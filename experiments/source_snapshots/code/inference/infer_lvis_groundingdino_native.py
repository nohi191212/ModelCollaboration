#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--categories", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--class-batch-size", type=int, required=True)
parser.add_argument("--prompt-batch-size", type=int, required=True)
parser.add_argument("--threshold", type=float, required=True)
parser.add_argument("--top-k", type=int, required=True)
args = parser.parse_args()

device = torch.device(args.device)
processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
categories = json.loads(args.categories.read_text(encoding="utf-8"))
if len(categories) != 1203:
    raise ValueError(f"expected 1203 LVIS categories, got {len(categories)}")
query_labels = [f"a {category['name'].replace('_', ' ')}" for category in categories]
category_by_query = {query: category for query, category in zip(query_labels, categories)}
class_batches = [
    query_labels[start : start + args.class_batch_size]
    for start in range(0, len(query_labels), args.class_batch_size)
]
dot_token_id = processor.tokenizer.convert_tokens_to_ids(".")

rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as destination:
    for index, row in enumerate(rows, 1):
        image = Image.open(row["image_path"]).convert("RGB")
        detections = []
        for start in range(0, len(class_batches), args.prompt_batch_size):
            prompt_batch = class_batches[start : start + args.prompt_batch_size]
            inputs = processor(
                images=[image] * len(prompt_batch),
                text=prompt_batch,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            for batch_index, prompt in enumerate(prompt_batch):
                dot_positions = (inputs.input_ids[batch_index] == dot_token_id).nonzero(as_tuple=True)[0].tolist()
                if len(dot_positions) != len(prompt):
                    raise ValueError(
                        f"expected {len(prompt)} class separators, got {len(dot_positions)}"
                    )
                positive_map = torch.zeros(
                    (len(prompt), outputs.logits.shape[-1]),
                    device=outputs.logits.device,
                )
                token_start = 1
                for class_index, token_end in enumerate(dot_positions):
                    if token_end <= token_start:
                        raise ValueError(f"empty token span for {prompt[class_index]!r}")
                    positive_map[class_index, token_start:token_end] = 1.0 / (token_end - token_start)
                    token_start = token_end + 1

                token_probabilities = outputs.logits[batch_index].sigmoid()
                class_scores = token_probabilities @ positive_map.T
                scores, class_indices = class_scores.max(dim=1)
                keep = scores > args.threshold
                scores = scores[keep]
                class_indices = class_indices[keep]
                boxes = outputs.pred_boxes[batch_index][keep]
                cx, cy, width, height = boxes.unbind(-1)
                boxes = torch.stack(
                    [cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height],
                    dim=-1,
                )
                boxes = boxes * torch.tensor(
                    [image.width, image.height, image.width, image.height],
                    device=boxes.device,
                )
                for box, score, class_index in zip(boxes, scores, class_indices):
                    query = prompt[int(class_index)]
                    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                    category = category_by_query[query]
                    detections.append({
                        "category_id": category["category_id"],
                        "category_name": category["name"],
                        "box_xyxy": [x1, y1, x2, y2],
                        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })
        detections.sort(key=lambda detection: detection["score"], reverse=True)
        output = dict(row)
        output.update({
            "model": "mm_grounding_dino_tiny_o365v1_goldg_v3det",
            "class_count": len(categories),
            "class_batch_size": args.class_batch_size,
            "class_batch_count": len(class_batches),
            "prompt_batch_size": args.prompt_batch_size,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "detections": detections[: args.top_k],
        })
        destination.write(json.dumps(output, ensure_ascii=False) + "\n")
        image.close()
        print(json.dumps({"completed": index, "total": len(rows), "detections": len(output["detections"])}), flush=True)
