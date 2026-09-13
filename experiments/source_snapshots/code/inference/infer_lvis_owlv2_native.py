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

rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as destination:
    for index, row in enumerate(rows, 1):
        image = Image.open(row["image_path"]).convert("RGB")
        text_labels = [query_labels]
        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=[image.size[::-1]],
            threshold=args.threshold,
            text_labels=text_labels,
        )[0]

        labels = result["text_labels"]
        detections = []
        for box, score, label in zip(result["boxes"], result["scores"], labels):
            query = str(label)
            if query not in category_by_query:
                raise ValueError(f"unexpected OWLv2 label: {query!r}")
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
            "model": "owlv2-base-patch16-ensemble",
            "class_count": len(categories),
            "threshold": args.threshold,
            "top_k": args.top_k,
            "detections": detections[: args.top_k],
        })
        destination.write(json.dumps(output, ensure_ascii=False) + "\n")
        image.close()
        print(json.dumps({"completed": index, "total": len(rows), "detections": len(output["detections"])}), flush=True)
