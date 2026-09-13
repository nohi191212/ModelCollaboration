#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--box-threshold", type=float, required=True)
parser.add_argument("--text-threshold", type=float, required=True)
args = parser.parse_args()

if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")

rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
if not rows:
    raise ValueError("input is empty")
sample_ids = [row["sample_id"] for row in rows]
if len(sample_ids) != len(set(sample_ids)):
    raise ValueError("input sample identifiers are not unique")
for row in rows:
    for field in ("sample_id", "split", "image_id", "image_path", "width", "height", "expression"):
        if field not in row:
            raise ValueError(f"sample {row.get('sample_id')} is missing {field}")
    if not Path(row["image_path"]).is_file():
        raise FileNotFoundError(row["image_path"])

device = torch.device(args.device)
processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
args.output.parent.mkdir(parents=True, exist_ok=True)

started = time.perf_counter()
with args.output.open("x", encoding="utf-8") as destination:
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch_rows]
        prompts = [row["expression"].strip() + "." for row in batch_rows]
        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(device)
        target_sizes = torch.tensor([[image.height, image.width] for image in images], device=device)
        for image in images:
            image.close()

        with torch.inference_mode():
            outputs = model(**inputs)
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.box_threshold - 1e-6,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes,
        )
        if len(processed) != len(batch_rows):
            raise ValueError(f"expected {len(batch_rows)} outputs, got {len(processed)}")

        for row, result in zip(batch_rows, processed):
            objects = []
            for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
                if float(score) < args.box_threshold:
                    continue
                coordinates = [float(value) for value in box.tolist()]
                if len(coordinates) != 4 or not all(torch.isfinite(torch.tensor(coordinates))):
                    raise ValueError(f"sample {row['sample_id']} produced a non-finite box")
                objects.append({
                    "bbox_xyxy_absolute": coordinates,
                    "score": float(score),
                    "grounded_text": str(label),
                })
            objects.sort(key=lambda item: item["score"], reverse=True)
            output = dict(row)
            output.update({
                "model": "mm_grounding_dino_tiny_o365v1_goldg_v3det",
                "prompt": row["expression"].strip() + ".",
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "objects": objects,
            })
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")
        destination.flush()
        completed = min(start + args.batch_size, len(rows))
        elapsed = time.perf_counter() - started
        print(json.dumps({
            "completed": completed,
            "total": len(rows),
            "elapsed_seconds": elapsed,
            "samples_per_second": completed / elapsed,
        }), flush=True)

