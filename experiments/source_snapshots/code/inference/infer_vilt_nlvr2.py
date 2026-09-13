#!/usr/bin/env python3
"""NLVR2 specialist inference using the compact HF-only ViLT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, ViltForImagesAndTextClassification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dandelin/vilt-b32-finetuned-nlvr2")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with image1, image2, sentence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    model = ViltForImagesAndTextClassification.from_pretrained(args.model).eval().to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image1 = Image.open(row["image1"]).convert("RGB")
            image2 = Image.open(row["image2"]).convert("RGB")
            inputs = processor(
                [image1, image2],
                row["sentence"],
                return_tensors="pt",
                truncation=True,
                max_length=model.config.max_position_embeddings,
            )
            with torch.inference_mode():
                logits = model(
                    input_ids=inputs.input_ids.to(device),
                    pixel_values=inputs.pixel_values.unsqueeze(0).to(device),
                ).logits
                probs = logits.softmax(-1)[0]
            pred = int(probs.argmax().item())
            out = dict(row)
            out.update({
                "expert_model": args.model,
                "expert_pred": pred,
                "expert_prediction": model.config.id2label[pred],
                "expert_confidence": float(probs[pred]),
                "expert_probabilities": {
                    model.config.id2label[index]: float(probability)
                    for index, probability in enumerate(probs)
                },
            })
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
