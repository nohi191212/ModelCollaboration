from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MM Grounding DINO on LVIS query JSONL")
    parser.add_argument("--model", required=True, help="Local MM Grounding DINO checkpoint directory")
    parser.add_argument("--input", required=True, help="JSONL with image/image_path and labels fields")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--image-root", default="", help="Prefix for relative image paths")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-items", type=int, default=-1)
    return parser.parse_args()


def load_rows(path: Path, max_items: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if max_items > 0 and len(rows) >= max_items:
                break
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    try:
        processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    except TypeError:
        processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    image_root = Path(args.image_root) if args.image_root else None
    rows = load_rows(Path(args.input), args.max_items)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_name = row.get("image_path", row.get("image"))
            if not image_name:
                raise ValueError("each input row needs image_path or image")
            image_path = Path(image_name)
            if image_root is not None and not image_path.is_absolute():
                image_path = image_root / image_path
            query_labels = [str(label).strip().lower() for label in row.get("labels", []) if str(label).strip()]
            if not query_labels:
                raise ValueError("each input row needs a non-empty labels list")

            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, text=[query_labels], return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=args.threshold,
                text_threshold=args.text_threshold,
                target_sizes=[image.size[::-1]],
                text_labels=[query_labels],
            )[0]

            detections = []
            labels = results.get("text_labels", results.get("labels", []))
            for box, score, label in zip(results["boxes"], results["scores"], labels):
                detections.append(
                    {
                        "box_xyxy": [round(float(value), 3) for value in box.tolist()],
                        "score": round(float(score), 6),
                        "label": str(label),
                    }
                )
            output = dict(row)
            output.update(
                {
                    "image": str(image_path),
                    "queries": query_labels,
                    "detections": detections,
                    "expert": "mm_grounding_dino_tiny_o365v1_goldg_v3det",
                }
            )
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
