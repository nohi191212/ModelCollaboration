#!/usr/bin/env python3
"""Run a YOLO construction/PPE expert and emit router features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with image or image_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expert-name", default="YOLO10X-ConstructionSafety")
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = row.get("image_path", row.get("image"))
            if not image_path:
                raise ValueError(f"line {line_no} needs image_path")
            result = model.predict(
                source=image_path, imgsz=args.imgsz, conf=args.conf,
                device=args.device, verbose=False,
            )[0]
            names = result.names
            detections = []
            counts = {}
            max_conf = 0.0
            if result.boxes is not None:
                for cls_id, conf, xyxy in zip(
                    result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()
                ):
                    label = names[int(cls_id)] if isinstance(names, dict) else names[int(cls_id)]
                    counts[label] = counts.get(label, 0) + 1
                    max_conf = max(max_conf, float(conf))
                    detections.append({"class": label, "confidence": float(conf), "xyxy": [float(v) for v in xyxy]})
            out = dict(row)
            out.update({
                "expert_model": args.expert_name,
                "detections": detections,
                "detection_counts": counts,
                "expert_confidence": max_conf,
                "router_features": [float(len(detections)), max_conf] + [float(v) for _, v in sorted(counts.items())],
            })
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
