#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import RTDETR, YOLO


EXPECTED_NAMES = {
    0: "rule_1_ppe_violation",
    1: "rule_2_fall_protection_violation",
    2: "rule_3_unprotected_edge_violation",
    3: "rule_4_excavator_proximity_violation",
}
MODEL_CLASSES = {"yolo26x": YOLO, "rtdetr_x": RTDETR}


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=sorted(MODEL_CLASSES), required=True)
parser.add_argument("--weights", type=Path, required=True)
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--half", action="store_true")
args = parser.parse_args()

for path in (args.weights, args.dataset_dir / "evaluation_ground_truth.jsonl"):
    if not path.exists():
        raise FileNotFoundError(path)
if args.output_dir.exists():
    raise FileExistsError(f"refusing to overwrite {args.output_dir}")
args.output_dir.mkdir(parents=True)

model = MODEL_CLASSES[args.model](args.weights)
model_names = {int(key): value for key, value in model.names.items()}
if model_names != EXPECTED_NAMES:
    raise ValueError(f"unexpected trained class map: {model_names}")

for split, expected_count in (("val", 701), ("test", 3004)):
    image_paths = sorted((args.dataset_dir / "images" / split).glob("*.jpg"))
    if len(image_paths) != expected_count:
        raise ValueError(f"expected {expected_count} {split} images, found {len(image_paths)}")
    output_path = args.output_dir / f"yolo4_events_{split}_predictions.jsonl"
    completed = 0
    with output_path.open("w", encoding="utf-8") as destination:
        results = model.predict(
            source=str(args.dataset_dir / "images" / split),
            stream=True,
            imgsz=args.imgsz,
            batch=args.batch,
            half=args.half,
            conf=0.001,
            iou=0.7,
            max_det=300,
            device=args.device,
            project=str(args.output_dir),
            name=f"predict_{split}",
            exist_ok=True,
            save=False,
            verbose=False,
        )
        for result in results:
            sample_id = Path(result.path).stem
            height, width = result.orig_shape
            boxes = []
            for coordinates, score, class_id in zip(
                result.boxes.xyxyn.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                integer_class_id = int(class_id)
                if integer_class_id not in EXPECTED_NAMES:
                    raise ValueError(f"out-of-range class {class_id} for {sample_id}")
                x1, y1, x2, y2 = map(float, coordinates)
                if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                    raise ValueError(f"invalid prediction box for {sample_id}: {coordinates}")
                if not (0.0 <= float(score) <= 1.0):
                    raise ValueError(f"invalid prediction score for {sample_id}: {score}")
                boxes.append(
                    {
                        "class_id": integer_class_id,
                        "class_name": EXPECTED_NAMES[integer_class_id],
                        "score": float(score),
                        "box_xyxy_normalized": [x1, y1, x2, y2],
                    }
                )
            destination.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "image_path": str(result.path),
                        "width": width,
                        "height": height,
                        "predictions": boxes,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            destination.flush()
            completed += 1
            print(json.dumps({"model": args.model, "split": split, "completed": completed, "total": expected_count}), flush=True)
    if completed != expected_count:
        raise ValueError(f"{split} inference produced {completed} rows, expected {expected_count}")
