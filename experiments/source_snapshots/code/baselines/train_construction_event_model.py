#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
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
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--patience", type=int, required=True)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--fraction", type=float, default=1.0)
args = parser.parse_args()

for path in (args.weights, args.data):
    if not path.is_file():
        raise FileNotFoundError(path)
if args.run_dir.exists():
    raise FileExistsError(f"refusing to overwrite {args.run_dir}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

args.run_dir.parent.mkdir(parents=True, exist_ok=True)
request_path = args.run_dir.parent / f"{args.run_dir.name}.request.json"
if request_path.exists():
    raise FileExistsError(f"refusing to overwrite {request_path}")
request_path.write_text(
    json.dumps(
        {
            "model": args.model,
            "weights": str(args.weights),
            "data": str(args.data),
            "run_dir": str(args.run_dir),
            "device": args.device,
            "epochs": args.epochs,
            "patience": args.patience,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

model = MODEL_CLASSES[args.model](args.weights)
model.train(
    data=str(args.data),
    epochs=args.epochs,
    patience=args.patience,
    imgsz=args.imgsz,
    batch=args.batch,
    workers=args.workers,
    device=args.device,
    optimizer="auto",
    seed=20260831,
    deterministic=True,
    amp=False,
    cache=False,
    fraction=args.fraction,
    plots=False,
    pretrained=True,
    project=str(args.run_dir.parent),
    name=args.run_dir.name,
    exist_ok=False,
    verbose=True,
)

best_path = args.run_dir / "weights" / "best.pt"
if not best_path.is_file():
    raise FileNotFoundError(f"training returned without expected checkpoint: {best_path}")
print(json.dumps({"model": args.model, "best_weights": str(best_path)}, ensure_ascii=False))
