#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_size = (right_bottom - left_top).clamp(min=0)
    intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union
    enclosing_left_top = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_size = (enclosing_right_bottom - enclosing_left_top).clamp(min=0)
    enclosing_area = enclosing_size[:, :, 0] * enclosing_size[:, :, 1]
    return iou - (enclosing_area - union) / enclosing_area


parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--labels", type=Path, required=True)
parser.add_argument("--metrics-output", type=Path, required=True)
parser.add_argument("--per-sample-output", type=Path, required=True)
parser.add_argument("--expected-rows", type=int, required=True)
parser.add_argument("--score-threshold", type=float, required=True)
parser.add_argument("--iou-threshold", type=float, required=True)
parser.add_argument("--f1-threshold", type=float, required=True)
args = parser.parse_args()

for path in (args.metrics_output, args.per_sample_output):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")

prediction_rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
input_rows = [json.loads(line) for line in args.inputs.read_text(encoding="utf-8").splitlines() if line]
label_rows = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line]
if len(prediction_rows) != args.expected_rows or len(input_rows) != args.expected_rows or len(label_rows) != args.expected_rows:
    raise ValueError("prediction, input, and label row counts do not match the expected count")
prediction_ids = [row["sample_id"] for row in prediction_rows]
input_ids = [row["sample_id"] for row in input_rows]
label_ids = [row["sample_id"] for row in label_rows]
if prediction_ids != input_ids or prediction_ids != label_ids:
    raise ValueError("prediction, input, and label sample order differs")
if len(set(prediction_ids)) != args.expected_rows:
    raise ValueError("sample identifiers are not unique")

counts = Counter()
f1_values = []
per_sample = []
for prediction, label in zip(prediction_rows, label_rows):
    sample_id = prediction["sample_id"]
    output = prediction["generalist_output"]
    if output.get("model") != "MiniCPM-V-4_5":
        raise ValueError(f"unexpected model for {sample_id}")
    parse_status = output["parse_status"]
    if parse_status == "valid":
        counts["valid_outputs"] += 1
    elif parse_status == "invalid_output_counted_wrong":
        if output.get("objects") != [] or not output.get("raw_answer") or not output.get("parse_error"):
            raise ValueError(f"invalid output was not preserved and emptied: {sample_id}")
        counts["invalid_outputs_counted_wrong"] += 1
    else:
        raise ValueError(f"unexpected parse status for {sample_id}: {parse_status}")

    if label["target_type"] not in {"single_target", "multi_target", "no_target"}:
        raise ValueError(f"invalid target type for {sample_id}")
    counts[f"rows_{label['target_type']}"] += 1
    target_boxes = [row["bbox_xyxy"] for row in label["target_boxes"]]
    objects = output["objects"]
    counts["predicted_boxes_before_threshold"] += len(objects)
    filtered_boxes = [
        item["bbox_xyxy_absolute"]
        for item in objects
        if float(item["score"]) >= args.score_threshold
    ]
    counts["predicted_boxes_after_threshold"] += len(filtered_boxes)

    if parse_status == "invalid_output_counted_wrong":
        f1 = 0.0
        correct = False
        counts["invalid_forced_wrong"] += 1
    elif label["target_type"] == "no_target":
        if target_boxes:
            raise ValueError(f"no-target row has target boxes: {sample_id}")
        correct = len(filtered_boxes) == 0
        counts["no_target_correct" if correct else "no_target_wrong"] += 1
        f1 = 1.0 if correct else 0.0
    else:
        if not target_boxes:
            raise ValueError(f"target row has no target boxes: {sample_id}")
        counts["target_has_prediction" if filtered_boxes else "target_no_prediction"] += 1
        prediction_tensor = torch.tensor(filtered_boxes, dtype=torch.float64).reshape(-1, 4)
        target_tensor = torch.tensor(target_boxes, dtype=torch.float64).reshape(-1, 4)
        true_positive = 0
        if prediction_tensor.numel() > 0:
            giou = generalized_box_iou(prediction_tensor, target_tensor)
            for _ in range(min(len(filtered_boxes), len(target_boxes))):
                top_value, top_index = torch.max(giou.reshape(-1), dim=0)
                if float(top_value) < args.iou_threshold:
                    break
                prediction_index = int(top_index) // len(target_boxes)
                target_index = int(top_index) % len(target_boxes)
                true_positive += 1
                giou[prediction_index, :] = 0
                giou[:, target_index] = 0
        false_positive = len(filtered_boxes) - true_positive
        false_negative = len(target_boxes) - true_positive
        f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
        correct = f1 >= args.f1_threshold

    f1_values.append(f1)
    counts["f1_exact_correct"] += int(correct)
    per_sample.append({
        "sample_id": sample_id,
        "target_type": label["target_type"],
        "parse_status": parse_status,
        "predicted_boxes": len(filtered_boxes),
        "target_boxes": len(target_boxes),
        "instance_f1": f1,
        "correct": correct,
    })

target_rows = counts["rows_single_target"] + counts["rows_multi_target"]
no_target_rows = counts["rows_no_target"]
metrics = {
    "status": "complete",
    "rows": len(prediction_ids),
    "unique_sample_ids": len(set(prediction_ids)),
    "sample_order_matches_fixed_input": True,
    "score_threshold": args.score_threshold,
    "iou_threshold": args.iou_threshold,
    "f1_threshold": args.f1_threshold,
    "F1_score": counts["f1_exact_correct"] / len(prediction_ids),
    "T_acc": counts["target_has_prediction"] / target_rows,
    "N_acc": counts["no_target_correct"] / no_target_rows,
    "mean_instance_f1": sum(f1_values) / len(f1_values),
    **dict(counts),
}
args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with args.per_sample_output.open("x", encoding="utf-8") as destination:
    for row in per_sample:
        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(metrics, ensure_ascii=False))
