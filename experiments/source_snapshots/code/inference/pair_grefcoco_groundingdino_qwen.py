#!/usr/bin/env python3
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
parser.add_argument("--labels", type=Path, required=True)
parser.add_argument("--qwen-predictions", type=Path, required=True)
parser.add_argument("--grounding-per-sample", type=Path, required=True)
parser.add_argument("--metrics-output", type=Path, required=True)
parser.add_argument("--details-output", type=Path, required=True)
parser.add_argument("--score-threshold", type=float, required=True)
parser.add_argument("--iou-threshold", type=float, required=True)
parser.add_argument("--f1-threshold", type=float, required=True)
args = parser.parse_args()

for path in (args.metrics_output, args.details_output):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")

labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line]
qwen_rows = [json.loads(line) for line in args.qwen_predictions.read_text(encoding="utf-8").splitlines() if line]
grounding_rows = [json.loads(line) for line in args.grounding_per_sample.read_text(encoding="utf-8").splitlines() if line]
if len(labels) != 14229 or len(qwen_rows) != 14229 or len(grounding_rows) != 14229:
    raise ValueError("all three inputs must contain exactly 14229 rows")
label_ids = [row["sample_id"] for row in labels]
qwen_ids = [row["sample_id"] for row in qwen_rows]
grounding_ids = [row["sample_id"] for row in grounding_rows]
if label_ids != qwen_ids or label_ids != grounding_ids:
    raise ValueError("sample order differs across labels, Qwen, and GroundingDINO")

counts = Counter()
details = []
for label, qwen_row, grounding_row in zip(labels, qwen_rows, grounding_rows):
    sample_id = label["sample_id"]
    output = qwen_row["generalist_output"]
    status = output.get("parse_status")
    invalid = status == "invalid_output_counted_wrong"
    if status not in {"valid", "invalid_output_counted_wrong"}:
        raise ValueError(f"unexpected Qwen parse status for {sample_id}: {status!r}")
    if invalid:
        if output.get("objects") != [] or not isinstance(output.get("raw_answer"), str):
            raise ValueError(f"invalid Qwen row is not preserved correctly: {sample_id}")
        filtered_boxes = []
    else:
        filtered_boxes = [
            item["bbox_xyxy_absolute"]
            for item in output["objects"]
            if float(item["score"]) >= args.score_threshold
        ]
    target_boxes = [item["bbox_xyxy"] for item in label["target_boxes"]]

    if label["target_type"] == "no_target":
        qwen_f1 = 1.0 if not invalid and len(filtered_boxes) == 0 else 0.0
    else:
        if not target_boxes:
            raise ValueError(f"target row has no target boxes: {sample_id}")
        true_positive = 0
        prediction_tensor = torch.tensor(filtered_boxes, dtype=torch.float64).reshape(-1, 4)
        target_tensor = torch.tensor(target_boxes, dtype=torch.float64).reshape(-1, 4)
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
        qwen_f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

    qwen_correct = qwen_f1 >= args.f1_threshold
    grounding_correct = grounding_row["correct"]
    if not isinstance(grounding_correct, bool):
        raise ValueError(f"GroundingDINO correctness is not boolean: {sample_id}")
    if not grounding_correct and qwen_correct:
        outcome = "rescued_by_qwen"
    elif grounding_correct and not qwen_correct:
        outcome = "harmed_by_qwen"
    elif grounding_correct and qwen_correct:
        outcome = "both_correct"
    else:
        outcome = "both_wrong"
    counts[outcome] += 1
    counts[f"{outcome}_{label['target_type']}"] += 1
    details.append({
        "sample_id": sample_id,
        "target_type": label["target_type"],
        "groundingdino_correct": grounding_correct,
        "groundingdino_instance_f1": grounding_row["instance_f1"],
        "qwen_correct": qwen_correct,
        "qwen_instance_f1": qwen_f1,
        "qwen_parse_status": status,
        "outcome": outcome,
    })

metrics = {
    "status": "complete",
    "rows": len(details),
    "score_threshold": args.score_threshold,
    "iou_threshold": args.iou_threshold,
    "f1_threshold": args.f1_threshold,
    "groundingdino_correct": counts["harmed_by_qwen"] + counts["both_correct"],
    "qwen_correct": counts["rescued_by_qwen"] + counts["both_correct"],
    "rescued_by_qwen": counts["rescued_by_qwen"],
    "harmed_by_qwen": counts["harmed_by_qwen"],
    "both_correct": counts["both_correct"],
    "both_wrong": counts["both_wrong"],
    "net_gain_if_all_upgraded_to_qwen": counts["rescued_by_qwen"] - counts["harmed_by_qwen"],
    "by_target_type": {
        target_type: {
            outcome: counts[f"{outcome}_{target_type}"]
            for outcome in ("rescued_by_qwen", "harmed_by_qwen", "both_correct", "both_wrong")
        }
        for target_type in ("single_target", "multi_target", "no_target")
    },
}
args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
args.details_output.parent.mkdir(parents=True, exist_ok=True)
args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with args.details_output.open("x", encoding="utf-8") as destination:
    for row in details:
        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(metrics, ensure_ascii=False))
