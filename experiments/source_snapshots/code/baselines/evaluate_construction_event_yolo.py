#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--predictions-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
class_names = {
    0: "rule_1_ppe_violation",
    1: "rule_2_fall_protection_violation",
    2: "rule_3_unprotected_edge_violation",
    3: "rule_4_excavator_proximity_violation",
}
expected_counts = {"val": 701, "test": 3004}
ground_truth = {split: {} for split in expected_counts}
predictions = {}

ground_truth_path = args.dataset_dir / "evaluation_ground_truth.jsonl"
ground_truth_rows = [
    json.loads(line)
    for line in ground_truth_path.read_text(encoding="utf-8").splitlines()
    if line
]
for row in ground_truth_rows:
    if set(row) != {"sample_id", "split", "events"}:
        raise ValueError(f"invalid evaluation ground-truth row: {row}")
    split = row["split"]
    if split not in ground_truth:
        continue
    if row["sample_id"] in ground_truth[split]:
        raise ValueError(f"duplicate {split} ground-truth identifier: {row['sample_id']}")
    if set(row["events"]) != set(class_names.values()):
        raise ValueError(f"invalid event fields for {row['sample_id']}")
    events_by_class = {}
    for class_id, class_name in class_names.items():
        event = row["events"][class_name]
        if not isinstance(event, dict) or set(event) != {"present", "reason", "boxes"}:
            raise ValueError(f"invalid {class_name} event for {row['sample_id']}")
        if not isinstance(event["present"], bool) or not isinstance(event["boxes"], list):
            raise ValueError(f"invalid {class_name} presence or boxes for {row['sample_id']}")
        boxes = []
        for box in event["boxes"]:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"invalid {class_name} ground-truth box for {row['sample_id']}: {box}")
            x1, y1, x2, y2 = map(float, box)
            if not (0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0):
                raise ValueError(f"out-of-range {class_name} ground-truth box for {row['sample_id']}: {box}")
            boxes.append([x1, y1, x2, y2])
        if not event["present"] and (event["reason"] is not None or boxes):
            raise ValueError(f"absent {class_name} has reason or boxes for {row['sample_id']}")
        events_by_class[class_id] = {"present": event["present"], "boxes": boxes}
    ground_truth[split][row["sample_id"]] = events_by_class

for split, expected_count in expected_counts.items():
    if len(ground_truth[split]) != expected_count:
        raise ValueError(f"expected {expected_count} {split} ground-truth rows, found {len(ground_truth[split])}")

for split in ("val", "test"):
    prediction_path = args.predictions_dir / f"yolo4_events_{split}_predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line]
    split_predictions = {row["sample_id"]: row["predictions"] for row in rows}
    if len(split_predictions) != len(rows):
        raise ValueError(f"duplicate {split} prediction identifiers")
    if set(split_predictions) != set(ground_truth[split]):
        raise ValueError(f"{split} prediction and label identifiers differ")
    predictions[split] = split_predictions

thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
validation_sweep = []
selected_threshold = None
selected_mean_positive_macro_iou = -1.0
selected_mean_presence_f1 = -1.0

for threshold in thresholds:
    per_class = {}
    for class_id, class_name in class_names.items():
        tp = fp = fn = 0
        positive_ious = []
        positive_without_boxes = 0
        for sample_id, events_by_class in ground_truth["val"].items():
            reference = events_by_class[class_id]
            candidate_boxes = [
                prediction["box_xyxy_normalized"]
                for prediction in predictions["val"][sample_id]
                if prediction["class_id"] == class_id and prediction["score"] >= threshold
            ]
            true_positive = reference["present"]
            predicted_positive = bool(candidate_boxes)
            tp += int(true_positive and predicted_positive)
            fp += int(not true_positive and predicted_positive)
            fn += int(true_positive and not predicted_positive)
            if true_positive and not reference["boxes"]:
                positive_without_boxes += 1
            if reference["boxes"]:
                reference_mask = np.zeros((100, 100), dtype=bool)
                candidate_mask = np.zeros((100, 100), dtype=bool)
                for x1, y1, x2, y2 in reference["boxes"]:
                    reference_mask[int(y1 * 100): int(y2 * 100) + 1, int(x1 * 100): int(x2 * 100) + 1] = True
                for x1, y1, x2, y2 in candidate_boxes:
                    candidate_mask[int(y1 * 100): int(y2 * 100) + 1, int(x1 * 100): int(x2 * 100) + 1] = True
                intersection = np.logical_and(reference_mask, candidate_mask).sum()
                union = np.logical_or(reference_mask, candidate_mask).sum()
                positive_ious.append(float(intersection / union))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "box_labeled_positive_rows": len(positive_ious),
            "positive_without_box_rows": positive_without_boxes,
            "positive_macro_mask_iou": float(np.mean(positive_ious)),
        }
    mean_presence_f1 = float(np.mean([metrics["f1"] for metrics in per_class.values()]))
    mean_positive_macro_iou = float(np.mean([metrics["positive_macro_mask_iou"] for metrics in per_class.values()]))
    validation_sweep.append({
        "threshold": threshold,
        "mean_presence_f1": mean_presence_f1,
        "mean_positive_macro_mask_iou": mean_positive_macro_iou,
        "per_class": per_class,
    })
    if mean_positive_macro_iou > selected_mean_positive_macro_iou:
        selected_threshold = threshold
        selected_mean_positive_macro_iou = mean_positive_macro_iou
        selected_mean_presence_f1 = mean_presence_f1
    elif mean_positive_macro_iou == selected_mean_positive_macro_iou and mean_presence_f1 > selected_mean_presence_f1:
        selected_threshold = threshold
        selected_mean_presence_f1 = mean_presence_f1
    elif (
        mean_positive_macro_iou == selected_mean_positive_macro_iou
        and mean_presence_f1 == selected_mean_presence_f1
        and threshold > selected_threshold
    ):
        selected_threshold = threshold

test_metrics = {}
for class_id, class_name in class_names.items():
    tp = fp = fn = 0
    positive_ious = []
    total_intersection = 0
    total_union = 0
    true_rows = 0
    predicted_rows = 0
    positive_without_boxes = 0
    for sample_id, events_by_class in ground_truth["test"].items():
        reference = events_by_class[class_id]
        candidate_boxes = [
            prediction["box_xyxy_normalized"]
            for prediction in predictions["test"][sample_id]
            if prediction["class_id"] == class_id and prediction["score"] >= selected_threshold
        ]
        true_positive = reference["present"]
        predicted_positive = bool(candidate_boxes)
        true_rows += int(true_positive)
        predicted_rows += int(predicted_positive)
        tp += int(true_positive and predicted_positive)
        fp += int(not true_positive and predicted_positive)
        fn += int(true_positive and not predicted_positive)
        if true_positive and not reference["boxes"]:
            positive_without_boxes += 1
        if reference["boxes"]:
            reference_mask = np.zeros((100, 100), dtype=bool)
            candidate_mask = np.zeros((100, 100), dtype=bool)
            for x1, y1, x2, y2 in reference["boxes"]:
                reference_mask[int(y1 * 100): int(y2 * 100) + 1, int(x1 * 100): int(x2 * 100) + 1] = True
            for x1, y1, x2, y2 in candidate_boxes:
                candidate_mask[int(y1 * 100): int(y2 * 100) + 1, int(x1 * 100): int(x2 * 100) + 1] = True
            intersection = int(np.logical_and(reference_mask, candidate_mask).sum())
            union = int(np.logical_or(reference_mask, candidate_mask).sum())
            positive_ious.append(intersection / union)
            total_intersection += intersection
            total_union += union
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    test_metrics[class_name] = {
        "true_positive_event_rows": true_rows,
        "box_labeled_positive_rows": len(positive_ious),
        "positive_without_box_rows": positive_without_boxes,
        "predicted_positive_rows": predicted_rows,
        "negative_rows": expected_counts["test"] - true_rows,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "negative_false_positive_rate": fp / (expected_counts["test"] - true_rows),
        "positive_macro_mask_iou": float(np.mean(positive_ious)),
        "positive_micro_mask_iou": total_intersection / total_union,
    }

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({
    "validation_images": expected_counts["val"],
    "test_images": expected_counts["test"],
    "threshold_selection": "single global threshold maximizing mean four-event positive macro mask IoU on fixed validation split; tie by mean presence F1, then higher threshold",
    "selected_threshold": selected_threshold,
    "validation_sweep": validation_sweep,
    "test_metrics": test_metrics,
    "mask_protocol": "official 100x100 union mask with integer endpoints; event presence includes source positives without boxes, localization excludes only those unboxable positives",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"selected_threshold": selected_threshold, "test_metrics": test_metrics}))
