import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from instancevg.evaluations.grefcoco import grec_evaluate_f1_nacc, generalized_box_iou

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--split', required=True)
parser.add_argument('--predictions', type=Path, required=True)
args = parser.parse_args()
prepared = args.root / 'outputs/experiments/20260827_grefcoco_native_grounding/prepared'
predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
inputs = [json.loads(line) for line in (prepared / f'grefcoco_{args.split}_inputs.jsonl').read_text().splitlines()]
labels = [json.loads(line) for line in (prepared / f'grefcoco_{args.split}_labels.jsonl').read_text().splitlines()]
assert [r['sample_id'] for r in predictions] == [r['sample_id'] for r in inputs] == [r['sample_id'] for r in labels]
assert len({r['sample_id'] for r in predictions}) == len(predictions)
official_predictions, targets, metas = [], [], []
counts = Counter()
rows = []
for pred, label in zip(predictions, labels):
    assert pred['model'] == 'InstanceVG-grefcoco'
    boxes = torch.tensor([obj['bbox_xyxy_absolute'] for obj in pred['objects']], dtype=torch.float32).reshape(-1, 4)
    scores = torch.tensor([obj['score'] for obj in pred['objects']], dtype=torch.float32)
    target = torch.tensor([obj['bbox_xyxy'] for obj in label['target_boxes']], dtype=torch.float32).reshape(-1, 4)
    empty = label['target_type'] == 'no_target'
    assert empty == (len(target) == 0)
    official_predictions.append(dict(boxes=boxes, scores=scores))
    targets.append(target)
    metas.append(dict(empty=empty))
    selected = boxes[scores >= 0.7]
    if empty:
        f1 = float(len(selected) == 0)
    else:
        giou = generalized_box_iou(selected, target)
        matched = 0
        for _ in range(min(len(selected), len(target))):
            value, index = giou.flatten().max(dim=0)
            if value < 0.5:
                break
            p, t = divmod(int(index), len(target))
            matched += 1
            giou[p, :] = 0
            giou[:, t] = 0
        f1 = 2 * matched / (len(selected) + len(target))
    correct = f1 >= 1.0
    counts['correct'] += int(correct)
    counts[label['target_type']] += 1
    counts[label['target_type'] + '_correct'] += int(correct)
    rows.append(dict(sample_id=pred['sample_id'], target_type=label['target_type'], predicted_boxes=len(selected),
                     target_boxes=len(target), instance_f1=f1, correct=correct))
f1, nacc, tacc = grec_evaluate_f1_nacc(official_predictions, targets, metas, thresh_score=0.7, thresh_iou=0.5, thresh_F1=1.0)
assert abs(f1 / 100 - counts['correct'] / len(rows)) < 1e-12
metrics = dict(status='complete', rows=len(rows), split=args.split, F1_score=f1/100, N_acc=nacc/100, T_acc=tacc/100,
               mean_instance_f1=sum(r['instance_f1'] for r in rows)/len(rows), score_threshold=0.7,
               matching='official GIoU greedy, threshold 0.5; exact expression F1=1.0', counts=dict(counts))
args.predictions.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
with args.predictions.with_suffix('.per_sample.jsonl').open('x') as out:
    for row in rows:
        out.write(json.dumps(row) + '\n')
print(json.dumps(metrics), flush=True)
