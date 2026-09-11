"""Recompute saved learned-routing curves from per-example outcomes (CPU only)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--artifacts', type=Path, required=True)
p.add_argument('--output', type=Path, default=Path('replayed_main.csv'))
a = p.parse_args()
root = a.artifacts / 'experiment_data'
groups = json.loads((root / 'test_results.json').read_text())['groups']
rows = []
for g in groups:
    for split, field in [('val', 'validation'), ('test', 'test')]:
        with np.load(root / 'pairs' / g['id'] / (split + '.npz')) as z:
            scores, small, large = z['scores'], z['small'], z['large']
        curve = g[field + '_curve']
        recomputed = []
        for point in curve:
            op = point['operator']
            if op == 'never': mask = np.zeros(len(scores), dtype=bool)
            elif op == 'always': mask = np.ones(len(scores), dtype=bool)
            elif op == 'ge': mask = scores >= point['threshold']
            elif op == 'gt': mask = scores > point['threshold']
            else: raise ValueError(op)
            if small.ndim == 1:
                metric = float(np.where(mask, large, small).mean())
            else:
                tp, fp, fn = np.where(mask[:, None, None], large, small).sum(0).T
                den = 2 * tp + fp + fn
                metric = float(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0).mean())
            call = float(mask.mean())
            if abs(metric - point['metric']) > 1e-9 or abs(call - point['actual_fraction']) > 1e-9:
                raise ValueError((g['id'], split, point, metric, call))
            recomputed.append((metric, call))
        i = g['best_budget_index']
        if split == 'val':
            selected = max(range(1, 100), key=lambda j: (recomputed[j][0], -recomputed[j][1], -j))
            assert selected == i, (g['id'], selected, i)
        metric, call = recomputed[i]
        rows.append({'pair': g['id'], 'split': split, 'metric': metric, 'call_fraction': call})
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(f'Verified {len(groups)} pairs, validation/test curves, and validation-selected points: {a.output}')
