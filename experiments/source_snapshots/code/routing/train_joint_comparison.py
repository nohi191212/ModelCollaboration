"""Compare per-pair router fitting with a shared multi-task router.

Both settings use the same 4M student image/text representation and error
target.  Validation chooses epochs and per-pair thresholds; test is final.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
WORK = ROOT / 'outputs/router_full_20260910'
EXPERTS = {'cub': 'cub', 'glsim': 'cub', 'nlvr2': 'nlvr2', 'vilt': 'nlvr2',
           'groundingdino': 'grefcoco', 'instancevg': 'grefcoco',
           'yolo26x': 'construction', 'rtdetr_x_fp32': 'construction'}


class JointRouter(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(1538, dim), nn.LayerNorm(dim))
        self.text = nn.Sequential(nn.Linear(768, dim), nn.LayerNorm(dim))
        self.head = nn.Sequential(nn.Linear(2 * dim, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, data, idx):
        image = data['image'][idx]
        valid = data['valid'][idx]
        image = self.image(torch.cat([image.flatten(1), valid.float()], dim=1))
        text = self.text(data['text'][idx])
        return self.head(torch.cat([image, text], dim=1))[:, 0]


def rows_and_features(expert, split, device):
    task = EXPERTS[expert]
    base = WORK / 'data' / expert / split
    rows = [json.loads(x) for x in (base / 'records.jsonl').read_text().splitlines() if x.strip()]
    feat = WORK / 'features' / '4M' / task / split
    ids = json.loads((feat / 'sample_ids.json').read_text())
    assert ids == [r['sample_id'] for r in rows]
    data = {'image': torch.from_numpy(np.load(feat / 'image.npy')).to(device=device, dtype=torch.float32),
            'text': torch.from_numpy(np.load(feat / 'text.npy')).to(device=device, dtype=torch.float32),
            'valid': torch.from_numpy(np.load(feat / 'valid.npy')).to(device=device, dtype=torch.bool)}
    return rows, data


def error_targets(rows, endpoint, device):
    if 'events' in rows[0]['small_label']:
        y = [1. - np.mean([r['small_label']['events'][e]['correct'] for e in
                           ['rule_1_ppe_violation', 'rule_2_fall_protection_violation',
                            'rule_3_unprotected_edge_violation', 'rule_4_excavator_proximity_violation']]) for r in rows]
    else:
        y = [1. - float(r['small_label']['correct']) for r in rows]
    return torch.tensor(y, device=device, dtype=torch.float32)


def train_one(train_rows, train_data, val_blocks, endpoint, out, device):
    out.mkdir(parents=True, exist_ok=True)
    model = JointRouter().to(device)
    y = error_targets(train_rows, endpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best = -1.; best_epoch = 0; stale = 0; started = time.time(); history = []
    order_all = torch.arange(len(train_rows), device=device)
    for epoch in range(1, 51):
        model.train(); order = order_all[torch.randperm(len(order_all), device=device)]; total = 0.; seen = 0
        for b in range(0, len(order), 2048):
            idx = order[b:b + 2048]; optimizer.zero_grad(set_to_none=True)
            pred = model(train_data, idx); loss = F.binary_cross_entropy_with_logits(pred, y[idx])
            if not torch.isfinite(loss): raise RuntimeError(('nonfinite joint loss', endpoint, epoch))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True); optimizer.step()
            total += loss.item() * len(idx); seen += len(idx)
        model.eval(); scores_by_expert = {}
        with torch.no_grad():
            for expert, (rows, data) in val_blocks.items():
                scores = []
                for b in range(0, len(rows), 8192):
                    idx = torch.arange(b, min(b + 8192, len(rows)), device=device)
                    scores.append(torch.sigmoid(model(data, idx)).cpu().numpy())
                scores_by_expert[expert] = np.concatenate(scores)
        # Macro-average the validation area across the eight model pairs.
        from routing.train_router_formal import ranked_curve
        areas = [ranked_curve(rows, endpoint, scores_by_expert[expert])['selection_score']
                 for expert, (rows, _) in val_blocks.items()]
        score = float(np.mean(areas)); history.append({'epoch': epoch, 'loss': total / max(1, seen), 'macro_validation_selection': score})
        print(out.name, history[-1], flush=True)
        if score > best:
            best = score; best_epoch = epoch; stale = 0
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, out / 'best.pt')
        else:
            stale += 1
            if stale >= 8: break
    with (out / 'history.jsonl').open('w') as f:
        f.write('\n'.join(json.dumps(x) for x in history) + '\n')
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True)['state_dict']); model.eval()
    return model, {'best_epoch': best_epoch, 'validation_selection': best, 'seconds': time.time() - started}


def scores(model, rows, data, device):
    vals = []
    with torch.no_grad():
        for b in range(0, len(rows), 8192):
            idx = torch.arange(b, min(b + 8192, len(rows)), device=device)
            vals.append(torch.sigmoid(model(data, idx)).cpu().numpy())
    return np.concatenate(vals)


def evaluate_pair(model, val_rows, val_data, test_rows, test_data, endpoint, device):
    from routing.train_router_formal import fixed_curve, ranked_curve, validation_thresholds
    val_scores = scores(model, val_rows, val_data, device)
    test_scores = scores(model, test_rows, test_data, device)
    thresholds = validation_thresholds(val_rows, endpoint, val_scores)
    return {'validation': ranked_curve(val_rows, endpoint, val_scores),
            'test_ranked': ranked_curve(test_rows, endpoint, test_scores),
            'test_fixed': fixed_curve(test_rows, endpoint, test_scores, thresholds),
            'validation_thresholds': thresholds}


def main():
    sys.path.insert(0, str(ROOT / 'code'))
    device = torch.device('cuda')
    single = {}; val_blocks = {}; test_blocks = {}; train_blocks = {}
    for expert in EXPERTS:
        train_blocks[expert] = rows_and_features(expert, 'train', device)
        val_blocks[expert] = rows_and_features(expert, 'val', device)
        test_blocks[expert] = rows_and_features(expert, 'test', device)
    for endpoint in ['Qwen3.8', 'MiniCPM']:
        endpoint_dir = WORK / 'joint_runs' / endpoint
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        # Individual optimization for each model pair.
        for expert, (tr_rows, tr_data) in train_blocks.items():
            out = endpoint_dir / 'single' / expert
            model, fit = train_one(tr_rows, tr_data, {expert: val_blocks[expert]}, endpoint, out, device)
            result = evaluate_pair(model, val_blocks[expert][0], val_blocks[expert][1],
                                   test_blocks[expert][0], test_blocks[expert][1], endpoint, device)
            (out / 'completion.json').write_text(json.dumps({'status': 'complete', 'mode': 'single_task',
                'expert': expert, 'endpoint': endpoint, **fit, **result}, indent=2))
            single[(expert, endpoint)] = result
        # Shared multi-task optimization over all eight model-pair datasets.
        train_rows = []; train_data_parts = []
        for expert in EXPERTS:
            train_rows.extend(train_blocks[expert][0]); train_data_parts.append(train_blocks[expert][1])
        train_data = {k: torch.cat([part[k] for part in train_data_parts], dim=0) for k in ['image', 'text', 'valid']}
        model, fit = train_one(train_rows, train_data, val_blocks, endpoint, endpoint_dir / 'joint', device)
        joint_results = {}
        for expert in EXPERTS:
            result = evaluate_pair(model, val_blocks[expert][0], val_blocks[expert][1],
                                   test_blocks[expert][0], test_blocks[expert][1], endpoint, device)
            joint_results[expert] = result
        (endpoint_dir / 'joint' / 'completion.json').write_text(json.dumps({'status': 'complete', 'mode': 'multi_task',
            'endpoint': endpoint, **fit, 'pairs': joint_results}, indent=2))
        comparison = []
        for expert in EXPERTS:
            a = next(x for x in single[(expert, endpoint)]['test_fixed']['curve'] if x['budget'] == .25)['metric']
            b = next(x for x in joint_results[expert]['test_fixed']['curve'] if x['budget'] == .25)['metric']
            comparison.append({'expert': expert, 'endpoint': endpoint, 'single_task_test_fixed25': a,
                               'multi_task_test_fixed25': b, 'delta_multi_minus_single': b - a})
        (endpoint_dir / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    summary = {'status': 'complete', 'endpoints': ['Qwen3.8', 'MiniCPM'], 'student': '4M',
               'branches': ['image', 'text'], 'target': 'error', 'test_used_for_selection': False}
    (WORK / 'joint_comparison_complete.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
