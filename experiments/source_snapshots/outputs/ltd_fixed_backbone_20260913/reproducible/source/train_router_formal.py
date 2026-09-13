"""Formal router training on strict train/validation/test splits.

The validation split selects the stopping epoch and fixed deployment
thresholds.  The test split is touched only once, after the model is fixed.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
WORK = ROOT / 'outputs/router_full_20260910'


class FeatureRouter(nn.Module):
    def __init__(self, cfg, hidden_dim, confidence_dim, output_dim):
        super().__init__()
        self.branches = cfg['branches']
        self.fusion = cfg['fusion']
        d = cfg['representation_dim']
        self.projections = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, d), nn.LayerNorm(d))
                                          for name, dim in [('text', 768), ('hidden', hidden_dim),
                                                            ('confidence', confidence_dim), ('output', output_dim)]
                                          if name in self.branches})
        self.head = nn.Sequential(nn.Linear(len(self.branches) * d, 128), nn.GELU(),
                                  nn.Linear(128, 4 if cfg['target'] == 'four_state' else 1))
        if 'image' in self.branches:
            self.image_projection = nn.Linear(1538 if self.fusion == 'concat' else 768, d)
            self.image_norm = nn.LayerNorm(d)
            if self.fusion == 'attention':
                self.positions = nn.Parameter(torch.empty(1, 2, d))
                self.query = nn.Parameter(torch.empty(1, 1, d))
                nn.init.normal_(self.positions, std=.02)
                nn.init.normal_(self.query, std=.02)
                self.attention = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.)

    def forward(self, data, idx):
        fused = []
        for branch in self.branches:
            if branch == 'image':
                images, valid = data['image'][idx], data['valid'][idx]
                if self.fusion == 'concat':
                    x = self.image_projection(torch.cat([images.flatten(1), valid.float()], dim=1))
                else:
                    tokens = self.image_projection(images) + self.positions
                    x = self.attention(self.query.expand(len(idx), -1, -1), tokens, tokens,
                                       key_padding_mask=~valid, need_weights=False)[0][:, 0]
                fused.append(self.image_norm(x))
            else:
                fused.append(self.projections[branch](data[branch][idx]))
        return self.head(torch.cat(fused, dim=1))


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_hidden(expert, split, layers):
    if not layers:
        return None
    positions = json.loads((WORK / 'data' / expert / split / 'index.json').read_text())
    arrays = []
    for layer in layers:
        values = np.empty((len(positions), 1), dtype=np.float32)
        by_shard = defaultdict(list)
        for out_index, item in enumerate(positions):
            by_shard[item['shard']].append((out_index, item['position']))
        for shard, selected in by_shard.items():
            with np.load(Path(shard) / 'states.npz') as states:
                source = states[layer]
                if values.shape[1] == 1:
                    values = np.empty((len(positions), source.shape[1]), dtype=np.float32)
                out_idx, src_idx = zip(*selected)
                picked = np.asarray(source[list(src_idx)], dtype=np.float32)
                if not np.isfinite(picked).all():
                    raise RuntimeError(('nonfinite hidden state', expert, split, layer, shard))
                values[list(out_idx)] = picked
        arrays.append(values)
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)


def load_data(cfg, split, device):
    expert, task = cfg['expert'], cfg['task']
    rows = read_rows(WORK / 'data' / expert / split / 'records.jsonl')
    feature = WORK / 'features' / cfg['student'] / task / split
    feature_ids = json.loads((feature / 'sample_ids.json').read_text())
    assert feature_ids == [r['sample_id'] for r in rows], (expert, split, 'student feature identity mismatch')
    data = {
        'image': torch.from_numpy(np.load(feature / 'image.npy')).to(device=device, dtype=torch.float32),
        'text': torch.from_numpy(np.load(feature / 'text.npy')).to(device=device, dtype=torch.float32),
        'valid': torch.from_numpy(np.load(feature / 'valid.npy')).to(device=device, dtype=torch.bool),
        'confidence': torch.from_numpy(np.load(WORK / 'data' / expert / split / 'confidence.npy')).to(device=device, dtype=torch.float32),
        'output': torch.from_numpy(np.load(WORK / 'data' / expert / split / 'output.npy')).to(device=device, dtype=torch.float32),
    }
    layers = (cfg.get('hidden_layers') or ([cfg['hidden_layer']] if cfg.get('hidden_layer') else [])) if 'hidden' in cfg['branches'] else []
    hidden = load_hidden(expert, split, layers)
    if hidden is not None:
        data['hidden'] = torch.from_numpy(hidden).to(device=device, dtype=torch.float32)
    assert all(len(v) == len(rows) for v in data.values())
    assert all(torch.isfinite(v).all() for v in data.values() if v.dtype != torch.bool)
    return rows, data


def correctness(records, endpoint):
    small = [r['small_label'] for r in records]
    large = [r['large_labels'][endpoint] for r in records]
    if 'events' in small[0]:
        events = ['rule_1_ppe_violation', 'rule_2_fall_protection_violation',
                  'rule_3_unprotected_edge_violation', 'rule_4_excavator_proximity_violation']
        s = np.asarray([[r['events'][e]['correct'] for e in events] for r in small], dtype=float)
        l = np.asarray([[r['events'][e]['correct'] for e in events] for r in large], dtype=float)
    else:
        s = np.asarray([[r['correct']] for r in small], dtype=float)
        l = np.asarray([[r['correct']] for r in large], dtype=float)
    return s, l


def metric(records, endpoint, upgrade):
    chosen = [r['large_labels'][endpoint] if use else r['small_label'] for r, use in zip(records, upgrade)]
    if 'events' not in chosen[0]:
        return float(np.mean([r['correct'] for r in chosen]))
    events = ['rule_1_ppe_violation', 'rule_2_fall_protection_violation',
              'rule_3_unprotected_edge_violation', 'rule_4_excavator_proximity_violation']
    vals = []
    for event in events:
        target = np.asarray([r['events'][event]['target'] for r in chosen], dtype=bool)
        positive = np.asarray([r['events'][event]['prediction'] is True for r in chosen])
        tp, fp, fn = np.sum(target & positive), np.sum(~target & positive), np.sum(target & ~positive)
        vals.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.)
    return float(np.mean(vals))


BUDGETS = [0, .05, .10, .15, .20, .25, .50, .75, 1.]


def ranked_curve(records, endpoint, scores):
    order = np.argsort(-np.asarray(scores), kind='stable')
    s, l = correctness(records, endpoint)
    rescue, harm = (s == 0) & (l == 1), (s == 1) & (l == 0)
    points = []
    for budget in BUDGETS:
        count = int(np.floor(len(records) * budget))
        upgrade = np.zeros(len(records), dtype=bool)
        upgrade[order[:count]] = True
        selected = upgrade[:, None]
        points.append({'budget': budget, 'actual_fraction': float(upgrade.mean()),
                       'metric': metric(records, endpoint, upgrade),
                       'rescue': int((rescue & selected).sum()), 'harm': int((harm & selected).sum()),
                       'missed_rescue': int((rescue & ~selected).sum()),
                       'wasted_events': int(((s == l) & selected).sum())})
    area = float(sum((a['metric'] + b['metric']) * .5 * (b['budget'] - a['budget'])
                     for a, b in zip(points[:5], points[1:6])) / .25)
    return {'curve': points, 'selection_score': area, 'score_std': float(np.asarray(scores).std())}


def validation_thresholds(records, endpoint, scores):
    order = np.argsort(-np.asarray(scores), kind='stable')
    thresholds = []
    for budget in BUDGETS:
        count = int(np.floor(len(records) * budget))
        if count == 0:
            thresholds.append({'budget': budget, 'threshold': None, 'operator': 'never'})
        elif count == len(records):
            thresholds.append({'budget': budget, 'threshold': None, 'operator': 'always'})
        else:
            threshold = float(scores[order[count - 1]])
            operator = 'ge' if int(np.sum(np.asarray(scores) >= threshold)) <= count else 'gt'
            thresholds.append({'budget': budget, 'threshold': threshold, 'operator': operator})
    return thresholds


def fixed_curve(records, endpoint, scores, thresholds):
    s, l = correctness(records, endpoint)
    rescue, harm = (s == 0) & (l == 1), (s == 1) & (l == 0)
    scores = np.asarray(scores)
    points = []
    for item in thresholds:
        if item['operator'] == 'never': upgrade = np.zeros(len(records), dtype=bool)
        elif item['operator'] == 'always': upgrade = np.ones(len(records), dtype=bool)
        elif item['operator'] == 'ge': upgrade = scores >= item['threshold']
        else: upgrade = scores > item['threshold']
        selected = upgrade[:, None]
        points.append({'budget': item['budget'], 'threshold': item['threshold'], 'operator': item['operator'],
                       'actual_fraction': float(upgrade.mean()), 'metric': metric(records, endpoint, upgrade),
                       'rescue': int((rescue & selected).sum()), 'harm': int((harm & selected).sum()),
                       'missed_rescue': int((rescue & ~selected).sum())})
    return {'curve': points, 'selection_split': 'validation'}


def targets(records, endpoint, kind, harm_cost):
    s, l = correctness(records, endpoint)
    classes = np.stack([(s * l).mean(1), ((1 - s) * l).mean(1), (s * (1 - l)).mean(1),
                        ((1 - s) * (1 - l)).mean(1)], axis=1)
    if kind == 'error': y = (1 - s).mean(1, keepdims=True)
    elif kind == 'gain': y = (classes[:, 1] - harm_cost * classes[:, 2])[:, None]
    else: y = classes
    return y, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    dest = Path(args.output_dir)
    if dest.exists() and (dest / 'completion.json').exists():
        raise FileExistsError(('formal run already complete', str(dest)))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'config.json').write_text(json.dumps(cfg, indent=2))
    torch.set_num_threads(int(cfg.get('cpu_threads', 4)))
    torch.manual_seed(int(cfg.get('seed', 42)))
    np.random.seed(int(cfg.get('seed', 42)))
    device = torch.device(args.device)
    records, data = {}, {}
    for split in ['train', 'val', 'test']:
        records[split], data[split] = load_data(cfg, split, device)
    indices = json.loads((WORK / 'data' / cfg['expert'] / 'fraction_indices.json').read_text())[str(cfg['fraction'])]['indices']
    if cfg['fraction'] != 1.0:
        selected = torch.tensor(indices, device=device, dtype=torch.long)
        records['train'] = [records['train'][i] for i in indices]
        data['train'] = {k: v.index_select(0, selected) for k, v in data['train'].items()}
    train_rows = len(records['train'])
    (dest / 'training_rows.json').write_text(json.dumps({'fraction': cfg['fraction'], 'train_rows': train_rows,
        'validation_rows': len(records['val']), 'test_rows': len(records['test']),
        'sample_ids': [r['sample_id'] for r in records['train']]}, indent=2))

    # Fit normalization only on the selected training rows.
    for key in ['hidden', 'confidence']:
        if key not in data['train']:
            continue
        mean = data['train'][key].mean(0)
        std = data['train'][key].std(0, unbiased=False).clamp_min(1e-6)
        for split in data:
            data[split][key] = (data[split][key] - mean) / std
        np.savez(dest / (key + '_normalization.npz'), mean=mean.cpu().numpy(), std=std.cpu().numpy())

    endpoints = ['Qwen3.8', 'MiniCPM'] if cfg.get('endpoint') == 'both' else [cfg['endpoint']]
    hidden_dim = data['train']['hidden'].shape[1] if 'hidden' in data['train'] else 1
    model = FeatureRouter(cfg, hidden_dim, data['train']['confidence'].shape[1], data['train']['output'].shape[1]).to(device)
    y_np, states_np = targets(records['train'], endpoints[0], cfg['target'], cfg.get('harm_cost', 1.0))
    strata = np.concatenate([1 - y_np, y_np], axis=1) if cfg['target'] == 'error' else states_np
    counts = strata.sum(0)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = len(y_np) / (present.sum() * counts[present])
    sample_weights = torch.tensor(strata @ weights, device=device, dtype=torch.float32)
    y = torch.tensor(y_np, device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', .001), weight_decay=cfg.get('weight_decay', .01))
    best = {endpoint: float('-inf') for endpoint in endpoints}
    best_epoch, best_thresholds = {}, {}
    stale, history, started = 0, [], time.time()
    all_train = torch.arange(train_rows, device=device)
    for epoch in range(1, int(cfg.get('max_epochs', 50)) + 1):
        model.train()
        order = torch.multinomial(sample_weights, train_rows, replacement=True) if cfg.get('strategy') == 'balanced' else all_train[torch.randperm(train_rows, device=device)]
        loss_sum = seen = 0
        for b in range(0, len(order), int(cfg.get('batch_size', 256))):
            ids = order[b:b + int(cfg.get('batch_size', 256))]
            optimizer.zero_grad(set_to_none=True)
            output = model(data['train'], ids)
            if cfg['target'] == 'error': loss = F.binary_cross_entropy_with_logits(output, y[ids], reduction='none').mean(1)
            elif cfg['target'] == 'gain': loss = (output - y[ids]).square().mean(1)
            else: loss = -(y[ids] * F.log_softmax(output, dim=1)).sum(1)
            if cfg.get('strategy') == 'weighted': loss = loss * sample_weights[ids]
            loss = loss.mean()
            if not torch.isfinite(loss): raise RuntimeError(('nonfinite training loss', cfg['id'], epoch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(ids); seen += len(ids)
        model.eval(); outputs = []
        with torch.no_grad():
            for b in range(0, len(records['val']), 8192):
                ids = torch.arange(b, min(b + 8192, len(records['val'])), device=device)
                outputs.append(model(data['val'], ids).detach().cpu())
        raw = torch.cat(outputs).numpy()
        scores = ((raw[:, 1] - cfg.get('harm_cost', 1.0) * raw[:, 2]) if cfg['target'] == 'four_state' else
                  (1 / (1 + np.exp(-raw[:, 0])) if cfg['target'] == 'error' else raw[:, 0]))
        results = {endpoint: ranked_curve(records['val'], endpoint, scores) for endpoint in endpoints}
        improved = False
        for endpoint, result in results.items():
            if result['selection_score'] > best[endpoint]:
                improved = True; best[endpoint] = result['selection_score']; best_epoch[endpoint] = epoch
                best_thresholds[endpoint] = validation_thresholds(records['val'], endpoint, scores)
                torch.save({'state_dict': model.state_dict(), 'config': cfg, 'epoch': epoch}, dest / (endpoint + '_best.pt'))
                (dest / (endpoint + '_validation_metrics.json')).write_text(json.dumps(result, indent=2))
                (dest / (endpoint + '_validation_thresholds.json')).write_text(json.dumps(best_thresholds[endpoint], indent=2))
        history.append({'epoch': epoch, 'training_loss': loss_sum / max(1, seen),
                        'validation_selection': {e: r['selection_score'] for e, r in results.items()},
                        'score_std': float(np.asarray(scores).std()), 'elapsed_seconds': time.time() - started})
        with (dest / 'history.jsonl').open('a') as f: f.write(json.dumps(history[-1]) + '\n')
        print(cfg['id'], history[-1], flush=True)
        stale = 0 if improved else stale + 1
        if stale >= int(cfg.get('patience', 8)): break

    test_summary = {}
    for endpoint in endpoints:
        checkpoint = torch.load(dest / (endpoint + '_best.pt'), map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict']); model.eval(); outputs = []
        with torch.no_grad():
            for b in range(0, len(records['test']), 8192):
                ids = torch.arange(b, min(b + 8192, len(records['test'])), device=device)
                outputs.append(model(data['test'], ids).detach().cpu())
        raw = torch.cat(outputs).numpy()
        scores = ((raw[:, 1] - cfg.get('harm_cost', 1.0) * raw[:, 2]) if cfg['target'] == 'four_state' else
                  (1 / (1 + np.exp(-raw[:, 0])) if cfg['target'] == 'error' else raw[:, 0]))
        thresholds = best_thresholds[endpoint]
        test_summary[endpoint] = {'best_epoch': best_epoch[endpoint], 'validation_selection': best[endpoint],
                                   'ranked_test': ranked_curve(records['test'], endpoint, scores),
                                   'fixed_threshold_test': fixed_curve(records['test'], endpoint, scores, thresholds),
                                   'score_std': float(scores.std()), 'score_min': float(scores.min()), 'score_max': float(scores.max())}
        np.save(dest / (endpoint + '_test_scores.npy'), scores)
        (dest / (endpoint + '_test_metrics.json')).write_text(json.dumps(test_summary[endpoint], indent=2))
    result = {'status': 'complete', 'id': cfg['id'], 'fraction': cfg['fraction'], 'epochs': epoch,
              'best_epoch': best_epoch, 'best_validation_selection': best, 'test': test_summary,
              'trainable_parameters': sum(p.numel() for p in model.parameters()), 'seconds': time.time() - started,
              'anomalies': []}
    if any(v['score_std'] < 1e-6 for v in test_summary.values()): result['anomalies'].append('constant test scores')
    (dest / 'completion.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': 'complete', 'id': cfg['id'], 'seconds': result['seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
