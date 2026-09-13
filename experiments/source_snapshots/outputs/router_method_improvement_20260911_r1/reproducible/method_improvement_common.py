"""Common data, model, metric, and validation-selection code for the 2026-09-11 study."""

from collections import defaultdict
from pathlib import Path

import json
import math
import numpy as np
import torch
from torch import nn


EVENTS = [
    'rule_1_ppe_violation',
    'rule_2_fall_protection_violation',
    'rule_3_unprotected_edge_violation',
    'rule_4_excavator_proximity_violation',
]


class FeatureRouter(nn.Module):
    def __init__(self, cfg, hidden_dim, confidence_dim, output_dim):
        super().__init__()
        self.branches = cfg['branches']
        self.fusion = cfg['fusion']
        d = cfg['representation_dim']
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, d), nn.LayerNorm(d))
            for name, dim in [
                ('text', 768),
                ('hidden', hidden_dim),
                ('confidence', confidence_dim),
                ('output', output_dim),
            ]
            if name in self.branches
        })
        self.head = nn.Sequential(
            nn.Linear(len(self.branches) * d, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )
        if 'image' in self.branches:
            if self.fusion != 'concat':
                raise ValueError(('method-improvement study requires concat fusion', self.fusion))
            self.image_projection = nn.Linear(1538, d)
            self.image_norm = nn.LayerNorm(d)

    def forward(self, data, idx):
        fused = []
        for branch in self.branches:
            if branch == 'image':
                images, valid = data['image'][idx], data['valid'][idx]
                x = self.image_projection(torch.cat([images.flatten(1), valid.float()], dim=1))
                fused.append(self.image_norm(x))
            else:
                fused.append(self.projections[branch](data[branch][idx]))
        return self.head(torch.cat(fused, dim=1))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def make_config(spec, expert, endpoint, seed, trajectory_id):
    expert_spec = spec['experts'][expert]
    cfg = dict(spec['fixed'])
    cfg.update({
        'student': spec['student'],
        'expert': expert,
        'task': expert_spec['task'],
        'endpoint': endpoint,
        'seed': seed,
        'hidden_layer': expert_spec['hidden_layer'],
        'branches': expert_spec['branches'],
        'trajectory_id': trajectory_id,
        'stage': 'method_improvement_20260911',
    })
    return cfg


def hidden_values(data_work, expert, split, layer):
    positions = read_json(data_work / 'data' / expert / split / 'index.json')
    values = None
    by_shard = defaultdict(list)
    for out_index, item in enumerate(positions):
        by_shard[item['shard']].append((out_index, item['position']))
    for shard, selected in by_shard.items():
        with np.load(Path(shard) / 'states.npz') as states:
            source = states[layer]
            if values is None:
                values = np.empty((len(positions), source.shape[1]), dtype=np.float32)
            out_indices, source_indices = zip(*selected)
            picked = np.asarray(source[list(source_indices)], dtype=np.float32)
            if not np.isfinite(picked).all():
                raise RuntimeError(('nonfinite hidden state', expert, split, layer, shard))
            values[list(out_indices)] = picked
    if values is None:
        raise ValueError(('empty hidden index', expert, split, layer))
    return values


def tensor_from_npy(path, device):
    values = np.load(path, mmap_mode='r')
    if not np.isfinite(values).all():
        raise RuntimeError(('nonfinite input array', str(path)))
    return torch.from_numpy(values).to(device=device, dtype=torch.float32)


def load_split(root, spec, cfg, split, device):
    data_work = root / spec['data_work']
    expert, task = cfg['expert'], cfg['task']
    split_dir = data_work / 'data' / expert / split
    rows = read_rows(split_dir / 'records.jsonl')
    feature_dir = data_work / 'features' / cfg['student'] / task / split
    feature_ids = read_json(feature_dir / 'sample_ids.json')
    row_ids = [row['sample_id'] for row in rows]
    if feature_ids != row_ids:
        raise ValueError(('student feature identity mismatch', expert, split))
    data = {
        'image': tensor_from_npy(feature_dir / 'image.npy', device),
        'valid': torch.from_numpy(np.load(feature_dir / 'valid.npy', mmap_mode='r')).to(device=device, dtype=torch.bool),
    }
    for branch in cfg['branches']:
        if branch in ('image', 'hidden'):
            continue
        data[branch] = tensor_from_npy(
            feature_dir / f'{branch}.npy' if branch in ('text',) else split_dir / f'{branch}.npy',
            device,
        )
    hidden = hidden_values(data_work, expert, split, cfg['hidden_layer'])
    data['hidden'] = torch.from_numpy(hidden).to(device=device, dtype=torch.float32)
    if any(len(value) != len(rows) for value in data.values()):
        raise ValueError(('input row count mismatch', expert, split, {key: len(value) for key, value in data.items()}, len(rows)))
    return rows, data


def normalize_splits(data_by_split, destination):
    for key in ('hidden', 'confidence'):
        if key not in data_by_split['train']:
            continue
        mean = data_by_split['train'][key].mean(0)
        std = data_by_split['train'][key].std(0, unbiased=False).clamp_min(1e-6)
        for split in data_by_split:
            data_by_split[split][key] = (data_by_split[split][key] - mean) / std
        np.savez(destination / f'{key}_normalization.npz', mean=mean.cpu().numpy(), std=std.cpu().numpy())


def forward_logits(model, data, row_count, device, batch_size=8192):
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, row_count, batch_size):
            indices = torch.arange(start, min(start + batch_size, row_count), device=device)
            chunks.append(model(data, indices).detach().cpu())
    logits = torch.cat(chunks).numpy()
    if not np.isfinite(logits).all():
        raise RuntimeError('nonfinite router logits')
    return logits


def correctness_arrays(records, endpoint):
    small = [row['small_label'] for row in records]
    large = [row['large_labels'][endpoint] for row in records]
    if 'events' not in small[0]:
        return np.asarray([row['correct'] for row in small], dtype=bool), np.asarray([row['correct'] for row in large], dtype=bool)
    small_values = np.asarray([[row['events'][event]['correct'] for event in EVENTS] for row in small], dtype=bool)
    large_values = np.asarray([[row['events'][event]['correct'] for event in EVENTS] for row in large], dtype=bool)
    return small_values, large_values


def metric(records, endpoint, upgrade):
    small, large = correctness_arrays(records, endpoint)
    chosen = np.where(upgrade[:, None], large, small) if small.ndim == 2 else np.where(upgrade, large, small)
    if chosen.ndim == 1:
        return float(chosen.mean())
    labels = [row['large_labels'][endpoint] if upgrade[index] else row['small_label'] for index, row in enumerate(records)]
    f1 = []
    for event in EVENTS:
        target = np.asarray([label['events'][event]['target'] for label in labels], dtype=bool)
        positive = np.asarray([label['events'][event]['prediction'] is True for label in labels], dtype=bool)
        tp = np.sum(target & positive)
        fp = np.sum(~target & positive)
        fn = np.sum(target & ~positive)
        denominator = 2 * tp + fp + fn
        f1.append(float(2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(f1))


def rescue_harm_counts(records, endpoint, upgrade):
    small, large = correctness_arrays(records, endpoint)
    if small.ndim == 1:
        selected = upgrade
        rescue = int(np.sum((~small) & large & selected))
        harm = int(np.sum(small & (~large) & selected))
        missed = int(np.sum((~small) & large & (~selected)))
        wasted = int(np.sum((small == large) & selected))
    else:
        selected = upgrade[:, None]
        rescue = int(np.sum((~small) & large & selected))
        harm = int(np.sum(small & (~large) & selected))
        missed = int(np.sum((~small) & large & (~selected)))
        wasted = int(np.sum((small == large) & selected))
    return {'rescue': rescue, 'harm': harm, 'missed_rescue': missed, 'wasted_events': wasted}


def ranked_mask(scores, budget):
    scores = np.asarray(scores)
    count = int(np.floor(len(scores) * budget))
    order = np.argsort(-scores, kind='stable')
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:count]] = True
    return mask


def ranked_point(records, endpoint, scores, budget):
    mask = ranked_mask(scores, budget)
    point = {
        'budget': float(budget),
        'threshold': None,
        'operator': 'ranked',
        'actual_fraction': float(mask.mean()),
        'metric': metric(records, endpoint, mask),
    }
    point.update(rescue_harm_counts(records, endpoint, mask))
    return point


def ranked_curve(records, endpoint, scores, budgets):
    points = [ranked_point(records, endpoint, scores, budget) for budget in budgets]
    area_points = [point for point in points if point['budget'] <= 0.25]
    area = float(sum(
        (left['metric'] + right['metric']) * 0.5 * (right['budget'] - left['budget'])
        for left, right in zip(area_points[:-1], area_points[1:])
    ) / 0.25)
    return {'curve': points, 'selection_score': area, 'score_std': float(np.asarray(scores).std())}


def validation_thresholds(scores, budgets):
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind='stable')
    thresholds = []
    for budget in budgets:
        count = int(np.floor(len(scores) * budget))
        if count == 0:
            thresholds.append({'budget': float(budget), 'threshold': None, 'operator': 'never'})
        elif count == len(scores):
            thresholds.append({'budget': float(budget), 'threshold': None, 'operator': 'always'})
        else:
            threshold = float(scores[order[count - 1]])
            operator = 'ge' if int(np.sum(scores >= threshold)) <= count else 'gt'
            thresholds.append({'budget': float(budget), 'threshold': threshold, 'operator': operator})
    return thresholds


def threshold_mask(scores, point):
    scores = np.asarray(scores)
    if point['operator'] == 'never':
        return np.zeros(len(scores), dtype=bool)
    if point['operator'] == 'always':
        return np.ones(len(scores), dtype=bool)
    if point['operator'] == 'ge':
        return scores >= point['threshold']
    if point['operator'] == 'gt':
        return scores > point['threshold']
    raise ValueError(('unknown threshold operator', point['operator']))


def threshold_curve(records, endpoint, scores, budgets):
    points = []
    for threshold in validation_thresholds(scores, budgets):
        mask = threshold_mask(scores, threshold)
        point = dict(threshold)
        point['actual_fraction'] = float(mask.mean())
        point['metric'] = metric(records, endpoint, mask)
        point.update(rescue_harm_counts(records, endpoint, mask))
        points.append(point)
    return points


def select_threshold(records, endpoint, scores, budgets, tie_epsilon=1e-12):
    candidates = threshold_curve(records, endpoint, scores, budgets)
    selected = candidates[0]
    for candidate in candidates[1:]:
        if candidate['metric'] > selected['metric'] + tie_epsilon:
            selected = candidate
        elif abs(candidate['metric'] - selected['metric']) <= tie_epsilon:
            if candidate['actual_fraction'] < selected['actual_fraction'] - tie_epsilon:
                selected = candidate
            elif abs(candidate['actual_fraction'] - selected['actual_fraction']) <= tie_epsilon and candidate['budget'] < selected['budget']:
                selected = candidate
    return dict(selected), candidates


def select_ranked_candidate(records, endpoint, scores, budgets, tie_epsilon=1e-12):
    candidates = [ranked_point(records, endpoint, scores, budget) for budget in budgets]
    selected = candidates[0]
    for candidate in candidates[1:]:
        if candidate['metric'] > selected['metric'] + tie_epsilon:
            selected = candidate
        elif abs(candidate['metric'] - selected['metric']) <= tie_epsilon:
            if candidate['actual_fraction'] < selected['actual_fraction'] - tie_epsilon:
                selected = candidate
            elif abs(candidate['actual_fraction'] - selected['actual_fraction']) <= tie_epsilon and candidate['budget'] < selected['budget']:
                selected = candidate
    return dict(selected), candidates


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def score_from_logits(logits, mode):
    logits = np.asarray(logits)
    if mode == 's_log':
        scores = logits[:, 1] - logits[:, 2]
    elif mode == 's_gain':
        probabilities = softmax(logits)
        scores = probabilities[:, 1] - probabilities[:, 2]
    else:
        raise ValueError(('unknown score mode', mode))
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError(('nonfinite route scores', mode))
    return scores


def confidence_route(root, spec, expert, split):
    data_work = root / spec['data_work']
    expert_spec = spec['experts'][expert]
    source = data_work / 'data' / expert / split
    confidence = np.load(source / 'confidence.npy', mmap_mode='r')
    metadata_path = root / 'outputs/router_exploration_25pct_20260910' / 'datasets' / expert / 'completion.json'
    columns = read_json(metadata_path)['confidence_columns']
    value_index = columns.index(expert_spec['confidence_column'])
    valid_index = columns.index(expert_spec['confidence_valid_column'])
    values = np.asarray(confidence[:, value_index], dtype=np.float64)
    valid = np.asarray(confidence[:, valid_index], dtype=bool)
    if not np.isfinite(values).all():
        raise RuntimeError(('nonfinite confidence', expert, split))
    route = np.where(valid, -values, 1.0)
    return route, values, valid, columns


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return checkpoint


def copy_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def group_name(record):
    sample_id = str(record['sample_id'])
    split = sample_id.split('-', 1)[0]
    if split not in ('testA', 'testB'):
        split = str(record.get('split', split))
    label = record.get('ground_truth', {})
    target_type = label.get('target_type')
    if target_type is None:
        target_type = record.get('target_type')
    if target_type is None:
        target_type = 'unknown'
    return split, target_type


def selected_metrics(records, endpoint, mask):
    return {
        'metric': metric(records, endpoint, mask),
        'actual_fraction': float(mask.mean()),
        **rescue_harm_counts(records, endpoint, mask),
    }
