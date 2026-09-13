"""Select validation thresholds and the shared fusion alpha without opening test data."""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from method_improvement_common import (
    FeatureRouter,
    confidence_route,
    forward_logits,
    load_checkpoint,
    load_split,
    metric,
    read_json,
    read_rows,
    score_from_logits,
    select_threshold,
    threshold_curve,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    return parser.parse_args()


def apply_saved_normalization(data, trajectory_dir):
    for key in ('hidden', 'confidence'):
        if key not in data:
            continue
        normalization = np.load(trajectory_dir / f'{key}_normalization.npz')
        mean = torch.from_numpy(normalization['mean']).to(data[key].device)
        std = torch.from_numpy(normalization['std']).to(data[key].device)
        data[key] = (data[key] - mean) / std


def score_file(condition_dir, mode):
    logits = np.load(condition_dir / 'validation_logits.npy')
    scores = score_from_logits(logits, mode)
    saved = np.load(condition_dir / 'validation_scores.npy')
    if len(saved) != len(scores) or not np.allclose(saved, scores, rtol=0.0, atol=1e-12):
        raise ValueError(('saved validation score mismatch', str(condition_dir), mode))
    return logits, scores


def save_validation_selection(directory, records, endpoint, scores, method, source_condition=None, endpoint_reused=False, tie_epsilon=1e-12, score_mode=None):
    directory.mkdir(parents=True, exist_ok=True)
    historical, historical_curve = select_threshold(
        records, endpoint, scores, [i / 100 for i in range(1, 100)], tie_epsilon=tie_epsilon,
    )
    endpoint_point, endpoint_curve = select_threshold(
        records, endpoint, scores, [i / 100 for i in range(0, 101)], tie_epsilon=tie_epsilon,
    )
    result = {
        'method': method,
        'source_condition': source_condition,
        'score_mode': score_mode or ('c_route' if method == 'confidence' else 's_gain' if method.endswith('gain') or method == 'both' else 's_log'),
        'endpoint_reused': endpoint_reused,
        'selection_protocol': {
            'historical': 'validation candidates 1%..99%; choose highest task metric, then lower actual calls, then lower nominal budget',
            'endpoint': 'validation candidates 0%..100%; choose highest task metric, then lower actual calls, then lower nominal budget',
        },
        'selected': {'historical': historical, 'endpoint': endpoint_point},
        'historical_curve': historical_curve,
        'endpoint_curve': endpoint_curve,
        'validation_rows': len(records),
        'test_loaded': False,
    }
    write_json(directory / 'validation_selection.json', result)
    return result


def save_endpoint_selection(directory, records, endpoint, method, tie_epsilon=1e-12):
    directory.mkdir(parents=True, exist_ok=True)
    selected_point = {
        'budget': 0.0 if method == 'small' else 1.0,
        'threshold': None,
        'operator': 'never' if method == 'small' else 'always',
        'actual_fraction': 0.0 if method == 'small' else 1.0,
        'metric': metric(records, endpoint, np.zeros(len(records), dtype=bool) if method == 'small' else np.ones(len(records), dtype=bool)),
    }
    result = {
        'method': method,
        'selected': {
            'historical': selected_point,
            'endpoint': selected_point,
        },
        'historical_curve': [selected_point],
        'endpoint_curve': [selected_point],
        'validation_rows': len(records),
        'test_loaded': False,
    }
    write_json(directory / 'validation_selection.json', result)
    return result


def trajectory_ids(spec):
    return [
        f'{expert}_{endpoint}_seed{seed}'
        for seed in spec['seeds']
        for expert in spec['experts']
        for endpoint in ('MiniCPM', 'Qwen3.8')
    ]


def main():
    args = parse_args()
    spec = read_json(args.spec)
    output = args.output
    if (output / 'validation_selection_complete.json').exists():
        raise FileExistsError(output / 'validation_selection_complete.json')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    all_trajectories = trajectory_ids(spec)
    conditions = {item['name']: item for item in spec['conditions']}
    records_cache = {}
    method_entries = []
    for trajectory_id in all_trajectories:
        trajectory_dir = output / 'trajectories' / trajectory_id
        trajectory_done = read_json(trajectory_dir / 'trajectory_completion.json')
        if trajectory_done.get('status') != 'complete':
            raise ValueError(('trajectory incomplete', trajectory_id))
        cfg = read_json(trajectory_dir / 'config.json')
        expert, endpoint, seed, task = cfg['expert'], cfg['endpoint'], int(cfg['seed']), cfg['task']
        records = read_rows(args.root / spec['data_work'] / 'data' / expert / 'val' / 'records.jsonl')
        expected_rows = spec['experts'][expert]['expected_rows']['val']
        if len(records) != expected_rows:
            raise ValueError(('validation row count mismatch', trajectory_id, len(records), expected_rows))
        records_cache[(expert, endpoint)] = records
        confidence, _, _, columns = confidence_route(args.root, spec, expert, 'val')
        confidence_dir = output / 'baselines' / f'confidence_{expert}_{endpoint}'
        if not (confidence_dir / 'validation_selection.json').exists():
            confidence_dir.mkdir(parents=True, exist_ok=True)
            np.save(confidence_dir / 'validation_scores.npy', confidence)
            write_json(confidence_dir / 'validation_sample_ids.json', [row['sample_id'] for row in records])
            write_json(confidence_dir / 'config.json', {
                'method': 'confidence',
                'expert': expert,
                'task': task,
                'endpoint': endpoint,
                'confidence_columns': columns,
                'confidence_definition': 'where(valid, -selected_confidence, 1.0)',
                'seed_variation': False,
            })
            save_validation_selection(confidence_dir, records, endpoint, confidence, 'confidence', tie_epsilon=spec['tie_epsilon'])
        confidence_result = read_json(confidence_dir / 'validation_selection.json')
        method_entries.append({
            'method': 'confidence',
            'method_id': f'confidence_{expert}_{endpoint}_seed_invariant',
            'trajectory_id': trajectory_id,
            'condition_dir': str(confidence_dir),
            'expert': expert,
            'task': task,
            'endpoint': endpoint,
            'seed': None,
            'selection': confidence_result,
        })
        for condition in spec['conditions']:
            name = condition['name']
            condition_id = f'{name}_{trajectory_id}'
            condition_dir = output / 'conditions' / condition_id
            logits, scores = score_file(condition_dir, condition['score_mode'])
            selection = save_validation_selection(
                condition_dir,
                records,
                endpoint,
                scores,
                name,
                tie_epsilon=spec['tie_epsilon'],
            )
            write_json(condition_dir / 'validation_sample_ids.json', [row['sample_id'] for row in records])
            method_entries.append({
                'method': name,
                'method_id': condition_id,
                'trajectory_id': trajectory_id,
                'condition_dir': str(condition_dir),
                'expert': expert,
                'task': task,
                'endpoint': endpoint,
                'seed': seed,
                'selection': selection,
            })
        score_only_dir = output / 'derived' / 'score_only_gain' / trajectory_id
        if not (score_only_dir / 'validation_selection.json').exists():
            original_logits = np.load(output / 'conditions' / f'original_{trajectory_id}' / 'validation_logits.npy')
            score_only = score_from_logits(original_logits, 's_gain')
            score_only_dir.mkdir(parents=True, exist_ok=True)
            np.save(score_only_dir / 'validation_logits.npy', original_logits)
            np.save(score_only_dir / 'validation_scores.npy', score_only)
            write_json(score_only_dir / 'config.json', {
                'method': 'score_only_gain',
                'source_condition': 'original',
                'trajectory_id': trajectory_id,
                'expert': expert,
                'task': task,
                'endpoint': endpoint,
                'seed': seed,
                'test_loaded': False,
            })
            write_json(score_only_dir / 'validation_sample_ids.json', [row['sample_id'] for row in records])
            save_validation_selection(
                score_only_dir,
                records,
                endpoint,
                score_only,
                'score_only_gain',
                source_condition='original',
                endpoint_reused=True,
                tie_epsilon=spec['tie_epsilon'],
            )
        score_only_result = read_json(score_only_dir / 'validation_selection.json')
        method_entries.append({
            'method': 'score_only_gain',
            'method_id': f'score_only_gain_{trajectory_id}',
            'trajectory_id': trajectory_id,
            'condition_dir': str(score_only_dir),
            'expert': expert,
            'task': task,
            'endpoint': endpoint,
            'seed': seed,
            'selection': score_only_result,
        })
        for endpoint_method, mask_value in (('small', False), ('large', True)):
            endpoint_dir = output / 'baselines' / f'{endpoint_method}_{expert}_{endpoint}'
            if not (endpoint_dir / 'validation_selection.json').exists():
                endpoint_dir.mkdir(parents=True, exist_ok=True)
                write_json(endpoint_dir / 'config.json', {
                    'method': endpoint_method,
                    'expert': expert,
                    'task': task,
                    'endpoint': endpoint,
                    'seed_variation': False,
                    'test_loaded': False,
                })
                save_endpoint_selection(endpoint_dir, records, endpoint, endpoint_method, spec['tie_epsilon'])
            endpoint_result = read_json(endpoint_dir / 'validation_selection.json')
            method_entries.append({
                'method': endpoint_method,
                'method_id': f'{endpoint_method}_{expert}_{endpoint}_seed_invariant',
                'trajectory_id': trajectory_id,
                'condition_dir': str(endpoint_dir),
                'expert': expert,
                'task': task,
                'endpoint': endpoint,
                'seed': None,
                'selection': endpoint_result,
            })
    if len(all_trajectories) != 48:
        raise AssertionError(len(all_trajectories))
    condition_count = sum(1 for entry in method_entries if entry['method'] in conditions)
    if condition_count != 192:
        raise AssertionError(('logical condition count', condition_count))

    fusion_entries = []
    for trajectory_id in all_trajectories:
        trajectory_dir = output / 'trajectories' / trajectory_id
        cfg = read_json(trajectory_dir / 'config.json')
        expert, endpoint, task, seed = cfg['expert'], cfg['endpoint'], cfg['task'], int(cfg['seed'])
        records = records_cache[(expert, endpoint)]
        both_dir = output / 'conditions' / f'both_{trajectory_id}'
        both_logits = np.load(both_dir / 'validation_logits.npy')
        val_gain = score_from_logits(both_logits, 's_gain')
        val_conf, _, _, columns = confidence_route(args.root, spec, expert, 'val')
        train_records, train_data = load_split(args.root, spec, cfg, 'train', device)
        apply_saved_normalization(train_data, trajectory_dir)
        model = FeatureRouter(cfg, train_data['hidden'].shape[1], train_data['confidence'].shape[1], train_data['output'].shape[1]).to(device)
        checkpoint = load_checkpoint(both_dir / 'best.pt', device)
        model.load_state_dict(checkpoint['state_dict'])
        train_logits = forward_logits(model, train_data, len(train_records), device)
        train_gain = score_from_logits(train_logits, 's_gain')
        train_conf, _, _, _ = confidence_route(args.root, spec, expert, 'train')
        if len(train_gain) != len(train_conf):
            raise ValueError(('fusion train length mismatch', trajectory_id))
        gain_mean, gain_std = float(train_gain.mean()), float(train_gain.std())
        conf_mean, conf_std = float(train_conf.mean()), float(train_conf.std())
        gain_std = max(gain_std, 1e-12)
        conf_std = max(conf_std, 1e-12)
        np.savez(trajectory_dir / 'fusion_stats.npz', gain_mean=gain_mean, gain_std=gain_std, confidence_mean=conf_mean, confidence_std=conf_std)
        write_json(trajectory_dir / 'fusion_stats.json', {
            'train_rows': len(train_records),
            's_gain_mean': gain_mean,
            's_gain_std': gain_std,
            'c_route_mean': conf_mean,
            'c_route_std': conf_std,
            'confidence_columns': columns,
            'fit_on_split': 'train',
            'test_loaded': False,
        })
        del model, checkpoint, train_data, train_records, train_logits, train_gain
        val_scores_by_alpha = {}
        for alpha in spec['fusion_alphas']:
            if alpha == 0.0:
                scores = val_conf.copy()
                endpoint_reused = True
            elif alpha == 1.0:
                scores = val_gain.copy()
                endpoint_reused = True
            else:
                scores = alpha * ((val_gain - gain_mean) / gain_std) + (1.0 - alpha) * ((val_conf - conf_mean) / conf_std)
                endpoint_reused = False
            alpha_label = f'alpha_{alpha:g}'
            fusion_dir = output / 'derived' / 'fusion' / f'{alpha_label}_{trajectory_id}'
            fusion_dir.mkdir(parents=True, exist_ok=True)
            np.save(fusion_dir / 'validation_scores.npy', scores)
            write_json(fusion_dir / 'config.json', {
                'method': 'fusion',
                'alpha': alpha,
                'checkpoint_source': f'both_{trajectory_id}',
                'trajectory_id': trajectory_id,
                'expert': expert,
                'task': task,
                'endpoint': endpoint,
                'seed': seed,
                'score_formula': 'alpha*standardized(s_gain)+(1-alpha)*standardized(c_route)' if 0.0 < alpha < 1.0 else 'reuse original endpoint score strategy',
                'endpoint_reused': endpoint_reused,
                'test_loaded': False,
            })
            write_json(fusion_dir / 'validation_sample_ids.json', [row['sample_id'] for row in records])
            selection = save_validation_selection(
                fusion_dir,
                records,
                endpoint,
                scores,
                f'fusion_alpha_{alpha:g}',
                source_condition=f'both_{trajectory_id}',
                endpoint_reused=endpoint_reused,
                tie_epsilon=spec['tie_epsilon'],
                score_mode='c_route' if alpha == 0.0 else 's_gain' if alpha == 1.0 else 's_mix',
            )
            fusion_entries.append({
                'method': 'fusion',
                'method_id': f'fusion_alpha_{alpha:g}_{trajectory_id}',
                'alpha': alpha,
                'trajectory_id': trajectory_id,
                'condition_dir': str(fusion_dir),
                'expert': expert,
                'task': task,
                'endpoint': endpoint,
                'seed': seed,
                'selection': selection,
            })

    alpha_lock = {}
    task_names = ['cub', 'grefcoco', 'nlvr2', 'construction']
    for rule in ('historical', 'endpoint'):
        candidates = {}
        for alpha in spec['fusion_alphas']:
            rows = [entry for entry in fusion_entries if entry['alpha'] == alpha]
            by_task = defaultdict(list)
            for row in rows:
                point = row['selection']['selected'][rule]
                by_task[row['task']].append(point)
            task_stats = {}
            for task_name in task_names:
                points = by_task[task_name]
                if len(points) != 12:
                    raise AssertionError(('fusion task row count', rule, alpha, task_name, len(points)))
                task_stats[task_name] = {
                    'mean_validation_metric': float(np.mean([point['metric'] for point in points])),
                    'mean_validation_actual_fraction': float(np.mean([point['actual_fraction'] for point in points])),
                    'rows': len(points),
                }
            overall_metric = float(np.mean([task_stats[task_name]['mean_validation_metric'] for task_name in task_names]))
            overall_actual = float(np.mean([task_stats[task_name]['mean_validation_actual_fraction'] for task_name in task_names]))
            candidates[str(alpha)] = {
                'alpha': alpha,
                'task_stats': task_stats,
                'equal_task_mean_validation_metric': overall_metric,
                'equal_task_mean_validation_actual_fraction': overall_actual,
            }
        selected = max(
            candidates.values(),
            key=lambda item: (
                item['equal_task_mean_validation_metric'],
                -item['equal_task_mean_validation_actual_fraction'],
                -item['alpha'],
            ),
        )
        alpha_lock[rule] = {'selected_alpha': selected['alpha'], 'candidates': candidates, 'test_loaded': False}
    write_json(output / 'fusion_alpha_selection.json', alpha_lock)
    write_json(output / 'validation_manifest.json', {
        'status': 'validation_complete',
        'training_trajectories': len(all_trajectories),
        'logical_training_conditions': condition_count,
        'methods_with_validation_selection': len(method_entries),
        'fusion_entries': len(fusion_entries),
        'entries': method_entries,
        'fusion_entries_manifest': fusion_entries,
        'alpha_selection_file': 'fusion_alpha_selection.json',
        'test_loaded': False,
    })
    write_json(output / 'validation_selection_complete.json', {
        'status': 'complete',
        'finished_at': time.time(),
        'logical_training_conditions': condition_count,
        'test_loaded': False,
        'thresholds_locked_on': 'validation',
        'fusion_alphas_locked_on': 'validation',
    })
    print('VALIDATION_SELECTION_COMPLETE', condition_count, flush=True)


if __name__ == '__main__':
    main()
