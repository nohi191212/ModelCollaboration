"""Evaluate locked checkpoints and thresholds on the test split only after validation is frozen."""

import argparse
import gc
import time
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
    rescue_harm_counts,
    score_from_logits,
    threshold_mask,
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


def score_test_point(records, endpoint, scores, validation_point):
    mask = threshold_mask(scores, validation_point)
    point = dict(validation_point)
    point.update({
        'validation_metric': validation_point['metric'],
        'validation_actual_fraction': validation_point['actual_fraction'],
        'metric': metric(records, endpoint, mask),
        'actual_fraction': float(mask.mean()),
    })
    point.update(rescue_harm_counts(records, endpoint, mask))
    return point, mask


def test_curve(records, endpoint, scores, curve):
    points = []
    for validation_point in curve:
        point, _ = score_test_point(records, endpoint, scores, validation_point)
        points.append(point)
    return points


def batch_curve(records, endpoint, scores, budgets):
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind='stable')
    points = []
    for budget in budgets:
        count = int(np.floor(len(records) * budget))
        mask = np.zeros(len(records), dtype=bool)
        mask[order[:count]] = True
        point = {
            'budget': float(budget),
            'actual_fraction': float(mask.mean()),
            'metric': metric(records, endpoint, mask),
        }
        point.update(rescue_harm_counts(records, endpoint, mask))
        points.append(point)
    return points


def write_method_arrays(directory, scores, masks, sample_ids, logits=None):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / 'test_scores.npy', np.asarray(scores, dtype=np.float64))
    if logits is not None:
        np.save(directory / 'test_logits.npy', np.asarray(logits, dtype=np.float32))
    for rule, mask in masks.items():
        np.save(directory / f'test_selection_mask_{rule}.npy', mask.astype(bool))
    write_json(directory / 'test_sample_ids.json', sample_ids)


def deployment_record(method, method_id, entry, records, endpoint, scores, directory, logits=None, batch_budgets=None):
    selection = entry['selection']
    deployments = {}
    masks = {}
    for rule, curve_key in (('historical', 'historical_curve'), ('endpoint', 'endpoint_curve')):
        validation_point = selection['selected'][rule]
        point, mask = score_test_point(records, endpoint, scores, validation_point)
        deployments[rule] = {
            'validation_point': validation_point,
            'test_point': point,
            'test_curve': test_curve(records, endpoint, scores, selection[curve_key]),
        }
        masks[rule] = mask
    write_method_arrays(directory, scores, masks, [row['sample_id'] for row in records], logits)
    result = {
        'method': method,
        'method_id': method_id,
        'task': entry['task'],
        'expert': entry['expert'],
        'endpoint': endpoint,
        'seed': entry.get('seed'),
        'trajectory_id': entry.get('trajectory_id'),
        'config_dir': str(directory),
        'validation_rows': int(entry['selection']['validation_rows']),
        'test_rows': len(records),
        'deployments': deployments,
        'score_summary': {
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
        },
        'batch_sorting_diagnostic': batch_curve(records, endpoint, scores, batch_budgets) if batch_budgets is not None else None,
        'test_loaded': True,
    }
    write_json(directory / 'test_result.json', result)
    return result, masks


def subgroup_metrics(records, endpoint, mask):
    groups = {}
    for index, record in enumerate(records):
        split, target_type = _group_name(record)
        groups.setdefault((split, target_type), []).append(index)
    output = {}
    for (split, target_type), indices in sorted(groups.items()):
        indices = np.asarray(indices, dtype=int)
        local_records = [records[index] for index in indices]
        local_mask = np.asarray(mask)[indices]
        output[f'{split}|{target_type}'] = {
            'split': split,
            'target_type': target_type,
            'rows': len(indices),
            'metric': metric(local_records, endpoint, local_mask),
            'actual_call_fraction_within_group': float(local_mask.mean()),
            'selected_rows': int(local_mask.sum()),
        }
    return output


def _group_name(record):
    sample_id = str(record['sample_id'])
    split = sample_id.split('-', 1)[0]
    if split not in ('testA', 'testB'):
        split = str(record.get('split', split))
    target_type = record.get('ground_truth', {}).get('target_type')
    if target_type is None:
        target_type = record.get('target_type', 'unknown')
    return split, target_type


def main():
    args = parse_args()
    spec = read_json(args.spec)
    output = args.output
    if not (output / 'validation_selection_complete.json').exists():
        raise FileNotFoundError('validation selection is not locked')
    if (output / 'test_evaluation_complete.json').exists():
        raise FileExistsError(output / 'test_evaluation_complete.json')
    lock = read_json(output / 'validation_selection_complete.json')
    if lock.get('test_loaded'):
        raise ValueError('validation lock already touched test')
    alpha_lock = read_json(output / 'fusion_alpha_selection.json')
    manifest = read_json(output / 'validation_manifest.json')
    entries = manifest['entries']
    fusion_entries = manifest['fusion_entries_manifest']
    route_entries = {
        (entry['method'], entry['trajectory_id']): entry
        for entry in entries
        if entry['method'] in {condition['name'] for condition in spec['conditions']}
    }
    score_only_entries = {entry['trajectory_id']: entry for entry in entries if entry['method'] == 'score_only_gain'}
    confidence_entries = {}
    endpoint_entries = {}
    for entry in entries:
        if entry['method'] == 'confidence':
            confidence_entries[(entry['expert'], entry['endpoint'])] = entry
        elif entry['method'] in ('small', 'large'):
            endpoint_entries[(entry['method'], entry['expert'], entry['endpoint'])] = entry
    fusion_lookup = {(entry['alpha'], entry['trajectory_id']): entry for entry in fusion_entries}
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    all_results = []
    subgroup_results = []
    trajectories = sorted({entry['trajectory_id'] for entry in entries if entry.get('seed') is not None})
    batch_budgets = spec['budgets']['batch_diagnostic']
    started = time.perf_counter()
    for trajectory_id in trajectories:
        trajectory_dir = output / 'trajectories' / trajectory_id
        cfg = read_json(trajectory_dir / 'config.json')
        expert, endpoint, task, seed = cfg['expert'], cfg['endpoint'], cfg['task'], int(cfg['seed'])
        records, data = load_split(args.root, spec, cfg, 'test', device)
        expected = spec['experts'][expert]['expected_rows']['test']
        if len(records) != expected:
            raise ValueError(('test row count mismatch', trajectory_id, len(records), expected))
        apply_saved_normalization(data, trajectory_dir)
        raw_by_condition = {}
        score_by_condition = {}
        for condition in spec['conditions']:
            name = condition['name']
            entry = route_entries[(name, trajectory_id)]
            condition_dir = output / 'conditions' / f'{name}_{trajectory_id}'
            checkpoint = load_checkpoint(condition_dir / 'best.pt', device)
            model = FeatureRouter(cfg, data['hidden'].shape[1], data['confidence'].shape[1], data['output'].shape[1]).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            raw = forward_logits(model, data, len(records), device)
            scores = score_from_logits(raw, condition['score_mode'])
            raw_by_condition[name] = raw
            score_by_condition[name] = scores
            result, masks = deployment_record(
                name,
                entry['method_id'],
                entry,
                records,
                endpoint,
                scores,
                condition_dir,
                logits=raw,
                batch_budgets=batch_budgets,
            )
            result['checkpoint_epoch'] = read_json(condition_dir / 'selection_record.json')['best_epoch']
            all_results.append(result)
            if task == 'grefcoco':
                subgroup_results.append({
                    'method': name,
                    'method_id': entry['method_id'],
                    'trajectory_id': trajectory_id,
                    'seed': seed,
                    'rule': 'historical',
                    'groups': subgroup_metrics(records, endpoint, masks['historical']),
                })
            del model, checkpoint, raw
            gc.collect()
        original_entry = score_only_entries[trajectory_id]
        score_only_dir = output / 'derived' / 'score_only_gain' / trajectory_id
        score_only_raw = raw_by_condition['original']
        score_only_scores = score_from_logits(score_only_raw, 's_gain')
        score_only_result, score_only_masks = deployment_record(
            'score_only_gain',
            original_entry['method_id'],
            original_entry,
            records,
            endpoint,
            score_only_scores,
            score_only_dir,
            logits=score_only_raw,
            batch_budgets=batch_budgets,
        )
        score_only_result['score_only_reuses_original_checkpoint'] = True
        all_results.append(score_only_result)
        if task == 'grefcoco':
            subgroup_results.append({
                'method': 'score_only_gain',
                'method_id': original_entry['method_id'],
                'trajectory_id': trajectory_id,
                'seed': seed,
                'rule': 'historical',
                'groups': subgroup_metrics(records, endpoint, score_only_masks['historical']),
            })
        confidence_scores, _, _, _ = confidence_route(args.root, spec, expert, 'test')
        confidence_entry = confidence_entries[(expert, endpoint)]
        confidence_dir = output / 'baselines' / f'confidence_{expert}_{endpoint}'
        confidence_result, confidence_masks = deployment_record(
            'confidence',
            confidence_entry['method_id'],
            confidence_entry,
            records,
            endpoint,
            confidence_scores,
            confidence_dir,
            batch_budgets=batch_budgets,
        )
        confidence_result['seed_variation'] = False
        all_results.append(confidence_result)
        if task == 'grefcoco':
            subgroup_results.append({
                'method': 'confidence',
                'method_id': confidence_entry['method_id'],
                'trajectory_id': trajectory_id,
                'seed': seed,
                'rule': 'historical',
                'groups': subgroup_metrics(records, endpoint, confidence_masks['historical']),
            })
        for endpoint_method in ('small', 'large'):
            baseline_entry = endpoint_entries[(endpoint_method, expert, endpoint)]
            baseline_dir = output / 'baselines' / f'{endpoint_method}_{expert}_{endpoint}'
            baseline_mask = np.zeros(len(records), dtype=bool) if endpoint_method == 'small' else np.ones(len(records), dtype=bool)
            baseline_scores = np.zeros(len(records), dtype=np.float64)
            baseline_point = deployment_record(
                endpoint_method,
                baseline_entry['method_id'],
                baseline_entry,
                records,
                endpoint,
                baseline_scores,
                baseline_dir,
                batch_budgets=None,
            )[0]
            if endpoint_method == 'large':
                # The endpoint baseline uses an always mask; deployment_record already follows its locked rule.
                pass
            baseline_point['seed_variation'] = False
            all_results.append(baseline_point)
        both_gain = score_by_condition['both']
        train_stats = np.load(trajectory_dir / 'fusion_stats.npz')
        val_conf, _, _, _ = confidence_route(args.root, spec, expert, 'val')
        test_fusion_scores = {}
        for alpha in spec['fusion_alphas']:
            if alpha == 0.0:
                scores = confidence_scores.copy()
                reuse_error = 0.0
            elif alpha == 1.0:
                scores = both_gain.copy()
                reuse_error = float(np.max(np.abs(scores - both_gain)))
            else:
                scores = alpha * ((both_gain - float(train_stats['gain_mean'])) / float(train_stats['gain_std'])) + (1.0 - alpha) * ((confidence_scores - float(train_stats['confidence_mean'])) / float(train_stats['confidence_std']))
                reuse_error = None
            fusion_entry = fusion_lookup[(alpha, trajectory_id)]
            alpha_label = f'alpha_{alpha:g}'
            fusion_dir = output / 'derived' / 'fusion' / f'{alpha_label}_{trajectory_id}'
            fusion_result, fusion_masks = deployment_record(
                'fusion',
                fusion_entry['method_id'],
                fusion_entry,
                records,
                endpoint,
                scores,
                fusion_dir,
                batch_budgets=batch_budgets,
            )
            fusion_result.update({
                'alpha': alpha,
                'selected_alpha_for_historical': alpha_lock['historical']['selected_alpha'],
                'selected_alpha_for_endpoint': alpha_lock['endpoint']['selected_alpha'],
                'endpoint_reuse_max_abs_error': reuse_error,
            })
            all_results.append(fusion_result)
            test_fusion_scores[alpha] = scores
        if task == 'grefcoco':
            for alpha in spec['fusion_alphas']:
                entry = fusion_lookup[(alpha, trajectory_id)]
                selection = entry['selection']
                mask = threshold_mask(test_fusion_scores[alpha], selection['selected']['historical'])
                subgroup_results.append({
                    'method': 'fusion',
                    'method_id': entry['method_id'],
                    'alpha': alpha,
                    'trajectory_id': trajectory_id,
                    'seed': seed,
                    'rule': 'historical',
                    'groups': subgroup_metrics(records, endpoint, mask),
                })
        del data, records, raw_by_condition, score_by_condition, confidence_scores, both_gain
        gc.collect()
        print('TEST_TRAJECTORY_COMPLETE', trajectory_id, flush=True)
    # Remove duplicate seed-invariant endpoint/confidence results while keeping one explicit row per pair.
    unique_results = {}
    for result in all_results:
        key = (result['method'], result['task'], result['expert'], result['endpoint'], result.get('trajectory_id') if result['method'] not in ('confidence', 'small', 'large') else None)
        unique_results[key] = result
    all_results = list(unique_results.values())
    if sum(1 for result in all_results if result['method'] in {condition['name'] for condition in spec['conditions']}) != 192:
        raise AssertionError(('test logical result count', sum(1 for result in all_results if result['method'] in {condition['name'] for condition in spec['conditions']})))
    write_json(output / 'grefcoco_subgroups.json', subgroup_results)
    write_json(output / 'test_results.json', {
        'status': 'complete',
        'protocol': {
            'thresholds_selected_on': 'validation',
            'historical_candidates': '1%..99%',
            'endpoint_candidates': '0%..100%',
            'test_selection': 'none',
            'batch_sorting_is_diagnostic': True,
        },
        'groups': all_results,
        'fusion_alpha_selection': alpha_lock,
        'logical_training_conditions': 192,
        'test_loaded': True,
        'seconds': time.perf_counter() - started,
    })
    write_json(output / 'test_evaluation_complete.json', {
        'status': 'complete',
        'finished_at': time.time(),
        'logical_training_conditions': 192,
        'test_loaded_after_validation_lock': True,
        'test_rows_evaluated': sum(result['test_rows'] for result in all_results if result['method'] in {condition['name'] for condition in spec['conditions']}),
    })
    print('ALL_TESTS_COMPLETE', len(all_results), flush=True)


if __name__ == '__main__':
    main()
