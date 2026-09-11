"""Re-evaluate fixed formal checkpoints on a 1% validation-threshold grid.

No training and no test-dependent model/budget selection. Original outputs are
read-only. Best budget is chosen among 1..99 on validation fixed-threshold score,
with lower actual usage breaking exact metric ties.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch


EVENTS = ['rule_1_ppe_violation', 'rule_2_fall_protection_violation',
          'rule_3_unprotected_edge_violation', 'rule_4_excavator_proximity_violation']


def label_arrays(records, endpoint):
    small = [r['small_label'] for r in records]
    large = [r['large_labels'][endpoint] for r in records]
    if 'events' not in small[0]:
        return {'small': np.array([r['correct'] for r in small], dtype=float),
                'large': np.array([r['correct'] for r in large], dtype=float)}
    result = {}
    for key, labels in [('small', small), ('large', large)]:
        target = np.array([[r['events'][e]['target'] for e in EVENTS] for r in labels], dtype=bool)
        pred = np.array([[r['events'][e]['prediction'] is True for e in EVENTS] for r in labels])
        result[key] = np.stack([target & pred, ~target & pred, target & ~pred], axis=-1).astype(float)
    return result


def score_metric(arrays, mask):
    s, l = arrays['small'], arrays['large']
    if s.ndim == 1:
        return float(np.where(mask, l, s).mean())
    counts = np.where(mask[:, None, None], l, s).sum(0)
    tp, fp, fn = counts.T
    return float(np.divide(2 * tp, 2 * tp + fp + fn,
                           out=np.zeros_like(tp), where=2 * tp + fp + fn > 0).mean())


def decision(scores, point):
    op = point['operator']
    if op == 'never':
        return np.zeros(len(scores), dtype=bool)
    if op == 'always':
        return np.ones(len(scores), dtype=bool)
    return scores >= point['threshold'] if op == 'ge' else scores > point['threshold']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / 'code'))
    from routing import train_router_formal as formal
    torch.set_num_threads(4)
    device = torch.device(args.device)
    work = args.root / 'outputs/router_full_20260910'
    selected = json.loads(args.selected.read_text(encoding='utf-8'))['groups']
    args.output.mkdir(parents=True, exist_ok=True)
    formal.BUDGETS = [i / 100 for i in range(101)]
    output = {
        'selection': 'Existing per-pair model chosen by validation area 0-25%; model unchanged. Budget selected from validation fixed-threshold metrics at nominal 1..99%, exact ties prefer lower validation actual usage then lower nominal budget.',
        'curve_protocol': '101 observed points, nominal budget calibrated on validation, deployed unchanged to test. Actual test usage can differ. No interpolation used to create samples.',
        'smoothing': 'Display-only 5-point moving average, edge padded; raw points used for all reporting and selection.',
        'restore_device': str(device),
        'groups': [],
    }
    for item in selected:
        old = item['best_by_validation']
        endpoint = old['endpoint']
        dest = work / 'formal_runs' / old['id']
        cfg = json.loads((dest / 'config.json').read_text())
        records, data = formal.load_data(cfg, 'val', device)
        for key in ['hidden', 'confidence']:
            if key not in data:
                continue
            norm = np.load(dest / f'{key}_normalization.npz')
            data[key] = (data[key] - torch.from_numpy(norm['mean']).to(device)) / torch.from_numpy(norm['std']).to(device)
        model = formal.FeatureRouter(cfg, data['hidden'].shape[1] if 'hidden' in data else 1,
                                     data['confidence'].shape[1], data['output'].shape[1]).to(device)
        checkpoint = torch.load(dest / f'{endpoint}_best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        with torch.inference_mode():
            raw = torch.cat([model(data, torch.arange(i, min(i + 8192, len(records)), device=device))
                             for i in range(0, len(records), 8192)]).cpu().numpy()
        if cfg['target'] == 'four_state':
            val_scores = raw[:, 1] - cfg.get('harm_cost', 1.0) * raw[:, 2]
        elif cfg['target'] == 'error':
            val_scores = 1 / (1 + np.exp(-raw[:, 0]))
        else:
            val_scores = raw[:, 0]
        thresholds = formal.validation_thresholds(records, endpoint, val_scores)
        val_arrays = label_arrays(records, endpoint)
        val_curve = []
        for point in thresholds:
            mask = decision(val_scores, point)
            val_curve.append(dict(point, actual_fraction=float(mask.mean()), metric=score_metric(val_arrays, mask)))
        # Compare restored CPU scores at original ranked budget points to saved validation metrics.
        saved_val = json.loads((dest / f'{endpoint}_validation_metrics.json').read_text())['curve']
        order = np.argsort(-val_scores, kind='stable')
        errors = []
        for old_point in saved_val:
            mask = np.zeros(len(records), dtype=bool)
            mask[order[:int(np.floor(len(records) * old_point['budget']))]] = True
            errors.append(abs(score_metric(val_arrays, mask) - old_point['metric']))
        max_error = max(errors)
        if max_error > 0.001:
            raise RuntimeError(('validation restoration mismatch', old['id'], max_error))
        best_budget_index = max(range(1, 100), key=lambda i: (val_curve[i]['metric'], -val_curve[i]['actual_fraction'], -i))
        # Selection has finished before test metrics are read.
        test_records = formal.read_rows(work / 'data' / cfg['expert'] / 'test/records.jsonl')
        test_scores = np.load(dest / f'{endpoint}_test_scores.npy')
        if len(test_records) != len(test_scores):
            raise ValueError('test score/record length mismatch')
        test_arrays = label_arrays(test_records, endpoint)
        test_curve = []
        for point in thresholds:
            mask = decision(test_scores, point)
            test_curve.append(dict(point, actual_fraction=float(mask.mean()), metric=score_metric(test_arrays, mask)))
        # Reproduce original fixed-threshold test results with the original threshold file.
        saved_test = json.loads((dest / f'{endpoint}_test_metrics.json').read_text())['fixed_threshold_test']['curve']
        max_test_error = max(abs(score_metric(test_arrays, decision(test_scores, p)) - p['metric']) for p in saved_test)
        if max_test_error > 1e-10:
            raise RuntimeError(('test metric mismatch', old['id'], max_test_error))
        run_out = args.output / (old['id'] + '_' + endpoint)
        run_out.mkdir(exist_ok=True)
        np.savez(run_out / 'paired_test_arrays.npz', scores=test_scores, **test_arrays)
        np.save(run_out / 'validation_scores.npy', val_scores)
        (run_out / 'test_sample_ids.json').write_text(json.dumps([r['sample_id'] for r in test_records]))
        group = {
            'key': item['key'], 'config': cfg, 'id': old['id'], 'endpoint': endpoint,
            'selection_area_0_25': old['validation_selection'],
            'validation_rows': len(records), 'test_rows': len(test_records),
            'best_epoch': checkpoint['epoch'],
            'best_budget_index': best_budget_index,
            'best_validation_point': val_curve[best_budget_index],
            'best_test_point': test_curve[best_budget_index],
            'validation_curve': val_curve, 'test_curve': test_curve,
            'validation_restore_max_metric_error': max_error,
            'test_reproduction_max_metric_error': max_test_error,
            'router_head_parameters': sum(p.numel() for p in model.parameters()),
        }
        output['groups'].append(group)
        (args.output / 'dense_results.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
        print(json.dumps({k: group[k] for k in ['key', 'id', 'best_validation_point', 'best_test_point', 'validation_restore_max_metric_error']}, ensure_ascii=False), flush=True)
        del records, test_records, data, model, checkpoint, test_arrays, val_arrays
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('complete', len(output['groups']), flush=True)


if __name__ == '__main__':
    main()
