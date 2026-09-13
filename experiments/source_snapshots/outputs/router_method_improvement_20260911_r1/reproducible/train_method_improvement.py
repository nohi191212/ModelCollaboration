"""Train one full train/validation trajectory and derive the four locked methods."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from method_improvement_common import (
    FeatureRouter,
    copy_state_dict,
    forward_logits,
    load_split,
    make_config,
    normalize_splits,
    ranked_curve,
    select_ranked_candidate,
    read_json,
    score_from_logits,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expert', required=True)
    parser.add_argument('--endpoint', required=True, choices=['MiniCPM', 'Qwen3.8'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=None)
    return parser.parse_args()


def selection_record(records, endpoint, scores, condition, epoch):
    if condition['selection_mode'] == 'a25':
        curve = ranked_curve(records, endpoint, scores, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        return {
            'primary_metric': curve['selection_score'],
            'selection_area_0_25': curve['selection_score'],
            'selection_candidate': None,
            'selection_curve': curve['curve'],
            'epoch': epoch,
        }
    if condition['selection_mode'] == 'max_task_performance':
        candidate, candidates = select_ranked_candidate(
            records,
            endpoint,
            scores,
            [i / 100 for i in range(1, 100)],
        )
        return {
            'primary_metric': candidate['metric'],
            'selection_area_0_25': None,
            'selection_candidate': candidate,
            'selection_curve': candidates,
            'epoch': epoch,
        }
    raise ValueError(('unknown selection mode', condition['selection_mode']))


def main():
    args = parse_args()
    spec = read_json(args.spec)
    if args.expert not in spec['experts']:
        raise ValueError(('unknown expert', args.expert))
    trajectory_id = f'{args.expert}_{args.endpoint}_seed{args.seed}'
    dest = args.output / 'trajectories' / trajectory_id
    completion_path = dest / 'trajectory_completion.json'
    if completion_path.exists():
        existing = read_json(completion_path)
        if existing.get('status') == 'complete':
            print('REUSED', trajectory_id, flush=True)
            return
        raise FileExistsError(('incomplete trajectory exists', str(dest)))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'epoch_validation_logits').mkdir(exist_ok=True)
    cfg = make_config(spec, args.expert, args.endpoint, args.seed, trajectory_id)
    cfg['runtime_device'] = args.device
    cfg['runtime_threads'] = args.threads if args.threads is not None else cfg['cpu_threads']
    write_json(dest / 'config.json', cfg)
    write_json(dest / 'protocol_snapshot.json', {
        'spec_name': spec['name'],
        'data_work': spec['data_work'],
        'fixed': spec['fixed'],
        'conditions': spec['conditions'],
        'seed': args.seed,
        'test_loaded_during_training': False,
    })
    torch.set_num_threads(cfg['runtime_threads'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    records, data = {}, {}
    for split in ('train', 'val'):
        records[split], data[split] = load_split(args.root, spec, cfg, split, device)
    expected = spec['experts'][args.expert]['expected_rows']
    for split in ('train', 'val'):
        if len(records[split]) != expected[split]:
            raise ValueError(('unexpected full-split row count', args.expert, split, len(records[split]), expected[split]))
    normalize_splits(data, dest)
    write_json(dest / 'training_rows.json', {
        'fraction': 1.0,
        'train_rows': len(records['train']),
        'validation_rows': len(records['val']),
        'test_loaded': False,
        'training_sample_ids': [row['sample_id'] for row in records['train']],
        'validation_sample_ids': [row['sample_id'] for row in records['val']],
    })
    model = FeatureRouter(
        cfg,
        data['train']['hidden'].shape[1],
        data['train']['confidence'].shape[1],
        data['train']['output'].shape[1],
    ).to(device)
    small_labels, large_labels = None, None
    from method_improvement_common import correctness_arrays
    small_labels, large_labels = correctness_arrays(records['train'], args.endpoint)
    if small_labels.ndim == 1:
        small_labels = small_labels[:, None]
        large_labels = large_labels[:, None]
    classes = np.stack([
        (small_labels * large_labels).mean(1),
        ((~small_labels) * large_labels).mean(1),
        (small_labels * (~large_labels)).mean(1),
        ((~small_labels) * (~large_labels)).mean(1),
    ], axis=1)
    y = torch.from_numpy(classes.astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    all_train = torch.arange(len(records['train']), device=device)
    conditions = spec['conditions']
    state = {
        condition['name']: {
            'best_primary_metric': float('-inf'),
            'best_epoch': None,
            'stale': 0,
            'early_stop_epoch': None,
            'best_state_dict': None,
            'best_validation_logits': None,
            'best_validation_scores': None,
            'best_selection': None,
            'history': [],
        }
        for condition in conditions
    }
    history_path = dest / 'history.jsonl'
    started = time.perf_counter()
    epoch = 0
    while epoch < cfg['max_epochs'] and any(item['early_stop_epoch'] is None for item in state.values()):
        epoch += 1
        model.train()
        order = all_train[torch.randperm(len(all_train), device=device)]
        loss_sum = 0.0
        for start in range(0, len(order), cfg['batch_size']):
            indices = order[start:start + cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            logits = model(data['train'], indices)
            loss = -(y[indices] * F.log_softmax(logits, dim=1)).sum(1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(('nonfinite training loss', trajectory_id, epoch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        validation_logits = forward_logits(model, data['val'], len(records['val']), device)
        np.save(dest / 'epoch_validation_logits' / f'epoch_{epoch:03d}.npy', validation_logits)
        epoch_record = {
            'epoch': epoch,
            'training_loss': loss_sum / len(order),
            'elapsed_seconds': time.perf_counter() - started,
            'conditions': {},
        }
        for condition in conditions:
            name = condition['name']
            scores = score_from_logits(validation_logits, condition['score_mode'])
            current = selection_record(records['val'], args.endpoint, scores, condition, epoch)
            item = state[name]
            improved = current['primary_metric'] > item['best_primary_metric']
            if improved:
                item['best_primary_metric'] = current['primary_metric']
                item['best_epoch'] = epoch
                item['stale'] = 0
                item['best_state_dict'] = copy_state_dict(model)
                item['best_validation_logits'] = validation_logits.copy()
                item['best_validation_scores'] = scores.copy()
                item['best_selection'] = current
            else:
                item['stale'] += 1
            if item['early_stop_epoch'] is None and item['stale'] >= cfg['patience']:
                item['early_stop_epoch'] = epoch
            item['history'].append({
                'epoch': epoch,
                'primary_metric': current['primary_metric'],
                'improved': improved,
                'stale': item['stale'],
                'selection_candidate': current['selection_candidate'],
                'selection_area_0_25': current['selection_area_0_25'],
            })
            epoch_record['conditions'][name] = {
                'primary_metric': current['primary_metric'],
                'improved': improved,
                'stale': item['stale'],
                'selection_candidate': current['selection_candidate'],
            }
        with history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + '\n')
        print(trajectory_id, json.dumps(epoch_record, ensure_ascii=False), flush=True)
    if any(item['best_state_dict'] is None for item in state.values()):
        raise RuntimeError(('no checkpoint selected', trajectory_id))
    condition_ids = []
    for condition in conditions:
        name = condition['name']
        item = state[name]
        condition_id = f'{name}_{trajectory_id}'
        condition_dir = args.output / 'conditions' / condition_id
        condition_dir.mkdir(parents=True, exist_ok=True)
        condition_cfg = dict(cfg)
        condition_cfg.update({
            'condition': name,
            'selection_mode': condition['selection_mode'],
            'score_mode': condition['score_mode'],
            'condition_id': condition_id,
        })
        write_json(condition_dir / 'config.json', condition_cfg)
        torch.save({
            'state_dict': item['best_state_dict'],
            'config': condition_cfg,
            'epoch': item['best_epoch'],
            'trajectory_id': trajectory_id,
        }, condition_dir / 'best.pt')
        np.save(condition_dir / 'validation_logits.npy', item['best_validation_logits'])
        np.save(condition_dir / 'validation_scores.npy', item['best_validation_scores'])
        write_json(condition_dir / 'validation_sample_ids.json', records['val'] and [row['sample_id'] for row in records['val']])
        with (condition_dir / 'history.jsonl').open('w', encoding='utf-8') as handle:
            for row in item['history']:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        write_json(condition_dir / 'selection_record.json', {
            'condition': name,
            'condition_id': condition_id,
            'trajectory_id': trajectory_id,
            'best_epoch': item['best_epoch'],
            'early_stop_epoch': item['early_stop_epoch'],
            'epochs_run_for_shared_trajectory': epoch,
            'best_primary_metric': item['best_primary_metric'],
            'best_selection': item['best_selection'],
            'test_loaded_during_training': False,
            'trajectory_reuse_note': 'The one training trajectory is shared only because all four conditions have identical loss, data, and seed; each condition keeps an independent checkpoint criterion and early-stop record.',
        })
        condition_ids.append(condition_id)
    write_json(completion_path, {
        'status': 'complete',
        'trajectory_id': trajectory_id,
        'config': cfg,
        'expert': args.expert,
        'endpoint': args.endpoint,
        'seed': args.seed,
        'epochs_run': epoch,
        'condition_ids': condition_ids,
        'conditions': {
            name: {
                'best_epoch': item['best_epoch'],
                'early_stop_epoch': item['early_stop_epoch'],
                'best_primary_metric': item['best_primary_metric'],
            }
            for name, item in state.items()
        },
        'train_rows': len(records['train']),
        'validation_rows': len(records['val']),
        'test_loaded_during_training': False,
        'seconds': time.perf_counter() - started,
    })
    print('TRAJECTORY_COMPLETE', trajectory_id, flush=True)


if __name__ == '__main__':
    main()
