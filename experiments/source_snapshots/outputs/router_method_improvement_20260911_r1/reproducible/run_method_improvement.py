"""Run all 48 shared full-data trajectories; each yields four logical conditions."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from method_improvement_common import read_json, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trainer', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    spec = read_json(args.spec)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'method_improvement_spec.json', spec)
    expected = [
        (expert, endpoint, seed)
        for seed in spec['seeds']
        for expert in spec['experts']
        for endpoint in ('MiniCPM', 'Qwen3.8')
    ]
    logs_dir = args.output / 'logs'
    logs_dir.mkdir(exist_ok=True)
    plan = [
        {
            'trajectory_id': f'{expert}_{endpoint}_seed{seed}',
            'expert': expert,
            'endpoint': endpoint,
            'seed': seed,
            'status': 'pending',
        }
        for expert, endpoint, seed in expected
    ]
    plan_path = args.output / 'training_plan.json'
    if plan_path.exists():
        old_plan = read_json(plan_path)
        if [item['trajectory_id'] for item in old_plan] != [item['trajectory_id'] for item in plan]:
            raise ValueError('existing training plan does not match the fixed experiment spec')
        plan = old_plan
    else:
        write_json(plan_path, plan)
    scheduler_log = args.output / 'scheduler.jsonl'

    def log(event):
        with scheduler_log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(event, time=time.time()), ensure_ascii=False) + '\n')

    pending = []
    for item in plan:
        done = args.output / 'trajectories' / item['trajectory_id'] / 'trajectory_completion.json'
        if done.exists() and read_json(done).get('status') == 'complete':
            item['status'] = 'complete'
        else:
            if (args.output / 'trajectories' / item['trajectory_id']).exists():
                raise FileExistsError(('incomplete trajectory output exists', item['trajectory_id']))
            item['status'] = 'pending'
            pending.append(item)
    write_json(plan_path, plan)
    active = []
    started = time.time()
    try:
        while pending or active:
            for job in active[:]:
                code = job['process'].poll()
                if code is None:
                    continue
                item = job['item']
                done = args.output / 'trajectories' / item['trajectory_id'] / 'trajectory_completion.json'
                valid = code == 0 and done.exists() and read_json(done).get('status') == 'complete'
                item['status'] = 'complete' if valid else 'failed'
                item['returncode'] = code
                item['seconds'] = time.time() - job['started']
                write_json(plan_path, plan)
                log({'event': 'finish', 'trajectory_id': item['trajectory_id'], 'returncode': code, 'valid': valid, 'seconds': item['seconds']})
                active.remove(job)
                if not valid:
                    raise RuntimeError(('trajectory failed', item['trajectory_id'], code, str(job['log_path'])))
            while pending and len(active) < args.workers:
                item = pending.pop(0)
                log_path = logs_dir / f"{item['trajectory_id']}.log"
                env = dict(os.environ)
                env.update({
                    'CUDA_VISIBLE_DEVICES': '',
                    'OMP_NUM_THREADS': str(args.threads),
                    'MKL_NUM_THREADS': str(args.threads),
                    'PYTHONUNBUFFERED': '1',
                })
                handle = log_path.open('a', encoding='utf-8')
                process = subprocess.Popen([
                    sys.executable, str(args.trainer),
                    '--root', str(args.root),
                    '--spec', str(args.spec),
                    '--output', str(args.output),
                    '--expert', item['expert'],
                    '--endpoint', item['endpoint'],
                    '--seed', str(item['seed']),
                    '--device', args.device,
                    '--threads', str(args.threads),
                ], stdout=handle, stderr=subprocess.STDOUT, env=env)
                handle.close()
                item['status'] = 'running'
                item['pid'] = process.pid
                write_json(plan_path, plan)
                job = {'item': item, 'process': process, 'started': time.time(), 'log_path': log_path}
                active.append(job)
                log({'event': 'start', 'trajectory_id': item['trajectory_id'], 'pid': process.pid, 'log': str(log_path)})
            if pending or active:
                time.sleep(2)
    finally:
        for job in active:
            if job['process'].poll() is None:
                job['process'].terminate()
        for job in active:
            try:
                job['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                job['process'].kill()
                job['process'].wait()
    complete = [item for item in plan if item['status'] == 'complete']
    if len(complete) != 48:
        raise AssertionError(('trajectory count', len(complete)))
    condition_count = sum(
        1
        for condition in spec['conditions']
        for item in complete
        if (args.output / 'conditions' / f"{condition['name']}_{item['trajectory_id']}" / 'selection_record.json').exists()
    )
    if condition_count != 192:
        raise AssertionError(('logical training condition count', condition_count))
    write_json(args.output / 'training_complete.json', {
        'status': 'complete',
        'finished_at': time.time(),
        'elapsed_seconds': time.time() - started,
        'trajectories': len(complete),
        'logical_training_conditions': condition_count,
        'test_loaded_during_training': False,
        'device': args.device,
        'workers': args.workers,
        'threads_per_worker': args.threads,
    })
    print('TRAINING_COMPLETE', condition_count, flush=True)


if __name__ == '__main__':
    main()
