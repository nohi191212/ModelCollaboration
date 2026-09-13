"""Drain legacy workers without killing fits, then multiplex remaining fits on GPUs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-gpu', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.per_gpu <= 32
    work = ROOT / 'outputs/router_exploration_25pct_20260910'
    queue = json.loads((work / 'followup_queue.json').read_text())
    workers = json.loads((work / 'followup_pipeline/worker_processes.json').read_text())
    assert set(workers) == {'0', '1', '2', '3'}
    for pid in workers.values():
        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
        assert 'run_queue.py' in command and 'followup_queue.json' in command, (pid, command)
    completed = set()
    for cfg in queue:
        dest = work / 'runs' / cfg['id']
        if not (dest / 'completion.json').exists():
            continue
        done = json.loads((dest / 'completion.json').read_text())
        assert done['status'] == 'complete' and not done['smoke']
        if done['anomalies']:
            assert json.loads((dest / 'anomaly_review.json').read_text())['decision'] == 'no_bug'
        completed.add(cfg['id'])
    if args.prepare_only:
        print(json.dumps({'planned': len(queue), 'complete': len(completed), 'per_gpu': args.per_gpu,
                          'legacy_workers_verified': workers, 'gpu_launched': False}))
        sys.exit(0)
    stage = work / 'parallel_followup'
    stage.mkdir(exist_ok=False)
    (stage / 'logs').mkdir()
    started = time.time()
    (stage / 'started.json').write_text(json.dumps({'pid': os.getpid(), 'unix_time': started, 'legacy_workers': workers}))
    (stage / 'control.json').write_text(json.dumps({'per_gpu': args.per_gpu}))
    # Only stop scheduling parents; their currently running training children continue.
    for pid in workers.values():
        os.kill(pid, signal.SIGSTOP)
    time.sleep(.2)
    children = []
    for pid in workers.values():
        state = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        assert state == 'T', (pid, state)
        child_listing = subprocess.run(['ps', '--ppid', str(pid), '-o', 'pid='], text=True, capture_output=True)
        assert child_listing.returncode in [0, 1], child_listing.stderr
        ids = child_listing.stdout.split()
        for child in ids:
            proc = Path('/proc') / child
            arguments = proc.joinpath('cmdline').read_bytes().split(b'\x00')
            if not arguments[0]:
                continue  # Already exited and waiting for the stopped parent to reap it.
            decoded = [x.decode() for x in arguments if x]
            assert any('train_router.py' in x for x in decoded), decoded
            config = Path(decoded[decoded.index('--config') + 1])
            children.append({'pid': int(child), 'id': json.loads(config.read_text())['id']})
    (stage / 'draining.json').write_text(json.dumps({'legacy_workers': workers, 'training_children': children}, indent=2))
    print(json.dumps({'status': 'draining_existing_fits', 'children': children}), flush=True)
    while True:
        live = []
        for child in children:
            proc = Path('/proc') / str(child['pid']) / 'stat'
            if proc.exists() and proc.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                live.append(child['pid'])
        if not live:
            break
        time.sleep(2)
    for child in children:
        assert (work / 'runs' / child['id'] / 'completion.json').exists(), child
    completed = set()
    for cfg in queue:
        dest = work / 'runs' / cfg['id']
        if (dest / 'completion.json').exists():
            done = json.loads((dest / 'completion.json').read_text())
            assert done['status'] == 'complete' and not done['smoke']
            if done['anomalies']:
                assert json.loads((dest / 'anomaly_review.json').read_text())['decision'] == 'no_bug'
            completed.add(cfg['id'])
        else:
            assert not dest.exists(), ('Incomplete run needs diagnosis', str(dest))
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(memory) == 4 and all(int(x) < 500 for x in memory), memory
    pending = [cfg for cfg in queue if cfg['id'] not in completed]
    active = {}
    problems = []
    cursor = 0
    starting_complete = len(completed)
    print(json.dumps({'status': 'parallel_training_started', 'completed_before': starting_complete, 'remaining': len(pending)}), flush=True)
    with (stage / 'events.jsonl').open('x') as events:
        while cursor < len(pending) or active:
            for cid, job in list(active.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                job['log'].close()
                elapsed = time.time() - job['started']
                resultfile = work / 'runs' / cid / 'completion.json'
                problem = None
                if code != 0 or not resultfile.exists():
                    problem = {'id': cid, 'exit_code': code, 'reason': 'training_failed'}
                else:
                    done = json.loads(resultfile.read_text())
                    assert done['status'] == 'complete' and not done['smoke']
                    if done['anomalies']:
                        review = resultfile.parent / 'anomaly_review.json'
                        if not review.exists() or json.loads(review.read_text())['decision'] != 'no_bug':
                            problem = {'id': cid, 'reason': 'anomaly_needs_diagnosis', 'anomalies': done['anomalies']}
                    if problem is None:
                        completed.add(cid)
                if problem:
                    problems.append(problem)
                events.write(json.dumps({'event': 'finished', 'id': cid, 'gpu': job['gpu'], 'unix_time': time.time(), 'seconds': elapsed, 'exit_code': code, 'problem': problem}) + '\n')
                events.flush()
                del active[cid]
            limit = json.loads((stage / 'control.json').read_text())['per_gpu']
            assert isinstance(limit, int) and 1 <= limit <= 32
            memory = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
            counts = Counter(job['gpu'] for job in active.values())
            if not problems:
                for gpu in range(4):
                    if cursor >= len(pending):
                        break
                    if counts[gpu] >= limit or memory[gpu] >= 65536:
                        continue
                    cfg = pending[cursor]
                    configfile = work / 'configs' / (cfg['id'] + '.json')
                    if configfile.exists():
                        assert json.loads(configfile.read_text()) == cfg
                    else:
                        configfile.write_text(json.dumps(cfg, indent=2))
                    log = (stage / 'logs' / (cfg['id'] + '.log')).open('x')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                    process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'code/routing/train_router.py'), '--config', str(configfile), '--cpu-threads', '1'], env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[cfg['id']] = {'process': process, 'log': log, 'gpu': gpu, 'started': time.time()}
                    cursor += 1
                    events.write(json.dumps({'event': 'started', 'id': cfg['id'], 'gpu': gpu, 'pid': process.pid, 'unix_time': time.time()}) + '\n')
                    events.flush()
            snapshot = {'status': 'draining_after_problem' if problems else 'running', 'pid': os.getpid(), 'unix_time': time.time(),
                        'completed': len(completed), 'starting_completed': starting_complete, 'planned': len(queue),
                        'pending': len(pending) - cursor, 'per_gpu': limit, 'memory_mib': memory, 'problems': problems,
                        'active': [{'id': cid, 'gpu': job['gpu'], 'pid': job['process'].pid, 'started': job['started']} for cid, job in active.items()]}
            (stage / 'state.tmp').write_text(json.dumps(snapshot, indent=2))
            (stage / 'state.tmp').replace(stage / 'state.json')
            if problems and not active:
                raise RuntimeError(problems)
            time.sleep(2)
    assert len(completed) == len(queue)
    (stage / 'completion.json').write_text(json.dumps({'status': 'parallel_queue_complete_legacy_supervisor_finishing', 'fits': len(completed), 'all_exploration_complete': False}))
    # Legacy workers skip now-complete fits and let their original master exit normally.
    for pid in workers.values():
        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
        assert 'run_queue.py' in command and 'followup_queue.json' in command
        os.kill(pid, signal.SIGCONT)
    print(json.dumps({'status': 'parallel_queue_complete', 'fits': len(completed)}), flush=True)
