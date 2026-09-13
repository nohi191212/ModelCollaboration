"""Dedicated source-only transfer launcher; does not run ordinary router training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(4))
    args = parser.parse_args()
    work = ROOT / 'outputs/router_exploration_25pct_20260910'
    queue = json.loads((work / 'transfer_queue.json').read_text())
    assert len(queue) == 16 and len({cfg['id'] for cfg in queue}) == 16
    assert all(cfg['seed'] == 42 for cfg in queue)
    for cfg in queue:
        assert cfg['branches'] == ['image', 'text'] and cfg['target'] == 'error'
        assert json.loads((work / 'transfer_configs' / (cfg['id'] + '.json')).read_text()) == cfg
    if args.prepare_only:
        print(json.dumps({'status': 'plan_checked_not_launched', 'fits': len(queue),
                          'jobs_per_worker': [len(queue[i::4]) for i in range(4)],
                          'formal_dependencies_checked': False}))
        sys.exit(0)
    directory = work / 'transfer_pipeline'
    if args.worker is None:
        sys.path.insert(0, str(ROOT / 'code'))
        from routing.student_artifacts import student_artifacts
        for cfg in queue:
            artifact = student_artifacts(cfg['student'], cfg['student_control'])
            control = json.loads((artifact['directory'] / 'config.json').read_text())['control_config']
            assert control['heldout_task'] == cfg['heldout_task']
            report = json.loads((work / 'features' / artifact['feature_key'] / 'completion.json').read_text())
            assert report['status'] == 'complete' and report['student'] == str(artifact['checkpoint'])
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        assert len(memory.splitlines()) == 4 and all(int(x) < 500 for x in memory.splitlines()), memory
        directory.mkdir(exist_ok=False)
        (directory / 'started.json').write_text(json.dumps({'pid': os.getpid(), 'unix_time': time.time()}))
        processes = []
        handles = []
        for worker in range(4):
            handle = (directory / f'worker{worker}.log').open('x')
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker))
            processes.append(subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(worker)], env=env, stdout=handle, stderr=subprocess.STDOUT))
        (directory / 'worker_processes.json').write_text(json.dumps({str(i): p.pid for i, p in enumerate(processes)}))
        exits = {str(i): p.wait() for i, p in enumerate(processes)}
        for handle in handles:
            handle.close()
        (directory / 'worker_exits.json').write_text(json.dumps(exits))
        assert all(code == 0 for code in exits.values()), exits
        (directory / 'completion.json').write_text(json.dumps({'status': 'transfer_training_complete_requires_result_review', 'fits': len(queue), 'all_exploration_complete': False}))
    else:
        statefile = directory / f'worker{args.worker}.json'
        for cfg in queue[args.worker::4]:
            config = work / 'transfer_configs' / (cfg['id'] + '.json')
            for smoke in [True, False]:
                phase = 'smoke' if smoke else 'formal'
                state = {'worker_pid': os.getpid(), 'active_id': cfg['id'], 'phase': phase, 'status': 'running', 'unix_time': time.time()}
                command = [sys.executable, '-u', str(ROOT / 'code/routing/train_transfer_router.py'), '--config', str(config)]
                if smoke:
                    command.append('--smoke')
                with (directory / (cfg['id'] + '_' + phase + '.log')).open('x') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = process.pid
                    statefile.write_text(json.dumps(state))
                    code = process.wait()
                if code:
                    state.update(status='failed_needs_diagnosis', exit_code=code)
                    statefile.write_text(json.dumps(state))
                    sys.exit(code)
                result = json.loads((work / 'transfer_runs' / phase / cfg['id'] / 'completion.json').read_text())
                assert result['status'] == 'complete' and result['smoke'] == smoke
                assert result['target_used_for_selection'] is False and result['target_training_rows_used'] == 0
                if not smoke:
                    assert result['strict_student_verified'] is True
                replay = subprocess.run([sys.executable, str(ROOT / 'code/routing/check_transfer_result.py'), '--run', str(work / 'transfer_runs' / phase / cfg['id'])], check=False)
                if replay.returncode:
                    state.update(status='result_check_needs_diagnosis', exit_code=replay.returncode)
                    statefile.write_text(json.dumps(state))
                    sys.exit(replay.returncode)
        statefile.write_text(json.dumps({'status': 'complete', 'worker_pid': os.getpid(), 'jobs': len(queue[args.worker::4]), 'unix_time': time.time()}))
