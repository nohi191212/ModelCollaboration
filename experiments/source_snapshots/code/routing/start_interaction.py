"""Start interactions and raw-image finetuning after reviewed single-factor fits."""
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
    args = parser.parse_args()
    work = ROOT / 'outputs/router_exploration_25pct_20260910'
    exits = json.loads((work / 'followup_pipeline/worker_exits.json').read_text())
    assert set(exits) == {'0', '1', '2', '3'} and all(code == 0 for code in exits.values())
    analysis = json.loads((work / 'followup_queue_analysis.json').read_text())
    assert analysis['all_queue_fits_complete']
    for warning in analysis['warnings']:
        assert json.loads((work / 'runs' / warning['id'] / 'anomaly_review.json').read_text())['decision'] == 'no_bug'
    queue = json.loads((work / 'interaction_queue.json').read_text())
    design = json.loads((work / 'interaction_design.json').read_text())
    assert queue and len(queue) == design['new_fits']
    assert all(cfg['seed'] == 42 for cfg in queue)
    assert len({cfg['id'] for cfg in queue}) == len(queue)
    covered = set()
    smokes = []
    for cfg in queue:
        assert cfg['fraction'] == .25
        coverage = {('inputs', cfg['expert'], tuple(cfg['branches'])),
                    ('target_strategy', cfg['target'], cfg['strategy']),
                    ('size_dimension', cfg['student'], cfg['representation_dim'])}
        if not cfg['freeze_encoder']:
            assert set(cfg['branches']) & {'image', 'text'}
            coverage.add(('finetune', cfg['task'], cfg['student'], tuple(cfg['branches']), cfg['target'], cfg['encoder_lr'], cfg['micro_batch_size']))
        if coverage - covered:
            covered.update(coverage)
            smokes.append(cfg)
    plan = {'fits': len(queue), 'smoke_ids': [cfg['id'] for cfg in smokes],
            'finetune_fits': sum(not cfg['freeze_encoder'] for cfg in queue),
            'all_exploration_complete': False}
    (work / 'interaction_launch_plan.json').write_text(json.dumps(plan, indent=2))
    if args.prepare_only:
        print(json.dumps(plan))
        sys.exit(0)
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(memory) == 4 and all(int(x) < 500 for x in memory), memory
    stage = work / 'interaction_pipeline'
    stage.mkdir(exist_ok=False)
    (stage / 'started.json').write_text(json.dumps({'pid': os.getpid(), 'unix_time': time.time()}))
    for start in range(0, len(smokes), 4):
        processes = []
        for gpu, cfg in enumerate(smokes[start:start + 4]):
            configfile = work / 'configs' / (cfg['id'] + '.json')
            configfile.write_text(json.dumps(cfg, indent=2))
            log = (stage / (cfg['id'] + '_smoke.log')).open('x')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2')
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'code/routing/train_router.py'), '--config', str(configfile), '--smoke'], env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((cfg['id'], process, log))
        (stage / 'smoke_processes.json').write_text(json.dumps({name: process.pid for name, process, log in processes}))
        exits = {}
        for name, process, log in processes:
            exits[name] = process.wait()
            log.close()
        (stage / f'smoke_batch_{start:04d}.json').write_text(json.dumps(exits))
        assert all(code == 0 for code in exits.values()), exits
        for name, process, log in processes:
            dest = work / 'smoke_runs' / name
            result = json.loads((dest / 'completion.json').read_text())
            assert result['status'] == 'complete' and result['smoke']
            if result['anomalies']:
                assert json.loads((dest / 'anomaly_review.json').read_text())['decision'] == 'no_bug'
    # Cached vector fits share each GPU; raw-image fits start only on an empty GPU.
    queue = sorted(queue, key=lambda cfg: not cfg['freeze_encoder'])
    active = {}; cursor = 0; completed = 0; problems = []
    with (stage / 'events.jsonl').open('x') as events:
        while cursor < len(queue) or active:
            for name, job in list(active.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                job['log'].close()
                resultfile = work / 'runs' / name / 'completion.json'
                problem = None
                if code != 0 or not resultfile.exists():
                    problem = {'id': name, 'exit_code': code}
                else:
                    result = json.loads(resultfile.read_text())
                    assert result['status'] == 'complete' and not result['smoke']
                    if result['anomalies']:
                        review = resultfile.parent / 'anomaly_review.json'
                        if not review.exists() or json.loads(review.read_text())['decision'] != 'no_bug':
                            problem = {'id': name, 'anomalies': result['anomalies']}
                if problem:
                    problems.append(problem)
                else:
                    completed += 1
                events.write(json.dumps({'event':'finished','id':name,'gpu':job['gpu'],'exit_code':code,'problem':problem,'seconds':time.time()-job['started']})+'\n'); events.flush()
                del active[name]
            memory = [int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
            if not problems:
                for gpu in range(4):
                    if cursor == len(queue): break
                    cfg = queue[cursor]
                    occupied = [job for job in active.values() if job['gpu'] == gpu]
                    if memory[gpu] >= 65536: continue
                    if occupied and (not cfg['freeze_encoder'] or not all(job['frozen'] for job in occupied)): continue
                    if len(occupied) >= 4: continue
                    configfile = work / 'configs' / (cfg['id'] + '.json')
                    assert json.loads(configfile.read_text()) == cfg if configfile.exists() else True
                    configfile.write_text(json.dumps(cfg,indent=2))
                    assert not (work / 'runs' / cfg['id']).exists(), cfg['id']
                    log = (stage / (cfg['id'] + '_formal.log')).open('x')
                    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    process = subprocess.Popen([sys.executable,'-u',str(ROOT/'code/routing/train_router.py'),'--config',str(configfile),'--cpu-threads','1'],env=env,stdout=log,stderr=subprocess.STDOUT)
                    active[cfg['id']] = {'process':process,'log':log,'gpu':gpu,'frozen':cfg['freeze_encoder'],'started':time.time()}
                    cursor += 1
                    events.write(json.dumps({'event':'started','id':cfg['id'],'gpu':gpu,'pid':process.pid,'unix_time':time.time()})+'\n'); events.flush()
            snapshot = {'completed':completed,'planned':len(queue),'pending':len(queue)-cursor,'problems':problems,'active':[{'id':name,'pid':job['process'].pid,'gpu':job['gpu'],'frozen':job['frozen']} for name,job in active.items()]}
            (stage/'state.tmp').write_text(json.dumps(snapshot,indent=2)); (stage/'state.tmp').replace(stage/'state.json')
            if problems and not active: raise RuntimeError(problems)
            time.sleep(2)
    (stage / 'completion.json').write_text(json.dumps({'status': 'interaction_training_complete_requires_replay', 'fits': len(queue), 'all_exploration_complete': False}))
