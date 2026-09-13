from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
import json,os,subprocess,sys,time
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
OUT=ROOT/'outputs/three_state_20260913'
HERE=Path(__file__).resolve().parent
REF=ROOT/'outputs/objective_multiseed_20260912/runs'
groups=json.loads((HERE/'source/main_table_reference.json').read_text())['groups']
jobs=[];reuse=[]
for seed in range(2026,2031):
    for group in groups:
        task,expert,endpoint=group['key']
        orig=REF/f'repeat_{expert}_{endpoint}_four_difference_s{seed}'
        gain=REF/f'repeat_{expert}_{endpoint}_gain_s{seed}'
        cfg=json.loads((orig/'config.json').read_text())
        gcfg=json.loads((gain/'config.json').read_text())
        assert cfg['strategy']=='natural' and cfg['harm_cost']==1
        allowed={'id','target','ranking','method'}
        diffs={k:[cfg.get(k),gcfg.get(k)] for k in set(cfg)|set(gcfg) if cfg.get(k)!=gcfg.get(k)}
        assert set(diffs)<=allowed,(gain,diffs)
        for path in [orig,gain]:
            assert all((path/name).exists() for name in ['completion.json','paired_logits.npz','test_evaluation.json'])
        reuse.append({'task':task,'expert':expert,'endpoint':endpoint,'seed':seed,'learned':str(orig),'gain':str(gain),'config_differences':diffs})
        cfg.update(id=f'{expert}_{endpoint}_three_s{seed}',ranking='three_difference',method='three_state',ltd_method='three_state')
        jobs.append(cfg)
assert len(jobs)==80
OUT.mkdir(exist_ok=True)
(OUT/'reused_runs.json').write_text(json.dumps(reuse,indent=2))
(OUT/'job_plan.json').write_text(json.dumps(jobs,indent=2))
(OUT/'method_lock.json').write_text(json.dumps({'method':'Three classes: rescue, harm, unchanged. unchanged = both correct + both wrong. Soft event proportions for construction; fixed original specialist predictions. Same original low-budget validation checkpoint selection.','seeds':list(range(2026,2031)),'new_fits':80,'reused_fits':160,'time':time.time()},indent=2))
queue=Queue()
for cfg in jobs:queue.put(cfg)
def worker(gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1')
    while True:
        try:cfg=queue.get_nowait()
        except Empty:return
        dest=OUT/'runs'/cfg['id'];dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'completion.json').exists():
            assert not (dest/'history.jsonl').exists(),('preserve interrupted attempt before restarting',dest)
            (dest/'config.json').write_text(json.dumps(cfg,indent=2))
            with (dest/'train.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train_three.py'),'--root',str(ROOT),'--config',str(dest/'config.json'),'--output',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (dest/'test_evaluation.json').exists():
            with (dest/'test.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'evaluate_three.py'),'--run',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'completed':cfg['id'],'gpu':gpu,'time':time.time()}),flush=True)
with ThreadPoolExecutor(max_workers=10) as pool:
    fs=[pool.submit(worker,i%2) for i in range(10)]
    for f in fs:f.result()
(OUT/'complete.json').write_text(json.dumps({'fits':80,'reused':160,'time':time.time()},indent=2))
