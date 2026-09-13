"""Original K-class answer semantics on the two single-label tasks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Event
import json
import os
import subprocess
import sys
import time

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/ltd_kclass_native_20260913'
groups=json.loads((HERE/'source/main_table_reference.json').read_text())['groups']
base=[json.loads((ROOT/'outputs/router_shared_image_hidden_20260910/runs'/g['run']/'config.json').read_text()) for g in groups if g['key'][0] in ['cub','nlvr2']]
assert len(base)==8,len(base)
plan=[]
for seed in range(2026,2031):
    for cfg in base:
        for method in ['ova','css']:
            c=dict(cfg);c.update(seed=seed,target='four_state',ranking='native_probability_difference',native_method=method)
            c['id']=f"{cfg['expert']}_{cfg['endpoint']}_{method}_s{seed}"
            plan.append(c)
OUT.mkdir(exist_ok=True)
(OUT/'job_plan.json').write_text(json.dumps(plan,indent=2))
(OUT/'method_lock.json').write_text(json.dumps({'methods':['ova','css'],'chosen_before_test':True,'K':{'cub':200,'nlvr2':2},
    'output_semantics':'K task classes plus expert correctness; retain the classifier argmax answer',
    'excluded':'gRefCOCO box sets and Construction multi-label events are not single-label K-class tasks',
    'checkpoint':'same validation Score25 and early stopping; final retained answers come from learned class branch',
    'decision':'expert probability minus max class probability; original zero threshold also reported',
    'cost':0,'historical_test_access':True,'time':time.time()},indent=2))
queues=[Queue(),Queue()]
for i,cfg in enumerate(plan):
    if not (OUT/'runs'/cfg['id']/'test_evaluation.json').exists():queues[(i//2)%2].put(cfg)
stopped=Event()

def worker(gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1')
    while not stopped.is_set():
        try:cfg=queues[gpu].get_nowait()
        except Empty:return
        dest=OUT/'runs'/cfg['id'];dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'completion.json').exists():
            (dest/'config.json').write_text(json.dumps(cfg,indent=2))
            with (dest/'train.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train_native.py'),'--root',str(ROOT),'--config',str(dest/'config.json'),'--output',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (dest/'test.log').open('w') as log:
            subprocess.run([sys.executable,str(HERE/'evaluate_native.py'),'--run',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'completed':cfg['id'],'gpu':gpu,'time':time.time()}),flush=True)

with ThreadPoolExecutor(max_workers=10) as pool:
    futures=[pool.submit(worker,gpu) for gpu in [0,1] for _ in range(5)]
    for f in as_completed(futures):
        if f.exception() is not None:
            stopped.set()
            (OUT/'failure.json').write_text(json.dumps({'error':str(f.exception()),'time':time.time()},indent=2))
        f.result()
(OUT/'complete.json').write_text(json.dumps({'runs':len(plan),'time':time.time()}))
