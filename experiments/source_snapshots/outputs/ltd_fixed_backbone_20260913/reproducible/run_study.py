"""Two GPU queues; all five seeds; no test-dependent choices."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import time

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/ltd_fixed_backbone_20260913'
METHODS={'css':('css',0.),'ova':('ova',0.),'posthoc_c0':('posthoc',0.),
         'posthoc_c005':('posthoc',.05),'posthoc_c01':('posthoc',.1),
         'posthoc_c02':('posthoc',.2),'posthoc_c04':('posthoc',.4)}
SEEDS=[2026,2027,2028,2029,2030]


def worker(gpu,queue):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1')
    for cfg in queue:
        dest=OUT/'runs'/cfg['id'];dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'completion.json').exists():
            (dest/'config.json').write_text(json.dumps(cfg,indent=2))
            with (dest/'train.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train_ltd.py'),'--root',str(ROOT),
                    '--config',str(dest/'config.json'),'--output',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (dest/'test_evaluation.json').exists():
            with (dest/'test.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'evaluate_ltd.py'),'--run',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'completed':cfg['id'],'gpu':gpu,'time':time.time()}),flush=True)


if __name__=='__main__':
    groups=json.loads((HERE/'source/main_table_reference.json').read_text())['groups']
    base=[json.loads((ROOT/'outputs/router_shared_image_hidden_20260910/runs'/g['run']/'config.json').read_text()) for g in groups]
    jobs=[]
    # Complete seed 2026 across all pairs first to make early evidence interpretable.
    for seed in SEEDS:
        for c in base:
            for name,(method,cost) in METHODS.items():
                cfg=dict(c);cfg.update(seed=seed,target='four_state',ranking='ltd_score',
                    ltd_method=method,method=name,call_cost=cost)
                cfg['id']=f"{c['expert']}_{c['endpoint']}_{name}_s{seed}"
                jobs.append(cfg)
    OUT.mkdir(exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps({'seeds':SEEDS,'methods':METHODS,'pairs':len(base),'fits':len(jobs),
        'original_reference':'objective_multiseed_20260912','fixed':'same pair configs, frozen 8M encoder and five input branches, optimizer, batch, epochs, early stopping, normalization, splits and prediction records',
        'adaptation':'CSS/OvA operate on two frozen actions, NOT K jointly trained classes. Output dimensions necessarily differ.',
        'posthoc':'Equation 14 with logistic loss; cost trained separately for each prespecified value; score thresholds use the existing protocol.',
        'chow':'reuse the existing confidence baseline; Chow has no learned routing backbone',
        'selection':'checkpoint by original validation Score25; thresholds validation-only; all cost variants reported; any preferred cost selected on aggregated validation only',
        'test':'same previously inspected test split, no claim of new blind holdout; test never enters training or selection',
        'construction':'event-mean correctness targets, single routing action per image, macro-F1 evaluation',
        'no_new_vlm_calls':True},indent=2))
    (OUT/'method_lock.json').write_text(json.dumps({'methods':METHODS,'fixed_before_test':True,'time':time.time()},indent=2))
    (OUT/'job_plan.json').write_text(json.dumps(jobs,indent=2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu,jobs[gpu::2]) for gpu in [0,1]]
        for f in futures:f.result()
    (OUT/'complete.json').write_text(json.dumps({'completed_fits':len(jobs),'time':time.time()},indent=2))
