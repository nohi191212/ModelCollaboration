"""Explicitly rerun the 400 released configurations on prepared feature caches.

No work occurs on import. This portable launcher removes the historical
seed-42 pilot from the current entry and selects by five-seed mean minus SD.
Training/evaluation algorithms and released configuration values are unchanged.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True,help='Project containing all prepared original feature caches')
    p.add_argument('--artifacts',type=Path,required=True,help='Downloaded HF repository with experiment_data/runs')
    p.add_argument('--output',type=Path,required=True,help='New directory for rerun results; never use the released evidence directory')
    p.add_argument('--gpus',type=int,nargs='+',required=True)
    a=p.parse_args()
    here=Path(__file__).resolve().parent
    configs=[json.loads(x.read_text()) for x in sorted((a.artifacts/'experiment_data/runs').glob('repeat_*/config.json'))]
    assert len(configs)==400
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)

    def worker(gpu,queue,evaluate):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1')
        for cfg in queue:
            dest=a.output/'runs'/cfg['id']
            if not evaluate:
                dest.mkdir(parents=True)
                config=dest/'config.json';config.write_text(json.dumps(cfg,indent=2))
                command=[sys.executable,str(here/'train_objective.py'),'--root',str(a.root),'--config',str(config),'--output',str(dest)]
            else:
                command=[sys.executable,str(here/'evaluate_objective.py'),'--root',str(a.root),'--run',str(dest)]
            with (dest/('test.log' if evaluate else 'train.log')).open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(json.dumps({'completed':cfg['id'],'gpu':gpu,'evaluated':evaluate}),flush=True)

    for evaluate in [False,True]:
        if evaluate:
            subprocess.run([sys.executable,str(here/'select_five_seeds.py'),'--results',str(a.output)],check=True)
        with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            futures=[pool.submit(worker,gpu,configs[i::len(a.gpus)],evaluate) for i,gpu in enumerate(a.gpus)]
            for future in futures:future.result()
    (a.output/'complete.json').write_text(json.dumps({'status':'complete','repeat_fits':400,'test_runs':400},indent=2))


if __name__=='__main__':main()
