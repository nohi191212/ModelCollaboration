from pathlib import Path
import os
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
O=R/'outputs/full_budget_20260913'
ap=argparse.ArgumentParser();ap.add_argument('--start',action='store_true');a=ap.parse_args()
if a.start:
    assert not O.exists(),O
    O.mkdir();(O/'runs').mkdir();(O/'logs').mkdir()
    prior=json.loads((R/'outputs/fixed_score_search_20260913/results.json').read_text())
    runs=[];audit=[]
    for seed in range(2026,2031):
        for p in prior['pairs']:
            source=R/'outputs/objective_multiseed_20260912/runs'/f"repeat_{p['expert']}_{p['endpoint']}_four_difference_s{seed}"
            checkpoints=[x.name for x in source.glob('*.pt')]
            audit.append({'source':str(source),'checkpoints':checkpoints})
            runs.append({'source':str(source),'id':f"full_{p['expert']}_{p['endpoint']}_s{seed}"})
    (O/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2))
    (O/'queue.json').write_text(json.dumps(runs,indent=2))
    protocol={'pairs':16,'seeds':list(range(2026,2031)),'runs':80,'gpu_ids':[0,1],'concurrent_runs':10,'training':'Same data, frozen backbone, paired cross-entropy, optimizer, random seeds, maximum 50 epochs and patience 8 as original. Only validation selection changes.','selection':'At each epoch, evaluate normalized score with alpha in [0,.25,.5,.75,1], beta=1 on validation exact-call curve at 101 points including endpoints. Maximize full-curve mean; same-epoch ties prefer smaller alpha; epoch ties retain earlier epoch. Early stopping follows this metric.','threshold':'Select maximum validation task score over budgets 0..100% inclusive; ties fewer calls then lower budget. Never uses test to select.','caveat':'Exploratory revision; test previously inspected. Historical comparison changes model selection and scoring jointly; same-checkpoint score controls are additionally reported.','time':time.time()}
    (O/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with (O/'queue.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (O/'queue_pid.json').write_text(json.dumps({'pid':process.pid}))
    print('STARTED',process.pid,'runs',len(runs),'all_only_best',all(x['checkpoints']==['best.pt'] for x in audit))
    sys.exit(0)
runs=json.loads((O/'queue.json').read_text())
def worker(worker_id):
    gpu=worker_id%2;done=[]
    for job in runs[worker_id::10]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (O/'logs'/(job['id']+'.log')).open('w') as log:
            process=subprocess.run([sys.executable,str(Path(__file__).with_name('full_budget_train.py')),'--source',job['source'],'--output',str(O/'runs'/job['id'])],stdout=log,stderr=subprocess.STDOUT,env=env)
        item={'id':job['id'],'gpu':gpu,'exit_code':process.returncode,'time':time.time()}
        (O/'logs'/(job['id']+'.exit.json')).write_text(json.dumps(item));done.append(item)
        print(json.dumps(item),flush=True)
        if process.returncode!=0:raise RuntimeError(item)
    return done
with ThreadPoolExecutor(max_workers=10) as pool:
    futures=[pool.submit(worker,i) for i in range(10)]
    completed=[row for future in as_completed(futures) for row in future.result()]
(O/'complete.json').write_text(json.dumps({'runs':len(completed),'status':'complete','time':time.time()},indent=2))
print('ALL COMPLETE',len(completed),flush=True)
