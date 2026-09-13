"""Five slots per assigned GPU, adopting in-flight jobs without rerunning them."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Event
import json
import time
from run_study import OUT, worker

if __name__=='__main__':
    plan=json.loads((OUT/'job_plan.json').read_text())
    handoff=json.loads((OUT/'concurrency_handoff.json').read_text())
    adopted={j['id']:j for j in handoff['active']}
    queues=[Queue(),Queue()]
    for cfg in plan:
        if cfg['id'] in adopted:
            queues[adopted[cfg['id']]['gpu']].put(cfg)
    pending=[cfg for cfg in plan if cfg['id'] not in adopted and not (OUT/'runs'/cfg['id']/'test_evaluation.json').exists()]
    for i,cfg in enumerate(pending):queues[i%2].put(cfg)
    stopped=Event()

    def slot(gpu):
        while not stopped.is_set():
            try:cfg=queues[gpu].get_nowait()
            except Empty:return
            if cfg['id'] in adopted:
                job=adopted[cfg['id']]
                print(json.dumps({'adopting':cfg['id'],'pid':job['pid'],'gpu':gpu}),flush=True)
                proc=Path(f"/proc/{job['pid']}/stat")
                while proc.exists():
                    # A finished orphan can briefly remain a zombie.
                    if proc.read_text().split()[2]=='Z':break
                    time.sleep(2)
                marker='completion.json' if job['phase']=='train' else 'test_evaluation.json'
                if not (OUT/'runs'/cfg['id']/marker).exists():
                    raise RuntimeError(f"Adopted {job['phase']} failed: {cfg['id']}; inspect its log")
            worker(gpu,[cfg])

    succeeded=False
    try:
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures=[pool.submit(slot,gpu) for gpu in [0,1] for _ in range(5)]
            for future in as_completed(futures):
                if future.exception() is not None:
                    stopped.set()
                    (OUT/'failure.json').write_text(json.dumps({'error':str(future.exception()),'time':time.time(),'inspect':'study_ten.log'},indent=2))
                future.result()
        missing=[cfg['id'] for cfg in plan if not (OUT/'runs'/cfg['id']/'test_evaluation.json').exists()]
        if missing:raise RuntimeError(('Missing evaluations',missing))
        (OUT/'complete.json').write_text(json.dumps({'completed_fits':len(plan),'time':time.time(),'concurrency':10},indent=2))
        succeeded=True
    finally:
        if not succeeded and not (OUT/'failure.json').exists():
            (OUT/'failure.json').write_text(json.dumps({'error':'Ten-slot scheduler failed','time':time.time(),'inspect':'study_ten.log'},indent=2))
