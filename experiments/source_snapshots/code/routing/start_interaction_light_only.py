"""Pause the old dispatcher, drain fits, and run only the approved frozen work."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    w=ROOT/'outputs/router_exploration_25pct_20260910'
    old=json.loads((w/'interaction_pipeline/started.json').read_text())['pid']
    cmd=(Path('/proc')/str(old)/'cmdline').read_bytes()
    assert b'start_interaction.py' in cmd
    stage=w/'interaction_light_pipeline';stage.mkdir(exist_ok=False)
    (stage/'started.json').write_text(json.dumps({'pid':os.getpid(),'paused_dispatcher':old,'unix_time':time.time()}))
    os.kill(old,signal.SIGSTOP)
    time.sleep(.2)
    listing=subprocess.run(['ps','--ppid',str(old),'-o','pid='],capture_output=True,text=True)
    assert listing.returncode in [0,1]
    children=[int(x) for x in listing.stdout.split()]
    (stage/'draining.json').write_text(json.dumps({'children':children}))
    while any((Path('/proc')/str(pid)/'stat').exists() and (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z' for pid in children):
        time.sleep(2)
    queue=json.loads((w/'interaction_queue.json').read_text())
    frozen=[c for c in queue if c['freeze_encoder']]
    deferred=[c['id'] for c in queue if not c['freeze_encoder']]
    assert len(frozen)==3127 and len(deferred)==87
    assert all(c['seed']==42 for c in frozen)
    for cfg in queue:
        dest=w/'runs'/cfg['id']
        if dest.exists():
            done=json.loads((dest/'completion.json').read_text())
            assert done['status']=='complete' and not done['smoke']
            assert not done['anomalies'],cfg['id']
        if not cfg['freeze_encoder']:assert not dest.exists(),cfg['id']
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(memory)==4 and all(int(x)<500 for x in memory),memory
    qfile=w/'interaction_light_queue.json'
    assert not qfile.exists()
    qfile.write_text(json.dumps(frozen,indent=2))
    (stage/'deferred.json').write_text(json.dumps({'reason':'Heavy experiments deferred pending user confirmation','ids':deferred,'do_not_resume_old_dispatcher':old},indent=2))
    jobs=[]
    for worker in range(16):
        log=(stage/f'worker{worker}.log').open('x')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(worker%4),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/routing/run_queue.py'),'--queue',str(qfile),'--worker',str(worker),'--workers','16','--cpu-threads','1'],env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((worker,p,log))
    (stage/'worker_processes.json').write_text(json.dumps({str(i):p.pid for i,p,log in jobs}))
    exits={}
    for i,p,log in jobs:exits[str(i)]=p.wait();log.close()
    (stage/'worker_exits.json').write_text(json.dumps(exits))
    assert all(code==0 for code in exits.values()),exits
    (stage/'completion.json').write_text(json.dumps({'status':'frozen_only_complete','fits':len(frozen),'deferred_finetunes':len(deferred),'all_exploration_complete':False}))
