"""One GPU worker, persistent per-run logs, stop on unreviewed abnormalities."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--queue',required=True);ap.add_argument('--worker',type=int,required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--cpu-threads',type=int,default=2);a=ap.parse_args()
    work=ROOT/'outputs/router_exploration_25pct_20260910';queue=json.loads(Path(a.queue).read_text())
    assert 0 <= a.worker < a.workers
    jobs=queue[a.worker::a.workers];configdir=work/'configs';configdir.mkdir(exist_ok=True)
    logdir=work/'queue_logs';logdir.mkdir(exist_ok=True)
    statefile=logdir/(Path(a.queue).stem+f'_worker{a.worker}.json')
    for cfg in jobs:
        resultfile=work/'runs'/cfg['id']/'completion.json'
        if not resultfile.exists():
            configfile=configdir/(cfg['id']+'.json');configfile.write_text(json.dumps(cfg,indent=2))
            state={'worker_pid':os.getpid(),'active_id':cfg['id'],'status':'running','unix_time':time.time()}
            with (logdir/(cfg['id']+'.log')).open('a') as log:
                p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/routing/train_router.py'),'--config',str(configfile),'--cpu-threads',str(a.cpu_threads)],stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=p.pid;statefile.write_text(json.dumps(state,indent=2));code=p.wait()
            if code:
                state.update(status='failed_needs_diagnosis',exit_code=code);statefile.write_text(json.dumps(state,indent=2));raise SystemExit(code)
        result=json.loads(resultfile.read_text())
        if result['anomalies']:
            review=resultfile.parent/'anomaly_review.json'
            if not review.exists() or json.loads(review.read_text())['decision']!='no_bug':
                statefile.write_text(json.dumps({'status':'anomaly_needs_diagnosis','active_id':cfg['id'],'worker_pid':os.getpid()},indent=2))
                raise RuntimeError(('Review anomaly before accepting',cfg['id'],result['anomalies']))
    statefile.write_text(json.dumps({'status':'complete','worker_pid':os.getpid(),'queue':a.queue,'jobs':len(jobs),'unix_time':time.time()},indent=2))
