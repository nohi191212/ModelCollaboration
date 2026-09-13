"""Run reviewed frozen-feature ablations; this is not the final exploration stage."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    assert json.loads((work/'initial_pipeline/completion.json').read_text())['status']=='initial_stage_complete_only'
    analysis=json.loads((work/'initial_queue_analysis.json').read_text())
    assert analysis['all_queue_fits_complete'] and analysis['fits_complete']==240
    for warning in analysis['warnings']:
        assert json.loads((work/'runs'/warning['id']/'anomaly_review.json').read_text())['decision']=='no_bug'
    queue=json.loads((work/'followup_queue.json').read_text())
    design=json.loads((work/'followup_design.json').read_text())
    assert len(queue)==design['new_fits'] and queue
    # Cover every expert/input combination and every new code path before full fits.
    covered=set();smokes=[]
    for cfg in queue:
        coverage={('expert_inputs',cfg['expert'],tuple(cfg['branches'])),
                  ('objective_strategy',cfg['target'],cfg['strategy']),
                  ('dimension',cfg['representation_dim']),('student',cfg['student']),
                  ('fraction',cfg['task'],cfg['fraction']),
                  ('layers',cfg['expert'],tuple(cfg.get('hidden_layers',[cfg['hidden_layer']]))),
                  ('shuffle',cfg['task'],cfg.get('hidden_shuffle',False))}
        if coverage-covered:
            smokes.append(cfg);covered.update(coverage)
    plan={'new_fits':len(queue),'smoke_count':len(smokes),'smoke_ids':[c['id'] for c in smokes],
          'coverage':[list(x) for x in sorted(covered,key=str)],'all_exploration_complete':False}
    (work/'followup_launch_plan.json').write_text(json.dumps(plan,indent=2))
    if args.prepare_only:
        print(json.dumps({'new_fits':len(queue),'smoke_count':len(smokes),'gpu_launched':False}));raise SystemExit(0)
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(memory)==4 and all(int(x)<500 for x in memory),memory
    stage=work/'followup_pipeline';stage.mkdir(exist_ok=True)
    (stage/'started.json').write_text(json.dumps({'pid':os.getpid(),'unix_time':time.time()}))
    cfgdir=work/'configs'
    # Four independent checks at a time; preserve failures and stop before queue launch.
    for start in range(0,len(smokes),4):
        processes=[]
        for gpu,cfg in enumerate(smokes[start:start+4]):
            result=work/'smoke_runs'/cfg['id']/'completion.json'
            if result.exists():
                completed=json.loads(result.read_text());assert completed['smoke'] and completed['status']=='complete'
                continue
            cfgfile=cfgdir/(cfg['id']+'.json');cfgfile.write_text(json.dumps(cfg,indent=2))
            log=(stage/(cfg['id']+'_smoke.log')).open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/routing/train_router.py'),'--config',str(cfgfile),'--smoke'],stdout=log,stderr=subprocess.STDOUT,env=env)
            processes.append((cfg['id'],p,log))
        exits={}
        for name,p,log in processes:exits[name]=p.wait();log.close()
        (stage/f'smoke_batch_{start:04d}.json').write_text(json.dumps(exits))
        assert all(code==0 for code in exits.values()),exits
    for cfg in smokes:
        path=work/'smoke_runs'/cfg['id'];result=json.loads((path/'completion.json').read_text())
        if result['anomalies']:
            assert json.loads((path/'anomaly_review.json').read_text())['decision']=='no_bug'
    processes=[]
    for gpu in range(4):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
        log=(stage/f'worker{gpu}.log').open('a')
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/routing/run_queue.py'),'--queue',str(work/'followup_queue.json'),'--worker',str(gpu)],stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((gpu,p,log))
    (stage/'worker_processes.json').write_text(json.dumps({str(g):p.pid for g,p,log in processes}))
    exits={}
    for gpu,p,log in processes:exits[str(gpu)]=p.wait();log.close()
    (stage/'worker_exits.json').write_text(json.dumps(exits))
    assert all(code==0 for code in exits.values()),exits
    (stage/'completion.json').write_text(json.dumps({'status':'frozen_single_factor_complete_only','unix_time':time.time(),'fits':len(queue),'all_exploration_complete':False}))
