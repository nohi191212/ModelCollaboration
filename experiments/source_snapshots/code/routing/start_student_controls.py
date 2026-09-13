"""Explicit four-GPU student controls. Never start while router workers own GPUs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--queue',choices=['student_control','task_holdout_student'],default='student_control');args=ap.parse_args()
    work=ROOT/'outputs/router_exploration_25pct_20260910/distillation_controls'
    queue=json.loads((work/(args.queue+'_queue.json')).read_text())
    plan='student_control_plan_precheck.json' if args.queue=='student_control' else 'task_holdout_student_plan.json'
    assert json.loads((work/plan).read_text())['configs_checked']==len(queue)
    if args.prepare_only:
        print(json.dumps({'students':len(queue),'workers':4,'jobs_per_worker':[len(queue[i::4]) for i in range(4)],'gpu_launched':False}));raise SystemExit(0)
    stage=work/('pipeline' if args.queue=='student_control' else 'task_holdout_pipeline');stage.mkdir(exist_ok=True)
    if args.worker is None:
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        assert len(memory)==4 and all(int(x)<500 for x in memory),memory
        if (stage/'started.json').exists():raise FileExistsError('Inspect the previous student-control run before restarting')
        (stage/'started.json').write_text(json.dumps({'pid':os.getpid(),'unix_time':time.time()}))
        processes=[]
        for gpu in range(4):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
            log=(stage/f'worker{gpu}.log').open('w')
            p=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(gpu),'--queue',args.queue],stdout=log,stderr=subprocess.STDOUT,env=env)
            processes.append((gpu,p,log))
        (stage/'worker_processes.json').write_text(json.dumps({str(g):p.pid for g,p,log in processes}))
        exits={}
        for gpu,p,log in processes:exits[str(gpu)]=p.wait();log.close()
        (stage/'worker_exits.json').write_text(json.dumps(exits))
        assert all(code==0 for code in exits.values()),exits
        (stage/'completion.json').write_text(json.dumps({'status':'students_and_features_complete_only','students':len(queue),'all_exploration_complete':False}))
    else:
        assert args.worker in range(4)
        for cfg in queue[args.worker::4]:
            config=work/'configs'/(cfg['id']+'.json')
            assert json.loads(config.read_text())==cfg
            # Same real batch size and architecture; outputs remain separate from formal students.
            for smoke in [True,False]:
                dest=work/('smoke_students' if smoke else 'students')/cfg['id']
                if not (dest/'completion.json').exists():
                    if dest.exists():raise FileExistsError(('Incomplete run: inspect before restarting',str(dest)))
                    label='smoke' if smoke else 'formal'
                    with (stage/(cfg['id']+'_'+label+'.log')).open('w') as log:
                        command=[sys.executable,'-u',str(ROOT/'code/routing/train_student_control.py'),'--config',str(config)]
                        if smoke:command+=['--smoke-steps','2']
                        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                        state={'worker_pid':os.getpid(),'child_pid':p.pid,'id':cfg['id'],'smoke':smoke,'status':'running','unix_time':time.time()}
                        statefile=stage/f'worker{args.worker}_state.json';statefile.write_text(json.dumps(state))
                        code=p.wait();state.update(status='complete' if code==0 else 'failed_needs_diagnosis',exit_code=code);statefile.write_text(json.dumps(state))
                    if code:raise SystemExit(code)
                result=json.loads((dest/'completion.json').read_text())
                assert result['status']=='complete' and result['smoke']==smoke
            features=work.parent/'features'/('control_'+cfg['id'])
            if not (features/'completion.json').exists():
                if features.exists():raise FileExistsError(('Partial features: inspect before restarting',str(features)))
                with (stage/(cfg['id']+'_features.log')).open('w') as log:
                    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/routing/extract_features.py'),'--control',cfg['id']],stdout=log,stderr=subprocess.STDOUT)
                    state={'worker_pid':os.getpid(),'child_pid':p.pid,'id':cfg['id'],'phase':'features','status':'running','unix_time':time.time()}
                    statefile=stage/f'worker{args.worker}_state.json';statefile.write_text(json.dumps(state))
                    code=p.wait();state.update(status='complete' if code==0 else 'failed_needs_diagnosis',exit_code=code);statefile.write_text(json.dumps(state))
                if code:raise SystemExit(code)
            feature_report=json.loads((features/'completion.json').read_text())
            assert feature_report['status']=='complete' and feature_report['student']==str(work/'students'/cfg['id']/'final.pt')
            assert feature_report['tokenizer']==str(work/'students'/cfg['id']/'student_tokenizer.json')
        (stage/f'worker{args.worker}_state.json').write_text(json.dumps({'status':'all_assigned_students_complete','worker_pid':os.getpid(),'unix_time':time.time()}))
