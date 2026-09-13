"""Start only after strict distillation finishes and all four GPUs are free."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910';students=ROOT/'outputs/mini_siglip_strict_20260910_300ep'
    assert (students/'pipeline.exit').read_text().strip()=='0'
    for size in ['1M','2M','4M','8M']:
        assert json.loads((students/size/'completion.json').read_text())['epochs']==300
        assert (students/size/'epoch_300.pt').is_file()
    from prepare_exploration import EXPERTS
    for expert in EXPERTS: assert json.loads((work/'datasets'/expert/'completion.json').read_text())['status']=='complete'
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(memory)==4 and all(int(x)<500 for x in memory),memory
    stage=work/'initial_pipeline';stage.mkdir()
    (stage/'started.json').write_text(json.dumps({'pid':os.getpid(),'unix_time':time.time()},indent=2))
    processes=[];logs=[]
    for gpu,size in enumerate(['1M','2M','4M','8M']):
        if (work/'features'/size/'completion.json').exists():continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        log=(stage/(size+'_features.log')).open('w');logs.append(log)
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/routing/extract_features.py'),'--size',size],stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((size,p))
    (stage/'feature_processes.json').write_text(json.dumps({s:p.pid for s,p in processes},indent=2))
    exits={s:p.wait() for s,p in processes}
    for log in logs:log.close()
    (stage/'feature_exits.json').write_text(json.dumps(exits,indent=2))
    assert all(code==0 for code in exits.values()),exits
    queue=json.loads((work/'initial_queue.json').read_text());cfgdir=work/'configs';cfgdir.mkdir(exist_ok=True)
    # Real end-to-end checks across all tasks, both fusions and all three targets.
    for cfg in queue:
        if cfg['expert'] not in ['cub','nlvr2','instancevg','yolo26x'] or cfg['seed']!=42 or cfg['endpoint'] not in ['both','Qwen3.8']:continue
        cfgfile=cfgdir/(cfg['id']+'.json');cfgfile.write_text(json.dumps(cfg,indent=2))
        with (stage/(cfg['id']+'_smoke.log')).open('w') as log:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2')
            code=subprocess.call([sys.executable,'-u',str(ROOT/'code/routing/train_router.py'),'--config',str(cfgfile),'--smoke'],stdout=log,stderr=subprocess.STDOUT,env=env)
        assert code==0,(cfg['id'],code)
    processes=[];logs=[]
    for gpu in range(4):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
        log=(stage/f'worker{gpu}.log').open('w');logs.append(log)
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/routing/run_queue.py'),'--queue',str(work/'initial_queue.json'),'--worker',str(gpu)],stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((gpu,p))
    (stage/'worker_processes.json').write_text(json.dumps({str(g):p.pid for g,p in processes},indent=2))
    exits={str(g):p.wait() for g,p in processes}
    for log in logs:log.close()
    (stage/'worker_exits.json').write_text(json.dumps(exits,indent=2))
    assert all(code==0 for code in exits.values()),exits
    (stage/'completion.json').write_text(json.dumps({'status':'initial_stage_complete_only','unix_time':time.time(),'fits':len(queue),'all_exploration_complete':False},indent=2))
