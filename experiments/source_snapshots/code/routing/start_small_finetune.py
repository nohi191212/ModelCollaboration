"""Run only eight authorized raw-image probes after lightweight fits finish."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    w=ROOT/'outputs/router_exploration_25pct_20260910'
    done=json.loads((w/'interaction_light_pipeline/completion.json').read_text())
    assert done['status']=='frozen_only_complete'
    queue=json.loads((w/'small_finetune_queue.json').read_text())
    assert len(queue)==8 and all(c['seed']==42 and c['fraction']==.25 and not c['freeze_encoder'] for c in queue)
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(memory)==4 and all(int(x)<500 for x in memory),memory
    stage=w/'small_finetune_pipeline';stage.mkdir(exist_ok=False)
    (stage/'started.json').write_text(json.dumps({'pid':os.getpid(),'unix_time':time.time()}))
    for smoke in [True,False]:
        for start in [0,4]:
            jobs=[]
            for gpu,cfg in enumerate(queue[start:start+4]):
                config=w/'configs'/(cfg['id']+'.json');config.write_text(json.dumps(cfg,indent=2))
                log=(stage/(cfg['id']+('_smoke.log' if smoke else '_formal.log'))).open('x')
                command=[sys.executable,'-u',str(ROOT/'code/routing/train_router.py'),'--config',str(config),'--cpu-threads','2']
                if smoke:command.append('--smoke')
                p=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
                jobs.append((cfg['id'],p,log))
            (stage/'active.json').write_text(json.dumps({'smoke':smoke,'processes':{name:p.pid for name,p,log in jobs}}))
            exits={}
            for name,p,log in jobs:exits[name]=p.wait();log.close()
            (stage/f'exits_{smoke}_{start}.json').write_text(json.dumps(exits))
            assert all(c==0 for c in exits.values()),exits
            for name,p,log in jobs:
                dest=w/('smoke_runs' if smoke else 'runs')/name
                result=json.loads((dest/'completion.json').read_text())
                assert result['status']=='complete' and result['smoke']==smoke
                assert result['encoder_trainable_parameters']>0
                if result['anomalies']:
                    assert json.loads((dest/'anomaly_review.json').read_text())['decision']=='no_bug'
    (stage/'completion.json').write_text(json.dumps({'status':'eight_finetunes_complete_requires_replay','fits':8,'all_exploration_complete':False}))
