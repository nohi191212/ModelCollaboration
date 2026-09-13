"""Read actual process and output state; never launch, kill, or restart a job."""
import json,subprocess,time
from collections import Counter
from pathlib import Path

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910';stage=work/'initial_pipeline'
    result={'unix_time':time.time(),'features':{},'smoke_completed':0,'initial_fits_completed':0,'other_fits_completed':0,'anomalies':[],'workers':[]}
    for size in ['1M','2M','4M','8M']:
        done=work/'features'/size/'completion.json';log=stage/(size+'_features.log')
        result['features'][size]={'complete':done.exists(),'last_log':log.read_text().splitlines()[-2:] if log.exists() else []}
    for file in sorted((work/'smoke_runs').glob('*/completion.json')):
        done=json.loads(file.read_text())
        assert done['smoke']
        result['smoke_completed']+=1
        if done['anomalies']:result['anomalies'].append({'run':file.parent.name,'smoke':True,'reasons':done['anomalies']})
    counts=Counter();stage_counts=Counter();latest=[]
    for file in sorted((work/'runs').glob('*/completion.json')):
        done=json.loads(file.read_text());cfg=json.loads((file.parent/'config.json').read_text())
        assert done['status']=='complete' and not done['smoke']
        field='initial_fits_completed' if cfg['stage']=='initial' else 'other_fits_completed'
        result[field]+=1;counts[cfg['expert']]+=1;stage_counts[cfg['stage']]+=1
        latest.append({'id':cfg['id'],'seconds':done['seconds'],'epochs':done['epochs'],'best_selection':done['best_selection'],'mtime':file.stat().st_mtime})
        if done['anomalies']:
            review=file.parent/'anomaly_review.json'
            result['anomalies'].append({'run':cfg['id'],'smoke':False,'reasons':done['anomalies'],'review':json.loads(review.read_text()) if review.exists() else None})
    result['fits_by_expert']=dict(counts);result['fits_by_stage']=dict(stage_counts);result['latest_fits']=sorted(latest,key=lambda row:row['mtime'])[-8:]
    result['worker_read_retries']=[]
    for file in sorted((work/'queue_logs').glob('*_worker*.json')):
        for attempt in range(3):
            try:
                worker=json.loads(file.read_text())
                break
            except json.JSONDecodeError:
                result['worker_read_retries'].append({'file':file.name,'attempt':attempt+1})
                if attempt==2:raise
                time.sleep(.05)
        result['workers'].append({'file':file.name,**worker})
    for name in ['feature_exits.json','worker_exits.json','completion.json']:
        if (stage/name).exists():result[name]=json.loads((stage/name).read_text())
    log=work/'launch_logs/initial_master.log'
    result['master_log_tail']=log.read_text().splitlines()[-15:] if log.exists() else []
    exit_file=work/'launch_logs/initial_master.exit'
    result['master_exit']=exit_file.read_text().strip() if exit_file.exists() else None
    followup=work/'followup_pipeline'
    result['followup']={}
    for name in ['started.json','worker_processes.json','worker_exits.json','completion.json']:
        if (followup/name).exists():result['followup'][name]=json.loads((followup/name).read_text())
    for suffix in ['log','exit']:
        file=work/'launch_logs'/('followup_master.'+suffix)
        result['followup']['master_'+suffix]=file.read_text().splitlines()[-15:] if file.exists() else None
    processes=subprocess.check_output(['ps','-eo','pid,ppid,etime,stat,args'],text=True).splitlines()
    result['live_processes']=[line for line in processes if '/code/routing/' in line or 'python -u code/routing/' in line]
    result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).splitlines()
    parallel=work/'parallel_followup';result['parallel_followup']={}
    for name in ['started.json','draining.json','state.json','completion.json']:
        if (parallel/name).exists():result['parallel_followup'][name]=json.loads((parallel/name).read_text())
    interaction=work/'interaction_pipeline';result['interaction']={}
    for name in ['started.json','smoke_processes.json','state.json','completion.json']:
        if (interaction/name).exists():result['interaction'][name]=json.loads((interaction/name).read_text())
    for suffix in ['log','exit']:
        file=work/'launch_logs'/('interaction_master.'+suffix)
        result['interaction']['master_'+suffix]=file.read_text().splitlines()[-20:] if file.exists() else None
    result['interaction']['smoke_batches_complete']=len(list(interaction.glob('smoke_batch_*.json')))
    light=work/'interaction_light_pipeline';result['interaction_light']={}
    for name in ['started.json','worker_processes.json','worker_exits.json','completion.json']:
        if (light/name).exists():result['interaction_light'][name]=json.loads((light/name).read_text())
    if (work/'interaction_light_queue.json').exists():
        light_queue=json.loads((work/'interaction_light_queue.json').read_text())
        result['interaction_light']['planned']=len(light_queue)
        result['interaction_light']['completed']=sum((work/'runs'/cfg['id']/'completion.json').exists() for cfg in light_queue)
        result['interaction_light']['deferred_finetunes']=87
    small=work/'small_finetune_pipeline';result['small_finetune']={}
    for name in ['started.json','active.json','completion.json']:
        if (small/name).exists():result['small_finetune'][name]=json.loads((small/name).read_text())
    result['small_finetune']['fits']=[json.loads(file.read_text()) for file in sorted((work/'runs').glob('finetune_small_*/completion.json'))]
    for suffix in ['log','exit']:
        file=work/'launch_logs'/('small_finetune_master.'+suffix)
        result['small_finetune']['master_'+suffix]=file.read_text().splitlines()[-15:] if file.exists() else None
    (work/'progress_snapshot.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
