"""Validation-only shared coordinate search, with bounded concurrent GPU fits."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--trainer',type=Path,required=True)
    ap.add_argument('--max-per-gpu',type=int,default=8)
    ap.add_argument('--gpus',default='0,1,2,3')
    a=ap.parse_args()
    w=a.work;protocol=json.loads((w/'protocol.json').read_text())
    for folder in ['configs','runs','logs']:(w/folder).mkdir(exist_ok=True)
    state_path=w/'DSE_STATE.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['status']=='search_complete':raise FileExistsError('Shared selection already locked')
    else:
        state={'status':'running','started_at':time.time(),'completed_stages':[],'selected':None,'candidates':[],
               'failures':[],'test_loaded':False}
    gpus=a.gpus.split(',')

    def save():state_path.write_text(json.dumps(state,indent=2))

    def candidate_for(settings):
        for item in state['candidates']:
            if item['settings']==settings:return item
        if len(state['candidates'])>=protocol['maximum_shared_settings']:raise ValueError('Declared shared-setting cap exceeded')
        index=len(state['candidates'])
        item={'id':f'shared_{index:02d}','settings':settings,'runs':[],'status':'prepared'}
        for expert,spec in protocol['experts'].items():
            for endpoint in ['MiniCPM','Qwen3.8']:
                run_id=f"{item['id']}_{expert}_{endpoint}"
                cfg=dict(protocol['fixed'],**settings,expert=expert,task=spec['task'],endpoint=endpoint,
                         branches=spec['branches'],hidden_layer=spec['layers'][str(settings['hidden_percent'])]['key'],
                         id=run_id,stage='shared_mandatory_inputs')
                (w/'configs'/(run_id+'.json')).write_text(json.dumps(cfg,indent=2))
                item['runs'].append(run_id)
        state['candidates'].append(item);save()
        return item

    def record(event):
        with (w/'scheduler.jsonl').open('a') as handle:handle.write(json.dumps(dict(event,time=time.time()))+'\n')

    def execute(items):
        queue=[run for item in items for run in item['runs'] if not (w/'runs'/run/'completion.json').exists() and run not in state['failures']]
        active=[]
        try:
            while queue or active:
                for job in active[:]:
                    code=job['process'].poll()
                    if code is None:continue
                    valid=code==0 and (w/'runs'/job['id']/'completion.json').exists()
                    record({'event':'finish','id':job['id'],'gpu':job['gpu'],'pid':job['process'].pid,
                            'returncode':code,'seconds':time.time()-job['started'],'status':'complete' if valid else 'failed'})
                    active.remove(job)
                    if not valid:
                        state['failures'].append(job['id']);save()
                if len(state['failures'])>=3:raise RuntimeError('Three training failures; stop and inspect saved logs')
                while queue and len(active)<len(gpus)*a.max_per_gpu:
                    gpu=min(gpus,key=lambda value:sum(job['gpu']==value for job in active))
                    run=queue.pop(0)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                    with (w/'logs'/(run+'.log')).open('a') as log:
                        process=subprocess.Popen([sys.executable,str(a.trainer),'--root',str(a.root),
                            '--config',str(w/'configs'/(run+'.json')),'--output',str(w/'runs'/run)],
                            stdout=log,stderr=subprocess.STDOUT,env=env)
                    job={'id':run,'gpu':gpu,'process':process,'started':time.time()};active.append(job)
                    record({'event':'start','id':run,'gpu':gpu,'pid':process.pid})
                if queue or active:time.sleep(1)
        finally:
            for job in active:
                if job['process'].poll() is None:job['process'].terminate()
            for job in active:
                try:job['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:job['process'].kill();job['process'].wait()
        for item in items:
            rows=[]
            for run in item['runs']:
                path=w/'runs'/run/'completion.json'
                if not path.exists():break
                r=json.loads(path.read_text())
                if r['status']!='validation_complete' or r['test_loaded']:raise ValueError(('not a validation-only fit',run))
                if r['config']!=json.loads((w/'configs'/(run+'.json')).read_text()):raise ValueError(('configuration differs',run))
                rows.append(r)
            if len(rows)!=16:item['status']='incomplete';continue
            per_task={task:sum(r['validation_selection_score'] for r in rows if r['config']['task']==task)/4
                      for task in ['cub','grefcoco','nlvr2','construction']}
            item.update(status='complete',task_means=per_task,validation_mean=sum(per_task.values())/4,
                        mean_active_parameters=sum(r['head_trainable_parameters']+r['encoder_active_parameters'] for r in rows)/16,
                        mean_fit_seconds=sum(r['seconds'] for r in rows)/16)
        save()
        with (w/'dse_log.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['id','status','validation_mean','mean_active_parameters','settings'])
            for item in state['candidates']:
                writer.writerow([item['id'],item['status'],item.get('validation_mean'),item.get('mean_active_parameters'),json.dumps(item['settings'],sort_keys=True)])

    if state['selected'] is None:
        base=candidate_for(protocol['base']);execute([base])
        if base['status']!='complete':raise RuntimeError('Initial shared baseline did not finish all 16 pairs')
        state['selected']=base['id'];save()
        print('BASELINE_COMPLETE',json.dumps(base),flush=True)
    for stage in protocol['stages']:
        name=stage['parameter']
        if name in state['completed_stages']:continue
        if time.time()-state['started_at']>=protocol['time_limit_seconds']:
            state['status']='time_limit_before_full_search';save();raise SystemExit(2)
        selected=next(c for c in state['candidates'] if c['id']==state['selected'])
        items=[candidate_for(dict(selected['settings'],**{name:value})) for value in stage['values']]
        execute(items)
        completed=[item for item in items if item['status']=='complete']
        if len(completed)!=len(items):raise RuntimeError(('Incomplete common comparison',name))
        best=max(completed,key=lambda c:(c['validation_mean'],-c['mean_active_parameters'],-int(c['id'].split('_')[-1])))
        state['selected']=best['id'];state['completed_stages'].append(name);save()
        print('STAGE_COMPLETE',name,json.dumps(best),flush=True)
    selected=next(c for c in state['candidates'] if c['id']==state['selected'])
    freeze={'selected':selected,'selection_locked_at':time.time(),'completed_stages':state['completed_stages'],
            'test_loaded_during_selection':False,'shared_settings_tested':len(state['candidates']),
            'individual_fits':sum(len(c['runs']) for c in state['candidates']),
            'boundary_note':'No expansion beyond declared ranges: 288-fit cap and only four already-distilled encoder sizes. This is the best observed coordinate-search setting, not a global optimum.'}
    (w/'selected_config.json').write_text(json.dumps(freeze,indent=2))
    state['status']='search_complete';state['finished_at']=time.time();save()
    print('SELECTION_LOCKED',json.dumps(freeze),flush=True)


if __name__=='__main__':main()
