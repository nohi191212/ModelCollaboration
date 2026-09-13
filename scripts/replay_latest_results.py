"""Recompute Tables 1 and 3 from released per-example outputs; no GPU or training."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from replay_five_seed_results import prepare_curve,point_at_count,validation_thresholds,fixed_curve

parser=argparse.ArgumentParser()
parser.add_argument('--artifacts',type=Path,required=True)
parser.add_argument('--paper-data',type=Path,default=Path(__file__).resolve().parents[1]/'paper/data')
parser.add_argument('--output',type=Path,default=Path('verification_latest.json'))
args=parser.parse_args();torch.set_num_threads(2)
root=args.artifacts;cache={};main=defaultdict(list);areas=defaultdict(list);curve_points=0

def outcomes(c,split):
    key=(c['expert'],c['endpoint'],split)
    if key not in cache:
        with np.load(root/'pairs'/('shared_12_'+c['expert']+'_'+c['endpoint'])/(split+'.npz')) as d:cache[key]={k:d[k] for k in d.files}
    return cache[key]

def prepared(score,c,split):
    d=outcomes(c,split)
    return prepare_curve(score,d['small'],d['large'],d['small_correct'],d['large_correct'])

def exact(score,c,split):
    p=prepared(score,c,split);count=np.floor(p['n']*np.arange(101)/100).astype(int)
    ys=np.array([point_at_count(p,int(k),k/p['n'])['metric'] for k in count])*100
    return float(np.trapezoid(ys,x=count/p['n']))

def normalized(raw,alpha):
    p=torch.from_numpy(raw).softmax(1).numpy()
    return (p[:,1]-p[:,2])/(p[:,1].astype(float)+p[:,2]+2*p[:,3]+1e-8)**alpha

for kind in ['main','four_difference','gain','three','posthoc']:
    runs=sorted((root/'paper_replay_runs'/kind).iterdir());assert len(runs)==80,(kind,len(runs))
    for run in runs:
        c=json.loads((run/'config.json').read_text())
        with np.load(run/'paired_logits.npz') as data:raw={k:data[k] for k in data.files}
        if kind=='main':
            record=json.loads((run/'evaluation.json').read_text())['methods']['normalized']
            alpha=json.loads((run/'selection.json').read_text())['alpha']
            scores={s:normalized(raw[k],alpha) for s,k in [('val','validation'),('test','test')]}
            threshold=validation_thresholds(prepared(scores['val'],c,'val'))
            curves={s:fixed_curve(prepared(scores[s],c,s),threshold) for s in ['val','test']}
            for split in curves:
                for i,point in enumerate(curves[split]):
                    assert abs(point['metric']*100-record[split]['score'][i])<1e-9,(run.name,split,i,'score')
                    assert abs(point['actual_fraction']*100-record[split]['calls'][i])<1e-9,(run.name,split,i,'calls')
                    curve_points+=1
            selected=record['selected_budget'];best=max(p['metric'] for p in curves['val'])
            assert abs(curves['val'][selected]['metric']-best)<1e-12
            main[(c['expert'],c['endpoint'])].append(curves['test'][selected])
        elif kind=='four_difference':
            alpha=max([0,.25,.5,.75,1],key=lambda a:(exact(normalized(raw['validation'],a),c,'val'),-a))
            z=torch.from_numpy(raw['test']);p=z.softmax(1).numpy()
            scores={'4S_difference':p[:,1]-p[:,2],'4S_RH':raw['test'][:,1]-raw['test'][:,2],
                    '4S_error_ratio':(torch.logsumexp(z[:,[1,3]],1)-torch.logsumexp(z[:,[2,3]],1)).numpy(),
                    '4S_normalized':normalized(raw['test'],alpha)}
            for method,score in scores.items():areas[(method,c['task'])].append(exact(score,c,'test'))
        else:
            z=raw['test']
            if kind=='three':
                p=torch.from_numpy(z).softmax(1).numpy();score=p[:,0]-p[:,1]
            else:score=z[:,0]
            areas[(kind,c['task'])].append(exact(score,c,'test'))
    print('Replayed',kind,len(runs),flush=True)
expected=json.loads((args.paper_data/'results.json').read_text())['groups']
for g in expected:
    key=tuple(g['key'][1:]);rr=main[key];assert len(rr)==5
    for field in ['metric','actual_fraction']:
        assert abs(np.mean([r[field] for r in rr])-g['methods']['learned']['test_point'][field])<1e-10,(key,field)
    c={'expert':key[0],'endpoint':key[1]};scores={s:np.where(outcomes(c,s)['confidence_valid'],-outcomes(c,s)['confidence'],1.) for s in ['val','test']}
    thresholds=validation_thresholds(prepared(scores['val'],c,'val'))
    for split,tag in [('val','validation'),('test','test')]:
        curve=fixed_curve(prepared(scores[split],c,split),thresholds)
        for a,b in zip(curve,g['methods']['confidence'][tag+'_curve']):
            assert abs(a['metric']-b['metric'])<1e-10,(key,tag,'confidence metric')
            assert abs(a['actual_fraction']-b['actual_fraction'])<1e-10
            curve_points+=1
summary=json.loads((args.paper_data/'lr_control_summary.json').read_text());table={}
for (method,task),values in areas.items():
    assert len(values)==20,(method,task,len(values))
    value=float(np.mean(values));assert abs(value-summary[method][task]['full'])<1e-8,(method,task,value,summary[method][task]['full'])
    table.setdefault(method,{})[task]=value
report={'status':'passed','main_runs':80,'table3_trained_runs':320,'main_and_confidence_curve_points':curve_points,'table3':table,'main_pairs':16}
args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
