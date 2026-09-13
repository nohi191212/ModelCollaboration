from pathlib import Path
import os
import json,sys
import numpy as np
import torch
ROOT=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
OUT=ROOT/'outputs/three_state_20260913'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing import train_router_formal as f
torch.set_num_threads(1)
reuse=json.loads((OUT/'reused_runs.json').read_text())
previous=json.loads((ROOT/'outputs/ltd_fixed_backbone_20260913/full_curve_comparison.json').read_text())
result={'protocol':'Three methods share original inputs, optimization, natural sampling and low-budget validation checkpoint selection. Three-state CE uses rescue/harm/unchanged probabilities; construction uses event proportions. Gain reuses MSE of event-mean signed gain. Exact calls at 0..100 percent; original archived points checked; no test-based model selection.','pairs':[]}
for old in previous['pairs']:
    task,expert,endpoint=old['task'],old['expert'],old['endpoint']
    rows=f.read_rows(ROOT/'outputs/router_full_20260910/data'/expert/'test/records.jsonl')
    n=len(rows);counts=np.floor(n*np.arange(101)/100).astype(int)
    small,large=f.correctness(rows,endpoint)
    if task=='construction':
        events=['rule_1_ppe_violation','rule_2_fall_protection_violation','rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']
        y=np.array([[r['small_label']['events'][e]['target'] for e in events] for r in rows],bool)
        sp=np.array([[r['small_label']['events'][e]['prediction'] is True for e in events] for r in rows])
        lp=np.array([[r['large_labels'][endpoint]['events'][e]['prediction'] is True for e in events] for r in rows])
        sc=np.stack([y&sp,~y&sp,y&~sp],axis=1).astype(int)
        lc=np.stack([y&lp,~y&lp,y&~lp],axis=1).astype(int)
        base=sc.sum(0);delta=lc-sc
    pair={k:old[k] for k in ['task','expert','endpoint','n','small','large']};pair['runs']=[]
    for seed in range(2026,2031):
        item=next(r for r in reuse if (r['expert'],r['endpoint'],r['seed'])==(expert,endpoint,seed))
        for method,dest in [('learned',Path(item['learned'])),('gain',Path(item['gain'])),('three',OUT/'runs'/f'{expert}_{endpoint}_three_s{seed}')]:
            if not (dest/'test_evaluation.json').exists():continue
            cfg=json.loads((dest/'config.json').read_text());key=cfg['ranking']
            ev=json.loads((dest/'test_evaluation.json').read_text())['comparisons'][key]
            if method=='learned':
                record=dict(next(r for r in old['runs'] if r['method']=='learned' and r['seed']==seed))
            else:
                raw=np.load(dest/'paired_logits.npz')['test']
                if method=='three':
                    assert raw.shape==(n,3)
                    prob=torch.from_numpy(raw).softmax(1).numpy();score=prob[:,0]-prob[:,1]
                else:
                    assert raw.shape==(n,1)
                    score=raw[:,0]
                order=np.argsort(-score,kind='stable')
                if task=='construction':
                    conf=base+np.concatenate([np.zeros((1,3,4),int),np.cumsum(delta[order],axis=0)])[counts]
                    den=2*conf[:,0]+conf[:,1]+conf[:,2]
                    metrics=np.divide(2*conf[:,0],den,out=np.zeros_like(den,dtype=float),where=den!=0).mean(1)
                else:
                    gain=(large-small)[:,0]
                    metrics=small.mean()+np.concatenate([[0],np.cumsum(gain[order])])[counts]/n
                for point in ev['test_exact_budget_diagnostic']:
                    assert abs(metrics[round(point['budget']*100)]-point['metric'])<1e-10,(dest,point)
                record={'seed':seed,'method':method,'exact_calls':(counts/n*100).tolist(),'exact_score':(metrics*100).tolist(),'transferred_calls':[p['actual_fraction']*100 for p in ev['test_curve']],'transferred_score':[p['metric']*100 for p in ev['test_curve']]}
            record['selected']={'calls':ev['test_selected']['actual_fraction']*100,'score':ev['test_selected']['metric']*100}
            pair['runs'].append(record)
    result['pairs'].append(pair)
    print('EXPORTED',expert,endpoint,len(pair['runs']),flush=True)
result['three_completed']=sum(r['method']=='three' for p in result['pairs'] for r in p['runs'])
(OUT/'curve_comparison.json').write_text(json.dumps(result))
print('COMPLETE',result['three_completed'])
