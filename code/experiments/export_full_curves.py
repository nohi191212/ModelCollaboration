from pathlib import Path
import os
from collections import defaultdict
import json
import sys
import numpy as np
import torch
ROOT=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
OUT=ROOT/'outputs/ltd_fixed_backbone_20260913'
REF=ROOT/'outputs/objective_multiseed_20260912/runs'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing import train_router_formal as f
torch.set_num_threads(1)
table=json.loads((OUT/'table1_posthoc.json').read_text())
chosen={(r['expert'],r['endpoint'],r['seed']):r['call_cost'] for r in table['selected_runs']}
plan=[c for c in json.loads((OUT/'job_plan.json').read_text()) if c['ltd_method']=='posthoc']
pairgroups=defaultdict(list)
for cfg in plan:pairgroups[(cfg['task'],cfg['expert'],cfg['endpoint'])].append(cfg)
output={'protocol':'Cached predictions only. Exact-call curves use stable descending ranking at floor(N*b), b=0..1 by .01; test labels only score, not select. Post-hoc main curve chooses cost per seed by original r>0 validation policy as in Table1, then sweeps that frozen score. All 5 costs separately retained. Native zero policies are separate points. Existing checkpoints remain selected by validation low-budget Score25.','pairs':[]}
for (task,expert,endpoint),configs in pairgroups.items():
    rows=f.read_rows(ROOT/'outputs/router_full_20260910/data'/expert/'test/records.jsonl')
    n=len(rows);counts=np.floor(n*np.arange(101)/100).astype(int)
    small,large=f.correctness(rows,endpoint)
    if task=='construction':
        events=['rule_1_ppe_violation','rule_2_fall_protection_violation','rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']
        y=np.array([[r['small_label']['events'][e]['target'] for e in events] for r in rows],bool)
        smallpred=np.array([[r['small_label']['events'][e]['prediction'] is True for e in events] for r in rows])
        largepred=np.array([[r['large_labels'][endpoint]['events'][e]['prediction'] is True for e in events] for r in rows])
        smallconf=np.stack([y&smallpred,~y&smallpred,y&~smallpred],axis=1).astype(int)
        largeconf=np.stack([y&largepred,~y&largepred,y&~largepred],axis=1).astype(int)
        base=smallconf.sum(0);delta=largeconf-smallconf
    pair={'task':task,'expert':expert,'endpoint':endpoint,'n':n,'small':f.metric(rows,endpoint,np.zeros(n,bool))*100,'large':f.metric(rows,endpoint,np.ones(n,bool))*100,'runs':[]}
    for seed in range(2026,2031):
        jobs=[('learned',None,REF/f'repeat_{expert}_{endpoint}_four_difference_s{seed}')]+[('posthoc',c['call_cost'],OUT/'runs'/c['id']) for c in configs if c['seed']==seed]
        for method,cost,dest in jobs:
            raw=np.load(dest/'paired_logits.npz')['test']
            if method=='learned':
                prob=torch.from_numpy(raw).softmax(1).numpy();score=prob[:,1]-prob[:,2];key='probability_difference'
            else:score=raw[:,0];key='ltd_score'
            order=np.argsort(-score,kind='stable')
            if task=='construction':
                conf=base+np.concatenate([np.zeros((1,3,4),int),np.cumsum(delta[order],axis=0)])[counts]
                den=2*conf[:,0]+conf[:,1]+conf[:,2]
                metrics=np.divide(2*conf[:,0],den,out=np.zeros_like(den,dtype=float),where=den!=0).mean(1)
            else:
                gain=(large-small)[:,0]
                metrics=small.mean()+np.concatenate([[0],np.cumsum(gain[order])])[counts]/n
            ev=json.loads((dest/'test_evaluation.json').read_text())['comparisons'][key]
            for point in ev['test_exact_budget_diagnostic']:
                assert abs(metrics[round(point['budget']*100)]-point['metric'])<1e-10,(dest,point)
            record={'seed':seed,'method':method,'cost':cost,'chosen_cost':cost==chosen[(expert,endpoint,seed)] if method=='posthoc' else True,'exact_calls':(counts/n*100).tolist(),'exact_score':(metrics*100).tolist(),'transferred_calls':[p['actual_fraction']*100 for p in ev['test_curve']],'transferred_score':[p['metric']*100 for p in ev['test_curve']]}
            if method=='posthoc':
                native=next(r for r in table['all_candidates'] if r['id']==dest.name)
                record['native']=native['test']
            pair['runs'].append(record)
    output['pairs'].append(pair)
    print('EXPORTED',task,expert,endpoint,flush=True)
(OUT/'full_curve_comparison.json').write_text(json.dumps(output))
print('COMPLETE',len(output['pairs']),sum(len(p['runs']) for p in output['pairs']))
