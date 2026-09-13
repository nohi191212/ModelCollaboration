from pathlib import Path
import os
import json,sys
import numpy as np
import torch
ROOT=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
OUT=ROOT/'outputs/ltd_fixed_backbone_20260913'
REF=ROOT/'outputs/objective_multiseed_20260912/runs'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing import train_router_formal as f
torch.set_num_threads(1)
previous=json.loads((OUT/'full_curve_comparison.json').read_text())
results=[]
for pair in previous['pairs']:
    task,expert,endpoint=pair['task'],pair['expert'],pair['endpoint']
    rows=f.read_rows(ROOT/'outputs/router_full_20260910/data'/expert/'test/records.jsonl')
    n=len(rows);counts=np.floor(n*np.arange(101)/100).astype(int);x=counts/n
    s,l=f.correctness(rows,endpoint)
    if task=='construction':
        events=['rule_1_ppe_violation','rule_2_fall_protection_violation','rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']
        y=np.array([[r['small_label']['events'][e]['target'] for e in events] for r in rows],bool)
        sp=np.array([[r['small_label']['events'][e]['prediction'] is True for e in events] for r in rows]);lp=np.array([[r['large_labels'][endpoint]['events'][e]['prediction'] is True for e in events] for r in rows])
        sc=np.stack([y&sp,~y&sp,y&~sp],axis=1).astype(int);lc=np.stack([y&lp,~y&lp,y&~lp],axis=1).astype(int)
        base=sc.sum(0);d=lc-sc
    for seed in range(2026,2031):
        raw=torch.from_numpy(np.load(REF/f'repeat_{expert}_{endpoint}_four_difference_s{seed}'/'paired_logits.npz')['test'])
        p=raw.softmax(1)
        scores={'difference':(p[:,1]-p[:,2]).numpy(),'marginal_error_ratio':(torch.logsumexp(raw[:,[1,3]],1)-torch.logsumexp(raw[:,[2,3]],1)).numpy(),'rescue_harm_ratio':(raw[:,1]-raw[:,2]).numpy()}
        for method,score in scores.items():
            order=np.argsort(-score,kind='stable')
            if task=='construction':
                cc=base+np.concatenate([np.zeros((1,3,4),int),np.cumsum(d[order],axis=0)])[counts];den=2*cc[:,0]+cc[:,1]+cc[:,2]
                yy=np.divide(2*cc[:,0],den,out=np.zeros_like(den,dtype=float),where=den!=0).mean(1)*100
            else:yy=(s.mean()+np.concatenate([[0],np.cumsum((l-s)[order,0])])[counts]/n)*100
            entry={'task':task,'expert':expert,'endpoint':endpoint,'seed':seed,'method':method,'curve':yy.tolist()}
            for lo,hi,name in [(0,25,'low'),(25,100,'mid_high'),(0,100,'full')]:
                entry[name]=float(np.sum((yy[lo+1:hi+1]+yy[lo:hi])*.5*np.diff(x[lo:hi+1]))/(x[hi]-x[lo]))
            if method=='difference':
                prior=next(r for r in pair['runs'] if r['method']=='learned' and r['seed']==seed)
                assert np.allclose(yy,prior['exact_score'],atol=1e-10)
            results.append(entry)
        prior=next(r for r in pair['runs'] if r['method']=='posthoc' and r['cost']==0 and r['seed']==seed)
        yy=np.array(prior['exact_score']);entry={'task':task,'expert':expert,'endpoint':endpoint,'seed':seed,'method':'posthoc_c0','curve':yy.tolist()}
        for lo,hi,name in [(0,25,'low'),(25,100,'mid_high'),(0,100,'full')]:
            entry[name]=float(np.sum((yy[lo+1:hi+1]+yy[lo:hi])*.5*np.diff(x[lo:hi+1]))/(x[hi]-x[lo]))
        results.append(entry)
summary=[]
for task in ['cub','grefcoco','nlvr2','construction']:
    for method in ['difference','marginal_error_ratio','rescue_harm_ratio','posthoc_c0']:
        rr=[r for r in results if (r['task'],r['method'])==(task,method)]
        summary.append({'task':task,'method':method,**{k:float(np.mean([r[k] for r in rr])) for k in ['low','mid_high','full']}})
(OUT/'score_geometry_audit.json').write_text(json.dumps({'protocol':'Same four-state checkpoints, same test inputs, score transformations only; no test-based checkpoint selection, no new training. Posthoc c0 shown separately to match cost zero ratio derivation.','summary':summary,'runs':results}))
print(json.dumps(summary,indent=2))
