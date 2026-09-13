from pathlib import Path
import os
from collections import Counter
import json,sys,time
import numpy as np
import torch
ROOT=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
REF=ROOT/'outputs/objective_multiseed_20260912/runs'
OUT=ROOT/'outputs/fixed_score_search_20260913'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing import train_router_formal as f
torch.set_num_threads(1)
grid=[(0.,1.)]+[(a,b) for a in [.25,.5,.75,1.] for b in [1.,.5,0.]]
protocol={'alpha':[0,.25,.5,.75,1],'beta':[1,.5,0],'unique_candidates':13,'epsilon':1e-8,'score':'(p_rescue-p_harm)/(p_rescue+p_harm+2*beta*p_bothwrong+epsilon)^alpha','selection':'Per pair and seed, maximize validation exact-call AUC over 0..100%; ties prefer smaller alpha then beta=1. One-dimensional selection fixes beta=1; two-dimensional selection uses full grid. Test labels only evaluate selected and prespecified reference scores.','fixed':'All original four-state checkpoints selected on validation Score25; no training, no model inference, no VLM calls.','time':time.time()}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
previous=json.loads((ROOT/'outputs/ltd_fixed_backbone_20260913/full_curve_comparison.json').read_text())
result={'protocol':protocol,'pairs':[]}
for original in previous['pairs']:
    task,expert,endpoint=original['task'],original['expert'],original['endpoint']
    pair={k:original[k] for k in ['task','expert','endpoint','small','large']};pair['runs']=[]
    prepared={}
    for split in ['val','test']:
        rows=f.read_rows(ROOT/'outputs/router_full_20260910/data'/expert/split/'records.jsonl');n=len(rows)
        counts=np.floor(n*np.arange(101)/100).astype(int);x=counts/n
        s,l=f.correctness(rows,endpoint)
        if task=='construction':
            events=['rule_1_ppe_violation','rule_2_fall_protection_violation','rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']
            y=np.array([[r['small_label']['events'][e]['target'] for e in events] for r in rows],bool)
            sp=np.array([[r['small_label']['events'][e]['prediction'] is True for e in events] for r in rows]);lp=np.array([[r['large_labels'][endpoint]['events'][e]['prediction'] is True for e in events] for r in rows])
            sc=np.stack([y&sp,~y&sp,y&~sp],axis=1).astype(int);lc=np.stack([y&lp,~y&lp,y&~lp],axis=1).astype(int)
            base=sc.sum(0);delta=lc-sc
        else:base=float(s.mean());delta=(l-s)[:,0]
        prepared[split]=(n,counts,x,base,delta)
    for seed in range(2026,2031):
        dest=REF/f'repeat_{expert}_{endpoint}_four_difference_s{seed}'
        raw=np.load(dest/'paired_logits.npz');selections={};validation=[]
        for split,key in [('val','validation'),('test','test')]:
            p=torch.from_numpy(raw[key]).softmax(1).numpy()
            diff=p[:,1]-p[:,2]
            n,counts,x,base,delta=prepared[split]
            if split=='val':candidates=grid
            else:candidates=list(dict.fromkeys([(0.,1.),(1.,1.),(1.,0.),selections['one_dim'],selections['two_dim']]))
            curves={}
            for alpha,beta in candidates:
                denom=p[:,1].astype(float)+p[:,2]+2*beta*p[:,3]+1e-8
                scores=diff/denom**alpha
                order=np.argsort(-scores,kind='stable')
                if task=='construction':
                    cc=base+np.concatenate([np.zeros((1,3,4),int),np.cumsum(delta[order],axis=0)])[counts];den=2*cc[:,0]+cc[:,1]+cc[:,2]
                    ys=np.divide(2*cc[:,0],den,out=np.zeros_like(den,dtype=float),where=den!=0).mean(1)*100
                else:ys=(base+np.concatenate([[0],np.cumsum(delta[order])])[counts]/n)*100
                entry={'alpha':alpha,'beta':beta}
                for limit in [25,100]:entry[f'area{limit}']=float(np.sum((ys[1:limit+1]+ys[:limit])*.5*np.diff(x[:limit+1]))/x[limit])
                if split=='val':validation.append(entry)
                else:curves[(alpha,beta)]={**entry,'calls':(x*100).tolist(),'curve':ys.tolist()}
            if split=='val':
                for name,subset in [('one_dim',[r for r in validation if r['beta']==1]),('two_dim',validation)]:
                    chosen=min(subset,key=lambda r:(-r['area100'],r['alpha'],-r['beta']))
                    selections[name]=(chosen['alpha'],chosen['beta'])
            else:
                baseline=next(r for r in original['runs'] if r['method']=='learned' and r['seed']==seed)
                assert np.allclose(curves[(0.,1.)]['curve'],baseline['exact_score'],atol=1e-10)
                methods={'original':(0.,1.),'one_dim':selections['one_dim'],'two_dim':selections['two_dim'],'error_ratio':(1.,1.),'rescue_harm_ratio':(1.,0.)}
                pair['runs'].append({'seed':seed,'validation_candidates':validation,'test':{m:curves[ab] for m,ab in methods.items()}})
    result['pairs'].append(pair)
    print('DONE',expert,endpoint,flush=True)
(OUT/'results.json').write_text(json.dumps(result))
(OUT/'complete.json').write_text(json.dumps({'pairs':16,'checkpoints':80,'validation_candidates_per_checkpoint':13,'training_runs':0,'time':time.time()},indent=2))
for task in ['cub','grefcoco','nlvr2','construction']:
    pp=[p for p in result['pairs'] if p['task']==task]
    print(task,{m:{k:float(np.mean([r['test'][m][k] for p in pp for r in p['runs']])) for k in ['area25','area100']} for m in ['original','one_dim','two_dim']})
