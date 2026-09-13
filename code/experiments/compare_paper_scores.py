from pathlib import Path
import os
import json,sys
import numpy as np
import torch
R=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
O=R/'outputs/paper_score_comparison_20260913';O.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing import train_router_formal as f
torch.set_num_threads(1);f.BUDGETS=[i/100 for i in range(101)]
old=json.loads((R/'outputs/fixed_score_search_20260913/results.json').read_text())
events=['rule_1_ppe_violation','rule_2_fall_protection_violation','rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']
result={'protocol':{'methods':['difference','normalized','rescue_harm_ratio','error_ratio'],'alpha':'Previous validation-only full-AUC selection, fixed beta=1','thresholds':'Original validation threshold/tie rule; selected policy maximizes validation metric over budgets 1..99%, ties fewer calls then lower budget','training':0},'pairs':[]}
for pair in old['pairs']:
    expert,endpoint,task=pair['expert'],pair['endpoint'],pair['task']
    out={k:pair[k] for k in ['expert','endpoint','task','small','large']};out['runs']=[];prepared={};rows={}
    for split in ['val','test']:
        rows[split]=f.read_rows(R/'outputs/router_full_20260910/data'/expert/split/'records.jsonl')
        rr=rows[split];s,l=f.correctness(rr,endpoint);n=len(rr)
        if task=='construction':
            y=np.array([[r['small_label']['events'][e]['target'] for e in events] for r in rr],bool)
            sp=np.array([[r['small_label']['events'][e]['prediction'] is True for e in events] for r in rr]);lp=np.array([[r['large_labels'][endpoint]['events'][e]['prediction'] is True for e in events] for r in rr])
            sc=np.stack([y&sp,~y&sp,y&~sp],axis=1).astype(int);lc=np.stack([y&lp,~y&lp,y&~lp],axis=1).astype(int)
            base=sc.sum(0);delta=lc-sc
        else:base=s.mean();delta=(l-s)[:,0]
        prepared[split]=(n,base,delta)
    for oldrun in pair['runs']:
        seed=oldrun['seed'];alpha=oldrun['test']['one_dim']['alpha']
        dest=R/'outputs/objective_multiseed_20260912/runs'/f'repeat_{expert}_{endpoint}_four_difference_s{seed}'
        raw=np.load(dest/'paired_logits.npz');ss={}
        for split,key in [('val','validation'),('test','test')]:
            z=torch.from_numpy(raw[key]);p=z.softmax(1).numpy();diff=p[:,1]-p[:,2]
            ss[split]={'difference':diff,'normalized':diff/(p[:,1].astype(float)+p[:,2]+2*p[:,3]+1e-8)**alpha,'rescue_harm_ratio':raw[key][:,1]-raw[key][:,2],'error_ratio':(torch.logsumexp(z[:,[1,3]],1)-torch.logsumexp(z[:,[2,3]],1)).numpy()}
        run={'seed':seed,'alpha':alpha,'methods':{}}
        for method in ss['val']:
            thresholds=f.validation_thresholds(rows['val'],endpoint,ss['val'][method]);entry={}
            for split in ['val','test']:
                n,base,delta=prepared[split];score=ss[split][method];order=np.argsort(-score,kind='stable');sorted_score=score[order]
                if task=='construction':
                    cc=base+np.concatenate([np.zeros((1,3,4),int),np.cumsum(delta[order],axis=0)])
                    den=2*cc[:,0]+cc[:,1]+cc[:,2];metric=np.divide(2*cc[:,0],den,out=np.zeros_like(den,dtype=float),where=den!=0).mean(1)
                else:metric=base+np.concatenate([[0],np.cumsum(delta[order])])/n
                counts=np.array([0 if t['operator']=='never' else n if t['operator']=='always' else np.searchsorted(-sorted_score,-t['threshold'],side='right' if t['operator']=='ge' else 'left') for t in thresholds])
                exact=np.floor(n*np.arange(101)/100).astype(int)
                yy=metric[counts];ex=metric[exact]
                entry[split]={'score':(yy*100).tolist(),'calls':(counts/n*100).tolist(),'exact_score':(ex*100).tolist(),'exact_calls':(exact/n*100).tolist(),'threshold_area25':float(np.trapz(yy[np.arange(0,26,5)],dx=.05)/.25*100),'exact_area100':float(np.trapz(ex,x=exact/n)*100)}
            best=min(range(1,100),key=lambda i:(-entry['val']['score'][i],entry['val']['calls'][i],i))
            entry['selected_budget']=best;entry['selected_threshold']=thresholds[best]
            entry['selected']={split:{'score':entry[split]['score'][best],'calls':entry[split]['calls'][best]} for split in ['val','test']}
            if method=='difference':
                history=json.loads((dest/'test_evaluation.json').read_text())['comparisons']['probability_difference']
                assert np.isclose(entry['selected']['test']['score'],history['test_selected']['metric']*100)
                assert np.isclose(entry['selected']['test']['calls'],history['test_selected']['actual_fraction']*100)
                assert np.allclose(entry['test']['score'],[v['metric']*100 for v in history['test_curve']])
            run['methods'][method]=entry
        out['runs'].append(run)
    result['pairs'].append(out)
    print('DONE',expert,endpoint,flush=True)
(O/'results.json').write_text(json.dumps(result))
for task in ['cub','grefcoco','nlvr2','construction']:
    pp=[p for p in result['pairs'] if p['task']==task]
    print(task,{m:{k:round(float(np.mean([r['methods'][m]['test'][k] for p in pp for r in p['runs']])),3) for k in ['threshold_area25','exact_area100']} for m in ['difference','normalized','rescue_harm_ratio','error_ratio']})
