"""Fixed-model, cross-fitted histogram calibration; no model training/inference."""
from pathlib import Path
import os
import json, sys, time
import numpy as np
import torch

ROOT=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
OUT=ROOT/'outputs/score_information_20260913'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing import train_router_formal as f
torch.set_num_threads(1)
EVENTS=['rule_1_ppe_violation','rule_2_fall_protection_violation','rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']
FAMILIES=['score_only','confidence','category','score_confidence','score_category']
WEIGHTS=[0.,.25,.5,1.]
protocol={
    'time':time.time(),'fixed_models':80,'new_training_runs':0,'new_inference_calls':0,
    'baseline':'Previous validation-selected alpha, beta=1; checkpoints unchanged.',
    'update':'p_new = normalize(max(p + weight * mean_shrunk(Y-p | group), 1e-8)); then original normalized score with fixed alpha.',
    'families':FAMILIES,'weights':WEIGHTS,'shrinkage_pseudocount':20,
    'groups':{'score_only':'10 quantile bins of baseline score','confidence':'10 quantile bins of max_probability or max_score; missing separate', 'category':'predicted class (classification), predicted box count capped at 3 (grounding), predicted event bitmask (construction)', 'score_confidence':'5 score bins x 5 confidence bins','score_category':'5 score bins x predicted category'},
    'selection':'5-fold group cross-fitting within validation. Each fold correction and bin edges estimated without its labels/features. Select weight per family and an overall candidate by validation full-curve AUC. Ties: weight=0, then lower weight, then family order. Reestimate correction tables on all validation, evaluate selected candidates on test. Test never selects weights or families.',
    'grouping':'Shared image_path kept in same fold; otherwise image1; otherwise sample_id. Fixed RNG seed 913, no target stratification.',
    'caveat':'Validation already used for checkpoint and alpha selection; cross-fitting removes direct same-row calibration leakage but is not fully nested model selection. Test previously inspected: exploratory result, not blind holdout.',
    'information':'Confidence and output were already router inputs; they are extra information relative to compressed four-state output, not new raw model observations.',
    'construction':'Four-state target is event-averaged distribution. Selection/evaluation uses actual macro-F1; no claim of additive gain optimality.',
}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
prior=json.loads((ROOT/'outputs/fixed_score_search_20260913/results.json').read_text())
result={'protocol':protocol,'pairs':[]}

def curve(score, prepared):
    n,counts,x,base,delta,construction=prepared
    order=np.argsort(-score,kind='stable')
    if construction:
        c=base+np.concatenate([np.zeros((1,3,4),int),np.cumsum(delta[order],axis=0)])[counts]
        den=2*c[:,0]+c[:,1]+c[:,2]
        ys=np.divide(2*c[:,0],den,out=np.zeros_like(den,dtype=float),where=den!=0).mean(1)*100
    else:
        ys=(base+np.concatenate([[0],np.cumsum(delta[order])])[counts]/n)*100
    out={'calls':(x*100).tolist(),'curve':ys.tolist()}
    for limit in [25,100]:
        out[f'area{limit}']=float(np.sum((ys[1:limit+1]+ys[:limit])*.5*np.diff(x[:limit+1]))/x[limit])
    out['area25_100']=float(np.sum((ys[26:]+ys[25:-1])*.5*np.diff(x[25:]))/(x[100]-x[25]))
    return out

def scores(p,alpha):
    return (p[:,1]-p[:,2])/(p[:,1].astype(float)+p[:,2]+2*p[:,3]+1e-8)**alpha

def quantile_bins(train,query,k):
    good=np.isfinite(train)
    assert good.any()
    edges=np.unique(np.quantile(train[good],np.arange(1,k)/k))
    a=np.searchsorted(edges,train,side='right');b=np.searchsorted(edges,query,side='right')
    a[~good]=k;b[~np.isfinite(query)]=k
    return a,b

def corrections(family, s_train, s_query, c_train, c_query, g_train, g_query, residual):
    k=5 if family.startswith('score_') and family!='score_only' else 10
    if family in ['score_only','score_confidence','score_category']:
        st,sq=quantile_bins(s_train,s_query,k)
    if family in ['confidence','score_confidence']:
        ct,cq=quantile_bins(c_train,c_query,k)
    if family=='score_only':a,b=st,sq
    elif family=='confidence':a,b=ct,cq
    elif family=='category':a,b=g_train,g_query
    elif family=='score_confidence':a,b=st*(k+1)+ct,sq*(k+1)+cq
    else:a,b=st*1000+g_train,sq*1000+g_query
    keys,inverse=np.unique(a,return_inverse=True)
    sums=np.zeros((len(keys),4));np.add.at(sums,inverse,residual)
    count=np.bincount(inverse,minlength=len(keys))
    means=sums/(count[:,None]+20)
    # Unseen groups explicitly fall back to unchanged probabilities.
    lookup={int(key):i for i,key in enumerate(keys)}
    pred=np.zeros((len(b),4));seen=np.array([int(key) in lookup for key in b])
    pred[seen]=means[[lookup[int(key)] for key in b[seen]]]
    return pred,{'unseen_fraction':float((~seen).mean()),'groups':len(keys),'median_group_count':float(np.median(count))}

def adjusted(p,change,weight):
    if weight==0:return p
    q=np.maximum(p+weight*change,1e-8)
    return q/q.sum(1,keepdims=True)

for old in prior['pairs']:
    task,expert,endpoint=old['task'],old['expert'],old['endpoint']
    pair={k:old[k] for k in ['task','expert','endpoint','small','large']};pair['runs']=[]
    prepared={};inputs={};targets={};rows_by_split={}
    columns=json.loads((ROOT/'outputs/router_training_cache_complete_20260908'/expert/'train/confidence_columns.json').read_text())
    conf_name='max_probability' if task in ['cub','nlvr2'] else 'max_score'
    col=columns.index(conf_name)
    for split in ['val','test']:
        d=ROOT/'outputs/router_full_20260910/data'/expert/split
        rows=f.read_rows(d/'records.jsonl');rows_by_split[split]=rows
        n=len(rows);counts=np.floor(n*np.arange(101)/100).astype(int);x=counts/n
        conf=np.load(d/'confidence.npy');confidence=conf[:,col].astype(float)
        confidence[conf[:,col+len(columns)]==0]=np.nan
        if task=='cub':group=np.array([r['small_label']['prediction'] if r['small_label']['valid_output'] else 999 for r in rows],int)
        elif task=='nlvr2':group=np.array([int(r['small_label']['prediction']=='True') if r['small_label']['valid_output'] else 999 for r in rows])
        elif task=='grefcoco':group=np.array([min(r['small_label']['predicted_boxes'],3) if r['small_label']['valid_output'] else 999 for r in rows])
        else:group=np.array([sum((1<<i)*int(r['small_label']['events'][e]['prediction'] is True) for i,e in enumerate(EVENTS)) for r in rows])
        inputs[split]=(confidence,group)
        s,l=f.correctness(rows,endpoint)
        targets[split]=np.stack([((s==1)&(l==1)).mean(1),((s==0)&(l==1)).mean(1),((s==1)&(l==0)).mean(1),((s==0)&(l==0)).mean(1)],axis=1)
        if task=='construction':
            y=np.array([[r['small_label']['events'][e]['target'] for e in EVENTS] for r in rows],bool)
            sp=np.array([[r['small_label']['events'][e]['prediction'] is True for e in EVENTS] for r in rows]);lp=np.array([[r['large_labels'][endpoint]['events'][e]['prediction'] is True for e in EVENTS] for r in rows])
            sc=np.stack([y&sp,~y&sp,y&~sp],axis=1).astype(int);lc=np.stack([y&lp,~y&lp,y&~lp],axis=1).astype(int)
            base=sc.sum(0);delta=lc-sc
        else:base=float(s.mean());delta=(l-s)[:,0]
        prepared[split]=(n,counts,x,base,delta,task=='construction')
    # Grouping uses only identifiers, not labels.
    groups=[str(r['image_path'] if 'image_path' in r else r['image1'] if 'image1' in r else r['sample_id']) for r in rows_by_split['val']]
    unique=np.unique(groups);np.random.default_rng(913).shuffle(unique)
    mapping={key:i%5 for i,key in enumerate(unique)};folds=np.array([mapping[g] for g in groups])
    pair['input_audit']={'confidence':conf_name,'validation_rows':len(folds),'validation_image_groups':len(unique),'confidence_already_router_input':True,'category_already_router_input':True}
    cv,gv=inputs['val'];ct,gt=inputs['test']
    for oldrun in old['runs']:
        seed=oldrun['seed'];alpha=oldrun['test']['one_dim']['alpha']
        raw=np.load(ROOT/'outputs/objective_multiseed_20260912/runs'/f'repeat_{expert}_{endpoint}_four_difference_s{seed}'/'paired_logits.npz')
        pv=torch.from_numpy(raw['validation']).softmax(1).numpy();pt=torch.from_numpy(raw['test']).softmax(1).numpy()
        sv,st=scores(pv,alpha),scores(pt,alpha)
        rv=targets['val']-pv
        baseline=curve(st,prepared['test'])
        assert np.allclose(baseline['curve'],oldrun['test']['one_dim']['curve'],atol=1e-10)
        validation=[];test={'baseline':baseline,'original':oldrun['test']['original']};selections={};diagnostics={}
        for family in FAMILIES:
            oof=np.zeros_like(pv,dtype=float)
            for fold in range(5):
                held=folds==fold;train=~held
                oof[held],_=corrections(family,sv[train],sv[held],cv[train],cv[held],gv[train],gv[held],rv[train])
            candidates=[]
            for weight in WEIGHTS:
                q=adjusted(pv,oof,weight);m=curve(scores(q,alpha),prepared['val'])
                entry={'family':family,'weight':weight,'area100':m['area100'],'area25':m['area25'],'brier':float(np.mean(np.sum((q-targets['val'])**2,axis=1)))}
                candidates.append(entry);validation.append(entry)
            chosen=min(candidates,key=lambda z:(-z['area100'],z['weight']))
            selections[family]=chosen
            full,audit=corrections(family,sv,st,cv,ct,gv,gt,rv)
            q=adjusted(pt,full,chosen['weight'])
            test[family]={**curve(scores(q,alpha),prepared['test']),'weight':chosen['weight'],'brier':float(np.mean(np.sum((q-targets['test'])**2,axis=1)))}
            diagnostics[family]=audit
        chosen=min(validation,key=lambda z:(-z['area100'],z['weight'],FAMILIES.index(z['family'])))
        test['selected']={**test[chosen['family']],'family':chosen['family']}
        pair['runs'].append({'seed':seed,'alpha':alpha,'validation_candidates':validation,'selections':selections,'overall_selection':chosen,'test':test,'diagnostics':diagnostics})
    result['pairs'].append(pair)
    print('DONE',expert,endpoint,{m:round(float(np.mean([r['test'][m]['area100']-r['test']['baseline']['area100'] for r in pair['runs']])),4) for m in FAMILIES+['selected']},flush=True)
    (OUT/'results.json').write_text(json.dumps(result))
(OUT/'complete.json').write_text(json.dumps({'pairs':len(result['pairs']),'checkpoints':sum(len(p['runs']) for p in result['pairs']),'training_runs':0,'inference_calls':0,'time':time.time()},indent=2))
