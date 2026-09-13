"""Uniform full-budget validation selection; frozen encoders, unchanged paired CE."""
from pathlib import Path
import os
import argparse,json,sys,time
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing import train_router_formal as f
ALPHAS=[0.,.25,.5,.75,1.]
EVENTS=['rule_1_ppe_violation','rule_2_fall_protection_violation','rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']

def prepare(rows,endpoint):
    s,l=f.correctness(rows,endpoint);n=len(rows)
    if 'events' in rows[0]['small_label']:
        y=np.array([[r['small_label']['events'][e]['target'] for e in EVENTS] for r in rows],bool)
        sp=np.array([[r['small_label']['events'][e]['prediction'] is True for e in EVENTS] for r in rows]);lp=np.array([[r['large_labels'][endpoint]['events'][e]['prediction'] is True for e in EVENTS] for r in rows])
        sc=np.stack([y&sp,~y&sp,y&~sp],axis=1).astype(int);lc=np.stack([y&lp,~y&lp,y&~lp],axis=1).astype(int)
        base=sc.sum(0);delta=lc-sc
    else:base=s.mean();delta=(l-s)[:,0]
    return {'n':n,'base':base,'delta':delta,'construction':'events' in rows[0]['small_label'],'rescue':((s==0)&(l==1)).sum(1),'harm':((s==1)&(l==0)).sum(1)}

def cumulative(score,a):
    order=np.argsort(-score,kind='stable')
    if a['construction']:
        c=a['base']+np.concatenate([np.zeros((1,3,4),int),np.cumsum(a['delta'][order],axis=0)])
        den=2*c[:,0]+c[:,1]+c[:,2]
        metric=np.divide(2*c[:,0],den,out=np.zeros_like(den,dtype=float),where=den!=0).mean(1)
    else:metric=a['base']+np.concatenate([[0],np.cumsum(a['delta'][order])])/a['n']
    return order,metric

def scores(raw,alpha):
    p=torch.from_numpy(raw).softmax(1).numpy()
    return (p[:,1]-p[:,2])/(p[:,1].astype(float)+p[:,2]+2*p[:,3]+1e-8)**alpha

def select(raw,a):
    counts=np.floor(a['n']*np.arange(101)/100).astype(int);x=counts/a['n'];values=[]
    for alpha in ALPHAS:
        _,metric=cumulative(scores(raw,alpha),a)
        values.append({'alpha':alpha,'area100':float(np.trapz(metric[counts],x=x))})
    return max(values,key=lambda v:(v['area100'],-v['alpha'])),values

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads((args.source/'config.json').read_text());dest=args.output
    assert not dest.exists(),dest
    dest.mkdir(parents=True)
    cfg.update(source_id=cfg['id'],id=dest.name,selection_range=[0,100],alpha_grid=ALPHAS,checkpoint_selection='joint epoch/alpha validation exact-call full-curve mean',policy_budget_range=[0,100])
    (dest/'config.json').write_text(json.dumps(cfg,indent=2))
    torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);device=torch.device('cuda')
    records={};data={};started=time.perf_counter()
    for split in ['train','val']:
        records[split],loaded=f.load_data(cfg,split,device)
        data[split]={k:v for k,v in loaded.items() if k in cfg['branches'] or k=='valid'}
        del loaded
    norms={}
    for key in ['hidden','confidence']:
        mean=data['train'][key].mean(0);std=data['train'][key].std(0,unbiased=False).clamp_min(1e-6);norms[key]=(mean,std)
        for split in data:data[split][key]=(data[split][key]-mean)/std
        np.savez(dest/(key+'_normalization.npz'),mean=mean.cpu().numpy(),std=std.cpu().numpy())
    model=f.FeatureRouter(cfg,data['train']['hidden'].shape[1],data['train']['confidence'].shape[1],data['train']['output'].shape[1]).to(device)
    y_np,_=f.targets(records['train'],cfg['endpoint'],cfg['target'],cfg['harm_cost']);y=torch.tensor(y_np,device=device,dtype=torch.float32)
    assert cfg['target']=='four_state' and cfg['strategy']=='natural'
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    val_prepared=prepare(records['val'],cfg['endpoint']);all_train=torch.arange(len(y),device=device)
    best=-float('inf');stale=0
    for epoch in range(1,cfg['max_epochs']+1):
        model.train();order=all_train[torch.randperm(len(y),device=device)];loss_sum=0.
        for b in range(0,len(y),cfg['batch_size']):
            ids=order[b:b+cfg['batch_size']];optimizer.zero_grad(set_to_none=True)
            raw=model(data['train'],ids);loss=-(y[ids]*F.log_softmax(raw,dim=1)).sum(1).mean()
            if not torch.isfinite(loss):raise ValueError(('nonfinite loss',cfg['id'],epoch))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True);optimizer.step();loss_sum+=loss.item()*len(ids)
        model.eval()
        with torch.inference_mode():
            raw=torch.cat([model(data['val'],torch.arange(b,min(b+8192,len(records['val'])),device=device)).cpu() for b in range(0,len(records['val']),8192)]).numpy()
        chosen,candidates=select(raw,val_prepared);improved=chosen['area100']>best
        if improved:
            best=chosen['area100'];best_epoch=epoch;best_alpha=chosen['alpha'];best_raw=raw
            torch.save({'state_dict':model.state_dict(),'config':cfg,'epoch':epoch,'alpha':best_alpha},dest/'best.pt')
            np.save(dest/'validation_logits.npy',raw)
        row={'epoch':epoch,'training_loss':loss_sum/len(y),'validation_candidates':candidates,'best_area100':best,'best_epoch':best_epoch,'best_alpha':best_alpha,'seconds':time.perf_counter()-started}
        with (dest/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);stale=0 if improved else stale+1
        if stale>=cfg['patience']:break
    # No test input or label was read before model and alpha were fixed.
    locked={'best_epoch':best_epoch,'epochs':epoch,'alpha':best_alpha,'validation_area100':best,'training_seconds':time.perf_counter()-started,'test_read_during_selection':False}
    (dest/'selection.json').write_text(json.dumps(locked,indent=2))
    del data,y,optimizer,all_train;torch.cuda.empty_cache()
    model.load_state_dict(torch.load(dest/'best.pt',map_location=device,weights_only=True)['state_dict']);model.eval()
    test_rows,loaded=f.load_data(cfg,'test',device)
    test_data={k:v for k,v in loaded.items() if k in cfg['branches'] or k=='valid'};del loaded
    for key,(mean,std) in norms.items():test_data[key]=(test_data[key]-mean)/std
    with torch.inference_mode():
        test_raw=torch.cat([model(test_data,torch.arange(b,min(b+8192,len(test_rows)),device=device)).cpu() for b in range(0,len(test_rows),8192)]).numpy()
    np.savez(dest/'paired_logits.npz',validation=best_raw,test=test_raw)
    ptest=prepare(test_rows,cfg['endpoint']);f.BUDGETS=[i/100 for i in range(101)]
    output={'config':cfg,'selection':locked,'methods':{}}
    for method in ['normalized','difference','rescue_harm_ratio','error_ratio']:
        ss={}
        for split,raw in [('val',best_raw),('test',test_raw)]:
            if method in ['normalized','difference']:ss[split]=scores(raw,best_alpha if method=='normalized' else 0)
            elif method=='rescue_harm_ratio':ss[split]=raw[:,1]-raw[:,2]
            else:
                z=torch.from_numpy(raw);ss[split]=(torch.logsumexp(z[:,[1,3]],1)-torch.logsumexp(z[:,[2,3]],1)).numpy()
        thresholds=f.validation_thresholds(records['val'],cfg['endpoint'],ss['val']);entry={'thresholds':thresholds}
        for split,a in [('val',val_prepared),('test',ptest)]:
            score=ss[split];order,metric=cumulative(score,a);n=a['n'];sorted_score=score[order]
            counts=np.array([0 if t['operator']=='never' else n if t['operator']=='always' else np.searchsorted(-sorted_score,-t['threshold'],side='right' if t['operator']=='ge' else 'left') for t in thresholds])
            exact=np.floor(n*np.arange(101)/100).astype(int);x=exact/n
            rescued=np.concatenate([[0],np.cumsum(a['rescue'][order])]);harmed=np.concatenate([[0],np.cumsum(a['harm'][order])])
            entry[split]={'score':(metric[counts]*100).tolist(),'calls':(counts/n*100).tolist(),'rescued':rescued[counts].tolist(),'harmed':harmed[counts].tolist(),'exact_score':(metric[exact]*100).tolist(),'exact_calls':(x*100).tolist(),'exact_area100':float(np.trapz(metric[exact],x=x)*100),'threshold_area100':float(np.trapz(metric[counts],dx=.01)*100),'threshold_area25':float(np.trapz(metric[counts[np.arange(0,26,5)]],dx=.05)/.25*100)}
        index=min(range(101),key=lambda i:(-entry['val']['score'][i],entry['val']['calls'][i],i))
        entry['selected_budget']=index
        entry['selected']={split:{k:entry[split][k][index] for k in ['score','calls','rescued','harmed']} for split in ['val','test']}
        output['methods'][method]=entry
    assert np.isclose(output['methods']['normalized']['val']['exact_area100'],best*100)
    (dest/'evaluation.json').write_text(json.dumps(output))
    (dest/'completion.json').write_text(json.dumps({'status':'complete',**locked,'seconds':time.perf_counter()-started,'seed':cfg['seed'],'expert':cfg['expert'],'endpoint':cfg['endpoint']},indent=2))
    print('COMPLETE',json.dumps(locked),flush=True)
