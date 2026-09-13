"""Fit a mandatory-input router on train/validation only; never open test data."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    import torch
    from torch.nn import functional as F
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from routing import train_router_formal as formal
    formal.ROOT=a.root.resolve()
    formal.WORK=formal.ROOT/'outputs/router_full_20260910'
    from routing.train_router_formal import FeatureRouter,load_data,targets,ranked_curve
    cfg=json.loads(a.config.read_text())
    if not {'image','hidden'}<=set(cfg['branches']):raise ValueError('Mandatory image and hidden branches missing')
    if cfg['task'] in ['nlvr2','grefcoco'] and 'text' not in cfg['branches']:raise ValueError('Real task text must be encoded')
    if cfg['endpoint'] not in ['MiniCPM','Qwen3.8']:raise ValueError('Each fit must have one explicit large-model endpoint')
    dest=a.output
    if (dest/'completion.json').exists():raise FileExistsError(dest/'completion.json')
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'config.json').write_text(json.dumps(cfg,indent=2))
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    device=torch.device(a.device)
    records={};data={}
    for split in ['train','val']:
        records[split],loaded=load_data(cfg,split,device)
        data[split]={k:v for k,v in loaded.items() if k in cfg['branches'] or k=='valid'}
        del loaded
    (dest/'training_rows.json').write_text(json.dumps({'fraction':1.0,'train_rows':len(records['train']),
        'validation_rows':len(records['val']),'test_loaded':False},indent=2))
    for key in ['hidden','confidence']:
        mean=data['train'][key].mean(0);std=data['train'][key].std(0,unbiased=False).clamp_min(1e-6)
        for split in data:data[split][key]=(data[split][key]-mean)/std
        np.savez(dest/(key+'_normalization.npz'),mean=mean.cpu().numpy(),std=std.cpu().numpy())
    model=FeatureRouter(cfg,data['train']['hidden'].shape[1],data['train']['confidence'].shape[1],data['train']['output'].shape[1]).to(device)
    y_np,states_np=targets(records['train'],cfg['endpoint'],cfg['target'],cfg['harm_cost'])
    if cfg['target']=='route':y_np=states_np[:,1:2]
    strata=np.concatenate([1-y_np,y_np],axis=1) if cfg['target']=='error' else states_np
    counts=strata.sum(0);weights=np.zeros_like(counts);present=counts>0
    weights[present]=len(y_np)/(present.sum()*counts[present])
    sample_weights=torch.tensor(strata@weights,device=device,dtype=torch.float32)
    y=torch.tensor(y_np,device=device,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    best=-float('inf');best_epoch=0;stale=0;started=time.perf_counter()
    all_train=torch.arange(len(y),device=device)
    for epoch in range(1,cfg['max_epochs']+1):
        model.train()
        order=torch.multinomial(sample_weights,len(y),replacement=True) if cfg['strategy']=='balanced' else all_train[torch.randperm(len(y),device=device)]
        loss_sum=0
        for b in range(0,len(y),cfg['batch_size']):
            ids=order[b:b+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            raw=model(data['train'],ids)
            if cfg['target'] in ['error','route']:loss=F.binary_cross_entropy_with_logits(raw,y[ids],reduction='none').mean(1)
            elif cfg['target']=='gain':loss=(raw-y[ids]).square().mean(1)
            else:loss=-(y[ids]*F.log_softmax(raw,dim=1)).sum(1)
            if cfg['strategy']=='weighted':loss=loss*sample_weights[ids]
            loss=loss.mean()
            if not torch.isfinite(loss):raise ValueError(('nonfinite loss',cfg['id'],epoch))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True);optimizer.step()
            loss_sum+=loss.item()*len(ids)
        model.eval()
        with torch.inference_mode():
            raw=torch.cat([model(data['val'],torch.arange(b,min(b+8192,len(records['val'])),device=device)).cpu()
                           for b in range(0,len(records['val']),8192)]).numpy()
        if cfg['target']=='four_state':
            probs=torch.from_numpy(raw).softmax(1).numpy()
            scores=probs[:,1]-probs[:,2] if cfg['ranking']=='probability_difference' else raw[:,1]-raw[:,2]
        else:scores=torch.from_numpy(raw[:,0]).sigmoid().numpy() if cfg['target'] in ['error','route'] else raw[:,0]
        metrics=ranked_curve(records['val'],cfg['endpoint'],scores)
        improved=metrics['selection_score']>best
        if improved:
            best=metrics['selection_score'];best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'config':cfg,'epoch':epoch},dest/'best.pt')
            np.save(dest/'validation_scores.npy',scores)
            np.save(dest/'validation_logits.npy',raw)
            (dest/'validation_metrics.json').write_text(json.dumps(metrics,indent=2))
        row={'epoch':epoch,'training_loss':loss_sum/len(y),'validation_score':metrics['selection_score'],
             'best_validation_score':best,'elapsed_seconds':time.perf_counter()-started}
        with (dest/'history.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        print(cfg['id'],json.dumps(row),flush=True)
        stale=0 if improved else stale+1
        if stale>=cfg['patience']:break
    parameters=json.loads((a.root/'outputs/mini_siglip_strict_20260910_300ep'/cfg['student']/'config.json').read_text())['parameters']
    active=parameters['vision']+(parameters['text'] if 'text' in cfg['branches'] else 0)
    result={'status':'validation_complete','id':cfg['id'],'config':cfg,'best_epoch':best_epoch,'epochs':epoch,
            'validation_selection_score':best,'seconds':time.perf_counter()-started,'test_loaded':False,
            'head_trainable_parameters':sum(p.numel() for p in model.parameters()),
            'encoder_stored_parameters':parameters['total'],'encoder_active_parameters':active}
    (dest/'completion.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
