"""Frozen-feature router fitting; one fit can produce separately selected endpoint models."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(ROOT/'code'))
from routing.evaluate_router import correctness,evaluate,fixed_threshold_curve
from routing.raw_student import RawStudent
from routing.student_artifacts import student_artifacts

class FeatureRouter(nn.Module):
    def __init__(self,cfg,hidden_dim,confidence_dim,output_dim=0):
        super().__init__();self.branches=cfg['branches'];self.fusion=cfg['fusion'];d=cfg['representation_dim']
        self.projections=nn.ModuleDict({name:nn.Sequential(nn.Linear(dim,d),nn.LayerNorm(d)) for name,dim in [('text',768),('hidden',hidden_dim),('confidence',confidence_dim),('output',output_dim)] if name in self.branches})
        self.head=nn.Sequential(nn.Linear(len(self.branches)*d,128),nn.GELU(),nn.Linear(128,4 if cfg['target']=='four_state' else 1))
        if 'image' in self.branches:
            self.image_projection=nn.Linear(1538 if self.fusion=='concat' else 768,d);self.image_norm=nn.LayerNorm(d)
            if self.fusion=='attention':
                self.positions=nn.Parameter(torch.empty(1,2,d));self.query=nn.Parameter(torch.empty(1,1,d))
                nn.init.normal_(self.positions,std=.02);nn.init.normal_(self.query,std=.02)
                self.attention=nn.MultiheadAttention(d,4,batch_first=True,dropout=0.)
    def forward(self,data,idx):
        fused=[]
        for branch in self.branches:
            if branch=='image':
                images=data['image'][idx];valid=data['valid'][idx]
                if self.fusion=='concat': x=self.image_projection(torch.cat([images.flatten(1),valid.float()],dim=1))
                else:
                    tokens=self.image_projection(images)+self.positions
                    x=self.attention(self.query.expand(len(idx),-1,-1),tokens,tokens,key_padding_mask=~valid,need_weights=False)[0][:,0]
                fused.append(self.image_norm(x))
            else: fused.append(self.projections[branch](data[branch][idx]))
        return self.head(torch.cat(fused,dim=1))

def targets(records,endpoint,kind,harm_cost):
    s,l=correctness(records,endpoint)
    classes=np.stack([(s*l).mean(1),((1-s)*l).mean(1),(s*(1-l)).mean(1),((1-s)*(1-l)).mean(1)],axis=1)
    if kind=='error': y=(1-s).mean(1,keepdims=True)
    elif kind=='gain': y=(classes[:,1]-harm_cost*classes[:,2])[:,None]
    else: y=classes
    return y,classes

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--smoke',action='store_true');ap.add_argument('--cpu-threads',type=int,default=2)
    a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());work=ROOT/'outputs/router_exploration_25pct_20260910'
    assert cfg['fraction'] in [.1,.25], 'Only frozen grouped exploration fractions are allowed'
    dest=work/('smoke_runs' if a.smoke else 'runs')/cfg['id']
    if dest.exists(): raise FileExistsError(('Run exists; inspect before rerunning',str(dest)))
    dest.mkdir(parents=True);(dest/'config.json').write_text(json.dumps(cfg,indent=2))
    torch.set_num_threads(a.cpu_threads);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    records={};data={};device=torch.device(a.device)
    artifact=student_artifacts(cfg['student'],cfg.get('student_control'))
    if cfg['freeze_encoder']:
        feature_report=json.loads((work/'features'/artifact['feature_key']/'completion.json').read_text())
        assert feature_report['student']==str(artifact['checkpoint']) and feature_report['status']=='complete'
    for split in ['train','val']:
        source=work/'datasets'/cfg['expert']/split
        records[split]=[json.loads(x) for x in (source/'records.jsonl').read_text().splitlines()]
        data[split]={}
        if cfg['freeze_encoder']:
            feature=work/'features'/artifact['feature_key']/cfg['task']/split
            assert json.loads((feature/'sample_ids.json').read_text())==[r['sample_id'] for r in records[split]]
            data[split]={key:torch.from_numpy(np.load(feature/(key+'.npy'))).to(device=device,dtype=torch.bool if key=='valid' else torch.float32) for key in ['image','text','valid']}
        hidden_layers=cfg.get('hidden_layers',[cfg['hidden_layer']])
        assert hidden_layers and len(hidden_layers)==len(set(hidden_layers))
        hidden_arrays=[np.load(source/(layer+'.npy')) for layer in hidden_layers]
        combined=hidden_arrays[0] if len(hidden_arrays)==1 else np.concatenate(hidden_arrays,axis=1)
        data[split]['hidden']=torch.from_numpy(combined).to(device=device,dtype=torch.float32)
        data[split]['confidence']=torch.from_numpy(np.load(source/'confidence.npy')).to(device=device,dtype=torch.float32)
        if 'output' in cfg['branches']:
            data[split]['output']=torch.from_numpy(np.load(source/'output.npy')).to(device=device,dtype=torch.float32)
        assert all(len(x)==len(records[split]) for x in data[split].values())
        assert all(torch.isfinite(x).all() for x in data[split].values())
    if cfg['fraction']==.1:
        subset=json.loads((work/'manifests'/cfg['task']/'train10_indices.json').read_text())
        indices=subset['indices']
        assert [records['train'][i]['sample_id'] for i in indices]==subset['sample_ids']
        records['train']=[records['train'][i] for i in indices]
        selected=torch.tensor(indices,device=device,dtype=torch.long)
        data['train']={key:value.index_select(0,selected) for key,value in data['train'].items()}
    eligible=json.loads((work/'preparation_complete.json').read_text())['counts'][cfg['task']]['eligible_train_rows']
    (dest/'training_rows.json').write_text(json.dumps({'fraction':cfg['fraction'],'fraction_definition':'source groups, not exact label rows','actual_row_fraction':len(records['train'])/eligible,'train_rows':len(records['train']),'validation_rows':len(records['val']),'sample_ids':[r['sample_id'] for r in records['train']]}))
    if cfg.get('hidden_shuffle',False):
        assert 'hidden' in cfg['branches']
        for split in ['train','val']:
            name='train10' if split=='train' and cfg['fraction']==.1 else split
            shuffle=json.loads((work/'manifests'/cfg['task']/('hidden_shuffle_'+name+'.json')).read_text())
            assert shuffle['sample_ids']==[r['sample_id'] for r in records[split]]
            indices=torch.tensor(shuffle['indices'],device=device,dtype=torch.long)
            data[split]['hidden']=data[split]['hidden'].index_select(0,indices)
            (dest/('hidden_shuffle_'+split+'.json')).write_text(json.dumps(shuffle))
    # Train-only feature scaling AFTER subsetting; no extra 25% rows inform a 10% run.
    for key in ['hidden','confidence']:
        mean=data['train'][key].mean(0);std=data['train'][key].std(0,unbiased=False).clamp_min(1e-6)
        for split in data: data[split][key]=(data[split][key]-mean)/std
        np.savez(dest/(key+'_normalization.npz'),mean=mean.cpu().numpy(),std=std.cpu().numpy())
    model=FeatureRouter(cfg,data['train']['hidden'].shape[1],data['train']['confidence'].shape[1],data['train']['output'].shape[1] if 'output' in cfg['branches'] else 0).to(device)
    if cfg.get('init_router_checkpoint'):
        assert cfg['freeze_encoder'] and cfg['branches']==['image','text'] and cfg['target']=='error'
        checkpoint=Path(cfg['init_router_checkpoint'])
        origin=json.loads((checkpoint.parent/'completion.json').read_text())
        assert origin['status']=='complete' and not origin['target_used_for_selection'] and origin['target_training_rows_used']==0
        if not a.smoke:assert not origin['smoke'] and origin['strict_student_verified']
        saved=torch.load(checkpoint,map_location=device,weights_only=True);source_cfg=saved['config']
        assert source_cfg['heldout_task']==cfg['task']
        for key in ['student','student_control','branches','target','fusion','representation_dim']:
            assert source_cfg.get(key)==cfg.get(key),(key,source_cfg.get(key),cfg.get(key))
        model.load_state_dict(saved['state_dict'],strict=True)
        (dest/'initialization.json').write_text(json.dumps({'source_checkpoint':str(checkpoint),'target_supervision':'explicit adaptation; not zero-shot','source_was_smoke':origin['smoke']}))
    raw_student=None if cfg['freeze_encoder'] else RawStudent(cfg,records,device)
    endpoints=['Qwen3.8','MiniCPM'] if cfg['endpoint']=='both' else [cfg['endpoint']]
    y,states=targets(records['train'],endpoints[0],cfg['target'],cfg['harm_cost'])
    # For an error probe sampling must also be independent of the large model.
    strata=np.concatenate([1-y,y],axis=1) if cfg['target']=='error' else states
    counts=strata.sum(0);weights=np.zeros_like(counts);present=counts>0
    weights[present]=len(y)/(present.sum()*counts[present])
    sample_weights=torch.tensor(strata@weights,device=device,dtype=torch.float32)
    y=torch.tensor(y,device=device,dtype=torch.float32)
    parameter_groups=[{'params':list(model.parameters()),'lr':cfg['lr']}]
    if raw_student is not None:
        parameter_groups.append({'params':[p for p in raw_student.parameters() if p.requires_grad],'lr':cfg['encoder_lr']})
    optimized=[p for group in parameter_groups for p in group['params']]
    optimizer=torch.optim.AdamW(parameter_groups,weight_decay=cfg['weight_decay'])
    best={m:float('-inf') for m in endpoints};best_epoch={};stale=0;started=time.time();history=[]
    all_train=torch.arange(len(y),device=device);max_epochs=1 if a.smoke else cfg['max_epochs']
    for epoch in range(1,max_epochs+1):
        model.train();loss_sum=0.;seen=0
        if raw_student is not None:raw_student.train()
        if cfg['strategy']=='balanced': order=torch.multinomial(sample_weights,len(y),replacement=True)
        else: order=all_train[torch.randperm(len(y),device=device)]
        for b in range(0,len(order),cfg['batch_size']):
            batch_ids=order[b:b+cfg['batch_size']];optimizer.zero_grad(set_to_none=True)
            micro=cfg['micro_batch_size'] if raw_student is not None else len(batch_ids)
            for ids in batch_ids.split(micro):
                if raw_student is None:output=model(data['train'],ids)
                else:
                    current={key:value[ids] for key,value in data['train'].items()}
                    current.update(raw_student('train',ids));output=model(current,torch.arange(len(ids),device=device))
                if cfg['target']=='error': loss=F.binary_cross_entropy_with_logits(output,y[ids],reduction='none').mean(1)
                elif cfg['target']=='gain': loss=(output-y[ids]).square().mean(1)
                else: loss=-(y[ids]*F.log_softmax(output,dim=1)).sum(1)
                if cfg['strategy']=='weighted': loss=loss*sample_weights[ids]
                loss=loss.mean()
                if not torch.isfinite(loss): raise RuntimeError(('nonfinite training loss',cfg['id'],epoch,b))
                (loss*len(ids)/len(batch_ids)).backward()
                loss_sum+=loss.item()*len(ids);seen+=len(ids)
            torch.nn.utils.clip_grad_norm_(optimized,float('inf'),error_if_nonfinite=True);optimizer.step()
            if a.smoke and seen>=cfg['batch_size']*3: break
        model.eval();raw=[]
        if raw_student is not None:raw_student.eval()
        validation_batch=cfg['micro_batch_size'] if raw_student is not None else 4096
        with torch.no_grad():
            for b in range(0,len(records['val']),validation_batch):
                ids=torch.arange(b,min(b+validation_batch,len(records['val'])),device=device)
                if raw_student is None:output=model(data['val'],ids)
                else:
                    current={key:value[ids] for key,value in data['val'].items()}
                    current.update(raw_student('val',ids));output=model(current,torch.arange(len(ids),device=device))
                raw.append(output.cpu())
        raw=torch.cat(raw)
        if cfg['target']=='four_state': p=raw.softmax(1);scores=(p[:,1]-cfg['harm_cost']*p[:,2]).numpy()
        elif cfg['target']=='error': scores=raw.sigmoid().numpy()[:,0]
        else: scores=raw.numpy()[:,0]
        results={m:evaluate(records['val'],m,scores) for m in endpoints};improved=False
        for endpoint,result in results.items():
            if result['selection_score']>best[endpoint]:
                best[endpoint]=result['selection_score'];best_epoch[endpoint]=epoch;improved=True
                saved={'state_dict':model.state_dict(),'config':cfg,'epoch':epoch}
                if raw_student is not None:saved['student_state_dict']=raw_student.student.state_dict()
                torch.save(saved,dest/(endpoint+'_best.pt'))
                np.save(dest/(endpoint+'_scores.npy'),scores)
                (dest/(endpoint+'_metrics.json')).write_text(json.dumps(result,indent=2))
                (dest/(endpoint+'_thresholds.json')).write_text(json.dumps(fixed_threshold_curve(records['val'],endpoint,scores),indent=2))
        stale=0 if improved else stale+1
        row={'epoch':epoch,'training_loss':loss_sum/seen,'validation_selection':{m:r['selection_score'] for m,r in results.items()},'score_std':float(scores.std()),'elapsed_seconds':time.time()-started}
        history.append(row)
        with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(cfg['id'],row,flush=True)
        if stale>=cfg['patience']:break
    encoder_counts=artifact['parameters']
    encoder=(encoder_counts['vision'] if 'image' in cfg['branches'] else 0)+(encoder_counts['text'] if 'text' in cfg['branches'] else 0)
    result={'status':'complete','id':cfg['id'],'epochs':epoch,'best_epoch':best_epoch,'best_selection':best,'seconds':time.time()-started,'router_trainable_parameters':sum(p.numel() for p in model.parameters()),'student_parameters':encoder,'total_parameters':encoder+sum(p.numel() for p in model.parameters()),'smoke':a.smoke,'anomalies':[]}
    if max(x['score_std'] for x in history)<1e-6:result['anomalies'].append('constant validation scores; inspect before accepting')
    result['encoder_trainable_parameters']=0 if raw_student is None else sum(p.numel() for p in raw_student.parameters() if p.requires_grad)
    result['total_trainable_parameters']=sum(p.numel() for p in optimized)
    result['input_mode']='cached_frozen_features' if raw_student is None else 'raw_pixels_and_strict_tokens_with_gradient'
    result['hidden_layers']=hidden_layers
    result['hidden_input_dimension']=data['train']['hidden'].shape[1]
    result['student_checkpoint']=str(artifact['checkpoint'])
    result['router_initialization']=cfg.get('init_router_checkpoint','random')
    result['cpu_threads']=a.cpu_threads
    (dest/'completion.json').write_text(json.dumps(result,indent=2))
