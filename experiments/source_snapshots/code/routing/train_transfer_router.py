"""Source-only error router using portable image/text representations."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(ROOT/'code'))
from routing.train_router import FeatureRouter
from routing.student_artifacts import student_artifacts
from routing.evaluate_router import correctness,evaluate
from routing.transfer_thresholds import source_thresholds,target_threshold_results

def load_data(work,artifact,expert,split,device,smoke):
    directory=work/'datasets'/expert
    task=json.loads((directory/'completion.json').read_text())['task']
    records=[json.loads(x) for x in (directory/split/'records.jsonl').read_text().splitlines()]
    features=work/'features'/artifact['feature_key']/task/split
    assert json.loads((features/'sample_ids.json').read_text())==[r['sample_id'] for r in records]
    if smoke:records=records[:8]
    data={key:torch.tensor(np.load(features/(key+'.npy'),mmap_mode='r')[:len(records)],device=device,dtype=torch.bool if key=='valid' else torch.float32) for key in ['image','text','valid']}
    assert all(torch.isfinite(x).all() for x in data.values())
    return records,data

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());torch.manual_seed(cfg['seed']);torch.set_num_threads(2)
    assert cfg['branches']==['image','text'] and cfg['target']=='error'
    work=ROOT/'outputs/router_exploration_25pct_20260910';device=torch.device(args.device)
    source=json.loads((work/'task_holdout/transfer_sources'/(cfg['heldout_task']+'.json')).read_text())
    artifact=student_artifacts(cfg['student'],cfg.get('student_control'))
    if not args.smoke:
        student_cfg=json.loads((artifact['directory']/'config.json').read_text())['control_config']
        assert student_cfg['heldout_task']==cfg['heldout_task']
    feature_report=json.loads((work/'features'/artifact['feature_key']/'completion.json').read_text())
    assert feature_report['student']==str(artifact['checkpoint']) and feature_report['status']=='complete'
    dest=work/'transfer_runs'/('smoke' if args.smoke else 'formal')/cfg['id'];dest.mkdir(parents=True,exist_ok=False)
    (dest/'config.json').write_text(json.dumps(cfg,indent=2))
    training=[];labels=[];weights=[];validation={};manifest={}
    for expert,population in source['source_experts'].items():
        assert population['task']!=cfg['heldout_task']
        records,data=load_data(work,artifact,expert,'train',device,args.smoke)
        s,_=correctness(records,'Qwen3.8');training.append(data)
        labels.append(torch.tensor((1-s).mean(1,keepdims=True),device=device,dtype=torch.float32))
        weights.append(torch.full((len(records),),1/len(records),device=device))
        manifest[expert]={'train_ids':[r['sample_id'] for r in records]}
        records,data=load_data(work,artifact,expert,'val',device,args.smoke)
        s,_=correctness(records,'Qwen3.8')
        validation[expert]=(data,torch.tensor((1-s).mean(1,keepdims=True),device=device,dtype=torch.float32))
        manifest[expert]['val_ids']=[r['sample_id'] for r in records]
    (dest/'source_samples.json').write_text(json.dumps(manifest))
    data={key:torch.cat([part[key] for part in training]) for key in ['image','text','valid']};y=torch.cat(labels);weights=torch.cat(weights)
    model=FeatureRouter(cfg,0,0).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    best=float('inf');stale=0;history=[];started=time.time()
    for epoch in range(1,(1 if args.smoke else cfg['max_epochs'])+1):
        model.train();losses=[]
        for step in range(2 if args.smoke else math.ceil(len(y)/cfg['batch_size'])):
            ids=torch.multinomial(weights,cfg['batch_size'],replacement=True)
            optimizer.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(data,ids),y[ids])
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True);optimizer.step();losses.append(loss.item())
        model.eval();val_losses={}
        with torch.no_grad():
            for expert,(part,labels) in validation.items():
                total=0.
                for start in range(0,len(labels),4096):
                    ids=torch.arange(start,min(start+4096,len(labels)),device=device)
                    total+=F.binary_cross_entropy_with_logits(model(part,ids),labels[ids],reduction='sum').item()
                val_losses[expert]=total/len(labels)
        selection=sum(val_losses.values())/len(val_losses);assert np.isfinite(selection)
        row={'epoch':epoch,'training_loss':sum(losses)/len(losses),'source_validation_loss':selection,'per_expert_validation_loss':val_losses};history.append(row)
        with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(row,flush=True)
        if selection<best:
            best=selection;stale=0;torch.save({'state_dict':model.state_dict(),'config':cfg,'epoch':epoch},dest/'best.pt')
        else:stale+=1
        if stale>=cfg['patience']:break
    model.load_state_dict(torch.load(dest/'best.pt',map_location=device,weights_only=True)['state_dict']);model.eval()
    source_scores={}
    with torch.no_grad():
        for expert,(part,labels) in validation.items():
            pieces=[]
            for start in range(0,len(labels),4096):
                ids=torch.arange(start,min(start+4096,len(labels)),device=device);pieces.append(model(part,ids).sigmoid().cpu().numpy()[:,0])
            source_scores[expert]=np.concatenate(pieces)
            np.save(dest/(expert+'_source_scores.npy'),source_scores[expert])
    calibration=source_thresholds(source_scores)
    (dest/'source_thresholds.json').write_text(json.dumps(calibration,indent=2))
    # Target data is first loaded after checkpoint selection has finished.
    results={};fixed_results={}
    for expert in source['target_experts']:
        records,part=load_data(work,artifact,expert,'val',device,args.smoke);scores=[]
        with torch.no_grad():
            for start in range(0,len(records),4096):
                ids=torch.arange(start,min(start+4096,len(records)),device=device);scores.append(model(part,ids).sigmoid().cpu().numpy()[:,0])
        scores=np.concatenate(scores);np.save(dest/(expert+'_scores.npy'),scores)
        results[expert]={endpoint:evaluate(records,endpoint,scores) for endpoint in ['Qwen3.8','MiniCPM']}
        fixed_results[expert]={endpoint:target_threshold_results(records,endpoint,scores,calibration) for endpoint in ['Qwen3.8','MiniCPM']}
    (dest/'target_metrics.json').write_text(json.dumps(results,indent=2))
    (dest/'target_fixed_threshold_metrics.json').write_text(json.dumps(fixed_results,indent=2))
    (dest/'completion.json').write_text(json.dumps({'status':'complete','smoke':args.smoke,'seconds':time.time()-started,'epochs':epoch,'source_validation_loss':best,'target_used_for_selection':False,'target_training_rows_used':0,'strict_student_verified':not args.smoke,'scope':'Portable image/text diagnostic; target budget ranking is not a source-calibrated deployment threshold.'},indent=2))
