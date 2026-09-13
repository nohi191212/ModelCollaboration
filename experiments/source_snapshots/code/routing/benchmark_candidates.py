"""Router-only timing, not upstream models or complete deployed system latency."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(root/'code'))
from routing.train_router import FeatureRouter
from routing.raw_student import RawStudent
from routing.student_artifacts import student_artifacts
from routing.evaluate_router import metric
ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,required=True);ap.add_argument('--audit',action='store_true');ap.add_argument('--audit-repeat',action='store_true');a=ap.parse_args()
assert not a.audit_repeat or a.audit
assert 0<=a.worker<4
torch.set_num_threads(2);device=torch.device('cuda')
w=root/'outputs/router_exploration_25pct_20260910'
figures=json.loads((w/'report_figures/figure_data.json').read_text())
out=w/('router_timing_audit_repeat' if a.audit_repeat else 'router_timing_audit' if a.audit else 'router_timing');out.mkdir(exist_ok=True)
report={'scope':'Router only: cached resized uint8 pixels to student encoding and router head. Hidden/confidence/output inputs already cached and resident on GPU. Excludes original image decoding/resizing, tokenizer preparation, upstream small/large model calls, model loading, network and deployment overhead. Full validation throughput is not test-set end-to-end latency.','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'worker':a.worker,'cpu_threads':2,'status':'running','rows':[]}
for figure in figures[a.worker::4]:
    cid=figure['run_id'];endpoint=figure['endpoint'];dest=w/'runs'/cid
    cfg=json.loads((dest/'config.json').read_text())
    if a.audit and not set(cfg['branches'])&{'image','text'}:continue
    assert cfg['freeze_encoder'] and not cfg.get('hidden_shuffle',False)
    source=w/'datasets'/cfg['expert']/'val'
    records=[json.loads(x) for x in (source/'records.jsonl').read_text().splitlines()]
    artifact=student_artifacts(cfg['student'],cfg.get('student_control'))
    feature=w/'features'/artifact['feature_key']/cfg['task']/'val'
    assert json.loads((feature/'sample_ids.json').read_text())==[r['sample_id'] for r in records]
    data={k:torch.from_numpy(np.load(feature/(k+'.npy'))).to(device=device,dtype=torch.bool if k=='valid' else torch.float32) for k in ['image','text','valid']}
    layers=cfg.get('hidden_layers',[cfg['hidden_layer']])
    data['hidden']=torch.from_numpy(np.concatenate([np.load(source/(layer+'.npy')) for layer in layers],axis=1)).float().to(device)
    data['confidence']=torch.from_numpy(np.load(source/'confidence.npy')).float().to(device)
    if 'output' in cfg['branches']:data['output']=torch.from_numpy(np.load(source/'output.npy')).float().to(device)
    for key in ['hidden','confidence']:
        norm=np.load(dest/(key+'_normalization.npz'))
        data[key]=(data[key]-torch.from_numpy(norm['mean']).to(device))/torch.from_numpy(norm['std']).to(device)
    model=FeatureRouter(cfg,data['hidden'].shape[1],data['confidence'].shape[1],data['output'].shape[1] if 'output' in data else 0).to(device)
    checkpoint=torch.load(dest/(endpoint+'_best.pt'),map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    student=RawStudent(cfg,{'val':records},device) if set(cfg['branches'])&{'image','text'} else None
    if student is not None:student.eval()
    row={'run_id':cid,'expert':cfg['expert'],'endpoint':endpoint,'branches':cfg['branches'],'student':cfg['student'],'validation_rows':len(records),'router_parameters':sum(p.numel() for p in model.parameters()),'student_loaded_parameters':sum(p.numel() for p in student.parameters()) if student is not None else 0,'measurements':[]}
    with torch.inference_mode():
        for mode in (['pixel_encoder_and_head'] if a.audit else ['cached_head','pixel_encoder_and_head']):
            for batch,limit in ([(16,len(records))] if a.audit else [(1,min(128,len(records))),(16,len(records))]):
                ids_all=np.unique(np.linspace(0,len(records)-1,limit,dtype=int))
                elapsed=[];outputs=[]
                for repeat in range(1 if a.audit else 4):
                    torch.cuda.synchronize();start=time.perf_counter()
                    current_outputs=[]
                    for offset in range(0,len(ids_all),batch):
                        ids=torch.tensor(ids_all[offset:offset+batch],device=device)
                        current={k:v[ids] for k,v in data.items()}
                        if mode=='pixel_encoder_and_head' and student is not None:current.update(student('val',ids))
                        prediction=model(current,torch.arange(len(ids),device=device))
                        current_outputs.append(prediction.cpu())
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                    if repeat or a.audit:elapsed.append(seconds)
                    outputs=current_outputs
                raw=torch.cat(outputs)
                if cfg['target']=='four_state':p=raw.softmax(1);scores=(p[:,1]-cfg['harm_cost']*p[:,2]).numpy()
                elif cfg['target']=='error':scores=raw.sigmoid().numpy()[:,0]
                else:scores=raw.numpy()[:,0]
                saved=np.load(dest/(endpoint+'_scores.npy'))[ids_all]
                assert np.isfinite(scores).all()
                row['measurements'].append({'mode':mode,'batch_size':batch,'rows':len(ids_all),'repeats_seconds':elapsed,'median_ms_per_row':float(np.median(elapsed)*1000/len(ids_all)),'max_saved_score_difference':float(np.max(np.abs(saved-scores))),'warmup':'one complete untimed pass over the same rows','output_copy_to_cpu_included':True})
                if a.audit:
                    point=next(p for p in json.loads((dest/(endpoint+'_thresholds.json')).read_text())['curve'] if p['budget']==.25)
                    decisions=[]
                    for values in [saved,scores]:
                        if point['operator']=='never':decision=np.zeros(len(values),dtype=bool)
                        elif point['operator']=='always':decision=np.ones(len(values),dtype=bool)
                        elif point['operator']=='ge':decision=values>=point['threshold']
                        else:
                            assert point['operator']=='gt'
                            decision=values>point['threshold']
                        decisions.append(decision)
                    assert metric(records,endpoint,decisions[0])==point['metric']
                    row['fixed_threshold_audit']={'threshold':point['threshold'],'operator':point['operator'],'decision_changes':int((decisions[0]!=decisions[1]).sum()),'saved_calls':int(decisions[0].sum()),'online_calls':int(decisions[1].sum()),'saved_metric':point['metric'],'online_metric':metric(records,endpoint,decisions[1]),'threshold_retuned':False}
                    row['measurements'][-1]['warmup']='none; diagnostic pass, not timing evidence'
                    np.save(out/(cid+'_'+endpoint+'_online_scores.npy'),scores)
    report['rows'].append(row)
    (out/f'worker{a.worker}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(row),flush=True)
    del model,student,data,checkpoint
report['status']='complete'
(out/f'worker{a.worker}.json').write_text(json.dumps(report,indent=2))
