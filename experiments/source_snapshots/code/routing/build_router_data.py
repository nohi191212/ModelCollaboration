"""Join cached same-forward expert states and labels by sample identity, CPU only."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(ROOT/'code'))
from large_label_collection.judge import judge, EVENTS
from routing.prepare_exploration import EXPERTS

def small_label(task,row,expert,threshold):
    pred=row['expert_output'];out={'parse_status':'valid','parse_error':None}
    if task=='cub': out['predicted_category_id']=pred['prediction']
    elif task=='nlvr2': out['prediction']=pred['prediction']
    elif task=='grefcoco':
        if expert=='groundingdino':
            assert pred['box_threshold']==.7
            out['objects']=pred['objects']
        else:
            boxes=pred['boxes_xyxy_absolute'];scores=pred['scores']
            assert len(scores)==len(boxes)
            out['objects']=[{'bbox_xyxy_absolute':b,'score':float(s)} for b,s in zip(boxes,scores)]
    else:
        selected=[x for x in pred['boxes_xyxy_score_class'] if x[4]>=threshold]
        # Presence supervision; no alteration of cached boxes. Mask-IoU is not used here.
        out['events']={event:{'present':any(int(x[5])==i for x in selected),'boxes':[]} for i,event in enumerate(EVENTS)}
    label=judge(task,row,out)
    if task=='construction':
        for value in label['events'].values(): value['positive_target_mask_iou']=None
        label['native_metric_note']='presence metrics only; cached expert masks not evaluated by this data adapter'
    return label

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--expert',choices=EXPERTS,required=True)
    args=ap.parse_args();expert=args.expert;task=EXPERTS[expert]
    work=ROOT/'outputs/router_exploration_25pct_20260910';dest=work/'datasets'/expert
    if (dest/'completion.json').exists(): raise FileExistsError('Already prepared; do not overwrite')
    dest.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'outputs/router_training_cache_complete_20260908'/expert
    splits={s:[json.loads(x) for x in (work/'manifests'/task/(s+'.jsonl')).read_text().splitlines()] for s in ['train','val']}
    lookup={x['sample_id']:(s,i) for s,rows in splits.items() for i,x in enumerate(rows)}
    assert len(lookup)==sum(map(len,splits.values()))
    large={}
    for model in ['Qwen3.8','MiniCPM']:
        large[model]={}
        source_splits=['train'] if task=='cub' else ['train','dev' if task=='nlvr2' else 'val']
        for split in source_splits:
            p=ROOT/'outputs/large_model_labels_h200_full_20260909'/model/task/split/'labels.jsonl'
            with p.open() as f:
                for line in f:
                    x=json.loads(line);sid=x['sample_id']
                    if sid in lookup:
                        assert sid not in large[model],(model,sid)
                        large[model][sid]=x
        assert set(large[model])==set(lookup),(expert,model,'missing labels',len(set(lookup)-set(large[model])))
    threshold={'yolo26x':.005,'rtdetr_x_fp32':.05}.get(expert)
    columns=json.loads((cache/'train/confidence_columns.json').read_text())
    dimensions=json.loads((cache/'train/shard_000000/manifest.json').read_text())['dimensions']
    arrays={};records={s:[None]*len(rows) for s,rows in splits.items()}
    for split,rows in splits.items():
        d=dest/split;d.mkdir(exist_ok=True)
        arrays[split]={key:np.lib.format.open_memmap(d/(key+'.npy'),mode='w+',dtype=np.float16,shape=(len(rows),shape[1])) for key,shape in dimensions.items()}
        arrays[split]['confidence']=np.lib.format.open_memmap(d/'confidence.npy',mode='w+',dtype=np.float32,shape=(len(rows),2*len(columns)))
    for source_split in source_splits:
        for shard in sorted((cache/source_split).glob('shard_*')):
            found=[]
            with gzip.open(shard/'records.jsonl.gz','rt') as f:
                for position,line in enumerate(f):
                    x=json.loads(line);sid=x['sample_id']
                    if sid not in lookup: continue
                    split,index=lookup[sid];expected=splits[split][index]
                    assert records[split][index] is None,('duplicate',sid)
                    for key in ['image_path','image1','image2','sentence','expression']:
                        if key in expected: assert x[key]==expected[key],(sid,key)
                    small=small_label(task,x,expert,threshold)
                    if task in ['cub','nlvr2']:
                        for model in large: assert large[model][sid]['target']==small['target']
                    if task=='construction':
                        for model in large:
                            assert all(large[model][sid]['events'][e]['target']==small['events'][e]['target'] for e in EVENTS)
                    if task=='grefcoco':
                        for model in large:
                            assert (large[model][sid]['target_type'],large[model][sid]['target_boxes'])==(small['target_type'],small['target_boxes'])
                    conf=x['expert_output']['confidence']
                    values=[conf[c] for c in columns]
                    mask=[v is not None for v in values]
                    encoded=np.array([float(v) if ok else 0.0 for v,ok in zip(values,mask)]+list(map(float,mask)),dtype=np.float32)
                    assert np.isfinite(encoded).all(),(expert,sid,'confidence nonfinite')
                    arrays[split]['confidence'][index]=encoded
                    record=dict(expected,small_label=small,large_labels={m:large[m][sid] for m in large},source_shard=str(shard),source_position=position)
                    records[split][index]=record
                    found.append((position,split,index))
            if found:
                with np.load(shard/'states.npz') as states:
                    for key in dimensions:
                        values=states[key]
                        for split in splits:
                            match=[(p,i) for p,s,i in found if s==split]
                            if match:
                                src,dst=zip(*match);selected=values[list(src)]
                                assert np.isfinite(selected).all(),(expert,key,shard)
                                arrays[split][key][list(dst)]=selected
                print(expert,source_split,shard.name,'selected',len(found),flush=True)
    for split in splits:
        assert all(x is not None for x in records[split]),(expert,split,'missing expert rows')
        for a in arrays[split].values(): a.flush()
        (dest/split/'records.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records[split]))
    summary={'expert':expert,'task':task,'rows':{s:len(x) for s,x in records.items()},'state_dimensions':{k:v[1] for k,v in dimensions.items()},'confidence_columns':columns+[c+'_valid' for c in columns],'missing_confidence':'explicit zero placeholder plus validity flag, not a fabricated probability','expert_detection_threshold':threshold,'threshold_source':'20260903_construction_event_experts_corrected_v2/results; unchanged existing thresholds','test_rows_used':0,'status':'complete'}
    (dest/'completion.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
