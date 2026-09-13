"""Freeze grouped 25% training manifests and the initial exploration queue, without GPUs."""
import collections
import json
from pathlib import Path
import random

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
SPLITS=ROOT/'data/router_splits_strict_20260910'
OUT=ROOT/'outputs/router_exploration_25pct_20260910'
EXPERTS={'cub':'cub','glsim':'cub','nlvr2':'nlvr2','vilt':'nlvr2','groundingdino':'grefcoco','instancevg':'grefcoco','yolo26x':'construction','rtdetr_x_fp32':'construction'}

if __name__=='__main__':
    if (OUT/'preparation_complete.json').exists(): raise FileExistsError('Exploration manifests already frozen; read existing files, do not redraw')
    OUT.mkdir(exist_ok=True)
    parents={}
    def root(key):
        parents.setdefault(key,key)
        while parents[key]!=key:
            parents[key]=parents[parents[key]];key=parents[key]
        return key
    def connect(a,b): parents[root(b)]=root(a)
    cub_map={x['sample_id']:x['official_ids'] for x in json.loads((SPLITS/'cub_image_correspondence.json').read_text())}
    for row,ids in cub_map.items():
        for i in ids: connect('cub:'+row,'cub_image:'+i)
    nlvr_rows=[json.loads(x) for x in (ROOT/'data/nlvr2/annotations/data/train.json').read_text().splitlines()]
    for x in nlvr_rows:
        key='nlvr:'+x['identifier'].split('-')[1]
        for side in ['left_url','right_url']: connect(key,'url:'+x[side])
    counts={}
    for task in ['cub','construction','grefcoco','nlvr2']:
        train=[json.loads(x) for x in (SPLITS/task/'train.jsonl').read_text().splitlines()]
        groups=collections.defaultdict(list)
        for row in train:
            if task=='cub': group=root('cub:'+row['sample_id'])
            elif task=='nlvr2': group=root('nlvr:'+row['sample_id'].split('-')[1])
            elif task=='grefcoco': group=str(row['image_id'])
            else: group=row['sample_id']
            groups[group].append(dict(row,source_group=group))
        rng=random.Random(42)
        ordered=sorted(groups);rng.shuffle(ordered)
        if task=='cub':
            per_class=collections.defaultdict(list)
            for group in ordered: per_class[groups[group][0]['ground_truth']['category_id']].append(group)
            selected=[]
            for category in sorted(per_class):
                candidates=per_class[category]
                selected.extend(candidates[:round(len(candidates)*.25)])
        else: selected=ordered[:round(len(ordered)*.25)]
        chosen=sorted([x for g in selected for x in groups[g]],key=lambda x:x['sample_id'])
        val_name='dev' if task=='nlvr2' else 'val'
        validation=[json.loads(x) for x in (SPLITS/task/(val_name+'.jsonl')).read_text().splitlines()]
        assert not {x['sample_id'] for x in chosen}&{x['sample_id'] for x in validation}
        dest=OUT/'manifests'/task;dest.mkdir(parents=True,exist_ok=True)
        (dest/'train.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in chosen))
        (dest/'val.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in validation))
        (dest/'group_order.json').write_text(json.dumps({'ordered_groups':ordered,'selected_groups':selected}))
        counts[task]={'eligible_train_rows':len(train),'eligible_groups':len(groups),'selected_groups':len(selected),'train_rows':len(chosen),'validation_rows':len(validation),'largest_group_rows':max(map(len,groups.values()))}
    default_layers={
        'cub':'residual_blocks__layer_13','glsim':'global_local_encoder__layer_10',
        'nlvr2':'multimodal_encoder__layer_10','vilt':'ordered_image_text__layer_10',
        'groundingdino':'query_decoder__layer_04','instancevg':'query_decoder__layer_01',
        'yolo26x':'feature_blocks__layer_14','rtdetr_x_fp32':'query_decoder__layer_04'}
    queue=[]
    for expert,task in EXPERTS.items():
        manifest=json.loads((ROOT/'outputs/router_training_cache_complete_20260908'/expert/'train/shard_000000/manifest.json').read_text())
        assert default_layers[expert] in manifest['dimensions']
        for fusion in ['concat','attention']:
            for target in ['error','gain','four_state']:
                for endpoint in (['both'] if target=='error' else ['Qwen3.8','MiniCPM']):
                    for seed in [42,43,44]:
                        cid=f'{expert}__{fusion}__{target}__{endpoint}__seed{seed}'
                        queue.append({'id':cid,'stage':'initial','task':task,'expert':expert,'endpoint':endpoint,'target':target,'seed':seed,'fusion':fusion,'student':'4M','branches':['image','text','hidden','confidence'],'representation_dim':512,'hidden_layer':default_layers[expert],'strategy':'natural','harm_cost':1.0,'batch_size':256,'lr':0.001,'weight_decay':0.01,'max_epochs':50,'patience':8,'freeze_encoder':True,'fraction':0.25})
    assert len(queue)==240
    (OUT/'initial_queue.json').write_text(json.dumps(queue,indent=2))
    (OUT/'preparation_complete.json').write_text(json.dumps({'status':'manifests_and_initial_configs_prepared','counts':counts,'initial_fits':240,'initial_pair_evaluations':288,'test_rows_loaded':0,'sample_seed':42,'selection':'validation normalized trapezoidal area over call rates 0, .05, .10, .15, .20, .25','all_exploration_complete':False},indent=2))
    print((OUT/'preparation_complete.json').read_text())
