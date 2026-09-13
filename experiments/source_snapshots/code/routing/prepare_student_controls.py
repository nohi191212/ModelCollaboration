"""Prepare explicit distillation/depth controls; never launch GPU work."""
import json,sys
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(ROOT/'code'))
from routing.train_student_control import budget_width

if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910/distillation_controls'
    dest=work/'configs';dest.mkdir(exist_ok=False)
    base={'seed':42,'batch_size':2048,'lr':1e-4,'weight_decay':.01,'temperature':.07,'relation_weight':1.,'cache_pixels_on_gpu':True}
    sources={'random':[], 'image':[{'kind':'image','epochs':300}],
             'relation':[{'kind':'relation','epochs':300}],
             'image_relation':[{'kind':'image','epochs':150},{'kind':'relation','epochs':150}]}
    queue=[]
    for size in [1,2,4,8]:
        for source,stages in sources.items():
            queue.append(dict(base,id=f'source_{source}_{size}M_seed42',budget_millions=size,depth=2,stages=stages,group='distillation_source'))
        for depth in [1,3,4,8]:
            queue.append(dict(base,id=f'depth_{depth}_{size}M_seed42',budget_millions=size,depth=depth,stages=[{'kind':'joint','epochs':300}],group='depth_fixed_budget'))
    width=budget_width(4,2)
    for depth in [1,3,4,8]:
        queue.append(dict(base,id=f'width_{width}_depth_{depth}_seed42',width=width,depth=depth,stages=[{'kind':'joint','epochs':300}],group='depth_fixed_width'))
    assert len(queue)==36 and len({cfg['id'] for cfg in queue})==36
    details=[]
    for cfg in queue:
        w=cfg.get('width') or budget_width(cfg['budget_millions'],cfg['depth'])
        count=24*cfg['depth']*w*w+(26*cfg['depth']+2602)*w+132608
        (dest/(cfg['id']+'.json')).write_text(json.dumps(cfg,indent=2))
        details.append({'id':cfg['id'],'width':w,'depth':cfg['depth'],'parameters':count,'stages':cfg['stages'],'group':cfg['group']})
    report={'status':'configs_prepared_not_trained','configs':details,'new_students':len(queue),
            'existing_joint_depth2_students':['1M','2M','4M','8M'],
            'seed_scope':'student seed42 only here; router repeats are separate; student-seed robustness remains pending',
            'compute_matching':'300 stage epochs total for trained controls, NOT matched steps/examples/FLOPs; image has fewer examples than joint/relation; random has zero updates',
            'image_relation_schedule':'150 image epochs then 150 relation epochs; this is a tested schedule, not assumed optimal',
            'data_scope':'full strict distillation training cache; router supervision remains fixed 25%; no heldout task generalization claim',
            'gpu_launched':False,'all_exploration_complete':False}
    (work/'student_control_queue.json').write_text(json.dumps(queue,indent=2))
    (work/'student_control_design.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'new_students':len(queue),'fixed_width':width,'parameter_range':[min(x['parameters'] for x in details),max(x['parameters'] for x in details)],'gpu_launched':False}))
