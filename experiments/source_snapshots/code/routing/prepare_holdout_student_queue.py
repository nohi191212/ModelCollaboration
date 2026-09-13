"""Four sizes per heldout task, joint representation distillation only."""
import json,math
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    assert json.loads((work/'task_holdout/training_cpu_precheck.json').read_text())['status']=='passed'
    controls=work/'distillation_controls';queue=[];counts=[]
    if (controls/'task_holdout_student_queue.json').exists():raise FileExistsError('Holdout queue already exists')
    for task in ['cub','construction','grefcoco','nlvr2']:
        source=json.loads((work/'task_holdout'/task/'preparation.json').read_text())
        assert source['heldout_task']==task and source['heldout_source_images']==source['heldout_normalized_text_overlap']==0
        for size in [1,2,4,8]:
            cfg={'id':f'holdout_{task}_joint_{size}M_seed42','heldout_task':task,'budget_millions':size,'depth':2,'seed':42,'batch_size':2048,'lr':.0001,'weight_decay':.01,'temperature':.07,'relation_weight':1.,'cache_pixels_on_gpu':True,'stages':[{'kind':'joint','epochs':300}],'group':'task_holdout_representation'}
            path=controls/'configs'/(cfg['id']+'.json')
            if path.exists():raise FileExistsError(path)
            path.write_text(json.dumps(cfg,indent=2));queue.append(cfg)
            counts.append({'id':cfg['id'],'heldout_task':task,'planned_steps':math.ceil(max(source['images'],source['texts'])/2048)*300,'images':source['images'],'texts':source['texts']})
    (controls/'task_holdout_student_queue.json').write_text(json.dumps(queue,indent=2))
    (controls/'task_holdout_student_plan.json').write_text(json.dumps({'status':'prepared_not_trained','configs_checked':len(queue),'configs':counts,'distillation':'joint image and text representation only; all four holdouts use the same objective','gpu_launched':False,'all_exploration_complete':False},indent=2))
    print(json.dumps({'configs_checked':len(queue),'gpu_launched':False}))
