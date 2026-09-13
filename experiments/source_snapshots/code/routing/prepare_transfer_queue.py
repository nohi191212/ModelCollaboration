"""Predeclare portable transfer and paired target adaptation; no training launch."""
import json
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    initial=json.loads((work/'initial_queue.json').read_text())
    students=json.loads((work/'distillation_controls/task_holdout_student_queue.json').read_text())
    directory=work/'transfer_configs';directory.mkdir(exist_ok=False)
    transfer=[];adaptation=[];pairs=[]
    for student in students:
        task=student['heldout_task'];size=str(student['budget_millions'])+'M'
        population=json.loads((work/'task_holdout/transfer_sources'/(task+'.json')).read_text())
        assert population['target_training_rows_used']==0
        for seed in [42]:
            cfg={'id':f'transfer_{task}_{size}_seed{seed}','heldout_task':task,'student':size,'student_control':student['id'],
                 'branches':['image','text'],'target':'error','fusion':'concat','representation_dim':512,'seed':seed,
                 'lr':.001,'weight_decay':.01,'batch_size':256,'max_epochs':50,'patience':8}
            transfer.append(cfg);(directory/(cfg['id']+'.json')).write_text(json.dumps(cfg,indent=2))
            for expert in population['target_experts']:
                base=next(x for x in initial if x['expert']==expert and x['target']=='error' and x['seed']==seed)
                for fraction in [.1,.25]:
                    paired=[]
                    for initialization in ['random','transfer']:
                        child=dict(base,id=f'adapt_{expert}_{size}_{int(fraction*100)}pct_{initialization}_seed{seed}',stage='target_adaptation',student=size,student_control=student['id'],branches=['image','text'],fusion='concat',representation_dim=512,fraction=fraction)
                        if initialization=='transfer':child['init_router_checkpoint']=str(work/'transfer_runs/formal'/cfg['id']/'best.pt')
                        adaptation.append(child);paired.append(child)
                    a={k:v for k,v in paired[0].items() if k not in ['id','init_router_checkpoint']}
                    b={k:v for k,v in paired[1].items() if k not in ['id','init_router_checkpoint']}
                    assert a==b
                    pairs.append({'task':task,'expert':expert,'student':size,'fraction':fraction,'seed':seed,'random_id':paired[0]['id'],'transfer_id':paired[1]['id'],'source_run':cfg['id']})
    assert len(transfer)==16 and len(adaptation)==128 and len(pairs)==64
    for queue in [transfer,adaptation]:assert len({cfg['id'] for cfg in queue})==len(queue)
    (work/'transfer_queue.json').write_text(json.dumps(transfer,indent=2))
    (work/'adaptation_queue.json').write_text(json.dumps(adaptation,indent=2))
    (work/'transfer_adaptation_design.json').write_text(json.dumps({'status':'prepared_not_trained','transfer_fits':16,'adaptation_fits':128,'matched_pairs':pairs,
        'source_model_selection':'Source validation loss only; fixed generic architecture, not target-selected best method.',
        'target_adaptation_selection':'Target validation selection explicitly allowed only in supervised adaptation, matched to random-init control.',
        'fraction_definition':'Existing nested source-group 10% and 25%, not necessarily exact label percentages.',
        'remaining':['strict student training','feature extraction','GPU validation','formal transfer and adaptation','full method per-task retraining','final report'],'all_exploration_complete':False},indent=2))
    print(json.dumps({'transfer_fits':16,'adaptation_fits':128,'matched_pairs':64,'gpu_launched':False}))
