"""Eight paired end-to-end probes selected from completed frozen exploration."""
import json
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    w=ROOT/'outputs/router_exploration_25pct_20260910'
    analysis=json.loads((w/'followup_queue_analysis.json').read_text())
    assert analysis['all_queue_fits_complete'] and not analysis['warnings']
    queue=[];pairs=[]
    for task in ['cub','construction','grefcoco','nlvr2']:
        eligible=[]
        for row in analysis['groups']:
            c=row['config']
            if c['task']!=task or c['fraction']!=.25 or not c['freeze_encoder'] or 'image' not in c['branches'] or c.get('hidden_shuffle',False):continue
            runs=[r for r in row['runs'] if r['seed']==42]
            assert len(runs)==1
            eligible.append((runs[0]['score'],runs[0]['id'],row['endpoint']))
        score,ref,endpoint=sorted(eligible,key=lambda r:(-r[0],r[1],r[2]))[0]
        base=json.loads((w/'runs'/ref/'config.json').read_text())
        for lr in [1e-5,5e-5]:
            cfg=dict(base,id=f'finetune_small_{len(queue)+1:03d}',stage='small_end_to_end',freeze_encoder=False,encoder_lr=lr,micro_batch_size=16)
            assert cfg['seed']==42 and cfg['fraction']==.25
            queue.append(cfg)
            pairs.append({'task':task,'endpoint_for_selection':endpoint,'reference_id':ref,'run_id':cfg['id'],'seed':42,'frozen_validation_score':score,'encoder_lr':lr})
    assert len(queue)==8
    for name in ['small_finetune_queue.json','small_finetune_design.json']:assert not (w/name).exists()
    (w/'small_finetune_queue.json').write_text(json.dumps(queue,indent=2))
    (w/'small_finetune_design.json').write_text(json.dumps({'new_fits':8,'comparisons':[dict(p,run_id=p['reference_id']) for p in pairs]+pairs,'pairs':pairs,'selection':'Per task highest seed42 native validation selection score among image-using frozen candidates in complete second-stage analysis; stable ID/endpoint tie-break. No third-stage partial results or test scores used. Each new fit keeps reference endpoint setting and all non-encoder training settings.','all_exploration_complete':False,'new_distillation':False},indent=2))
    print(json.dumps({'new_fits':8,'pairs':pairs,'gpu_launched':False}))
