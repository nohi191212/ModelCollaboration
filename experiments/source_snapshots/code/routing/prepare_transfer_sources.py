"""Verify source-only router populations for heldout-task transfer."""
import json
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    experts=['cub','glsim','nlvr2','vilt','groundingdino','instancevg','yolo26x','rtdetr_x_fp32']
    populations={}
    for expert in experts:
        directory=work/'datasets'/expert;metadata=json.loads((directory/'completion.json').read_text())
        parts={}
        for split in ['train','val']:
            records=[json.loads(x) for x in (directory/split/'records.jsonl').read_text().splitlines()]
            assert len(records)==metadata['rows'][split]
            identities=[r['sample_id'] for r in records]
            assert len(set(identities))==len(identities)
            parts[split]={'records_file':str(directory/split/'records.jsonl'),'rows':len(records),'sample_ids':identities}
        assert not set(parts['train']['sample_ids'])&set(parts['val']['sample_ids'])
        populations[expert]={'task':metadata['task'],'splits':parts,'hidden_dimensions':metadata['state_dimensions'],'confidence_dimensions':len(metadata['confidence_columns'])}
    out=work/'task_holdout/transfer_sources';out.mkdir(exist_ok=False);summary=[]
    for target in ['cub','construction','grefcoco','nlvr2']:
        sources={expert:rows for expert,rows in populations.items() if rows['task']!=target}
        targets=[expert for expert,rows in populations.items() if rows['task']==target]
        assert len(sources)==6 and len(targets)==2 and all(x['task']!=target for x in sources.values())
        report={'heldout_task':target,'source_experts':sources,'target_experts':targets,
                'source_router_supervision':'existing fixed 25% training manifests only',
                'portable_inputs':['image','text'],
                'scope':'Universal image/text diagnostic, not transfer of heterogeneous hidden-state adapters. Full-method per-task retraining remains separate.',
                'checkpoint_selection':'Source validation mean error-prediction BCE, equally weighted by expert; never average disparate native task metrics.',
                'target_training_rows_used':0,'target_validation_used_for_selection':False,'training_complete':False}
        (out/(target+'.json')).write_text(json.dumps(report,indent=2))
        summary.append({'heldout_task':target,'source_experts':list(sources),'target_experts':targets,'source_train_rows':sum(x['splits']['train']['rows'] for x in sources.values()),'source_val_rows':sum(x['splits']['val']['rows'] for x in sources.values())})
    (out/'preparation.json').write_text(json.dumps({'status':'source_populations_verified_not_trained','groups':summary},indent=2));print(json.dumps(summary,indent=2))
