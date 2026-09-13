"""Prepare interactions and true encoder updates after all single-factor results."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path
from prepare_followup_queue import canonical,signature

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

def variants(anchor):
    inputs=[['image'],['image','text'],['hidden'],['confidence'],['output'],['output','confidence'],
            ['image','text','hidden'],['image','text','confidence'],['hidden','confidence'],['image','text','hidden','confidence']]
    for branches in inputs:
        for target in ['error','gain','four_state']:
            for strategy in ['natural','weighted','balanced']:
                yield 'inputs_targets_strategies',dict(anchor,branches=branches,target=target,strategy=strategy)
    if set(anchor['branches'])&{'image','text'}:
        for size in ['1M','2M','4M','8M']:
            for dim in [32,64,128,256,512]:
                yield 'sizes_dimensions',dict(anchor,student=size,representation_dim=dim)
        yield 'freeze_reference',dict(anchor)
        for lr in [1e-5,5e-5,1e-4]:
            yield 'finetune',dict(anchor,freeze_encoder=False,encoder_lr=lr,micro_batch_size=16)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--precheck',action='store_true');args=ap.parse_args()
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    initial=json.loads((work/'initial_queue.json').read_text())
    if args.precheck:
        checked=0
        for anchor in initial:
            if anchor['seed']!=42:continue
            for group,cfg in variants(anchor):
                cfg=canonical(cfg,anchor['hidden_layer'],'Qwen3.8' if anchor['endpoint']=='both' else anchor['endpoint'])
                assert cfg['fraction']==.25
                if not cfg['freeze_encoder']:
                    assert set(cfg['branches'])&{'image','text'} and cfg['encoder_lr']>0 and cfg['micro_batch_size']<=cfg['batch_size']
                checked+=1
        result={'status':'passed','config_checks':checked,'real_results_selected':False,'gpu_launched':False}
        (work/'interaction_config_precheck.json').write_text(json.dumps(result));print(json.dumps(result));raise SystemExit(0)
    analysis=json.loads((work/'followup_queue_analysis.json').read_text())
    assert analysis['all_queue_fits_complete'], 'Finish and replay the full followup queue before selection'
    for warning in analysis['warnings']:
        assert json.loads((work/'runs'/warning['id']/'anomaly_review.json').read_text())['decision']=='no_bug'
    followup=json.loads((work/'followup_queue.json').read_text())
    prior=initial+followup;used={signature(cfg):cfg['id'] for cfg in prior}
    eligible=defaultdict(list)
    for row in analysis['groups']:
        cfg=row['config']
        if cfg['fraction']!=.25 or cfg.get('hidden_shuffle',False) or not cfg['freeze_encoder']:continue
        assert row['seeds_complete']
        scores=[r['score'] for r in row['runs'] if r['seed'] in [42,43,44]]
        assert len(scores)==3
        eligible[(cfg['expert'],row['endpoint'])].append((statistics.mean(scores),cfg,row['runs']))
    anchors=[];queue=[];comparisons=[]
    for (expert,endpoint),rows in sorted(eligible.items()):
        ranked=sorted(rows,key=lambda x:(-x[0],json.dumps(x[1],sort_keys=True)))
        selected=ranked[:2]
        # Fine-tune must have an actual image/text encoder, even if a hidden-only model wins.
        for candidate in [r for r in ranked if set(r[1]['branches'])&{'image','text'}][:2]:
            if candidate not in selected:selected.append(candidate)
        for rank,(mean,base,runs) in enumerate(selected,1):
            anchor=dict(base,endpoint=endpoint,seed=42)
            anchors.append({'expert':expert,'endpoint':endpoint,'rank':rank,'mean_three_seed':mean,'config':anchor,'source_ids':[r['id'] for r in runs]})
            for group,variant in variants(anchor):
                for seed in [42]:
                    cfg=canonical(dict(variant,seed=seed),anchor['hidden_layer'],endpoint)
                    key=signature(cfg)
                    if key not in used:
                        cfg.update(id=f'interaction_{len(queue)+1:06d}',stage='interaction_and_finetune')
                        queue.append(cfg);used[key]=cfg['id']
                    comparisons.append({'expert':expert,'endpoint':endpoint,'anchor_rank':rank,'group':group,'seed':seed,'run_id':used[key]})
    assert len(eligible)==16
    for filename in ['interaction_queue.json','interaction_design.json']:
        if (work/filename).exists():raise FileExistsError(filename)
    (work/'interaction_queue.json').write_text(json.dumps(queue,indent=2))
    (work/'interaction_design.json').write_text(json.dumps({'anchors':anchors,'comparisons':comparisons,'new_fits':len(queue),
        'selection':'Per pair top two by the same seeds42/43/44; add top two encoder-using candidates if needed for actual finetuning.',
        'all_exploration_complete':False,'test_rows_used':0},indent=2))
    print(json.dumps({'new_fits':len(queue),'anchors':len(anchors),'gpu_launched':False}))
