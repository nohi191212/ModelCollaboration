"""Generate the next frozen-encoder ablations from completed, reviewed initial results.

This is one exploration stage, NOT completion of depth/distillation/generalization.
"""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
IGNORED={'id','stage','seed'}

def signature(cfg,include_seed=True):
    ignored={'id','stage'} if include_seed else IGNORED
    return json.dumps({k:v for k,v in cfg.items() if k not in ignored},sort_keys=True)

def variants(anchor,layers):
    yield 'anchor',dict(anchor)
    # I = image+text; keep the pure-image control distinct.
    inputs=[['image','text'],['hidden'],['confidence'],['image','text','hidden'],
            ['image','text','confidence'],['hidden','confidence'],['image','text','hidden','confidence'],['image']]
    for branches in inputs:yield 'inputs',dict(anchor,branches=branches)
    for branches in [['output'],['output','confidence']]:yield 'output_only_controls',dict(anchor,branches=branches)
    by_branch=defaultdict(list)
    for layer in layers:by_branch[layer.split('__layer_')[0]].append(layer)
    for branch in sorted(by_branch):
        ordered=sorted(by_branch[branch])
        for index in sorted({0,len(ordered)//3,2*len(ordered)//3,len(ordered)-1}):
            yield 'hidden_positions',dict(anchor,hidden_layer=ordered[index])
        combined=[ordered[i] for i in sorted({0,len(ordered)//2,len(ordered)-1})]
        if len(combined)>1:yield 'hidden_combinations',dict(anchor,hidden_layers=combined)
    for d in [32,64,128,256,512]:yield 'dimensions',dict(anchor,representation_dim=d)
    for size in ['1M','2M','4M','8M']:yield 'student_sizes',dict(anchor,student=size)
    for target in ['error','gain','four_state']:
        for strategy in ['natural','weighted','balanced']:
            yield 'targets_and_imbalance',dict(anchor,target=target,strategy=strategy)
    for target in ['gain','four_state']:
        for cost in [0.,.5,1.,2.,4.]:yield 'harm_cost',dict(anchor,target=target,harm_cost=cost)
    yield 'nested_fraction',dict(anchor,fraction=.1)
    yield 'hidden_group_shuffle',dict(anchor,hidden_shuffle=True)
    for seed in range(42,50):yield 'anchor_seeds',dict(anchor,seed=seed)

def canonical(cfg,default_layer,endpoint):
    cfg=dict(cfg)
    if cfg['target']=='error':cfg.update(endpoint='both',harm_cost=1.)
    else:cfg['endpoint']=endpoint
    if 'image' not in cfg['branches']:cfg['fusion']='concat'
    if not set(cfg['branches'])&{'image','text'}:cfg['student']='4M'
    if 'hidden' not in cfg['branches']:
        cfg['hidden_layer']=default_layer;cfg.pop('hidden_layers',None);cfg.pop('hidden_shuffle',None)
    elif 'hidden_layers' in cfg:cfg['hidden_layer']=cfg['hidden_layers'][0]
    return cfg

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--precheck',action='store_true');a=ap.parse_args()
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    initial=json.loads((work/'initial_queue.json').read_text())
    if a.precheck:
        # Uses real configs and cached layer dimensions, not fabricated training results.
        checked=0
        for cfg in initial:
            layers=json.loads((work/'datasets'/cfg['expert']/'completion.json').read_text())['state_dimensions']
            for group,variant in variants(cfg,layers):
                endpoint='Qwen3.8' if cfg['endpoint']=='both' else cfg['endpoint']
                variant=canonical(variant,cfg['hidden_layer'],endpoint)
                assert variant['hidden_layer'] in layers and variant['freeze_encoder']
                assert all(layer in layers for layer in variant.get('hidden_layers',[]))
                assert variant['fraction'] in [.1,.25] and variant['representation_dim'] in [32,64,128,256,512]
                assert variant['branches'] and len(set(variant['branches']))==len(variant['branches'])
                assert set(variant['branches'])<={'image','text','hidden','confidence','output'}
                checked+=1
        result={'status':'config_precheck_passed','variant_checks':checked,'training_results_used':False,'gpu_launched':False}
        (work/'followup_config_precheck.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0)
    # No candidate selection until every initial fit is complete and anomalies reviewed.
    analysis=json.loads((work/'initial_queue_analysis.json').read_text())
    assert analysis['all_queue_fits_complete'] and analysis['fits_complete']==analysis['fits_planned']==len(initial)
    for warning in analysis['warnings']:
        assert json.loads((work/'runs'/warning['id']/'anomaly_review.json').read_text())['decision']=='no_bug'
    candidates=defaultdict(lambda:defaultdict(list))
    used={signature(cfg):cfg['id'] for cfg in initial}
    for cfg in initial:
        dest=work/'runs'/cfg['id'];done=json.loads((dest/'completion.json').read_text())
        assert done['status']=='complete' and not done['smoke']
        if done['anomalies']:
            assert json.loads((dest/'anomaly_review.json').read_text())['decision']=='no_bug'
        endpoints=['Qwen3.8','MiniCPM'] if cfg['endpoint']=='both' else [cfg['endpoint']]
        for endpoint in endpoints:
            score=done['best_selection'][endpoint]
            assert isinstance(score,(int,float)) and 0<=score<=1
            candidates[(cfg['expert'],endpoint)][signature(cfg,False)].append((cfg,score))
    anchors=[];queue=[];comparisons=[]
    for (expert,endpoint),groups in sorted(candidates.items()):
        ranked=[]
        for group,runs in groups.items():
            assert sorted(cfg['seed'] for cfg,score in runs)==[42,43,44]
            ranked.append((statistics.mean(score for cfg,score in runs),group,runs))
        ranked.sort(key=lambda item:(-item[0],item[1]))
        for rank,(mean,group,runs) in enumerate(ranked[:2],1):
            anchor=dict(runs[0][0]);anchor['seed']=42
            anchors.append({'expert':expert,'endpoint':endpoint,'rank':rank,'mean_selection':mean,'source_ids':[cfg['id'] for cfg,score in runs],'config':anchor})
            layers=json.loads((work/'datasets'/expert/'completion.json').read_text())['state_dimensions']
            for ablation,variant in variants(anchor,layers):
                seeds=[variant['seed']] if ablation=='anchor_seeds' else [42,43,44]
                for seed in seeds:
                    cfg=canonical(dict(variant,seed=seed),anchor['hidden_layer'],endpoint)
                    key=signature(cfg)
                    if key not in used:
                        cfg.update(id=f'followup_{len(queue)+1:06d}',stage='frozen_single_factor')
                        used[key]=cfg['id'];queue.append(cfg)
                    comparisons.append({'expert':expert,'endpoint':endpoint,'anchor_rank':rank,'group':ablation,'seed':seed,'run_id':used[key]})
    payload={'anchors':anchors,'comparisons':comparisons,'new_fits':len(queue),'initial_fits_reused':len({x['run_id'] for x in comparisons if not x['run_id'].startswith('followup_')}),
             'selection':'Top two structures per expert/endpoint by mean initial validation score across seeds42/43/44; stable config tie-break.',
             'remaining_groups':['interactions after single-factor results','fine-tuning','distillation source controls','depth controls','task generalization','cost and figures','final HTML/Markdown'],
             'all_exploration_complete':False,'test_rows_used':0}
    for name in ['followup_queue.json','followup_design.json']:
        if (work/name).exists():raise FileExistsError(name)
    (work/'followup_queue.json').write_text(json.dumps(queue,indent=2))
    (work/'followup_design.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps({'new_fits':len(queue),'anchors':len(anchors),'initial_fits_reused':payload['initial_fits_reused']}),flush=True)
