"""Replay completed scores and summarize matched seed groups; validation only."""
import argparse,json,statistics,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(ROOT/'code'))
from routing.evaluate_router import evaluate,fixed_threshold_curve

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--queue',default='initial_queue');args=ap.parse_args()
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    queue=json.loads((work/(args.queue+'.json')).read_text());primary_ids={cfg['id'] for cfg in queue}
    design_file=work/(args.queue.removesuffix('_queue')+'_design.json')
    if design_file.exists():
        design=json.loads(design_file.read_text())
        reused_ids={row['run_id'] for row in design['comparisons']}-primary_ids
        queue=queue+[json.loads((work/'runs'/cid/'config.json').read_text()) for cid in sorted(reused_ids)]
    assert len({cfg['id'] for cfg in queue})==len(queue)
    groups=defaultdict(list);expected=defaultdict(set)
    datasets={};checks=[];completed=0;new_completed=0;warnings=[]
    for cfg in queue:
        endpoints=['Qwen3.8','MiniCPM'] if cfg['endpoint']=='both' else [cfg['endpoint']]
        base={k:v for k,v in cfg.items() if k not in ['id','stage','seed','endpoint']}
        for endpoint in endpoints:expected[(json.dumps(base,sort_keys=True),endpoint)].add(cfg['seed'])
        dest=work/'runs'/cfg['id']
        if not (dest/'completion.json').exists():continue
        done=json.loads((dest/'completion.json').read_text());completed+=1
        new_completed+=int(cfg['id'] in primary_ids)
        assert done['status']=='complete' and not done['smoke']
        assert json.loads((dest/'config.json').read_text())==cfg
        if cfg['expert'] not in datasets:
            datasets[cfg['expert']]=[json.loads(x) for x in (work/'datasets'/cfg['expert']/'val/records.jsonl').read_text().splitlines()]
        records=datasets[cfg['expert']]
        history=[json.loads(x) for x in (dest/'history.jsonl').read_text().splitlines()]
        assert all(np.isfinite(row['training_loss']) for row in history)
        for endpoint in endpoints:
            scores=np.load(dest/(endpoint+'_scores.npy'));replayed=evaluate(records,endpoint,scores)
            saved=json.loads((dest/(endpoint+'_metrics.json')).read_text())
            assert abs(replayed['selection_score']-saved['selection_score'])<1e-12
            assert abs(done['best_selection'][endpoint]-saved['selection_score'])<1e-12
            assert abs(max(row['validation_selection'][endpoint] for row in history)-saved['selection_score'])<1e-12
            assert replayed['curve']==saved['curve'],(cfg['id'],endpoint,'curve replay mismatch')
            thresholds=json.loads((dest/(endpoint+'_thresholds.json')).read_text())
            assert fixed_threshold_curve(records,endpoint,scores)==thresholds,(cfg['id'],endpoint,'fixed threshold replay mismatch')
            # Reapply the persisted deployment rule directly, including ties.
            for point in thresholds['curve']:
                operator=point['operator'];threshold=point['threshold']
                assert operator in ['never','always','ge','gt']
                if operator=='never':selected=np.zeros(len(scores),dtype=bool)
                elif operator=='always':selected=np.ones(len(scores),dtype=bool)
                elif operator=='ge':selected=scores>=threshold
                else:selected=scores>threshold
                assert int(selected.sum())<=int(np.floor(len(scores)*point['budget']))
                assert float(selected.mean())==point['actual_fraction']
            if saved['score_std']<1e-6:warnings.append({'id':cfg['id'],'endpoint':endpoint,'reason':'Selected checkpoint scores nearly constant; inspect before accepting.'})
            checks.append({'id':cfg['id'],'endpoint':endpoint,'score_replay':'passed','history_best':'passed','saved_threshold_and_call_fraction':'passed'})
            groups[(json.dumps(base,sort_keys=True),endpoint)].append({'id':cfg['id'],'seed':cfg['seed'],'score':saved['selection_score'],'seconds':done['seconds'],'epochs':done['epochs']})
    rows=[]
    for (key,endpoint),runs in groups.items():
        cfg=json.loads(key);reference=json.loads((work/'references'/(cfg['expert']+'_'+endpoint+'.json')).read_text())
        scores=[run['score'] for run in runs];mean=statistics.mean(scores)
        rows.append({'config':cfg,'endpoint':endpoint,'runs':runs,'seeds_complete':{r['seed'] for r in runs}==expected[(key,endpoint)],
                     'expected_seeds':sorted(expected[(key,endpoint)]),'mean':mean,'std':statistics.stdev(scores) if len(scores)>1 else None,
                     'random_mean':reference['random_area_mean'],'confidence_score':reference['confidence_area'],'small_metric':reference['small_metric'],
                     'delta_random':mean-reference['random_area_mean'],'delta_confidence':mean-reference['confidence_area'],'delta_no_upgrade':mean-reference['small_metric']})
    rows.sort(key=lambda r:(r['config']['expert'],r['endpoint'],-r['mean']))
    result={'queue':args.queue,'unix_time':time.time(),'fits_complete':completed,'fits_planned':len(queue),'all_queue_fits_complete':completed==len(queue),
            'new_fits_complete':new_completed,'new_fits_planned':len(primary_ids),'reused_fits_complete':completed-new_completed,
            'replayed_endpoints':len(checks),'checks':checks,'warnings':warnings,'groups':rows,'test_rows_used':0,
            'scope':'Validation selection summaries, not independent test results. Seed std is not a confidence interval. No best-method claim while queue incomplete.'}
    (work/(args.queue+'_analysis.json')).write_text(json.dumps(result,indent=2))
    top={}
    for row in rows:
        if row['seeds_complete']:top.setdefault(row['config']['expert']+' / '+row['endpoint'],row)
    names={'cub':'ResNet／CUB','glsim':'GLSim／CUB','nlvr2':'BEiT3／NLVR2','vilt':'ViLT／NLVR2','groundingdino':'GroundingDINO／gRefCOCO','instancevg':'InstanceVG／gRefCOCO','yolo26x':'YOLO26x／工地','rtdetr_x_fp32':'RT-DETR-X／工地'}
    target_names={'error':'预测小模型出错','gain':'预测净收益','four_state':'四种对错组合'}
    lines=['# 阶段性结果（不是最终结论）','','## 1. 做了什么','',f'已核查{completed}/{len(queue)}次训练结果（含复用），其中本队列新训练{new_completed}/{len(primary_ids)}次；重算{len(checks)}组大模型配对的保存分数，与报告曲线及训练历史最优值一致。',
           '', '## 2. 为什么做','','只比较种子已齐的配置，汇总均值与标准差，并同时对比随机调用、置信规则和完全不调用大模型。',
           '', '## 3. 当前结果','','以下仅列当前已完成重复组中均值最高者；未完成配置不能参加排名，也不代表最终最佳。数字为0%—25%调用区间的曲线平均表现×100，不是25%单点成绩。工地使用事件存在性F1，其余使用严格逐样本正确率，不跨任务混算。',
           '', '| 专用模型／任务 | 大模型 | 当前候选目标 | 重复次数 | 均值±标准差 | 比置信规则 | 比不调用 |', '|---|---|---|---:|---:|---:|---:|']
    for row in top.values():
        c=row['config'];std='—' if row['std'] is None else f"{row['std']*100:.3f}"
        lines.append(f"| {names[c['expert']]} | {row['endpoint']} | {target_names[c['target']]} | {len(row['runs'])} | {row['mean']*100:.3f}±{std} | {row['delta_confidence']*100:+.3f} | {row['delta_no_upgrade']*100:+.3f} |")
    lines+=['','注意：这些是用于选方案的验证结果，不是独立测试成绩；标准差不是置信区间，不宣称统计显著。CUB的ResNet训练见过当前内部验证图片，GLSim训练来源未独立确认，须保留限制。超过随机或置信规则不等于超过专用模型。',
            '',f'需要排查的近常数分数记录：{len(warnings)}。后续仍需完成全部配置、异常复核及完整消融。',
            '', '## 4. 文件在哪里','',f'实际远端目录：`{work}`。原始核查与全部配置均值：`{args.queue}_analysis.json`；逐配置权重、曲线和分数：`runs/`。本文件为滚动阶段记录，不代替最终HTML报告。','']
    (work/(args.queue+'_analysis.md')).write_text('\n'.join(lines))
    print(json.dumps({'fits_complete':completed,'fits_planned':len(queue),'replayed_endpoints':len(checks),'warnings':warnings,
                      'current_top_complete_seed_groups':[{k:v for k,v in row.items() if k!='runs'} for row in top.values()]},indent=2),flush=True)
