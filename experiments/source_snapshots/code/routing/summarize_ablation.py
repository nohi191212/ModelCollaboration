"""Paired three-seed ablation deltas, only from replay-verified results."""
import json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    design=json.loads((work/'followup_design.json').read_text())
    analysis=json.loads((work/'followup_queue_analysis.json').read_text())
    excluded=set()
    for warning in analysis['warnings']:
        review=work/'runs'/warning['id']/'anomaly_review.json'
        if not review.exists() or json.loads(review.read_text())['decision']!='no_bug':excluded.add(warning['id'])
    verified={(x['id'],x['endpoint']) for x in analysis['checks'] if x['id'] not in excluded}
    configs={};scores={}
    for cid,endpoint in verified:
        dest=work/'runs'/cid
        if cid not in configs:configs[cid]=json.loads((dest/'config.json').read_text())
        scores[(cid,endpoint)]=json.loads((dest/(endpoint+'_metrics.json')).read_text())['selection_score']
    anchors=defaultdict(dict)
    for item in design['comparisons']:
        if item['group']=='anchor' and (item['run_id'],item['endpoint']) in verified:
            anchors[(item['expert'],item['endpoint'],item['anchor_rank'])][item['seed']]=scores[(item['run_id'],item['endpoint'])]
    grouped=defaultdict(dict);representatives={}
    for item in design['comparisons']:
        if item['group'] in ['anchor','anchor_seeds'] or item['seed'] not in [42,43,44]:continue
        cid=item['run_id'];endpoint=item['endpoint']
        if (cid,endpoint) not in verified:continue
        config={k:v for k,v in configs[cid].items() if k not in ['id','stage','seed','endpoint']}
        key=(item['expert'],endpoint,item['anchor_rank'],item['group'],json.dumps(config,sort_keys=True))
        if item['seed'] in grouped[key]:assert grouped[key][item['seed']]['run_id']==cid
        grouped[key][item['seed']]={'run_id':cid,'score':scores[(cid,endpoint)]}
        representatives[key]=config
    rows=[];incomplete=0
    for key,runs in grouped.items():
        expert,endpoint,rank,group,_=key;base=anchors[(expert,endpoint,rank)]
        if set(runs)!={42,43,44} or not {42,43,44}<=set(base):incomplete+=1;continue
        differences=[runs[seed]['score']-base[seed] for seed in [42,43,44]]
        rows.append({'expert':expert,'endpoint':endpoint,'anchor_rank':rank,'group':group,'config':representatives[key],
                     'mean':statistics.mean(r['score'] for r in runs.values()),'anchor_mean':statistics.mean(base[s] for s in [42,43,44]),
                     'paired_delta_mean':statistics.mean(differences),'paired_delta_std':statistics.stdev(differences),
                     'paired_deltas':differences,'run_ids':[runs[s]['run_id'] for s in [42,43,44]],'seeds':[42,43,44]})
    rows.sort(key=lambda x:(x['expert'],x['endpoint'],x['anchor_rank'],x['group'],-x['paired_delta_mean']))
    report={'status':'partial' if not analysis['all_queue_fits_complete'] else 'queue_complete',
            'completed_three_seed_comparisons':len(rows),'incomplete_started_groups':incomplete,'excluded_unreviewed_runs':sorted(excluded),
            'rows':rows,'note':'Paired validation selection deltas, not independent test or statistical significance. Full rows include negative and zero findings.'}
    (work/'paired_ablation_summary.json').write_text(json.dumps(report,indent=2))
    by_group=defaultdict(list)
    for row in rows:by_group[(row['expert'],row['endpoint'],row['group'])].append(row['paired_delta_mean']*100)
    lines=['# 已核查的配对消融差值（阶段记录）','','## 1. 做了什么','',f'对{len(rows)}个已凑齐三个相同种子的对照，计算相对于各自候选的逐种子差值；只读取已复算通过的结果。',
           '','## 2. 为什么做','','相同种子配对，避免把不同候选或不同重复次数混在一起。负值和零值全部保留，不只汇报最大提升。',
           '','## 3. 当前范围','','下表是已完成配置差值的范围，不是最终最佳方案，也不是显著性结论。单位为0%—25%预算曲线平均表现的百分点；不同任务不混算。',
           '','| 小模型 | 大模型 | 对照组 | 完整配置数 | 最小差值 | 最大差值 |','|---|---|---|---:|---:|---:|']
    names={'inputs':'输入信息','output_only_controls':'仅预测信息','hidden_positions':'隐藏层位置','hidden_combinations':'多层组合','dimensions':'表示大小','student_sizes':'学生尺寸','targets_and_imbalance':'目标与平衡策略','harm_cost':'改坏代价','nested_fraction':'训练数据量','hidden_group_shuffle':'打乱隐藏信息'}
    experts={'cub':'ResNet／CUB','glsim':'GLSim／CUB','nlvr2':'BEiT3／NLVR2','vilt':'ViLT／NLVR2','groundingdino':'GroundingDINO／gRefCOCO','instancevg':'InstanceVG／gRefCOCO','yolo26x':'YOLO26x／工地','rtdetr_x_fp32':'RT-DETR-X／工地'}
    for (expert,endpoint,group),values in sorted(by_group.items()):lines.append(f'| {experts[expert]} | {endpoint} | {names[group]} | {len(values)} | {min(values):+.3f} | {max(values):+.3f} |')
    lines+=['','CUB既有小模型的验证图片重叠限制仍适用。这里的配对差值不能消除该限制。','','## 4. 文件','','全部配置、逐种子差值及来源保存在远端探索目录 `paired_ablation_summary.json`；未完成对照不进入此表。']
    (work/'paired_ablation_summary.md').write_text('\n'.join(lines))
    print(json.dumps({'completed_comparisons':len(rows),'incomplete_started_groups':incomplete,'excluded_runs':len(excluded),'negative_comparisons':sum(x['paired_delta_mean']<0 for x in rows)}))
