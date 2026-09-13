"""Paired, seed-matched summaries; never rank methods on incomplete task sets."""
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
import csv
import json
import time

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/ltd_fixed_backbone_20260913'
REF=ROOT/'outputs/objective_multiseed_20260912/runs'
groups=json.loads((HERE/'source/main_table_reference.json').read_text())['groups']
by_pair={tuple(g['key'][1:]):g for g in groups}
plan=json.loads((OUT/'job_plan.json').read_text())
rows=[]
for cfg in plan:
    path=OUT/'runs'/cfg['id']/'test_evaluation.json'
    if not path.exists():continue
    result=json.loads(path.read_text())['comparisons']['ltd_score']
    old=json.loads((REF/f"repeat_{cfg['expert']}_{cfg['endpoint']}_four_difference_s{cfg['seed']}"/'test_evaluation.json').read_text())['comparisons']['probability_difference']
    conf=by_pair[(cfg['expert'],cfg['endpoint'])]['methods']['confidence']
    curve=conf['test_curve']
    conf25=sum((curve[i]['metric']+curve[i+5]['metric'])*.5*.05 for i in range(0,25,5))/.25
    done=json.loads((path.parent/'completion.json').read_text())
    rows.append({'task':cfg['task'],'expert':cfg['expert'],'endpoint':cfg['endpoint'],'seed':cfg['seed'],
        'method':cfg['method'],'call_cost':cfg['call_cost'],'validation_score25':result['validation_score25']*100,
        'test_fixed_score25':result['test_fixed_score25']*100,'test_exact_score25':result['test_ranked_score25']*100,
        'test_selected_metric':result['test_selected']['metric']*100,'test_selected_calls':result['test_selected']['actual_fraction']*100,
        'ours_fixed_score25':old['test_fixed_score25']*100,'ours_exact_score25':old['test_ranked_score25']*100,
        'ours_selected_metric':old['test_selected']['metric']*100,'ours_selected_calls':old['test_selected']['actual_fraction']*100,
        'confidence_fixed_score25':conf25*100,'confidence_selected_metric':conf['test_point']['metric']*100,
        'confidence_selected_calls':conf['test_point']['actual_fraction']*100,
        'delta_fixed25_vs_ours':(result['test_fixed_score25']-old['test_fixed_score25'])*100,
        'delta_exact25_vs_ours':(result['test_ranked_score25']-old['test_ranked_score25'])*100,
        'epochs':done['epochs'],'fit_seconds':done['seconds']})
if not rows:raise RuntimeError('No completed evaluations yet')
with (OUT/'per_run.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pair_groups=defaultdict(list)
for r in rows:pair_groups[(r['task'],r['expert'],r['endpoint'],r['method'])].append(r)
metrics=['test_fixed_score25','test_exact_score25','test_selected_metric','test_selected_calls',
         'ours_fixed_score25','ours_selected_metric','ours_selected_calls','delta_fixed25_vs_ours']
pair_rows=[]
for (task,expert,endpoint,method),items in pair_groups.items():
    row=dict(task=task,expert=expert,endpoint=endpoint,method=method,seeds=len(items))
    for key in metrics:
        row[key]=mean(r[key] for r in items)
        row[key+'_std']=stdev(r[key] for r in items) if len(items)>1 else None
    pair_rows.append(row)
with (OUT/'pair_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(pair_rows[0]));w.writeheader();w.writerows(pair_rows)
complete_seed_method=[]
for method in sorted(set(r['method'] for r in rows)):
    for seed in range(2026,2031):
        items=[r for r in rows if r['seed']==seed and r['method']==method]
        if len(items)!=16:continue
        summary={'method':method,'seed':seed}
        for key in ['validation_score25']+metrics:
            summary[key]=mean(mean(r[key] for r in items if r['task']==task) for task in sorted(set(r['task'] for r in items)))
        complete_seed_method.append(summary)
method_rows=[]
for method in sorted(set(r['method'] for r in complete_seed_method)):
    items=[r for r in complete_seed_method if r['method']==method]
    row={'method':method,'complete_seeds':len(items)}
    for key in ['validation_score25']+metrics:
        row[key]=mean(r[key] for r in items)
        row[key+'_std']=stdev(r[key] for r in items) if len(items)>1 else None
    method_rows.append(row)
status={'finished':len(rows),'planned':len(plan),'time':time.time(),'methods_complete_across_16_pairs':method_rows}
(OUT/'summary.json').write_text(json.dumps(status,indent=2))
text=['# 固定主干的转交方法比较：运行结果','',f'已完成 {len(rows)}/{len(plan)} 组训练及测试。未完成前只作阶段性判断。',
      '', '## 1．做了什么','', '在相同数据、冻结模型、路由器主干和训练规则下比较两动作分类损失与事后转交损失。',
      '', '## 2．为什么做','', '检验四状态训练的实际收益；分类损失均为固定动作适配版，不是原始分类器联合训练。',
      '', '## 3．当前结果','', '下面总表只纳入已完成全部16个配对的种子。分数是原协议下0至25%调用预算的测试平均表现，数值越大越好。',
      '', '| 方法 | 完整种子数 | 固定阈值分数 | 同种子四状态分数 | 差值 | 等调用分数 |', '|---|---:|---:|---:|---:|---:|']
names={'css':'两动作CSS','ova':'两动作OvA','posthoc_c0':'事后转交，代价0','posthoc_c005':'事后转交，代价0.05',
       'posthoc_c01':'事后转交，代价0.1','posthoc_c02':'事后转交，代价0.2','posthoc_c04':'事后转交，代价0.4'}
for row in method_rows:
    text.append(f"| {names[row['method']]} | {row['complete_seeds']} | {row['test_fixed_score25']:.3f} | {row['ours_fixed_score25']:.3f} | {row['delta_fixed25_vs_ours']:+.3f} | {row['test_exact_score25']:.3f} |")
if not method_rows:text.extend(['','暂时没有覆盖全部16个配对的完整方法结果，因此不计算混合总分。'])
text.extend(['','各配对、各种子明细见 per_run.csv；配对平均与标准差见 pair_summary.csv。单一种子的标准差留空。',
             '', '## 4．文件在哪','',str(OUT),'','实验设置见 exp_settings.md；逐组检查点、训练日志与测试曲线在 runs。'])
(OUT/'结果报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps(status),flush=True)
