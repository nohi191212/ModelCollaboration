from pathlib import Path
from collections import defaultdict
from statistics import mean,stdev
import csv
import json

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
OUT=ROOT/'outputs/ltd_kclass_native_20260913'
OLD=ROOT/'outputs/objective_multiseed_20260912/runs'
plan=json.loads((OUT/'job_plan.json').read_text())
rows=[]
for cfg in plan:
    path=OUT/'runs'/cfg['id']/'test_evaluation.json'
    if not path.exists():continue
    result=json.loads(path.read_text());r=result['comparisons']['native_probability_difference']
    ref=json.loads((OLD/f"repeat_{cfg['expert']}_{cfg['endpoint']}_four_difference_s{cfg['seed']}"/'test_evaluation.json').read_text())
    prior=ref['comparisons']['probability_difference']
    rows.append({'task':cfg['task'],'expert':cfg['expert'],'endpoint':cfg['endpoint'],'seed':cfg['seed'],'method':cfg['native_method'],
        'new_classifier_alone':result['small_test']*100,'old_specialist_alone':ref['small_test']*100,'large_alone':result['large_test']*100,
        'test_fixed25':r['test_fixed_score25']*100,'test_exact25':r['test_ranked_score25']*100,
        'test_selected':r['test_selected']['metric']*100,'test_calls':r['test_selected']['actual_fraction']*100,
        'original_zero_threshold_metric':result['native_zero_threshold']['test']['metric']*100,
        'original_zero_threshold_calls':result['native_zero_threshold']['test']['calls']*100,
        'ours_fixed25':prior['test_fixed_score25']*100,'ours_exact25':prior['test_ranked_score25']*100,
        'ours_selected':prior['test_selected']['metric']*100,'ours_calls':prior['test_selected']['actual_fraction']*100})
if rows:
    with (OUT/'per_run.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics=['new_classifier_alone','old_specialist_alone','large_alone','test_fixed25','test_exact25','test_selected','test_calls',
         'original_zero_threshold_metric','original_zero_threshold_calls','ours_fixed25','ours_exact25','ours_selected','ours_calls']
summary=[]
for task in ['cub','nlvr2']:
    for method in ['ova','css']:
        complete=[]
        for seed in range(2026,2031):
            rr=[r for r in rows if (r['task'],r['method'],r['seed'])==(task,method,seed)]
            if len(rr)==4:complete.append({k:mean(r[k] for r in rr) for k in metrics})
        if not complete:continue
        row={'task':task,'method':method,'complete_seeds':len(complete)}
        for key in metrics:
            row[key]=mean(r[key] for r in complete)
            row[key+'_std']=stdev(r[key] for r in complete) if len(complete)>1 else None
        summary.append(row)
state={'completed':len(rows),'total':len(plan),'finished':(OUT/'complete.json').exists(),'task_means':summary}
(OUT/'summary.json').write_text(json.dumps(state,indent=2))
names={'cub':'鸟类分类','nlvr2':'双图推理'}
text=['# K类别与转交规则实验结果','', '## 1．做了什么','',f'完成{len(rows)}/{len(plan)}组。保留真实K类别输出和专家输出，未转交时由类别分支预测答案。',
      '', '## 2．为什么做','', '纠正此前只预测两个模型正确率的适配。本次复现OvA/CSS的分类损失与原始答案规则，采用我们的冻结特征与路由主干。',
      '', '## 3．结果','', '下表只纳入该任务全部4个模型配对完成的种子。先看新分类器与原专用模型的差别，再看协作成绩。','',
      '| 任务 | 方法 | 完整种子数 | 新分类器单独 | 原专用模型单独 | 固定阈值低预算 | 同任务四状态 | 选定阈值成绩 | 调用率 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:
    text.append(f"| {names[r['task']]} | {r['method']} | {r['complete_seeds']} | {r['new_classifier_alone']:.3f} | {r['old_specialist_alone']:.3f} | {r['test_fixed25']:.3f} | {r['ours_fixed25']:.3f} | {r['test_selected']:.3f} | {r['test_calls']:.2f}% |")
text.extend(['','新方法与四状态的基础答案不同，分数差不能全部解释为路由收益。原始零阈值成绩和等调用成绩见per_run.csv；标准差见summary.json。',
             '', '定位任务没有固定K类输出，施工为多标签，不强行纳入此单标签实验。',
             '', '## 4．文件在哪','',str(OUT),'','实验设置exp_settings.md，逐组日志、检查点和曲线在runs。'])
(OUT/'结果报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps(state),flush=True)
