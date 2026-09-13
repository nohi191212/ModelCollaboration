from pathlib import Path
import os
from collections import defaultdict
from statistics import mean
import csv
import json
import time
from run_study import OUT, ROOT

plan=[c for c in json.loads((OUT/'job_plan.json').read_text()) if c['ltd_method']=='posthoc']
pid=int((OUT/'posthoc_resume.pid').read_text())
while True:
    rows=[]
    for cfg in plan:
        path=OUT/'runs'/cfg['id']/'test_evaluation.json'
        if not path.exists():
            continue
        result=json.loads(path.read_text())['comparisons']['ltd_score']
        ref=ROOT/'outputs/objective_multiseed_20260912/runs'/f"repeat_{cfg['expert']}_{cfg['endpoint']}_four_difference_s{cfg['seed']}"/'test_evaluation.json'
        ours=json.loads(ref.read_text())['comparisons']['probability_difference']
        rows.append(dict(task=cfg['task'],expert=cfg['expert'],endpoint=cfg['endpoint'],seed=cfg['seed'],method=cfg['method'],call_cost=cfg['call_cost'],validation_score25=result['validation_score25']*100,test_fixed_score25=result['test_fixed_score25']*100,test_exact_score25=result['test_ranked_score25']*100,ours_fixed_score25=ours['test_fixed_score25']*100,selected_metric=result['test_selected']['metric']*100,selected_calls=result['test_selected']['actual_fraction']*100))
    with (OUT/'posthoc_per_run.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups=defaultdict(list)
    for row in rows:
        groups[(row['task'],row['method'])].append(row)
    table=['# 事后转交方法续跑结果','','1. 做了什么：只统计有效的事后转交实验，复用此前已完成结果。','2. 为什么做：分类输出修正不影响这一方法；旧两动作 CSS/OvA 不纳入本表。',f'3. 当前效果：已完成 {len(rows)}/400 组。下表仅列出四个配对、五个种子全部完成的任务和代价组合。','','| 任务 | 调用代价 | 组数 | 固定预算分数 | 我们的方法 | 差值 |','|---|---:|---:|---:|---:|---:|']
    for (task,method),items in sorted(groups.items()):
        if len(items)!=20:
            continue
        score=mean(r['test_fixed_score25'] for r in items)
        baseline=mean(r['ours_fixed_score25'] for r in items)
        table.append(f"| {task} | {items[0]['call_cost']} | 20 | {score:.3f} | {baseline:.3f} | {score-baseline:+.3f} |")
    table.extend(['','分数表示原协议下 0—25% 调用预算的平均测试表现。五种代价分别报告，不按测试结果挑选。','','4. 文件：同目录 posthoc_per_run.csv 为逐组明细，posthoc_resume.log 为续跑日志；原始结果保留在 runs。'])
    (OUT/'事后转交续跑报告.md').write_text('\n'.join(table)+'\n')
    (OUT/'posthoc_summary.json').write_text(json.dumps({'finished':len(rows),'planned':400,'time':time.time()},indent=2))
    if (OUT/'posthoc_complete.json').exists():
        assert len(rows)==400
        break
    if not Path(f'/proc/{pid}').exists() or Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':
        raise RuntimeError('Continuation exited before completing 400 fits; inspect posthoc_resume.log')
    time.sleep(60)
