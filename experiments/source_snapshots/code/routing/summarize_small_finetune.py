"""Matched frozen versus eight authorized end-to-end probes; validation only."""
import json
from pathlib import Path
w=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/router_exploration_25pct_20260910')
design=json.loads((w/'small_finetune_design.json').read_text())
analysis=json.loads((w/'small_finetune_queue_analysis.json').read_text())
assert not analysis['warnings']
verified={(r['id'],r['endpoint']) for r in analysis['checks']}
rows=[];pending=[]
for p in design['pairs']:
    dest=w/'runs'/p['run_id'];ref=w/'runs'/p['reference_id']
    if not (dest/'completion.json').exists():pending.append(p['run_id']);continue
    result=json.loads((dest/'completion.json').read_text())
    endpoint=p['endpoint_for_selection']
    if (p['run_id'],endpoint) not in verified:pending.append(p['run_id']);continue
    assert (p['reference_id'],endpoint) in verified
    base=json.loads((ref/'completion.json').read_text())
    assert result['encoder_trainable_parameters']>0 and not base['encoder_trainable_parameters']
    delta=result['best_selection'][endpoint]-base['best_selection'][endpoint]
    rows.append(dict(p,frozen_score=base['best_selection'][endpoint],finetune_score=result['best_selection'][endpoint],delta=delta,epochs=result['epochs'],seconds=result['seconds'],encoder_trainable_parameters=result['encoder_trainable_parameters']))
names={'cub':'CUB鸟类分类','construction':'工地事件','grefcoco':'gRefCOCO指代表达','nlvr2':'NLVR2图文推理'}
lines=['# 原图端到端微调与冻结参照','','## 做了什么','','只对已完成且分数/曲线复核通过的微调，与同配置、同种子42的冻结参照比较。每任务两个编码器学习率，总计8项；未完成项不参与结论。','','| 任务 | 编码器学习率 | 冻结曲线面积×100 | 微调曲线面积×100 | 差值 | 轮数 | 训练循环秒数 |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {names[r['task']]} | {r['encoder_lr']:g} | {r['frozen_score']*100:.3f} | {r['finetune_score']*100:.3f} | {r['delta']*100:+.3f} | {r['epochs']} | {r['seconds']:.1f} |")
lines+=['','## 为什么做与如何理解','','这些配置确实更新图像编码器，而不是重新蒸馏。比较的是验证集0至25%调用预算曲线面积，不是25%单点成绩或独立测试效果。正差值说明该次验证比较更高，不代表统计显著；负差值照常保留。任务原生指标不同，不跨任务平均差值。训练循环计时不含前期缓存加载，不等于端到端推理速度。','','尚未完成或尚未复核：'+str(len(pending))+'项。','','## 文件在哪里','',str(w/'small_finetune_paired_summary.json'),'']
(w/'small_finetune_paired_summary.json').write_text(json.dumps({'rows':rows,'pending_ids':pending,'all_eight_reviewed':len(rows)==8,'validation_only':True},indent=2))
(w/'small_finetune_paired_summary.md').write_text('\n'.join(lines))
print(json.dumps({'reviewed':len(rows),'pending':len(pending),'positive':sum(r['delta']>0 for r in rows),'negative':sum(r['delta']<0 for r in rows)}))
