"""Render an explicitly unfinished, evidence-linked HTML/Markdown report."""
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import markdown
root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
w=root/'outputs/router_exploration_25pct_20260910'
p=json.loads((w/'progress_snapshot.json').read_text())
light=p['interaction_light'];stamp=datetime.fromtimestamp(p['unix_time'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
small_completed=len(list((w/'runs').glob('finetune_small_*/completion.json')))
text=f'''# 25%数据探索报告 · 未完成草稿

数据快照：{stamp}（北京时间）。本文件不是最终交付，也不是测试集成绩。

## 1、目前做了什么

| 部分 | 实际状态 |
|---|---|
| 第一批初筛 | 240项训练完成 |
| 第二批单因素消融 | 3101项新增训练完成，完整结果已复核 |
| 第三批轻量组合 | {light['completed']} / {light['planned']}项完成 |
| 代表性原图微调 | {small_completed}/8项正式训练完成 |
| 新增学生蒸馏与深度对照 | 按用户要求暂缓 |
| 依赖新学生的严格任务留出 | 暂缓，不能称为已验证的泛化 |
| 路由器成本、图表、案例与统一设置复核 | 已完成；整套端到端成本与严格留出泛化未完成 |
| 最后占用四张显卡 | 未执行，不能提前执行 |

## 2、为什么这样做

先用缓存表示筛选路由输入与训练目标，再用8项代表性原图微调检查是否值得联合更新编码器。当前不重新蒸馏学生，不将所有配置展开多个随机种子。

冻结表示实验有真实路由标签监督，但冻结学生编码器，只更新输入投影、归一化和融合后的判断网络。带图片分支的配置使用原图编码后的缓存向量；没有图片分支的对照不使用图像信息。8项微调读取缓存的原始像素、更新学生编码器，不是教师蒸馏；完成数量见上表。

### 固定实验设置

| 设置 | 当前执行 |
|---|---|
| 路由训练数据 | 固定25%来源组；另有嵌套10%诊断，不扩展到50%或100% |
| 验证 | 使用完整固定验证集；不依据测试成绩选方法 |
| 最多轮数与停止条件 | 50轮；验证选择指标连续8轮不提高则停止 |
| 每批数据与优化器 | 256条，AdamW，学习率0.001，权重衰减0.01 |
| 随机种子 | 前两批已有多种子结果保留；之后固定42 |
| 主要选择指标 | 大模型调用预算0至25%区间的原生指标曲线归一化面积 |
| 轻量并行 | 每张卡最多4项，四张卡共16项 |

训练耗时不等于线上推理耗时。缓存表示实验不包含重新调用大小模型；读取缓存之前的初始化时间也不在训练循环计时中。路由器判断头和现场编码路径已单独计时，但整套端到端成本（含大小模型和网络）尚未实测。

## 3、目前结果与限制

下面引用第二批和第三批完整复核结果。验证集同时用于筛选与报告，不能解释为独立泛化效果，不能用单种子声称统计显著。

工地任务使用事件存在性的宏平均F1，不是框定位指标；其他任务使用当前严格逐样本正确性定义，不跨任务混合百分数。工地救回和改坏按事件计数，不能直接除以图片数解释F1变化。

CUB既有ResNet训练见过当前内部验证图片；GLSim训练来源尚未独立确认。严格重蒸馏学生排除了当前验证数据，但这不能消除既有专用模型带来的限制。NLVR2嵌套10%来源组实际占14.47%的标签行，不能称为严格10%标签。

负结果保留；高于随机调用或置信规则并不一定高于完全不调用大模型。排序预算曲线遇到同分时，与固定阈值实际调用比例可能不同。

'''
for title,name in [('第二批完整复核汇总','followup_queue_analysis.md'),('第二批成对消融','paired_ablation_summary.md'),('第二批救回与改坏','second_stage_rescue_harm.md'),('第三批完整复核汇总','interaction_light_queue_analysis.md')]:
    source=(w/name).read_text()
    source='\n'.join('##'+line if line.startswith('#') else line for line in source.splitlines())
    text+='\n## '+title+'\n\n'+source+'\n'
for old,new in [('groundingdino','GroundingDINO'),('rtdetr_x_fp32','RT-DETR-X'),('instancevg','InstanceVG'),('yolo26x','YOLO26x'),('glsim','GLSim'),('vilt','ViLT')]:
    text=text.replace('| '+old+' |','| '+new+' |')
if (w/'small_finetune_paired_summary.md').exists():
    text+='\n\n'+(w/'small_finetune_paired_summary.md').read_text()
if (w/'shared_settings_analysis.md').exists():
    text+='\n\n'+(w/'shared_settings_analysis.md').read_text()
if (w/'candidate_changes.md').exists():
    text+='\n\n'+(w/'candidate_changes.md').read_text()
if (w/'case_study.md').exists():
    text+='\n\n'+(w/'case_study.md').read_text()
if (w/'router_timing_summary.md').exists():
    text+='\n\n'+(w/'router_timing_summary.md').read_text()
if (w/'provisional_candidates.md').exists():
    text+='\n\n'+(w/'provisional_candidates.md').read_text()
if (w/'report_figures/figure_data.json').exists():
    text+='\n\n## 完整轻量探索的候选预算曲线\n\n以下16张图覆盖全部模型配对，不将验证选择结果称为测试结果。\n'
    for figure in json.loads((w/'report_figures/figure_data.json').read_text()):
        text+='\n'+figure['caption']+'\n\n[查看原尺寸图片](report_figures/'+figure['name']+'.png)\n\n!['+figure['name']+'](report_figures/'+figure['name']+'.png)\n'
text+='\n## 4、文件与后续交付\n\n原始结果目录：`'+str(w)+'`。每项配置、逐轮历史、最佳权重、曲线和阈值保存在 `runs/`。\n\n当前已补齐路由器路径耗时、案例、错误变化、统一设置和暂定候选表。仍缺整套端到端成本、严格留出泛化训练及原始清单中暂停的学生蒸馏/深度对照；暂停项不等于完成，未经用户确认不得恢复。该文件是当前授权范围的阶段报告，不标记原始完整目标完成，不执行最后占卡。\n'
body=markdown.markdown(text,extensions=['tables','fenced_code'])
style='''body{margin:0;background:#f3f5f8;color:#17283b;font:16px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1160px;margin:36px auto;background:white;padding:44px 48px;border-radius:16px;box-shadow:0 8px 30px #16324f0c}h1{font-size:32px;color:#163e68}h2{margin-top:40px;border-bottom:2px solid #dce8f4;padding-bottom:10px}h3{color:#45617e}table{display:block;overflow:auto;border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{border:1px solid #dae3ed;padding:10px 14px;text-align:left}th{background:#eaf1f8;white-space:nowrap}tr:nth-child(even){background:#f8fafc}code{overflow-wrap:anywhere;font-size:13px;color:#234f7b}p{overflow-wrap:anywhere}.notice{background:#fff1cf;border-left:5px solid #da9b21;padding:16px 20px;border-radius:6px}@media(max-width:700px){main{margin:12px;padding:22px}h1{font-size:25px}}'''
(w/'exploration_report_draft.md').write_text(text)
(w/'exploration_report_draft.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>探索报告·未完成草稿</title><style>'+style+'img{max-width:100%;height:auto}</style><main><div class="notice">未完成草稿：实验汇总和验收仍在进行，不能作为最终结论。</div>'+body+'</main></html>')
stage_title=text.replace('# 25%数据探索报告 · 未完成草稿','# 25%数据探索阶段报告（当前授权范围）',1).replace('本文件不是最终交付，也不是测试集成绩。','本文件汇总当前已授权范围；暂停项单独列出，不是原始完整清单的终版，也不是测试集成绩。',1)
stage_body=markdown.markdown(stage_title,extensions=['tables','fenced_code'])
(w/'exploration_report_25pct.md').write_text(stage_title)
(w/'exploration_report_25pct.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>25%数据探索阶段报告</title><style>'+style+'img{max-width:100%;height:auto}</style><main><div class="notice">阶段报告：当前授权范围的训练与复核已汇总；暂停实验未执行，不能解释为原始完整清单全部完成。</div>'+stage_body+'</main></html>')
print(json.dumps({'draft_only':True,'stage_report':True,'snapshot':stamp,'html_bytes':(w/'exploration_report_25pct.html').stat().st_size,'goal_complete':False}))
