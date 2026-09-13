"""Cross-task setting selection from the fixed initial grid; no new training."""
import json
from pathlib import Path
from statistics import mean

root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
w=root/'outputs/router_exploration_25pct_20260910'
queue=json.loads((w/'initial_queue.json').read_text())
tasks=['cub','construction','grefcoco','nlvr2']
labels={'cub':'鸟类分类','construction':'工地事件','grefcoco':'指代表达','nlvr2':'图文推理'}
expert_labels={'cub':'ResNet','glsim':'GLSim','yolo26x':'YOLO26x','rtdetr_x_fp32':'RT-DETR-X','groundingdino':'GroundingDINO','instancevg':'InstanceVG','nlvr2':'NLVR2专用模型','vilt':'ViLT'}
setting_labels={('concat','error'):'拼接＋预测小模型错误',('concat','gain'):'拼接＋预测收益',('concat','four_state'):'拼接＋四状态分类',('attention','error'):'注意力融合＋预测小模型错误',('attention','gain'):'注意力融合＋预测收益',('attention','four_state'):'注意力融合＋四状态分类'}
pairs={}
base_by_expert={}
for cfg in queue:
    if cfg['seed']!=42:continue
    assert cfg['freeze_encoder'] and cfg['fraction']==.25
    common={k:v for k,v in cfg.items() if k not in ['id','endpoint','target','fusion']}
    if cfg['expert'] in base_by_expert:assert base_by_expert[cfg['expert']]==common
    else:base_by_expert[cfg['expert']]=common
    dest=w/'runs'/cfg['id']
    done=json.loads((dest/'completion.json').read_text())
    assert done['status']=='complete' and not done['smoke']
    assert json.loads((dest/'config.json').read_text())==cfg
    setting=(cfg['fusion'],cfg['target'])
    for endpoint in (['Qwen3.8','MiniCPM'] if cfg['endpoint']=='both' else [cfg['endpoint']]):
        metrics=json.loads((dest/(endpoint+'_metrics.json')).read_text())
        score=metrics['selection_score']
        assert score==done['best_selection'][endpoint]
        pair=(cfg['task'],cfg['expert'],endpoint)
        assert setting not in pairs.setdefault(pair,{})
        pairs[pair][setting]={'score':score,'run_id':cfg['id']}
assert len(pairs)==16
settings=sorted(setting_labels)
for candidates in pairs.values():assert set(candidates)==set(settings)
# Mean tied ranks, computed separately per expert/large-model pair.
rank={}
for pair,candidates in pairs.items():
    rank[pair]={s:1+sum(v['score']>candidates[s]['score'] for v in candidates.values())+.5*(sum(v['score']==candidates[s]['score'] for v in candidates.values())-1) for s in settings}
task_rank={task:{s:mean(rank[p][s] for p in pairs if p[0]==task) for s in settings} for task in tasks}
rows=[];selections=[]
for heldout in tasks:
    source_rank={s:mean(task_rank[t][s] for t in tasks if t!=heldout) for s in settings}
    selected=min(settings,key=lambda s:(source_rank[s],s))
    selections.append({'heldout_task':heldout,'fusion':selected[0],'target':selected[1],'source_mean_rank':source_rank[selected],'candidate_source_ranks':[{'fusion':s[0],'target':s[1],'mean_rank':source_rank[s]} for s in settings]})
    for pair,candidates in sorted(pairs.items()):
        if pair[0]!=heldout:continue
        best=min(settings,key=lambda s:(-candidates[s]['score'],s))
        shared=candidates[selected];oracle=candidates[best]
        rows.append({'task':heldout,'expert':pair[1],'endpoint':pair[2],'selected_setting':setting_labels[selected],'run_id':shared['run_id'],'shared_score':shared['score'],'task_selected_score':oracle['score'],'task_selected_run_id':oracle['run_id'],'difference':shared['score']-oracle['score'],'relative_difference_percent':100*(shared['score']/oracle['score']-1) if oracle['score'] else None})
report={'scope':'Setting transfer only: six preplanned initial configurations, seed42, frozen 4M student, 25% task-specific supervised training. Each held-out task is excluded from setting selection but not student pretraining or router training. Initial expert-specific hidden-layer choices remain fixed. Not zero-shot or independent test generalization. No adaptive followup candidates used.','new_training':False,'selection':'Mean tied rank within each expert/endpoint, then equal task average over three source tasks; deterministic lexical tie break. Target comparator is best of the SAME SIX settings, not best of all exploration.','selections':selections,'rows':rows,'source_runs':sorted({v['run_id'] for p in pairs.values() for v in p.values()}),'status':'complete'}
(w/'shared_settings_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
text='# 统一设置跨任务比较\n\n## 1、做了什么\n\n复用初筛中固定的6种设置：两种融合方式×三种训练目标，学生冻结为4M，种子42，各任务25%训练数据。没有新增训练。每次用另外三个任务选设置，再查看剩下一个任务的验证结果。\n\n'
text+='## 2、为什么这样做\n\n检验设置能否通用，而不是每个任务单独选设置。先在各模型配对内部比较名次，再让三个任务等权参与选择，不把不同任务的指标直接平均。每个专用模型的初始隐藏层位置保持不变，没有检验统一层位置。\n\n'
text+='## 3、结果\n\n| 留出的任务 | 另外三任务选出的设置 | 来源任务平均名次（越小越好） |\n|---|---|---:|\n'
for row in selections:text+=f"| {labels[row['heldout_task']]} | {setting_labels[(row['fusion'],row['target'])]} | {row['source_mean_rank']:.3f} |\n"
text+='\n| 任务 | 专用模型 | 大模型 | 统一设置面积×100 | 本任务六选一面积×100 | 差值 |\n|---|---|---|---:|---:|---:|\n'
for row in rows:text+=f"| {labels[row['task']]} | {expert_labels[row['expert']]} | {row['endpoint']} | {100*row['shared_score']:.3f} | {100*row['task_selected_score']:.3f} | {100*row['difference']:+.3f} |\n"
text+='\n观察：图文推理四个配对的差距为0至0.213，鸟类分类为1.700至4.058（面积×100单位）。这说明本次设置迁移的损失因任务而异，尚不能认定存在一套通用最优设置。参照是本任务六选一的最好结果，因此差值按定义不大于零；负差值本身不是模型失效证据。后续应补充更大共同候选范围的比较，并保留任务差异，不将六种初筛设置代表所有方案。\n'
text+='\n限制：这里的参照只是在同样6种设置中选最好，不是全部探索中的最好。验证集参与训练轮次选择；学生预训练也没有排除留出任务。因此不能称为未见任务泛化或独立测试。单种子不做显著性声明。若希望检验更大候选范围，需要另行明确共同候选范围，不将本结果冒充完整方法泛化。\n\n## 4、文件在哪里\n\n原始来源编号、全部候选名次和精确数值：`'+str(w/'shared_settings_analysis.json')+'`。\n'
(w/'shared_settings_analysis.md').write_text(text)
print(json.dumps({'pairs':len(rows),'reused_fits':len(report['source_runs']),'selections':selections},ensure_ascii=False))
