"""Persist provisional per-pair candidates from the complete light exploration."""
import json
from pathlib import Path
w=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/router_exploration_25pct_20260910')
figures=json.loads((w/'report_figures/figure_data.json').read_text())
assert len(figures)==16 and len({(x['expert'],x['endpoint']) for x in figures})==16
names={'cub':'ResNet','glsim':'GLSim','yolo26x':'YOLO26x','rtdetr_x_fp32':'RT-DETR-X','groundingdino':'GroundingDINO','instancevg':'InstanceVG','nlvr2':'NLVR2专用模型','vilt':'ViLT'}
rows=[]
for f in sorted(figures,key=lambda x:(x['expert'],x['endpoint'])):
    dest=w/'runs'/f['run_id'];cfg=json.loads((dest/'config.json').read_text());done=json.loads((dest/'completion.json').read_text())
    assert cfg['freeze_encoder'] and cfg['seed']==42 and done['status']=='complete' and not done['smoke']
    curve=f['curves']['本轮候选'];baseline=f['curves']['不调用大模型'];
    assert f['budgets']==[0,5,10,15,20,25]
    rows.append({'task':cfg['task'],'expert':f['expert'],'expert_display':names[f['expert']],'endpoint':f['endpoint'],'run_id':f['run_id'],'student':cfg['student'],'branches':cfg['branches'],'fusion':cfg['fusion'],'target':cfg['target'],'hidden_layer':cfg.get('hidden_layer'),'hidden_layers':cfg.get('hidden_layers'),'representation_dim':cfg['representation_dim'],'strategy':cfg['strategy'],'harm_cost':cfg['harm_cost'],'selection_area_0_25':float(done['best_selection'][f['endpoint']]),'curve_metric_0_25':[float(x) for x in curve],'no_upgrade_metric_0_25':[float(x) for x in baseline],'metric_at_25_delta':float(curve[-1]-baseline[-1]),'source':'complete interaction_light analysis, seed42; reused completed reference where applicable','selection_status':'provisional_validation_candidate'})
report={'scope':'Provisional candidate table for the 16 expert/large-model pairs. Chosen by highest seed42 normalized validation area over budgets 0,5,10,15,20,25% among complete interaction-light candidates, with reused completed references included. This is validation selection, not test performance or a single universal method; no new training. Cached-feature candidate scores and runtime-calibrated thresholds are kept separately.','rows':rows,'count':len(rows),'student_distillation_new':False,'final_method_claim':False}
(w/'provisional_candidates.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
text='# 16个模型配对的暂定候选配置\n\n## 1、选择规则\n\n从第三批轻量实验的完整结果中，对每个专用模型和大模型配对选取种子42下0%、5%、10%、15%、20%、25%预算曲线归一化面积最高的配置；完整的已完成参照可复用，不把未完成结果纳入。这里是验证集探索候选，不是独立测试结果，也不宣称所有任务共用一套设置。\n\n| 专用模型 | 大模型 | 学生 | 输入分支 | 融合 | 目标 | 隐藏层 | 维度 | 策略 | 0至25%面积 | 25%点相对不调用 | 运行编号 |\n|---|---|---|---|---|---|---|---:|---|---:|---:|---|\n'
for r in rows:
    hidden='/'.join(r['hidden_layers']) if r['hidden_layers'] else r['hidden_layer'] or '—'
    text+=f"| {r['expert_display']} | {r['endpoint']} | {r['student']} | {'+'.join(r['branches'])} | {r['fusion']} | {r['target']} | {hidden} | {r['representation_dim']} | {r['strategy']} | {100*r['selection_area_0_25']:.3f} | {100*r['metric_at_25_delta']:+.3f} | {r['run_id']} |\n"
text+='\n## 2、如何使用\n\n这张表只固定当前验证探索的候选，不能替代最终方案确认。缓存特征训练和现场重新编码的阈值校准分开保存；部署时要固定编码批量和阈值来源，不能把缓存分数直接当成现场分数。原图端到端微调的8项结果单独作为消融，不覆盖这张冻结候选表。新增学生蒸馏、深度和严格任务留出目前按用户要求暂停。\n\n## 3、文件\n\n完整精确配置、曲线和来源身份：`provisional_candidates.json`；原始权重与训练记录在远端 `runs/`。\n'
(w/'provisional_candidates.md').write_text(text)
print(json.dumps({'candidates':len(rows),'new_training':False,'final_method_claim':False}))
