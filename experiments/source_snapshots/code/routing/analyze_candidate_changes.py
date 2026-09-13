"""Replay fixed-threshold changes and deterministic case indices for plotted candidates."""
import json
import sys
from pathlib import Path
import numpy as np
root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(root/'code'))
from routing.evaluate_router import correctness,metric,fixed_threshold_curve,EVENTS
w=root/'outputs/router_exploration_25pct_20260910'
figures=json.loads((w/'report_figures/figure_data.json').read_text())
assert len(figures)==16
rows=[];cases=[]
for figure in figures:
    dest=w/'runs'/figure['run_id'];endpoint=figure['endpoint']
    cfg=json.loads((dest/'config.json').read_text())
    done=json.loads((dest/'completion.json').read_text())
    assert cfg['freeze_encoder'] and cfg['seed']==42 and cfg['fraction']==.25
    assert done['status']=='complete' and not done['smoke'] and cfg['expert']==figure['expert']
    records=[json.loads(x) for x in (w/'datasets'/cfg['expert']/'val/records.jsonl').read_text().splitlines()]
    scores=np.load(dest/(endpoint+'_scores.npy'))
    saved=json.loads((dest/(endpoint+'_thresholds.json')).read_text())
    assert saved==fixed_threshold_curve(records,endpoint,scores)
    point=next(p for p in saved['curve'] if p['budget']==.25)
    op=point['operator'];threshold=point['threshold']
    if op=='never':chosen=np.zeros(len(records),dtype=bool)
    elif op=='always':chosen=np.ones(len(records),dtype=bool)
    elif op=='ge':chosen=scores>=threshold
    else:
        assert op=='gt'
        chosen=scores>threshold
    s,l=correctness(records,endpoint);after=np.where(chosen[:,None],l,s)
    transition=[[int(((s==a)&(after==b)).sum()) for b in [0,1]] for a in [0,1]]
    assert sum(map(sum,transition))==s.size
    assert transition[0][1]==point['rescue'] and transition[1][0]==point['harm']
    before_metric=metric(records,endpoint,np.zeros(len(records),dtype=bool))
    after_metric=metric(records,endpoint,chosen)
    assert after_metric==point['metric']
    if cfg['task']!='construction':assert abs(after_metric-before_metric-(transition[0][1]-transition[1][0])/len(records))<1e-12
    event_confusions=[]
    if cfg['task']=='construction':
        for event in EVENTS:
            counts=[]
            for upgraded in [np.zeros(len(records),dtype=bool),chosen]:
                labels=[r['large_labels'][endpoint] if u else r['small_label'] for r,u in zip(records,upgraded)]
                target=np.array([r['events'][event]['target'] for r in labels],dtype=bool)
                positive=np.array([r['events'][event]['prediction'] is True for r in labels])
                counts.append({'tp':int((target&positive).sum()),'fp':int((~target&positive).sum()),'fn':int((target&~positive).sum()),'tn':int((~target&~positive).sum()),'invalid_predictions':sum(r['events'][event]['prediction'] is None for r in labels)})
            event_confusions.append({'event':event,'before':counts[0],'after':counts[1]})
    rows.append({'task':cfg['task'],'expert':cfg['expert'],'endpoint':endpoint,'run_id':cfg['id'],'rows':len(records),'unit':'event' if s.shape[1]>1 else 'sample','before_metric':before_metric,'after_metric':after_metric,'difference':after_metric-before_metric,'fixed_threshold':point,'correctness_transition_rows_before_columns_after':transition,'event_confusions':event_confusions})
    masks={'救回':chosen&((s==0)&(l==1)).any(1),'改坏':chosen&((s==1)&(l==0)).any(1),'漏掉可救回':~chosen&((s==0)&(l==1)).any(1)}
    for kind,mask in masks.items():
        eligible=np.flatnonzero(mask)
        if not len(eligible):
            cases.append({'expert':cfg['expert'],'endpoint':endpoint,'type':kind,'available':False})
            continue
        idx=min(eligible,key=lambda i:records[i]['sample_id']);record=records[idx]
        cases.append({'expert':cfg['expert'],'task':cfg['task'],'endpoint':endpoint,'run_id':cfg['id'],'type':kind,'available':True,'sample_id':record['sample_id'],'row_index':int(idx),'score':float(scores[idx]),'upgraded':bool(chosen[idx]),'small_correct':s[idx].tolist(),'large_correct':l[idx].tolist(),'record':record})
report={'scope':'Same 16 frozen candidates as current plots; validation-selected fixed threshold at maximum 25% budget. Correctness transition matrix is NOT a category confusion matrix. Construction has separate event presence confusion counts; invalid predictions counted explicitly and remain unmodified.','rows':rows,'cases':cases,'case_selection':'Lexically first sample ID in each rescue/harm/missed-rescue group per pair; construction images may belong to multiple groups because events can change differently. This is a case index, not a visually reviewed case study.','new_training':False}
(w/'candidate_changes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
names={'cub':'ResNet','glsim':'GLSim','yolo26x':'YOLO26x','rtdetr_x_fp32':'RT-DETR-X','groundingdino':'GroundingDINO','instancevg':'InstanceVG','nlvr2':'NLVR2专用模型','vilt':'ViLT'}
text='# 当前冻结候选：救回、改坏与错误变化\n\n## 1、做了什么\n\n对报告16张曲线的同一批候选，用验证集确定的25%预算固定阈值重新统计。核对阈值、调用数量和指标与原结果完全一致。没有训练或新模型调用。\n\n## 2、为什么做\n\n分清哪些收益来自救回，哪些损失来自改坏。固定阈值遇到同分时可能少于25%调用，不能与强制排序取25%的曲线混用。\n\n## 3、结果\n\n| 专用模型 | 大模型 | 计数单位 | 实际调用 | 错误仍错误 | 救回 | 改坏 | 正确仍正确 | 指标变化×100 |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
for row in rows:
    m=row['correctness_transition_rows_before_columns_after']
    text+=f"| {names[row['expert']]} | {row['endpoint']} | {'事件' if row['unit']=='event' else '样本'} | {100*row['fixed_threshold']['actual_fraction']:.2f}% | {m[0][0]} | {m[0][1]} | {m[1][0]} | {m[1][1]} | {100*row['difference']:+.3f} |\n"
negative=sum(r['difference']<0 for r in rows)
text+=f'\n有{negative}个配对在此固定阈值下低于完全不调用大模型。负结果保留；不能因为优于随机调用就宣称优于不调用。工地指标为事件宏平均F1，不等于净救回除以图片数。表格是正确/错误状态迁移，不是类别混淆矩阵。\n\n已按样本编号顺序选取救回、改坏、漏救回的案例索引，不根据图片好不好看挑选；尚未核对图片并写成案例分析。工地同一图片可能同时包含救回与改坏事件。\n\n## 4、文件在哪里\n\n完整精确数值、工地各事件混淆计数、原始案例记录：`'+str(w/'candidate_changes.json')+'`。候选仍为验证探索结果，不是最终方法或独立测试；既有CUB专用模型见过内部验证图的限制仍然存在。\n'
(w/'candidate_changes.md').write_text(text)
print(json.dumps({'pairs':len(rows),'negative_pairs':negative,'available_cases':sum(c['available'] for c in cases),'checks':'passed','new_training':False}))
