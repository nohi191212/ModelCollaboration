"""Descriptive validation help/harm table for completed second-stage candidates."""
import json
from pathlib import Path
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    w=ROOT/'outputs/router_exploration_25pct_20260910'
    analysis=json.loads((w/'followup_queue_analysis.json').read_text())
    assert analysis['all_queue_fits_complete'] and not analysis['warnings']
    best={}
    for row in analysis['groups']:
        c=row['config']
        if c['fraction']!=.25 or c.get('hidden_shuffle',False):continue
        run=next(r for r in row['runs'] if r['seed']==42)
        key=(c['expert'],row['endpoint'])
        candidate=(run['score'],run['id'],row)
        if key not in best or (-candidate[0],candidate[1])<(-best[key][0],best[key][1]):best[key]=candidate
    results=[]
    for (expert,endpoint),(score,cid,row) in sorted(best.items()):
        dest=w/'runs'/cid
        ranked=json.loads((dest/(endpoint+'_metrics.json')).read_text())
        fixed=json.loads((dest/(endpoint+'_thresholds.json')).read_text())
        zero=ranked['curve'][0]['metric']
        rank=next(p for p in ranked['curve'] if p['budget']==.25)
        threshold=next(p for p in fixed['curve'] if p['budget']==.25)
        n=json.loads((dest/'training_rows.json').read_text())['validation_rows']
        if row['config']['task']!='construction':
            for p in [rank,threshold]:assert abs(p['metric']-zero-(p['rescue']-p['harm'])/n)<1e-12
        results.append({'expert':expert,'endpoint':endpoint,'run_id':cid,'task':row['config']['task'],'seed':42,'selection_area':score,'small_metric':zero,'count_unit':'event' if ranked['event_counts']>1 else 'sample','rank_budget25':rank,'fixed_budget25':threshold})
    assert len(results)==16
    (w/'second_stage_rescue_harm.json').write_text(json.dumps({'scope':'Completed second-stage seed42 winners, descriptive validation only; not final method; construction counts events and macro F1 is not net rescue divided by samples.','rows':results},indent=2))
    lines=['# 第二批候选：救回与改坏（验证集，非最终方法）','','## 做了什么与为什么','','对每个模型配对，按第二批种子42的0至25%曲线面积选一个候选，展示25%预算处的结果。用于解释效果来源，不是测试成绩，也不是从第三批未完成结果挑优。','','| 专用模型 | 后接大模型 | 排序曲线救回/改坏 | 固定阈值实际调用 | 固定阈值救回/改坏 | 固定阈值指标变化 |','|---|---|---:|---:|---:|---:|']
    for r in results:
        a=r['rank_budget25'];b=r['fixed_budget25']
        lines.append(f"| {r['expert']} | {r['endpoint']} | {a['rescue']}/{a['harm']} | {b['actual_fraction']*100:.2f}% | {b['rescue']}/{b['harm']} | {(b['metric']-r['small_metric'])*100:+.3f} |")
    lines+=['','## 如何解释','','排序曲线与固定阈值遇到同分时可能不同，固定阈值不保证刚好调用25%。指标变化为相对不调用大模型的百分点差。工地统计单位为事件，其原生指标为事件宏平均F1，不能用救回减改坏除以图片数解释F1变化。其他任务已核对净救回与正确率变化的一致性。','','这只是诊断表；出现改坏和负差值应如实保留。最终需结合后续完成结果、参数量和实际耗时选方案。CUB的既有专用模型训练见过内部验证图片的限制仍需披露。','','## 文件在哪里','',str(w/'second_stage_rescue_harm.json'),'']
    (w/'second_stage_rescue_harm.md').write_text('\n'.join(lines))
    print(json.dumps({'pairs':len(results),'negative_fixed_budget25':sum(r['fixed_budget25']['metric']<r['small_metric'] for r in results),'gpu_launched':False}))
