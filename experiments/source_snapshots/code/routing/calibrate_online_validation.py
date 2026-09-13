"""Calibrate on first online validation pass; verify unchanged thresholds on second."""
import json,sys
from pathlib import Path
import numpy as np
root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(root/'code'))
from routing.evaluate_router import fixed_threshold_curve,metric
w=root/'outputs/router_exploration_25pct_20260910'
first=w/'router_timing_audit';second=w/'router_timing_audit_repeat'
assert (first/'master.exit').read_text().strip()=='0'
assert (second/'master.exit').read_text().strip()=='0'
dest=w/'online_validation_calibration';dest.mkdir(exist_ok=True)
rows=[]
for worker in range(4):
    report=json.loads((first/f'worker{worker}.json').read_text())
    assert report['status']=='complete'
    for row in report['rows']:
        cid=row['run_id'];endpoint=row['endpoint'];filename=cid+'_'+endpoint+'_online_scores.npy'
        a=np.load(first/filename);b=np.load(second/filename)
        records=[json.loads(x) for x in (w/'datasets'/row['expert']/'val/records.jsonl').read_text().splitlines()]
        calibrated=fixed_threshold_curve(records,endpoint,a)
        checks=[]
        for point in calibrated['curve']:
            masks=[]
            for values in [a,b]:
                if point['operator']=='never':mask=np.zeros(len(values),dtype=bool)
                elif point['operator']=='always':mask=np.ones(len(values),dtype=bool)
                elif point['operator']=='ge':mask=values>=point['threshold']
                else:
                    assert point['operator']=='gt'
                    mask=values>point['threshold']
                masks.append(mask)
            checks.append({'budget':point['budget'],'decision_changes':int((masks[0]!=masks[1]).sum()),'repeat_calls':int(masks[1].sum()),'within_validation_budget':bool(masks[1].sum()<=int(np.floor(len(records)*point['budget']))),'repeat_metric':metric(records,endpoint,masks[1])})
        calibrated['runtime']='Same A800, resized-pixel input, batch16, bfloat16 encoder then float16 features, unchanged saved router; same validation order.'
        (dest/(cid+'_'+endpoint+'_thresholds.json')).write_text(json.dumps(calibrated,indent=2))
        rows.append({'run_id':cid,'expert':row['expert'],'endpoint':endpoint,'max_repeat_score_difference':float(np.max(np.abs(a-b))),'score_exact_repeat':bool(np.array_equal(a,b)),'checks':checks,'previous_cached_threshold_audit':row['fixed_threshold_audit'],'calibrated_budget25':next(p for p in calibrated['curve'] if p['budget']==.25)})
assert len(rows)==6
passed=all(c['decision_changes']==0 and c['within_validation_budget'] for row in rows for c in row['checks'])
result={'scope':'Six image/text candidates, recalibrated using first online validation pass and verified on second forward pass of the SAME validation data. This is runtime repeatability, not independent validation/test performance. Original cached thresholds retained. No tolerance added. Future data/order/batch/hardware can change scores and call fraction; a fixed threshold is not a hard future budget guarantee.','rows':rows,'status':'passed' if passed else 'requires_investigation','new_training':False}
(dest/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'candidates':len(rows),'passed':passed,'exact_repeats':sum(r['score_exact_repeat'] for r in rows),'budget25':[{'expert':r['expert'],'endpoint':r['endpoint'],'metric':r['calibrated_budget25']['metric'],'calls_fraction':r['calibrated_budget25']['actual_fraction']} for r in rows]}))
