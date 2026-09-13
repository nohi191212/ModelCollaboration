"""Validation-only, no-training references for all fixed expert/large-model pairs."""
import json,sys
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(ROOT/'code'))
from routing.evaluate_router import correctness,evaluate,fixed_threshold_curve

if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    dest=work/'references'
    dest.mkdir(exist_ok=False)
    summary=[]
    for expert in ['cub','glsim','nlvr2','vilt','groundingdino','instancevg','yolo26x','rtdetr_x_fp32']:
        source=work/'datasets'/expert
        meta=json.loads((source/'completion.json').read_text())
        assert meta['status']=='complete'
        records=[json.loads(line) for line in (source/'val/records.jsonl').read_text().splitlines()]
        confidence=np.load(source/'val/confidence.npy')
        columns=meta['confidence_columns']
        column='max_probability' if 'max_probability' in columns else 'max_score'
        values=confidence[:,columns.index(column)]
        valid=confidence[:,columns.index(column+'_valid')].astype(bool)
        assert len(confidence)==len(records) and np.isfinite(confidence).all()
        assert ((values[valid]>=0)&(values[valid]<=1)).all()
        # Missing confidence is explicitly routed first, never interpreted as a probability.
        confidence_scores=np.where(valid,-values,1.)
        for endpoint in ['Qwen3.8','MiniCPM']:
            random=[]
            for seed in range(42,50):
                scores=np.random.default_rng(seed).random(len(records))
                random.append({'seed':seed,**evaluate(records,endpoint,scores)})
            confidence_result=evaluate(records,endpoint,confidence_scores)
            thresholds=fixed_threshold_curve(records,endpoint,confidence_scores)
            s,l=correctness(records,endpoint)
            # Diagnostic only: this uses labels, so it cannot be deployed or selected as a router.
            # For construction it is NOT a macro-F1 oracle, because F1 is non-additive.
            diagnostic=evaluate(records,endpoint,(l-s).mean(1))
            opportunity={'both_correct':int(((s==1)&(l==1)).sum()),
                         'rescuable':int(((s==0)&(l==1)).sum()),
                         'harmful':int(((s==1)&(l==0)).sum()),
                         'both_wrong':int(((s==0)&(l==0)).sum()),
                         'count_unit':'event' if s.shape[1]>1 else 'sample'}
            small=random[0]['curve'][0]['metric'];large=random[0]['curve'][-1]['metric']
            for result in random+[confidence_result,diagnostic]:
                assert result['curve'][0]['metric']==small and result['curve'][-1]['metric']==large
                if s.shape[1]==1:
                    for point in result['curve']:
                        assert abs(point['metric']-small-(point['rescue']-point['harm'])/len(records))<1e-10
            areas=np.array([r['selection_score'] for r in random])
            means=[]
            for i,point in enumerate(random[0]['curve']):
                metrics=np.array([r['curve'][i]['metric'] for r in random])
                means.append({'budget':point['budget'],'metric_mean':float(metrics.mean()),'metric_std':float(metrics.std(ddof=1))})
            row={'expert':expert,'task':meta['task'],'endpoint':endpoint,'validation_rows':len(records),
                 'metric_name':'macro_event_presence_f1' if s.shape[1]>1 else 'accuracy',
                 'small_metric':small,'large_metric':large,
                 'random_area_mean':float(areas.mean()),'random_area_std':float(areas.std(ddof=1)),
                 'confidence_area':confidence_result['selection_score'],
                 'confidence_delta_from_random':confidence_result['selection_score']-float(areas.mean()),
                 'confidence_valid_rows':int(valid.sum()),'opportunity':opportunity}
            report={**row,'random_runs':random,'random_curve_summary':means,
                    'confidence_curve':confidence_result,'confidence_thresholds':thresholds,
                    'confidence_rule':{'column':column,'direction':'lower confidence routes first','missing':'routes first; explicitly missing, not a probability'},
                    'label_informed_diagnostic':diagnostic,
                    'diagnostic_warning':'Uses ground truth. Not deployable. For event-F1 not an upper bound.',
                    'test_rows_used':0,'cub_warning':'ResNet previously trained on these 1200 validation images; GLSim training provenance unconfirmed.' if meta['task']=='cub' else None}
            (dest/(expert+'_'+endpoint+'.json')).write_text(json.dumps(report,indent=2))
            summary.append(row)
            print(json.dumps(row),flush=True)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    (dest/'completion.json').write_text(json.dumps({'status':'complete','pairs':len(summary),'random_seeds':list(range(42,50)),'test_rows_used':0},indent=2))
