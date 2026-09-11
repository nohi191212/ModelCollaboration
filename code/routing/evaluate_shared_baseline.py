"""Open test data only after the single shared configuration has been locked."""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True)
    a=ap.parse_args()
    w=a.work
    frozen=json.loads((w/'selected_config.json').read_text())
    state=json.loads((w/'DSE_STATE.json').read_text())
    if state['status']!='search_complete' or frozen['test_loaded_during_selection']:raise ValueError('Global selection is not locked')
    output=w/'test_results.json'
    if output.exists():raise FileExistsError(output)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from routing import train_router_formal as formal
    formal.ROOT=a.root; formal.WORK=a.root/'outputs/router_full_20260910'
    from evaluate_formal_dense_curves import label_arrays,score_metric,decision
    formal.BUDGETS=[i/100 for i in range(101)]
    torch.set_num_threads(2)
    device=torch.device('cuda')
    result={'shared_setting':frozen['selected']['settings'],'selection_protocol':'Shared validation-only coordinate search; test is evaluated only after selected_config.json is frozen',
            'threshold_protocol':'Per-pair validation best budget in 1..99%; test never selects thresholds','groups':[]}
    temp=w/'test_evaluation'
    temp.mkdir(exist_ok=True)
    for run in frozen['selected']['runs']:
        dest=w/'runs'/run
        cfg=json.loads((dest/'config.json').read_text())
        done=json.loads((dest/'completion.json').read_text())
        endpoint=cfg['endpoint']
        val_records=formal.read_rows(formal.WORK/'data'/cfg['expert']/'val/records.jsonl')
        val_scores=np.load(dest/'validation_scores.npy')
        if len(val_records)!=len(val_scores):raise ValueError(('validation identity length',run))
        thresholds=formal.validation_thresholds(val_records,endpoint,val_scores)
        val_arrays=label_arrays(val_records,endpoint)
        val_curve=[dict(p,actual_fraction=float(decision(val_scores,p).mean()),metric=score_metric(val_arrays,decision(val_scores,p))) for p in thresholds]
        best_index=max(range(1,100),key=lambda i:(val_curve[i]['metric'],-val_curve[i]['actual_fraction'],-i))
        # Threshold selection has now finished, before reading this pair's test data.
        records,data=formal.load_data(cfg,'test',device)
        for key in ['hidden','confidence']:
            norm=np.load(dest/(key+'_normalization.npz'))
            data[key]=(data[key]-torch.from_numpy(norm['mean']).to(device))/torch.from_numpy(norm['std']).to(device)
        model=formal.FeatureRouter(cfg,data['hidden'].shape[1],data['confidence'].shape[1],data['output'].shape[1]).to(device)
        checkpoint=torch.load(dest/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.inference_mode():
            raw=torch.cat([model(data,torch.arange(b,min(b+8192,len(records)),device=device)).cpu()
                           for b in range(0,len(records),8192)]).numpy()
        scores=raw[:,1]-cfg['harm_cost']*raw[:,2] if cfg['target']=='four_state' else 1/(1+np.exp(-raw[:,0])) if cfg['target']=='error' else raw[:,0]
        if not np.isfinite(scores).all():raise ValueError(('nonfinite test scores',run))
        arrays=label_arrays(records,endpoint)
        test_curve=[dict(p,actual_fraction=float(decision(scores,p).mean()),metric=score_metric(arrays,decision(scores,p))) for p in thresholds]
        key=(cfg['task'],cfg['expert'],endpoint)
        row={'id':run,'key':list(key),'config':cfg,'endpoint':endpoint,'validation_rows':len(val_records),'test_rows':len(records),
             'best_epoch':done['best_epoch'],'selection_area_0_25':done['validation_selection_score'],
             'best_budget_index':best_index,'best_validation_point':val_curve[best_index],'best_test_point':test_curve[best_index],
             'validation_curve':val_curve,'test_curve':test_curve,
             'small_metric':score_metric(arrays,np.zeros(len(records),dtype=bool)),
             'large_metric':score_metric(arrays,np.ones(len(records),dtype=bool)),
             'head_trainable_parameters':done['head_trainable_parameters'],'encoder_active_parameters':done['encoder_active_parameters'],
             'encoder_stored_parameters':done['encoder_stored_parameters']}
        result['groups'].append(row)
        case=temp/run;case.mkdir(exist_ok=True)
        np.savez(case/'paired_test_arrays.npz',scores=scores,**arrays)
        (case/'test_sample_ids.json').write_text(json.dumps([r['sample_id'] for r in records]))
        (temp/'partial_results.json').write_text(json.dumps(result,indent=2))
        print('TEST_EVALUATED',json.dumps({k:row[k] for k in ['key','best_test_point']}),flush=True)
        del model,checkpoint,data,records,arrays,val_arrays
        gc.collect();torch.cuda.empty_cache()
    result['completed']=True
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('ALL_TESTS_COMPLETE',output,flush=True)


if __name__=='__main__':main()
