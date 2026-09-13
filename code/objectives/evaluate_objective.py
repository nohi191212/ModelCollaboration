"""Evaluate fixed checkpoints using validation-selected thresholds.

Algorithm copied from the executed five-seed evaluator. Only path/device
arguments and the main guard differ from the executed script.
"""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True,help='Prepared project with outputs/router_full_20260910 caches')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    import torch
    from routing import train_router_formal as formal
    formal.ROOT=a.root.resolve()
    formal.WORK=formal.ROOT/'outputs/router_full_20260910'
    cfg=json.loads((a.run/'config.json').read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    device=torch.device(a.device)
    records={};logits={}
    model=None
    for split in ['val','test']:
        rows,data=formal.load_data(cfg,split,device);records[split]=rows
        for key in ['hidden','confidence']:
            norm=np.load(a.run/(key+'_normalization.npz'))
            data[key]=(data[key]-torch.from_numpy(norm['mean']).to(device))/torch.from_numpy(norm['std']).to(device)
        if model is None:
            model=formal.FeatureRouter(cfg,data['hidden'].shape[1],data['confidence'].shape[1],data['output'].shape[1]).to(device)
            model.load_state_dict(torch.load(a.run/'best.pt',map_location=device,weights_only=True)['state_dict']);model.eval()
        with torch.inference_mode():
            logits[split]=torch.cat([model(data,torch.arange(i,min(i+8192,len(rows)),device=device)).cpu() for i in range(0,len(rows),8192)]).numpy()
        del data
    saved_validation=np.load(a.run/'validation_logits.npy')
    assert np.allclose(logits['val'],saved_validation,rtol=1e-5,atol=1e-6), 'Validation logits changed after checkpoint reload'
    logits['val']=saved_validation
    scores={}
    for split,raw in logits.items():
        if cfg['target']=='four_state':
            prob=torch.from_numpy(raw).softmax(1).numpy()
            scores[split]={'log_ratio':raw[:,1]-raw[:,2],'probability_difference':prob[:,1]-prob[:,2]}
        else:scores[split]={cfg['ranking']:torch.from_numpy(raw[:,0]).sigmoid().numpy() if cfg['target'] in ['error','route'] else raw[:,0]}
    np.savez(a.run/'paired_logits.npz',validation=logits['val'],test=logits['test'])
    result={'config':cfg,'test_rows':len(records['test']),'validation_rows':len(records['val']),'comparisons':{}}
    for ranking in scores['val']:
        val,test=scores['val'][ranking],scores['test'][ranking]
        selection_score=formal.ranked_curve(records['val'],cfg['endpoint'],val)['selection_score']
        ranked_test=formal.ranked_curve(records['test'],cfg['endpoint'],test)
        formal.BUDGETS=[i/100 for i in range(101)]
        thresholds=formal.validation_thresholds(records['val'],cfg['endpoint'],val)
        vc=formal.fixed_curve(records['val'],cfg['endpoint'],val,thresholds)['curve']
        tc=formal.fixed_curve(records['test'],cfg['endpoint'],test,thresholds)['curve']
        best=min(range(1,100),key=lambda i:(-vc[i]['metric'],vc[i]['actual_fraction'],vc[i]['budget']))
        fixed25=sum((tc[i]['metric']+tc[i+5]['metric'])*.5*.05 for i in range(0,25,5))/.25
        result['comparisons'][ranking]={'validation_score25':selection_score,'test_ranked_score25':ranked_test['selection_score'],
           'test_exact_budget_diagnostic':ranked_test['curve'],'validation_curve':vc,'test_curve':tc,
           'validation_selected':vc[best],'test_selected':tc[best],'test_fixed_score25':fixed25,
           'checkpoint_ranking':cfg['ranking'],'same_checkpoint_control':ranking!=cfg['ranking']}
        formal.BUDGETS=[0,.05,.10,.15,.20,.25,.50,.75,1.]
    result['small_test']=formal.metric(records['test'],cfg['endpoint'],np.zeros(len(records['test']),dtype=bool))
    result['large_test']=formal.metric(records['test'],cfg['endpoint'],np.ones(len(records['test']),dtype=bool))
    (a.run/'test_evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'run':cfg['id'],'rankings':list(result['comparisons'])}),flush=True)


if __name__=='__main__':main()
