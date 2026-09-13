from pathlib import Path
import os
import json,sys
import numpy as np
R=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]));O=R/'outputs/full_budget_20260913'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).parent))
from routing import train_router_formal as f
from full_budget_train import scores
d=json.loads((O/'results.json').read_text());checks=0
for p in d['pairs']:
    rows={split:f.read_rows(R/'outputs/router_full_20260910/data'/p['expert']/split/'records.jsonl') for split in ['val','test']}
    for r in p['runs']:
        dest=O/'runs'/f"full_{p['expert']}_{p['endpoint']}_s{r['seed']}"
        raw=np.load(dest/'paired_logits.npz');m=r['methods']['normalized'];idx=m['selected_budget'];threshold=m['thresholds'][idx]
        history=[json.loads(x) for x in (dest/'history.jsonl').read_text().splitlines()]
        assert np.isclose(max(v['area100'] for h in history for v in h['validation_candidates']),r['selection']['validation_area100'])
        assert m['selected']['val']['score']>=max(m['val']['score'][0],m['val']['score'][-1])-1e-10
        for split,key in [('val','validation'),('test','test')]:
            score=scores(raw[key],r['selection']['alpha'])
            if threshold['operator']=='never':upgrade=np.zeros(len(score),bool)
            elif threshold['operator']=='always':upgrade=np.ones(len(score),bool)
            elif threshold['operator']=='ge':upgrade=score>=threshold['threshold']
            else:upgrade=score>threshold['threshold']
            actual=f.metric(rows[split],p['endpoint'],upgrade)*100
            assert np.isclose(actual,m['selected'][split]['score'],atol=1e-9),(p['expert'],r['seed'],split,actual)
            assert np.isclose(upgrade.mean()*100,m['selected'][split]['calls'],atol=1e-9)
            assert np.isclose(f.metric(rows[split],p['endpoint'],np.zeros(len(score),bool))*100,m[split]['score'][0],atol=1e-9)
            assert np.isclose(f.metric(rows[split],p['endpoint'],np.ones(len(score),bool))*100,m[split]['score'][-1],atol=1e-9)
            checks+=1
result={'status':'passed','runs':80,'direct_metric_split_checks':checks,'checkpoints_match_history_best':True,'validation_selected_score_at_least_endpoints':True,'zero_and_full_endpoints_match_original_metric':True}
(O/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
