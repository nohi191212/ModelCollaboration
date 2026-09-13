from pathlib import Path
import os
from collections import defaultdict
from statistics import mean,stdev
import json
import sys
import numpy as np
ROOT=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
OUT=ROOT/'outputs/ltd_fixed_backbone_20260913'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing import train_router_formal as f
plan=[c for c in json.loads((OUT/'job_plan.json').read_text()) if c['ltd_method']=='posthoc']
assert len(plan)==400
cache={}
groups=defaultdict(list)
for c in plan:
    raw=np.load(OUT/'runs'/c['id']/'paired_logits.npz')
    row={k:c[k] for k in ['id','task','expert','endpoint','seed','call_cost']}
    for split,key in [('val','validation'),('test','test')]:
        if (c['expert'],split) not in cache:
            cache[c['expert'],split]=f.read_rows(ROOT/'outputs/router_full_20260910/data'/c['expert']/split/'records.jsonl')
        records=cache[c['expert'],split]
        mask=raw[key][:,0]>0
        row[split]={'score':f.metric(records,c['endpoint'],mask)*100,'calls':float(mask.mean())*100}
    groups[(c['expert'],c['endpoint'],c['seed'])].append(row)
selected=[]
for group,rows in groups.items():
    assert len(rows)==5
    selected.append(min(rows,key=lambda r:(-r['val']['score'],r['val']['calls'],r['call_cost'])))
summary=[]
for expert,endpoint in sorted({(r['expert'],r['endpoint']) for r in selected}):
    rows=[r for r in selected if (r['expert'],r['endpoint'])==(expert,endpoint)]
    assert len(rows)==5
    summary.append(dict(task=rows[0]['task'],expert=expert,endpoint=endpoint,posthoc=mean(r['test']['score'] for r in rows),call_posthoc=mean(r['test']['calls'] for r in rows),posthoc_std=stdev(r['test']['score'] for r in rows),call_posthoc_std=stdev(r['test']['calls'] for r in rows),selected_costs={str(r['seed']):r['call_cost'] for r in rows}))
result={'protocol':'Original r>0 rule; choose cost among 0,.05,.1,.2,.4 per pair and seed by maximum validation task metric, then minimum validation calls, then smaller cost. Existing checkpoints selected by validation Score25; no new training. Test labels never used for cost selection.','summary':summary,'selected_runs':selected,'all_candidates':[r for rows in groups.values() for r in rows]}
(OUT/'table1_posthoc.json').write_text(json.dumps(result,indent=2))
print(json.dumps(summary,indent=2))
