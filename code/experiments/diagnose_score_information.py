from pathlib import Path
import os
import json,sys
import numpy as np
import torch
R=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing import train_router_formal as f
torch.set_num_threads(1)
d=json.loads((R/'outputs/score_information_20260913/results.json').read_text())
out=[]
for pair in d['pairs']:
    expert,endpoint=pair['expert'],pair['endpoint'];entry={'expert':expert,'endpoint':endpoint,'splits':{}}
    for split,key in [('val','validation'),('test','test')]:
        rows=f.read_rows(R/'outputs/router_full_20260910/data'/expert/split/'records.jsonl')
        s,l=f.correctness(rows,endpoint)
        true=np.stack([((s==1)&(l==1)).mean(1),((s==0)&(l==1)).mean(1),((s==1)&(l==0)).mean(1),((s==0)&(l==0)).mean(1)],axis=1)
        pp=[]
        for seed in range(2026,2031):
            z=np.load(R/'outputs/objective_multiseed_20260912/runs'/f'repeat_{expert}_{endpoint}_four_difference_s{seed}'/'paired_logits.npz')[key]
            pp.append(torch.from_numpy(z).softmax(1).numpy())
        p=np.mean(pp,axis=0)
        item={'true_states':true.mean(0).tolist(),'predicted_states':p.mean(0).tolist(),'residual':(true-p).mean(0).tolist(),'small_accuracy':float(s.mean()),'large_accuracy':float(l.mean())}
        if pair['task']=='grefcoco':
            groups=np.array([min(r['small_label']['predicted_boxes'],3) for r in rows])
            item['box_groups']={str(g):{'count':int((groups==g).sum()),'true_gain':float((true[groups==g,1]-true[groups==g,2]).mean()),'predicted_gain':float((p[groups==g,1]-p[groups==g,2]).mean()),'state_residual':(true[groups==g]-p[groups==g]).mean(0).tolist()} for g in np.unique(groups)}
        entry['splits'][split]=item
    out.append(entry)
(R/'outputs/score_information_20260913/diagnostics.json').write_text(json.dumps(out,indent=2))
for e in out:
    if e['expert'] in ['groundingdino','instancevg']:
        print(e['expert'],e['endpoint'])
        for split,m in e['splits'].items():print(split,'actual_gain',round(m['large_accuracy']-m['small_accuracy'],4),'residual',np.round(m['residual'],4).tolist(),'box_groups',m['box_groups'])
