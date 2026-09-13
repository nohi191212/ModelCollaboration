from pathlib import Path
import os
import json,gzip,sys
import numpy as np
R=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing.build_router_data import small_label
experts={'cub':'cub','glsim':'cub','nlvr2':'nlvr2','vilt':'nlvr2','groundingdino':'grefcoco','instancevg':'grefcoco','yolo26x':'construction','rtdetr_x_fp32':'construction'}
checks=[]
for expert,task in experts.items():
    cols=json.loads((R/'outputs/router_training_cache_complete_20260908'/expert/'train/confidence_columns.json').read_text())
    for split in ['val','test']:
        path=R/'outputs/router_full_20260910/data'/expert/split
        rows=[json.loads(line) for line in (path/'records.jsonl').read_text().splitlines()]
        index=json.loads((path/'index.json').read_text());conf=np.load(path/'confidence.npy')
        for i in [0,len(rows)//2,len(rows)-1]:
            ref=index[i]
            with gzip.open(Path(ref['shard'])/'records.jsonl.gz','rt') as stream:
                for pos,line in enumerate(stream):
                    if pos==ref['position']:source=json.loads(line);break
            assert source['sample_id']==rows[i]['sample_id'],(expert,split,i,'identity')
            values=[source['expert_output']['confidence'][c] for c in cols]
            encoded=np.array([v if v is not None else 0. for v in values]+[float(v is not None) for v in values],dtype=np.float32)
            assert np.allclose(encoded,conf[i],atol=1e-6),(expert,split,i,'confidence')
            judge_input=dict(rows[i],split=split,expert_output=source['expert_output'])
            label=small_label(task,judge_input,expert,{'yolo26x':.005,'rtdetr_x_fp32':.05}.get(expert))
            old=rows[i]['small_label']
            if task=='grefcoco':fields=['predicted_boxes','target_type','target_boxes','correct']
            elif task=='construction':fields=['correct_event_count','events']
            else:fields=['prediction','target','correct']
            for field in fields:assert label[field]==old[field],(expert,split,i,field)
        checks.append({'expert':expert,'split':split,'samples_checked':3,'identity_confidence_labels':'passed'})
for p in json.loads((R/'outputs/score_information_20260913/results.json').read_text())['pairs']:
    for run in p['runs']:
        dest=R/'outputs/objective_multiseed_20260912/runs'/f"repeat_{p['expert']}_{p['endpoint']}_four_difference_s{run['seed']}"
        assert np.array_equal(np.load(dest/'validation_logits.npy'),np.load(dest/'paired_logits.npz')['validation'])
(R/'outputs/score_information_20260913/alignment_checks.json').write_text(json.dumps({'cache_checks':checks,'validation_logits_checks':80},indent=2))
print('PASSED: 48 cached sample identity/confidence/label checks; 80 validation-logit equality checks.')
