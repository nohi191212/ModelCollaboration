from pathlib import Path
import json
from collections import Counter

base=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/router_full_20260910/data')
out=[]
for expert in ['yolo26x','rtdetr_x_fp32']:
    rows=[json.loads(s) for s in (base/expert/'test/records.jsonl').read_text().splitlines()]
    for endpoint in ['MiniCPM','Qwen3.8']:
        c=Counter()
        for r in rows:
            for event,s in r['small_label']['events'].items():
                l=r['large_labels'][endpoint]['events'][event]
                assert s['target']==l['target']
                state={(True,True):'both_correct',(False,True):'rescuable',(True,False):'harmful',(False,False):'both_wrong'}[(bool(s['correct']),bool(l['correct']))]
                c[state]+=1
                c['positive' if s['target'] else 'negative']+=1
                if state=='both_correct':c['both_correct_positive' if s['target'] else 'both_correct_negative']+=1
                if l['prediction'] is None:c['null_predictions']+=1
                score_correct=(l['prediction'] is True)==bool(l['target'])
                if score_correct!=bool(l['correct']):
                    c['metric_vs_state_difference']+=1
                    c['difference_null' if l['prediction'] is None else 'difference_other']+=1
        out.append(dict(expert=expert,endpoint=endpoint,images=len(rows),counts=dict(c)))
print(json.dumps(out,indent=2))
