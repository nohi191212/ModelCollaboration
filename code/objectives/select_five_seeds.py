"""Select one global recipe from five-seed validation completion records."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True,help='Directory containing the 400 runs/repeat_* directories')
    a=p.parse_args()
    root=a.results
    seeds=[2026,2027,2028,2029,2030];rows=[]
    for run in sorted((root/'runs').glob('repeat_*')):
        done=json.loads((run/'completion.json').read_text())
        cfg=done['config']
        assert done['status']=='validation_complete' and not done['test_loaded']
        rows.append({'method':cfg['method'],'seed':cfg['seed'],'task':cfg['task'],
          'expert':cfg['expert'],'endpoint':cfg['endpoint'],'score':done['validation_selection_score']})
    assert len(rows)==400 and len({(r['method'],r['seed'],r['expert'],r['endpoint']) for r in rows})==400
    comparison=[]
    for method in sorted({r['method'] for r in rows}):
        values=[]
        for seed in seeds:
            task_means=[]
            for task in ['cub','grefcoco','nlvr2','construction']:
                rr=[r for r in rows if r['method']==method and r['seed']==seed and r['task']==task]
                assert len(rr)==4
                task_means.append(np.mean([r['score'] for r in rr]))
            values.append(float(np.mean(task_means)))
        avg=float(np.mean(values));sd=float(np.std(values,ddof=1))
        comparison.append({'method':method,'validation_seed_scores':values,'validation_mean':avg,
          'validation_std':sd,'selection_value':avg-sd})
    best=min(comparison,key=lambda r:(-r['selection_value'],r['method']!='four_ratio',r['method']))
    lock={'selected_method':best['method'],'selection':'task-balanced validation Score25 mean minus sample standard deviation over five router seeds',
     'repeat_seeds':seeds,'comparison':comparison,'selector_reads_test_records':False,
     'historical_pilot_selected_method':'four_ratio',
     'rule_amendment':'The authors changed the selection criterion after the seed-42 pilot and partial CUB test results had been reported. The final numerical criterion uses validation records only; this is not a prospectively untouched test set.'}
    (root/'method_lock_five_seeds.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps(lock,indent=2))


if __name__=='__main__':main()
