from pathlib import Path
import os
import json,sys
R=Path(os.environ.get('ICASSP_DATA_ROOT', Path(__file__).resolve().parents[2]));O=R/'outputs/full_budget_20260913'
done=list((O/'runs').glob('*/completion.json'));exits=[json.loads(p.read_text()) for p in (O/'logs').glob('*.exit.json')];failed=[r for r in exits if r['exit_code']!=0]
running=[]
for p in (O/'runs').iterdir():
    if (p/'completion.json').exists():continue
    if (p/'history.jsonl').exists():
        last=json.loads((p/'history.jsonl').read_text().splitlines()[-1]);running.append({'id':p.name,'epoch':last['epoch'],'seconds':round(last['seconds'],1)})
    else:running.append({'id':p.name,'stage':'loading'})
print(json.dumps({'completed':len(done),'total':80,'failed':failed,'active':running},ensure_ascii=False))
if len(done)!=80:sys.exit(0)
old=json.loads((R/'outputs/paper_score_comparison_20260913/results.json').read_text())
result={'protocol':json.loads((O/'protocol.json').read_text()),'pairs':[]}
for p in old['pairs']:
    pair={k:p[k] for k in ['expert','endpoint','task','small','large']};pair['runs']=[]
    for previous in p['runs']:
        seed=previous['seed'];folder=O/'runs'/f"full_{p['expert']}_{p['endpoint']}_s{seed}"
        new=json.loads((folder/'evaluation.json').read_text());oldnorm=previous['methods']['normalized']
        index=min(range(101),key=lambda i:(-oldnorm['val']['score'][i],oldnorm['val']['calls'][i],i))
        baseline={'old_difference_interior':previous['methods']['difference']['selected'],'old_normalized_interior':oldnorm['selected'],'old_normalized_endpoints':{split:{k:oldnorm[split][k][index] for k in ['score','calls']} for split in ['val','test']}}
        pair['runs'].append({'seed':seed,'selection':new['selection'],'methods':new['methods'],'baseline_policies':baseline,'baseline_old_normalized':{split:oldnorm[split] for split in ['val','test']},'old_normalized_endpoint_budget':index})
    result['pairs'].append(pair)
(O/'results.json').write_text(json.dumps(result))
print('COLLECTED',len(result['pairs']),'pairs')
