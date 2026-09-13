from pathlib import Path
from collections import defaultdict
import json
import time

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
OUT=ROOT/'outputs/ltd_fixed_backbone_20260913'
plan=json.loads((OUT/'job_plan.json').read_text())
done=[];active=[];historical=defaultdict(list)
for cfg in plan:
    dest=OUT/'runs'/cfg['id']
    if (dest/'test_evaluation.json').exists():done.append(cfg)
    elif dest.exists():
        history=dest/'history.jsonl'
        last=json.loads(history.read_text().splitlines()[-1]) if history.exists() and history.stat().st_size else None
        active.append({'id':cfg['id'],'epoch':last,'test_started':(dest/'test.log').exists()})
for p in (ROOT/'outputs/objective_multiseed_20260912/runs').glob('repeat_*four_difference*/completion.json'):
    r=json.loads(p.read_text());historical[r['config']['expert']].append(r['seconds'])
remaining=sum(sum(historical[c['expert']])/len(historical[c['expert']]) for c in plan if c not in done)/2
print(json.dumps({'completed':len(done),'total':len(plan),'active':active,
 'rough_remaining_training_hours_from_previous_four_state':remaining/3600,'test_and_IO_time_extra':True,'failure':(OUT/'failure.json').exists()}),flush=True)
