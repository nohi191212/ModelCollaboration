"""Boundary sanity check and exact replay of one prior training trajectory."""
from pathlib import Path
import json
import subprocess
import sys
import torch
from ltd_methods import loss_values

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
OUT=ROOT/'outputs/ltd_fixed_backbone_20260913'
states=torch.eye(4)
for method,dim in [('css',2),('ova',2),('posthoc',1),('four_state',4)]:
    raw=torch.zeros(4,dim,requires_grad=True)
    loss=loss_values(raw,states,method,0.)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(raw.grad).all()
    if method=='posthoc':
        torch.testing.assert_close(raw.grad[:,0],torch.tensor([0.,-.5,.5,0.]))
    print(method,'losses',loss.detach().tolist(),'gradients',raw.grad.tolist(),flush=True)

groups=json.loads((HERE/'source/main_table_reference.json').read_text())['groups']
g=next(g for g in groups if 'yolo26x' in g['run'] and 'Qwen3.8' in g['run'])
base=json.loads((ROOT/'outputs/router_shared_image_hidden_20260910/runs'/g['run']/'config.json').read_text())
for method in ['four_state','css','ova','posthoc']:
    cfg=dict(base);cfg.update(seed=2026,target='four_state',ranking='ltd_score',ltd_method=method,call_cost=.1 if method=='posthoc' else 0.,max_epochs=1,id='smoke_'+method)
    dest=OUT/'preflight'/cfg['id'];dest.mkdir(parents=True,exist_ok=True)
    (dest/'config.json').write_text(json.dumps(cfg,indent=2))
    subprocess.run([sys.executable,str(HERE/'train_ltd.py'),'--root',str(ROOT),'--config',str(dest/'config.json'),'--output',str(dest)],check=True)
    if method=='four_state':
        old=ROOT/'outputs/objective_multiseed_20260912/runs/repeat_yolo26x_Qwen3.8_four_difference_s2026/history.jsonl'
        reference=json.loads(old.read_text().splitlines()[0])
        actual=json.loads((dest/'history.jsonl').read_text().splitlines()[0])
        assert abs(reference['training_loss']-actual['training_loss'])<1e-7,(reference,actual)
        assert abs(reference['validation_score']-actual['validation_score'])<1e-10,(reference,actual)
        print('EXACT FIRST-EPOCH REPLAY PASSED',flush=True)
print('PREFLIGHT PASSED; test not loaded',flush=True)
