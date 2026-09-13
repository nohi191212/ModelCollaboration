from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from k_class_losses import native_records,native_scores

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
OUT=ROOT/'outputs/ltd_kclass_native_20260913'
groups=json.loads((HERE/'source/main_table_reference.json').read_text())['groups']
for task,expert,k in [('cub','cub',200),('nlvr2','nlvr2',2)]:
    group=next(g for g in groups if g['key'][1:]==[expert,'MiniCPM'])
    cfg=json.loads((ROOT/'outputs/router_shared_image_hidden_20260910/runs'/group['run']/'config.json').read_text())
    cfg.update(seed=2026,target='four_state',ranking='native_probability_difference',native_method='ova',max_epochs=1,id='smoke_'+task)
    dest=OUT/'preflight'/cfg['id'];dest.mkdir(parents=True,exist_ok=True)
    (dest/'config.json').write_text(json.dumps(cfg,indent=2))
    subprocess.run([sys.executable,str(HERE/'train_native.py'),'--root',str(ROOT),'--config',str(dest/'config.json'),'--output',str(dest)],check=True)
    logits=np.load(dest/'validation_logits.npy')
    assert logits.shape[1]==k+1,logits.shape
    assert np.allclose(np.load(dest/'validation_scores.npy'),native_scores(logits,'ova'))
    with (ROOT/'outputs/router_full_20260910/data'/expert/'train/records.jsonl').open() as f:row=json.loads(next(f))
    raw=np.zeros((1,k+1),dtype=np.float32)
    trueindex=int(row['small_label']['target'])-1 if task=='cub' else {'False':0,'True':1}[row['small_label']['target']]
    wrong=(trueindex+1)%k;raw[0,wrong]=10
    changed=native_records([row],raw,task)[0]
    assert not changed['small_label']['correct']
    assert changed['large_labels']==row['large_labels']
    print('PASS',task,k+1,'class-branch answer used; expert unchanged; no test loaded',flush=True)
