"""Pause only our scheduler, preserve its training children, replace orchestration."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OUT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/ltd_fixed_backbone_20260913')
old=int((OUT/'study.pid').read_text())
cmd=Path(f'/proc/{old}/cmdline').read_bytes().split(b'\0')
assert str(HERE/'run_study.py').encode() in cmd,cmd
os.kill(old,signal.SIGSTOP)
children=subprocess.check_output(['ps','--ppid',str(old),'-o','pid='],text=True).split()
active=[]
for p in children:
    pid=int(p);args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    if str(HERE/'train_ltd.py') in args:
        dest=Path(args[args.index('--output')+1]);phase='train'
    elif str(HERE/'evaluate_ltd.py') in args:
        dest=Path(args[args.index('--run')+1]);phase='test'
    else:raise RuntimeError(('Unexpected child; original scheduler left paused',pid,args))
    env=Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0')
    gpu=int(next(x.split('=',1)[1] for x in env if x.startswith('CUDA_VISIBLE_DEVICES=')))
    active.append({'id':dest.name,'pid':pid,'gpu':gpu,'phase':phase})
report=int((OUT/'report.pid').read_text())
assert str(HERE/'finish_report.py').encode() in Path(f'/proc/{report}/cmdline').read_bytes().split(b'\0')
os.kill(report,signal.SIGTERM)
handoff={'previous_scheduler':old,'active':active,'new_concurrency':10,'per_gpu':5,'time':time.time(),'training_children_interrupted':False}
(OUT/'concurrency_handoff.json').write_text(json.dumps(handoff,indent=2))
with (OUT/'study_ten.log').open('w') as log:
    new=subprocess.Popen([sys.executable,str(HERE/'run_ten.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'study.pid').write_text(str(new.pid)+'\n')
# Terminating the paused parent does not signal its still-running children.
os.kill(old,signal.SIGTERM)
os.kill(old,signal.SIGCONT)
with (OUT/'report_ten.log').open('w') as log:
    rep=subprocess.Popen([sys.executable,str(HERE/'finish_report.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'report.pid').write_text(str(rep.pid)+'\n')
protocol=json.loads((OUT/'protocol.json').read_text())
protocol['execution']={'concurrency':10,'gpu_slots':{'0':5,'1':5},'changed_at':time.time(),'model_settings_changed':False}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
with (OUT/'exp_settings.md').open('a') as f:
    f.write('\n## 并发调整\n\n用户要求提高到同时10组；0、1号卡各5组。正在运行的训练由新队列接管，未中断；模型和评测设置不变。新队列日志为study_ten.log。\n')
for name in ['run_ten.py','switch_ten.py']:
    (OUT/'reproducible'/name).write_bytes((HERE/name).read_bytes())
print(json.dumps({'scheduler':new.pid,'reporter':rep.pid,**handoff}),flush=True)
