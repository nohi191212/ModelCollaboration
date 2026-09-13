from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
out=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/ltd_kclass_native_20260913')
while True:
    subprocess.run([sys.executable,str(here/'summarize_native.py')],check=True)
    if (out/'complete.json').exists():break
    if (out/'failure.json').exists():raise RuntimeError((out/'failure.json').read_text())
    pid=int((out/'study.pid').read_text())
    proc=Path(f'/proc/{pid}/stat')
    if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError('Native scheduler exited before completion; inspect study.log')
    time.sleep(30)
