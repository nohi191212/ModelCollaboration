"""Keep the experiment report current and expose worker failure."""
from pathlib import Path
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OUT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/ltd_fixed_backbone_20260913')
pid=int((OUT/'study.pid').read_text())
while True:
    subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True)
    if (OUT/'complete.json').exists():
        print('ALL TRAINING AND REPORTS COMPLETE',flush=True)
        break
    if not Path(f'/proc/{pid}').exists() or Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':
        (OUT/'failure.json').write_text(json.dumps({'error':'study process exited before completion','time':time.time(),'inspect':'study.log'},indent=2))
        raise RuntimeError('Training exited before completion; inspect study.log')
    time.sleep(60)
