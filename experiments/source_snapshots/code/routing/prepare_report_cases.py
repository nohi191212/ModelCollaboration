"""Copy three predetermined case images, preserving originals and label provenance."""
import json
import shutil
from pathlib import Path
root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
w=root/'outputs/router_exploration_25pct_20260910'
data=json.loads((w/'candidate_changes.json').read_text())
cases=[c for c in data['cases'] if c['available'] and c['expert']=='instancevg' and c['endpoint']=='Qwen3.8']
assert len(cases)==3 and {c['type'] for c in cases}=={'救回','改坏','漏掉可救回'}
dest=w/'report_cases';dest.mkdir(exist_ok=True)
for c in cases:
    source=Path(c['record']['image_path'])
    output=dest/(c['sample_id']+source.suffix)
    shutil.copy2(source,output)
    c['report_image']='report_cases/'+output.name
(dest/'cases.json').write_text(json.dumps({'selection':'First sample ID per outcome, as selected before inspecting images; one fixed InstanceVG/Qwen pair, no curated replacement.','cases':cases},ensure_ascii=False,indent=2))
print(json.dumps({'images_copied':len(cases),'image_editing':False,'new_inference':False}))
