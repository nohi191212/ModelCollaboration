from pathlib import Path
import json
import shutil
root=Path(__file__).resolve().parents[2]
out=root/'dse_results/main_table_cost_20260912'
out.mkdir(exist_ok=True)
(out/'components').mkdir(exist_ok=True)
source=root/'paper_materials/draft_20260912_revision19/data/results.json'
data=json.loads(source.read_text(encoding='utf-8'))
groups=[g for g in data['groups'] if g['key'] in [['grefcoco','instancevg','MiniCPM'],['construction','yolo26x','Qwen3.8']]]
assert len(groups)==2
(out/'main_table_results.json').write_text(json.dumps({'source':str(source),'protocol':data['protocol'],'groups':groups},indent=2),encoding='utf-8')
scripts=out/'reproducible'
scripts.mkdir(exist_ok=True)
for p in Path(__file__).resolve().parent.glob('*.py'):
    shutil.copy2(p,scripts/p.name)
for p in Path(__file__).resolve().parent.glob('*.sh'):
    shutil.copy2(p,scripts/p.name)
print(out)
