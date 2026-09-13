from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
data=json.loads((root/'paper_materials/draft_20260912_revision19/data/results.json').read_text(encoding='utf-8'))
print('TOP',list(data))
for g in data['groups']:
    if g['key'] in [['grefcoco','instancevg','MiniCPM'],['construction','yolo26x','Qwen3.8']]:
        print(json.dumps({k:v for k,v in g.items() if k!='methods'},ensure_ascii=False)[:6000])
        for m,v in g['methods'].items():
            print(m, {k:v1 for k,v1 in v.items() if 'curve' not in k})
