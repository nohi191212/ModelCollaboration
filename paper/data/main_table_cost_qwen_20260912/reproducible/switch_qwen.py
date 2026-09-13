from pathlib import Path
import json
import shutil
root=Path(__file__).resolve().parents[2]
old=root/'dse_results/main_table_cost_20260912'
out=root/'dse_results/main_table_cost_qwen_20260912'
out.mkdir(exist_ok=True)
(out/'components').mkdir(exist_ok=True)
(out/'reproducible').mkdir(exist_ok=True)
(out/'logs').mkdir(exist_ok=True)
source=root/'paper_materials/draft_20260912_revision19/data/results.json'
data=json.loads(source.read_text(encoding='utf-8'))
groups=[next(g for g in data['groups'] if g['key']==key) for key in [['grefcoco','instancevg','Qwen3.8'],['construction','yolo26x','Qwen3.8']]]
(out/'main_table_results.json').write_text(json.dumps({'source':str(source),'protocol':data['protocol'],'groups':groups},indent=2),encoding='utf-8')
for name in ['instancevg_specialist.json','yolo26x_specialist.json','yolo26x_router.json','qwen_latency.json','qwen_flops.json']:
    shutil.copy2(old/'components'/name,out/'components'/name)
for name in ['build_figures.py','verify_and_report.py','latex_includes.tex']:
    text=(old/name).read_text(encoding='utf-8').replace("['minicpm','qwen']","['qwen_gref','qwen']").replace('gRefCOCO / M','gRefCOCO / Q').replace('InstanceVG--MiniCPM','InstanceVG--Qwen').replace('main_table_cost_20260912','main_table_cost_qwen_20260912')
    (out/name).write_text(text,encoding='utf-8')
for p in (old/'reproducible').iterdir():
    if p.is_file(): shutil.copy2(p,out/'reproducible'/p.name)
for g in groups:
    print(g['key'],g['run'],{m:g['methods'][m]['test_point'] for m in ['learned','confidence']})
