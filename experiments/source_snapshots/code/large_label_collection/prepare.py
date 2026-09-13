"""冻结全量输入、提示词及判分口径，不读取测试成绩选设置。"""
import json
from pathlib import Path
import shutil

root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
base=root/'outputs/large_model_labels_all_20260909'
old=root/'outputs/experiments/20260902_four_dataset_prompt_optimization_70plus8'
cache=root/'outputs/router_training_cache_complete_20260908'
assert json.loads((cache/'collection_complete.json').read_text())['all_official_sources_complete']
base.mkdir(exist_ok=True)
assert not (base/'plan.json').exists()
classes=root/'outputs/experiments/20260831_cub_qwen_prompt_enhancement/inputs_v2/cub_classes_with_train_traits.json'
shutil.copy2(classes,base/'classes.json')
class_rows=json.loads(classes.read_text())
assert [r['category_id'] for r in class_rows]==list(range(1,201))
tasks={}
for task,source in [('cub','CUB'),('nlvr2','NLVR2'),('construction','Construction10K'),('grefcoco','gRefCOCO')]:
    splits={}
    for file in sorted((cache/'inputs'/task).glob('*.jsonl')):
        rows=[json.loads(line) for line in file.open()]
        assert len({r['sample_id'] for r in rows})==len(rows)
        for row in rows:
            if 'split' in row:
                assert row['split']==file.stem
            else:
                assert task=='cub' and file.stem=='test'
                row['split']=file.stem  # 已核实的旧CUB测试清单以文件名表达划分。
        if task=='cub':
            assert all(class_rows[r['ground_truth']['category_id']-1]['official_name']==r['ground_truth']['official_name'] for r in rows)
        directory=base/'inputs'/task
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/file.name).open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        splits[file.stem]=len(rows)
    prompts={}
    for model in ['Qwen3.8','MiniCPM']:
        destination=base/'prompts'/task/f'{model}.txt'
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(old/'best_prompts'/source/f'{model}.txt',destination)
        prompts[model]=str(destination)
    tasks[task]=dict(splits=splits,prompts=prompts)
plan=dict(tasks=tasks,source_cache=str(cache),models={
    'Qwen3.8':dict(served_model='Qwen3.8-27B-FP8',gpu=0,port=18101),
    'MiniCPM':dict(served_model='MiniCPM-V-4_5',gpu=1,port=18102)},
    tensor_parallel_size=1,temperature=0,max_model_len=16384,concurrency=32,
    reuse_old_answers=False,reuse_reason='fresh consistent TP1 collection; old TP2 results retained separately; CUB IDs not assumed equivalent',
    grefcoco_unit='one request per image-expression sample, explicitly authorized 2026-09-09',
    scoring='classification exact; grefcoco existing strict GIoU rule; construction per-event correctness plus all-four exact and mask IoU',
    invalid_policy='raw answer and reason retained, empty valid prediction, correct false, continue',
    transport_policy='persist request failure, no correctness label and no automatic retry; stop affected model worker',
    split_policy='all official splits collected separately; test labels forbidden for router tuning',
    samples_per_model=sum(sum(t['splits'].values()) for t in tasks.values()))
(base/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
print(json.dumps(plan,ensure_ascii=False,indent=2))
