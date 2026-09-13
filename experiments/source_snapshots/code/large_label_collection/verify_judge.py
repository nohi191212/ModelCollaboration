"""使用已审计的真实测试产物核对判分器；不运行模型、不选择设置。"""
import json
import math
from pathlib import Path
from judge import judge

root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
old=root/'outputs/experiments/20260902_four_dataset_prompt_optimization_70plus8'
base=root/'outputs/large_model_labels_all_20260909'
report={}
for task,old_task,splits in [('grefcoco','gRefCOCO',['testA','testB']),('construction','Construction10K',['test'])]:
    reference={}
    for split in splits:
        for line in (base/'inputs'/task/f'{split}.jsonl').open():
            row=json.loads(line)
            reference[row['sample_id']]=row
    for model in ['Qwen3.8','MiniCPM']:
        correct=invalid=total=0
        per_sample={r['sample_id']:r for r in map(json.loads,(old/'audits'/old_task/model/'final_test_per_sample.jsonl').open())} if task=='grefcoco' else None
        for line in (old/'model_outputs'/old_task/model/'final_test.jsonl').open():
            row=json.loads(line)
            result=judge(task,reference[row['sample_id']],row['generalist_output'])
            if per_sample is not None:
                saved=per_sample[row['sample_id']]
                assert result['correct']==saved['correct'],row['sample_id']
                assert math.isclose(result['instance_f1'],saved['instance_f1'],abs_tol=1e-12)
            if not result['valid_output']:
                assert result['correct'] is False
            total+=1
            correct+=result['correct']
            invalid+=not result['valid_output']
        saved_metrics=json.loads((old/'audits'/old_task/model/'final_test_metrics.json').read_text())
        if task=='grefcoco':
            assert correct==saved_metrics['f1_exact_correct']
        else:
            assert math.isclose(correct/total,saved_metrics['exact_four_event_set_accuracy'],abs_tol=1e-12)
        report[f'{task}/{model}']=dict(rows=total,correct=correct,invalid_always_wrong=invalid,matched_old_scoring=True)
        print('SCORER_VERIFIED',task,model,total,correct,invalid,flush=True)
(base/'judge_verified.json').write_text(json.dumps(report,indent=2))
