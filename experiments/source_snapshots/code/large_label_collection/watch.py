"""读取进度文件；全量结束后读回答案与对错标签并再次核对判分。"""
import gzip
import json
from pathlib import Path
import time
from datetime import datetime,timezone,timedelta
from judge import judge

root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
base=root/'outputs/large_model_labels_all_20260909'
plan=json.loads((base/'plan.json').read_text())
names={'cub':'鸟类分类','nlvr2':'双图判断','construction':'工地事件','grefcoco':'指代表达式定位'}
docpath=root/'paper_draft/大模型全量标注_实时进度.md'
while True:
    lines=[]
    done=[]
    failed=[]
    for model in plan['models']:
        if (base/model/'completion.json').exists():
            record=json.loads((base/model/'completion.json').read_text())
            assert record['rows']==plan['samples_per_model'] and record['all_splits_verified']
            done.append(model)
            lines.append(f"| {model} | {record['rows']:,}／{plan['samples_per_model']:,} | 采集与分划分导出完成，等待总检查 |")
            continue
        exitpath=base/'logs'/f'{model}_full.exit'
        if exitpath.exists() and exitpath.read_text().strip()!='0':
            failed.append(model)
        path=base/model/'progress.json'
        if path.exists():
            p=json.loads(path.read_text())
            phase='小样本检查' if p['smoke'] else '全量采集'
            status='已报错，需检查日志' if model in failed else f"{phase}：{names[p['task']]} {p['split']}，{p['split_completed']:,}／{p['split_total']:,}"
            lines.append(f"| {model} | {p['total_completed']:,}／{p['total_expected']:,} | {status} |")
        else:
            lines.append(f'| {model} | — | 等待启动 |')
    stamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
    doc='# 大模型全量答案及正确性标注进度\n\n更新时间：'+stamp+'。\n\n'
    doc+='| 模型 | 已保存答案与标签 | 当前状态 |\n|---|---:|---|\n'+'\n'.join(lines)+'\n\n'
    doc+='每个模型覆盖全部四任务387,929条；两个模型共775,858条。训练、验证、测试分别保存。非法模型回答记错并保留原文，服务故障不冒充模型答错。\n\n'
    doc+='当前输出：`'+str(base)+'`。采集中逐条结果在每个模型的 `answers.sqlite` 中，完成的划分导出 `labels.jsonl` 和 `audit.jsonl.gz`。旧实验结果保留，不与本次单卡配置混拼。\n'
    temp=docpath.with_suffix('.partial.md')
    temp.write_text(doc)
    temp.replace(docpath)
    print('COLLECTION_STATUS',stamp,'done',done,'failed',failed,flush=True)
    if failed:
        (base/'collection_failed.json').write_text(json.dumps(dict(models=failed,time=stamp)))
        raise RuntimeError('全量采集有任务失败：'+','.join(failed))
    if len(done)==len(plan['models']):
        break
    time.sleep(60)

verified={}
for model in plan['models']:
    verified[model]={}
    for task,info in plan['tasks'].items():
        for split,n in info['splits'].items():
            directory=base/model/task/split
            count=0
            with (base/'inputs'/task/f'{split}.jsonl').open() as inputs,(directory/'labels.jsonl').open() as labels,gzip.open(directory/'audit.jsonl.gz','rt') as audits:
                for input_line,label_line,audit_line in zip(inputs,labels,audits,strict=True):
                    row,label,audit=map(json.loads,[input_line,label_line,audit_line])
                    assert row['sample_id']==label['sample_id']==audit['sample_id']
                    assert row['split']==label['split']==audit['split']==split
                    assert label==judge(task,row,audit['generalist_output'])
                    assert 'raw_answer' in audit and 'request_error' not in audit
                    count+=1
            assert count==n
            verified[model][task+'/'+split]=count
            print('LABEL_AUDIT_READBACK_PASSED',model,task,split,count,flush=True)
(base/'collection_complete.json').write_text(json.dumps(dict(status='complete',verified=verified,total_rows=sum(sum(v.values()) for v in verified.values())),indent=2))
with docpath.open('a') as handle:
    handle.write('\n全部答案与正确性标签已完成，逐条读回并重新判分检查通过。完成证明：`collection_complete.json`。\n')
print('ALL_LARGE_MODEL_LABELS_COMPLETE',flush=True)
