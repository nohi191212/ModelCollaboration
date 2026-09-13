"""边推理边判分：提交日志先落盘，回答逐条提交，断点续跑不重发未知请求。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import time
import urllib.error
import urllib.request

from judge import judge

root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
old=root/'outputs/experiments/20260902_four_dataset_prompt_optimization_70plus8'
sys.path.insert(0,str(old/'scripts'))
from infer_grefcoco_http import parse_answer as parse_gref
from infer_event_prompt_http import parse_answer as parse_events, output_schema

parser=argparse.ArgumentParser()
parser.add_argument('--model',choices=['Qwen3.8','MiniCPM'],required=True)
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
base=root/'outputs/large_model_labels_all_20260909'
plan=json.loads((base/'plan.json').read_text())
settings=plan['models'][args.model]
output_dir=base/args.model
output_dir.mkdir(exist_ok=True)
connection=sqlite3.connect(output_dir/'answers.sqlite')
connection.execute('PRAGMA journal_mode=WAL')
connection.execute('CREATE TABLE IF NOT EXISTS samples(task TEXT,split TEXT,sample_id TEXT,input_index INTEGER,status TEXT,audit TEXT,label TEXT,PRIMARY KEY(task,split,sample_id))')
connection.commit()
unresolved=connection.execute("SELECT task,split,sample_id,status FROM samples WHERE status!='complete' LIMIT 5").fetchall()
if unresolved:
    raise RuntimeError(f'发现未明确完成的旧请求，不自动重试：{unresolved}')
classes=json.loads((base/'classes.json').read_text())
class_names='\n'.join(f"{r['category_id']}: {r['prompt_name']}" for r in classes)
class_traits='\n'.join(f"{r['category_id']}: {r['prompt_name']}; "+'; '.join(f"{t['family']} {t['value']}" for t in r['traits'][:3]) for r in classes)
prompts={task:Path(info['prompts'][args.model]).read_text().strip() for task,info in plan['tasks'].items()}
assert prompts['cub'].count('{class_names}')+prompts['cub'].count('{class_traits}')==1
assert prompts['nlvr2'].count('{sentence}')==1 and prompts['grefcoco'].count('{expression}')==1
prompts['cub']=prompts['cub'].replace('{class_names}',class_names).replace('{class_traits}',class_traits)
gref_schema={'type':'object','properties':{'objects':{'type':'array','maxItems':18,'items':{
    'type':'object','properties':{'bbox_2d':{'type':'array','items':{'type':'number','minimum':0,'maximum':1000},'minItems':4,'maxItems':4},
    'score':{'type':'number','minimum':0,'maximum':1}},'required':['bbox_2d','score'],'additionalProperties':False}}},
    'required':['objects'],'additionalProperties':False}
endpoint=f"http://127.0.0.1:{settings['port']}/v1/chat/completions"


def run_one(task,row):
    prompt=prompts[task]
    if task=='nlvr2':
        prompt=prompt.replace('{sentence}',row['sentence'])
        content=[{'type':'text','text':'Left image:'},{'type':'image_url','image_url':{'url':Path(row['image1']).as_uri()}},
                 {'type':'text','text':'Right image:'},{'type':'image_url','image_url':{'url':Path(row['image2']).as_uri()}},
                 {'type':'text','text':prompt}]
    else:
        if task=='grefcoco':
            prompt=prompt.replace('{expression}',row['expression'])
        content=[{'type':'image_url','image_url':{'url':Path(row['image_path']).as_uri()}},{'type':'text','text':prompt}]
    payload=dict(model=settings['served_model'],messages=[dict(role='user',content=content)],temperature=0,
                 max_tokens=8 if task in ['cub','nlvr2'] else 512 if task=='grefcoco' else 768)
    if task in ['cub','nlvr2']:
        if args.model=='Qwen3.8':
            payload['chat_template_kwargs']={'enable_thinking':False}
        else:
            payload['structured_outputs']={'json':{'type':'integer','minimum':1,'maximum':200}} if task=='cub' else {'choice':['True','False']}
    else:
        payload['structured_outputs']={'json':gref_schema if task=='grefcoco' else output_schema()}
    audit=dict(sample_id=row['sample_id'],split=row['split'],model=settings['served_model'],
        prompt_file=plan['tasks'][task]['prompts'][args.model],request_settings={k:v for k,v in payload.items() if k!='messages'},
        image_paths=[row['image1'],row['image2']] if task=='nlvr2' else [row['image_path']],
        started_at=time.time(),tensor_parallel_size=1,model_requests_for_sample=1)
    request=urllib.request.Request(endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    started=time.monotonic()
    try:
        with urllib.request.urlopen(request,timeout=900) as response:
            body=response.read().decode('utf-8')
    except (urllib.error.URLError,TimeoutError,OSError) as error:
        audit.update(request_error=str(error),elapsed_seconds=time.monotonic()-started)
        if isinstance(error,urllib.error.HTTPError):
            audit['http_error_body']=error.read().decode('utf-8',errors='replace')
        return audit,None,'request_error'
    audit.update(response_body=body,elapsed_seconds=time.monotonic()-started)
    try:
        response=json.loads(body)
        choice=response['choices'][0]
        raw=choice['message']['content']
    except (ValueError,KeyError,IndexError,TypeError) as error:
        audit['request_error']='Malformed service response: '+str(error)
        return audit,None,'request_error'
    audit['raw_answer']=raw
    audit['finish_reason']=choice.get('finish_reason')
    answer=raw.strip() if isinstance(raw,str) else raw
    parsed=dict(parse_status='valid')
    try:
        if not isinstance(answer,str):
            raise ValueError('model content is not text')
        if task=='cub':
            if not answer.isdigit() or answer!=str(int(answer)) or not 1<=int(answer)<=200:
                raise ValueError('answer must be exactly one standard integer in 1..200')
            parsed['predicted_category_id']=int(answer)
        elif task=='nlvr2':
            if answer not in ['True','False']:
                raise ValueError('answer must be exactly True or False')
            parsed['prediction']=answer
        elif task=='grefcoco':
            parsed['objects']=parse_gref(answer,row['width'],row['height'],18)
        else:
            parsed['events']=parse_events(answer)
    except (ValueError,TypeError) as error:
        parsed=dict(parse_status='invalid_output_counted_wrong',parse_error=str(error))
        if task=='cub': parsed['predicted_category_id']=None
        elif task=='nlvr2': parsed['prediction']=None
        elif task=='grefcoco': parsed['objects']=[]
        else: parsed['events']=None
    audit['generalist_output']=parsed
    return audit,judge(task,row,parsed),'complete'


started=time.monotonic()
new_count=0
for task,info in plan['tasks'].items():
    order=[split for split in ['train','dev','val','test','test1','test2','testA','testB'] if split in info['splits']]
    for split in order:
        rows=[json.loads(line) for line in (base/'inputs'/task/f'{split}.jsonl').open()]
        assert len(rows)==info['splits'][split]
        complete={row[0] for row in connection.execute("SELECT sample_id FROM samples WHERE task=? AND split=? AND status='complete'",(task,split))}
        assert complete <= {r['sample_id'] for r in rows}
        pending=[(index,row) for index,row in enumerate(rows) if row['sample_id'] not in complete]
        if args.smoke:
            pending=pending[:1]
        print('SPLIT_START',args.model,task,split,'existing',len(complete),'new',len(pending),flush=True)
        with ThreadPoolExecutor(max_workers=plan['concurrency']) as executor:
            for offset in range(0,len(pending),plan['concurrency']):
                batch=pending[offset:offset+plan['concurrency']]
                for index,row in batch:
                    paths=[row['image1'],row['image2']] if task=='nlvr2' else [row['image_path']]
                    if not all(Path(path).is_file() for path in paths):
                        raise FileNotFoundError(paths)
                    connection.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?)',(task,split,row['sample_id'],index,'submitted',None,None))
                connection.commit()
                futures={executor.submit(run_one,task,row):(index,row) for index,row in batch}
                errors=[]
                for future in as_completed(futures):
                    index,row=futures[future]
                    audit,label,status=future.result()
                    connection.execute('UPDATE samples SET status=?,audit=?,label=? WHERE task=? AND split=? AND sample_id=?',
                        (status,json.dumps(audit,ensure_ascii=False),json.dumps(label,ensure_ascii=False) if label is not None else None,task,split,row['sample_id']))
                    connection.commit()
                    if status!='complete':
                        errors.append(row['sample_id'])
                    else:
                        new_count+=1
                        complete.add(row['sample_id'])
                progress=dict(model=args.model,task=task,split=split,split_completed=len(complete),split_total=len(rows),
                    total_completed=connection.execute("SELECT count(*) FROM samples WHERE status='complete'").fetchone()[0],
                    total_expected=plan['samples_per_model'],new_count=new_count,samples_per_second=new_count/(time.monotonic()-started),
                    updated_at=time.time(),smoke=args.smoke,request_errors=errors)
                temp=output_dir/'progress.partial.json'
                temp.write_text(json.dumps(progress,ensure_ascii=False,indent=2))
                temp.replace(output_dir/'progress.json')
                print('PROGRESS',json.dumps(progress,ensure_ascii=False),flush=True)
                if errors:
                    raise RuntimeError(f'服务请求失败，不生成对错标签，不自动重试：{errors}')
        if len(complete)==len(rows):
            destination=output_dir/task/split
            destination.mkdir(parents=True,exist_ok=True)
            correct=invalid=0
            with (destination/'labels.partial.jsonl').open('w') as labels,gzip.open(destination/'audit.partial.jsonl.gz','wt') as audits:
                records=connection.execute('SELECT sample_id,audit,label FROM samples WHERE task=? AND split=? ORDER BY input_index',(task,split))
                for expected,(sample_id,audit,label) in zip(rows,records,strict=True):
                    assert sample_id==expected['sample_id'] and label is not None
                    value=json.loads(label)
                    assert value['sample_id']==sample_id and value['split']==split
                    correct+=value['correct']
                    invalid+=not value['valid_output']
                    labels.write(label+'\n')
                    audits.write(audit+'\n')
            (destination/'labels.partial.jsonl').replace(destination/'labels.jsonl')
            (destination/'audit.partial.jsonl.gz').replace(destination/'audit.jsonl.gz')
            (destination/'completion.json').write_text(json.dumps(dict(rows=len(rows),correct=correct,invalid=invalid,
                sample_correctness_rate=correct/len(rows),metric_note='construction correctness is exact event set, NOT macro F1',
                source_input=str(base/'inputs'/task/f'{split}.jsonl')),indent=2))
            print('SPLIT_COMPLETE',args.model,task,split,len(rows),correct,invalid,flush=True)
if args.smoke:
    (output_dir/'smoke_complete.json').write_text(json.dumps(dict(status='complete',new_samples=new_count)))
else:
    total=connection.execute("SELECT count(*) FROM samples WHERE status='complete'").fetchone()[0]
    assert total==plan['samples_per_model']
    assert all((output_dir/task/split/'completion.json').exists() for task,info in plan['tasks'].items() for split in info['splits'])
    (output_dir/'completion.json').write_text(json.dumps(dict(status='complete',rows=total,all_splits_verified=True)))
connection.close()
print('MODEL_COLLECTION_COMPLETE',args.model,'smoke',args.smoke,flush=True)
