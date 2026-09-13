"""四卡独立比较 Qwen MTP；独立审计，不写正式答案。"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import infer_8gpu as core

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
MODEL = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/checkpoints/Qwen3.8-27B-FP8')
DATA = ROOT/'outputs/large_model_labels_all_20260909'
CONFIGS = [dict(name='baseline',mtp=0,seqs=32,tokens=8192),
           dict(name='mtp1',mtp=1,seqs=32,tokens=8192),
           dict(name='mtp2',mtp=2,seqs=32,tokens=8192),
           dict(name='mtp1_large',mtp=1,seqs=64,tokens=16384)]

async def measure(client, endpoint, a, task, rows, prompt, concurrency, audit_file):
    iterator = iter(rows)
    results = []
    started = time.monotonic()
    async def worker():
        for row in iterator:
            payload = core.make_payload(a,task,row,prompt)
            before = time.monotonic()
            response = await client.post(endpoint,json=payload)
            record = dict(task=task,sample_id=row['sample_id'],response_body=response.text,
                          status_code=response.status_code,elapsed=time.monotonic()-before)
            audit_file.write(json.dumps(record,ensure_ascii=False)+'\n')
            audit_file.flush()
            response.raise_for_status()
            value = response.json()
            raw = value['choices'][0]['message']['content']
            parsed, label = core.score_answer(task,row,raw)
            results.append(dict(valid=label['valid_output'],correct=label['correct'],
                                tokens=value.get('usage',{}).get('completion_tokens',0),
                                latency=record['elapsed']))
    jobs = [asyncio.create_task(worker()) for _ in range(concurrency)]
    settled = await asyncio.gather(*jobs,return_exceptions=True)
    for value in settled:
        if isinstance(value,BaseException):
            raise value
    elapsed = time.monotonic()-started
    return dict(task=task,concurrency=concurrency,rows=len(results),seconds=elapsed,
                samples_per_second=len(results)/elapsed,
                output_tokens_per_second=sum(r['tokens'] for r in results)/elapsed,
                mean_output_tokens=sum(r['tokens'] for r in results)/len(results),
                valid=sum(r['valid'] for r in results),correct=sum(r['correct'] for r in results))

async def one(rank, config, output, samples):
    a = SimpleNamespace(model_path=MODEL,model_kind='Qwen3.8',data_path=DATA,
        media_root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang'),old_media_root=None,
        tp=1,port_base=19300,max_model_len=16384,max_num_seqs=config['seqs'],
        max_num_batched_tokens=config['tokens'],gpu_memory_utilization=0.90)
    cmd = core.service_command(a,rank)
    if config['mtp']:
        cmd += ['--speculative-config',json.dumps(dict(method='mtp',num_speculative_tokens=config['mtp']))]
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    dest = output/config['name']
    dest.mkdir()
    core.save_json(dest/'settings.json',dict(config=config,command=cmd,gpu=rank,samples=samples))
    process = None
    try:
        with (dest/'service.log').open('wb') as logfile:
            process = subprocess.Popen(cmd,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
        endpoint = f'http://127.0.0.1:{19300+rank}'
        async with httpx.AsyncClient(timeout=900,trust_env=False,
                    limits=httpx.Limits(max_connections=128,max_keepalive_connections=128)) as client:
            deadline = time.monotonic()+1800
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'service exited {process.returncode}; see {dest}/service.log')
                if time.monotonic()>deadline:
                    raise TimeoutError('service startup exceeded 1800s')
                try:
                    health = await client.get(endpoint+'/health',timeout=2)
                except httpx.TransportError:
                    await asyncio.sleep(2)
                    continue
                if health.status_code == 200:
                    break
                await asyncio.sleep(2)
            print('READY',config['name'],flush=True)
            classes = json.loads((DATA/'classes.json').read_text())
            names = '\n'.join(f"{r['category_id']}: {r['prompt_name']}" for r in classes)
            traits = '\n'.join(f"{r['category_id']}: {r['prompt_name']}; "+'; '.join(f"{t['family']} {t['value']}" for t in r['traits'][:3]) for r in classes)
            summaries = []
            with (dest/'audit.jsonl').open('w') as audit:
                for task in core.TASKS:
                    prompt = (DATA/'prompts'/task/'Qwen3.8.txt').read_text().strip().replace('{class_names}',names).replace('{class_traits}',traits)
                    rows = [row for _,row in core.read_rows(a,task,'train',samples+8)]
                    await measure(client,endpoint+'/v1/chat/completions',a,task,rows[:8],prompt,8,audit)
                    # 各配置测同一批样本，预热样本与计时样本分开。
                    result = await measure(client,endpoint+'/v1/chat/completions',a,task,rows[8:],prompt,
                                           config['seqs']*2,audit)
                    summaries.append(result)
                    core.save_json(dest/'results.json',summaries)
                    print('RESULT',config['name'],json.dumps(result),flush=True)
                core.save_json(dest/'complete.json',dict(status='complete',results=summaries))
            return dict(config=config,status='complete',results=summaries)
    except (RuntimeError,TimeoutError,httpx.HTTPError,ValueError,KeyError) as error:
        # 配置不兼容或请求失败保留现场；不重试、不改正式结果，其他比较组继续。
        core.save_json(dest/'failure.json',dict(error=str(error),type=type(error).__name__))
        print('FAILED',config['name'],repr(error),flush=True)
        return dict(config=config,status='failed',error=str(error))
    finally:
        if process is not None:
            await core.stop_services([process])

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    devices = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used = {int(row.split(',')[0]):int(row.split(',')[1]) for row in devices.splitlines()}
    assert all(g in used and used[g]<500 for g in range(4)),devices
    results = await asyncio.gather(*(one(i,c,args.output,args.samples) for i,c in enumerate(CONFIGS)))
    core.save_json(args.output/'summary.json',results)
    print('BENCHMARK_FINISHED',flush=True)

if __name__=='__main__':
    asyncio.run(main())
