"""整套四卡吞吐比较：DP4/TP1、DP2/TP2、DP1/TP4。"""
import argparse
import asyncio
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import httpx
import infer_8gpu as core

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
DATA=ROOT/'outputs/large_model_labels_all_20260909'
COUNTS={'cub':11788,'nlvr2':107292,'construction':10013,'grefcoco':258836}

async def measure(a,clients,task,rows,prompt,concurrency,audit,stop):
    iterator=iter(rows)
    stats=[]
    started=time.monotonic()
    async def worker(rank):
        for row in iterator:
            if stop.is_set():
                return
            payload=core.make_payload(a,task,row,prompt)
            before=time.monotonic()
            try:
                response=await clients[rank].post(f'http://127.0.0.1:{a.port_base+rank}/v1/chat/completions',json=payload)
            except httpx.TransportError as error:
                audit.write(json.dumps(dict(task=task,sample_id=row['sample_id'],replica=rank,request_error=str(error)))+'\n')
                audit.flush()
                stop.set()
                raise
            record=dict(task=task,sample_id=row['sample_id'],replica=rank,status_code=response.status_code,
                        response_body=response.text,elapsed=time.monotonic()-before)
            audit.write(json.dumps(record,ensure_ascii=False)+'\n')
            audit.flush()
            if response.status_code!=200:
                stop.set()
                response.raise_for_status()
            answer=response.json()
            parsed,label=core.score_answer(task,row,answer['choices'][0]['message']['content'])
            stats.append(dict(replica=rank,valid=label['valid_output'],correct=label['correct'],
                              tokens=answer.get('usage',{}).get('completion_tokens',0)))
    jobs=[asyncio.create_task(worker(rank)) for rank in range(a.replicas) for _ in range(concurrency)]
    settled=await asyncio.gather(*jobs,return_exceptions=True)
    for value in settled:
        if isinstance(value,BaseException):
            raise value
    elapsed=time.monotonic()-started
    if len(stats)!=len(rows):
        raise InterruptedError('已停止新请求，已发出的请求已保存')
    return dict(task=task,concurrency_per_replica=concurrency,rows=len(stats),seconds=elapsed,
                samples_per_second=len(stats)/elapsed,valid_samples_per_second=sum(r['valid'] for r in stats)/elapsed,
                valid=sum(r['valid'] for r in stats),correct=sum(r['correct'] for r in stats),
                output_tokens_per_second=sum(r['tokens'] for r in stats)/elapsed,
                mean_output_tokens=sum(r['tokens'] for r in stats)/len(stats),
                per_replica_rows=[sum(r['replica']==rank for r in stats) for rank in range(a.replicas)])

async def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    stop=asyncio.Event()
    loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM):
        loop.add_signal_handler(sig,stop.set)
    a=SimpleNamespace(model_path=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/checkpoints/Qwen3.8-27B-FP8'),
        model_kind='Qwen3.8',data_path=DATA,output_path=args.output,
        media_root=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang'),old_media_root=None,
        tp=args.tp,port_base=args.port_base,max_model_len=16384,max_num_seqs=args.seqs,
        max_num_batched_tokens=args.tokens,gpu_memory_utilization=0.90,
        gpu_ids=['0','1','2','3'],replicas=4//args.tp,startup_timeout=1800)
    original=core.service_command
    def command(config,rank):
        result=original(config,rank)
        if args.mtp:
            result+=['--speculative-config',json.dumps(dict(method='mtp',num_speculative_tokens=args.mtp))]
        return result
    core.service_command=command
    (args.output/'logs').mkdir()
    core.save_json(args.output/'settings.json',dict(tp=args.tp,dp=a.replicas,mtp=args.mtp,seqs=args.seqs,
        tokens=args.tokens,concurrency=args.concurrency,samples=args.samples,
        commands=[command(a,i) for i in range(a.replicas)]))
    processes=[]
    clients=[]
    started=time.monotonic()
    try:
        await core.start_services(a,processes,stop)
        clients=[httpx.AsyncClient(timeout=900,trust_env=False,
            limits=httpx.Limits(max_connections=args.concurrency,max_keepalive_connections=args.concurrency))
            for _ in range(a.replicas)]
        classes=json.loads((DATA/'classes.json').read_text())
        names='\n'.join(f"{r['category_id']}: {r['prompt_name']}" for r in classes)
        traits='\n'.join(f"{r['category_id']}: {r['prompt_name']}; "+'; '.join(f"{t['family']} {t['value']}" for t in r['traits'][:3]) for r in classes)
        results=[]
        with (args.output/'audit.jsonl').open('w') as audit:
            for task in core.TASKS:
                prompt=(DATA/'prompts'/task/'Qwen3.8.txt').read_text().strip().replace('{class_names}',names).replace('{class_traits}',traits)
                rows=[r for _,r in core.read_rows(a,task,'train',args.samples+32)]
                await measure(a,clients,task,rows[:32],prompt,8,audit,stop)
                result=await measure(a,clients,task,rows[32:],prompt,args.concurrency,audit,stop)
                results.append(result)
                core.save_json(args.output/'results.json',results)
                print('RESULT',json.dumps(result),flush=True)
        estimated=sum(COUNTS[r['task']]/r['samples_per_second'] for r in results)
        core.save_json(args.output/'complete.json',dict(status='complete',results=results,
            estimated_full_model_seconds=estimated,weighted_samples_per_second=sum(COUNTS.values())/estimated,
            total_seconds_including_load=time.monotonic()-started))
        print('LAYOUT_COMPLETE',args.output,estimated,flush=True)
    finally:
        for client in clients:
            await client.aclose()
        await core.stop_services(processes)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tp',type=int,choices=[1,2,4],required=True)
    p.add_argument('--mtp',type=int,default=0)
    p.add_argument('--seqs',type=int,default=32)
    p.add_argument('--tokens',type=int,default=8192)
    p.add_argument('--concurrency',type=int,default=64)
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--port-base',type=int,default=19400)
    asyncio.run(run(p.parse_args()))
