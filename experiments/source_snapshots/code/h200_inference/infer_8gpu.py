#!/usr/bin/env python3
"""八卡多副本推理。单文件可移动；沿用四任务冻结提示词、解析和判分口径。"""
from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import traceback

import httpx

"""生成路由监督；非法模型回答强制记错，运输失败不生成模型对错标签。"""
import numpy as np

EVENTS = ['rule_1_ppe_violation','rule_2_fall_protection_violation',
          'rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']


def judge(task, row, output):
    valid = output['parse_status'] == 'valid'
    result = dict(sample_id=row['sample_id'], split=row['split'], valid_output=valid,
                  correct=False, error_reason=output.get('parse_error'))
    gt = row['ground_truth']
    if task == 'cub':
        result.update(correct=valid and output['predicted_category_id']==gt['category_id'],
                      target=gt['category_id'], prediction=output['predicted_category_id'],
                      correctness_rule='category_exact_match')
    elif task == 'nlvr2':
        result.update(correct=valid and output['prediction']==gt, target=gt,
                      prediction=output['prediction'], correctness_rule='True_False_exact_match')
    elif task == 'grefcoco':
        targets = gt['target_boxes']
        boxes = [o['bbox_xyxy_absolute'] for o in output['objects'] if o['score']>=0.7] if valid else []
        assert gt['target_type'] in ['no_target','single_target','multi_target']
        assert bool(targets) == (gt['target_type']!='no_target')
        tp = 0
        if valid and boxes and targets:
            a = np.asarray(boxes,dtype=np.float64)
            b = np.asarray([o['bbox_xyxy'] for o in targets],dtype=np.float64)
            sizes = np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0)
            intersection = sizes[:,:,0]*sizes[:,:,1]
            union = ((a[:,2]-a[:,0])*(a[:,3]-a[:,1]))[:,None] + ((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))[None,:]-intersection
            enclosure = np.maximum(a[:,None,2:],b[None,:,2:])-np.minimum(a[:,None,:2],b[None,:,:2])
            area = enclosure[:,:,0]*enclosure[:,:,1]
            giou = intersection/union-(area-union)/area
            for _ in range(min(len(a),len(b))):
                index = np.argmax(giou)
                i,j = np.unravel_index(index,giou.shape)
                if giou[i,j]<0.5:
                    break
                tp += 1
                giou[i,:]=0
                giou[:,j]=0
        f1 = (2*tp/(len(boxes)+len(targets)) if targets else float(not boxes)) if valid else 0.0
        result.update(correct=bool(valid and f1>=1.0),instance_f1=f1,target_type=gt['target_type'],
                      predicted_boxes=len(boxes),target_boxes=len(targets),matched_boxes=tp,
                      correctness_rule='score_0.7_greedy_GIoU_0.5_sample_F1_1; invalid_always_wrong')
    else:
        assert task=='construction' and set(gt['events'])==set(EVENTS)
        labels={}
        for event in EVENTS:
            reference=gt['events'][event]
            candidate=output['events'][event] if valid else None
            predicted=candidate['present'] if valid else None
            mask_iou=None
            if reference['boxes']:
                masks=[]
                for boxes in [reference['boxes'],candidate['boxes'] if valid else []]:
                    mask=np.zeros((100,100),dtype=bool)
                    for x1,y1,x2,y2 in boxes:
                        mask[int(y1*100):int(y2*100)+1,int(x1*100):int(x2*100)+1]=True
                    masks.append(mask)
                mask_iou=float(np.logical_and(*masks).sum()/np.logical_or(*masks).sum())
            labels[event]=dict(target=reference['present'],prediction=predicted,
                               correct=bool(valid and predicted==reference['present']),
                               positive_target_mask_iou=mask_iou)
        result.update(correct=bool(valid and all(x['correct'] for x in labels.values())),
            events=labels,correct_event_count=sum(x['correct'] for x in labels.values()),
            correctness_rule='all_four_event_presence_exact_match; invalid_always_wrong',
            native_metric_note='macro_presence_F1 must be recomputed over a dataset; correct is not F1')
    return result

def parse_gref(answer: str, width: int, height: int, max_objects: int) -> list[dict]:
    payload = json.loads(answer)
    if not isinstance(payload, dict) or set(payload) != {"objects"}:
        raise ValueError("top-level JSON must contain only objects")
    if not isinstance(payload["objects"], list) or len(payload["objects"]) > max_objects:
        raise ValueError(f"objects must be a list with at most {max_objects} entries")
    parsed = []
    for index, item in enumerate(payload["objects"]):
        if not isinstance(item, dict) or set(item) != {"bbox_2d", "score"}:
            raise ValueError(f"object {index} must contain only bbox_2d and score")
        box = item["bbox_2d"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"object {index} bbox_2d must contain four coordinates")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
            raise ValueError(f"object {index} coordinates must be numbers")
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError(f"object {index} coordinates are outside ordered [0,1000] xyxy")
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
            raise ValueError(f"object {index} score is not a number in [0,1]")
        parsed.append({
            "bbox_2d_normalized": [x1, y1, x2, y2],
            "bbox_xyxy_absolute": [
                x1 * width / 1000,
                y1 * height / 1000,
                x2 * width / 1000,
                y2 * height / 1000,
            ],
            "score": float(score),
        })
    return parsed



EVENT_MAX_BOXES = {
    "rule_1_ppe_violation": 13,
    "rule_2_fall_protection_violation": 6,
    "rule_3_unprotected_edge_violation": 3,
    "rule_4_excavator_proximity_violation": 2,
}


def parse_events(answer: str) -> dict:
    payload = json.loads(answer)
    if not isinstance(payload, dict) or set(payload) != {"events"}:
        raise ValueError("top-level JSON must contain only events")
    events = payload["events"]
    if not isinstance(events, dict) or set(events) != set(EVENT_MAX_BOXES):
        raise ValueError("events must contain exactly the four fixed safety rules")

    parsed = {}
    for event_name, max_boxes in EVENT_MAX_BOXES.items():
        event = events[event_name]
        if not isinstance(event, dict) or set(event) != {"present", "reason", "boxes"}:
            raise ValueError(f"{event_name} must contain only present, reason, and boxes")
        if not isinstance(event["present"], bool):
            raise ValueError(f"{event_name}.present must be boolean")
        if not isinstance(event["boxes"], list) or len(event["boxes"]) > max_boxes:
            raise ValueError(f"{event_name}.boxes exceeds the training-derived limit {max_boxes}")
        boxes = []
        for index, box in enumerate(event["boxes"]):
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"{event_name}.boxes[{index}] must contain four coordinates")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
                raise ValueError(f"{event_name}.boxes[{index}] coordinates must be integers")
            x1, y1, x2, y2 = map(float, box)
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError(f"{event_name}.boxes[{index}] is outside ordered [0,1000] xyxy")
            boxes.append([x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000])
        if event["present"]:
            if not isinstance(event["reason"], str) or not event["reason"].strip() or not boxes:
                raise ValueError(f"{event_name} present=true requires a non-empty reason and at least one box")
            reason = event["reason"].strip()
        else:
            if event["reason"] is not None or boxes:
                raise ValueError(f"{event_name} present=false requires reason=null and boxes=[]")
            reason = None
        parsed[event_name] = {"present": event["present"], "reason": reason, "boxes": boxes}
    return parsed


def output_schema() -> dict:
    box_schema = {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
        "minItems": 4,
        "maxItems": 4,
    }
    event_properties = {}
    for event_name, max_boxes in EVENT_MAX_BOXES.items():
        event_properties[event_name] = {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "reason": {"type": ["string", "null"]},
                "boxes": {"type": "array", "items": box_schema, "maxItems": max_boxes},
            },
            "required": ["present", "reason", "boxes"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "events": {
                "type": "object",
                "properties": event_properties,
                "required": list(EVENT_MAX_BOXES),
                "additionalProperties": False,
            }
        },
        "required": ["events"],
        "additionalProperties": False,
    }



TASKS = ('cub', 'nlvr2', 'construction', 'grefcoco')
SPLITS = ('train', 'dev', 'val', 'test', 'test1', 'test2', 'testA', 'testB')
SERVED_NAME = 'router-generalist'


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--model-kind', choices=['Qwen3.8', 'MiniCPM'], required=True)
    p.add_argument('--data-path', type=Path, required=True,
                   help='包含 plan.json、classes.json、inputs/、prompts/ 的目录')
    p.add_argument('--output-path', type=Path, required=True)
    p.add_argument('--work-path', type=Path, help='本地工作目录；结果后台同步到 output-path')
    p.add_argument('--media-root', type=Path, required=True,
                   help='当前服务器图片的共同父目录；服务只允许读取这个范围')
    p.add_argument('--old-media-root', type=Path,
                   help='清单记录的旧图片根目录；迁移服务器时替换成 media-root')
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--tp', type=int, choices=[1, 2, 4, 8], default=1,
                   help='每个副本使用的卡数；副本数=卡数/tp，默认8副本×1卡')
    p.add_argument('--concurrency-per-replica', type=int, default=64)
    p.add_argument('--max-num-seqs', type=int, default=32)
    p.add_argument('--max-num-batched-tokens', type=int, default=8192)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    p.add_argument('--port-base', type=int, default=19100)
    p.add_argument('--request-timeout', type=float, default=900)
    p.add_argument('--startup-timeout', type=float, default=1800)
    p.add_argument('--limit-per-split', type=int,
                   help='测速时每个划分只运行前若干条；必须使用单独输出目录')
    p.add_argument('--check-only', action='store_true',
                   help='只检查数据、图片路径与数量，不加载模型，不创建输出')
    a = p.parse_args()
    a.work_path = (a.work_path or a.output_path).absolute()
    for name in ('model_path', 'data_path', 'output_path', 'media_root'):
        setattr(a, name, getattr(a, name).resolve())
    if a.old_media_root:
        if not a.old_media_root.is_absolute():
            p.error('--old-media-root 必须是绝对路径')
    a.gpu_ids = a.gpus.split(',')
    if len(a.gpu_ids) % a.tp or len(set(a.gpu_ids)) != len(a.gpu_ids):
        p.error('显卡编号不能重复，卡数必须能整除tp')
    if any(not item.isdigit() for item in a.gpu_ids):
        p.error('--gpus 使用逗号分隔的物理显卡编号')
    a.replicas = len(a.gpu_ids) // a.tp
    if min(a.concurrency_per_replica, a.max_num_seqs, a.max_num_batched_tokens) < 1:
        p.error('并发数和批处理上限必须为正数')
    if a.limit_per_split is not None and a.limit_per_split < 1:
        p.error('--limit-per-split 必须为正数')
    return a


def mapped_image(a, name):
    path = Path(name)
    if not path.is_absolute():
        path = a.media_root / path
    elif a.old_media_root:
        path = a.media_root / path.relative_to(a.old_media_root)
    # 路径归一化不逐层访问共享盘；图片存在性在输入检查时逐图检查一次。
    # 服务端仍根据 allowed-local-media-path 检查实际可访问范围。
    path = Path(os.path.abspath(path))
    path.relative_to(a.media_root)
    return path


def read_rows(a, task, split, limit=None):
    source = getattr(a, 'input_cache', a.data_path / 'inputs') / task / (split + '.jsonl')
    with source.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            if row['split'] != split:
                raise ValueError(f'划分不一致：{source}:{index + 1}')
            for field in (('image1', 'image2') if task == 'nlvr2' else ('image_path',)):
                row[field] = str(mapped_image(a, row[field]))
            yield index, row


def prepare(a):
    if not a.model_path.is_dir() or not a.media_root.is_dir():
        raise FileNotFoundError('模型目录或图片根目录不存在')
    plan = json.loads((a.data_path / 'plan.json').read_text(encoding='utf-8'))
    if set(plan['tasks']) != set(TASKS):
        raise ValueError('本脚本支持当前四任务清单，不接受未知任务')
    classes = json.loads((a.data_path / 'classes.json').read_text(encoding='utf-8'))
    names = '\n'.join(f"{r['category_id']}: {r['prompt_name']}" for r in classes)
    traits = '\n'.join(f"{r['category_id']}: {r['prompt_name']}; " +
                       '; '.join(f"{t['family']} {t['value']}" for t in r['traits'][:3])
                       for r in classes)
    prompts = {}
    split_counts = {}
    checked_images = set()
    for task in TASKS:
        # 忽略旧 plan 中的绝对提示词路径，始终相对用户指定的数据目录读取。
        prompt = (a.data_path / 'prompts' / task / (a.model_kind + '.txt')).read_text(encoding='utf-8').strip()
        if task == 'cub':
            if prompt.count('{class_names}') + prompt.count('{class_traits}') != 1:
                raise ValueError('CUB 提示词占位符错误')
            prompt = prompt.replace('{class_names}', names).replace('{class_traits}', traits)
        if task in ('nlvr2', 'grefcoco'):
            placeholder = '{sentence}' if task == 'nlvr2' else '{expression}'
            if prompt.count(placeholder) != 1:
                raise ValueError(f'{task} 提示词占位符错误')
        prompts[task] = prompt
        for split in SPLITS:
            if split not in plan['tasks'][task]['splits']:
                continue
            ids = set()
            for index, row in read_rows(a, task, split):
                if row['sample_id'] in ids:
                    raise ValueError(f'重复样本：{task}/{split}/{row["sample_id"]}')
                ids.add(row['sample_id'])
                if len(ids) % 20000 == 0:
                    print('INPUT_CHECK_PROGRESS', task, split, len(ids), flush=True)
                row['ground_truth']
                if task == 'nlvr2':
                    row['sentence']
                if task == 'grefcoco':
                    row['expression'], row['width'], row['height']
                for field in (('image1', 'image2') if task == 'nlvr2' else ('image_path',)):
                    path = row[field]
                    if a.check_only and path not in checked_images:
                        if not Path(path).is_file():
                            raise FileNotFoundError(path)
                        checked_images.add(path)
            if len(ids) != plan['tasks'][task]['splits'][split]:
                raise ValueError(f'{task}/{split} 清单数量和 plan 不一致')
            split_counts[f'{task}/{split}'] = min(len(ids), a.limit_per_split) if a.limit_per_split else len(ids)
            print('INPUT_CHECK', task, split, len(ids), flush=True)
    if sum(info['splits'][s] for info in plan['tasks'].values() for s in info['splits']) != plan['samples_per_model']:
        raise ValueError('plan 总样本数不一致')
    return prompts, split_counts


def make_payload(a, task, row, prompt):
    if task == 'nlvr2':
        prompt = prompt.replace('{sentence}', row['sentence'])
        content = [{'type': 'text', 'text': 'Left image:'},
                   {'type': 'image_url', 'image_url': {'url': Path(row['image1']).as_uri()}},
                   {'type': 'text', 'text': 'Right image:'},
                   {'type': 'image_url', 'image_url': {'url': Path(row['image2']).as_uri()}},
                   {'type': 'text', 'text': prompt}]
    else:
        if task == 'grefcoco':
            prompt = prompt.replace('{expression}', row['expression'])
        content = [{'type': 'image_url', 'image_url': {'url': Path(row['image_path']).as_uri()}},
                   {'type': 'text', 'text': prompt}]
    payload = dict(model=SERVED_NAME, messages=[dict(role='user', content=content)], temperature=0,
                   max_tokens=8 if task in ('cub', 'nlvr2') else 512 if task == 'grefcoco' else 768)
    if task in ('cub', 'nlvr2'):
        if a.model_kind == 'Qwen3.8':
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        else:
            payload['structured_outputs'] = {'json': {'type': 'integer', 'minimum': 1, 'maximum': 200}} if task == 'cub' else {'choice': ['True', 'False']}
    else:
        schema = {'type': 'object', 'properties': {'objects': {'type': 'array', 'maxItems': 18,
                  'items': {'type': 'object', 'properties': {
                      'bbox_2d': {'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 1000}, 'minItems': 4, 'maxItems': 4},
                      'score': {'type': 'number', 'minimum': 0, 'maximum': 1}},
                      'required': ['bbox_2d', 'score'], 'additionalProperties': False}}},
                  'required': ['objects'], 'additionalProperties': False}
        payload['structured_outputs'] = {'json': schema if task == 'grefcoco' else output_schema()}
    return payload


def score_answer(task, row, raw):
    answer = raw.strip() if isinstance(raw, str) else raw
    parsed = dict(parse_status='valid')
    try:
        if not isinstance(answer, str):
            raise ValueError('model content is not text')
        if task == 'cub':
            if not answer.isdigit() or answer != str(int(answer)) or not 1 <= int(answer) <= 200:
                raise ValueError('answer must be exactly one standard integer in 1..200')
            parsed['predicted_category_id'] = int(answer)
        elif task == 'nlvr2':
            if answer not in ('True', 'False'):
                raise ValueError('answer must be exactly True or False')
            parsed['prediction'] = answer
        elif task == 'grefcoco':
            parsed['objects'] = parse_gref(answer, row['width'], row['height'], 18)
        else:
            parsed['events'] = parse_events(answer)
    except (ValueError, TypeError) as error:
        parsed = dict(parse_status='invalid_output_counted_wrong', parse_error=str(error))
        if task == 'cub':
            parsed['predicted_category_id'] = None
        elif task == 'nlvr2':
            parsed['prediction'] = None
        elif task == 'grefcoco':
            parsed['objects'] = []
        else:
            parsed['events'] = None
    return parsed, judge(task, row, parsed)


def service_command(a, rank):
    cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
           '--model', str(a.model_path), '--served-model-name', SERVED_NAME,
           '--host', '127.0.0.1', '--port', str(a.port_base + rank),
           '--trust-remote-code', '--tensor-parallel-size', str(a.tp),
           '--max-model-len', str(a.max_model_len), '--max-num-seqs', str(a.max_num_seqs),
           '--max-num-batched-tokens', str(a.max_num_batched_tokens),
           '--gpu-memory-utilization', str(a.gpu_memory_utilization),
           '--limit-mm-per-prompt', '{"image":2}', '--allowed-local-media-path', str(a.media_root),
           '--enable-prefix-caching', '--enable-chunked-prefill', '--generation-config', 'vllm']
    if a.model_kind == 'Qwen3.8':
        cmd += ['--mm-processor-kwargs', '{"size":{"shortest_edge":65536,"longest_edge":1003520}}']
    return cmd


async def start_services(a, processes, stop):
    devices = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    used = {item.split(',')[0].strip(): int(item.split(',')[1]) for item in devices.splitlines()}
    for gpu in a.gpu_ids:
        if gpu not in used or used[gpu] >= 500:
            raise RuntimeError(f'显卡 {gpu} 不存在或已占用；本脚本不停止已有任务')
    for rank in range(a.replicas):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', a.port_base + rank))
    for rank in range(a.replicas):
        if stop.is_set():
            raise InterruptedError('用户在加载期间停止')
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = ','.join(a.gpu_ids[rank * a.tp:(rank + 1) * a.tp])
        env.setdefault('OMP_NUM_THREADS', '2')
        env.setdefault('TOKENIZERS_PARALLELISM', 'false')
        log = a.work_path / 'logs' / f'service_{rank}.log'
        with log.open('ab', buffering=0) as stream:
            process = subprocess.Popen(service_command(a, rank), env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        print('SERVICE_START', rank, env['CUDA_VISIBLE_DEVICES'], process.pid, flush=True)
        await asyncio.sleep(2)
    waiting = set(range(a.replicas))
    deadline = time.monotonic() + a.startup_timeout
    async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
        while waiting:
            if stop.is_set():
                raise InterruptedError('用户在加载期间停止')
            if time.monotonic() > deadline:
                raise TimeoutError(f'服务加载超时：{sorted(waiting)}，查看 logs/service_*.log')
            for rank in list(waiting):
                if processes[rank].poll() is not None:
                    raise RuntimeError(f'副本 {rank} 启动失败，查看 logs/service_{rank}.log')
                try:
                    response = await client.get(f'http://127.0.0.1:{a.port_base + rank}/health')
                except httpx.TransportError:
                    continue  # 仅加载健康探测可重复，不重发推理请求。
                if response.status_code == 200:
                    waiting.remove(rank)
                    print('SERVICE_READY', rank, flush=True)
            if waiting:
                await asyncio.sleep(2)


async def stop_services(processes):
    # 只停止本次创建的进程组，包括推理子进程，不触碰其他服务。
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 45
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        await asyncio.sleep(1)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await asyncio.to_thread(process.wait)


def save_json(path, value):
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    partial.replace(path)


async def collect_split(a, db, task, split, expected, prompt, stop, processes):
    completed = {r[0] for r in db.execute(
        "SELECT sample_id FROM samples WHERE task=? AND split=? AND status='complete'", (task, split))}
    if len(completed) > expected:
        raise ValueError('输出记录数超出当前清单')
    rows = ((index, row) for index, row in read_rows(a, task, split, expected)
            if row['sample_id'] not in completed)
    started = time.monotonic()
    initial = len(completed)
    failed = []
    total_latency = 0.0
    last_report = started
    checked_images = set()
    queue = asyncio.Queue(maxsize=1024)
    writer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='result-writer')

    def commit_batch(batch):
        with db:
            for sql, parameters, acknowledgement in batch:
                db.execute(sql, parameters)

    async def writer():
        while True:
            first = await queue.get()
            if first is None:
                return
            batch = [first]
            deadline = asyncio.get_running_loop().time() + 0.2
            while len(batch) < 100:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                batch.append(item)
            await asyncio.get_running_loop().run_in_executor(writer_pool, commit_batch, batch)
            for sql, parameters, acknowledgement in batch:
                acknowledgement.set_result(None)

    writer_task = asyncio.create_task(writer())

    async def persist(sql, parameters):
        acknowledgement = asyncio.get_running_loop().create_future()
        await queue.put((sql, parameters, acknowledgement))
        done, pending = await asyncio.wait([acknowledgement, writer_task], return_when=asyncio.FIRST_COMPLETED)
        if writer_task in done:
            await writer_task  # 写盘失败向上传递，不能让请求永远等确认。
            raise RuntimeError('保存线程提前退出')
        await acknowledgement

    async def progress():
        elapsed = time.monotonic() - started
        new = len(completed) - initial
        rate = new / elapsed if elapsed else 0
        value = dict(model=a.model_kind, task=task, split=split, completed=len(completed),
                     expected=expected, new_completed=new, elapsed_seconds=elapsed,
                     samples_per_second=rate,
                     remaining_seconds=(expected-len(completed))/rate if rate else None,
                     mean_request_seconds=total_latency/new if new else None,
                     request_errors=failed, updated_at=time.time(),
                     open_file_descriptors=len(list(Path('/proc/self/fd').iterdir())),
                     connection_pools=[repr(c._transport._pool) for c in clients],
                     save_queue_size=queue.qsize(), save_queue_limit=1024,
                     work_database=str(a.work_path / 'answers.sqlite'))
        await asyncio.to_thread(save_json, a.work_path / 'progress.json', value)
        print('PROGRESS', json.dumps(value, ensure_ascii=False), flush=True)
        return value

    async def worker(rank, client):
        nonlocal total_latency, last_report
        while not stop.is_set():
            if any(p.poll() is not None for p in processes):
                stop.set()
                raise RuntimeError('推理服务意外退出；保留未明确完成的请求，禁止自动重发')
            try:
                index, row = next(rows)
            except StopIteration:
                return
            for field in (('image1', 'image2') if task == 'nlvr2' else ('image_path',)):
                path = row[field]
                if path not in checked_images:
                    if not await asyncio.to_thread(Path(path).is_file):
                        stop.set()
                        raise FileNotFoundError(path)
                    checked_images.add(path)
            payload = make_payload(a, task, row, prompt)
            audit = dict(sample_id=row['sample_id'], split=split, model=a.model_kind,
                         model_path=str(a.model_path), replica=rank, tensor_parallel_size=a.tp,
                         model_requests_for_sample=0, request_failures=[], started_at=time.time(),
                         request_settings={k: v for k, v in payload.items() if k != 'messages'},
                         image_paths=[row['image1'], row['image2']] if task == 'nlvr2' else [row['image_path']])
            started_request = time.monotonic()
            status, label = 'complete', None
            response = None
            for attempt in range(4):
                # 已获用户授权：通信失败最多重试三次；提交前写明次数。
                audit['model_requests_for_sample'] = attempt + 1
                request_id = f'{task}/{split}/{row["sample_id"]}/{time.time_ns()}'
                audit['last_request_id'] = request_id
                if attempt == 0:
                    await persist('INSERT INTO samples VALUES (?,?,?,?,?,?,?)',
                                  (task, split, row['sample_id'], index, 'submitted',
                                   json.dumps(audit, ensure_ascii=False), None))
                else:
                    await persist('UPDATE samples SET audit=? WHERE task=? AND split=? AND sample_id=?',
                                  (json.dumps(audit, ensure_ascii=False), task, split, row['sample_id']))
                attempt_started = time.monotonic()
                response = None
                try:
                    url = f'http://127.0.0.1:{a.port_base + rank}/v1/chat/completions'
                    headers = {'X-Request-Id': request_id}
                    if attempt == 0:
                        response = await client.post(url, json=payload, headers=headers)
                    else:
                        # 重试独占新连接，退出时关闭，不扰动其他在途请求。
                        async with httpx.AsyncClient(timeout=a.request_timeout, trust_env=False,
                                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)) as retry_client:
                            response = await retry_client.post(url, json=payload, headers=headers)
                    audit['response_body'] = response.text
                    response.raise_for_status()
                except (httpx.TransportError, httpx.HTTPStatusError) as error:
                    failure = dict(attempt=attempt + 1, request_id=request_id, failed_at=time.time(),
                                   elapsed_seconds=time.monotonic() - attempt_started,
                                   request_error=str(error), exception_type=type(error).__name__,
                                   exception_repr=repr(error), exception_cause=repr(error.__cause__),
                                   exception_context=repr(error.__context__), traceback=traceback.format_exc(),
                                   connection_pool=repr(client._transport._pool),
                                   open_file_descriptors=len(list(Path('/proc/self/fd').iterdir())))
                    if response is not None:
                        failure.update(status_code=response.status_code, response_body=response.text)
                    audit['request_failures'].append(failure)
                    # 明确 HTTP 错误、配置错误不盲目重试；非法模型答案也不会进入这里。
                    retryable = isinstance(error, (httpx.NetworkError, httpx.RemoteProtocolError,
                                                  httpx.TimeoutException))
                    retrying = retryable and attempt < 3 and not stop.is_set() and all(p.poll() is None for p in processes)
                    await persist('UPDATE samples SET audit=? WHERE task=? AND split=? AND sample_id=?',
                                  (json.dumps(audit, ensure_ascii=False), task, split, row['sample_id']))
                    print('REQUEST_FAILURE', json.dumps(dict(sample_id=row['sample_id'], replica=rank,
                          retrying=retrying, **failure), ensure_ascii=False), flush=True)
                    if not retrying:
                        status = 'request_error'
                        audit['request_error'] = str(error)
                        break
                    await asyncio.sleep(2 ** attempt)
                    if stop.is_set() or any(p.poll() is not None for p in processes):
                        status = 'request_error'
                        audit['request_error'] = '重试前已停止或服务退出；原始错误保留在 request_failures'
                        break
                else:
                    break
                finally:
                    if response is not None:
                        await response.aclose()
            if status == 'complete':
                try:
                    choice = response.json()['choices'][0]
                    raw = choice['message']['content']
                    audit['raw_answer'] = raw
                    audit['finish_reason'] = choice.get('finish_reason')
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    status = 'request_error'
                    audit['request_error'] = 'Malformed service response: ' + str(error)
                else:
                    audit['generalist_output'], label = score_answer(task, row, raw)
            audit['elapsed_seconds'] = time.monotonic() - started_request
            await persist('UPDATE samples SET status=?,audit=?,label=? WHERE task=? AND split=? AND sample_id=?',
                          (status, json.dumps(audit, ensure_ascii=False),
                           json.dumps(label, ensure_ascii=False) if label is not None else None,
                           task, split, row['sample_id']))
            if status == 'complete':
                completed.add(row['sample_id'])
                total_latency += audit['elapsed_seconds']
            else:
                failed.append(row['sample_id'])
                stop.set()  # 不再发新请求，已发出的请求仍收完保存。
            if time.monotonic() - last_report >= 10:
                last_report = time.monotonic()
                await progress()

    clients = [httpx.AsyncClient(timeout=a.request_timeout, trust_env=False,
               limits=httpx.Limits(max_connections=a.concurrency_per_replica,
                                  max_keepalive_connections=a.concurrency_per_replica))
               for _ in range(a.replicas)]
    try:
        # 每个副本有独立连接池，所有副本共同领取剩余样本，快卡自动多做。
        jobs = [asyncio.create_task(worker(rank, clients[rank]))
                for rank in range(a.replicas) for _ in range(a.concurrency_per_replica)]
        done, pending = await asyncio.wait(jobs, return_when=asyncio.FIRST_EXCEPTION)
        if any(job.exception() is not None for job in done):
            stop.set()
        # 出错也先等其他已提交请求落盘，再把异常原样抛出。
        results = await asyncio.gather(*jobs, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
    finally:
        for client in clients:
            await client.aclose()
        if not writer_task.done():
            await queue.put(None)  # worker 全部结束后才发退出标记，队列此时已确认写完。
        try:
            await writer_task
        finally:
            writer_pool.shutdown(wait=True)
    summary = await progress()
    if failed:
        raise RuntimeError(f'服务请求失败：{failed}；已按策略有限重试或错误不可重试，没有生成对错标签')
    if len(completed) != expected:
        raise InterruptedError('已停止补充请求，已提交请求已保存；再次运行同命令可继续')
    destination = a.work_path / task / split
    destination.mkdir(parents=True, exist_ok=True)
    correct = invalid = count = 0
    records = db.execute('SELECT sample_id,audit,label FROM samples WHERE task=? AND split=? ORDER BY input_index',
                         (task, split))
    selected = (row for index, row in read_rows(a, task, split, expected))
    with (destination / 'labels.partial.jsonl').open('w', encoding='utf-8') as labels, \
         gzip.open(destination / 'audit.partial.jsonl.gz', 'wt', encoding='utf-8') as audits:
        for row, (sample_id, audit, label) in zip(selected, records, strict=True):
            if row['sample_id'] != sample_id or label is None:
                raise ValueError('输出与输入清单不一致')
            value = json.loads(label)
            correct += value['correct']
            invalid += not value['valid_output']
            labels.write(label + '\n')
            audits.write(audit + '\n')
            count += 1
    if count != expected:
        raise ValueError('导出数量错误')
    (destination / 'labels.partial.jsonl').replace(destination / 'labels.jsonl')
    (destination / 'audit.partial.jsonl.gz').replace(destination / 'audit.jsonl.gz')
    summary.update(rows=count, correct=correct, invalid=invalid,
                   sample_correctness_rate=correct/count,
                   metric_note='construction correctness is exact event set, NOT macro F1')
    # 已完成划分保留第一次完整测速记录，续跑不覆盖为零吞吐。
    if not (destination / 'completion.json').exists():
        save_json(destination / 'completion.json', summary)


def publish_checkpoint(a):
    # 先在本地形成一致快照；共享存储复制完全不持有工作库连接。
    snapshot = a.work_path / 'checkpoint.sqlite'
    source = sqlite3.connect(f'file:{a.work_path / "answers.sqlite"}?mode=ro', uri=True)
    destination = sqlite3.connect(snapshot)
    try:
        source.backup(destination)
        destination.execute('PRAGMA journal_mode=DELETE')
        counts = destination.execute('SELECT status,count(*) FROM samples GROUP BY status').fetchall()
    finally:
        destination.close()
        source.close()
    temporary = a.output_path / 'answers_checkpoint.sqlite.partial'
    shutil.copyfile(snapshot, temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(a.output_path / 'answers_checkpoint.sqlite')
    save_json(a.output_path / 'checkpoint_status.json', dict(published_at=time.time(), counts=dict(counts),
              source=str(a.work_path / 'answers.sqlite'), interval_seconds=60))
    if (a.work_path / 'progress.json').exists():
        shutil.copyfile(a.work_path / 'progress.json', a.output_path / 'progress.json.partial')
        (a.output_path / 'progress.json.partial').replace(a.output_path / 'progress.json')
    for task in TASKS:
        directory = a.work_path / task
        if directory.exists():
            shutil.copytree(directory, a.output_path / task, dirs_exist_ok=True)
    shutil.copytree(a.work_path / 'logs', a.output_path / 'logs', dirs_exist_ok=True)
    if (a.work_path / 'runner.log').exists():
        shutil.copyfile(a.work_path / 'runner.log', a.output_path / 'runner.log')


async def run(a, prompts, counts, db):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    processes = []
    started = time.monotonic()
    completion = None
    backup_stop = asyncio.Event()

    async def checkpoints():
        while not backup_stop.is_set():
            await asyncio.to_thread(publish_checkpoint, a)
            try:
                await asyncio.wait_for(backup_stop.wait(), timeout=60)
            except TimeoutError:
                continue
        await asyncio.to_thread(publish_checkpoint, a)

    checkpoint_task = asyncio.create_task(checkpoints()) if a.work_path != a.output_path else None
    if checkpoint_task is not None:
        checkpoint_task.add_done_callback(lambda task: stop.set() if not task.cancelled() and task.exception() is not None else None)
    try:
        await start_services(a, processes, stop)
        for key, expected in counts.items():
            task, split = key.split('/')
            if stop.is_set():
                raise InterruptedError('停止后不再开始下一个划分')
            await collect_split(a, db, task, split, expected, prompts[task], stop, processes)
        total = db.execute("SELECT count(*) FROM samples WHERE status='complete'").fetchone()[0]
        if total != sum(counts.values()):
            raise ValueError('最终记录数不一致')
        completion = dict(status='complete', rows=total, subset=a.limit_per_split is not None,
                          model=a.model_kind, wall_seconds_this_invocation=time.monotonic()-started,
                          all_splits_verified=True)
    finally:
        try:
            await stop_services(processes)
        finally:
            if checkpoint_task is not None:
                backup_stop.set()
                await checkpoint_task
    if completion is not None:
        save_json(a.output_path / 'completion.json', completion)


def main():
    a = arguments()
    prompts, counts = prepare(a)
    print('PLAN', json.dumps(dict(samples=sum(counts.values()), replicas=a.replicas, tp=a.tp,
          inflight_limit=a.replicas*a.concurrency_per_replica,
          active_sequence_limit=a.replicas*a.max_num_seqs)), flush=True)
    if a.check_only:
        return
    version = importlib.metadata.version('vllm')
    a.output_path.mkdir(parents=True, exist_ok=True)
    # 防止同一个输出目录被两个采集进程同时使用。
    with (a.output_path / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = dict(model_path=str(a.model_path), model_kind=a.model_kind, data_path=str(a.data_path),
                      media_root=str(a.media_root), old_media_root=str(a.old_media_root),
                      tp=a.tp, vllm_version=version, max_model_len=a.max_model_len,
                      counts=counts, prompts=prompts, mtp=False,
                      scoring='Original frozen four-task rules; invalid answer is wrong; transport failure has no label')
        config = a.output_path / 'settings.json'
        if (a.output_path / 'answers.sqlite').exists() and not config.exists():
            raise ValueError('输出目录已有其他采集程序的数据库，请指定新目录，禁止混写旧结果')
        if config.exists():
            if json.loads(config.read_text(encoding='utf-8')) != frozen:
                raise ValueError('输出目录的模型、数据、提示词或判分运行配置不同，请指定新目录')
        else:
            save_json(config, frozen)
        if (a.output_path / 'completion.json').exists():
            print('ALREADY_COMPLETE', a.output_path, flush=True)
            return
        (a.output_path / 'logs').mkdir(exist_ok=True)
        a.work_path.mkdir(parents=True, exist_ok=True)
        (a.work_path / 'logs').mkdir(exist_ok=True)
        if a.work_path != a.output_path:
            origin = a.work_path / 'origin.json'
            identity = dict(output_path=str(a.output_path), model_path=str(a.model_path))
            if origin.exists() and json.loads(origin.read_text()) != identity:
                raise ValueError('本地工作目录属于其他任务，禁止混用')
            if not origin.exists() and (a.work_path / 'answers.sqlite').exists():
                raise ValueError('本地数据库来源未记录，禁止直接使用')
            save_json(origin, identity)
            if not (a.work_path / 'answers.sqlite').exists():
                if (a.output_path / 'work_database.json').exists():
                    raise RuntimeError('本地工作库缺失；共享快照可能落后，需要人工审计后恢复，禁止自动重发')
                previous = a.output_path / 'answers_checkpoint.sqlite'
                if not previous.exists():
                    previous = a.output_path / 'answers.sqlite'
                if previous.exists():
                    with sqlite3.connect(f'file:{previous}?mode=ro', uri=True) as source:
                        destination = sqlite3.connect(a.work_path / 'answers.sqlite')
                        try:
                            source.backup(destination)
                        finally:
                            destination.close()
                    source.close()
            save_json(a.output_path / 'work_database.json', dict(path=str(a.work_path / 'answers.sqlite'),
                      checkpoint='answers_checkpoint.sqlite', interval_seconds=60,
                      missing_local_database='requires_manual_audit', original_database='answers.sqlite is pre-migration archive'))
            for task in TASKS:
                for previous in (a.output_path / task).glob('*/completion.json'):
                    destination = a.work_path / previous.relative_to(a.output_path)
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(previous, destination)
            a.input_cache = a.work_path / 'inputs'
            shutil.copytree(a.data_path / 'inputs', a.input_cache, dirs_exist_ok=True)
        save_json(a.output_path / 'launch.json', dict(arguments={k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(a).items()}, commands=[service_command(a, r) for r in range(a.replicas)],
                  transport_policy='Network/remote-protocol/timeout errors: at most 3 retries, 1/2/4 seconds; fresh connection on retries; no retry for HTTP errors or invalid model answers',
                  started_at=time.time()))
        db = sqlite3.connect(a.work_path / 'answers.sqlite', check_same_thread=False)
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS samples(task TEXT,split TEXT,sample_id TEXT,input_index INTEGER,status TEXT,audit TEXT,label TEXT,PRIMARY KEY(task,split,sample_id))')
            db.commit()
            unresolved = db.execute("SELECT task,split,sample_id,status FROM samples WHERE status!='complete' LIMIT 5").fetchall()
            if unresolved:
                raise RuntimeError(f'有未明确完成的旧请求，禁止自动重发，需人工审计：{unresolved}')
            asyncio.run(run(a, prompts, counts, db))
        finally:
            db.close()


if __name__ == '__main__':
    main()
