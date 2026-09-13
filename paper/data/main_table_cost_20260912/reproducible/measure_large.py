#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import jsonschema
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


EVENTS = [
    'rule_1_ppe_violation',
    'rule_2_fall_protection_violation',
    'rule_3_unprotected_edge_violation',
    'rule_4_excavator_proximity_violation',
]


def gref_schema():
    return {
        'type': 'object',
        'properties': {
            'objects': {
                'type': 'array', 'maxItems': 18,
                'items': {
                    'type': 'object',
                    'properties': {
                        'bbox_2d': {'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 1000}, 'minItems': 4, 'maxItems': 4},
                        'score': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    },
                    'required': ['bbox_2d', 'score'], 'additionalProperties': False,
                },
            },
        },
        'required': ['objects'], 'additionalProperties': False,
    }


def construction_schema():
    box = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000}, 'minItems': 4, 'maxItems': 4}
    max_boxes = [13, 6, 3, 2]
    events = {}
    for event, limit in zip(EVENTS, max_boxes):
        events[event] = {
            'type': 'object',
            'properties': {
                'present': {'type': 'boolean'},
                'reason': {'type': ['string', 'null']},
                'boxes': {'type': 'array', 'items': box, 'maxItems': limit},
            },
            'required': ['present', 'reason', 'boxes'], 'additionalProperties': False,
        }
    return {'type': 'object', 'properties': {'events': {'type': 'object', 'properties': events, 'required': EVENTS, 'additionalProperties': False}}, 'required': ['events'], 'additionalProperties': False}


def make_prompt(root: Path, model_kind: str, task: str, row: dict, processor_or_tokenizer):
    prompt = (root / 'outputs/large_model_labels_all_20260909/prompts' / task / f'{model_kind}.txt').read_text(encoding='utf-8').strip()
    if task == 'grefcoco':
        prompt = prompt.replace('{expression}', row['expression'])
    if model_kind == 'Qwen3.8':
        converted = [{'type': 'image'}]
        converted.append({'type': 'text', 'text': '\n'})
        converted.append({'type': 'text', 'text': prompt})
        messages = [{'role': 'user', 'content': converted}]
        return processor_or_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    message = '(<image>./</image>)\n' + prompt
    return processor_or_tokenizer.apply_chat_template([{'role': 'user', 'content': message}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-kind', choices=['Qwen3.8', 'MiniCPM'], required=True)
    parser.add_argument('--task', choices=['grefcoco', 'construction'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--max-model-len', type=int, default=16384)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.samples < 3 or args.batch_size < 1:
        raise ValueError('samples must include warmup rows and batch size must be positive')
    expert = 'instancevg' if args.task == 'grefcoco' else 'yolo26x'
    rows = [json.loads(line) for line in (args.root / f'outputs/router_full_20260910/data/{expert}/test/records.jsonl').open(encoding='utf-8')]
    if args.samples > len(rows):
        raise ValueError(f'samples {args.samples} exceeds {len(rows)}')
    selected_indices = np.random.default_rng(20260911).choice(len(rows), args.samples, replace=False)
    selected = [rows[int(index)] for index in selected_indices]
    checkpoint_root = args.root.parents[1] / 'checkpoints'
    if not checkpoint_root.exists():
        checkpoint_root = args.root / 'checkpoints'
    model_path = checkpoint_root / ('Qwen3.8-27B-FP8' if args.model_kind == 'Qwen3.8' else 'MiniCPM-V-4_5')
    if args.model_kind == 'Qwen3.8':
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        model_kwargs = {'mm_processor_kwargs': {'size': {'shortest_edge': 65536, 'longest_edge': 1003520}}}
    else:
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        model_kwargs = {}
    model = LLM(model=str(model_path), trust_remote_code=True, tensor_parallel_size=1, max_model_len=args.max_model_len,
                limit_mm_per_prompt={'image': 1}, max_num_seqs=args.batch_size,
                gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, **model_kwargs)
    schema = gref_schema() if args.task == 'grefcoco' else construction_schema()
    sampling = SamplingParams(temperature=0, max_tokens=512 if args.task == 'grefcoco' else 768,
                              structured_outputs=StructuredOutputsParams(json=schema, disable_additional_properties=True))

    def generate(batch_rows):
        images = []
        requests = []
        for row in batch_rows:
            required = {'sample_id', 'image_path'}
            if args.task == 'grefcoco':
                required.add('expression')
            missing = required - row.keys()
            if missing:
                raise ValueError(f'missing fields {sorted(missing)} for {row["sample_id"]}')
            image = Image.open(row['image_path']).convert('RGB')
            images.append(image)
            requests.append({'prompt': make_prompt(args.root, args.model_kind, args.task, row, processor), 'multi_modal_data': {'image': image}})
        started = time.perf_counter()
        generated = model.generate(requests, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        for image in images:
            image.close()
        if len(generated) != len(batch_rows):
            raise ValueError('vLLM returned an unexpected number of results')
        return elapsed, generated

    warmup_count = min(2, len(selected) - 1)
    for row in selected[:warmup_count]:
        generate([row])
    measured = selected[warmup_count:]
    batches = []
    for start in range(0, len(measured), args.batch_size):
        batch_rows = measured[start:start + args.batch_size]
        batch_started = time.perf_counter()
        elapsed, generated = generate(batch_rows)
        wall = time.perf_counter() - batch_started
        output_records = []
        for result in generated:
            raw = result.outputs[0].text
            error = None
            try:
                parsed = json.loads(raw)
                jsonschema.validate(parsed, schema)
                boxes = [o['bbox_2d'] for o in parsed['objects']] if args.task == 'grefcoco' else [b for e in parsed['events'].values() for b in e['boxes']]
                if any(b[0] >= b[2] or b[1] >= b[3] for b in boxes):
                    raise ValueError('non-positive box size')
            except (json.JSONDecodeError, jsonschema.ValidationError, ValueError) as exc:
                parsed = None
                error = str(exc)
            output_records.append({'raw_response':raw,'valid_prediction':parsed,'error':error,
                'finish_reason':result.outputs[0].finish_reason,'completion_tokens':len(result.outputs[0].token_ids),
                'prompt_tokens':len(result.prompt_token_ids)})
        item = {
            'sample_ids': [row['sample_id'] for row in batch_rows],
            'batch_size': len(batch_rows),
            'generation_latency_seconds': elapsed,
            'batch_wall_latency_seconds': wall,
            'amortized_generation_ms_per_sample': elapsed * 1000 / len(batch_rows),
            'amortized_batch_wall_ms_per_sample': wall * 1000 / len(batch_rows),
            'outputs': [
                {
                    'finish_reason': result.outputs[0].finish_reason,
                    'stop_reason': result.outputs[0].stop_reason,
                    'prompt_tokens': len(result.prompt_token_ids) if result.prompt_token_ids is not None else None,
                    'completion_tokens': len(result.outputs[0].token_ids),
                    'text_characters': len(result.outputs[0].text),
                }
                for result in generated
            ],
        }
        batches.append(item)
        item['outputs'] = output_records
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'status': 'running', 'model_kind': args.model_kind, 'task': args.task, 'model_path': str(model_path),
            'sample_count_requested': args.samples, 'warmup_count': warmup_count, 'batch_size': args.batch_size,
            'sample_ids': [row['sample_id'] for row in selected], 'batches': batches,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'completed_samples': sum(item['batch_size'] for item in batches), 'total_measured': len(measured), 'batch_latency_seconds': elapsed}), flush=True)
    all_ms = [x['amortized_generation_ms_per_sample'] for x in batches]
    all_wall_ms = [x['amortized_batch_wall_ms_per_sample'] for x in batches]
    output = {
        'status': 'complete', 'model_kind': args.model_kind, 'task': args.task, 'expert': expert,
        'model_path': str(model_path), 'device': 'cuda:0', 'gpu_memory_utilization': args.gpu_memory_utilization,
        'sample_count_requested': args.samples, 'warmup_count': warmup_count, 'measured_count': len(measured),
        'batch_size': args.batch_size, 'sample_ids': [row['sample_id'] for row in selected], 'batches': batches,
        'mean_generation_ms_per_sample': float(np.mean(all_ms)), 'median_generation_ms_per_sample': float(np.median(all_ms)),
        'mean_batch_wall_ms_per_sample': sum(b['batch_wall_latency_seconds'] for b in batches)*1000/len(measured), 'median_batch_wall_ms_per_sample': float(np.median(all_wall_ms)),
        'mean_prompt_tokens': float(np.mean([x['prompt_tokens'] for batch in batches for x in batch['outputs'] if x['prompt_tokens'] is not None])),
        'mean_completion_tokens': float(np.mean([x['completion_tokens'] for batch in batches for x in batch['outputs']])),
        'latency_scope': 'A800 vLLM-0.18.0 native generation latency with the frozen task prompt and structured-output schema; model loading and warmup excluded. Batch wall time includes the real request/image preparation path.',
        'flops_scope': 'This file measures latency only. Counted FLOPs are measured separately by the HF reference probe and are not claimed to be complete vLLM kernel FLOPs.',
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'mean_generation_ms_per_sample': output['mean_generation_ms_per_sample'], 'mean_completion_tokens': output['mean_completion_tokens']}), flush=True)


if __name__ == '__main__':
    main()
