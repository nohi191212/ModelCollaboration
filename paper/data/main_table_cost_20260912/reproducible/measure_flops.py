#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode


def count(call):
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        value = call()
    torch.cuda.synchronize()
    counts = counter.get_flop_counts().get('Global', {})
    return int(counter.get_total_flops()), value, {str(k): int(v) for k, v in counts.items()}


def make_text(root: Path, model_kind: str, task: str, row: dict, processor):
    prompt = (root / 'outputs/large_model_labels_all_20260909/prompts' / task / f'{model_kind}.txt').read_text(encoding='utf-8').strip()
    if task == 'grefcoco':
        prompt = prompt.replace('{expression}', row['expression'])
    if model_kind == 'Qwen3.8':
        content = [{'type': 'image'}, {'type': 'text', 'text': '\n'}, {'type': 'text', 'text': prompt}]
        return processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    message = '(<image>./</image>)\n' + prompt
    return processor.tokenizer.apply_chat_template([{'role': 'user', 'content': message}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prepare(root: Path, model_kind: str, task: str, row: dict, processor):
    image = Image.open(row['image_path']).convert('RGB')
    text = make_text(root, model_kind, task, row, processor)
    if model_kind == 'Qwen3.8':
        data = processor(text=[text], images=[image], return_tensors='pt', size={'shortest_edge': 65536, 'longest_edge': 1003520})
    else:
        data = processor([text], [[image]], return_tensors='pt', max_length=16384)
    image.close()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-kind', choices=['Qwen3.8', 'MiniCPM'], required=True)
    parser.add_argument('--task', choices=['grefcoco', 'construction'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expert = 'instancevg' if args.task == 'grefcoco' else 'yolo26x'
    rows = [json.loads(line) for line in (args.root / f'outputs/router_full_20260910/data/{expert}/test/records.jsonl').open(encoding='utf-8')]
    if args.samples > len(rows):
        raise ValueError('sample count exceeds test rows')
    selected_indices = np.random.default_rng(20260911).choice(len(rows), args.samples, replace=False)
    selected = [rows[int(index)] for index in selected_indices]
    checkpoint_root = args.root.parents[1] / 'checkpoints'
    if not checkpoint_root.exists():
        checkpoint_root = args.root / 'checkpoints'
    model_path = checkpoint_root / ('Qwen3.8-27B-FP8' if args.model_kind == 'Qwen3.8' else 'MiniCPM-V-4_5')
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if args.model_kind == 'Qwen3.8':
        from transformers import FineGrainedFP8Config, Qwen3_5ForConditionalGeneration
        import transformers.models.qwen3_5.modeling_qwen3_5 as implementation
        implementation.causal_conv1d_fn = None
        implementation.causal_conv1d_update = None
        implementation.chunk_gated_delta_rule = None
        implementation.fused_recurrent_gated_delta_rule = None
        implementation.FusedRMSNormGated = None
        implementation.is_fast_path_available = False
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            attn_implementation='eager', quantization_config=FineGrainedFP8Config(dequantize=False),
        )
    else:
        model = processor  # replaced below to keep imports local to the selected environment
        from transformers import AutoModel
        from transformers.modeling_utils import PreTrainedModel
        if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
            def get_tied_weights(self):
                return getattr(self, '_all_tied_weights_keys', getattr(self, '_tied_weights_keys', {}) or {})
            def set_tied_weights(self, value):
                self._all_tied_weights_keys = value
            PreTrainedModel.all_tied_weights_keys = property(get_tied_weights, set_tied_weights)
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, attn_implementation='eager',
        )
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    model.eval().requires_grad_(False)
    records = []
    for row in selected:
        data = prepare(args.root, args.model_kind, args.task, row, processor).to('cuda')
        prompt_tokens = int(data['input_ids'].shape[-1])
        if args.model_kind == 'MiniCPM':
            vision_flops, embedding_result, vision_ops = count(lambda: model.get_vllm_embedding(data))
            embeddings, _ = embedding_result
            prefill_flops, prefill_output, prefill_ops = count(lambda: model.llm(inputs_embeds=embeddings, attention_mask=data['attention_mask'], use_cache=True, logits_to_keep=1))
            prefill_flops += vision_flops
            prefill_ops.update({'vision:'+k:v for k,v in vision_ops.items()})
            length = int(data['input_ids'].shape[-1])
            known_token = data['input_ids'][:, -1:]
            kwargs = {
                'input_ids': known_token,
                'past_key_values': prefill_output.past_key_values,
                'attention_mask': torch.ones((1, length + 1), device='cuda', dtype=torch.long),
                'use_cache': True,
                'logits_to_keep': 1,
            }
            decode_flops, _, decode_ops = count(lambda: model.llm(**kwargs))
        else:
            prefill_flops, prefill_output, prefill_ops = count(lambda: model(**data, use_cache=True, logits_to_keep=1))
            length = int(data['input_ids'].shape[-1])
            known_token = data['input_ids'][:, -1:]
            kwargs = {
                'input_ids': known_token,
                'past_key_values': prefill_output.past_key_values,
                'attention_mask': torch.ones((1, length + 1), device='cuda', dtype=torch.long),
                'use_cache': True,
                'logits_to_keep': 1,
                'cache_position': torch.tensor([length], device='cuda'),
            }
            kwargs.pop('attention_mask')
            decode_flops, _, decode_ops = count(lambda: model(**kwargs))
        records.append({
            'sample_id': row['sample_id'], 'prompt_tokens_in_reference_tensor': prompt_tokens,
            'input_shapes': {key: list(value.shape) for key, value in data.items() if hasattr(value, 'shape')},
            'prefill_flops_counted': prefill_flops, 'decode_one_token_flops_counted': decode_flops,
            'prefill_gflops_counted': prefill_flops / 1e9, 'decode_one_token_gflops_counted': decode_flops / 1e9,
            'prefill_operators': prefill_ops, 'decode_operators': decode_ops,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'status': 'running', 'model_kind': args.model_kind, 'task': args.task, 'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'completed': len(records), 'sample_id': row['sample_id'], 'prefill_gflops': prefill_flops / 1e9, 'decode_gflops': decode_flops / 1e9}), flush=True)
    args.output.write_text(json.dumps({
        'status': 'complete', 'model_kind': args.model_kind, 'task': args.task, 'model_path': str(model_path),
        'sample_count': len(records), 'records': records,
        'mean_prefill_gflops_counted': float(np.mean([row['prefill_gflops_counted'] for row in records])),
        'mean_decode_one_token_gflops_counted': float(np.mean([row['decode_one_token_gflops_counted'] for row in records])),
        'scope': 'HF reference execution on the A800 with real task inputs; dense products/convolutions/supported attention counted with multiply-add=2 FLOPs. vLLM custom kernels, elementwise/normalization/nonlinearities and other unsupported operations are not complete.',
        'answer_generation': 'not performed; one known-token decode probe only',
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
