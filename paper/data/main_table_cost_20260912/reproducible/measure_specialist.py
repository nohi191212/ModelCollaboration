#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode


def timed_cuda(call):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = call()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), value


def count_flops(call):
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        value = call()
    torch.cuda.synchronize()
    global_counts = counter.get_flop_counts().get('Global', {})
    return int(counter.get_total_flops()), value, {str(k): int(v) for k, v in global_counts.items()}


def instancevg_measure(root: Path, output: Path, samples: int, batch_size: int, device: str) -> None:
    import mmcv
    from mmcv import Config
    from transformers import XLMRobertaTokenizer
    from instancevg.models import build_model
    from instancevg.datasets.pipelines.transforms import Normalize, Resize

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(str(root / 'code/third_party/InstanceVG/configs/gres/InstanceVG-grefcoco.py'))
    cfg.model.vis_enc.pretrain = None
    cfg.model.process_visual = False
    model = build_model(cfg.model)
    checkpoint_path = root / 'models/grefcoco/instancevg/InstanceVG-grefcoco.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['state_dict']
    if all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    model.eval().to(device).requires_grad_(False)
    tokenizer_path = root / 'models/nlvr2/Raghavan--beit3_base_patch16_224_nlvr2/sentencepiece.bpe.model'
    tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
    resize = Resize(img_scale=(320, 320), keep_ratio=False)
    normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    rows = [json.loads(line) for line in (root / 'outputs/router_full_20260910/data/instancevg/test/records.jsonl').open(encoding='utf-8')]
    if samples > len(rows):
        raise ValueError(f'samples {samples} exceeds {len(rows)}')
    indices = np.random.default_rng(20260911).choice(len(rows), samples, replace=False)
    chosen = [rows[int(index)] for index in indices]

    def prepare(batch_rows):
        images, metas, tokens, masks = [], [], [], []
        for row in batch_rows:
            image = mmcv.imread(row['image_path'])
            if image.shape[:2] != (row['height'], row['width']):
                raise ValueError(f'image shape mismatch for {row["sample_id"]}')
            expression = row['expression']
            expression = __import__('re').sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(expression))
            if not token_ids:
                raise ValueError(f'empty token sequence for {row["sample_id"]}')
            token_ids = [tokenizer.bos_token_id] + token_ids[:48] + [tokenizer.eos_token_id]
            if len(token_ids) > 50:
                raise ValueError(f'token sequence exceeds fixed length for {row["sample_id"]}')
            masks.append([0] * len(token_ids) + [1] * (50 - len(token_ids)))
            tokens.append(token_ids + [tokenizer.pad_token_id] * (50 - len(token_ids)))
            data = dict(img=image, filename=row['image_path'], img_shape=image.shape, ori_shape=image.shape,
                        expression=expression, empty=None, with_bbox=False, with_mask=False)
            data = normalize(resize(data))
            images.append(torch.from_numpy(np.ascontiguousarray(data['img'].transpose(2, 0, 1))))
            metas.append({key: data[key] for key in ['filename', 'expression', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'empty']})
        image_tensor = torch.stack(images).to(device)
        token_tensor = torch.tensor(tokens, device=device)
        mask_tensor = torch.tensor(masks, device=device)

        def forward():
            return model(img=image_tensor, ref_expr_inds=token_tensor, text_attention_mask=mask_tensor,
                         img_metas=metas, return_loss=False, rescale=True, with_bbox=True, with_mask=False)

        return forward

    batches = [chosen[start:start + batch_size] for start in range(0, len(chosen), batch_size)]
    prepared = [prepare(batch) for batch in batches]
    for forward in prepared[:2]:
        forward()
        torch.cuda.synchronize()
    forward_ms = []
    end_to_end_ms = []
    for batch in batches:
        forward = prepare(batch)
        elapsed, _ = timed_cuda(forward)
        forward_ms.append(elapsed / len(batch))
        started = time.perf_counter()
        forward = prepare(batch)
        forward()
        torch.cuda.synchronize()
        end_to_end_ms.append((time.perf_counter() - started) * 1000 / len(batch))
    flop_rows = []
    for batch in batches[:min(4, len(batches))]:
        forward = prepare(batch)
        total, _, operators = count_flops(forward)
        flop_rows.append({'batch_size': len(batch), 'flops_counted': total, 'gflops_counted_per_sample': total / len(batch) / 1e9, 'operators': operators})
    output.write_text(json.dumps({
        'component': 'specialist', 'expert': 'instancevg', 'task': 'grefcoco', 'device': device,
        'checkpoint': str(checkpoint_path), 'sample_count': len(chosen), 'batch_size': batch_size,
        'sample_ids': [row['sample_id'] for row in chosen],
        'model_load_seconds_excluded_from_latency': True,
        'forward_latency_ms_per_sample': forward_ms,
        'end_to_end_latency_ms_per_sample': end_to_end_ms,
        'mean_forward_latency_ms_per_sample': float(np.mean(forward_ms)),
        'median_forward_latency_ms_per_sample': float(np.median(forward_ms)),
        'mean_end_to_end_latency_ms_per_sample': float(np.mean(end_to_end_ms)),
        'median_end_to_end_latency_ms_per_sample': float(np.median(end_to_end_ms)),
        'flop_profiles': flop_rows,
        'mean_counted_gflops_per_sample': float(np.mean([row['gflops_counted_per_sample'] for row in flop_rows])),
        'flops_scope': 'GPU FlopCounterMode dense products, convolutions and supported attention; multiply-add counted as 2 FLOPs. Elementwise, normalization, custom ops and postprocessing may be missing.',
        'latency_scope': 'A800 wall-clock CUDA forward latency and end-to-end latency including image read/preprocess, excluding model load; batch processing uses the recorded batch size.',
        'completed': True,
    }, ensure_ascii=False, indent=2))


def yolo_measure(root: Path, output: Path, samples: int, batch_size: int, device: str) -> None:
    from ultralytics import YOLO

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_path = root / 'models/construction/yolo26x.pt'
    model = YOLO(str(checkpoint_path))
    rows = [json.loads(line) for line in (root / 'outputs/router_full_20260910/data/yolo26x/test/records.jsonl').open(encoding='utf-8')]
    if samples > len(rows):
        raise ValueError(f'samples {samples} exceeds {len(rows)}')
    indices = np.random.default_rng(20260911).choice(len(rows), samples, replace=False)
    chosen = [rows[int(index)] for index in indices]
    batches = [chosen[start:start + batch_size] for start in range(0, len(chosen), batch_size)]

    def predict(batch_rows):
        return model.predict(source=[row['image_path'] for row in batch_rows], imgsz=640, batch=len(batch_rows),
                             device=device, half=False, conf=0.001, iou=0.7, max_det=300, save=False, verbose=False)

    for batch in batches[:2]:
        predict(batch)
        torch.cuda.synchronize()
    forward_ms, end_to_end_ms = [], []
    for batch in batches:
        elapsed, _ = timed_cuda(lambda batch=batch: predict(batch))
        forward_ms.append(elapsed / len(batch))
        started = time.perf_counter()
        predict(batch)
        torch.cuda.synchronize()
        end_to_end_ms.append((time.perf_counter() - started) * 1000 / len(batch))
    flop_rows = []
    for batch in batches[:min(4, len(batches))]:
        total, _, operators = count_flops(lambda batch=batch: predict(batch))
        flop_rows.append({'batch_size': len(batch), 'flops_counted': total, 'gflops_counted_per_sample': total / len(batch) / 1e9, 'operators': operators})
    output.write_text(json.dumps({
        'component': 'specialist', 'expert': 'yolo26x', 'task': 'construction', 'device': device,
        'checkpoint': str(checkpoint_path), 'sample_count': len(chosen), 'batch_size': batch_size,
        'sample_ids': [row['sample_id'] for row in chosen],
        'model_load_seconds_excluded_from_latency': True,
        'forward_latency_ms_per_sample': forward_ms,
        'end_to_end_latency_ms_per_sample': end_to_end_ms,
        'mean_forward_latency_ms_per_sample': float(np.mean(forward_ms)),
        'median_forward_latency_ms_per_sample': float(np.median(forward_ms)),
        'mean_end_to_end_latency_ms_per_sample': float(np.mean(end_to_end_ms)),
        'median_end_to_end_latency_ms_per_sample': float(np.median(end_to_end_ms)),
        'flop_profiles': flop_rows,
        'mean_counted_gflops_per_sample': float(np.mean([row['gflops_counted_per_sample'] for row in flop_rows])),
        'flops_scope': 'GPU FlopCounterMode around the real Ultralytics predict path; multiply-add counted as 2 FLOPs. Elementwise, postprocessing and unsupported custom ops may be missing.',
        'latency_scope': 'A800 wall-clock latency for real Ultralytics prediction including image loading/preprocess, forward and result postprocessing, excluding model load; batch processing uses the recorded batch size.',
        'completed': True,
    }, ensure_ascii=False, indent=2))


def router_measure(root: Path, output: Path, expert: str, endpoint: str, condition: str, seed: int, samples: int, batch_size: int, device: str) -> None:
    work = root / 'outputs/router_method_improvement_20260911_r1/reproducible'
    sys.path.insert(0, str(work))
    common = __import__('method_improvement_common')
    spec = json.loads((root / 'outputs/router_method_improvement_20260911_r1/method_improvement_spec.json').read_text(encoding='utf-8'))
    trajectory_id = f'{expert}_{endpoint}_seed{seed}'
    cfg = common.make_config(spec, expert, endpoint, seed, trajectory_id)
    rows, data = common.load_split(root, spec, cfg, 'test', device)
    checkpoint_path = root / f'outputs/router_method_improvement_20260911_r1/conditions/{condition}_{expert}_{endpoint}_seed{seed}/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = common.FeatureRouter(cfg, data['hidden'].shape[1], data['confidence'].shape[1], data['output'].shape[1]).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    if samples > len(rows):
        raise ValueError(f'samples {samples} exceeds {len(rows)}')
    indices = np.random.default_rng(20260911).choice(len(rows), samples, replace=False)
    chunks = [torch.as_tensor(indices[start:start + batch_size], device=device, dtype=torch.long) for start in range(0, samples, batch_size)]
    for index in chunks[:2]:
        model(data, index)
        torch.cuda.synchronize()
    latency = []
    for index in chunks:
        elapsed, _ = timed_cuda(lambda index=index: model(data, index))
        latency.append(elapsed / len(index))
    index = chunks[0]
    total, _, operators = count_flops(lambda: model(data, index))
    output.write_text(json.dumps({
        'component': 'router', 'expert': expert, 'task': cfg['task'], 'endpoint': endpoint,
        'condition': condition, 'seed': seed, 'device': device, 'checkpoint': str(checkpoint_path),
        'sample_count': samples, 'batch_size': batch_size, 'row_count_loaded': len(rows),
        'mean_latency_ms_per_sample': float(np.mean(latency)), 'median_latency_ms_per_sample': float(np.median(latency)),
        'latency_ms_per_sample': latency, 'flops_counted_per_batch': total,
        'gflops_counted_per_sample': total / len(index) / 1e9, 'operators': operators,
        'flops_scope': 'GPU FlopCounterMode for the router forward; multiply-add counted as 2 FLOPs.',
        'latency_scope': 'A800 CUDA forward latency on cached router features, excluding feature loading and model load.',
        'completed': True,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component', choices=['specialist', 'router'], required=True)
    parser.add_argument('--expert', choices=['instancevg', 'yolo26x'], required=True)
    parser.add_argument('--endpoint', choices=['MiniCPM', 'Qwen3.8'])
    parser.add_argument('--condition', default='both')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if args.component == 'specialist' and args.expert == 'instancevg':
        instancevg_measure(args.root, args.output, args.samples, args.batch_size, args.device)
    elif args.component == 'specialist':
        yolo_measure(args.root, args.output, args.samples, args.batch_size, args.device)
    else:
        if args.endpoint is None:
            raise ValueError('--endpoint is required for router')
        router_measure(args.root, args.output, args.expert, args.endpoint, args.condition, args.seed, args.samples, args.batch_size, args.device)


if __name__ == '__main__':
    main()
