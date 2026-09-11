"""专用模型的分层状态读取：明确计数范围，百分位选层，即时汇聚。"""
from functools import partial
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

PERCENTAGES = tuple(range(10, 100, 10))


def layer_groups(task, model):
    """返回独立的选层轴，避免把并行视觉/文本主干混成一个深度。"""
    groups = {}
    if task == 'cub':
        groups['residual_blocks'] = [
            (f'layer{stage}.{index}', block, 'spatial')
            for stage in range(1, 5)
            for index, block in enumerate(getattr(model, f'layer{stage}'))]
    elif task == 'nlvr2':
        groups['multimodal_encoder'] = [
            (f'beit3.encoder.layers.{i}', block, 'ordered_image_text')
            for i, block in enumerate(model.beit3.encoder.layers)]
    elif task == 'grefcoco':
        backbone = model.backbone[0].body
        groups['visual_residual_blocks'] = [
            (f'backbone.0.body.layer{stage}.{index}', block, 'spatial')
            for stage in range(1, 5)
            for index, block in enumerate(getattr(backbone, f'layer{stage}'))]
        groups['text_encoder'] = [
            (f'transformer.text_encoder.encoder.layer.{i}', block, 'text')
            for i, block in enumerate(model.transformer.text_encoder.encoder.layer)]
        groups['multimodal_encoder'] = [
            (f'transformer.encoder.layers.{i}', block, 'sequence_first_masked')
            for i, block in enumerate(model.transformer.encoder.layers)]
        groups['query_decoder'] = [
            (f'transformer.decoder.layers.{i}', block, 'sequence_first_queries')
            for i, block in enumerate(model.transformer.decoder.layers)]
    elif task in ('yolo26x', 'rtdetr_x'):
        # 统计实际顶层有参数的特征块，不将上采样、拼接和最终预测头计为一层。
        groups['feature_blocks'] = [
            (f'model.{i}', block, 'spatial')
            for i, block in enumerate(model.model[:-1])
            if sum(parameter.numel() for parameter in block.parameters()) > 0]
        if task == 'rtdetr_x':
            decoder = model.model[-1].decoder
            groups['query_decoder'] = [
                (f'model.{len(model.model)-1}.decoder.layers.{i}', block, 'batch_first_queries')
                for i, block in enumerate(decoder.layers[:decoder.eval_idx+1])]
    else:
        raise ValueError(f'未知任务：{task}')
    if not groups or any(not blocks for blocks in groups.values()):
        raise ValueError('没有找到所声明的层组')
    return groups


class PercentileStateCollector:
    """读取单次完整推理的各百分位状态；不保存原始特征图。

    begin_sample() → 原模型正常推理 → finish_sample()。
    返回 CPU float16 张量 [B,D]。多百分位落同一模块时只挂一个读取器。
    """

    def __init__(self, task, model, groups=None):
        self.task = task
        self.model = model
        self.groups = layer_groups(task, model) if groups is None else groups
        self.selected = {}
        self.mapping = {}
        self.inventory = {}
        self.handles = []
        self.values = {}
        self.context = None
        for group_name, blocks in self.groups.items():
            count = len(blocks)
            self.inventory[group_name] = [
                {'rank_1based': rank, 'module': name, 'type': type(block).__name__, 'pooling': kind}
                for rank, (name, block, kind) in enumerate(blocks, 1)]
            self.mapping[group_name] = []
            for percentage in PERCENTAGES:
                rank = (percentage * count + 99) // 100
                name, block, kind = blocks[rank-1]
                key = f'{group_name}__layer_{rank:02d}'
                self.mapping[group_name].append({
                    'requested_percent': percentage, 'layer_count': count,
                    'rank_1based': rank, 'actual_percent': 100 * rank / count,
                    'module': name, 'state_key': key, 'file': key + '.npy'})
                if key not in self.selected:
                    self.selected[key] = (block, kind)

    def attach(self):
        if self.handles:
            raise RuntimeError('状态读取器已安装')
        for key, (block, kind) in self.selected.items():
            self.handles.append(block.register_forward_hook(
                partial(self._capture, key, kind), with_kwargs=True))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def begin_sample(self, *, text_valid=None, image_padding=None):
        self.values = {}
        self.context = {'text_valid': text_valid, 'image_padding': image_padding}

    def _capture(self, key, kind, _module, _inputs, kwargs, output):
        if self.context is None:
            raise RuntimeError('推理前必须调用 begin_sample')
        if key in self.values:
            raise RuntimeError(f'同一次样本推理重复进入所选模块：{key}')
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{key} 输出不是张量')
        if kind == 'spatial':
            padding = self.context['image_padding']
            if padding is None or not padding.any():
                pooled = tensor.float().mean(dim=(-2, -1))
            else:
                valid = ~F.interpolate(padding[:, None].float(), size=tensor.shape[-2:], mode='nearest').bool()
                pooled = (tensor.float() * valid).sum((-2, -1)) / valid.sum((-2, -1))
        elif kind == 'ordered_image_text':
            text_valid = self.context['text_valid']
            if text_valid is None:
                raise ValueError('双图图文汇聚必须提供文本有效标记')
            batch = text_valid.shape[0]
            visual_length = tensor.shape[1] - text_valid.shape[1]
            valid = torch.cat((torch.ones((2*batch, visual_length), device=tensor.device),
                               text_valid.repeat(2, 1)), dim=1)
            per_image = (tensor.float() * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            pooled = torch.cat((per_image[:batch], per_image[batch:]), dim=-1)
        elif kind == 'text':
            valid = self.context['text_valid']
            if valid is None:
                raise ValueError('文本汇聚必须提供文本有效标记')
            pooled = (tensor.float() * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        elif kind == 'sequence_first_masked':
            valid = ~kwargs['src_key_padding_mask']
            per_batch = tensor.transpose(0, 1).float()
            pooled = (per_batch * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        elif kind == 'sequence_first_queries':
            pooled = tensor.float().mean(0)
        elif kind == 'batch_first_queries':
            pooled = tensor.float().mean(1)
        else:
            raise ValueError(f'未知汇聚规则：{kind}')
        vector = pooled.to(dtype=torch.float16, device='cpu')
        if vector.ndim != 2 or not torch.isfinite(vector).all():
            raise RuntimeError(f'{key} 汇聚结果维度错误或含非有限值')
        self.values[key] = {
            'vector': vector, 'raw_shape': list(tensor.shape),
            'raw_dtype': str(tensor.dtype), 'raw_bytes': tensor.numel()*tensor.element_size(),
            'pooled_dim': vector.shape[-1], 'pooled_bytes': vector.numel()*2}

    def finish_sample(self):
        missing = set(self.selected) - set(self.values)
        if missing:
            raise RuntimeError(f'部分选定模块没有执行：{sorted(missing)}')
        result = self.values
        self.values = {}
        self.context = None
        return result


def run_percentile_probe(task, model, rows, run, output_dir):
    """已有三样本试提取脚本的多位置入口；每个样本只需一次带读取的推理。"""
    output_dir = Path(output_dir)
    collector = PercentileStateCollector(task, model)
    with torch.inference_mode():
        run(rows[0][1])
    torch.cuda.synchronize()
    saved = {key: [] for key in collector.selected}
    measurements, predictions = [], []
    for sample_id, data in rows:
        text_valid, image_padding = None, None
        if task == 'nlvr2':
            text_valid = data['attention_mask']
        elif task == 'grefcoco':
            tokenized = model.transformer.tokenizer.batch_encode_plus(data[1], padding='longest', return_tensors='pt')
            text_valid = tokenized['attention_mask'].to(data[0].tensors.device)
            image_padding = data[0].mask
        collector.begin_sample(text_valid=text_valid, image_padding=image_padding)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            original = run(data).detach().cpu()
        torch.cuda.synchronize()
        baseline_seconds = time.perf_counter() - started
        baseline_peak = torch.cuda.max_memory_allocated()
        collector.attach()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                observed = run(data).detach().cpu()
            torch.cuda.synchronize()
            hooked_seconds = time.perf_counter() - started
            hooked_peak = torch.cuda.max_memory_allocated()
            captured = collector.finish_sample()
        finally:
            collector.remove()
        if original.shape != observed.shape or not torch.equal(original, observed):
            raise RuntimeError(f'{sample_id} 带状态读取前后预测不同')
        states = {}
        for key, record in captured.items():
            vector = record.pop('vector')
            if vector.shape[0] != 1:
                raise ValueError('此试提取脚本每次只接受一个样本')
            saved[key].append(vector.numpy()[0])
            states[key] = record
        measurements.append({'sample_id': sample_id, 'states': states,
                             'baseline_seconds': baseline_seconds, 'hooked_seconds': hooked_seconds,
                             'baseline_peak_bytes': baseline_peak, 'hooked_peak_bytes': hooked_peak,
                             'prediction_unchanged': True})
        predictions.append({'sample_id': sample_id, 'output': observed.tolist()})
        print('PERCENTILE_SAMPLE', task, sample_id, len(states), flush=True)
    for key, vectors in saved.items():
        np.save(output_dir / (key + '.npy'), np.stack(vectors))
    mapping = {'task': task, 'rounding': 'ceil(percent * layer_count / 100), rank starts at 1',
               'deduplicate_same_layer': True, 'groups': collector.mapping, 'inventory': collector.inventory}
    (output_dir / 'layer_map.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    payload_bytes = sum(array[0].nbytes for array in saved.values())
    summary = {'task': task, 'sample_count': len(rows), 'split': 'train',
               'group_layer_counts': {name: len(blocks) for name, blocks in collector.groups.items()},
               'unique_saved_layers': len(saved), 'vector_bytes_per_sample': payload_bytes,
               'vector_MiB_per_100k': payload_bytes*100000/2**20,
               'array_files_bytes': sum((output_dir / (key + '.npy')).stat().st_size for key in saved),
               'measurements': measurements}
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_dir / 'predictions.json').write_text(json.dumps(predictions))
    print('PERCENTILE_COMPLETE', task, summary['group_layer_counts'], summary['unique_saved_layers'], payload_bytes, flush=True)
