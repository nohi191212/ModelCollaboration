"""GroundingDINO、ViLT、GLSim 的选层和汇聚；保留多次调用的先后顺序。"""
import torch
from torch.nn import functional as F
from state_pooling import PercentileStateCollector


class AddedStateCollector(PercentileStateCollector):
    def __init__(self, task, model):
        if task == 'groundingdino':
            groups = {
                'visual_blocks': [(f'model.backbone.conv_encoder.model.encoder.layers.{s}.blocks.{i}', block, 'swin')
                                  for s, stage in enumerate(model.model.backbone.conv_encoder.model.encoder.layers)
                                  for i, block in enumerate(stage.blocks)],
                'text_encoder': [(f'model.text_backbone.encoder.layer.{i}', block, 'text')
                                 for i, block in enumerate(model.model.text_backbone.encoder.layer)],
                'multimodal_encoder': [(f'model.encoder.layers.{i}', block, 'grounded_multimodal')
                                       for i, block in enumerate(model.model.encoder.layers)],
                'query_decoder': [(f'model.decoder.layers.{i}', block, 'batch_first_queries')
                                  for i, block in enumerate(model.model.decoder.layers)]}
        elif task == 'vilt':
            groups = {'ordered_image_text': [(f'vilt.encoder.layer.{i}', block, 'vilt_two_passes')
                                             for i, block in enumerate(model.vilt.encoder.layer)]}
        elif task == 'glsim':
            groups = {
                'global_local_encoder': [(f'model.encoder.blocks.{i}', block, 'glsim_two_passes')
                                         for i, block in enumerate(model.model.encoder.blocks)],
                'aggregator': [(f'model.aggregator.blocks.{i}', block, 'batch_first_queries')
                               for i, block in enumerate(model.model.aggregator.blocks)]}
        else:
            raise ValueError(task)
        super().__init__(task, model, groups=groups)

    def begin_sample(self, **context):
        super().begin_sample(**context)
        self.calls = {}

    def _capture(self, key, kind, module, inputs, kwargs, output):
        if kind not in ('swin', 'grounded_multimodal', 'vilt_two_passes', 'glsim_two_passes'):
            return super()._capture(key, kind, module, inputs, kwargs, output)
        if self.context is None:
            raise RuntimeError('推理前必须调用 begin_sample')
        tensor = output[0] if isinstance(output, tuple) else output
        if kind == 'grounded_multimodal':
            vision, text = tensor
            valid_v = ~kwargs['key_padding_mask']
            valid_t = ~kwargs['text_attention_mask']
            vector_v = (vision.float() * valid_v.unsqueeze(-1)).sum(1) / valid_v.sum(1, keepdim=True)
            vector_t = (text.float() * valid_t.unsqueeze(-1)).sum(1) / valid_t.sum(1, keepdim=True)
            vector = torch.cat((vector_v, vector_t), -1)
        elif kind == 'swin':
            dimensions = kwargs['input_dimensions'] if 'input_dimensions' in kwargs else inputs[1]
            padding = self.context['image_padding']
            valid = ~F.interpolate(padding[:, None].float(), size=dimensions, mode='nearest').bool().flatten(2).squeeze(1)
            vector = (tensor.float() * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        elif kind == 'vilt_two_passes':
            mask = kwargs['attention_mask'] if 'attention_mask' in kwargs else inputs[1]
            valid = (mask[:, 0, 0, :] == 0)
            vector = (tensor.float() * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        else:
            vector = tensor.float().mean(1)
        vector = vector.to(device='cpu', dtype=torch.float16)
        if vector.ndim != 2 or vector.shape[0] != 1 or not torch.isfinite(vector).all():
            raise RuntimeError(f'{key} 汇聚状态维度或数值错误')
        self.calls.setdefault(key, []).append(vector)
        expected = 2 if kind in ('vilt_two_passes', 'glsim_two_passes') else 1
        if len(self.calls[key]) > expected:
            raise RuntimeError(f'{key} 执行次数超过预期 {expected}')
        if len(self.calls[key]) == expected:
            self.values[key] = {'vector': torch.cat(self.calls[key], dim=-1)}
