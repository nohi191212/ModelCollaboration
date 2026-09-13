"""统一四路路由器：图像采用拼接投影或带掩码的注意力汇聚。"""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn


def prepare_images(image_groups):
    """每个样本提供一张图或有序的左右两张 PIL 图；单图补真实灰图。"""
    if not image_groups or any(len(group) not in (1, 2) for group in image_groups):
        raise ValueError("每个样本必须提供一张或两张图片，批次不能为空")
    gray = Image.new("RGB", (224, 224), (128, 128, 128))
    pixels, valid = [], []
    for group in image_groups:
        pair = list(group) if len(group) == 2 else [group[0], gray]
        pixels.append(np.stack([
            np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
            .transpose(2, 0, 1) for im in pair
        ]))
        valid.append([True, len(group) == 2])
    # 与蒸馏完全相同；灰色 128 归一化后是约 0.00392，不是零张量。
    return torch.from_numpy(np.stack(pixels)).float().div(127.5).sub(1), torch.tensor(valid)


def load_distilled_router(checkpoint_path, **router_kwargs):
    """加载已有双编码器和相邻词表；调用方显式提供隐藏、置信信息维度。"""
    from siglip_distillation.mini_siglip import MiniSiglip
    from tokenizers import Tokenizer

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder = MiniSiglip(**checkpoint["model_config"])
    encoder.load_state_dict(checkpoint["model"], strict=True)
    tokenizer = Tokenizer.from_file(str(checkpoint_path.parent / "student_tokenizer.json"))
    if tokenizer.token_to_id("[PAD]") != 0:
        raise ValueError("学生词表的填充编号必须为 0")
    tokenizer.enable_truncation(max_length=encoder.config["text_length"])
    tokenizer.enable_padding(length=encoder.config["text_length"], pad_id=0, pad_token="[PAD]")
    return UnifiedRouter(encoder, **router_kwargs), tokenizer


class UnifiedRouter(nn.Module):
    """各任务结构相同；这里只输出原始调用分数，不固定损失或阈值。

    hidden_state 是同一次小模型回答的已汇聚状态 [B, H]。
    confidence 是真实可用的固定字段 [B, C]，本模块不伪造缺失信息。
    """

    def __init__(self, encoder, hidden_dim, confidence_dim, image_fusion="concat",
                 representation_dim=512, head_dim=128, attention_heads=4,
                 freeze_encoder=True):
        super().__init__()
        if image_fusion not in ("concat", "attention"):
            raise ValueError("image_fusion 只能是 concat 或 attention")
        if min(hidden_dim, confidence_dim, representation_dim, head_dim) < 1:
            raise ValueError("输入维度和表示维度必须为正整数")
        if image_fusion == "attention" and (
                attention_heads < 1 or representation_dim % attention_heads):
            raise ValueError("表示维度必须能够被注意力头数整除")
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.encoder.requires_grad_(not freeze_encoder)
        if freeze_encoder:
            self.encoder.eval()
        self.feature_dim = encoder.config["output_dim"]
        self.hidden_dim = hidden_dim
        self.confidence_dim = confidence_dim
        self.image_fusion = image_fusion
        self.config = dict(hidden_dim=hidden_dim, confidence_dim=confidence_dim,
                           image_fusion=image_fusion, representation_dim=representation_dim,
                           head_dim=head_dim, attention_heads=attention_heads,
                           freeze_encoder=freeze_encoder)
        d = representation_dim
        # 先初始化公共分支；相同随机种子下切换图像方案不会改变这些初值。
        self.text_branch = nn.Sequential(nn.Linear(self.feature_dim, d), nn.LayerNorm(d))
        self.hidden_branch = nn.Sequential(nn.Linear(hidden_dim, d), nn.LayerNorm(d))
        self.confidence_branch = nn.Sequential(nn.Linear(confidence_dim, d), nn.LayerNorm(d))
        self.head = nn.Sequential(nn.Linear(4 * d, head_dim), nn.GELU(), nn.Linear(head_dim, 1))
        if image_fusion == "concat":
            # 保留灰图表示；有效标记作为输入，不把该表示乘零。
            self.image_projection = nn.Linear(2 * self.feature_dim + 2, d)
        else:
            self.image_projection = nn.Linear(self.feature_dim, d)
            self.image_positions = nn.Parameter(torch.empty(1, 2, d))
            self.image_query = nn.Parameter(torch.empty(1, 1, d))
            nn.init.normal_(self.image_positions, std=0.02)
            nn.init.normal_(self.image_query, std=0.02)
            self.image_attention = nn.MultiheadAttention(d, attention_heads, dropout=0,
                                                         batch_first=True)
        self.image_norm = nn.LayerNorm(d)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def encode_inputs(self, images, text_ids):
        """得到 [B,2,768] 和 [B,768]；冻结时可由调用方缓存并复用。

        此入口按两张图原样编码，包括灰图，不在特征端补零。
        缓存属于调用方；微调期间不能复用旧权重产生的表示。
        """
        if images.ndim != 5 or images.shape[1:] != (2, 3, 224, 224):
            raise ValueError("图片必须是 [批次,2,3,224,224]")
        if not images.is_floating_point():
            raise ValueError("图片必须先按像素值 / 127.5 - 1 归一化")
        if (text_ids.ndim != 2 or text_ids.shape[0] != images.shape[0]
                or not 1 <= text_ids.shape[1] <= self.encoder.config["text_length"]
                or text_ids.dtype != torch.long):
            raise ValueError("文本编号必须是同批次的二维 long 张量，长度不超过学生上限")
        if not text_ids.ne(0).any(dim=1).all():
            raise ValueError("文本不能全部是填充符；固定任务也应传入任务说明")
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            image_features = self.encoder.encode_image(images.flatten(0, 1))
            text_features = self.encoder.encode_text(text_ids)
        return image_features.reshape(-1, 2, self.feature_dim), text_features

    def forward_features(self, image_features, image_valid, text_features,
                         hidden_state, confidence):
        """直接使用已编码表示；灰图槽仍须为灰图表示，不能补零。

        返回 scores [B]；注意力方案另外返回两张图的平均注意力 [B,2]。
        scores 未经 Sigmoid，可用于收益回归或交给带 logits 的分类损失。
        """
        batch = image_features.shape[0]
        if batch == 0 or image_features.shape != (batch, 2, self.feature_dim):
            raise ValueError("图像表示必须为非空的 [批次,2,编码维度]")
        if image_valid.shape != (batch, 2) or image_valid.dtype != torch.bool:
            raise ValueError("图片有效标记必须为 [批次,2] 的布尔张量")
        if not image_valid[:, 0].all():
            raise ValueError("每个样本第一张图片必须有效，第二张才允许为占位图")
        if (text_features.shape != (batch, self.feature_dim)
                or hidden_state.shape != (batch, self.hidden_dim)
                or confidence.shape != (batch, self.confidence_dim)):
            raise ValueError("文本、中间状态或置信信息的批次/维度与配置不符")
        if self.image_fusion == "concat":
            image_input = torch.cat((image_features.flatten(1),
                                     image_valid.to(image_features.dtype)), dim=-1)
            image_summary = self.image_projection(image_input)
            attention = None
        else:
            tokens = self.image_projection(image_features) + self.image_positions
            query = self.image_query.expand(batch, -1, -1)
            # PyTorch 的 True 表示禁止关注；不能把注意力分数直接设为 0。
            image_summary, attention = self.image_attention(
                query, tokens, tokens, key_padding_mask=~image_valid,
                need_weights=True, average_attn_weights=True)
            image_summary = image_summary.squeeze(1)
            attention = attention.squeeze(1)
        fused = torch.cat((self.image_norm(image_summary),
                           self.text_branch(text_features),
                           self.hidden_branch(hidden_state),
                           self.confidence_branch(confidence)), dim=-1)
        scores = self.head(fused).squeeze(-1)
        if not torch.isfinite(scores).all():
            raise RuntimeError("路由分数出现非有限值，请检查输入和训练状态")
        return {"scores": scores, "image_attention": attention}

    def forward(self, images, image_valid, text_ids, hidden_state, confidence):
        image_features, text_features = self.encode_inputs(images, text_ids)
        return self.forward_features(image_features, image_valid, text_features,
                                     hidden_state, confidence)
