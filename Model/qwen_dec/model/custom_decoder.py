"""
自训练 TransformerDecoder 检测头（移植自 Change3D_caption_v2 caption_decoder.py）。

做什么:
    1) PositionalEncoding：标准正余弦位置编码。
    2) Mesh_TransformerDecoderLayer：自注意力 + encoder-decoder cross-attention 单层。
    3) CaptionDecoder：Embedding + TransformerDecoder(N 层) + wdc 输出投影。
    4) SiglipDecoderDetector：顶层组装——SigLIP2 提取器 + projector + decoder。

为什么:
    用自训练小 decoder 替换 Qwen3-VL LLM，把坐标输出改为自建词表上的原子 token
    （如 547 是 1 个 token，而非 BPE 的 "5"/"47"）。

与 Change3D 原版的差异（3 处清理）:
    - 删除 PositionalEncoding 里死的 self.embedding_1D = nn.Embedding(52,...)。
    - mask.cuda() 改为 mask.to(tgt.device)，设备无关。
    - 不复制 CrossTransformer/MCCFormers_diff_as_Q 等未使用类。

用法:
    detector = SiglipDecoderDetector(siglip_path=..., embed_dim=256,
                                     vocab_size=1017, n_head=8, n_layer=3)
    pred, caps_sorted, decode_lengths, sort_ind = detector(images, caps, caplens)
    memory = detector.encode_memory(images)  # 推理用
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from model.external_feature_adapter import (
    EXTERNAL_FEATURE_CHANNELS,
    SIGLIP2_PATCH_GRID_SIZE,
    ExternalFeatureProjector,
    Siglip2BaseFeatureExtractor,
)


class PositionalEncoding(nn.Module):
    """入参: d_model 维度, dropout, max_len。
    方法: 标准正余弦位置编码（与 Change3D 一致），广播加到 (T,B,d_model) 序列上。
    出参: 加了位置编码的序列。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Mesh_TransformerDecoderLayer(nn.Module):
    """入参: d_model, nhead, dim_feedforward, dropout 等。
    方法: 标准 TransformerDecoderLayer 等价实现——causal self-attention + 对 memory
          的 cross-attention（Q=文本嵌入, K=V=视觉 memory）+ 残差 + LayerNorm。
    出参: (T, B, d_model)。"""

    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # cross-attention（Q=tgt, K=V=memory）
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN（原版未调用，但保留以兼容参数集；保留后若有需要可启用）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: Optional[bool] = None,
    ) -> Tensor:
        # causal self-attention block
        sa_out = self._sa_block(tgt, tgt_mask, tgt_key_padding_mask)
        self_att_tgt = self.norm1(tgt + sa_out)
        # cross-attention block: Q=self_att_tgt, K=V=memory
        ca_out, _ = self.multihead_attn2(
            self_att_tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
        )
        ca_out = self.dropout2(ca_out)
        x = self.norm2(self_att_tgt + ca_out)
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        out = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(out)


class CaptionDecoder(nn.Module):
    """自训练 TransformerDecoder：Embedding → TransformerDecoder(N 层, cross-attn) → wdc 输出投影。
    移植自 Change3D caption_decoder.CaptionDecoder，签名保持一致。"""

    def __init__(self, embed_dim: int, vocab_size: int, n_head: int = 8, n_layer: int = 3, dropout: float = 0.1):
        super().__init__()
        print(f"CaptionDecoder: n_layer={n_layer} n_head={n_head} embed_dim={embed_dim} vocab_size={vocab_size}")
        self.vocab_embedding = nn.Embedding(vocab_size, embed_dim)
        decoder_layer = Mesh_TransformerDecoderLayer(
            embed_dim, n_head,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layer)
        self.position_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        self.wdc = nn.Linear(embed_dim, vocab_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)
        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(
        self,
        memory: Tensor,
        encoded_captions: Tensor,
        caption_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, List[int], Tensor]:
        """入参:
            memory: (S, B, C) 视觉特征序列，S=256 (16x16 patch)，C=embed_dim。
            encoded_captions: (B, L) token id 序列，含 <start>/<end>/<pad>。
            caption_lengths: (B, 1) 真实长度（含 start/end）。
        出参:
            pred: (B, L, vocab_size) logits（已按长度降序重排）。
            encoded_captions: (B, L) 重排后的输入。
            decode_lengths: list[int]，每条样本的有效预测步数（不含 <start>）。
            sort_ind: (B,) 重排索引，用于把同 batch 的其他张量对齐。
        """
        tgt = encoded_captions.permute(1, 0)  # (L, B)
        tgt_length = tgt.size(0)

        # causal mask（上三角 -inf）
        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        mask = mask.to(tgt.device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)

        # mask 必须与 query 同 dtype（bf16 训练下 mask 默认 float32 会触发 SDPA bias 类型错误）
        mask = mask.to(tgt_embedding.dtype)
        pred = self.transformer(tgt_embedding, memory, tgt_mask=mask)
        pred = self.wdc(self.dropout_layer(pred)).permute(1, 0, 2)  # (B, L, vocab_size)

        # 按长度降序排序（pack_padded_sequence 要求）
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        return pred, encoded_captions, decode_lengths, sort_ind


class SiglipDecoderDetector(nn.Module):
    """顶层检测模型：冻结 SigLIP2-base + 可训练空间适配/projector + 自训练 TransformerDecoder。

    forward 链路:
        images -> Siglip2BaseFeatureExtractor (主干冻结) -> [B,256,16,16]
               -> ExternalFeatureProjector (256->embed_dim + 2D 位置编码) -> [B,256,embed_dim]
               -> permute(1,0,2) -> memory [256, B, embed_dim]
        memory + caps -> CaptionDecoder (cross-attn) -> token logits [B, L, vocab_size]
    """

    def __init__(
        self,
        siglip_path: str,
        vocab_size: int,
        embed_dim: int = 256,
        n_head: int = 8,
        n_layer: int = 3,
        dropout: float = 0.1,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if embed_dim != EXTERNAL_FEATURE_CHANNELS:
            # projector 默认 input_dim=256，输出 hidden_size=embed_dim。要求 embed_dim % 4 == 0。
            # 这里 embed_dim 同时是 decoder 的 d_model 和 memory 通道——必须等于 adapter 输出通道(256)，
            # 否则 cross-attention 维度不匹配。强制约束避免静默错误。
            raise ValueError(
                f"embed_dim must equal EXTERNAL_FEATURE_CHANNELS={EXTERNAL_FEATURE_CHANNELS} "
                f"(no projection between projector output and decoder); got {embed_dim}."
            )
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.extractor = Siglip2BaseFeatureExtractor(siglip_path, device=device, dtype=dtype)
        # projector: 256 -> embed_dim=256，并叠加 2D 正余弦位置编码
        self.projector = ExternalFeatureProjector(
            hidden_size=embed_dim,
            input_dim=EXTERNAL_FEATURE_CHANNELS,
            output_grid_size=SIGLIP2_PATCH_GRID_SIZE,
        ).to(device=device, dtype=dtype)
        # decoder 全 bf16 与 extractor/projector 对齐
        self.decoder = CaptionDecoder(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
        ).to(device=device, dtype=dtype)

    def forward(
        self,
        images,
        encoded_captions: Tensor,
        caption_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, List[int], Tensor]:
        """入参: images (路径/PIL/tensor batch), encoded_captions (B,L), caption_lengths (B,1)。
        出参: CaptionDecoder 的 (pred, caps_sorted, decode_lengths, sort_ind)。"""
        feat = self.extractor(images)               # [B,256,16,16] 主干冻结
        memory = self.projector(feat)               # [B,256,embed_dim]
        memory = memory.permute(1, 0, 2)            # [256, B, embed_dim]
        return self.decoder(memory, encoded_captions, caption_lengths)

    @torch.no_grad()
    def encode_memory(self, images) -> Tensor:
        """入参: images batch。方法: 仅跑视觉链路出 memory（推理解码用）。
        出参: (S, B, embed_dim)。"""
        feat = self.extractor(images)
        memory = self.projector(feat)
        return memory.permute(1, 0, 2)
