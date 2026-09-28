"""FPGADecoder：频域感知门控注意力文本引导解码器。

FPGADecoder（Frequency-aware Prompt-Gated Attention Decoder）采用单向 V→T
频域感知残差注意力融合：视觉特征查询文本 token，经频域 Token 选择与门控
残差精炼，实现场景语义引导。
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from visual_modules import UpsampleBlock


# ===========================================================================
# 频域 Token 选择器（LAST-ViT 风格）
# ===========================================================================


class FreqTokenSelector(nn.Module):
    """
    频率域 Token 选择器（LAST-ViT 风格）。

    入参:
    - kernel_size: 高斯核大小，默认 3。
    - sigma: 高斯核标准差，同时作为 softmax 温度参数，默认 1.0。

    方法:
    - FFT 变换到频域，高斯低通滤波提取低频分量。
    - 由 |原始 - 低频| 计算高频能量占比作为 token 重要性分数。
    - softmax 温度加权调制特征。

    出参:
    - forward 返回 (加权特征 [B, N, D], 重要性分数 [B, N])。
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())

    def _create_gaussian_kernel(self) -> torch.Tensor:
        k = self.kernel_size
        xx = torch.arange(0, k, dtype=torch.float32)
        kernel = torch.exp(-(xx - k // 2) ** 2 / (2. * self.sigma ** 2))
        return kernel / kernel.sum()

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = features.shape
        feat_fft = torch.fft.fft(features, dim=-1)
        feat_fft_shifted = torch.fft.fftshift(feat_fft, dim=-1)

        if self.gaussian_kernel.size(0) < D:
            gaussian_kernel = F.interpolate(
                self.gaussian_kernel.unsqueeze(0).unsqueeze(0),
                size=D, mode='linear', align_corners=False).squeeze()
        elif self.gaussian_kernel.size(0) > D:
            indices = torch.linspace(0, self.kernel_size - 1, D).long()
            gaussian_kernel = self.gaussian_kernel[indices]
        else:
            gaussian_kernel = self.gaussian_kernel

        feat_filtered = feat_fft_shifted * gaussian_kernel.unsqueeze(0).unsqueeze(0)
        feat_filtered = torch.fft.ifftshift(feat_filtered, dim=-1)
        feat_low_freq = torch.fft.ifft(feat_filtered, dim=-1).real

        high_freq_energy = torch.abs(features - feat_low_freq).sum(dim=-1)
        total_energy = torch.abs(features).sum(dim=-1) + 1e-8
        importance_scores = high_freq_energy / total_energy

        attention_weights = F.softmax(importance_scores / self.sigma, dim=1)
        selected_feat = features * attention_weights.unsqueeze(-1)
        return selected_feat, importance_scores


# ===========================================================================
# 频域感知残差注意力块（单向 V→T）
# ===========================================================================


class FPGABlock(nn.Module):
    """
    频域感知残差注意力块（pre-norm，单向 V→T）。

    入参:
    - d_model: 特征维度。
    - nhead/dim_feedforward/dropout: 注意力与 FFN 配置。
    - use_freq_rescaling: 是否对 cross-attention 输出执行通道频域响应重标定。

    方法:
    - cross-attention(视觉 query 文本 KV) → 可选频域响应重标定 →
      门控残差(σ(Wg·src)⊙src + attn_out) → FFN 残差。

    出参:
    - forward(src, memory, src_pos, memory_key_padding_mask) 返回精炼后的视觉特征 [B, N, d_model]。
    """

    def __init__(self, d_model, nhead=16, dim_feedforward=2048, dropout=0.1,
                 use_freq_rescaling=True):
        super().__init__()
        self.use_freq_rescaling = use_freq_rescaling
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.token_selector = FreqTokenSelector(kernel_size=3, sigma=1.0)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, src, memory, src_pos=None, memory_key_padding_mask=None):
        src_input = src + src_pos if src_pos is not None else src
        attn_out = self.norm1(src_input)
        attn_out = self.cross_attn(
            query=attn_out, key=memory, value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        if self.use_freq_rescaling:
            attn_out, _ = self.token_selector(attn_out)
        g = self.gate_proj(src)
        gated = src * g + self.dropout1(attn_out)
        ffined = self.norm2(gated)
        ffined = self.linear2(self.dropout(self.activation(self.linear1(ffined))))
        return gated + self.dropout2(ffined)


# ===========================================================================
# 单向视觉-文本融合模块（仅 V→T）
# ===========================================================================


class FPGAStage(nn.Module):
    """
    单向视觉-文本融合模块（仅 V→T）。

    入参:
    - in_dim/out_dim/text_dim: 视觉输入、输出和文本维度。
    - nhead/dim_feedforward/dropout: 注意力配置。
    - use_freq_rescaling: 是否启用通道频域响应重标定。

    方法:
    - 文本双池化(avg+max)广播加法注入视觉特征。
    - 生成 2D sine 位置编码，视觉 token 查询文本 token 做可选频域重标定的残差注意力。
    - 深度可分离卷积转换通道。

    出参:
    - forward(src_feat, text_feat, text_mask) 返回融合特征 [B, out_dim, H, W]。
    """

    def __init__(self, in_dim, out_dim, text_dim, nhead=16,
                 dim_feedforward=2048, dropout=0.1,
                 use_freq_rescaling=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.text_pooling_proj = nn.Linear(text_dim, in_dim)
        self.v2t_block = FPGABlock(
            in_dim,
            nhead,
            dim_feedforward,
            dropout,
            use_freq_rescaling=use_freq_rescaling,
        )
        self.dim_convert = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, groups=in_dim),
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_sine_pos_encoding(self, feat):
        bs, c, h, w = feat.shape
        device, dtype = feat.device, feat.dtype
        half_c = c // 2
        base_dim = max(half_c // 2, 1)
        dim_t = torch.arange(base_dim, device=device, dtype=dtype)
        dim_t = 10000 ** (2 * (dim_t // 2) / base_dim)
        y_pos = torch.arange(h, device=device, dtype=dtype).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w, device=device, dtype=dtype).unsqueeze(0).repeat(h, 1)
        pos_x = x_pos.flatten()[None, :, None] / dim_t
        pos_y = y_pos.flatten()[None, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)
        base_pos = torch.cat([pos_y, pos_x], dim=2)
        if base_pos.shape[2] < half_c:
            base_pos = F.pad(base_pos, (0, half_c - base_pos.shape[2]))
        elif base_pos.shape[2] > half_c:
            base_pos = base_pos[:, :, :half_c]
        pos = torch.cat([base_pos, base_pos], dim=2)
        if pos.shape[2] < c:
            pos = F.pad(pos, (0, c - pos.shape[2]))
        elif pos.shape[2] > c:
            pos = pos[:, :, :c]
        pos = pos.transpose(1, 2).contiguous().view(1, c, h, w)
        return pos.expand(bs, -1, -1, -1)

    def forward(self, src_feat, text_feat, text_key_padding_mask=None):
        B, C, H, W = src_feat.shape
        avg_pool = text_feat.mean(dim=1)
        max_pool = text_feat.max(dim=1)[0]
        pooled_text = self.text_pooling_proj(avg_pool + max_pool)[:, :, None, None]
        src_feat = src_feat + pooled_text
        pos = self._generate_sine_pos_encoding(src_feat)
        feat_flat = src_feat.flatten(2).transpose(1, 2)
        pos_flat = pos.flatten(2).transpose(1, 2)
        visual_enhanced = self.v2t_block(
            feat_flat, text_feat, src_pos=pos_flat,
            memory_key_padding_mask=text_key_padding_mask,
        )
        visual_output = visual_enhanced.transpose(1, 2).view(B, C, H, W)
        return self.dim_convert(visual_output)


class FreqBandRefinement(nn.Module):
    """
    解码阶段频域精化：对解码输出做 rfft2 低/高频带分解与逐通道可学习重组。

    入参:
    - channels: 解码输出通道数。

    方法:
    - rfft2 后按归一化径向频率划分低频（<=0.25）与高频（>0.25）掩码，
      逆变换回空间域得到低/高频分量；每通道两个可学习权重做加性重组，
      权重零初始化使模块初始为恒等映射，保证残差安全启动。

    出参:
    - forward 返回与输入同形状的精化特征 [B, channels, H, W]。
    """

    def __init__(self, channels: int, cutoff: float = 0.25):
        super().__init__()
        self.cutoff = cutoff
        self.low_weight = nn.Parameter(torch.zeros(channels))
        self.high_weight = nn.Parameter(torch.zeros(channels))
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def _radial_masks(self, height: int, width: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """
        入参:
        - height/width: 特征空间尺寸；device/dtype: 掩码目标属性。

        方法:
        - 由 fftfreq/rfftfreq 构造归一化径向频率，按 cutoff 划分低/高频掩码并缓存。

        出参:
        - tuple: (低频掩码, 高频掩码)，形状 [H, W//2+1]。
        """
        key = (height, width, device, dtype)
        if key not in self._mask_cache:
            fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
            fx = torch.fft.rfftfreq(width, device=device, dtype=dtype)
            radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
            low = (radius <= self.cutoff).to(dtype)
            self._mask_cache[key] = (low, 1.0 - low)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入参:
        - x: [B, channels, H, W] 解码输出特征。

        方法:
        - 正交 rfft2 分解，低通逆变换得低频分量，高频分量由原特征减低频得到；
          逐通道权重加性重组后残差注入原特征。

        出参:
        - torch.Tensor: [B, channels, H, W] 频域精化特征，初始恒等。
        """
        spectrum = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
        height, width = x.shape[-2:]
        low_mask, high_mask = self._radial_masks(height, width, x.device, x.float().dtype)
        low = torch.fft.irfft2(spectrum * low_mask, s=(height, width), norm="ortho")
        high = x.float() - low
        low_gain = self.low_weight.view(1, -1, 1, 1).to(x.dtype)
        high_gain = self.high_weight.view(1, -1, 1, 1).to(x.dtype)
        return x + (low_gain * low + high_gain * high).to(x.dtype)


# ===========================================================================
# 文本引导解码器（四尺度粗到细，单向 V→T 融合）
# ===========================================================================


class FPGADecoder(nn.Module):
    """
    FPGADecoder：频域感知门控注意力文本引导解码器（单向 V→T 融合）。

    入参:
    - d_model/decoder_dim/text_feat_dim: 跨模态、输出和文本特征维度。
    - nhead/fpn_dim: 注意力头数与兼容 FPN 配置。
    - use_freq_rescaling: 是否在四个尺度启用通道频域响应重标定。
    - concat_features: T1/T2 拼接后的四尺度视觉特征。
    - t1_multiscale/t2_multiscale: 接口兼容保留，当前不使用。
    - text_global: 接口兼容保留，当前不使用。
    - text_tokens: Qwen 完整文本 token 序列 [B, T, text_feat_dim]。
    - text_key_padding_mask: 文本 padding 掩码，True 表示 padding。

    方法:
    - 从最粗尺度开始，每尺度执行一次 V→T 单向残差注意力融合；
      融合输出与融合前特征做残差连接（x + fused），不直接替换。
    - 上采样后与跳连分支等权相加，融合后特征直接进入解码通路。
    - 解码输出（final_upsample 后）经 FreqBandRefinement 做频域低/高频带
      逐通道精化，权重零初始化恒等启动，输出 decoder_dim 通道特征。

    出参:
    - forward 返回供变化检测头使用的高分辨率文本引导特征。
    """

    def __init__(self, d_model=512, decoder_dim=128, text_feat_dim=512,
                 nhead=16, fpn_dim=256, use_freq_rescaling=True):
        super().__init__()
        self.d_model = d_model
        self.decoder_dim = decoder_dim
        self.text_feat_dim = text_feat_dim
        self.use_freq_rescaling = use_freq_rescaling
        self.text_proj = (
            nn.Identity() if text_feat_dim == d_model
            else nn.Linear(text_feat_dim, d_model)
        )
        self.fusion_blocks = nn.ModuleList([
            FPGAStage(
                in_dim=d_model,
                out_dim=d_model,
                text_dim=d_model,
                nhead=nhead,
                use_freq_rescaling=use_freq_rescaling,
            )
            for _ in range(4)
        ])
        # 解码阶段频域精化：作用于 final_upsample 输出，逐通道零初始化残差注入
        self.freq_refine = FreqBandRefinement(decoder_dim)
        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(d_model, d_model) for _ in range(3)
        ])
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d_model, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )

    def _align_features(self, upsampled_feat, target_feat):
        diff_y = target_feat.size(2) - upsampled_feat.size(2)
        diff_x = target_feat.size(3) - upsampled_feat.size(3)
        if diff_y > 0 or diff_x > 0:
            upsampled_feat = F.pad(upsampled_feat, [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2,
            ])
        return upsampled_feat

    def forward(self, concat_features, t1_multiscale, t2_multiscale,
                text_global, text_tokens, text_key_padding_mask=None,
                use_text=True):
        """
        入参:
        - concat_features: 四尺度视觉特征（由细到粗）。
        - t1_multiscale/t2_multiscale/text_global: 接口兼容保留，当前不使用。
        - text_tokens: Qwen 文本 token [B,T,text_feat_dim]；use_text=False 时不消费。
        - text_key_padding_mask: 文本 padding 掩码；use_text=False 时不消费。
        - use_text: 文本提示开关；True 执行每尺度 V→T 融合，False 跳过全部
          fusion_blocks 计算，视觉特征直接进入上采样解码通路（参数保留在
          state_dict 中，保证与开启态 checkpoint 结构兼容）。

        方法:
        - 粗到细解码：开启时每尺度先做 V→T 残差融合（x + fused）再上采样、
          跳连等权相加；关闭时直接以视觉特征执行同样的上采样与跳连解码；
          两种状态最终均经 final_upsample 与 FreqBandRefinement 频域精化。

        出参:
        - torch.Tensor: [B,decoder_dim,H,W] 解码特征。
        """
        if len(concat_features) != 4:
            raise ValueError("concat_features must contain four scales")

        if use_text:
            text_tokens = self.text_proj(text_tokens)
            x = concat_features[3] + self.fusion_blocks[3](concat_features[3], text_tokens, text_key_padding_mask)
        else:
            x = concat_features[3]
        for upsample_index, fusion_index, skip_index in ((0, 2, 2), (1, 1, 1), (2, 0, 0)):
            x = self.upsample_blocks[upsample_index](x)
            x = self._align_features(x, concat_features[skip_index])
            x = x + concat_features[skip_index]
            if use_text:
                x = x + self.fusion_blocks[fusion_index](x, text_tokens, text_key_padding_mask)
        return self.freq_refine(self.final_upsample(x))
