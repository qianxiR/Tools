"""Text embedding cache loader for the Qwen-guided segmentation network.

入参(全局):
- 无；模块被训练/推理脚本 import。

方法:
- precompute_text_embedding.py 用 Qwen3-VL-4B 编码提示词后，将
  {"hidden": [1,T,D], "mask": [1,T], "hidden_size": D, "prompt": str}
  存为 torch 缓存；本模块负责加载并广播到当前 batch。

出参:
- load_text_embedding(path) -> (hidden [1,T,D], mask [1,T] bool, hidden_size int)。
"""

from __future__ import annotations

from pathlib import Path

import torch


def parse_embedding(ckpt: dict, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    入参:
    - ckpt: torch.load 的缓存字典（含 hidden/mask/hidden_size）。
    - device: 目标设备。

    方法:
    - 校验必需字段，返回文本 token 特征与 padding 掩码。

    出参:
    - tuple: (hidden [1,T,D] float, mask [1,T] bool, hidden_size int)。
    """
    hidden = ckpt["hidden"]
    mask = ckpt["mask"]
    hidden_size = int(ckpt.get("hidden_size", hidden.shape[-1]))
    hidden = hidden.to(device=device, dtype=torch.float32)
    # 缓存保存的是 attention_mask（True=有效 token）；网络端纹理掩码语义为
    # True=padding，因此取反后返回，避免全有效序列把全部 key 屏蔽导致 nan。
    padding_mask = (~mask.bool()).to(device=device)
    return hidden, padding_mask, hidden_size


def load_text_embedding(path: str | Path, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, int]:
    """入参: 缓存路径与设备；方法: 读取并解析缓存；出参: (hidden, mask, hidden_size)。"""
    if not (Path(path)).is_file():
        raise FileNotFoundError(f"Missing text embedding cache: {path}")
    ckpt = torch.load(Path(path).resolve(), map_location="cpu")
    if not isinstance(ckpt, dict) or "hidden" not in ckpt or "mask" not in ckpt:
        raise ValueError(f"Invalid text embedding cache format: {path} (expect hidden/mask keys)")
    return parse_embedding(ckpt, device)


def expand_for_batch(hidden: torch.Tensor, mask: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    入参:
    - hidden: [1,T,D] 文本 token 特征。
    - mask: [1,T] padding 掩码（True 为 padding）。
    - batch_size: 当前 batch 大小。

    方法:
    - 将单条提示的文本特征广播到 batch 维（同一提示词、不同图像样本共享文本）。

    出参:
    - tuple: ([B,T,D], [B,T])。
    """
    return hidden.expand(batch_size, -1, -1), mask.expand(batch_size, -1)
