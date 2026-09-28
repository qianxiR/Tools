# -*- coding: utf-8 -*-
"""
模型加载与推理工具
提供带缓存的模型加载、GPU 内存管理和批次大小自适应调整。

入参:
- checkpoint: 权重文件路径
- device: 'cuda:0' 或 'cpu'
- patch_size: 滑窗窗口尺寸
- model_class: 模型类（如 Trainer）
- model_fn: 模型构建函数

方法:
- load_model_cached: 带全局缓存的模型加载
- clear_model_cache: 清空缓存并释放 GPU 显存
- estimate_memory_usage: 估算单批次显存占用
- adjust_batch_size: 根据可用显存自适应调整 batch_size

出参:
- load_model_cached → 模型实例
- adjust_batch_size → int (建议 batch_size)
"""

import torch

# 全局模型缓存
_MODEL_CACHE = {}


def load_model_cached(model_fn, checkpoint, device, **kwargs):
    """
    入参:
    - model_fn (callable): 无参模型构造函数，返回 nn.Module
    - checkpoint (str): 权重文件路径
    - device (str): 设备
    - **kwargs: 传递给 model_fn 的额外参数

    方法:
    - 以 (checkpoint, device) 为键缓存模型
    - 支持 state_dict 和直接 checkpoint 两种权重格式
    - 加载后设为 eval 模式

    出参:
    - return (nn.Module): 已加载权重的模型
    """
    cache_key = f"{checkpoint}_{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = model_fn(**kwargs).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def clear_model_cache():
    """
    入参: 无

    方法:
    - 清空全局模型缓存字典并释放 GPU 显存

    出参: 无
    """
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_memory_usage(patch_size, batch_size=1):
    """
    入参:
    - patch_size (int): 窗口边长
    - batch_size (int): 批次大小

    方法:
    - 粗略估算显存：输入+输出+特征图 + 100MB 模型开销

    出参:
    - return (float): 估算内存 (MB)
    """
    bpv = 4  # float32
    input_per = 2 * 3 * patch_size * patch_size   # 前后时相各 3 通道
    output_per = patch_size * patch_size
    features_per = 20 * 3 * patch_size * patch_size
    model_overhead = 100 * 1024 * 1024  # 100MB
    batch_mem = batch_size * (input_per + output_per + features_per) * bpv
    return (model_overhead + batch_mem) / (1024 * 1024)


def adjust_batch_size(patch_size, base_batch_size, device='cuda:0', auto=True):
    """
    入参:
    - patch_size (int): 窗口边长
    - base_batch_size (int): 用户指定的默认批次大小
    - device (str): 设备
    - auto (bool): 是否启用自适应调整

    方法:
    - CPU 模式：大 patch 限 2，小 patch 限 4
    - GPU 模式：读取可用显存，估算单样本显存，计算最优 batch_size
    - 结果限制在 [1, 16] 范围

    出参:
    - return (int): 调整后的 batch_size
    """
    if not auto or device == 'cpu':
        if patch_size > 512:
            return min(base_batch_size, 2)
        return min(base_batch_size, 4)

    if not torch.cuda.is_available():
        return base_batch_size

    try:
        dev_idx = int(device.split(':')[-1]) if ':' in device else 0
        total = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 2)
        allocated = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
        available = (total - allocated) * 0.8
        per_sample = estimate_memory_usage(patch_size, 1)
        optimal = max(1, min(int(available / per_sample), 16))
        return optimal if optimal != base_batch_size else base_batch_size
    except Exception:
        return base_batch_size


def pad_image_to_multiple(image, divisor=256):
    """
    入参:
    - image (np.ndarray): 影像数组 (H, W, C)
    - divisor (int): 对齐除数

    方法:
    - 使用 reflect 模式填充影像，使 H 和 W 为 divisor 的整数倍

    出参:
    - return (tuple): (padded_image, pad_h, pad_w)
    """
    import numpy as np
    h, w = image.shape[:2]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        if image.ndim == 3:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    return image, pad_h, pad_w
