# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Euclidean distance transform (EDT) — scipy fallback for Windows compatibility"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as scipy_edt


def edt_triton(data: torch.Tensor):
    """
    入参: data - shape (B, H, W) 的二值张量，True 表示前景
    方法: 用 scipy.ndimage.distance_transform_edt 替代 triton 内核，
          对 batch 中每张图逐个计算欧氏距离变换，再拼回张量
    出参: 与 data 同 shape 的距离场张量，等价于 cv2.distanceTransform
    """
    assert data.dim() == 3
    B, H, W = data.shape

    # 转到 CPU numpy，逐样本计算 EDT 后拼回
    device = data.device
    data_np = data.cpu().numpy().astype(bool)
    results = np.stack([scipy_edt(~data_np[i]) for i in range(B)], axis=0)

    return torch.from_numpy(results).float().to(device)
