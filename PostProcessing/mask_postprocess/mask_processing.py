# -*- coding: utf-8 -*-
"""
掩码处理工具
对二值掩码进行形态学操作、边缘平滑和后处理。

入参:
- mask: 二值掩码 numpy 数组
- kernel_size: 形态学核大小
- iterations: 迭代次数

方法:
- smooth_mask: 边缘平滑（中值 + 形态学 + 高斯）
- remove_small_objects: 按面积去除小连通域
- binary_threshold: 智能二值化（支持多种 dtype）

出参:
- 处理后的掩码数组
"""

import numpy as np
import cv2


def smooth_mask(mask, median_k=5, morph_k=3, gaussian_k=5, sigma=1.0):
    """
    入参:
    - mask (np.ndarray): uint8 掩码 (0/255 或 0/1)
    - median_k (int): 中值滤波核大小（奇数）
    - morph_k (int): 形态学核大小
    - gaussian_k (int): 高斯核大小（奇数）
    - sigma (float): 高斯 sigma

    方法:
    - 中值滤波 → 开运算（去噪）→ 闭运算（填洞）→ 高斯模糊 → 二值化

    出参:
    - return (np.ndarray): 平滑后的 uint8 掩码
    """
    binary = binary_threshold(mask)
    result = cv2.medianBlur(binary, median_k)
    kernel = np.ones((morph_k, morph_k), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    blurred = cv2.GaussianBlur(result.astype(np.float32), (gaussian_k, gaussian_k), sigma)
    return (blurred > 0.5).astype(np.uint8) * 255


def remove_small_objects(mask, min_area=100, connectivity=8):
    """
    入参:
    - mask (np.ndarray): 二值掩码
    - min_area (int): 最小连通域面积（像素）
    - connectivity (int): 4 或 8 连通

    方法:
    - cv2.connectedComponents 标记连通域
    - 按面积过滤后重建掩码

    出参:
    - return (np.ndarray): 过滤后的 uint8 掩码
    """
    binary = binary_threshold(mask)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=connectivity)
    result = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        if np.sum(labels == label_id) >= min_area:
            result[labels == label_id] = 255
    return result


def binary_threshold(mask, threshold=127):
    """
    入参:
    - mask (np.ndarray): 任意 dtype 的掩码
    - threshold (int|float): 二值化阈值

    方法:
    - bool → 直接转 0/255
    - float → 0.5 阈值
    - uint8 → threshold 阈值

    出参:
    - return (np.ndarray): uint8 二值掩码 (0 或 255)
    """
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    if mask.ndim > 2:
        mask = mask[:, :, 0]

    if mask.dtype in (np.bool_, bool):
        return mask.astype(np.uint8) * 255
    if np.issubdtype(mask.dtype, np.floating):
        return (mask > 0.5).astype(np.uint8) * 255
    if mask.max() <= 1:
        return (mask > 0).astype(np.uint8) * 255
    return (mask > threshold).astype(np.uint8) * 255
