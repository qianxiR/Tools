# -*- coding: utf-8 -*-
"""
边缘提取工具
使用欧几里得距离变换从二值掩码中提取边界。

入参:
- mask: 二值掩码 (H, W) numpy 数组
- threshold: 距离阈值
- padding_size: 边缘填充大小

方法:
- distance_transform_boundary: 单张掩码的距离变换边界提取
- batch_extract_boundaries: 批量处理文件夹中的掩码图像

出参:
- boundary 掩码数组（float32，0-1 范围）
"""

import os
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def distance_transform_boundary(mask, threshold=2, padding_size=10):
    """
    入参:
    - mask (np.ndarray): 二值掩码 (H, W)，值域 0/1 或 0/255
    - threshold (float): 距离阈值，控制边界粗细
    - padding_size (int): 边缘填充像素数，避免边缘信息丢失

    方法:
    - padding 后分别计算前景和背景的欧几里得距离变换
    - 距离之和 ≤ threshold 且前景/背景距离均 ≤ threshold 的像素为边界
    - 移除 padding 恢复原始尺寸

    出参:
    - return (np.ndarray): 边界掩码 float32 (H, W)，值域 0.0-1.0
    """
    binary = (mask > 0).astype(np.float32)
    # 边缘填充避免边界信息丢失
    padded = np.pad(binary, padding_size, mode='constant', constant_values=0)

    # 前景距离：每个前景像素到最近背景像素的距离
    fg_dist = distance_transform_edt(padded > 0)
    # 背景距离：每个背景像素到最近前景像素的距离
    bg_dist = distance_transform_edt(padded == 0)

    combined = fg_dist + bg_dist
    boundary = (combined <= threshold).astype(np.float32)
    # 仅保留前景与背景交界区域
    boundary = boundary * (fg_dist <= threshold) * (bg_dist <= threshold)

    # 移除 padding
    return boundary[padding_size:-padding_size, padding_size:-padding_size]


def canny_edge(mask, low_threshold=50, high_threshold=150):
    """
    入参:
    - mask (np.ndarray): uint8 掩码
    - low_threshold (int): Canny 低阈值
    - high_threshold (int): Canny 高阈值

    方法:
    - OpenCV Canny 边缘检测

    出参:
    - return (np.ndarray): uint8 边缘图
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    return cv2.Canny(mask, low_threshold, high_threshold)


def sobel_edge(image):
    """
    入参:
    - image (np.ndarray): 灰度或彩色图像

    方法:
    - Sobel 算子计算 X/Y 方向梯度，合并取绝对值

    出参:
    - return (np.ndarray): uint8 边缘图
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.uint8)


def batch_extract_boundaries(input_folder, output_folder, method='distance',
                             threshold=2, padding_size=10):
    """
    入参:
    - input_folder (str): 输入掩码文件夹
    - output_folder (str): 输出边界文件夹
    - method (str): 'distance' 距离变换 | 'canny' | 'sobel'
    - threshold (float): 距离变换阈值（仅 method='distance'）
    - padding_size (int): 填充大小（仅 method='distance'）

    方法:
    - 遍历文件夹中所有 PNG 文件，批量提取边界并保存

    出参:
    - return (int): 成功处理的文件数
    """
    os.makedirs(output_folder, exist_ok=True)
    files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith('.png'))
    count = 0

    for fname in files:
        img = cv2.imread(os.path.join(input_folder, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if method == 'distance':
            binary = (img > 0).astype(np.float32)
            edge = distance_transform_boundary(binary, threshold, padding_size)
            result = (edge * 255).astype(np.uint8)
        elif method == 'canny':
            result = canny_edge(img)
        elif method == 'sobel':
            result = sobel_edge(img)
        else:
            continue

        cv2.imwrite(os.path.join(output_folder, f"edge_{fname}"), result)
        count += 1

    return count
