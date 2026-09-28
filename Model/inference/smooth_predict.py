# -*- coding: utf-8 -*-
"""
平滑分块预测工具
使用样条窗口函数对大影像进行分块预测并平滑融合重叠区域。
支持旋转镜像增强（TTA）提升预测一致性。

入参:
- input_img: numpy 图像 (H, W, C)
- window_size: 窗口边长
- subdivisions: 细分数（控制重叠程度，≥2）
- pred_func: 预测函数

方法:
- predict_img_with_smooth_windowing: 平滑滑窗预测（支持 TTA）
- cheap_tiling_prediction: 简单无重叠分块预测

出参:
- 预测结果数组
"""

import numpy as np
import scipy.signal


def _spline_window(window_size, power=2):
    """
    入参:
    - window_size (int): 窗口大小
    - power (int): 样条幂次

    方法:
    - 构造 1D 样条窗口函数

    出参:
    - return (np.ndarray): 1D 窗口权重
    """
    intersection = int(window_size / 4)
    outer = (abs(2 * scipy.signal.triang(window_size)) ** power) / 2
    outer[intersection:-intersection] = 0

    inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    inner[:intersection] = 0
    inner[-intersection:] = 0

    wind = (inner + outer)
    return wind / np.average(wind)


_cached_2d = {}


def _window_2d(window_size, power=2):
    """
    入参:
    - window_size (int): 窗口大小
    - power (int): 样条幂次

    方法:
    - 由 1D 窗口推断 2D 窗口（外积），带缓存

    出参:
    - return (np.ndarray): (window_size, window_size, 1, 1) 窗口权重
    """
    key = f"{window_size}_{power}"
    if key not in _cached_2d:
        w = _spline_window(window_size, power)
        w = w[np.newaxis, :, np.newaxis, np.newaxis]
        _cached_2d[key] = w * w.transpose(1, 0, 2)
    return _cached_2d[key]


def _pad_img(img, window_size, subdivisions):
    """reflect 填充使图像适合窗口划分"""
    aug = int(round(window_size * (1 - 1.0 / subdivisions)))
    return np.pad(img, ((aug, aug), (aug, aug), (0, 0)), mode='reflect')


def _unpad_img(img, window_size, subdivisions):
    """移除 padding"""
    aug = int(round(window_size * (1 - 1.0 / subdivisions)))
    return img[aug:-aug, aug:-aug, :]


def _rotate_mirror_do(img, slices=1):
    """
    入参:
    - img (np.ndarray): 图像
    - slices (int): 1/2/4/8，旋转镜像增强的变换数量

    方法:
    - 生成 img 的旋转/镜像变体列表

    出参:
    - return (list): 变换后的图像列表
    """
    mirrs = [np.array(img)]
    if slices >= 2:
        mirrs.append(np.rot90(img, k=1))
    if slices >= 4:
        flipped = img[:, ::-1]
        mirrs.append(flipped)
        mirrs.append(np.rot90(flipped, k=1))
    if slices >= 8:
        mirrs.append(np.rot90(img, k=2))
        mirrs.append(np.rot90(img, k=3))
        mirrs.append(np.rot90(flipped, k=2))
        mirrs.append(np.rot90(flipped, k=3))
    return mirrs


def _rotate_mirror_undo(imgs, slices=1):
    """将旋转/镜像变体逆向变换后取平均"""
    inv = [imgs[0]]
    if slices >= 2:
        inv.append(np.rot90(imgs[1], k=3))
    if slices >= 4:
        inv.append(imgs[2][:, ::-1])
        inv.append(np.rot90(imgs[3], k=3)[:, ::-1])
    if slices >= 8:
        inv.append(np.rot90(imgs[4], k=2))
        inv.append(np.rot90(imgs[5], k=1))
        inv.append(imgs[6][:, ::-1])
        inv.append(np.rot90(imgs[7], k=1)[:, ::-1])
    return sum(inv) / slices


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions,
                                      nb_classes, pred_func, tta_slices=1):
    """
    入参:
    - input_img (np.ndarray): (H, W, C) 输入图像
    - window_size (int): 窗口边长
    - subdivisions (int): 细分倍数（2 = 50% 重叠）
    - nb_classes (int): 输出类别数
    - pred_func (callable): 输入 (N, window_size, window_size, C) → (N, H, W, nb_classes)
    - tta_slices (int): TTA 变换数 1/2/4/8

    方法:
    - pad → rotate/mirror → 分块 → pred_func → 样条窗口加权 → 拼接 → 逆变换 → unpad

    出参:
    - return (np.ndarray): (H, W, nb_classes) 预测概率图
    """
    win2d = _window_2d(window_size)
    step = window_size // subdivisions
    padded = _pad_img(input_img, window_size, subdivisions)
    variants = _rotate_mirror_do(padded, tta_slices)

    results = []
    for var in variants:
        ph, pw = var.shape[:2]
        patches = []
        for i in range(0, ph - window_size + 1, step):
            for j in range(0, pw - window_size + 1, step):
                patches.append(var[i:i + window_size, j:j + window_size])

        patches = np.array(patches)
        preds = pred_func(patches).astype(np.float32)
        # 样条窗口加权
        preds = preds * win2d.squeeze()

        rows = (ph - window_size) // step + 1
        cols = (pw - window_size) // step + 1
        preds = preds.reshape(rows, cols, window_size, window_size, nb_classes)

        # 拼接
        merged = np.zeros(list(padded.shape[:2]) + [nb_classes], dtype=np.float32)
        count = np.zeros(list(padded.shape[:2]) + [nb_classes], dtype=np.float32)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                i, j = r * step, c * step
                merged[i:i + window_size, j:j + window_size] += preds[r, c]
                count[i:i + window_size, j:j + window_size] += win2d.squeeze()
                idx += 1
        count = np.maximum(count, 1e-6)
        results.append(merged / count)

    # 逆 TTA 并取平均
    merged_all = _rotate_mirror_undo(results, tta_slices)
    final = _unpad_img(merged_all, window_size, subdivisions)
    return final[:input_img.shape[0], :input_img.shape[1]]


def cheap_tiling_prediction(input_img, window_size, nb_classes, pred_func):
    """
    入参:
    - input_img (np.ndarray): (H, W, C)
    - window_size (int): 窗口边长
    - nb_classes (int): 类别数
    - pred_func (callable): 同上

    方法:
    - 无重叠分块预测，简单拼接

    出参:
    - return (np.ndarray): (H, W, nb_classes)
    """
    h, w = input_img.shape[:2]
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        padded = np.pad(input_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded = input_img

    ph, pw = padded.shape[:2]
    result = np.zeros((ph, pw, nb_classes), dtype=np.float32)

    for i in range(0, ph, window_size):
        for j in range(0, pw, window_size):
            patch = padded[i:i + window_size, j:j + window_size]
            if patch.shape[0] == window_size and patch.shape[1] == window_size:
                pred = pred_func(patch[np.newaxis])[0]
                result[i:i + window_size, j:j + window_size] = pred

    return result[:h, :w]
