# -*- coding: utf-8 -*-
"""
滑动窗口工具
为大尺寸栅格影像提供滑窗裁剪与结果拼接能力，支持重叠区域权重融合。

入参:
- image_shape: 影像 (H, W) 或 (H, W, C) 尺寸
- patch_size: 窗口边长
- stride: 滑动步长
- weight_type: 权重类型 'gaussian' / 'linear' / 'uniform'

方法:
- create_sliding_windows: 生成所有窗口坐标列表
- create_weight_map: 生成窗口权重图（用于重叠区域融合）
- stitch_result: 将窗口预测结果拼接为完整图

出参:
- 各函数返回值见函数签名
"""

import numpy as np


def create_sliding_windows(image_shape, patch_size, stride):
    """
    入参:
    - image_shape (tuple): 影像尺寸 (H, W) 或 (H, W, C)
    - patch_size (int): 窗口边长
    - stride (int): 步长

    方法:
    - 仅在能完整容纳窗口的区域生成坐标，保证所有窗口尺寸一致

    出参:
    - return (list[tuple]): [(x1, y1, x2, y2), ...]
    """
    h, w = image_shape[:2]
    windows = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            windows.append((x, y, x + patch_size, y + patch_size))
    return windows


def create_weight_map(patch_size, stride, weight_type='gaussian'):
    """
    入参:
    - patch_size (int): 窗口边长
    - stride (int): 步长（用于控制衰减宽度）
    - weight_type (str): 'gaussian' 高斯 | 'linear' 线性衰减 | 'uniform' 均匀

    方法:
    - gaussian: 二维高斯核，sigma=patch_size/4，中心为 1
    - linear: 四边线性衰减到 0，中心为 1
    - uniform: 全 1

    出参:
    - return (np.ndarray): shape=(patch_size, patch_size) 的权重图
    """
    if weight_type == 'gaussian':
        sigma = patch_size / 4
        x = np.linspace(-patch_size / 2, patch_size / 2, patch_size)
        xx, yy = np.meshgrid(x, x)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return kernel / kernel.max()

    if weight_type == 'linear':
        half = patch_size // 2
        x = np.linspace(0, 1, half)
        xx, yy = np.meshgrid(x, x)
        tl = xx * yy
        tr = np.fliplr(xx) * yy
        bl = xx * np.flipud(yy)
        center = np.ones((half, half))
        kernel = np.zeros((patch_size, patch_size))
        kernel[:half, :half] = tl
        kernel[:half, half:] = tr
        kernel[half:, :half] = bl
        kernel[half:, half:] = center
        return kernel

    return np.ones((patch_size, patch_size))


def stitch_result(result_shape, predictions, windows, patch_size,
                  overlap_weights=False, weight_map=None):
    """
    入参:
    - result_shape (tuple): 输出图尺寸 (H, W)
    - predictions (list[np.ndarray]): 每个窗口的预测结果
    - windows (list[tuple]): 窗口坐标列表
    - patch_size (int): 窗口边长
    - overlap_weights (bool): 是否使用权重融合重叠区域
    - weight_map (np.ndarray|None): 权重图

    方法:
    - 直接覆盖模式：后写入的窗口覆盖先写入的
    - 权重融合模式：累加预测值*权重，最终除以权重之和

    出参:
    - return (np.ndarray): 拼接后的完整预测图
    """
    h, w = result_shape
    result = np.zeros((h, w), dtype=np.float32)

    if overlap_weights and weight_map is not None:
        count_map = np.zeros((h, w), dtype=np.float32)
        for idx, (x1, y1, x2, y2) in enumerate(windows):
            if idx >= len(predictions):
                break
            result[y1:y2, x1:x2] += predictions[idx] * weight_map
            count_map[y1:y2, x1:x2] += weight_map
        count_map = np.maximum(count_map, 1e-6)
        result = result / count_map
    else:
        for idx, (x1, y1, x2, y2) in enumerate(windows):
            if idx >= len(predictions):
                break
            result[y1:y2, x1:x2] = predictions[idx]

    return result


def tif_cropping_array(img, side_length, img_size):
    """
    入参:
    - img (np.ndarray): 大影像数组 (H, W, C)
    - side_length (int): 重叠边长
    - img_size (int): 裁剪块尺寸

    方法:
    - 滑动窗口裁剪，处理行列末尾剩余区域
    - 返回二维数组 [行][列] 的裁剪块列表

    出参:
    - return (tuple): (TifArray, RowOver, ColumnOver)
    """
    result = []
    col_num = int((img.shape[0] - side_length * 2) / (img_size - side_length * 2))
    row_num = int((img.shape[1] - side_length * 2) / (img_size - side_length * 2))
    step = img_size - side_length * 2

    for i in range(col_num):
        row_arr = []
        for j in range(row_num):
            row_arr.append(img[i * step:i * step + img_size, j * step:j * step + img_size])
        result.append(row_arr)

    # 末尾列
    for i in range(col_num):
        cropped = img[i * step:i * step + img_size, img.shape[1] - img_size:img.shape[1]]
        result[i].append(cropped)

    # 末尾行 + 右下角
    last_row = []
    for j in range(row_num):
        last_row.append(img[img.shape[0] - img_size:img.shape[0], j * step:j * step + img_size])
    last_row.append(img[img.shape[0] - img_size:img.shape[0], img.shape[1] - img_size:img.shape[1]])
    result.append(last_row)

    col_over = (img.shape[0] - side_length * 2) % step + side_length
    row_over = (img.shape[1] - side_length * 2) % step + side_length
    return result, row_over, col_over


def stitch_tif_result(shape, tif_array, predictions, repetitive_length,
                      row_over, col_over, img_size):
    """
    入参:
    - shape (tuple): 原始影像 (H, W)
    - tif_array: TifCroppingArray 返回的裁剪结构
    - predictions (list): 每块的预测结果
    - repetitive_length (int): 重复边长
    - row_over (int): 行剩余像素
    - col_over (int): 列剩余像素
    - img_size (int): 裁剪块尺寸

    方法:
    - 按 tif_array 的网格结构拼接预测结果
    - 处理边缘和角落区域的重叠

    出参:
    - return (np.ndarray): 拼接后的完整结果 (H, W) uint8
    """
    result = np.zeros(shape, np.uint8)
    cols_per_row = len(tif_array[0])
    j = 0
    step = img_size - 2 * repetitive_length

    for i, pred in enumerate(predictions):
        col_idx = i % cols_per_row
        if col_idx == 0 and i > 0:
            j += 1

        # 四个角落和边缘的特殊处理
        is_first_row = (j == 0)
        is_last_row = (j == len(tif_array) - 1)
        is_first_col = (col_idx == 0)
        is_last_col = (col_idx == cols_per_row - 1)

        y_src_start = 0 if is_first_row else repetitive_length
        y_src_end = img_size - repetitive_length if not is_last_row else img_size
        x_src_start = 0 if is_first_col else repetitive_length
        x_src_end = img_size - repetitive_length if not is_last_col else img_size

        y_dst_start = 0 if is_first_row else j * step + repetitive_length
        if is_last_row:
            y_dst_end = shape[0]
        else:
            y_dst_end = (j + 1) * step + repetitive_length

        x_dst_start = 0 if is_first_col else col_idx * step + repetitive_length
        if is_last_col:
            x_dst_end = shape[1]
        else:
            x_dst_end = (col_idx + 1) * step + repetitive_length

        result[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = pred[y_src_start:y_src_end, x_src_start:x_src_end]

    return result
