# -*- coding: utf-8 -*-
"""
结果保存与可视化工具
保存带有地理参考的 GeoTIFF 掩码，以及前后对比可视化四联图。

入参:
- mask: 预测掩码 (H, W)
- output_path: 输出路径
- geo_transform / projection: 地理参考信息
- pre_img / post_img / pred_mask: 用于可视化的图像数组

方法:
- save_geotiff_mask: 保存掩码为 GeoTIFF（含边缘平滑）
- visualize_quad_view: 生成前后时相 + 检测结果 + 二值掩码四联图

出参:
- save_geotiff_mask → bool
- visualize_quad_view → bool
"""

import os
import numpy as np
import cv2
from osgeo import gdal


def save_geotiff_mask(mask, output_path, geo_transform=None, projection=None,
                      smooth=True, nodata=0):
    """
    入参:
    - mask (np.ndarray): 预测掩码 (H, W)，uint8 或 float32
    - output_path (str): 输出文件路径
    - geo_transform (tuple|None): 仿射变换参数
    - projection (str|None): 投影 WKT
    - smooth (bool): 是否对二值掩码做边缘平滑
    - nodata: NoData 值

    方法:
    - smooth 模式：中值滤波 → 形态学开闭 → 高斯模糊 → 二值化
    - GDAL 创建 GeoTIFF，LZW 压缩，写入地理参考

    出参:
    - return (bool): 是否保存成功
    """
    try:
        if smooth and mask.dtype == np.uint8:
            m = cv2.medianBlur(mask, 5)
            k = np.ones((3, 3), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
            m = cv2.GaussianBlur(m.astype(np.float32), (5, 5), 1.0)
            mask = (m > 127).astype(np.uint8) * 255

        dtype = gdal.GDT_Float32 if mask.dtype in (np.float32, np.float64) else gdal.GDT_Byte
        h, w = mask.shape[:2]

        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(output_path, w, h, 1, dtype, options=['COMPRESS=LZW'])

        if geo_transform is not None:
            ds.SetGeoTransform(geo_transform)
        if projection is not None:
            ds.SetProjection(projection)

        ds.GetRasterBand(1).WriteArray(mask)
        ds.GetRasterBand(1).SetNoDataValue(nodata)
        ds.FlushCache()
        ds = None
        return True
    except Exception:
        try:
            cv2.imwrite(output_path, mask)
            return True
        except Exception:
            return False


def visualize_quad_view(pre_img, post_img, pred_mask, save_path,
                        is_raw_output=False, max_display_size=1200):
    """
    入参:
    - pre_img (np.ndarray): 前时相图像 BGR (H, W, 3)
    - post_img (np.ndarray): 后时相图像 BGR (H, W, 3)
    - pred_mask (np.ndarray): 预测掩码 (H, W)
    - save_path (str): 输出路径
    - is_raw_output (bool): 是否为概率图（True 则用热力图着色）
    - max_display_size (int): 最大显示边长（超过则缩放）

    方法:
    - 缩放到 max_display_size 以内
    - raw_output: JET 热力图叠加后时相
    - 二值输出: 红色标记变化区域
    - 拼接四联图：前时相 | 后时相 | 叠加结果 | 二值掩码

    出参:
    - return (bool): 是否保存成功
    """
    if pred_mask is None or pred_mask.size == 0:
        return False

    if pred_mask.ndim > 2:
        pred_mask = pred_mask[:, :, 0]

    # 尺寸对齐
    h, w = pre_img.shape[:2]
    if pred_mask.shape[:2] != (h, w):
        interp = cv2.INTER_LINEAR if is_raw_output else cv2.INTER_NEAREST
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=interp)

    # 边缘平滑（仅二值模式）
    if not is_raw_output:
        m = cv2.medianBlur(pred_mask, 5)
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
        m = cv2.GaussianBlur(m.astype(np.float32), (5, 5), 1.0)
        pred_mask = (m > 127).astype(np.uint8) * 255

    # 缩放
    if h > max_display_size or w > max_display_size:
        scale = min(max_display_size / h, max_display_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pre_img = cv2.resize(pre_img, (new_w, new_h))
        post_img = cv2.resize(post_img, (new_w, new_h))
        interp = cv2.INTER_LINEAR if is_raw_output else cv2.INTER_NEAREST
        pred_mask = cv2.resize(pred_mask, (new_w, new_h), interpolation=interp)
        h, w = new_h, new_w

    # 叠加着色
    if is_raw_output:
        heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(post_img, 0.7, heatmap, 0.3, 0)
    else:
        overlay = post_img.copy()
        mask_cond = pred_mask > 127
        overlay[..., 0] = np.where(mask_cond, 255, post_img[..., 0])
        overlay[..., 1] = np.where(mask_cond, 0, post_img[..., 1])
        overlay[..., 2] = np.where(mask_cond, 0, post_img[..., 2])

    # 二值掩码着色（绿色）
    binary = np.zeros((h, w, 3), dtype=np.uint8)
    if is_raw_output:
        binary[..., 1] = (pred_mask > 0.5).astype(np.uint8) * 255
    else:
        binary[..., 1] = pred_mask

    # 四联图拼接
    gap = 5
    canvas = np.ones((h, w * 4 + gap * 3, 3), dtype=np.uint8) * 255
    canvas[:, :w] = pre_img
    canvas[:, w + gap:w * 2 + gap] = post_img
    canvas[:, w * 2 + gap * 2:w * 3 + gap * 2] = overlay
    canvas[:, w * 3 + gap * 3:w * 4 + gap * 3] = binary

    # 标题
    canvas = cv2.copyMakeBorder(canvas, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    titles = ["Pre-event", "Post-event", "Detection", "Binary Mask"]
    for i, t in enumerate(titles):
        x = int((i + 0.5) * w + i * gap - len(t) * 5)
        cv2.putText(canvas, t, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 保存
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(save_path)
        return True
    except Exception:
        return False
