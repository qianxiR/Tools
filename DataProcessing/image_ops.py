# -*- coding: utf-8 -*-
"""
影像增强工具
位深转换（16bit→8bit）、直方图匹配、光谱指数计算。

入参:
- input_path / output_path: 输入输出文件路径
- nodata: 无效值
- percentiles: 百分位裁剪范围
- index_type: 'NDVI' / 'NDWI'

方法:
- convert_to_8bit: 16bit 影像转 8bit（百分位裁剪 / 最小最大值）
- calc_spectral_index: 计算 NDVI/NDWI 指数
- histogram_match: 直方图匹配

出参:
- 各函数返回值见签名
"""

import os
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def convert_to_8bit(input_path, output_path, nodata=65535,
                    percentiles=None, block_h=10000):
    """
    入参:
    - input_path (str): 输入栅格路径
    - output_path (str): 输出 8bit 栅格路径
    - nodata (int|float): NoData 值
    - percentiles (list|None): [p_min, p_max] 百分位裁剪范围，None 则用 min/max
    - block_h (int): 分块高度

    方法:
    - 先扫描全图统计每波段的 min/max（或百分位值）
    - 分块处理：线性拉伸到 0-254，NoData 填 255
    - 复制地理参考信息到输出文件

    出参:
    - return (bool): 是否转换成功
    """
    ds = gdal.Open(input_path)
    if ds is None:
        return False

    height, width = ds.RasterYSize, ds.RasterXSize
    n_bands = ds.RasterCount
    mins = np.zeros(n_bands, np.float32)
    maxs = np.zeros(n_bands, np.float32)

    # 统计每波段值域
    for i in range(n_bands):
        band = ds.GetRasterBand(i + 1)
        band.SetNoDataValue(nodata)
        arr = band.ReadAsArray().astype(np.float32)
        mask = arr == nodata
        arr[mask] = np.nan

        if percentiles:
            mins[i] = np.nanpercentile(arr, percentiles[0])
            maxs[i] = np.nanpercentile(arr, percentiles[1])
        else:
            mins[i] = np.nanmin(arr)
            maxs[i] = np.nanmax(arr)

        if mins[i] < 0:
            mins[i] = 0

    # 创建输出
    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(output_path, width, height, n_bands, gdal.GDT_Byte,
                        options=['COMPRESS=LZW'])
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())

    bk = min(block_h, height)
    n_blocks = (height + bk - 1) // bk

    for bi in range(n_blocks):
        y0 = bi * bk
        yh = min(bk, height - y0)
        img = ds.ReadAsArray(0, y0, width, yh).transpose(1, 2, 0)

        for i in range(n_bands):
            ch = img[:, :, i].astype(np.float32)
            nodata_mask = ch == nodata
            ch[nodata_mask] = mins[i]
            stretched = 254.0 * (ch - mins[i]) / max(maxs[i] - mins[i], 1e-6)
            stretched = np.clip(stretched, 0, 254).astype(np.uint8)
            stretched[nodata_mask] = 255
            out.GetRasterBand(i + 1).WriteArray(stretched, 0, y0)

    ds = None
    out = None
    return True


def calc_spectral_index(image_path, output_path='', index_type='NDVI',
                        nodata=65535, block_h=10000):
    """
    入参:
    - image_path (str): 输入 4 波段 BGRN 影像路径
    - output_path (str): 输出路径，空则自动命名
    - index_type (str): 'NDVI' 或 'NDWI'
    - nodata: NoData 值
    - block_h (int): 分块高度

    方法:
    - NDVI = (NIR - Red) / (NIR + Red)
    - NDWI = (Green - NIR) / (Green + NIR)
    - 结果映射到 0-254，NoData 填 255

    出参:
    - return (str|None): 输出文件路径，失败返回 None
    """
    ds = gdal.Open(image_path)
    if ds is None or ds.RasterCount < 4:
        return None

    height, width = ds.RasterYSize, ds.RasterXSize

    if not output_path:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(os.path.dirname(image_path), f"{base}_{index_type}.tif")

    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(output_path, width, height, 1, gdal.GDT_Byte,
                        options=['COMPRESS=LZW'])
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())

    bk = min(block_h, height)
    n_blocks = (height + bk - 1) // bk
    eps = 1e-8

    for bi in range(n_blocks):
        y0 = bi * bk
        yh = min(bk, height - y0)
        img = ds.ReadAsArray(0, y0, width, yh).transpose(1, 2, 0).astype(np.float32)
        nodata_mask = img[:, :, 0] == nodata

        blue, green, red, nir = [img[:, :, i] for i in range(4)]

        if index_type == 'NDVI':
            idx = (nir - red) / (nir + red + eps)
        elif index_type == 'NDWI':
            idx = (green - nir) / (green + nir + eps)
        else:
            ds = None
            out = None
            return None

        # [-1, 1] → [0, 254]
        result = ((idx + 1.0) * 127.0).clip(0, 254).astype(np.uint8)
        result[nodata_mask] = 255
        out.GetRasterBand(1).WriteArray(result, 0, y0)

    ds = None
    out = None
    return output_path


def histogram_match(source_img, reference_img):
    """
    入参:
    - source_img (np.ndarray): 待匹配图像 (H, W) 或 (H, W, C)
    - reference_img (np.ndarray): 参考图像，形状可不同，通道数须一致

    方法:
    - 逐通道计算累积直方图，通过映射表将 source 匹配到 reference 分布

    出参:
    - return (np.ndarray): 匹配后的图像，dtype 与 source 相同
    """
    if source_img.ndim == 2:
        source_img = source_img[:, :, np.newaxis]
        reference_img = reference_img[:, :, np.newaxis]

    result = np.zeros_like(source_img)
    for c in range(source_img.shape[2]):
        src = source_img[:, :, c].flatten()
        ref = reference_img[:, :, c].flatten()
        src_vals, src_idx, src_cnt = np.unique(src, return_inverse=True, return_counts=True)
        ref_vals, ref_cnt = np.unique(ref, return_counts=True)

        src_cdf = np.cumsum(src_cnt).astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf = np.cumsum(ref_cnt).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        # 最近邻匹配
        mapping = np.interp(src_cdf, ref_cdf, ref_vals)
        result[:, :, c] = mapping[src_idx].reshape(source_img.shape[:2])

    return result.squeeze() if result.shape[2] == 1 else result
