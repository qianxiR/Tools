# -*- coding: utf-8 -*-
"""
波段提取工具
从本地 GeoTIFF 或坐标点提取波段值，支持智能波段映射检测。

入参:
- tiff_path: GeoTIFF 文件路径
- points_df: 含经纬度列的 DataFrame
- band_mapping: 显式波段名→索引映射（可选）

方法:
- detect_band_mapping: 从波段描述/标签自动检测 Sentinel-2 等波段映射
- extract_band_values_at_points: 在 WGS84 坐标点提取各波段值
- geotiff_to_csv: 逐像素提取经纬度和波段值

出参:
- 各函数返回值见签名
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer


def detect_band_mapping(tiff_path):
    """
    入参:
    - tiff_path (str): GeoTIFF 文件路径

    方法:
    - 读取每个波段的 description / 标签
    - 匹配 Sentinel-2 波段名（B1-B12）
    - 若无描述，根据波段数量推断默认映射

    出参:
    - return (dict): {波段名: 波段索引(1-based)}，如 {'B2': 1, 'B3': 2, ...}
    """
    S2_PATTERN = {'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'}
    DEFAULT_4B = {'B2': 1, 'B3': 2, 'B4': 3, 'B8': 4}
    DEFAULT_3B = {'B2': 1, 'B3': 2, 'B4': 3}

    mapping = {}
    with rasterio.open(tiff_path) as src:
        n = src.count
        for i in range(1, n + 1):
            desc = src.descriptions[i - 1] if src.descriptions else None
            tags = src.tags(i)
            name = None

            # 从 description 识别
            if desc:
                upper = desc.upper().strip()
                for b in S2_PATTERN:
                    if b in upper:
                        name = b
                        break

            # 从标签识别
            if name is None and tags:
                for v in tags.values():
                    upper = str(v).upper().strip()
                    for b in S2_PATTERN:
                        if b in upper:
                            name = b
                            break
                    if name:
                        break

            if name:
                mapping[name] = i

        # 无描述时根据波段数量推断
        if not mapping:
            if n >= 4:
                return dict(DEFAULT_4B)
            if n == 3:
                return dict(DEFAULT_3B)
            return {f'B{i}': i for i in range(1, n + 1)}

    return mapping


def extract_band_values_at_points(tiff_path, lon, lat, band_indices=None):
    """
    入参:
    - tiff_path (str): GeoTIFF 路径
    - lon (array-like): WGS84 经度
    - lat (array-like): WGS84 纬度
    - band_indices (list[int]|None): 1-based 波段列表，None 则提取所有波段

    方法:
    - 自动将 WGS84 坐标转换到影像 CRS
    - 使用 rasterio.transform.rowcol 计算像素行列
    - 读取对应像素值，应用 scales 偏移

    出参:
    - return (np.ndarray): shape=(N, n_bands) 的波段值矩阵
    """
    with rasterio.open(tiff_path) as src:
        # 坐标转换
        if src.crs and src.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = np.asarray(lon), np.asarray(lat)

        rows, cols = rowcol(src.transform, x, y)
        bands = band_indices or list(range(1, src.count + 1))
        n = len(rows)
        result = np.full((n, len(bands)), np.nan, dtype=np.float32)

        h, w = src.height, src.width
        scales = [src.scales[b - 1] if src.scales else 1.0 for b in bands]
        offsets = [src.offsets[b - 1] if src.offsets else 0.0 for b in bands]

        for j, b in enumerate(bands):
            data = src.read(b)
            for i in range(n):
                r, c = rows[i], cols[i]
                if 0 <= r < h and 0 <= c < w:
                    result[i, j] = data[r, c] * scales[j] + offsets[j]

    return result


def geotiff_to_csv(tiff_path, output_csv_path, chunk_size=None, progress_fn=None):
    """
    入参:
    - tiff_path (str): 输入 GeoTIFF 路径
    - output_csv_path (str): 输出 CSV 路径
    - chunk_size (int|None): 分块行数，None 则整幅处理
    - progress_fn (callable|None): 进度回调 fn(current, total)

    方法:
    - 逐像素提取经纬度和各波段值
    - 大影像自动分块处理避免内存溢出
    - 写入 CSV（列：lon, lat, B2, B3, ...）

    出参:
    - return (int): 写入的总行数
    """
    import csv
    import pandas as pd

    with rasterio.open(tiff_path) as src:
        n_bands = src.count
        mapping = detect_band_mapping(tiff_path)
        inv = {v: k for k, v in mapping.items()}
        col_names = ['lon', 'lat']
        for b in range(1, n_bands + 1):
            col_names.append(inv.get(b, f'band_{b}'))

        transform = src.transform
        total_rows = src.height

        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(col_names)

            written = 0
            chunk = chunk_size or total_rows

            for y_start in range(0, total_rows, chunk):
                y_end = min(y_start + chunk, total_rows)
                h = y_end - y_start
                w = src.width

                cols_arr, rows_arr = np.meshgrid(np.arange(w), np.arange(h))
                xs, ys = rasterio.transform.xy(transform, rows_arr + y_start, cols_arr)

                lons = np.asarray(xs).flatten()
                lats = np.asarray(ys).flatten()

                rows_data = []
                for b in range(1, n_bands + 1):
                    data = src.read(b, window=((y_start, y_end), (0, w)))
                    rows_data.append(data.flatten())

                batch = np.column_stack([lons, lats] + rows_data)
                for row in batch:
                    writer.writerow(row.tolist())

                written += len(lons)
                if progress_fn:
                    progress_fn(y_end, total_rows)

    return written
