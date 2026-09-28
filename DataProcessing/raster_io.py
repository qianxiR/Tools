# -*- coding: utf-8 -*-
"""
栅格影像读写工具
统一封装 GeoTIFF 栅格文件的读取与写入操作，保留地理参考信息。

入参:
- raster_path: 栅格文件路径
- selected_bands: 波段索引列表，默认 [1,2,3]
- img_data: numpy 数组 (H,W) 或 (H,W,C)
- output_path: 输出文件路径
- im_proj: 投影 WKT 字符串
- im_geotrans: 仿射变换六参数元组

方法:
- read_geotiff: 读取 GeoTIFF，返回 uint8 RGB 数组 + 地理信息
- read_img_with_geo: 读取影像原始数据 + 地理信息（不做归一化）
- write_img_with_geo: 将数组写入 GeoTIFF 并写入地理信息
- get_geo_info: 仅获取地理参考信息（不读取像素）

出参:
- 各函数返回值见函数签名
"""

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def read_geotiff(raster_path, selected_bands=None):
    """
    入参:
    - raster_path (str): GeoTIFF 文件路径
    - selected_bands (list[int]): 1-indexed 波段列表，默认 [1,2,3]

    方法:
    - 读取 GeoTIFF，对每个波段做 2%-98% 百分位拉伸归一化到 0-255
    - 统一地理变换参数精度：坐标 2 位小数，分辨率 6 位小数

    出参:
    - return (tuple): (image_uint8, geo_transform, projection, gdal_dataset)
    """
    ds = gdal.Open(raster_path)
    if ds is None:
        raise ValueError(f"无法打开栅格文件: {raster_path}")

    geo_transform = list(ds.GetGeoTransform())
    # 坐标值保留2位小数，分辨率保留6位
    geo_transform[0] = float(f"{geo_transform[0]:.2f}")
    geo_transform[3] = float(f"{geo_transform[3]:.2f}")
    for i in (1, 2, 4, 5):
        geo_transform[i] = float(f"{geo_transform[i]:.6f}")
    geo_transform = tuple(geo_transform)

    projection = ds.GetProjection()

    # 默认读取 RGB 三波段
    if selected_bands is None:
        selected_bands = [1, 2, 3]

    bands = []
    for band_idx in selected_bands:
        bands.append(ds.GetRasterBand(band_idx).ReadAsArray())

    image = np.stack(bands, axis=2)

    # 百分位拉伸归一化到 0-255
    for i in range(image.shape[2]):
        ch = image[:, :, i]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        ch = np.clip(ch, p2, p98)
        ch = ((ch - p2) / max(p98 - p2, 1e-6) * 255).astype(np.uint8)
        image[:, :, i] = ch

    image = np.nan_to_num(image, nan=0, posinf=255, neginf=0).astype(np.uint8)
    return image, geo_transform, projection, ds


def read_img_with_geo(src_path):
    """
    入参:
    - src_path (str): 图像文件路径（支持 .tif / .img）

    方法:
    - 读取原始像素值（不做归一化），保留完整地理参考
    - 多波段按 BGR 顺序堆叠（与 OpenCV 一致）

    出参:
    - return (tuple): (image_HWC, im_proj, im_geotrans)
    """
    dataset = gdal.Open(src_path, gdal.GA_ReadOnly)
    if dataset is None:
        alt = src_path.replace('.tif', '.img').replace('.TIF', '.IMG')
        dataset = gdal.Open(alt, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"无法打开图像文件: {src_path}")

    im_proj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()
    bands = dataset.RasterCount

    if bands >= 3:
        # BGR 顺序，与 OpenCV/mmcv 一致
        b = dataset.GetRasterBand(1).ReadAsArray()
        g = dataset.GetRasterBand(2).ReadAsArray()
        r = dataset.GetRasterBand(3).ReadAsArray()
        image = np.dstack((b, g, r))
    else:
        image = dataset.GetRasterBand(1).ReadAsArray()
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

    dataset = None
    return image, im_proj, im_geotrans


def write_img_with_geo(img_data, output_path, im_proj, im_geotrans, dtype=gdal.GDT_Byte):
    """
    入参:
    - img_data (np.ndarray): (H,W) 或 (H,W,C)
    - output_path (str): 输出路径
    - im_proj (str): 投影 WKT
    - im_geotrans (tuple): 仿射变换参数
    - dtype: GDAL 数据类型，默认 GDT_Byte

    方法:
    - 将数组写入 GeoTIFF 并设置地理参考

    出参:
    - 无返回值
    """
    if len(img_data.shape) == 2:
        h, w = img_data.shape
        bands = 1
    else:
        h, w, bands = img_data.shape

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(output_path, w, h, bands, dtype, options=['COMPRESS=LZW'])

    if im_proj:
        ds.SetProjection(im_proj)
    if im_geotrans:
        ds.SetGeoTransform(im_geotrans)

    if bands == 1:
        ds.GetRasterBand(1).WriteArray(img_data)
    else:
        for i in range(bands):
            ds.GetRasterBand(i + 1).WriteArray(img_data[:, :, i])

    ds.FlushCache()
    ds = None


def get_geo_info(raster_path):
    """
    入参:
    - raster_path (str): 栅格文件路径

    方法:
    - 仅读取投影和仿射变换参数，不加载像素数据

    出参:
    - return (tuple): (im_proj, im_geotrans)
    """
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        return None, None
    im_proj = ds.GetProjection()
    im_geotrans = ds.GetGeoTransform()
    ds = None
    return im_proj, im_geotrans


def copy_geo_to_file(src_path, dst_path):
    """
    入参:
    - src_path (str): 源栅格文件（读取地理信息）
    - dst_path (str): 目标栅格文件（写入地理信息）

    方法:
    - 将源文件的投影和仿射变换参数复制到目标文件

    出参:
    - 无返回值
    """
    im_proj, im_geotrans = get_geo_info(src_path)
    if im_proj is None:
        return
    ds = gdal.Open(dst_path, 1)
    if ds is None:
        return
    ds.SetProjection(im_proj)
    ds.SetGeoTransform(im_geotrans)
    ds = None
