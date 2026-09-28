#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → GeoTIFF/PNG 转换器

入参:
- input_csv (str): 输入CSV文件路径（包含longitude, latitude, red, green, blue, nir列）
- output_file (str): 输出文件路径（.tif/.tiff 或 .png）
- format (str): 输出格式，'tiff' 或 'png'（默认根据文件扩展名判断）
- bands (str): 输出波段选择，'rgb' 或 'rgbn'（默认，仅TIFF格式）

方法:
- 读取CSV文件中的坐标和波段数据
- 根据经纬度重建影像的空间结构
- TIFF格式：创建地理变换矩阵，保留坐标系信息
- PNG格式：生成RGB可视化图片

出参:
- GeoTIFF文件（3波段RGB或4波段RGBN）或PNG图片（3波段RGB）
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
import argparse
import os
from collections import defaultdict

# ==================== 全局配置参数 ====================
# 入参配置 - rgb_255_to_reflectance函数
DEFAULT_CLIP_MIN = 0.005               # 原始裁剪下限：反射率最小值
DEFAULT_CLIP_MAX = 0.3                 # 原始裁剪上限：反射率最大值
DEFAULT_GAMMA = 2.2                    # Gamma校正系数

# 入参配置 - csv_to_geotiff函数
DEFAULT_BANDS = 'rgbn'                 # 默认输出波段：'rgb'(3波段) 或 'rgbn'(4波段)
DEFAULT_CRS = 'EPSG:4326'              # 默认坐标系：WGS84
DEFAULT_RESTORE_REFLECTANCE = True     # 是否将0-255转回反射率值
DEFAULT_OUTPUT_FORMAT = 'png'          # 默认输出格式：'tiff' 或 'png'

# CSV列名配置
CSV_COL_LONGITUDE = 'longitude'        # 经度列名
CSV_COL_LATITUDE = 'latitude'          # 纬度列名
CSV_COL_RED = 'red'                    # 红波段列名
CSV_COL_GREEN = 'green'                # 绿波段列名
CSV_COL_BLUE = 'blue'                  # 蓝波段列名
CSV_COL_NIR = 'nir'                    # 近红外波段列名

# 反射率转换参数
REFLECTANCE_SCALE = 10000.0            # 反射率缩放因子：转换到0-10000范围
RGB_MAX_VALUE = 255                    # RGB最大值
REFLECTANCE_MIN_VALUE = 0              # 反射率最小值
REFLECTANCE_MAX_VALUE = 10000          # 反射率最大值

# 输出配置
OUTPUT_DTYPE_REFLECTANCE = np.uint16   # 反射率模式数据类型
OUTPUT_DTYPE_RGB = np.uint8            # RGB模式数据类型
COMPRESSION_TYPE = 'lzw'               # GeoTIFF压缩类型

# 默认分辨率
DEFAULT_RESOLUTION = 0.0001            # 默认像素分辨率（当只有1个像素时）

# 波段数量
BAND_COUNT_RGB = 3                     # RGB波段数量
BAND_COUNT_RGBN = 4                    # RGBN波段数量

# 波段名称
BAND_NAMES_RGB = ['red', 'green', 'blue']
BAND_NAMES_RGBN = ['red', 'green', 'blue', 'nir']

# 波段描述
BAND_DESC_RED = 'Red (B4)'
BAND_DESC_GREEN = 'Green (B3)'
BAND_DESC_BLUE = 'Blue (B2)'
BAND_DESC_NIR = 'NIR (B8)'

# 进度显示配置
PROGRESS_INTERVAL = 10000              # 每处理多少像素显示一次进度

# 出参说明
# rgb_255_to_reflectance 返回: numpy.ndarray (uint16) - 反射率值(0-10000)
# csv_to_geotiff 返回: 无（直接保存GeoTIFF文件）
# ====================================================

def rgb_255_to_reflectance(rgb_array, clip_min=None, clip_max=None, gamma=None):
    """
    将RGB 0-255值反向转换为反射率值（0-10000范围）
    
    入参:
    - rgb_array (np.ndarray): RGB值数组（0-255，uint8）
    - clip_min (float): 原始裁剪下限，None时使用全局DEFAULT_CLIP_MIN
    - clip_max (float): 原始裁剪上限，None时使用全局DEFAULT_CLIP_MAX
    - gamma (float): Gamma校正系数，None时使用全局DEFAULT_GAMMA
    
    方法:
    - 将0-255缩放回0-1
    - 逆Gamma校正（应用gamma次方）
    - 逆Min-Max拉伸
    - 恢复到裁剪前的反射率范围
    - 乘以10000转回原始整数值
    
    出参:
    - np.ndarray: 反射率值（0-10000范围，uint16类型）
    """
    # 使用全局默认值
    if clip_min is None:
        clip_min = DEFAULT_CLIP_MIN
    if clip_max is None:
        clip_max = DEFAULT_CLIP_MAX
    if gamma is None:
        gamma = DEFAULT_GAMMA
    
    # 0-255 → 0-1
    normalized = rgb_array.astype(np.float32) / RGB_MAX_VALUE
    
    # 逆Gamma校正
    gamma_inverse = np.power(normalized, gamma)
    
    # 逆Min-Max拉伸
    reflectance = gamma_inverse * (clip_max - clip_min) + clip_min
    
    # 转换为0-10000范围
    reflectance_10000 = reflectance * REFLECTANCE_SCALE
    
    # 裁剪到合理范围并转为uint16
    reflectance_10000 = np.clip(reflectance_10000, REFLECTANCE_MIN_VALUE, REFLECTANCE_MAX_VALUE).astype(OUTPUT_DTYPE_REFLECTANCE)
    
    return reflectance_10000


def reflectance_to_rgb_255(reflectance_array, clip_min=None, clip_max=None, gamma=None):
    """
    将反射率值（0-10000范围）转换为RGB 0-255值用于可视化
    
    入参:
    - reflectance_array (np.ndarray): 反射率值数组（0-10000，uint16或float）
    - clip_min (float): 裁剪下限，None时使用全局DEFAULT_CLIP_MIN
    - clip_max (float): 裁剪上限，None时使用全局DEFAULT_CLIP_MAX
    - gamma (float): Gamma校正系数，None时使用全局DEFAULT_GAMMA
    
    方法:
    - 将0-10000范围转为0-1
    - 应用裁剪范围
    - Min-Max拉伸到0-1
    - Gamma校正
    - 缩放到0-255
    
    出参:
    - np.ndarray: RGB值（0-255范围，uint8类型）
    """
    # 使用全局默认值
    if clip_min is None:
        clip_min = DEFAULT_CLIP_MIN
    if clip_max is None:
        clip_max = DEFAULT_CLIP_MAX
    if gamma is None:
        gamma = DEFAULT_GAMMA
    
    # 0-10000 → 0-1
    reflectance = reflectance_array.astype(np.float32) / REFLECTANCE_SCALE
    
    # 裁剪到指定范围
    reflectance = np.clip(reflectance, clip_min, clip_max)
    
    # Min-Max拉伸到0-1
    normalized = (reflectance - clip_min) / (clip_max - clip_min)
    
    # Gamma校正
    gamma_corrected = np.power(normalized, 1.0 / gamma)
    
    # 缩放到0-255
    rgb_255 = (gamma_corrected * RGB_MAX_VALUE).astype(OUTPUT_DTYPE_RGB)
    
    return rgb_255


def csv_to_geotiff(input_csv, output_tiff, bands=None, crs=None, 
                   restore_reflectance=None, clip_min=None, clip_max=None, gamma=None):
    """
    将CSV文件转换为GeoTIFF影像
    
    入参:
    - input_csv (str): 输入CSV文件路径
    - output_tiff (str): 输出GeoTIFF文件路径
    - bands (str): 输出波段，None时使用全局DEFAULT_BANDS
    - crs (str): 坐标系，None时使用全局DEFAULT_CRS
    - restore_reflectance (bool): 是否将0-255转回反射率值，None时使用全局DEFAULT_RESTORE_REFLECTANCE
    - clip_min (float): 原始裁剪下限，None时使用全局DEFAULT_CLIP_MIN
    - clip_max (float): 原始裁剪上限，None时使用全局DEFAULT_CLIP_MAX
    - gamma (float): Gamma校正系数，None时使用全局DEFAULT_GAMMA
    
    方法:
    - 读取CSV数据并解析经纬度坐标
    - 根据坐标计算影像尺寸和地理变换
    - 将0-255的RGB值反向转换为反射率值
    - 将像素值填充到对应的空间位置
    - 写入GeoTIFF文件
    
    出参:
    - 无（直接保存GeoTIFF文件）
    """
    # 使用全局默认值
    if bands is None:
        bands = DEFAULT_BANDS
    if crs is None:
        crs = DEFAULT_CRS
    if restore_reflectance is None:
        restore_reflectance = DEFAULT_RESTORE_REFLECTANCE
    if clip_min is None:
        clip_min = DEFAULT_CLIP_MIN
    if clip_max is None:
        clip_max = DEFAULT_CLIP_MAX
    if gamma is None:
        gamma = DEFAULT_GAMMA
    
    print(f"正在读取CSV文件: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 验证必需的列
    required_cols = [CSV_COL_LONGITUDE, CSV_COL_LATITUDE, CSV_COL_RED, CSV_COL_GREEN, CSV_COL_BLUE]
    if bands == DEFAULT_BANDS:
        required_cols.append(CSV_COL_NIR)
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必需的列: {col}")
    
    print(f"CSV数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 获取唯一的经纬度值并排序
    unique_lons = sorted(df[CSV_COL_LONGITUDE].unique())
    unique_lats = sorted(df[CSV_COL_LATITUDE].unique(), reverse=True)  # 纬度从大到小
    
    print(f"\n影像信息:")
    print(f"  宽度（列数）: {len(unique_lons)}")
    print(f"  高度（行数）: {len(unique_lats)}")
    print(f"  经度范围: {min(unique_lons):.6f} - {max(unique_lons):.6f}")
    print(f"  纬度范围: {min(unique_lats):.6f} - {max(unique_lats):.6f}")
    
    width = len(unique_lons)
    height = len(unique_lats)
    
    # 计算像素分辨率
    if width > 1:
        lon_res = (max(unique_lons) - min(unique_lons)) / (width - 1)
    else:
        lon_res = DEFAULT_RESOLUTION
    
    if height > 1:
        lat_res = (max(unique_lats) - min(unique_lats)) / (height - 1)
    else:
        lat_res = DEFAULT_RESOLUTION
    
    print(f"  像素分辨率: {lon_res:.6f} (经度) × {lat_res:.6f} (纬度)")
    
    # 创建经纬度到像素坐标的映射
    lon_to_col = {lon: i for i, lon in enumerate(unique_lons)}
    lat_to_row = {lat: i for i, lat in enumerate(unique_lats)}
    
    # 初始化波段数组
    if bands == DEFAULT_BANDS:
        band_count = BAND_COUNT_RGBN
        band_names = BAND_NAMES_RGBN
        print(f"  输出波段: RGBN（{BAND_COUNT_RGBN}波段）")
    else:
        band_count = BAND_COUNT_RGB
        band_names = BAND_NAMES_RGB
        print(f"  输出波段: RGB（{BAND_COUNT_RGB}波段）")
    
    # 根据是否恢复反射率值选择数据类型
    if restore_reflectance:
        dtype = OUTPUT_DTYPE_REFLECTANCE
        print(f"  数据类型: {OUTPUT_DTYPE_REFLECTANCE.__name__} (反射率值 {REFLECTANCE_MIN_VALUE}-{REFLECTANCE_MAX_VALUE})")
        print(f"  输出模式: 反射率模式（用于科学分析）")
    else:
        dtype = OUTPUT_DTYPE_RGB
        print(f"  数据类型: {OUTPUT_DTYPE_RGB.__name__} (RGB值 0-{RGB_MAX_VALUE})")
        print(f"  输出模式: RGB模式（用于可视化）")
    
    # 创建空数组（初始化为0）
    image_data = np.zeros((band_count, height, width), dtype=dtype)
    
    print(f"\n正在填充像素数据...")
    
    # 填充数据
    filled_pixels = 0
    
    # 如果需要恢复反射率，先收集所有RGB数据进行批量转换
    if restore_reflectance:
        # 临时存储RGB值
        temp_data = np.zeros((band_count, height, width), dtype=OUTPUT_DTYPE_RGB)
        
        for idx, row in df.iterrows():
            lon = row[CSV_COL_LONGITUDE]
            lat = row[CSV_COL_LATITUDE]
            
            col = lon_to_col.get(lon)
            row_idx = lat_to_row.get(lat)
            
            if col is not None and row_idx is not None:
                for band_idx, band_name in enumerate(band_names):
                    temp_data[band_idx, row_idx, col] = int(row[band_name])
                filled_pixels += 1
            
            if (idx + 1) % PROGRESS_INTERVAL == 0:
                print(f"  已处理 {idx + 1} / {len(df)} 像素")
        
        print(f"  成功填充 {filled_pixels} 个像素")
        print(f"\n正在将RGB(0-255)转换回反射率值(0-10000)...")
        
        # 批量转换所有波段
        for band_idx in range(band_count):
            image_data[band_idx] = rgb_255_to_reflectance(temp_data[band_idx], clip_min, clip_max, gamma)
            print(f"  已转换波段 {band_idx+1}/{band_count}")
        
    else:
        # 直接填充，不转换
        for idx, row in df.iterrows():
            lon = row[CSV_COL_LONGITUDE]
            lat = row[CSV_COL_LATITUDE]
            
            col = lon_to_col.get(lon)
            row_idx = lat_to_row.get(lat)
            
            if col is not None and row_idx is not None:
                for band_idx, band_name in enumerate(band_names):
                    image_data[band_idx, row_idx, col] = int(row[band_name])
                filled_pixels += 1
            
            if (idx + 1) % PROGRESS_INTERVAL == 0:
                print(f"  已处理 {idx + 1} / {len(df)} 像素")
        
        print(f"  成功填充 {filled_pixels} 个像素")
    
    # 计算地理变换（Affine变换矩阵）
    # 从边界计算，像素中心对齐
    left = min(unique_lons) - lon_res / 2
    top = max(unique_lats) + lat_res / 2
    right = max(unique_lons) + lon_res / 2
    bottom = min(unique_lats) - lat_res / 2
    
    transform = from_bounds(left, bottom, right, top, width, height)
    
    print(f"\n地理变换信息:")
    print(f"  左边界: {left:.6f}")
    print(f"  右边界: {right:.6f}")
    print(f"  上边界: {top:.6f}")
    print(f"  下边界: {bottom:.6f}")
    print(f"  变换矩阵: {transform}")
    
    # 写入GeoTIFF文件
    print(f"\n正在写入GeoTIFF文件: {output_tiff}")
    
    with rasterio.open(
        output_tiff,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=band_count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress=COMPRESSION_TYPE
    ) as dst:
        # 写入所有波段
        for band_idx in range(band_count):
            dst.write(image_data[band_idx], band_idx + 1)
            
        # 设置波段描述
        if bands == DEFAULT_BANDS:
            dst.set_band_description(1, BAND_DESC_RED)
            dst.set_band_description(2, BAND_DESC_GREEN)
            dst.set_band_description(3, BAND_DESC_BLUE)
            dst.set_band_description(4, BAND_DESC_NIR)
        else:
            dst.set_band_description(1, BAND_DESC_RED)
            dst.set_band_description(2, BAND_DESC_GREEN)
            dst.set_band_description(3, BAND_DESC_BLUE)
    
    print(f"✅ GeoTIFF文件创建成功!")
    print(f"   文件路径: {output_tiff}")
    print(f"   影像尺寸: {width} × {height}")
    print(f"   波段数量: {band_count}")
    print(f"   坐标系: {crs}")
    
    # 显示数据统计
    print(f"\n波段数值统计:")
    for band_idx, band_name in enumerate(band_names):
        band_data = image_data[band_idx]
        non_zero = band_data[band_data > 0]
        if len(non_zero) > 0:
            print(f"  {band_name.upper()}: 最小={non_zero.min()}, 最大={non_zero.max()}, 平均={non_zero.mean():.1f}")
        else:
            print(f"  {band_name.upper()}: 无数据")


def csv_to_png(input_csv, output_png, clip_min=None, clip_max=None, gamma=None):
    """
    将CSV文件转换为PNG可视化图片
    
    入参:
    - input_csv (str): 输入CSV文件路径
    - output_png (str): 输出PNG文件路径
    - clip_min (float): 反射率裁剪下限，None时使用全局DEFAULT_CLIP_MIN
    - clip_max (float): 反射率裁剪上限，None时使用全局DEFAULT_CLIP_MAX
    - gamma (float): Gamma校正系数，None时使用全局DEFAULT_GAMMA
    
    方法:
    - 读取CSV数据并解析经纬度坐标
    - 根据坐标计算影像尺寸
    - 将反射率值（0-10000或CSV中的0-255）转换为RGB可视化（0-255）
    - 将像素值填充到对应的空间位置
    - 保存为PNG图片
    
    出参:
    - 无（直接保存PNG文件）
    """
    # 使用全局默认值
    if clip_min is None:
        clip_min = DEFAULT_CLIP_MIN
    if clip_max is None:
        clip_max = DEFAULT_CLIP_MAX
    if gamma is None:
        gamma = DEFAULT_GAMMA
    
    print(f"正在读取CSV文件: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 验证必需的列
    required_cols = [CSV_COL_LONGITUDE, CSV_COL_LATITUDE, CSV_COL_RED, CSV_COL_GREEN, CSV_COL_BLUE]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必需的列: {col}")
    
    print(f"CSV数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 获取唯一的经纬度值并排序
    unique_lons = sorted(df[CSV_COL_LONGITUDE].unique())
    unique_lats = sorted(df[CSV_COL_LATITUDE].unique(), reverse=True)  # 纬度从大到小
    
    print(f"\n影像信息:")
    print(f"  宽度（列数）: {len(unique_lons)}")
    print(f"  高度（行数）: {len(unique_lats)}")
    print(f"  经度范围: {min(unique_lons):.6f} - {max(unique_lons):.6f}")
    print(f"  纬度范围: {min(unique_lats):.6f} - {max(unique_lats):.6f}")
    
    width = len(unique_lons)
    height = len(unique_lats)
    
    # 创建经纬度到像素坐标的映射
    lon_to_col = {lon: i for i, lon in enumerate(unique_lons)}
    lat_to_row = {lat: i for i, lat in enumerate(unique_lats)}
    
    # 初始化RGB数组（只输出RGB 3波段用于PNG）
    print(f"  输出格式: PNG (RGB 3波段)")
    print(f"  数据类型: uint8 (RGB值 0-255)")
    
    # 创建空数组（初始化为0）
    image_data = np.zeros((BAND_COUNT_RGB, height, width), dtype=OUTPUT_DTYPE_RGB)
    
    print(f"\n正在填充像素数据...")
    
    # 填充数据
    filled_pixels = 0
    band_names = BAND_NAMES_RGB
    
    for idx, row in df.iterrows():
        lon = row[CSV_COL_LONGITUDE]
        lat = row[CSV_COL_LATITUDE]
        
        col = lon_to_col.get(lon)
        row_idx = lat_to_row.get(lat)
        
        if col is not None and row_idx is not None:
            for band_idx, band_name in enumerate(band_names):
                image_data[band_idx, row_idx, col] = int(row[band_name])
            filled_pixels += 1
        
        if (idx + 1) % PROGRESS_INTERVAL == 0:
            print(f"  已处理 {idx + 1} / {len(df)} 像素")
    
    print(f"  成功填充 {filled_pixels} 个像素")
    
    # 转换为PIL Image格式（Height, Width, Channels）
    # numpy数组格式：(Channels, Height, Width) → (Height, Width, Channels)
    image_array = np.transpose(image_data, (1, 2, 0))
    
    # 创建PIL图像
    print(f"\n正在创建PNG图像...")
    img = Image.fromarray(image_array, mode='RGB')
    
    # 保存PNG文件
    print(f"正在保存PNG文件: {output_png}")
    img.save(output_png, 'PNG')
    
    print(f"✅ PNG文件创建成功!")
    print(f"   文件路径: {output_png}")
    print(f"   影像尺寸: {width} × {height}")
    print(f"   波段数量: 3 (RGB)")
    
    # 显示数据统计
    print(f"\n波段数值统计:")
    for band_idx, band_name in enumerate(band_names):
        band_data = image_data[band_idx]
        non_zero = band_data[band_data > 0]
        if len(non_zero) > 0:
            print(f"  {band_name.upper()}: 最小={non_zero.min()}, 最大={non_zero.max()}, 平均={non_zero.mean():.1f}")
        else:
            print(f"  {band_name.upper()}: 无数据")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将CSV文件转换为GeoTIFF影像或PNG图片')
    parser.add_argument('--input', '-i', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径（.tif/.tiff 或 .png）')
    parser.add_argument('--format', '-f', choices=['tiff', 'png'], 
                       help='输出格式: tiff 或 png（默认根据输出文件扩展名自动判断）')
    parser.add_argument('--bands', '-b', choices=['rgb', 'rgbn'], default='rgbn',
                       help='输出波段: rgb(3波段) 或 rgbn(4波段，默认) - 仅用于TIFF格式')
    parser.add_argument('--crs', default='EPSG:4326',
                       help='坐标参考系统 (默认: EPSG:4326) - 仅用于TIFF格式')
    parser.add_argument('--no-restore', action='store_true',
                       help='不恢复反射率，保持0-255的RGB值 - 仅用于TIFF格式')
    parser.add_argument('--clip-min', type=float, default=0.005,
                       help='反射率裁剪下限 (默认: 0.005)')
    parser.add_argument('--clip-max', type=float, default=0.3,
                       help='反射率裁剪上限 (默认: 0.3)')
    parser.add_argument('--gamma', type=float, default=2.2,
                       help='Gamma校正系数 (默认: 2.2)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    # 根据输出文件扩展名或--format参数确定输出格式
    output_ext = os.path.splitext(args.output)[1].lower()
    if args.format:
        output_format = args.format
    elif output_ext in ['.png']:
        output_format = 'png'
    elif output_ext in ['.tif', '.tiff']:
        output_format = 'tiff'
    else:
        print(f"❌ 错误: 无法识别输出文件格式，请使用 .tif/.tiff 或 .png 扩展名，或指定 --format 参数")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        if output_format == 'png':
            print(f"\n=== 转换为PNG格式 ===\n")
            csv_to_png(args.input, args.output,
                      clip_min=args.clip_min,
                      clip_max=args.clip_max,
                      gamma=args.gamma)
        else:  # tiff
            print(f"\n=== 转换为GeoTIFF格式 ===\n")
            csv_to_geotiff(args.input, args.output, args.bands, args.crs,
                          restore_reflectance=not args.no_restore,
                          clip_min=args.clip_min,
                          clip_max=args.clip_max,
                          gamma=args.gamma)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

