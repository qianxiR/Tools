#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoTIFF RGB+近红外波段数据提取器

入参:
- input_path (str): 输入GeoTIFF文件路径
- output_path (str): 输出CSV文件路径
- band_order (list): 波段顺序，默认为['B4','B3','B2','B8']（红绿蓝近红外）

方法:
- 读取GeoTIFF文件的4个波段数据
- 提取每个像素的经纬度坐标
- 将RGB和近红外数据与坐标信息组合
- 保存为CSV表格格式

出参:
- CSV文件包含列：longitude, latitude, red, green, blue, nir
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from PIL import Image
import argparse
import os
from tqdm import tqdm

# ==================== 全局配置参数 ====================
# 入参配置 - convert_to_rgb_255函数
DEFAULT_CLIP_MIN = 0.005               # 反射率下限：去掉极暗区域
DEFAULT_CLIP_MAX = 0.3                 # 反射率上限：去掉极亮区域
DEFAULT_GAMMA = 2.2                    # Gamma校正系数：增强对比度

# 入参配置 - extract_bands_to_csv函数
DEFAULT_BAND_ORDER = ['B4', 'B3', 'B2', 'B8']  # 默认波段顺序：红绿蓝近红外
DEFAULT_CONVERT_RGB = True             # 是否将RGBN转换为0-255范围
DEFAULT_SAVE_CLIPPED_TIFF = True       # 是否保存处理后的原始影像
DEFAULT_STRETCH_255 = True             # 是否在转换后再拉伸到完整0-255范围

# 反射率转换参数
REFLECTANCE_SCALE = 10000.0            # 反射率缩放因子：0-10000范围
RGB_MAX_VALUE = 255                    # RGB最大值

# 数据类型配置
OUTPUT_DTYPE_RGB = np.uint8            # RGB输出数据类型

# 波段索引配置
RED_BAND_INDEX = 0                     # 红波段索引
GREEN_BAND_INDEX = 1                   # 绿波段索引
BLUE_BAND_INDEX = 2                    # 蓝波段索引
NIR_BAND_INDEX = 3                     # 近红外波段索引

# 期望波段数量
EXPECTED_BAND_COUNT = 4                # 期望的波段数量

# 输出文件后缀
CLIPPED_SUFFIX = "_clipped.tif"               # 处理后原始影像文件后缀
RGB_CONVERTED_SUFFIX = "_rgb_converted.tif"   # RGB转换影像文件后缀

# 波段描述
BAND_DESC_RED_RGB = 'Red (0-255)'
BAND_DESC_GREEN_RGB = 'Green (0-255)'
BAND_DESC_BLUE_RGB = 'Blue (0-255)'
BAND_DESC_NIR_RGB = 'NIR (0-255)'

# GeoTIFF配置
COMPRESSION_TYPE = 'lzw'               # GeoTIFF压缩类型

# CSV列名配置
CSV_COL_WATER_MASK = 'water_mask'      # 水体掩膜列名

# 出参说明
# convert_to_rgb_255 返回: numpy.ndarray (uint8) - RGB值(0-255)
# extract_bands_to_csv 返回: 无（直接保存CSV和GeoTIFF文件）
# extract_bands_to_csv_with_mask 返回: 无（直接保存CSV和GeoTIFF文件，包含water_mask列）
# ====================================================

def convert_to_rgb_255(band_array, clip_min=None, clip_max=None, gamma=None, stretch_255=None):
    """
    将反射率转换为RGB 0-255范围（标准遥感影像处理流程）
    
    入参:
    - band_array (np.ndarray): 输入波段数组（反射率值，通常0-10000）
    - clip_min (float): 反射率下限，None时使用全局DEFAULT_CLIP_MIN
    - clip_max (float): 反射率上限，None时使用全局DEFAULT_CLIP_MAX
    - gamma (float): Gamma校正系数，None时使用全局DEFAULT_GAMMA
    - stretch_255 (bool): 是否在转换后再拉伸到完整0-255范围，None时使用全局DEFAULT_STRETCH_255
    
    方法:
    ① 将0-10000范围转换为0-1反射率
    ② 裁剪到有效反射率范围（clip_min ~ clip_max）
    ③ Min-Max拉伸到0-1
    ④ 应用Gamma校正（γ=2.2）增强显示效果
    ⑤ 缩放到0-255并转为uint8
    ⑥ (可选) 按255比例拉伸：将实际范围拉伸到完整0-255
    
    出参:
    - np.ndarray: RGB值（0-255，uint8类型）
    """
    # 使用全局默认值
    if clip_min is None:
        clip_min = DEFAULT_CLIP_MIN
    if clip_max is None:
        clip_max = DEFAULT_CLIP_MAX
    if gamma is None:
        gamma = DEFAULT_GAMMA
    if stretch_255 is None:
        stretch_255 = DEFAULT_STRETCH_255
    
    # ① 转换为0-1反射率（假设输入是0-10000范围）
    reflectance = band_array / REFLECTANCE_SCALE
    
    # ② 裁剪到有效范围（增强层次，去掉极暗与极亮）
    reflectance = np.clip(reflectance, clip_min, clip_max)
    
    # ③ Min-Max拉伸到0-1
    normalized = (reflectance - clip_min) / (clip_max - clip_min)
    
    # ④ Gamma校正（应用1/gamma增强对比度）
    gamma_corrected = np.power(normalized, 1.0 / gamma)
    
    # ⑤ 缩放到0-255
    rgb_255 = (gamma_corrected * RGB_MAX_VALUE).astype(OUTPUT_DTYPE_RGB)
    
    # ⑥ 按255比例拉伸（增强对比度）
    if stretch_255:
        # 计算当前实际范围
        actual_min = rgb_255.min()
        actual_max = rgb_255.max()
        
        # 如果有动态范围，则拉伸到完整0-255
        if actual_max > actual_min:
            # 线性拉伸到0-255
            rgb_255 = ((rgb_255.astype(np.float32) - actual_min) / (actual_max - actual_min) * RGB_MAX_VALUE).astype(OUTPUT_DTYPE_RGB)
    
    return rgb_255


def extract_bands_to_csv(input_path, output_path, band_order=None, 
                         convert_rgb=None, clip_min=None, clip_max=None, gamma=None,
                         save_clipped_tiff=None, stretch_255=None):
    """
    从GeoTIFF文件提取RGB和近红外波段数据并保存为CSV
    
    入参:
    - input_path (str): 输入GeoTIFF文件路径
    - output_path (str): 输出CSV文件路径  
    - band_order (list): 波段顺序，None时使用全局DEFAULT_BAND_ORDER
    - convert_rgb (bool): 是否将RGBN转换为0-255范围，None时使用全局DEFAULT_CONVERT_RGB
    - clip_min (float): 反射率裁剪下限，None时使用全局DEFAULT_CLIP_MIN
    - clip_max (float): 反射率裁剪上限，None时使用全局DEFAULT_CLIP_MAX
    - gamma (float): Gamma校正系数，None时使用全局DEFAULT_GAMMA
    - save_clipped_tiff (bool): 是否保存处理后的原始影像，None时使用全局DEFAULT_SAVE_CLIPPED_TIFF
    - stretch_255 (bool): 是否在转换后再拉伸到完整0-255范围，None时使用全局DEFAULT_STRETCH_255
    
    方法:
    - 使用rasterio读取GeoTIFF文件
    - 提取整个图像的所有像素
    - 保存处理后的原始影像（保持原始反射率值）
    - 提取每个像素的经纬度坐标
    - 将RGB和近红外波段转换为0-255标准范围
    - 应用反射率裁剪和Gamma校正
    - 按255比例拉伸以增强对比度
    
    出参:
    - 无（直接保存CSV文件和处理后的GeoTIFF文件）
    """
    # 使用全局默认值
    if band_order is None:
        band_order = DEFAULT_BAND_ORDER
    if convert_rgb is None:
        convert_rgb = DEFAULT_CONVERT_RGB
    if clip_min is None:
        clip_min = DEFAULT_CLIP_MIN
    if clip_max is None:
        clip_max = DEFAULT_CLIP_MAX
    if gamma is None:
        gamma = DEFAULT_GAMMA
    if save_clipped_tiff is None:
        save_clipped_tiff = DEFAULT_SAVE_CLIPPED_TIFF
    if stretch_255 is None:
        stretch_255 = DEFAULT_STRETCH_255
    
    print(f"正在读取文件: {input_path}")
    
    # 打开GeoTIFF文件
    with rasterio.open(input_path) as src:
        # 读取所有波段数据
        bands_data = src.read()  # 形状为 (bands, height, width)
        height, width = bands_data.shape[1], bands_data.shape[2]
        
        print(f"文件信息:")
        print(f"  波段数量: {src.count}")
        print(f"  影像尺寸: {width} x {height}")
        print(f"  数据类型: {src.dtypes}")
        print(f"  坐标系: {src.crs}")
        print(f"  地理变换: {src.transform}")
        
        # 验证波段数量
        if src.count != EXPECTED_BAND_COUNT:
            raise ValueError(f"期望{EXPECTED_BAND_COUNT}个波段，但文件包含{src.count}个波段")
        
        # 按照指定顺序提取波段
        red_band_raw = bands_data[RED_BAND_INDEX]      # B4 - 红波段
        green_band_raw = bands_data[GREEN_BAND_INDEX]  # B3 - 绿波段  
        blue_band_raw = bands_data[BLUE_BAND_INDEX]    # B2 - 蓝波段
        nir_band = bands_data[NIR_BAND_INDEX]          # B8 - 近红外波段
        
        print(f"波段顺序: {band_order}")
        print(f"  波段1 ({band_order[0]}): 红波段, 原始范围: {red_band_raw.min():.3f} - {red_band_raw.max():.3f}")
        print(f"  波段2 ({band_order[1]}): 绿波段, 原始范围: {green_band_raw.min():.3f} - {green_band_raw.max():.3f}")
        print(f"  波段3 ({band_order[2]}): 蓝波段, 原始范围: {blue_band_raw.min():.3f} - {blue_band_raw.max():.3f}")
        print(f"  波段4 ({band_order[3]}): 近红外波段, 原始范围: {nir_band.min():.3f} - {nir_band.max():.3f}")
        
        print("\n正在提取数据...")
        
        # 创建数据列表
        data_list = []
        
        # 处理整个图像区域
        start_i = 0
        end_i = height
        start_j = 0
        end_j = width
        
        # 使用整个图像数据
        red_band_clipped = red_band_raw
        green_band_clipped = green_band_raw
        blue_band_clipped = blue_band_raw
        nir_band_clipped = nir_band
        
        # 转换RGBN到0-255范围
        if convert_rgb:
            print(f"\n图像的原始数值范围:")
            print(f"  红波段: {red_band_clipped.min():.1f} - {red_band_clipped.max():.1f}")
            print(f"  绿波段: {green_band_clipped.min():.1f} - {green_band_clipped.max():.1f}")
            print(f"  蓝波段: {blue_band_clipped.min():.1f} - {blue_band_clipped.max():.1f}")
            print(f"  近红外: {nir_band_clipped.min():.1f} - {nir_band_clipped.max():.1f}")
            
            print(f"\n正在转换RGBN到0-255范围（标准遥感流程）...")
            print(f"  裁剪范围: {clip_min}-{clip_max}")
            print(f"  Gamma校正: γ={gamma}")
            print(f"  255拉伸: {'启用' if stretch_255 else '禁用'}")
            red_band = convert_to_rgb_255(red_band_clipped, clip_min, clip_max, gamma, stretch_255)
            green_band = convert_to_rgb_255(green_band_clipped, clip_min, clip_max, gamma, stretch_255)
            blue_band = convert_to_rgb_255(blue_band_clipped, clip_min, clip_max, gamma, stretch_255)
            nir_band = convert_to_rgb_255(nir_band_clipped, clip_min, clip_max, gamma, stretch_255)
            print(f"  转换后红波段范围: {red_band.min()} - {red_band.max()}")
            print(f"  转换后绿波段范围: {green_band.min()} - {green_band.max()}")
            print(f"  转换后蓝波段范围: {blue_band.min()} - {blue_band.max()}")
            print(f"  转换后近红外波段范围: {nir_band.min()} - {nir_band.max()}")
        else:
            red_band = red_band_clipped
            green_band = green_band_clipped
            blue_band = blue_band_clipped
            nir_band = nir_band_clipped
        
        print(f"处理区域: 行 {start_i}-{end_i}, 列 {start_j}-{end_j}")
        print(f"实际处理尺寸: {end_i-start_i} x {end_j-start_j}")
        
        # 保存原始影像（保持原始反射率值）
        if save_clipped_tiff:
            # 使用整个图像的原始数据
            clipped_data = bands_data
            
            # 使用原始的地理变换矩阵（无需重新计算）
            clipped_transform = src.transform
            
            # 生成输出文件名（在CSV同目录下）
            output_dir = os.path.dirname(output_path)
            csv_name = os.path.splitext(os.path.basename(output_path))[0]
            clipped_tiff_path = os.path.join(output_dir, f"{csv_name}{CLIPPED_SUFFIX}")
            
            print(f"\n正在保存原始影像: {clipped_tiff_path}")
            
            # 写入GeoTIFF文件
            with rasterio.open(
                clipped_tiff_path,
                'w',
                driver='GTiff',
                height=clipped_data.shape[1],
                width=clipped_data.shape[2],
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=clipped_transform,
                compress=COMPRESSION_TYPE
            ) as dst:
                dst.write(clipped_data)
                # 复制波段描述
                for i in range(1, src.count + 1):
                    if src.descriptions[i-1]:
                        dst.set_band_description(i, src.descriptions[i-1])
            
            print(f"✅ 原始影像已保存（尺寸: {clipped_data.shape[2]}x{clipped_data.shape[1]}, 数据类型: {src.dtypes[0]}）")
        
        # 保存转换后的RGB图像（如果启用了转换）
        if save_clipped_tiff and convert_rgb:
            # 创建转换后的RGB影像数组（4个波段：RGBN）
            rgb_converted_data = np.zeros((EXPECTED_BAND_COUNT, end_i-start_i, end_j-start_j), dtype=OUTPUT_DTYPE_RGB)
            rgb_converted_data[RED_BAND_INDEX] = red_band      # Red
            rgb_converted_data[GREEN_BAND_INDEX] = green_band  # Green
            rgb_converted_data[BLUE_BAND_INDEX] = blue_band    # Blue
            rgb_converted_data[NIR_BAND_INDEX] = nir_band      # NIR
            
            # 生成转换后影像的文件名
            rgb_tiff_path = os.path.join(output_dir, f"{csv_name}{RGB_CONVERTED_SUFFIX}")
            
            print(f"正在保存RGB转换后的影像: {rgb_tiff_path}")
            
            # 写入转换后的GeoTIFF文件
            with rasterio.open(
                rgb_tiff_path,
                'w',
                driver='GTiff',
                height=rgb_converted_data.shape[1],
                width=rgb_converted_data.shape[2],
                count=EXPECTED_BAND_COUNT,
                dtype=OUTPUT_DTYPE_RGB,
                crs=src.crs,
                transform=clipped_transform,
                compress=COMPRESSION_TYPE
            ) as dst:
                dst.write(rgb_converted_data)
                # 设置波段描述
                dst.set_band_description(1, BAND_DESC_RED_RGB)
                dst.set_band_description(2, BAND_DESC_GREEN_RGB)
                dst.set_band_description(3, BAND_DESC_BLUE_RGB)
                dst.set_band_description(4, BAND_DESC_NIR_RGB)
            
            print(f"✅ RGB转换影像已保存（尺寸: {rgb_converted_data.shape[2]}x{rgb_converted_data.shape[1]}, 数据类型: {OUTPUT_DTYPE_RGB.__name__}, 范围: 0-{RGB_MAX_VALUE}）")
        
        # 使用进度条处理数据（优化：向量化处理）
        total_pixels = (end_i - start_i) * (end_j - start_j)
        print(f"\n正在提取CSV数据（{total_pixels:,} 像素）...")
        
        # 创建有效像素的索引
        i_indices, j_indices = np.meshgrid(
            np.arange(end_i - start_i), 
            np.arange(end_j - start_j), 
            indexing='ij'
        )
        
        # 检查有效数据
        valid_mask = (
            ~np.isnan(red_band) & 
            ~np.isnan(green_band) & 
            ~np.isnan(blue_band) & 
            ~np.isnan(nir_band)
        )
        
        # 获取有效像素的索引
        valid_i = i_indices[valid_mask]
        valid_j = j_indices[valid_mask]
        
        print(f"有效像素数量: {len(valid_i):,}")
        
        # 批量转换坐标（优化：向量化）
        print("正在批量转换坐标...")
        
        # 准备批量处理的数据
        batch_size = 50000  # 每批处理50000个像素
        total_batches = (len(valid_i) + batch_size - 1) // batch_size
        
        with tqdm(total=len(valid_i), desc="提取像素") as pbar:
            for batch_num in range(total_batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, len(valid_i))
                
                batch_i = valid_i[batch_start:batch_end].astype(np.int32)
                batch_j = valid_j[batch_start:batch_end].astype(np.int32)
                
                # 批量转换坐标
                batch_lons = []
                batch_lats = []
                for i, j in zip(batch_i, batch_j):
                    lon, lat = xy(src.transform, start_i + i, start_j + j)
                    batch_lons.append(lon)
                    batch_lats.append(lat)
                
                # 批量创建数据记录
                for idx in range(len(batch_lons)):
                    data_list.append({
                        'longitude': batch_lons[idx],
                        'latitude': batch_lats[idx],
                        'red': float(red_band[batch_i[idx], batch_j[idx]]),
                        'green': float(green_band[batch_i[idx], batch_j[idx]]),
                        'blue': float(blue_band[batch_i[idx], batch_j[idx]]),
                        'nir': float(nir_band[batch_i[idx], batch_j[idx]])
                    })
                
                pbar.update(len(batch_i))
                
                # 定期打印进度
                if (batch_num + 1) % 10 == 0:
                    print(f"  已处理 {(batch_num + 1) * batch_size:,} / {len(valid_i):,} 像素")
        
        # 创建DataFrame
        df = pd.DataFrame(data_list)
        
        # 保存为CSV
        print(f"正在保存到: {output_path}")
        df.to_csv(output_path, index=False)
        
        print("✅ 数据提取完成!")
        print(f"CSV文件包含 {len(df)} 行数据")
        print(f"列名: {list(df.columns)}")
        
        # 显示数据统计信息
        print("\n数据统计:")
        print(df.describe())

def extract_bands_to_csv_with_mask(input_path, mask_path, output_path, band_order=None, 
                                    convert_rgb=None, clip_min=None, clip_max=None, gamma=None,
                                    save_clipped_tiff=None, stretch_255=None):
    """
    从GeoTIFF文件提取RGB和近红外波段数据，并从掩膜文件读取water_mask
    
    入参:
    - input_path (str): 输入GeoTIFF文件路径
    - mask_path (str): 输入掩膜文件路径（PNG/TIF格式，二值图像：1=水体，0=非水体）
    - output_path (str): 输出CSV文件路径  
    - band_order (list): 波段顺序，None时使用全局DEFAULT_BAND_ORDER
    - convert_rgb (bool): 是否将RGBN转换为0-255范围，None时使用全局DEFAULT_CONVERT_RGB
    - clip_min (float): 反射率裁剪下限，None时使用全局DEFAULT_CLIP_MIN
    - clip_max (float): 反射率裁剪上限，None时使用全局DEFAULT_CLIP_MAX
    - gamma (float): Gamma校正系数，None时使用全局DEFAULT_GAMMA
    - save_clipped_tiff (bool): 是否保存处理后的原始影像，None时使用全局DEFAULT_SAVE_CLIPPED_TIFF
    - stretch_255 (bool): 是否在转换后再拉伸到完整0-255范围，None时使用全局DEFAULT_STRETCH_255
    
    方法:
    ① 调用extract_bands_to_csv提取影像数据到CSV
    ② 读取掩膜文件（支持PNG和TIF格式）
    ③ 验证掩膜尺寸与影像一致
    ④ 将掩膜数据转换为二值（>0视为水体）
    ⑤ 将掩膜数据追加到CSV的water_mask列
    ⑥ 保存更新后的CSV文件
    
    出参:
    - 无（直接保存CSV文件，包含water_mask列）
    """
    # 使用全局默认值
    if band_order is None:
        band_order = DEFAULT_BAND_ORDER
    if convert_rgb is None:
        convert_rgb = DEFAULT_CONVERT_RGB
    if clip_min is None:
        clip_min = DEFAULT_CLIP_MIN
    if clip_max is None:
        clip_max = DEFAULT_CLIP_MAX
    if gamma is None:
        gamma = DEFAULT_GAMMA
    if save_clipped_tiff is None:
        save_clipped_tiff = DEFAULT_SAVE_CLIPPED_TIFF
    if stretch_255 is None:
        stretch_255 = DEFAULT_STRETCH_255
    
    # ① 先调用原始函数提取影像数据
    print(f"\n【步骤1/2】提取影像波段数据...")
    extract_bands_to_csv(
        input_path=input_path,
        output_path=output_path,
        band_order=band_order,
        convert_rgb=convert_rgb,
        clip_min=clip_min,
        clip_max=clip_max,
        gamma=gamma,
        save_clipped_tiff=save_clipped_tiff,
        stretch_255=stretch_255
    )
    
    # ② 读取掩膜文件
    print(f"\n【步骤2/2】读取掩膜文件并追加到CSV...")
    print(f"掩膜文件: {mask_path}")
    
    # 判断掩膜文件格式
    mask_ext = os.path.splitext(mask_path)[1].lower()
    
    if mask_ext in ['.png', '.jpg', '.jpeg']:
        # 使用PIL读取PNG等图像格式
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)
        
        # 如果是RGB图像，取第一个通道
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
            
    elif mask_ext in ['.tif', '.tiff']:
        # 使用rasterio读取GeoTIFF格式
        with rasterio.open(mask_path) as mask_src:
            mask_array = mask_src.read(1)  # 读取第一个波段
    else:
        raise ValueError(f"不支持的掩膜文件格式: {mask_ext}（支持: .png, .jpg, .tif）")
    
    print(f"掩膜数据形状: {mask_array.shape}")
    print(f"掩膜值范围: {mask_array.min()} - {mask_array.max()}")
    
    # ③ 将掩膜转换为二值（>0视为水体=1，否则=0）
    mask_binary = (mask_array > 0).astype(np.uint8)
    water_pixels = (mask_binary == 1).sum()
    total_pixels = mask_binary.size
    water_percentage = water_pixels / total_pixels * 100
    
    print(f"掩膜统计:")
    print(f"  水体像素: {water_pixels} ({water_percentage:.2f}%)")
    print(f"  非水体像素: {total_pixels - water_pixels} ({100-water_percentage:.2f}%)")
    
    # ④ 读取已生成的CSV文件
    print(f"\n正在读取CSV文件: {output_path}")
    df = pd.read_csv(output_path)
    
    print(f"CSV数据形状: {df.shape}")
    print(f"CSV列名: {list(df.columns)}")
    
    # ⑤ 读取影像文件获取地理变换信息
    with rasterio.open(input_path) as src:
        # 验证掩膜尺寸
        if mask_binary.shape != (src.height, src.width):
            raise ValueError(
                f"掩膜尺寸 {mask_binary.shape} 与影像尺寸 ({src.height}, {src.width}) 不匹配"
            )
        
        # ⑥ 为CSV中的每个像素分配掩膜值
        print(f"\n正在将掩膜数据追加到CSV...")
        mask_values = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="匹配掩膜值"):
            lon = row['longitude']
            lat = row['latitude']
            
            # 将地理坐标转换回像素坐标
            py, px = rasterio.transform.rowcol(src.transform, lon, lat)
            
            # 获取对应的掩膜值
            if 0 <= py < mask_binary.shape[0] and 0 <= px < mask_binary.shape[1]:
                mask_value = mask_binary[py, px]
            else:
                mask_value = 0  # 超出范围默认为非水体
            
            mask_values.append(mask_value)
        
        # ⑦ 添加water_mask列到DataFrame
        df[CSV_COL_WATER_MASK] = mask_values
        
        # ⑧ 保存更新后的CSV
        print(f"\n正在保存更新后的CSV文件: {output_path}")
        df.to_csv(output_path, index=False)
        
        print(f"✅ 掩膜数据已追加!")
        print(f"新增列: {CSV_COL_WATER_MASK}")
        print(f"CSV最终列名: {list(df.columns)}")
        
        # 验证掩膜统计
        csv_water_pixels = (df[CSV_COL_WATER_MASK] == 1).sum()
        csv_water_percentage = csv_water_pixels / len(df) * 100
        print(f"\nCSV中水体统计:")
        print(f"  水体像素: {csv_water_pixels} ({csv_water_percentage:.2f}%)")
        print(f"  非水体像素: {len(df) - csv_water_pixels} ({100-csv_water_percentage:.2f}%)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从GeoTIFF文件提取RGB和近红外波段数据')
    parser.add_argument('--input', '-i', required=True, help='输入GeoTIFF文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出CSV文件路径')
    parser.add_argument('--bands', '-b', nargs=4, default=['B4', 'B3', 'B2', 'B8'],
                       help='波段顺序 (默认: B4 B3 B2 B8)')
    parser.add_argument('--no-convert', action='store_true', 
                       help='不转换RGBN，保留原始反射率值')
    parser.add_argument('--clip-min', type=float, default=0.005,
                       help='反射率裁剪下限 (默认: 0.005)')
    parser.add_argument('--clip-max', type=float, default=0.3,
                       help='反射率裁剪上限 (默认: 0.3)')
    parser.add_argument('--gamma', type=float, default=2.2,
                       help='Gamma校正系数 (默认: 2.2)')
    parser.add_argument('--no-save-tiff', action='store_true',
                       help='不保存处理后的原始GeoTIFF影像')
    parser.add_argument('--no-stretch-255', action='store_true',
                       help='不进行255范围拉伸（默认启用拉伸以增强对比度）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        extract_bands_to_csv(args.input, args.output, args.bands, 
                            convert_rgb=not args.no_convert,
                            clip_min=args.clip_min,
                            clip_max=args.clip_max,
                            gamma=args.gamma,
                            save_clipped_tiff=not args.no_save_tiff,
                            stretch_255=not args.no_stretch_255)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")

if __name__ == "__main__":
    main()
