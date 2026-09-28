#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIFF 图像尺寸调整器 - 调整为500x500像素

入参:
- input_path (str): 输入TIFF文件路径
- output_path (str): 输出TIFF文件路径

方法:
- 读取TIFF文件及其地理参考信息
- 计算缩放比例和新的地理变换矩阵
- 使用双线性插值重采样到500x500像素
- 保存输出文件并保持地理参考信息

出参:
- 无（直接保存TIFF文件到指定路径）
"""

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
import numpy as np
import os

# ==================== 全局配置参数 ====================
# 输入文件列表配置
INPUT_FILES = [
    r"E:\1代码\模型\gee\栅格与表互相转换\原始数据\image.tif",
    r"E:\1代码\模型\gee\栅格与表互相转换\原始数据\mask.tif"
]

# 输出目录配置
OUTPUT_DIR = r"E:\1代码\模型\gee\栅格与表互相转换\处理结果"  # 输出到原始数据目录

# 目标尺寸配置
TARGET_WIDTH = 500                    # 目标宽度：500像素
TARGET_HEIGHT = 500                   # 目标高度：500像素

# 重采样方法配置
RESAMPLING_METHOD = Resampling.bilinear  # 双线性插值：平衡质量和速度

# GeoTIFF配置
COMPRESSION_TYPE = 'lzw'              # GeoTIFF压缩类型：LZW无损压缩

# 输出文件后缀
OUTPUT_SUFFIX = '_resized_500x500.tif'  # 输出文件后缀

# 出参说明
# resize_tiff_to_500x500 返回: 无（直接保存文件）
# ====================================================


def resize_tiff_to_500x500(input_path, output_path=None):
    """
    将TIFF图像调整为500x500像素（保持地理参考信息）
    
    入参:
    - input_path (str): 输入TIFF文件路径
    - output_path (str): 输出TIFF文件路径，None时自动生成（在输入文件同目录）
    
    方法:
    ① 使用rasterio读取原始TIFF文件和元数据（包括nodata值）
    ② 计算原始尺寸到目标尺寸(500x500)的缩放比例
    ③ 计算新的地理变换矩阵（调整像素分辨率）
    ④ 使用双线性插值重采样所有波段到目标尺寸，自动过滤nodata值
    ⑤ 保存输出文件并保留所有地理参考信息（坐标系、变换矩阵、nodata值等）
    
    出参:
    - 无（直接保存TIFF文件，并打印处理信息）
    """
    # ① 读取输入文件
    print(f"正在读取文件: {input_path}")
    
    with rasterio.open(input_path) as src:
        # 获取原始影像信息
        original_width = src.width
        original_height = src.height
        band_count = src.count
        nodata_value = src.nodata  # 读取nodata值
        
        print(f"\n原始影像信息:")
        print(f"  尺寸: {original_width} x {original_height}")
        print(f"  波段数量: {band_count}")
        print(f"  数据类型: {src.dtypes[0]}")
        print(f"  坐标系: {src.crs}")
        print(f"  NoData值: {nodata_value if nodata_value is not None else '未设置'}")
        
        # ② 计算缩放比例
        scale_x = TARGET_WIDTH / original_width
        scale_y = TARGET_HEIGHT / original_height
        print(f"\n缩放比例:")
        print(f"  X方向: {scale_x:.4f} ({original_width} -> {TARGET_WIDTH})")
        print(f"  Y方向: {scale_y:.4f} ({original_height} -> {TARGET_HEIGHT})")
        
        # ③ 计算新的地理变换矩阵
        # 原变换矩阵: [a, b, c, d, e, f] 表示 [x_pixel_size, rotation, x_offset, rotation, -y_pixel_size, y_offset]
        # 新变换矩阵需要调整像素分辨率（除以缩放比例）
        original_transform = src.transform
        new_transform = Affine(
            original_transform.a / scale_x,  # 调整X方向像素大小
            original_transform.b,            # 保持旋转
            original_transform.c,            # 保持X方向偏移
            original_transform.d,            # 保持旋转
            original_transform.e / scale_y,  # 调整Y方向像素大小
            original_transform.f             # 保持Y方向偏移
        )
        
        print(f"\n地理变换矩阵:")
        print(f"  原始: {original_transform}")
        print(f"  新的: {new_transform}")
        
        # ④ 创建输出数组并重采样所有波段
        print(f"\n正在重采样到 {TARGET_WIDTH}x{TARGET_HEIGHT} 像素...")
        
        # 如果有nodata值，提示正在过滤
        if nodata_value is not None:
            print(f"  ✓ 检测到NoData值 ({nodata_value})，重采样时将自动过滤")
            print(f"  ✓ NoData区域不会参与插值计算")
        
        # 创建目标数组（形状：波段数 x 高度 x 宽度）
        output_data = np.zeros((band_count, TARGET_HEIGHT, TARGET_WIDTH), dtype=src.dtypes[0])
        
        # 对每个波段进行重采样
        for band_idx in range(1, band_count + 1):
            print(f"  处理波段 {band_idx}/{band_count}...")
            
            # 使用rasterio的read方法直接重采样
            # masked=True: 自动将nodata值mask掉，不参与插值计算
            resampled_data = src.read(
                band_idx,
                out_shape=(TARGET_HEIGHT, TARGET_WIDTH),
                resampling=RESAMPLING_METHOD,
                masked=True  # 关键参数：mask掉nodata值
            )
            
            # 如果返回的是masked array，需要填充nodata值
            if np.ma.is_masked(resampled_data):
                # 将masked区域填充为nodata值
                if nodata_value is not None:
                    output_data[band_idx - 1] = resampled_data.filled(nodata_value)
                else:
                    # 如果原始文件没有nodata，使用0填充
                    output_data[band_idx - 1] = resampled_data.filled(0)
            else:
                output_data[band_idx - 1] = resampled_data
        
        # ⑤ 保存输出文件
        # 如果未指定输出路径，则使用全局配置的输出目录
        if output_path is None:
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(OUTPUT_DIR, f"{input_name}{OUTPUT_SUFFIX}")
        
        print(f"\n正在保存到: {output_path}")
        
        # 更新元数据
        output_meta = src.meta.copy()
        output_meta.update({
            'width': TARGET_WIDTH,
            'height': TARGET_HEIGHT,
            'transform': new_transform,
            'compress': COMPRESSION_TYPE,
            'nodata': nodata_value  # 保留原始nodata设置
        })
        
        # 打印nodata处理信息
        if nodata_value is not None:
            print(f"  ✓ 输出文件已设置NoData值: {nodata_value}")
        
        # 写入输出文件
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            dst.write(output_data)
            
            # 复制波段描述信息
            for band_idx in range(1, band_count + 1):
                if src.descriptions[band_idx - 1]:
                    dst.set_band_description(band_idx, src.descriptions[band_idx - 1])
        
        print(f"\n✅ 处理完成!")
        print(f"输出文件: {output_path}")
        print(f"输出尺寸: {TARGET_WIDTH} x {TARGET_HEIGHT}")
        print(f"数据类型: {output_meta['dtype']}")


def main():
    """
    主函数：批量处理多个图像文件
    
    入参:
    - 无（使用全局变量 INPUT_FILES 中定义的文件列表）
    
    方法:
    ① 获取全局变量中的文件列表
    ② 遍历文件列表，对每个文件进行检查和处理
    ③ 跳过不存在的文件，并记录错误信息
    ④ 统计处理成功和失败的文件数量
    
    出参:
    - 无（打印处理结果统计）
    """
    print("=" * 60)
    print("批量图像尺寸调整工具 - 调整为500x500像素")
    print("=" * 60)
    print(f"\n待处理文件数量: {len(INPUT_FILES)}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✓ 已创建输出目录: {OUTPUT_DIR}")
    
    print("-" * 60)
    
    # 统计变量
    success_count = 0  # 成功处理的文件数
    failed_count = 0   # 失败的文件数
    
    # ① 遍历全局配置中的文件列表
    for idx, input_file in enumerate(INPUT_FILES, start=1):
        print(f"\n[{idx}/{len(INPUT_FILES)}] 开始处理文件:")
        print(f"  路径: {input_file}")
        
        # ② 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"  ❌ 错误: 文件不存在，跳过处理")
            failed_count += 1
            continue
        
        # ③ 执行调整大小操作（输出路径为None时自动生成）
        try:
            resize_tiff_to_500x500(input_file, output_path=None)
            success_count += 1
            print(f"  ✅ 文件处理成功")
        except Exception as e:
            print(f"  ❌ 处理失败: {str(e)}")
            failed_count += 1
    
    # ④ 输出处理结果统计
    print("\n" + "=" * 60)
    print("批量处理完成!")
    print("=" * 60)
    print(f"总文件数: {len(INPUT_FILES)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()

