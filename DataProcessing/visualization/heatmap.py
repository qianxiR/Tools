#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灰度通道和水体掩膜可视化工具

入参:
- input_csv (str): 输入CSV文件路径（包含gray和water_mask列）
- output_image (str): 输出图像文件路径

方法:
- 读取CSV文件中的灰度和掩膜数据
- 根据经纬度重建空间结构
- 使用matplotlib绘制热力图
- 保存为图像文件

出参:
- 图像文件（PNG格式，可选输出灰度图和掩膜图）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os
from collections import defaultdict
from PIL import Image

# ==================== 全局配置参数 ====================
# CSV列名配置
CSV_COL_LONGITUDE = 'longitude'        # 经度列名
CSV_COL_LATITUDE = 'latitude'          # 纬度列名
CSV_COL_GRAY = 'gray'                  # 灰度通道列名
CSV_COL_WATER_MASK = 'water_mask'      # 水体掩膜列名

# 可视化配置
DEFAULT_FIGSIZE = (16, 7)              # 图像尺寸（宽x高，英寸）
DEFAULT_DPI = 150                      # 图像分辨率
DEFAULT_CMAP_GRAY = 'rainbow'          # 灰度色图（彩虹色）
DEFAULT_CMAP_MASK = 'binary'           # 掩膜色图（黑白）

# 色图范围
GRAY_VMIN = 0                          # 灰度最小值
GRAY_VMAX = 255                        # 灰度最大值
MASK_VMIN = 0                          # 掩膜最小值
MASK_VMAX = 1                          # 掩膜最大值

# 热力图样式
HEATMAP_INTERPOLATION = 'nearest'      # 插值方法：nearest（无插值），bilinear（双线性）
COLORBAR_SHRINK = 0.8                  # 色条收缩比例
COLORBAR_ASPECT = 20                   # 色条宽高比

# 标题和标签字体大小
TITLE_FONTSIZE = 16                    # 标题字体大小
LABEL_FONTSIZE = 12                    # 轴标签字体大小
TICK_FONTSIZE = 10                     # 刻度字体大小

# 输出格式
DEFAULT_OUTPUT_FORMAT = 'png'          # 输出图像格式
DEFAULT_OUTPUT_BBOX = 'tight'          # 边界框裁剪

# 出参说明
# csv_to_heatmap 返回: 无（直接保存图像文件）
# ====================================================


def csv_to_heatmap(input_csv, output_image, figsize=None, dpi=None, 
                   cmap_gray=None, cmap_mask=None,
                   gray_image_path=None, mask_image_path=None,
                   rgb_image_path=None, rgb_mask_image_path=None):
    """
    从CSV文件读取数据并生成热力图
    
    入参:
    - input_csv (str): 输入CSV文件路径
    - output_image (str): 输出图像文件路径
    - figsize (tuple): 图像尺寸，None时使用全局DEFAULT_FIGSIZE
    - dpi (int): 图像分辨率，None时使用全局DEFAULT_DPI
    - cmap_gray (str): 灰度色图，None时使用全局DEFAULT_CMAP_GRAY
    - cmap_mask (str): 掩膜色图，None时使用全局DEFAULT_CMAP_MASK
    - gray_image_path (str): 灰度通道单独保存路径，None时不另存
    - mask_image_path (str): 水体掩膜单独保存路径，None时不另存
    - rgb_image_path (str): RGB三通道图像保存路径，None时不另存
    - rgb_mask_image_path (str): RGB对应的掩膜图像保存路径，None时不另存
    
    方法:
    ① 读取CSV文件并验证必需的列
    ② 提取经纬度和灰度/掩膜数据
    ③ 根据经纬度坐标重建2D矩阵
    ④ 使用matplotlib绘制并排的两个热力图
    ⑤ 添加色条、标题和标签
    ⑥ 保存为图像文件
    ⑦ 可选：保存RGB和掩膜图像
    
    出参:
    - 无（直接保存图像文件）
    """
    # 使用全局默认值
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    if dpi is None:
        dpi = DEFAULT_DPI
    if cmap_gray is None:
        cmap_gray = DEFAULT_CMAP_GRAY
    if cmap_mask is None:
        cmap_mask = DEFAULT_CMAP_MASK
    
    print(f"正在读取CSV文件: {input_csv}")
    
    # ① 读取CSV文件
    df = pd.read_csv(input_csv)
    
    print(f"CSV数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # ② 验证必需的列
    required_cols = [CSV_COL_LONGITUDE, CSV_COL_LATITUDE, CSV_COL_GRAY, CSV_COL_WATER_MASK]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必需的列: {', '.join(missing_cols)}")
    
    # 检查是否需要生成RGB图像（检查是否存在RGB波段列）
    has_rgb = all(col in df.columns for col in ['red', 'green', 'blue'])
    if rgb_image_path and not has_rgb:
        print(f"⚠️  警告: CSV文件缺少RGB波段列(red/green/blue)，跳过RGB图像生成")
        rgb_image_path = None
        rgb_mask_image_path = None
    
    # ③ 获取唯一的经纬度值并排序
    unique_lons = sorted(df[CSV_COL_LONGITUDE].unique())
    unique_lats = sorted(df[CSV_COL_LATITUDE].unique(), reverse=True)  # 纬度从大到小
    
    print(f"\n空间信息:")
    print(f"  宽度（列数）: {len(unique_lons)}")
    print(f"  高度（行数）: {len(unique_lats)}")
    print(f"  经度范围: {min(unique_lons):.6f} - {max(unique_lons):.6f}")
    print(f"  纬度范围: {min(unique_lats):.6f} - {max(unique_lats):.6f}")
    
    width = len(unique_lons)
    height = len(unique_lats)
    
    # ④ 创建经纬度到像素坐标的映射
    lon_to_col = {lon: i for i, lon in enumerate(unique_lons)}
    lat_to_row = {lat: i for i, lat in enumerate(unique_lats)}
    
    # ⑤ 初始化2D数组（使用NaN填充，用于标识缺失数据）
    gray_matrix = np.full((height, width), np.nan, dtype=np.float32)
    mask_matrix = np.full((height, width), np.nan, dtype=np.float32)
    
    # 如果需要生成RGB图像，初始化RGB矩阵
    if rgb_image_path and has_rgb:
        rgb_matrix = np.full((height, width, 3), np.nan, dtype=np.float32)
    
    print(f"\n正在重建空间结构...")
    
    # ⑥ 填充数据到2D矩阵
    for idx, row in df.iterrows():
        lon = row[CSV_COL_LONGITUDE]
        lat = row[CSV_COL_LATITUDE]
        
        col = lon_to_col.get(lon)
        row_idx = lat_to_row.get(lat)
        
        if col is not None and row_idx is not None:
            gray_matrix[row_idx, col] = row[CSV_COL_GRAY]
            mask_matrix[row_idx, col] = row[CSV_COL_WATER_MASK]
            
            # 如果需要生成RGB图像，同时填充RGB数据
            if rgb_image_path and has_rgb:
                rgb_matrix[row_idx, col, 0] = row['red']
                rgb_matrix[row_idx, col, 1] = row['green']
                rgb_matrix[row_idx, col, 2] = row['blue']
        
        if (idx + 1) % 10000 == 0:
            print(f"  已处理 {idx + 1} / {len(df)} 像素")
    
    print(f"✓ 空间结构重建完成")
    
    # ⑦ 统计信息
    gray_valid = gray_matrix[~np.isnan(gray_matrix)]
    mask_valid = mask_matrix[~np.isnan(mask_matrix)]
    
    print(f"\n数据统计:")
    print(f"  灰度通道:")
    print(f"    有效像素: {len(gray_valid)}")
    print(f"    最小值: {gray_valid.min():.2f}")
    print(f"    最大值: {gray_valid.max():.2f}")
    print(f"    平均值: {gray_valid.mean():.2f}")
    print(f"  水体掩膜:")
    print(f"    有效像素: {len(mask_valid)}")
    print(f"    水体像素: {(mask_valid == 1).sum()} ({(mask_valid == 1).sum()/len(mask_valid)*100:.2f}%)")
    print(f"    非水体像素: {(mask_valid == 0).sum()} ({(mask_valid == 0).sum()/len(mask_valid)*100:.2f}%)")
    
    # ⑧ 创建图形和子图
    print(f"\n正在生成热力图...")

    # ⑧a 可选：单独保存灰度和掩膜图像
    if gray_image_path:
        gray_dir = os.path.dirname(gray_image_path)
        if gray_dir and not os.path.exists(gray_dir):
            os.makedirs(gray_dir, exist_ok=True)
        gray_to_save = np.nan_to_num(gray_matrix, nan=0).astype(np.uint8)
        Image.fromarray(gray_to_save, mode='L').save(gray_image_path)
        print(f"  ✓ 灰度图像已保存: {gray_image_path}")

    if mask_image_path:
        mask_dir = os.path.dirname(mask_image_path)
        if mask_dir and not os.path.exists(mask_dir):
            os.makedirs(mask_dir, exist_ok=True)
        mask_to_save = (np.nan_to_num(mask_matrix, nan=0) * 255).astype(np.uint8)
        Image.fromarray(mask_to_save, mode='L').save(mask_image_path)
        print(f"  ✓ 掩膜图像已保存: {mask_image_path}")
    
    # ⑧b 可选：单独保存RGB三通道图像和对应的掩膜
    if rgb_image_path and has_rgb:
        rgb_dir = os.path.dirname(rgb_image_path)
        if rgb_dir and not os.path.exists(rgb_dir):
            os.makedirs(rgb_dir, exist_ok=True)
        
        # 处理NaN值，用0填充
        rgb_to_save = np.nan_to_num(rgb_matrix, nan=0)
        # 确保值在0-255范围内
        rgb_to_save = np.clip(rgb_to_save, 0, 255).astype(np.uint8)
        
        Image.fromarray(rgb_to_save, mode='RGB').save(rgb_image_path)
        print(f"  ✓ RGB三通道图像已保存: {rgb_image_path}")
        
        # 如果指定了掩膜路径，也保存掩膜图
        if rgb_mask_image_path:
            rgb_mask_dir = os.path.dirname(rgb_mask_image_path)
            if rgb_mask_dir and not os.path.exists(rgb_mask_dir):
                os.makedirs(rgb_mask_dir, exist_ok=True)
            rgb_mask_to_save = (np.nan_to_num(mask_matrix, nan=0) * 255).astype(np.uint8)
            Image.fromarray(rgb_mask_to_save, mode='L').save(rgb_mask_image_path)
            print(f"  ✓ RGB掩膜图像已保存: {rgb_mask_image_path}")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # 子图1：灰度通道热力图
    im1 = axes[0].imshow(gray_matrix, cmap=cmap_gray, 
                         vmin=GRAY_VMIN, vmax=GRAY_VMAX,
                         interpolation=HEATMAP_INTERPOLATION,
                         aspect='auto')
    axes[0].set_title('Gray Channel (Rainbow Colormap)', fontsize=TITLE_FONTSIZE, fontweight='bold')
    axes[0].set_xlabel('Longitude Index', fontsize=LABEL_FONTSIZE)
    axes[0].set_ylabel('Latitude Index', fontsize=LABEL_FONTSIZE)
    axes[0].tick_params(labelsize=TICK_FONTSIZE)
    
    # 添加色条
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=COLORBAR_SHRINK, aspect=COLORBAR_ASPECT)
    cbar1.set_label('Gray Value (0-255)', fontsize=LABEL_FONTSIZE)
    cbar1.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # 子图2：水体掩膜热力图
    im2 = axes[1].imshow(mask_matrix, cmap=cmap_mask,
                         vmin=MASK_VMIN, vmax=MASK_VMAX,
                         interpolation=HEATMAP_INTERPOLATION,
                         aspect='auto')
    axes[1].set_title('Water Mask (Binary)', fontsize=TITLE_FONTSIZE, fontweight='bold')
    axes[1].set_xlabel('Longitude Index', fontsize=LABEL_FONTSIZE)
    axes[1].set_ylabel('Latitude Index', fontsize=LABEL_FONTSIZE)
    axes[1].tick_params(labelsize=TICK_FONTSIZE)
    
    # 添加色条
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=COLORBAR_SHRINK, aspect=COLORBAR_ASPECT)
    cbar2.set_label('Mask Value (0=Non-Water, 1=Water)', fontsize=LABEL_FONTSIZE)
    cbar2.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # ⑨ 添加总标题和统计信息
    water_percentage = (mask_valid == 1).sum() / len(mask_valid) * 100
    fig.suptitle(f'Gray Channel and Water Mask Heatmap\n'
                 f'Image Size: {width}×{height} | Water Coverage: {water_percentage:.2f}%',
                 fontsize=TITLE_FONTSIZE + 2, fontweight='bold', y=0.98)
    
    # ⑩ 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # ⑪ 保存图像
    print(f"\n正在保存图像到: {output_image}")
    plt.savefig(output_image, dpi=dpi, bbox_inches=DEFAULT_OUTPUT_BBOX, 
                format=DEFAULT_OUTPUT_FORMAT)
    plt.close()
    
    print(f"✅ 热力图生成完成!")
    print(f"输出文件: {output_image}")
    print(f"图像尺寸: {width}×{height} 像素")
    print(f"分辨率: {dpi} DPI")


def main():
    """
    主函数 - 解析命令行参数并执行可视化
    
    入参: 命令行参数
    
    方法:
    - 解析输入输出路径
    - 验证文件存在性
    - 调用csv_to_heatmap执行可视化
    
    出参: 无
    """
    parser = argparse.ArgumentParser(
        description='Visualize Gray Channel and Water Mask Heatmap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python step3_visualize_mask_gray.py -i input.csv -o output.png
  
Input CSV Format Requirements:
  Required columns: longitude, latitude, gray, water_mask
  
Output Format:
  PNG image with two side-by-side heatmaps:
  - Left: Gray Channel (Rainbow colormap)
  - Right: Water Mask (Binary black-white)
        """
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, 
                       help='Output image file path (PNG format)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 7],
                       help='Figure size (width height), default: 16 7')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Image resolution (DPI), default: 150')
    
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
        csv_to_heatmap(args.input, args.output, 
                      figsize=tuple(args.figsize), 
                      dpi=args.dpi)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

