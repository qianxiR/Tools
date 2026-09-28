#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDWI和NDVI指数计算器

入参:
- input_csv (str): 输入CSV文件路径（包含longitude, latitude, red, green, blue, nir列）
- output_csv (str): 输出CSV文件路径

方法:
- 读取CSV文件中的RGB和NIR波段数据
- 计算NDWI (归一化差异水体指数)
- 计算NDVI (归一化差异植被指数)
- 保存包含原始数据和计算结果的新CSV文件

出参:
- CSV文件包含列：longitude, latitude, red, green, blue, nir, gray, ndwi, ndvi, ndwi_255, ndvi_255, water_mask（最后一列）
"""

import numpy as np
import pandas as pd
import argparse
import os

# ==================== 全局配置参数 ====================
# CSV列名配置
CSV_COL_LONGITUDE = 'longitude'        # 经度列名
CSV_COL_LATITUDE = 'latitude'          # 纬度列名
CSV_COL_RED = 'red'                    # 红波段列名
CSV_COL_GREEN = 'green'                # 绿波段列名
CSV_COL_BLUE = 'blue'                  # 蓝波段列名
CSV_COL_NIR = 'nir'                    # 近红外波段列名
CSV_COL_GRAY = 'gray'                  # 灰度通道列名
CSV_COL_NDWI = 'ndwi'                  # NDWI列名
CSV_COL_NDVI = 'ndvi'                  # NDVI列名
CSV_COL_NDWI_255 = 'ndwi_255'          # NDWI拉伸到0-255列名
CSV_COL_NDVI_255 = 'ndvi_255'          # NDVI拉伸到0-255列名
CSV_COL_WATER_MASK = 'water_mask'      # 水体掩膜列名

# 计算参数
EPSILON = 1e-10                        # 防止除零的极小值
NDWI_MIN = -1.0                        # NDWI最小值
NDWI_MAX = 1.0                         # NDWI最大值
NDVI_MIN = -1.0                        # NDVI最小值
NDVI_MAX = 1.0                         # NDVI最大值

# 拉伸参数
RGB_MAX_VALUE = 255                    # RGB最大值（用于拉伸）
STRETCH_MIN = 0                        # 拉伸后最小值
STRETCH_MAX = 255                      # 拉伸后最大值

# 水体掩膜配置（注意：掩膜现在从step1的CSV中读取，不再计算）
# WATER_NDWI_THRESHOLD = 0.1             # 已废弃：掩膜从外部文件读取
# WATER_NDVI_THRESHOLD = 0.1             # 已废弃：掩膜从外部文件读取

# 灰度转换权重（ITU-R BT.601标准）
GRAY_WEIGHT_RED = 0.299                # 红通道权重
GRAY_WEIGHT_GREEN = 0.587              # 绿通道权重
GRAY_WEIGHT_BLUE = 0.114               # 蓝通道权重

# 输出配置
OUTPUT_DECIMAL_PLACES = 6              # 输出小数位数

# 统计配置
DISPLAY_STATS = True                   # 是否显示统计信息
DISPLAY_HISTOGRAM_BINS = 10            # 直方图分组数量

# 出参说明
# calculate_ndwi 返回: float - NDWI值 (-1.0 到 1.0)
# calculate_ndvi 返回: float - NDVI值 (-1.0 到 1.0)
# stretch_to_255 返回: int - 拉伸到0-255范围的值
# rgb_to_gray 返回: int - 灰度值 (0-255)
# compute_indices 返回: 无（直接保存CSV文件）
# 输出CSV列顺序: longitude, latitude, red, green, blue, nir, gray, ndwi, ndvi, ndwi_255, ndvi_255, water_mask
# 注意：water_mask现在从step1的CSV中直接读取，不再计算，保存时移到最后一列
# ====================================================


def calculate_ndwi(green, nir, epsilon=None):
    """
    计算NDWI (归一化差异水体指数) - McFeeters 1996
    
    入参:
    - green (float/array): 绿波段值（0-255）
    - nir (float/array): 近红外波段值（0-255）
    - epsilon (float): 防除零的极小值，None时使用全局EPSILON
    
    方法:
    ① 使用公式: NDWI = (Green - NIR) / (Green + NIR)
    ② 分母加上极小值epsilon防止除零错误
    ③ 结果范围: -1（无水体）到 +1（纯水体）
    ④ 正值表示水体，负值表示非水体
    
    出参:
    - float/array: NDWI值，范围 -1.0 到 1.0
    """
    # 使用全局默认值
    if epsilon is None:
        epsilon = EPSILON
    
    # 计算NDWI公式: (Green - NIR) / (Green + NIR)
    # 分母加epsilon防止除零
    ndwi = (green - nir) / (green + nir + epsilon)
    
    # 确保结果在合理范围内
    ndwi = np.clip(ndwi, NDWI_MIN, NDWI_MAX)
    
    return ndwi


def calculate_ndvi(red, nir, epsilon=None):
    """
    计算NDVI (归一化差异植被指数) - Rouse 1974
    
    入参:
    - red (float/array): 红波段值（0-255）
    - nir (float/array): 近红外波段值（0-255）
    - epsilon (float): 防除零的极小值，None时使用全局EPSILON
    
    方法:
    ① 使用公式: NDVI = (NIR - Red) / (NIR + Red)
    ② 分母加上极小值epsilon防止除零错误
    ③ 结果范围: -1（无植被）到 +1（茂密植被）
    ④ 正值表示植被，负值表示非植被（水体、裸土等）
    
    出参:
    - float/array: NDVI值，范围 -1.0 到 1.0
    """
    # 使用全局默认值
    if epsilon is None:
        epsilon = EPSILON
    
    # 计算NDVI公式: (NIR - Red) / (NIR + Red)
    # 分母加epsilon防止除零
    ndvi = (nir - red) / (nir + red + epsilon)
    
    # 确保结果在合理范围内
    ndvi = np.clip(ndvi, NDVI_MIN, NDVI_MAX)
    
    return ndvi


def stretch_to_255(index_value, min_val=None, max_val=None):
    """
    将归一化指数（-1到1）拉伸到0-255范围
    
    入参:
    - index_value (float/array): 指数值（-1到1范围）
    - min_val (float): 输入最小值，None时默认为-1.0
    - max_val (float): 输入最大值，None时默认为1.0
    
    方法:
    ① 线性拉伸公式: value_255 = (value - min_val) / (max_val - min_val) * 255
    ② 简化形式（当min=-1, max=1）: value_255 = (value + 1) / 2 * 255
    ③ 裁剪到0-255范围
    ④ 转换为整数类型（uint8）
    
    出参:
    - int/array: 拉伸后的值，范围 0 到 255
    """
    # 使用默认值
    if min_val is None:
        min_val = -1.0
    if max_val is None:
        max_val = 1.0
    
    # ① 线性拉伸到0-255范围
    stretched = (index_value - min_val) / (max_val - min_val) * STRETCH_MAX
    
    # ② 裁剪到合理范围
    stretched = np.clip(stretched, STRETCH_MIN, STRETCH_MAX)
    
    # ③ 转换为整数
    stretched = stretched.astype(np.uint8)
    
    return stretched


def rgb_to_gray(red, green, blue, weight_r=None, weight_g=None, weight_b=None):
    """
    将RGB三通道合并为单通道灰度值
    
    入参:
    - red (float/array): 红波段值（0-255）
    - green (float/array): 绿波段值（0-255）
    - blue (float/array): 蓝波段值（0-255）
    - weight_r (float): 红通道权重，None时使用全局GRAY_WEIGHT_RED
    - weight_g (float): 绿通道权重，None时使用全局GRAY_WEIGHT_GREEN
    - weight_b (float): 蓝通道权重，None时使用全局GRAY_WEIGHT_BLUE
    
    方法:
    ① 使用ITU-R BT.601标准灰度转换公式
    ② Gray = 0.299*R + 0.587*G + 0.114*B
    ③ 该公式考虑了人眼对不同颜色的敏感度
    ④ 裁剪到0-255范围并转为uint8
    
    出参:
    - int/array: 灰度值，范围 0 到 255
    """
    # 使用全局默认值
    if weight_r is None:
        weight_r = GRAY_WEIGHT_RED
    if weight_g is None:
        weight_g = GRAY_WEIGHT_GREEN
    if weight_b is None:
        weight_b = GRAY_WEIGHT_BLUE
    
    # ① 加权平均计算灰度值
    gray = weight_r * red + weight_g * green + weight_b * blue
    
    # ② 裁剪到0-255范围
    gray = np.clip(gray, 0, 255)
    
    # ③ 转换为uint8整数
    gray = gray.astype(np.uint8)
    
    return gray


# create_water_mask 函数已废弃
# 水体掩膜现在直接从step1的CSV中读取（来自外部掩膜文件）
# 不再通过NDWI/NDVI阈值计算


def compute_indices(input_csv, output_csv, display_stats=None):
    """
    从CSV文件读取波段数据并计算NDWI和NDVI
    
    入参:
    - input_csv (str): 输入CSV文件路径
    - output_csv (str): 输出CSV文件路径
    - display_stats (bool): 是否显示统计信息，None时使用全局DISPLAY_STATS
    
    方法:
    ① 读取CSV文件并验证必需的列（包括water_mask）
    ② 提取红、绿、蓝、近红外波段数据
    ③ 计算灰度通道（RGB合并）
    ④ 使用向量化操作批量计算NDWI和NDVI
    ⑤ 将NDWI和NDVI拉伸到0-255范围
    ⑥ 验证water_mask列存在（来自step1）
    ⑦ 调整列顺序（将water_mask移到最后）
    ⑧ 保存为新的CSV文件
    ⑨ 输出统计信息和数据分布
    
    出参:
    - 无（直接保存CSV文件）
    """
    # 使用全局默认值
    if display_stats is None:
        display_stats = DISPLAY_STATS
    
    print(f"正在读取CSV文件: {input_csv}")
    
    # ① 读取CSV文件
    df = pd.read_csv(input_csv)
    
    print(f"CSV数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # ② 验证必需的列（包括water_mask）
    required_cols = [CSV_COL_LONGITUDE, CSV_COL_LATITUDE, 
                    CSV_COL_RED, CSV_COL_GREEN, CSV_COL_BLUE, CSV_COL_NIR, CSV_COL_WATER_MASK]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必需的列: {', '.join(missing_cols)}\n"
                        f"提示：water_mask列应该来自step1（从外部掩膜文件读取）")
    
    print(f"\n原始数据统计:")
    print(df[required_cols].describe())
    
    # ③ 提取波段数据（转换为numpy数组进行向量化计算）
    red = df[CSV_COL_RED].values
    green = df[CSV_COL_GREEN].values
    blue = df[CSV_COL_BLUE].values
    nir = df[CSV_COL_NIR].values
    
    # ④ 计算灰度通道（RGB合并为单通道）
    print(f"\n正在计算灰度通道...")
    df[CSV_COL_GRAY] = rgb_to_gray(red, green, blue)
    print(f"✓ 灰度通道计算完成 (范围: {df[CSV_COL_GRAY].min()}-{df[CSV_COL_GRAY].max()})")
    
    # ⑤ 批量计算NDWI和NDVI（向量化操作，高效）
    print(f"\n正在计算NDWI和NDVI...")
    df[CSV_COL_NDWI] = calculate_ndwi(green, nir)
    df[CSV_COL_NDVI] = calculate_ndvi(red, nir)
    
    print(f"✓ NDWI计算完成")
    print(f"✓ NDVI计算完成")
    
    # ⑥ 将NDWI和NDVI拉伸到0-255范围（便于可视化和保存为图像）
    print(f"\n正在将指数拉伸到0-255范围...")
    df[CSV_COL_NDWI_255] = stretch_to_255(df[CSV_COL_NDWI].values)
    df[CSV_COL_NDVI_255] = stretch_to_255(df[CSV_COL_NDVI].values)
    
    print(f"✓ NDWI拉伸完成 (范围: {df[CSV_COL_NDWI_255].min()}-{df[CSV_COL_NDWI_255].max()})")
    print(f"✓ NDVI拉伸完成 (范围: {df[CSV_COL_NDVI_255].min()}-{df[CSV_COL_NDVI_255].max()})")
    
    # ⑦ 验证water_mask列（来自step1，从外部掩膜文件读取）
    print(f"\n验证water_mask列（来自step1）...")
    if CSV_COL_WATER_MASK not in df.columns:
        raise ValueError(f"CSV文件缺少water_mask列！请确保使用step1生成的CSV文件。")
    
    # 统计水体像素数量
    water_pixels = (df[CSV_COL_WATER_MASK] == 1).sum()
    water_percentage = water_pixels / len(df) * 100
    print(f"✓ water_mask列验证通过（来自外部掩膜文件）")
    print(f"  水体像素数量: {water_pixels} ({water_percentage:.2f}%)")
    print(f"  非水体像素数量: {len(df) - water_pixels} ({100-water_percentage:.2f}%)")
    
    # ⑧ 调整列顺序：将water_mask移到最后
    print(f"\n正在调整列顺序（water_mask移到最后）...")
    column_order = ['longitude', 'latitude', 'red', 'green', 'blue', 'nir', 
                    'gray', 'ndwi', 'ndvi', 'ndwi_255', 'ndvi_255', 'water_mask']
    df = df[column_order]
    print(f"✓ 列顺序已调整: {list(df.columns)}")
    
    # ⑨ 保存到新CSV文件
    print(f"\n正在保存到: {output_csv}")
    df.to_csv(output_csv, index=False, float_format=f'%.{OUTPUT_DECIMAL_PLACES}f')
    
    print(f"✅ 计算完成!")
    print(f"输出文件: {output_csv}")
    print(f"数据行数: {len(df)}")
    print(f"输出列: {list(df.columns)}")
    
    # ⑩ 显示统计信息
    if display_stats:
        print(f"\n" + "="*60)
        print("NDWI和NDVI统计信息")
        print("="*60)
        
        # NDWI统计
        print(f"\n【NDWI - 归一化差异水体指数】")
        print(f"  公式: (Green - NIR) / (Green + NIR)")
        print(f"  范围: -1 (非水体) 到 +1 (纯水体)")
        print(f"  最小值: {df[CSV_COL_NDWI].min():.6f}")
        print(f"  最大值: {df[CSV_COL_NDWI].max():.6f}")
        print(f"  平均值: {df[CSV_COL_NDWI].mean():.6f}")
        print(f"  中位数: {df[CSV_COL_NDWI].median():.6f}")
        print(f"  标准差: {df[CSV_COL_NDWI].std():.6f}")
        
        # NDWI分布统计
        ndwi_positive = (df[CSV_COL_NDWI] > 0).sum()
        ndwi_negative = (df[CSV_COL_NDWI] <= 0).sum()
        print(f"\n  分布统计:")
        print(f"    正值像素 (可能为水体): {ndwi_positive} ({ndwi_positive/len(df)*100:.2f}%)")
        print(f"    负值像素 (可能为非水体): {ndwi_negative} ({ndwi_negative/len(df)*100:.2f}%)")
        
        # NDVI统计
        print(f"\n【NDVI - 归一化差异植被指数】")
        print(f"  公式: (NIR - Red) / (NIR + Red)")
        print(f"  范围: -1 (无植被) 到 +1 (茂密植被)")
        print(f"  最小值: {df[CSV_COL_NDVI].min():.6f}")
        print(f"  最大值: {df[CSV_COL_NDVI].max():.6f}")
        print(f"  平均值: {df[CSV_COL_NDVI].mean():.6f}")
        print(f"  中位数: {df[CSV_COL_NDVI].median():.6f}")
        print(f"  标准差: {df[CSV_COL_NDVI].std():.6f}")
        
        # NDVI分类统计
        ndvi_water = (df[CSV_COL_NDVI] < 0).sum()
        ndvi_bare = ((df[CSV_COL_NDVI] >= 0) & (df[CSV_COL_NDVI] < 0.2)).sum()
        ndvi_sparse = ((df[CSV_COL_NDVI] >= 0.2) & (df[CSV_COL_NDVI] < 0.4)).sum()
        ndvi_moderate = ((df[CSV_COL_NDVI] >= 0.4) & (df[CSV_COL_NDVI] < 0.6)).sum()
        ndvi_dense = (df[CSV_COL_NDVI] >= 0.6).sum()
        
        print(f"\n  分类统计:")
        print(f"    水体/裸土 (NDVI < 0): {ndvi_water} ({ndvi_water/len(df)*100:.2f}%)")
        print(f"    稀疏植被 (0 ≤ NDVI < 0.2): {ndvi_bare} ({ndvi_bare/len(df)*100:.2f}%)")
        print(f"    低覆盖植被 (0.2 ≤ NDVI < 0.4): {ndvi_sparse} ({ndvi_sparse/len(df)*100:.2f}%)")
        print(f"    中等植被 (0.4 ≤ NDVI < 0.6): {ndvi_moderate} ({ndvi_moderate/len(df)*100:.2f}%)")
        print(f"    茂密植被 (NDVI ≥ 0.6): {ndvi_dense} ({ndvi_dense/len(df)*100:.2f}%)")
        
        # 灰度通道统计
        print(f"\n【灰度通道 - RGB合并】")
        print(f"  公式: Gray = 0.299*R + 0.587*G + 0.114*B")
        print(f"  最小值: {df[CSV_COL_GRAY].min()}")
        print(f"  最大值: {df[CSV_COL_GRAY].max()}")
        print(f"  平均值: {df[CSV_COL_GRAY].mean():.2f}")
        print(f"  中位数: {df[CSV_COL_GRAY].median():.2f}")
        print(f"  说明: 灰度通道用于单波段可视化和分析")
        
        # 水体掩膜统计
        print(f"\n【水体掩膜 - 外部掩膜文件】")
        print(f"  来源: step1从外部掩膜文件读取")
        print(f"  说明: 二值掩膜（1=水体，0=非水体）")
        water_count = (df[CSV_COL_WATER_MASK] == 1).sum()
        non_water_count = (df[CSV_COL_WATER_MASK] == 0).sum()
        print(f"  水体像素: {water_count} ({water_count/len(df)*100:.2f}%)")
        print(f"  非水体像素: {non_water_count} ({non_water_count/len(df)*100:.2f}%)")
        print(f"  说明: 1=水体, 0=非水体")
        
        # 相关性分析
        print(f"\n【相关性分析】")
        correlation = df[CSV_COL_NDWI].corr(df[CSV_COL_NDVI])
        print(f"  NDWI与NDVI相关系数: {correlation:.6f}")
        if correlation < -0.5:
            print(f"  → 强负相关：水体和植被分布明显相反")
        elif correlation < -0.3:
            print(f"  → 中等负相关：水体区域植被较少")
        else:
            print(f"  → 弱相关或无相关")
        
        # 0-255范围统计
        print(f"\n【0-255范围拉伸值】")
        print(f"  NDWI_255统计:")
        print(f"    最小值: {df[CSV_COL_NDWI_255].min()}")
        print(f"    最大值: {df[CSV_COL_NDWI_255].max()}")
        print(f"    平均值: {df[CSV_COL_NDWI_255].mean():.2f}")
        print(f"    中位数: {df[CSV_COL_NDWI_255].median():.2f}")
        print(f"  NDVI_255统计:")
        print(f"    最小值: {df[CSV_COL_NDVI_255].min()}")
        print(f"    最大值: {df[CSV_COL_NDVI_255].max()}")
        print(f"    平均值: {df[CSV_COL_NDVI_255].mean():.2f}")
        print(f"    中位数: {df[CSV_COL_NDVI_255].median():.2f}")
        print(f"  说明: 0-255范围适合直接保存为图像或可视化")
        
        print("="*60)


def main():
    """
    主函数 - 解析命令行参数并执行计算
    
    入参: 命令行参数
    
    方法:
    - 解析输入输出路径
    - 验证文件存在性
    - 调用compute_indices执行计算
    
    出参: 无
    """
    parser = argparse.ArgumentParser(
        description='计算NDWI和NDVI指数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python ndwi_ndvi_calculator.py -i input.csv -o output.csv
  
输入CSV格式要求:
  必需列: longitude, latitude, red, green, blue, nir
  
输出CSV格式:
  原始列 + gray + ndwi + ndvi + ndwi_255 + ndvi_255 + water_mask
  
计算说明:
  Gray = 0.299*R + 0.587*G + 0.114*B  (灰度通道)
  NDWI = (Green - NIR) / (Green + NIR)  范围: -1 到 1
  NDVI = (NIR - Red) / (NIR + Red)      范围: -1 到 1
  NDWI_255/NDVI_255: 拉伸到0-255范围，适合可视化
  Water_Mask: 水体掩膜 (1=水体, 0=非水体, 基于lene.js算法)
        """
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='输入CSV文件路径')
    parser.add_argument('--output', '-o', required=True, 
                       help='输出CSV文件路径')
    parser.add_argument('--no-stats', action='store_true',
                       help='不显示统计信息')
    
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
        compute_indices(args.input, args.output, display_stats=not args.no_stats)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

