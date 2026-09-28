"""
CGCS2000投影坐标转WGS84坐标脚本
功能：
- 读取指定目录下的所有Shapefile点文件（.shp）
- 源坐标系：EPSG:4520 (CGCS2000 / 3-degree Gauss-Kruger zone 32)
- 两步转换：EPSG:4520 → EPSG:4490 (CGCS2000地理) → EPSG:4326 (WGS84)
- 保存转换后的结果为CSV格式
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import glob
from pathlib import Path

# ========== 配置参数 ==========
INPUT_FOLDER = r'E:\1代码\模型\gee\逐波段提取数据\玉树水深\玉树所有'
OUTPUT_FOLDER = r'E:\1代码\模型\gee\逐波段提取数据\玉树水深\转换后'

# CGCS2000 坐标系统说明
# 3度带 zone 32: 中央子午线 = 3° * 32 = 96°E
# EPSG:4520 = CGCS2000 / 3-degree Gauss-Kruger zone 32
#
# 转换流程:
# 1. EPSG:4520 (CGCS2000投影) → EPSG:4490 (CGCS2000地理坐标系)
# 2. EPSG:4490 (CGCS2000地理) → EPSG:4326 (WGS84地理)

SOURCE_EPSG = "EPSG:4520"  # CGCS2000 / 3-degree Gauss-Kruger zone 32 (CM 96E)
INTERMEDIATE_EPSG = "EPSG:4490"  # CGCS2000地理坐标系
TARGET_EPSG = "EPSG:4326"  # WGS84地理坐标系
ZONE_NUMBER = 32  # 3度带 zone 32
CENTRAL_MERIDIAN = 96  # 中央子午线 96°E

# 支持的Shapefile文件扩展名
SHAPEFILE_EXTENSIONS = ['*.shp']


def create_output_folder():
    """创建输出文件夹"""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f'✓ 创建输出文件夹: {OUTPUT_FOLDER}')
    else:
        print(f'✓ 输出文件夹已存在: {OUTPUT_FOLDER}')


def read_shapefile(file_path):
    """
    入参:
    - file_path (str): Shapefile文件路径
    
    方法:
    - 使用geopandas读取Shapefile点文件
    - 自动识别坐标系信息
    - 提取几何信息和属性数据
    
    出参:
    - gdf (GeoDataFrame): 成功返回GeoDataFrame，失败返回None
    """
    try:
        # 读取Shapefile
        gdf = gpd.read_file(file_path, encoding='utf-8')
        
        # 检查是否为点要素
        if not all(gdf.geometry.geom_type == 'Point'):
            print(f'  ⚠ 警告: 文件包含非点要素，将只处理点要素')
            gdf = gdf[gdf.geometry.geom_type == 'Point']
        
        print(f'  ✓ 成功读取Shapefile')
        print(f'  原始坐标系: {gdf.crs}')
        
        return gdf
        
    except Exception as e:
        print(f'  ✗ 读取Shapefile失败: {os.path.basename(file_path)} - {str(e)}')
        return None


def convert_gdf_coordinates(gdf):
    """
    入参:
    - gdf (GeoDataFrame): 包含CGCS2000投影坐标的GeoDataFrame
    
    方法:
    - 两步坐标转换
    - 步骤1: EPSG:4520 (CGCS2000投影) → EPSG:4490 (CGCS2000地理)
    - 步骤2: EPSG:4490 (CGCS2000地理) → EPSG:4326 (WGS84地理)
    - 从geometry中提取经纬度到独立列
    
    出参:
    - gdf_wgs84 (GeoDataFrame): 转换为WGS84的GeoDataFrame
    """
    print(f'  两步坐标转换:')
    print(f'    步骤1: {SOURCE_EPSG} → {INTERMEDIATE_EPSG} (CGCS2000地理)')
    print(f'    步骤2: {INTERMEDIATE_EPSG} → {TARGET_EPSG} (WGS84)')
    
    # 保存原始坐标
    gdf['原始X'] = gdf.geometry.x
    gdf['原始Y'] = gdf.geometry.y
    
    # 设置源坐标系（如果未设置）
    if gdf.crs is None:
        print(f'  ⚠ 未检测到坐标系，强制设置为 {SOURCE_EPSG}')
        gdf = gdf.set_crs(SOURCE_EPSG)
    elif gdf.crs.to_string() != SOURCE_EPSG:
        print(f'  ⚠ 检测到坐标系 {gdf.crs}，但将使用 {SOURCE_EPSG} 进行转换')
        gdf = gdf.set_crs(SOURCE_EPSG, allow_override=True)
    
    # 步骤1: 转换到CGCS2000地理坐标系
    gdf_cgcs = gdf.to_crs(INTERMEDIATE_EPSG)
    gdf_cgcs['CGCS2000_lon'] = gdf_cgcs.geometry.x
    gdf_cgcs['CGCS2000_lat'] = gdf_cgcs.geometry.y
    print(f'    ✓ 步骤1完成')
    
    # 步骤2: 转换到WGS84
    gdf_wgs84 = gdf_cgcs.to_crs(TARGET_EPSG)
    gdf_wgs84['lon'] = gdf_wgs84.geometry.x
    gdf_wgs84['lat'] = gdf_wgs84.geometry.y
    print(f'    ✓ 步骤2完成')
    
    return gdf_wgs84


def process_single_file(file_path):
    """
    入参:
    - file_path (str): Shapefile文件路径
    
    方法:
    - 读取Shapefile点文件
    - 执行两步坐标转换（EPSG:4520 → EPSG:4490 → EPSG:4326）
    - 保存完整版和精简版CSV
    - 完整版包含所有原始属性+坐标转换结果
    - 精简版仅包含经纬度和水深
    
    出参:
    - success (bool): 处理成功返回True，失败返回False
    """
    file_name = os.path.basename(file_path)
    print(f'\n处理文件: {file_name}')
    print('-' * 80)

    # 读取Shapefile
    print('  [1/3] 读取Shapefile...')
    gdf = read_shapefile(file_path)
    if gdf is None:
        return False

    print(f'  ✓ 成功读取 {len(gdf)} 个点要素')
    print(f'  属性字段: {[col for col in gdf.columns if col != "geometry"]}')
    
    # 显示原始坐标范围
    x_values = gdf.geometry.x
    y_values = gdf.geometry.y
    print(f'  原始X坐标范围: {x_values.min():.2f} ~ {x_values.max():.2f}')
    print(f'  原始Y坐标范围: {y_values.min():.2f} ~ {y_values.max():.2f}')

    # 转换坐标
    print('\n  [2/3] 执行坐标转换...')
    gdf_wgs84 = convert_gdf_coordinates(gdf)

    # 统计转换成功率
    valid_count = gdf_wgs84['lon'].notna().sum()
    success_rate = (valid_count / len(gdf_wgs84)) * 100

    print(f'  ✓ 转换完成')
    print(f'  成功: {valid_count}/{len(gdf_wgs84)} ({success_rate:.1f}%)')

    # 显示转换结果示例
    print('\n  转换结果示例（前3条）:')
    for idx in range(min(3, len(gdf_wgs84))):
        row = gdf_wgs84.iloc[idx]
        print(f'    原始CGCS2000投影: ({row["原始X"]:.2f}, {row["原始Y"]:.2f})')
        print(f'    → CGCS2000地理: ({row["CGCS2000_lon"]:.6f}, {row["CGCS2000_lat"]:.6f})')
        print(f'    → WGS84: ({row["lon"]:.6f}, {row["lat"]:.6f})')
        print()

    # 转换为DataFrame用于CSV保存（移除geometry列）
    df_result = pd.DataFrame(gdf_wgs84.drop(columns='geometry'))

    # 保存完整结果
    print('  [3/3] 保存结果...')
    output_name = os.path.splitext(file_name)[0] + '_转换后.csv'
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'  ✓ 完整结果已保存: {output_name}')

    # 保存精简版（仅经纬度和水深）
    print('\n  保存精简版（经纬度+水深）...')

    # 查找水深列
    depth_col = None
    possible_depth_names = ['水深', 'depth', 'Depth', 'DEPTH', '深度', 'z', 'Z', 'H', 'h', 'shuishen']

    for col in df_result.columns:
        for possible_name in possible_depth_names:
            if possible_name.lower() in str(col).lower():
                depth_col = col
                break
        if depth_col:
            break

    if depth_col:
        # 创建精简DataFrame（经纬度+水深）
        simple_df = df_result[['lon', 'lat', depth_col]].copy()

        # 重命名列为标准名称
        simple_df.columns = ['lon', 'lat', 'depth']

        # 保存精简版
        simple_output_name = os.path.splitext(file_name)[0] + '_精简版.csv'
        simple_output_path = os.path.join(OUTPUT_FOLDER, simple_output_name)
        simple_df.to_csv(simple_output_path, index=False, encoding='utf-8-sig')

        print(f'  ✓ 精简版已保存: {simple_output_name}')
        print(f'  精简版列名: {simple_df.columns.tolist()}')
        print(f'  精简版行数: {len(simple_df)}')
    else:
        print(f'  ⚠ 未找到水深列，跳过精简版保存')
        print(f'  可用列名: {df_result.columns.tolist()}')

    return True


def main():
    """
    入参:
    - 无
    
    方法:
    - 扫描输入目录中的所有Shapefile文件
    - 批量处理每个文件进行坐标转换
    - 输出完整版和精简版CSV结果
    
    出参:
    - 无（直接执行转换流程）
    """
    print('=' * 80)
    print('CGCS2000投影坐标转WGS84脚本')
    print('=' * 80)
    print(f'输入文件夹: {INPUT_FOLDER}')
    print(f'输出文件夹: {OUTPUT_FOLDER}')
    print(f'源坐标系: {SOURCE_EPSG} (CGCS2000 / 3-degree GK zone 32)')
    print(f'中间坐标系: {INTERMEDIATE_EPSG} (CGCS2000地理)')
    print(f'目标坐标系: {TARGET_EPSG} (WGS84)')
    print(f'中央子午线: {CENTRAL_MERIDIAN}°E')
    print('=' * 80)

    # 检查输入文件夹
    if not os.path.exists(INPUT_FOLDER):
        print(f'\n✗ 输入文件夹不存在: {INPUT_FOLDER}')
        return

    # 创建输出文件夹
    create_output_folder()

    # 查找所有Shapefile文件
    print('\n扫描Shapefile文件...')
    input_files = []
    for pattern in SHAPEFILE_EXTENSIONS:
        input_files.extend(glob.glob(os.path.join(INPUT_FOLDER, pattern)))

    # 过滤掉已转换的文件
    input_files = [f for f in input_files if '转换后' not in os.path.basename(f)]

    if not input_files:
        print('✗ 未找到任何Shapefile文件（.shp）')
        return

    print(f'✓ 找到 {len(input_files)} 个Shapefile')
    for idx, file_path in enumerate(input_files, 1):
        file_size = os.path.getsize(file_path) / 1024
        print(f'  [{idx}] {os.path.basename(file_path)} ({file_size:.2f} KB)')

    # 处理所有文件
    print('\n开始处理...')
    success_count = 0
    fail_count = 0

    for file_path in input_files:
        if process_single_file(file_path):
            success_count += 1
        else:
            fail_count += 1

    # 总结
    print('\n' + '=' * 80)
    print('处理完成')
    print('=' * 80)
    print(f'总文件数: {len(input_files)}')
    print(f'成功: {success_count}')
    print(f'失败: {fail_count}')
    print(f'输出目录: {OUTPUT_FOLDER}')
    print('=' * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n⚠ 用户中断')
    except Exception as e:
        print(f'\n✗ 发生错误: {str(e)}')
        import traceback
        traceback.print_exc()
