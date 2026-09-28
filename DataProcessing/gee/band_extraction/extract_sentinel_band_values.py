"""
哨兵影像波段值提取脚本
入参:
- input_csv: 包含WGS84经纬度坐标(lon, lat, depth)的CSV文件路径
- output_csv: 输出结果的CSV文件路径
方法:
- 读取WGS84经纬度坐标点数据
- 使用GEE API提取Sentinel-2 SR波段值（B2-B12）
- 分批处理避免API限制
出参:
- CSV文件：包含经纬度、11个波段值、水深
"""

import ee
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime
import os
from pathlib import Path
import glob

# 导入配置文件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *

# ========== 配置参数 ==========
# 使用配置文件中的参数
INPUT_FOLDER_PATH = INPUT_FOLDER
INPUT_CSV = INPUT_CSV  # 保留兼容性
OUTPUT_CSV = os.path.join(BASE_DIR, OUTPUT_EXTRACT_CSV)
BATCH_SIZE = BATCH_SIZE
START_DATE = START_DATE
END_DATE = END_DATE
CLOUD_THRESHOLD = CLOUD_THRESHOLD
PROJECT_ID = PROJECT_ID

try:
    ee.Initialize(project=PROJECT_ID)
    print(f'✓ GEE已成功初始化 (Project: {PROJECT_ID})')
except Exception as e:
    print(f'× GEE初始化失败: {str(e)}')
    print('请先运行认证脚本: python 认证/python_认证.py')
    sys.exit(1)


def extract_sentinel_bands_batch(points_data):
    """
    批量提取多个点的Sentinel-2波段值
    入参:
    - points_data (list): 包含(lon, lat, depth)元组的列表
    方法:
    - 将所有点转换为FeatureCollection
    - 一次性获取整个区域的Sentinel-2影像并中位值合成
    - 使用sampleRegions批量提取所有点的波段值
    出参:
    - results (list): 包含所有点结果的字典列表
    """
    try:
        # 创建点要素集合
        features = []
        for idx, (lon, lat, depth) in enumerate(points_data):
            point = ee.Geometry.Point([lon, lat])
            # 处理NaN值，GEE不支持NaN
            depth_value = float(depth) if not np.isnan(depth) else -9999
            feature = ee.Feature(point, {
                'point_id': idx,
                'lon': lon,
                'lat': lat,
                'depth': depth_value
            })
            features.append(feature)
        
        points_fc = ee.FeatureCollection(features)
        
        # 获取所有点的边界
        bounds = points_fc.geometry().bounds()
        
        # 获取Sentinel-2 SR影像集合
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(bounds) \
            .filterDate(START_DATE, END_DATE) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
        
        # 中位值合成并选择B2-B12波段
        s2_median = s2_collection.median().select(
            ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        )
        
        # 批量采样所有点
        sampled = s2_median.sampleRegions(
            collection=points_fc,
            scale=10,
            geometries=False
        )
        
        # 获取结果
        sampled_list = sampled.getInfo()['features']
        
        # 组装结果
        results = []
        for feature in sampled_list:
            props = feature['properties']
            # 将-9999转换回NaN
            depth_value = props['depth']
            if depth_value == -9999:
                depth_value = np.nan
            
            result = {
                'lon': props['lon'],
                'lat': props['lat'],
                'band_2': props.get('B2', np.nan),
                'band_3': props.get('B3', np.nan),
                'band_4': props.get('B4', np.nan),
                'band_5': props.get('B5', np.nan),
                'band_6': props.get('B6', np.nan),
                'band_7': props.get('B7', np.nan),
                'band_8': props.get('B8', np.nan),
                'band_8a': props.get('B8A', np.nan),
                'band_9': props.get('B9', np.nan),
                'band_11': props.get('B11', np.nan),
                'band_12': props.get('B12', np.nan),
                'depth': depth_value
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f'  错误: 批量提取失败: {str(e)}')
        return []


def process_batch(df_batch, batch_idx, total_batches):
    """
    批量处理点数据
    入参:
    - df_batch (DataFrame): 当前批次的数据框（必须包含lon, lat, depth列）
    - batch_idx (int): 当前批次索引
    - total_batches (int): 总批次数
    方法:
    - 读取WGS84经纬度坐标
    - 一次性调用GEE提取所有点的波段值
    - 组装结果数据
    出参:
    - results (list): 包含所有点结果的列表
    """
    total_points = len(df_batch)
    
    print(f'\n========== 处理批次 {batch_idx}/{total_batches} (共{total_points}个点) ==========')
    print(f'  步骤1: 读取坐标数据...', end=' ')
    
    # 读取经纬度和水深数据
    points_data = []
    invalid_count = 0
    
    for idx, row in df_batch.iterrows():
        # 检查必需的列
        if 'lon' not in row or 'lat' not in row:
            print(f'  ✗ 错误: 未找到lon/lat列，请确保输入CSV包含WGS84经纬度坐标')
            return []
        
        lon = row['lon']
        lat = row['lat']
        
        # 过滤无效坐标值（NaN, Infinity, -Infinity）
        if not np.isfinite(lon) or not np.isfinite(lat):
            invalid_count += 1
            continue
        
        # 基本的经纬度范围检查
        if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
            invalid_count += 1
            continue
        
        # 支持多种深度列名
        if 'depth' in row:
            depth = row['depth']
        elif '水深' in row:
            depth = row['水深']
        else:
            depth = np.nan

        points_data.append((lon, lat, depth))
    
    if invalid_count > 0:
        print(f'⚠ (过滤了 {invalid_count} 个无效坐标) ', end='')
    print('✓')
    
    # 检查是否有有效点
    if len(points_data) == 0:
        print(f'  ⚠ 警告: 本批次没有有效坐标点，跳过处理')
        return []
    
    # 显示区域范围
    lons = [p[0] for p in points_data]
    lats = [p[1] for p in points_data]
    print(f'  有效点数: {len(points_data)}/{total_points}')
    print(f'  区域范围: 经度[{min(lons):.4f}, {max(lons):.4f}], 纬度[{min(lats):.4f}, {max(lats):.4f}]')
    
    # 批量提取波段值
    print(f'  步骤2: 批量提取波段值 (一次性处理{len(points_data)}个点)...', end=' ')
    results = extract_sentinel_bands_batch(points_data)
    print('✓')
    
    success_count = len(results)
    fail_count = len(points_data) - success_count
    
    print(f'  结果: 成功 {success_count} 个, 失败 {fail_count} 个')
    
    return results


def read_table_file(file_path):
    """
    读取表格文件（支持CSV/XLS/XLSX）
    入参:
    - file_path (str): 文件路径
    方法:
    - 根据文件扩展名选择读取方式
    - 尝试多种编码方式
    出参:
    - df (DataFrame): 读取的数据框
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.csv':
            # 尝试多种编码方式读取CSV
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='gb18030')
        elif file_ext in ['.xls', '.xlsx']:
            # 读取Excel文件
            df = pd.read_excel(file_path, engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
        else:
            raise ValueError(f'不支持的文件格式: {file_ext}')

        return df
    except Exception as e:
        print(f'  ✗ 读取文件失败: {os.path.basename(file_path)} - {str(e)}')
        return None


def get_input_files():
    """
    获取输入文件列表
    入参:
    - 无
    方法:
    - 从INPUT_FOLDER_PATH文件夹中查找所有CSV/XLS/XLSX文件
    出参:
    - files (list): 文件路径列表
    """
    if not os.path.exists(INPUT_FOLDER_PATH):
        print(f'✗ 输入文件夹不存在: {INPUT_FOLDER_PATH}')
        return []

    # 查找所有支持的文件格式
    patterns = ['*.csv', '*.xls', '*.xlsx']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER_PATH, pattern)))

    # 过滤掉输出文件和临时文件（只过滤真正的输出文件）
    files = [f for f in files if not any(x in os.path.basename(f) for x in
                                         ['波段值', 'temp_batch', '湖库波段值_合并', '预处理后', '缩放'])]

    return sorted(files)


def process_single_file(input_file, file_idx, total_files):
    """
    处理单个输入文件
    入参:
    - input_file (str): 输入文件路径
    - file_idx (int): 当前文件索引
    - total_files (int): 总文件数
    方法:
    - 读取文件
    - 分批处理数据点
    - 保存中间结果
    出参:
    - success (bool): 是否成功处理
    """
    file_name = os.path.basename(input_file)
    print('\n' + '=' * 80)
    print(f'处理文件 [{file_idx}/{total_files}]: {file_name}')
    print('=' * 80)

    # 读取文件
    print(f'[1/2] 读取文件...')
    df = read_table_file(input_file)
    if df is None:
        return False

    print(f'✓ 成功读取 {len(df)} 个点')
    print(f'  列名: {df.columns.tolist()}')

    # 分批处理
    print(f'\n[2/2] 开始分批处理...')
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    all_results = []

    start_time = time.time()

    for i in range(0, len(df), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE + 1
        df_batch = df.iloc[i:i+BATCH_SIZE]

        batch_results = process_batch(df_batch, batch_idx, total_batches)
        all_results.extend(batch_results)

        # 保存中间结果（添加文件名前缀）
        if len(batch_results) > 0:
            temp_df = pd.DataFrame(batch_results)
            # 使用文件名（不含扩展名）作为前缀
            file_prefix = os.path.splitext(file_name)[0]
            temp_output = os.path.join(BASE_DIR, f'{file_prefix}_temp_batch{batch_idx}.csv')
            temp_df.to_csv(temp_output, index=False, encoding='utf-8')
            print(f'  ✓ 中间结果已保存: {os.path.basename(temp_output)}')

    elapsed_time = time.time() - start_time

    # 统计信息
    print(f'\n文件处理完成:')
    print(f'  总点数: {len(df)}')
    print(f'  成功提取: {len(all_results)}')
    print(f'  失败/跳过: {len(df) - len(all_results)}')
    print(f'  耗时: {elapsed_time/60:.2f} 分钟')
    if len(all_results) > 0:
        print(f'  平均速度: {len(all_results)/(elapsed_time/60):.2f} 点/分钟')

    return len(all_results) > 0


def main():
    """
    主函数
    入参:
    - 无
    方法:
    - 从INPUT_FOLDER遍历读取所有WGS84坐标CSV文件（lon, lat, depth格式）
    - 分批处理所有点，调用GEE提取波段值
    - 保存中间结果
    出参:
    - 无，结果保存到文件
    """
    print('=' * 80)
    print('哨兵影像波段值提取脚本')
    print('=' * 80)
    print(f'输入文件夹: {INPUT_FOLDER_PATH}')
    print(f'输入格式: WGS84经纬度 (lon, lat, depth)')
    print(f'时间范围: {START_DATE} 至 {END_DATE}')
    print(f'云覆盖阈值: {CLOUD_THRESHOLD}%')
    print(f'批次大小: {BATCH_SIZE}个点/批')
    print('=' * 80)

    # 获取输入文件列表
    print('\n[1/2] 扫描输入文件...')
    input_files = get_input_files()

    if not input_files:
        print('✗ 未找到任何输入文件（支持格式：CSV/XLS/XLSX）')
        print(f'  请在文件夹中放入数据文件: {INPUT_FOLDER_PATH}')
        sys.exit(1)

    print(f'✓ 找到 {len(input_files)} 个文件待处理:')
    for idx, file_path in enumerate(input_files, 1):
        file_size = os.path.getsize(file_path) / 1024
        print(f'  [{idx}] {os.path.basename(file_path)} ({file_size:.2f} KB)')

    # 处理所有文件
    print(f'\n[2/2] 开始处理所有文件...')
    total_start_time = time.time()
    success_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, 1):
        if process_single_file(input_file, idx, len(input_files)):
            success_count += 1
        else:
            fail_count += 1

    total_elapsed_time = time.time() - total_start_time

    # 总体统计信息
    print('\n' + '=' * 80)
    print('所有文件处理完成统计')
    print('=' * 80)
    print(f'总文件数: {len(input_files)}')
    print(f'成功处理: {success_count} 个文件')
    print(f'失败: {fail_count} 个文件')
    print(f'总耗时: {total_elapsed_time/60:.2f} 分钟')
    print('=' * 80)

    # 提示后续步骤
    print('\n下一步:')
    print('  运行 merge_and_preprocess.py 合并所有批次文件')
    print('=' * 80)


if __name__ == '__main__':
    main()

