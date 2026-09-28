"""
本地影像波段值提取脚本
入参:
- tif_path: 本地GeoTIFF影像文件路径
- input_csv: 包含坐标点的CSV文件路径(lon, lat列)
- output_csv: 输出结果的CSV文件路径
方法:
- 读取本地GeoTIFF影像
- 读取CSV中的WGS84坐标点
- 将坐标点转换到影像坐标系
- 批量提取各位置的波段值
出参:
- CSV文件：包含经纬度、所有波段值、深度(如有)

python extract_from_local_tiff.py --tif "H:\一院\水深反演\3水深遥感反演系统V2.0\2测试数据\多模型推理\image_ori" --csv "E:\1代码\模型\影像处理\在线逐波段提取数据\train.csv" --output "E:\1代码\模型\影像处理\在线逐波段提取数据\band_values_extracted.csv"

"""

import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer, CRS
from tqdm import tqdm
import re
import argparse
import glob
from typing import Dict, List, Tuple, Optional
import time


def detect_band_mapping(src) -> Dict[int, str]:
    """
    智能检测波段映射关系
    入参:
    - src: rasterio.DatasetReader，打开的影像文件
    方法:
    - 从波段描述或标签中读取波段名称
    - 解析Sentinel-2波段编号
    - 根据波段数量和位置推断映射关系
    出参:
    - dict: {band_index: 'band_name'} 例如 {1: 'band_2', 2: 'band_3', 3: 'band_4', 4: 'band_8'}
    """
    band_count = src.count
    band_mapping = {}

    # 尝试从波段描述中读取波段名称
    band_names = []
    if hasattr(src, 'descriptions') and src.descriptions:
        band_names = [desc for desc in src.descriptions if desc]

    # 如果描述为空，尝试从标签中读取
    if not band_names:
        for i in range(1, band_count + 1):
            try:
                tags = src.tags(i)
                if 'BANDNAME' in tags:
                    band_names.append(tags['BANDNAME'])
                elif 'band_name' in tags:
                    band_names.append(tags['band_name'])
                else:
                    band_names.append(None)
            except:
                band_names.append(None)

    # 解析波段名称并建立映射
    for i in range(band_count):
        band_idx = i + 1  # rasterio使用1-based索引
        band_name = band_names[i] if i < len(band_names) else None

        # 尝试从名称中提取波段编号（如 "B2", "B3", "B4", "B8"）
        sentinel_band_num = None
        if band_name:
            match = re.search(r'B(\d+)', str(band_name).upper())
            if match:
                sentinel_band_num = int(match.group(1))

        # 如果没有找到波段名称，根据波段数量和位置推断
        if sentinel_band_num is None:
            if band_count == 4:
                # 对于4个波段的情况，假设是 B2, B3, B4, B8
                if i == 0:
                    sentinel_band_num = 2
                elif i == 1:
                    sentinel_band_num = 3
                elif i == 2:
                    sentinel_band_num = 4
                elif i == 3:
                    sentinel_band_num = 8
            elif band_count >= 11:
                # 标准 Sentinel-2 波段顺序
                standard_order = [2, 3, 4, 5, 6, 7, 8, 8, 9, 11, 12]
                if i < len(standard_order):
                    sentinel_band_num = standard_order[i]
                else:
                    sentinel_band_num = i + 2
            else:
                sentinel_band_num = i + 2

        # 映射所有检测到的波段
        band_mapping[band_idx] = f'band_{sentinel_band_num}'

    return band_mapping


def read_coordinate_file(file_path: str) -> pd.DataFrame:
    """
    读取坐标文件（支持CSV/XLS/XLSX）
    入参:
    - file_path: str，文件路径
    方法:
    - 根据文件扩展名选择读取方式
    - 尝试多种编码方式
    出参:
    - pd.DataFrame，包含lon, lat列的数据框
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.csv':
            # 尝试多种编码方式
            for encoding in ['utf-8', 'gbk', 'gb18030']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError('无法解码CSV文件')
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path, engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
        else:
            raise ValueError(f'不支持的文件格式: {file_ext}')

        return df
    except Exception as e:
        print(f'  ✗ 读取文件失败: {os.path.basename(file_path)} - {str(e)}')
        return None


def validate_coordinates(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    验证并提取坐标数据
    入参:
    - df: pd.DataFrame，包含坐标的数据框
    方法:
    - 检查必需的列（lon, lat）
    - 过滤无效坐标值
    - 提取深度数据（如有）
    出参:
    - tuple: (lon_array, lat_array, depth_array or None)
    """
    # 检查必需的列
    if 'lon' not in df.columns or 'lat' not in df.columns:
        raise ValueError('CSV文件必须包含 lon 和 lat 列')

    lons = df['lon'].values
    lats = df['lat'].values

    # 过滤无效坐标
    valid_mask = np.isfinite(lons) & np.isfinite(lats)
    valid_mask &= (lons >= -180) & (lons <= 180)
    valid_mask &= (lats >= -90) & (lats <= 90)

    valid_lons = lons[valid_mask]
    valid_lats = lats[valid_mask]

    # 提取深度数据（如有）
    depths = None
    if 'depth' in df.columns:
        depths = df['depth'].values[valid_mask]
    elif '水深' in df.columns:
        depths = df['水深'].values[valid_mask]

    return valid_lons, valid_lats, depths


def extract_band_values_at_points(
    src,
    lons: np.ndarray,
    lats: np.ndarray,
    transformer: Transformer,
    band_mapping: Dict[int, str],
    scale_factor: float = 1.0
) -> pd.DataFrame:
    """
    在指定坐标点提取波段值
    入参:
    - src: rasterio.DatasetReader，打开的影像文件
    - lons: np.ndarray，经度数组（WGS84）
    - lats: np.ndarray，纬度数组（WGS84）
    - transformer: Transformer，坐标转换器（WGS84 -> 影像CRS）
    - band_mapping: dict，波段映射关系
    - scale_factor: float，缩放系数
    方法:
    - 将WGS84坐标转换到影像坐标系
    - 将地理坐标转换为像素坐标
    - 提取每个点的波段值
    出参:
    - pd.DataFrame，包含坐标和波段值
    """
    # 转换坐标系：WGS84 -> 影像CRS
    x_coords, y_coords = transformer.transform(lons, lats)

    # 转换为像素坐标
    rows, cols = rasterio.transform.rowcol(src.transform, x_coords, y_coords)

    # 检查点是否在影像范围内
    height, width = src.height, src.width
    valid_mask = (np.array(rows) >= 0) & (np.array(rows) < height)
    valid_mask &= (np.array(cols) >= 0) & (np.array(cols) < width)

    valid_rows = np.array(rows)[valid_mask]
    valid_cols = np.array(cols)[valid_mask]
    valid_lons = lons[valid_mask]
    valid_lats = lats[valid_mask]

    print(f'  有效点数: {len(valid_rows)}/{len(lons)}')

    # 提取波段值
    results = {
        'lon': valid_lons,
        'lat': valid_lats
    }

    for band_idx, band_name in band_mapping.items():
        if band_idx <= src.count:
            # 读取波段数据
            band_data = src.read(band_idx)

            # 提取指定位置的值
            values = band_data[valid_rows, valid_cols]

            # 应用缩放系数并保留四位小数
            if scale_factor != 1.0:
                values = values * scale_factor

            results[band_name] = np.round(values, 4)
        else:
            # 波段不存在，填充NaN
            results[band_name] = np.full(len(valid_rows), np.nan)

    return pd.DataFrame(results)


def scan_tif_files(folder_path: str) -> List[str]:
    """
    扫描文件夹中的所有TIF影像文件
    入参:
    - folder_path: str，文件夹路径
    方法:
    - 递归扫描文件夹
    - 筛选.tif和.tiff文件
    出参:
    - list: TIF文件路径列表
    """
    if not os.path.exists(folder_path):
        raise ValueError(f'文件夹不存在: {folder_path}')

    tif_files = []
    patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']

    for pattern in patterns:
        tif_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        # 递归搜索子文件夹
        tif_files.extend(glob.glob(os.path.join(folder_path, '**', pattern), recursive=True))

    return sorted(list(set(tif_files)))  # 去重并排序


def extract_from_local_tiff(
    tif_folder: str,
    input_csv: str,
    output_csv: str,
    scale_factor: float = 0.0001
) -> None:
    """
    主函数：从本地GeoTIFF影像文件夹批量提取指定点的波段值，合并到单个CSV
    入参:
    - tif_folder: str，GeoTIFF影像文件夹路径
    - input_csv: str，输入CSV文件路径
    - output_csv: str，输出CSV文件路径（合并所有影像结果）
    - scale_factor: float，缩放系数（默认0.0001）
    方法:
    - 扫描文件夹中的所有TIF影像
    - 读取CSV坐标点
    - 批量提取每个影像的波段值
    - 合并所有结果到单个CSV文件
    出参:
    - None，副作用：生成CSV文件
    """
    print('=' * 80)
    print('本地影像波段值提取 - 批量处理模式')
    print('=' * 80)
    print(f'影像文件夹: {tif_folder}')
    print(f'坐标文件: {os.path.basename(input_csv)}')
    print(f'输出文件: {output_csv}')
    print(f'缩放系数: {scale_factor}')
    print('=' * 80)

    # 扫描TIF文件
    print('\n[1/3] 扫描影像文件...')
    tif_files = scan_tif_files(tif_folder)

    if not tif_files:
        print('  ✗ 未找到任何TIF影像文件')
        return

    print(f'  找到 {len(tif_files)} 个影像文件:')
    for idx, f in enumerate(tif_files, 1):
        file_size = os.path.getsize(f) / (1024 * 1024)  # MB
        rel_path = os.path.relpath(f, tif_folder)
        print(f'    [{idx}] {rel_path} ({file_size:.2f} MB)')

    # 读取坐标文件
    print('\n[2/3] 读取坐标文件...')
    df = read_coordinate_file(input_csv)
    if df is None:
        return

    print(f'  总点数: {len(df)}')

    # 验证坐标
    try:
        lons, lats, depths = validate_coordinates(df)
        print(f'  有效坐标点: {len(lons)}')
    except Exception as e:
        print(f'  ✗ 坐标验证失败: {str(e)}')
        return

    # 批量处理并合并结果
    print(f'\n[3/3] 批量处理影像文件...')
    total_start = time.time()
    all_results = []
    success_count = 0
    fail_count = 0

    for tif_file in tif_files:
        print('\n' + '-' * 80)
        try:
            result_df = process_single_tiff(
                tif_file,
                lons,
                lats,
                depths,
                scale_factor
            )
            all_results.append(result_df)
            success_count += 1
        except Exception as e:
            print(f'  ✗ 处理失败: {str(e)}')
            fail_count += 1

    total_elapsed = time.time() - total_start

    # 合并所有结果
    if all_results:
        print('\n' + '=' * 80)
        print('合并结果...')
        merged_df = pd.concat(all_results, ignore_index=True)

        # 保存合并结果
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        merged_df.to_csv(output_csv, index=False, encoding='utf-8-sig', float_format='%.4f')

        print(f'  ✓ 合并完成: {len(merged_df)} 行数据')
        print(f'  ✓ 输出文件: {output_csv}')

    # 统计信息
    print('\n' + '=' * 80)
    print('批量处理完成')
    print('=' * 80)
    print(f'总文件数: {len(tif_files)}')
    print(f'成功: {success_count} 个')
    print(f'失败: {fail_count} 个')
    print(f'总提取点数: {sum(len(df) for df in all_results) if all_results else 0}')
    print(f'总耗时: {total_elapsed/60:.2f} 分钟')
    print('=' * 80)


def process_single_tiff(
    tif_path: str,
    lons: np.ndarray,
    lats: np.ndarray,
    depths: Optional[np.ndarray],
    scale_factor: float
) -> pd.DataFrame:
    """
    处理单个TIF影像文件
    入参:
    - tif_path: str，GeoTIFF影像文件路径
    - lons: np.ndarray，经度数组
    - lats: np.ndarray，纬度数组
    - depths: np.ndarray or None，深度数组
    - scale_factor: float，缩放系数
    方法:
    - 读取单个GeoTIFF影像
    - 提取坐标点波段值
    - 返回结果DataFrame
    出参:
    - pd.DataFrame，包含提取的波段值
    """
    print(f'处理影像: {os.path.basename(tif_path)}')

    start_time = time.time()

    # 1. 打开影像文件
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f'影像文件不存在: {tif_path}')

    with rasterio.open(tif_path) as src:
        print(f'  尺寸: {src.width}x{src.height}')
        print(f'  波段数: {src.count}')
        print(f'  坐标系: {src.crs}')

        # 检测波段映射
        band_mapping = detect_band_mapping(src)
        print(f'  检测到波段: {list(band_mapping.values())}')

        # 创建坐标转换器（WGS84 -> 影像CRS）
        transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)

        # 2. 提取波段值
        results_df = extract_band_values_at_points(
            src, lons, lats, transformer, band_mapping, scale_factor
        )

        # 3. 添加深度数据（如有）
        if depths is not None:
            valid_mask = np.isfinite(lons) & np.isfinite(lats)
            valid_depths = depths[valid_mask]
            # 只保留有效点的深度数据
            results_df['depth'] = valid_depths[:len(results_df)]

    elapsed_time = time.time() - start_time
    print(f'  ✓ 完成: {len(results_df)} 个点, 耗时 {elapsed_time:.2f} 秒')

    return results_df


def main():
    """
    命令行入口
    入参:
    - 无（从命令行参数获取）
    方法:
    - 解析命令行参数
    - 调用extract_from_local_tiff批量处理
    - 默认缩放系数0.0001
    出参:
    - 无
    """
    parser = argparse.ArgumentParser(
        description='从本地GeoTIFF影像文件夹批量提取指定点的波段值，合并到单个CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python extract_from_local_tiff.py --tif ./images/ --csv points.csv --output result.csv

  指定缩放系数:
  python extract_from_local_tiff.py --tif ./images/ --csv points.csv --output result.csv --scale 0.0001
        '''
    )
    parser.add_argument('--tif', required=True, help='GeoTIFF影像文件夹路径')
    parser.add_argument('--csv', required=True, help='输入CSV文件路径（包含lon, lat列）')
    parser.add_argument('--output', required=True, help='输出CSV文件路径（合并所有影像结果）')
    parser.add_argument('--scale', type=float, default=0.0001, help='缩放系数（默认0.0001）')

    args = parser.parse_args()

    extract_from_local_tiff(
        args.tif,
        args.csv,
        args.output,
        args.scale
    )


if __name__ == '__main__':
    main()
