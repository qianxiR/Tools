import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from pyproj import Transformer
from tqdm import tqdm
import re


def geotiff_to_csv(tif_path: str, output_csv_path: str, scale_factor: float = 0.0001, chunk_size: int = 2048) -> None:
    """
    从多波段 GeoTIFF 逐像素提取经纬度与主波段(B2、B3、B4、B8)，乘以 scale_factor 并保留四位小数。
    
    使用分块处理以避免大影像的内存溢出问题。
    
    入参:
    - tif_path: str，输入GeoTIFF文件路径
    - output_csv_path: str，输出CSV文件路径
    - scale_factor: float，缩放系数（默认0.0001）
    - chunk_size: int，分块大小（像素），默认2048
    
    方法:
    - 逐像素采样，不遗漏任何像素
    - 对于大影像（超过1000万像素），使用分块处理
    - 保留所有像素，包括无效像元
    
    出参:
    - None。副作用：生成CSV文件
    """
    print(f"[>] 开始处理: {os.path.basename(tif_path)}")

    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        band_count = src.count
        transform = src.transform
        crs = src.crs

        print(f"   尺寸: {width}x{height} | 波段: {band_count} | 坐标系: {crs}")
        
        total_pixels = height * width
        print(f"   影像总像素数: {total_pixels:,} (逐像素采样模式)")
        
        use_chunked = total_pixels > 10_000_000  # 如果总像素数超过1000万，使用分块处理
        
        if use_chunked:
            print(f"   检测到大影像，使用分块处理 (块大小: {chunk_size}x{chunk_size})")
            df = _process_chunked(src, scale_factor, chunk_size)
        else:
            print(f"   使用标准处理模式")
            df = _process_standard(src, scale_factor)

        # 过滤0值和NaN值
        original_count = len(df)
        band_cols = [c for c in df.columns if c.startswith('band_')]
        
        if len(band_cols) > 0:
            # 过滤所有波段都为NaN的行
            df = df.dropna(subset=band_cols, how='all')
            after_dropna_count = len(df)
            
            # 过滤所有波段都为0或NaN的行
            # 保留至少有一个波段值大于0的行（排除NaN和0）
            # 使用fillna(0)将NaN替换为0，然后检查是否所有值都<=0
            band_data = df[band_cols].fillna(0)
            df = df[(band_data > 0).any(axis=1)]
            after_filter_count = len(df)
            
            print(f"   过滤统计：")
            print(f"     原始像素数: {original_count:,}")
            if after_dropna_count < original_count:
                print(f"     过滤全NaN像素: {original_count - after_dropna_count:,} 个")
            if after_filter_count < after_dropna_count:
                print(f"     过滤全0像素: {after_dropna_count - after_filter_count:,} 个")
            print(f"     有效像素数: {after_filter_count:,}")
            
            # 所有波段四位小数格式化（防止浮点误差）
            df[band_cols] = df[band_cols].round(4)
        else:
            print(f"   未检测到波段列，保留所有 {original_count:,} 个像素")

        # 保存 CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        row_count = len(df)
        print(f"   正在保存CSV文件（{row_count:,} 行数据，请耐心等待）...")

        # 修复：将浮点列转换为float64，避免Buffer dtype mismatch错误
        # pandas的to_csv内部C扩展期望float64类型
        float_cols = df.select_dtypes(include=['float32', 'float16']).columns
        if len(float_cols) > 0:
            df = df.copy()
            df[float_cols] = df[float_cols].astype(np.float64)

        # 对于大文件，使用分块写入以提高性能和显示进度
        if row_count > 1_000_000:  # 超过100万行使用分块写入
            _save_csv_chunked(df, output_csv_path)
        else:
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig', float_format='%.4f')
        
        print(f"   CSV文件保存完成: {output_csv_path}")

        # 汇总输出
        print(f"   处理完成: {os.path.basename(tif_path)}")
        print(f"     总像素数: {len(df):,}")
        print(f"     缩放系数: {scale_factor} (已保留四位小数)")
        print(f"     输出文件: {output_csv_path}\n")


def _detect_band_mapping(src):
    """
    智能检测波段映射关系
    
    入参:
    - src: rasterio.DatasetReader，打开的影像文件
    
    方法:
    - 从波段描述或标签中读取波段名称
    - 解析Sentinel-2波段编号
    - 根据波段数量和位置推断映射关系
    
    出参:
    - dict: {band_index: 'band_name'} 例如 {0: 'band_2', 1: 'band_3', 2: 'band_4', 3: 'band_8'}
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
        band_idx = i  # 0-based索引
        band_name = band_names[i] if i < len(band_names) else None
        
        # 尝试从名称中提取波段编号（如 "B2", "B3", "B4", "B8"）
        sentinel_band_num = None
        if band_name:
            # 匹配 B2, B3, B4, B8, B11, B12 等
            match = re.search(r'B(\d+)', str(band_name).upper())
            if match:
                sentinel_band_num = int(match.group(1))
        
        # 如果没有找到波段名称，根据波段数量和位置推断
        if sentinel_band_num is None:
            # 对于4个波段的情况，假设是 B2, B3, B4, B8
            if band_count == 4:
                if band_idx == 0:
                    sentinel_band_num = 2  # B2
                elif band_idx == 1:
                    sentinel_band_num = 3  # B3
                elif band_idx == 2:
                    sentinel_band_num = 4  # B4
                elif band_idx == 3:
                    sentinel_band_num = 8  # B8
            # 对于标准 Sentinel-2 多波段情况，假设按顺序排列
            elif band_count >= 11:
                # 标准 Sentinel-2 波段顺序：B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
                standard_order = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
                if band_idx < len(standard_order):
                    sentinel_band_num = standard_order[band_idx]
                else:
                    sentinel_band_num = band_idx + 2  # 默认假设从B2开始
            else:
                # 其他情况，假设从B2开始顺序排列
                sentinel_band_num = band_idx + 2
        
        # 只映射 Sentinel-2 主波段（B2、B3、B4、B8）
        if sentinel_band_num in [2, 3, 4, 8]:
            band_mapping[band_idx] = f'band_{sentinel_band_num}'
    
    return band_mapping


def _process_standard(src, scale_factor: float) -> pd.DataFrame:
    """
    标准处理模式：适用于小到中等大小的影像，逐像素采样
    
    入参:
    - src: rasterio.DatasetReader，打开的影像文件
    - scale_factor: float，缩放系数
    
    方法:
    - 一次性读取所有波段到内存
    - 逐像素采样所有像素
    - 提取每个像素的所有波段值
    
    出参:
    - pd.DataFrame，包含所有像素的坐标和波段值
    """
    height, width = src.height, src.width
    band_count = src.count
    transform = src.transform
    crs = src.crs
    
    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)

    # 检测波段映射关系
    band_mapping = _detect_band_mapping(src)
    if band_mapping:
        print(f"   检测到波段映射: {band_mapping}")
    else:
        print(f"  ️ 无法检测波段映射，使用默认映射")

    # 读取所有波段（显示进度）
    bands_data = []
    for i in tqdm(range(1, band_count + 1), desc="   读取波段", unit="band"):
        try:
            masked_band = src.read(i, masked=True)
            # 确保数据类型正确转换
            if masked_band.dtype == np.object_:
                # 如果是object类型，尝试转换为float
                band_data = np.array(masked_band.filled(np.nan), dtype=np.float32)
            else:
                # 先转换为float，再填充NaN
                band_data = masked_band.astype(np.float32).filled(np.nan)
            bands_data.append(band_data)
        except Exception as e:
            print(f"  ️ 警告: 波段 {i} 读取异常: {e}")
            # 创建一个全NaN的数组作为备用
            band_data = np.full((height, width), np.nan, dtype=np.float32)
            bands_data.append(band_data)

    # 逐像素采样：包含所有像素（0到height-1，0到width-1）
    rows = np.arange(0, height, dtype=np.int32)
    cols = np.arange(0, width, dtype=np.int32)
    print(f"   逐像素采样：将采样 {height * width:,} 个像素")
    
    row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
    row_coords, col_coords = row_grid.flatten(), col_grid.flatten()
    
    # 验证采样数量
    sampled_count = len(row_coords)
    expected_count = height * width
    if sampled_count != expected_count:
        print(f"  ️ 警告：采样数量不匹配！期望 {expected_count:,}，实际 {sampled_count:,}")
    else:
        print(f"   采样验证通过：{sampled_count:,} 个像素")

    # 经纬度
    x_coords, y_coords = xy(transform, row_coords, col_coords)
    lon_coords, lat_coords = transformer.transform(x_coords, y_coords)

    # 提取波段值
    band_values = [bands_data[i][row_coords, col_coords] for i in range(band_count)]

    # 组织数据
    data_dict = {
        'lon': lon_coords,
        'lat': lat_coords
    }

    # 使用检测到的波段映射
    if band_mapping:
        for band_idx, col_name in band_mapping.items():
            if band_idx < len(band_values):
                band_data = np.array(band_values[band_idx], dtype=np.float32)
                scaled = band_data * scale_factor
                data_dict[col_name] = np.round(scaled, 4)
    else:
        # 如果没有检测到映射，使用默认映射（只提取B2、B3、B4、B8）
        default_mapping = {
            2: 'band_2', 3: 'band_3', 4: 'band_4', 8: 'band_8'
        }
        for idx, col in default_mapping.items():
            if idx <= band_count:
                band_data = np.array(band_values[idx - 1], dtype=np.float32)
                scaled = band_data * scale_factor
                data_dict[col] = np.round(scaled, 4)
            else:
                data_dict[col] = np.full_like(band_values[0], np.nan, dtype=np.float32)

    return pd.DataFrame(data_dict)


def _process_chunked(src, scale_factor: float, chunk_size: int) -> pd.DataFrame:
    """
    分块处理模式：适用于大影像，逐块读取和处理，逐像素采样
    
    入参:
    - src: rasterio.DatasetReader，打开的影像文件
    - scale_factor: float，缩放系数
    - chunk_size: int，分块大小（像素）
    
    方法:
    - 将影像分成chunk_size x chunk_size的块
    - 逐块读取和处理
    - 每个块内逐像素采样
    
    出参:
    - pd.DataFrame，包含所有像素的坐标和波段值
    """
    height, width = src.height, src.width
    band_count = src.count
    transform = src.transform
    crs = src.crs
    
    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)

    # 检测波段映射关系
    band_mapping = _detect_band_mapping(src)
    if band_mapping:
        print(f"   检测到波段映射: {band_mapping}")
    else:
        print(f"  ️ 无法检测波段映射，使用默认映射")
    
    # 计算需要处理的块数
    num_chunks_y = (height + chunk_size - 1) // chunk_size
    num_chunks_x = (width + chunk_size - 1) // chunk_size
    total_chunks = num_chunks_y * num_chunks_x
    
    all_data = []
    
    # 逐块处理
    with tqdm(total=total_chunks, desc="   处理分块", unit="chunk") as pbar:
        for chunk_y in range(num_chunks_y):
            for chunk_x in range(num_chunks_x):
                # 计算当前块的窗口
                row_start = chunk_y * chunk_size
                col_start = chunk_x * chunk_size
                row_end = min(row_start + chunk_size, height)
                col_end = min(col_start + chunk_size, width)
                
                window = Window.from_slices(
                    (row_start, row_end),
                    (col_start, col_end)
                )
                
                # 读取当前块的所有波段
                try:
                    chunk_bands = []
                    for i in range(1, band_count + 1):
                        masked_band = src.read(i, window=window, masked=True)
                        if masked_band.dtype == np.object_:
                            band_data = np.array(masked_band.filled(np.nan), dtype=np.float32)
                        else:
                            band_data = masked_band.astype(np.float32).filled(np.nan)
                        chunk_bands.append(band_data)
                    
                    # 获取当前块的变换矩阵
                    chunk_transform = rasterio.windows.transform(window, transform)

                    # 逐像素采样：包含块内所有像素
                    chunk_height, chunk_width = chunk_bands[0].shape
                    rows = np.arange(0, chunk_height, dtype=np.int32)
                    cols = np.arange(0, chunk_width, dtype=np.int32)
                    
                    row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
                    row_coords, col_coords = row_grid.flatten(), col_grid.flatten()
                    
                    # 验证块内采样数量
                    expected_chunk_pixels = chunk_height * chunk_width
                    actual_chunk_pixels = len(row_coords)
                    if actual_chunk_pixels != expected_chunk_pixels:
                        print(f"  ️ 警告：块 ({chunk_y}, {chunk_x}) 采样数量不匹配！期望 {expected_chunk_pixels:,}，实际 {actual_chunk_pixels:,}")
                    
                    # 经纬度
                    x_coords, y_coords = xy(chunk_transform, row_coords, col_coords)
                    lon_coords, lat_coords = transformer.transform(x_coords, y_coords)

                    # 提取波段值
                    chunk_data = {
                        'lon': lon_coords,
                        'lat': lat_coords
                    }
                    
                    # 提取每个波段的值（使用检测到的映射）
                    if band_mapping:
                        for band_idx, col_name in band_mapping.items():
                            if band_idx < len(chunk_bands):
                                band_data = chunk_bands[band_idx][row_coords, col_coords]
                                band_data = np.array(band_data, dtype=np.float32)
                                scaled = band_data * scale_factor
                                chunk_data[col_name] = np.round(scaled, 4)
                    else:
                        # 如果没有检测到映射，使用默认映射（向后兼容）
                        default_mapping = {
                            2: 'band_2', 3: 'band_3', 4: 'band_4', 5: 'band_5',
                            6: 'band_6', 7: 'band_7', 8: 'band_8', 9: 'band_9',
                            11: 'band_11', 12: 'band_12'
                        }
                        for idx, col in default_mapping.items():
                            if idx <= band_count:
                                band_data = chunk_bands[idx - 1][row_coords, col_coords]
                                band_data = np.array(band_data, dtype=np.float32)
                                scaled = band_data * scale_factor
                                chunk_data[col] = np.round(scaled, 4)
                            else:
                                chunk_data[col] = np.full(len(row_coords), np.nan, dtype=np.float32)
                    
                    # 转换为DataFrame并添加到列表
                    chunk_df = pd.DataFrame(chunk_data)
                    all_data.append(chunk_df)
                    
                except Exception as e:
                    print(f"  ️ 警告: 处理块 ({chunk_y}, {chunk_x}) 时出错: {e}")
                    continue
                
                pbar.update(1)
    
    # 合并所有块的数据
    if not all_data:
        # 如果没有数据，返回空DataFrame
        band_cols = [f'band_{i}' for i in [2, 3, 4, 8]]
        return pd.DataFrame(columns=['lon', 'lat'] + band_cols)
    
    df = pd.concat(all_data, ignore_index=True)
    
    # 验证总像素数是否正确
    expected_total_pixels = height * width
    actual_total_pixels = len(df)
    if actual_total_pixels != expected_total_pixels:
        print(f"  ️ 警告：总采样数量不匹配！期望 {expected_total_pixels:,}，实际 {actual_total_pixels:,}")
        print(f"     可能原因：某些块处理失败")
    else:
        print(f"   总采样验证通过：{actual_total_pixels:,} 个像素（逐像素采样完成）")
    
    return df


def _save_csv_chunked(df: pd.DataFrame, output_csv_path: str, chunk_size: int = 1_000_000) -> None:
    """
    分块保存CSV文件，适用于大文件，显示进度

    入参:
    - df: pd.DataFrame，要保存的数据
    - output_csv_path: str，输出CSV文件路径
    - chunk_size: int，每块的行数（默认100万行）

    方法:
    - 将DataFrame分成多个块
    - 逐块写入CSV文件
    - 显示进度条
    - 修复：将float32/float16转换为float64，避免Buffer dtype mismatch错误

    出参:
    - None。副作用：生成CSV文件
    """
    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    # 修复：在分块前将浮点列转换为float64
    # pandas的to_csv内部C扩展期望float64类型
    float_cols = df.select_dtypes(include=['float32', 'float16']).columns
    if len(float_cols) > 0:
        df = df.copy()
        df[float_cols] = df[float_cols].astype(np.float64)

    # 使用追加模式，第一块写入表头
    header = True

    with tqdm(total=num_chunks, desc="   写入CSV分块", unit="chunk") as pbar:
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx]

            # 将DataFrame转换为CSV字符串（不指定文件路径，返回字符串）
            csv_string = chunk_df.to_csv(
                index=False,
                float_format='%.4f',
                header=header
            )
            
            # 写入文件（第一块用'w'模式，后续用'a'模式）
            file_mode = 'w' if header else 'a'
            with open(output_csv_path, file_mode, encoding='utf-8-sig', newline='') as f:
                f.write(csv_string)
            
            # 后续块不写入表头
            header = False
            
            pbar.update(1)

