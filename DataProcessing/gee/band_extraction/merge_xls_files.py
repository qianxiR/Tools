"""
合并多个CSV/XLS/XLSX文件
仅合并数据，不做任何额外处理
期待的列: lon, lat, depth
"""

import pandas as pd
import os
import glob

# ========== 配置参数 ==========
INPUT_FOLDER = r'E:\1代码\模型\gee\逐波段提取数据\西藏水深\1027\转换后'
OUTPUT_FILE = r'E:\1代码\模型\gee\逐波段提取数据\西藏水深\1027\合并结果.csv'

# 支持的文件格式
SUPPORTED_FORMATS = ['*.csv', '*.xls', '*.xlsx']


def read_file(file_path):
    """读取单个文件"""
    ext = os.path.splitext(file_path)[1].lower()
    name = os.path.basename(file_path)

    try:
        if ext == '.csv':
            # 尝试不同编码读取CSV
            for encoding in ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f'  OK: {name} ({len(df)} rows, encoding={encoding})')
                    return df
                except UnicodeDecodeError:
                    continue
            print(f'  FAIL: {name} - encoding error')
            return None

        elif ext in ['.xls', '.xlsx']:
            engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
            df = pd.read_excel(file_path, engine=engine)
            print(f'  OK: {name} ({len(df)} rows)')
            return df
        else:
            print(f'  SKIP: {name} - unsupported format')
            return None

    except Exception as e:
        print(f'  FAIL: {name} - {str(e)}')
        return None


def main():
    """主函数"""
    print('=' * 80)
    print('Merge Table Files')
    print('=' * 80)
    print(f'Input folder: {INPUT_FOLDER}')
    print(f'Output file: {OUTPUT_FILE}')
    print('=' * 80)

    # 检查输入文件夹
    if not os.path.exists(INPUT_FOLDER):
        print(f'\nERROR: Input folder does not exist')
        return

    # 扫描所有文件
    print('\nScanning files...')
    all_files = []
    for pattern in SUPPORTED_FORMATS:
        all_files.extend(glob.glob(os.path.join(INPUT_FOLDER, pattern)))

    all_files.sort()

    if not all_files:
        print('ERROR: No files found')
        return

    print(f'Found {len(all_files)} files\n')

    # 读取并合并所有文件
    print('Reading files...')
    df_list = []
    success = 0

    for i, file_path in enumerate(all_files, 1):
        print(f'[{i}/{len(all_files)}]', end=' ')
        df = read_file(file_path)

        if df is not None:
            # 显示列信息
            print(f'      Columns: {list(df.columns)}')
            df_list.append(df)
            success += 1

    if not df_list:
        print('\nERROR: No files successfully read')
        return

    # 合并数据（不做任何处理）
    print(f'\nMerging {len(df_list)} dataframes...')
    merged_df = pd.concat(df_list, ignore_index=True)

    print('=' * 80)
    print('Merge Summary')
    print('=' * 80)
    print(f'Success: {success}/{len(all_files)} files')
    print(f'Total rows: {len(merged_df)}')
    print(f'Total columns: {len(merged_df.columns)}')
    print(f'Columns: {list(merged_df.columns)}')

    # 保存结果
    print(f'\nSaving to: {OUTPUT_FILE}')
    try:
        # 创建输出目录
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存为CSV
        merged_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        file_size = os.path.getsize(OUTPUT_FILE) / 1024
        print(f'SUCCESS: Saved ({file_size:.2f} KB)')

        # 数据预览
        print('\nPreview (first 5 rows):')
        print(merged_df.head())

        # 显示数据范围
        if 'lon' in merged_df.columns and 'lat' in merged_df.columns:
            print(f'\nData range:')
            print(f'  lon: {merged_df["lon"].min():.6f} ~ {merged_df["lon"].max():.6f}')
            print(f'  lat: {merged_df["lat"].min():.6f} ~ {merged_df["lat"].max():.6f}')
        if 'depth' in merged_df.columns:
            print(f'  depth: {merged_df["depth"].min():.2f} ~ {merged_df["depth"].max():.2f}')

    except Exception as e:
        print(f'ERROR: Save failed - {str(e)}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nInterrupted by user')
    except Exception as e:
        print(f'\nERROR: {str(e)}')
        import traceback
        traceback.print_exc()