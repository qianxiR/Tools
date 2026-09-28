"""
批次数据合并与预处理脚本
入参:
- 临时批次CSV文件目录
方法:
- 合并所有batch临时文件
- 数据清洗与统计分析
- 数据预处理（缺失值处理、异常值检测、数据清洗）
出参:
- 合并后的完整CSV
- 预处理后的训练数据CSV
- 数据分析报告
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import warnings
import sys
warnings.filterwarnings('ignore')

# 导入配置文件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = [FIGURE_FONTSIZE]
plt.rcParams['axes.unicode_minus'] = False

# ========== 配置参数 ==========
# 使用配置文件中的参数
WORK_DIR = BASE_DIR
TEMP_FILE_PATTERN = TEMP_BATCH_PATTERN
OUTPUT_MERGED = OUTPUT_MERGED_CSV
OUTPUT_PROCESSED = OUTPUT_PROCESSED_CSV
ANALYSIS_REPORT = OUTPUT_REPORT_TXT


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


def merge_batch_files():
    """
    合并所有批次文件
    入参:
    - 无
    方法:
    - 查找所有临时batch文件（支持CSV/XLS/XLSX）
    - 按批次号排序
    - 合并为一个DataFrame
    - 去重
    出参:
    - merged_df (DataFrame): 合并后的完整数据框
    """
    print('=' * 80)
    print('[1/5] 合并批次文件')
    print('=' * 80)

    # 查找所有临时文件（包含temp_batch的文件）
    temp_files = []
    for pattern in ['*temp_batch*.csv', '*temp_batch*.xls', '*temp_batch*.xlsx']:
        temp_files.extend(glob.glob(os.path.join(WORK_DIR, pattern)))

    temp_files = sorted(set(temp_files))  # 去重并排序

    if len(temp_files) == 0:
        print('× 未找到临时批次文件')
        print('  提示: 请先运行 extract_sentinel_band_values.py 提取波段值')
        return None

    print(f'✓ 找到 {len(temp_files)} 个批次文件')

    # 读取并合并
    df_list = []
    for temp_file in temp_files:
        print(f'  读取: {os.path.basename(temp_file)}')
        df = read_table_file(temp_file)
        if df is not None:
            df_list.append(df)

    if not df_list:
        print('× 没有成功读取任何批次文件')
        return None

    merged_df = pd.concat(df_list, ignore_index=True)

    # 去重
    original_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['lon', 'lat'], keep='first')
    duplicate_count = original_count - len(merged_df)

    print(f'\n✓ 合并完成')
    print(f'  原始记录数: {original_count}')
    print(f'  去重后记录数: {len(merged_df)}')
    print(f'  重复记录数: {duplicate_count}')

    return merged_df


def data_quality_analysis(df):
    """
    数据质量分析
    入参:
    - df (DataFrame): 数据框
    方法:
    - 统计缺失值
    - 检测异常值
    - 生成描述性统计
    出参:
    - report (dict): 数据质量报告字典
    """
    print('\n' + '=' * 80)
    print('[2/5] 数据质量分析')
    print('=' * 80)
    
    report = {}
    
    # 基本信息
    print(f'\n数据集形状: {df.shape}')
    print(f'样本数量: {df.shape[0]}')
    print(f'特征数量: {df.shape[1]}')
    report['shape'] = df.shape
    
    # 列名
    print(f'\n列名: {df.columns.tolist()}')
    report['columns'] = df.columns.tolist()
    
    # 缺失值统计
    print('\n缺失值统计:')
    missing_stats = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        '缺失数量': missing_stats,
        '缺失比例(%)': missing_pct
    })
    print(missing_df[missing_df['缺失数量'] > 0])
    report['missing'] = missing_df
    
    # 波段数据统计
    band_cols = [col for col in df.columns if col.startswith('band_')]
    if band_cols:
        print(f'\n波段列: {band_cols}')
        print('\n波段数据描述性统计:')
        band_stats = df[band_cols].describe()
        print(band_stats)
        report['band_stats'] = band_stats
        
        # 检测异常值（使用IQR方法）
        print('\n异常值检测 (IQR方法):')
        outlier_counts = {}
        for col in band_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_counts[col] = len(outliers)
            if len(outliers) > 0:
                print(f'  {col}: {len(outliers)} 个异常值 ({len(outliers)/len(df)*100:.2f}%)')
        report['outliers'] = outlier_counts
    
    # 坐标范围
    print('\n坐标范围:')
    print(f'  经度: {df["lon"].min():.6f} - {df["lon"].max():.6f}')
    print(f'  纬度: {df["lat"].min():.6f} - {df["lat"].max():.6f}')
    report['coord_range'] = {
        'lon': (df["lon"].min(), df["lon"].max()),
        'lat': (df["lat"].min(), df["lat"].max())
    }
    
    return report


def preprocess_data(df):
    """
    机器学习数据预处理
    入参:
    - df (DataFrame): 原始数据框
    方法:
    - 处理缺失值（中位数填充）
    - 移除全NaN的样本
    - 特征工程（计算水体指数NDWI等）
    - 标准化（可选）
    出参:
    - processed_df (DataFrame): 预处理后的数据框
    """
    print('\n' + '=' * 80)
    print('[3/5] 数据预处理')
    print('=' * 80)
    
    df_processed = df.copy()
    original_count = len(df_processed)
    
    # 识别波段列
    band_cols = [col for col in df.columns if col.startswith('band_')]

    # 步骤1: 波段值缩放（转换为地表反射率）
    print('\n步骤1: 波段值缩放（转换为地表反射率）')
    print('  将所有波段值缩放 0.0001 倍')
    for col in band_cols:
        df_processed[col] = df_processed[col] * 0.0001
    print(f'  ✓ 已完成 {len(band_cols)} 个波段的缩放')

    # 移除波段全部为NaN的样本
    print('\n步骤2: 移除无效样本')
    df_processed = df_processed.dropna(subset=band_cols, how='all')
    removed_count = original_count - len(df_processed)
    print(f'  移除 {removed_count} 个波段全NaN的样本')
    print(f'  剩余样本: {len(df_processed)}')
    
    # 处理缺失值 - 使用中位数填充
    print('\n步骤3: 缺失值处理（中位数填充）')
    imputer = SimpleImputer(strategy='median')
    df_processed[band_cols] = imputer.fit_transform(df_processed[band_cols])
    print(f'  波段列缺失值已填充')
    
    # 处理深度缺失值
    if 'depth' in df_processed.columns:
        if df_processed['depth'].isnull().sum() > 0:
            depth_median = df_processed['depth'].median()
            df_processed['depth'].fillna(depth_median, inplace=True)
            print(f'  深度缺失值已用中位数填充: {depth_median:.2f}')
    
    # 异常值处理（温和裁剪）
    print(f'\n步骤4: 异常值处理（{OUTLIER_CLIP_UPPER*100:.1f}%分位数裁剪）')
    for col in band_cols:
        upper_limit = df_processed[col].quantile(OUTLIER_CLIP_UPPER)
        lower_limit = df_processed[col].quantile(OUTLIER_CLIP_LOWER)
        df_processed[col] = df_processed[col].clip(lower_limit, upper_limit)
    print('  ✓ 波段值裁剪完成')
    
    # 数据类型优化
    print('\n步骤5: 数据类型优化')
    for col in band_cols + ['depth']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('float32')
    print('  ✓ 波段和深度转换为float32')
    
    print(f'\n✓ 预处理完成，最终样本数: {len(df_processed)}')
    
    return df_processed


def generate_visualizations(df):
    """
    生成可视化分析图表
    入参:
    - df (DataFrame): 预处理后的数据框
    方法:
    - 绘制波段分布图
    - 绘制相关性热图
    - 绘制深度分布图
    出参:
    - 保存图表到文件
    """
    print('\n' + '=' * 80)
    print('[3/5] 生成可视化图表')
    print('=' * 80)
    
    band_cols = [col for col in df.columns if col.startswith('band_')]
    
    # 1. 相关性热图
    print('\n生成图表1: 波段相关性热图')
    feature_cols = band_cols.copy()
    
    plt.figure(figsize=(14, 12))
    corr_matrix = df[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('波段相关性热图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = os.path.join(WORK_DIR, '波段相关性热图.png')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f'  ✓ 已保存: {output_path}')
    


def save_analysis_report(df, report):
    """
    保存数据分析报告
    入参:
    - df (DataFrame): 预处理后的数据框
    - report (dict): 分析报告字典
    方法:
    - 生成文本格式的详细分析报告
    出参:
    - 保存报告文件
    """
    print('\n' + '=' * 80)
    print('[4/5] 保存分析报告')
    print('=' * 80)
    
    output_path = os.path.join(WORK_DIR, ANALYSIS_REPORT)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('湖库波段值数据分析报告\n')
        f.write('=' * 80 + '\n\n')
        
        f.write('1. 数据集基本信息\n')
        f.write('-' * 80 + '\n')
        f.write(f'样本数量: {report["shape"][0]}\n')
        f.write(f'特征数量: {report["shape"][1]}\n')
        f.write(f'列名: {", ".join(report["columns"])}\n\n')
        
        f.write('2. 缺失值统计\n')
        f.write('-' * 80 + '\n')
        missing_df = report['missing'][report['missing']['缺失数量'] > 0]
        if len(missing_df) > 0:
            f.write(missing_df.to_string() + '\n\n')
        else:
            f.write('无缺失值\n\n')
        
        if 'band_stats' in report:
            f.write('3. 波段数据统计\n')
            f.write('-' * 80 + '\n')
            f.write(report['band_stats'].to_string() + '\n\n')
        
        if 'outliers' in report:
            f.write('4. 异常值统计 (IQR方法)\n')
            f.write('-' * 80 + '\n')
            for band, count in report['outliers'].items():
                f.write(f'{band}: {count} 个异常值\n')
            f.write('\n')
        
        f.write('5. 坐标范围\n')
        f.write('-' * 80 + '\n')
        lon_range = report['coord_range']['lon']
        lat_range = report['coord_range']['lat']
        f.write(f'经度范围: {lon_range[0]:.6f} - {lon_range[1]:.6f}\n')
        f.write(f'纬度范围: {lat_range[0]:.6f} - {lat_range[1]:.6f}\n\n')
        
        f.write('6. 数据质量评估\n')
        f.write('-' * 80 + '\n')
        f.write('✓ 数据已完成预处理，可用于机器学习建模\n')
        f.write('✓ 建议使用的特征: 所有波段值\n')
        f.write('✓ 目标变量: depth (水深)\n\n')
    
    print(f'✓ 分析报告已保存: {output_path}')


def main():
    """
    主函数
    入参:
    - 无
    方法:
    - 合并批次文件
    - 数据质量分析
    - 数据预处理
    - 生成可视化
    - 保存报告
    出参:
    - 无，结果保存到文件
    """
    print('\n' + '=' * 80)
    print('批次数据合并与预处理脚本')
    print('=' * 80)
    
    # 1. 合并批次文件
    merged_df = merge_batch_files()
    if merged_df is None:
        print('× 未找到批次文件，程序退出')
        return
    
    # 2. 数据质量分析
    report = data_quality_analysis(merged_df)
    
    # 3. 数据预处理
    processed_df = preprocess_data(merged_df)

    # 保存合并后的完整数据（原始波段值，经纬度保留6位小数）
    output_path = os.path.join(WORK_DIR, OUTPUT_MERGED)
    merged_df['lon'] = merged_df['lon'].round(6)
    merged_df['lat'] = merged_df['lat'].round(6)
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f'\n✓ 合并数据已保存（原始波段值）: {output_path}')

    # 保存缩放后的完整数据（地表反射率，经纬度保留6位小数）
    output_scaled_path = os.path.join(WORK_DIR, '湖库波段值_缩放后.csv')
    scaled_df = merged_df.copy()
    band_cols = [col for col in scaled_df.columns if col.startswith('band_')]
    for col in band_cols:
        scaled_df[col] = scaled_df[col] * 0.0001
    scaled_df.to_csv(output_scaled_path, index=False, encoding='utf-8')
    print(f'✓ 缩放后数据已保存（地表反射率）: {output_scaled_path}')

    # 保存预处理后的数据（经纬度保留6位小数）
    output_path = os.path.join(WORK_DIR, OUTPUT_PROCESSED)
    processed_df['lon'] = processed_df['lon'].round(6)
    processed_df['lat'] = processed_df['lat'].round(6)
    processed_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f'\n✓ 预处理数据已保存: {output_path}')
    
    # 4. 生成可视化
    generate_visualizations(processed_df)
    
    # 5. 保存分析报告
    save_analysis_report(processed_df, report)
    
    # 6. 清理临时文件
    print('\n' + '=' * 80)
    print('[5/5] 清理临时文件')
    print('=' * 80)

    # 查找所有临时文件（支持多种格式）
    temp_files = []
    for pattern in ['*temp_batch*.csv', '*temp_batch*.xls', '*temp_batch*.xlsx']:
        temp_files.extend(glob.glob(os.path.join(WORK_DIR, pattern)))

    temp_files = list(set(temp_files))  # 去重

    if temp_files:
        print(f'找到 {len(temp_files)} 个临时批次文件')
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f'  ✓ 已删除: {os.path.basename(temp_file)}')
            except Exception as e:
                print(f'  ✗ 删除失败: {os.path.basename(temp_file)} - {str(e)}')
        print(f'\n✓ 所有临时文件已清理')
    else:
        print('  ⚠ 未找到临时批次文件')
    
    print('\n' + '=' * 80)
    print('所有处理完成！')
    print('=' * 80)
    print(f'✓ 合并文件（原始波段值）: {OUTPUT_MERGED}')
    print(f'✓ 缩放文件（地表反射率）: 湖库波段值_缩放后.csv')
    print(f'✓ 预处理文件: {OUTPUT_PROCESSED}')
    print(f'✓ 分析报告: {ANALYSIS_REPORT}')
    print(f'✓ 可视化图表: 波段相关性热图.png')
    print('=' * 80)


if __name__ == '__main__':
    main()

