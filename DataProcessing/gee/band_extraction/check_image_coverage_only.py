"""
影像覆盖率检查脚本（仅检查，不提取波段值）
入参:
- input_csv: 包含坐标的CSV文件路径（支持WGS84经纬度或CGCS2000投影坐标）
方法:
- 读取点数据（自动识别坐标格式）
- 按批次划分区域
- 检查每个批次区域的Sentinel-2影像覆盖情况
- 生成覆盖率分析报告
出参:
- 覆盖率汇总CSV文件
- 可视化覆盖率分布图
"""

import ee
import pandas as pd
import numpy as np
from pyproj import Transformer
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 配置参数 ==========
INPUT_CSV = r'E:\1代码\模型\gee\逐波段提取数据\西藏水深\1027\转换后\合并结果_精简版.csv'
OUTPUT_REPORT = r'E:\1代码\模型\gee\逐波段提取数据\西藏水深\1027\转换后\影像覆盖率检查报告.csv'
BATCH_SIZE = 10000  # 每批检查的点数
START_DATE = '2025-06-01'
END_DATE = '2025-07-31'
CLOUD_THRESHOLDS = [10, 20, 30, 50, 80]  # 检查多个云覆盖阈值

# ========== 初始化GEE ==========
"""
初始化Google Earth Engine
"""
PROJECT_ID = 'applied-pipe-453411-k9'

try:
    ee.Initialize(project=PROJECT_ID)
    print(f'✓ GEE已成功初始化 (Project: {PROJECT_ID})')
except Exception as e:
    print(f'× GEE初始化失败: {str(e)}')
    print('请先运行认证脚本: python 认证/python_认证.py')
    sys.exit(1)

# ========== 坐标转换器 ==========
"""
创建CGCS2000 → WGS84坐标转换器
"""
transformer = Transformer.from_crs("EPSG:4549", "EPSG:4326", always_xy=True)
print('✓ 坐标转换器已创建: EPSG:4549 → EPSG:4326')


def convert_coordinates(northing, easting):
    """
    坐标转换
    入参:
    - northing (float): CGCS2000北向坐标
    - easting (float): CGCS2000东向坐标
    方法:
    - 使用pyproj进行坐标转换
    出参:
    - lon, lat: WGS84经纬度
    """
    lon, lat = transformer.transform(easting, northing)
    return lon, lat


def check_batch_coverage(points_data, batch_idx):
    """
    检查一个批次的影像覆盖情况
    入参:
    - points_data (list): 点数据列表 [(lon, lat), ...]
    - batch_idx (int): 批次编号
    方法:
    - 创建区域边界
    - 查询Sentinel-2影像
    - 统计不同云覆盖阈值下的影像数量
    - 分析影像质量
    出参:
    - coverage_info (dict): 覆盖率信息字典
    """
    try:
        # 计算区域范围
        lons = [p[0] for p in points_data]
        lats = [p[1] for p in points_data]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # 创建边界矩形
        bounds = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        
        print(f'  区域范围: 经度[{min_lon:.6f}, {max_lon:.6f}], 纬度[{min_lat:.6f}, {max_lat:.6f}]')
        
        # 获取该区域的所有影像
        s2_all = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(bounds) \
            .filterDate(START_DATE, END_DATE)
        
        # 统计总影像数
        total_count = s2_all.size().getInfo()
        
        if total_count == 0:
            print(f'  ✗ 该区域没有任何Sentinel-2影像')
            return {
                'batch_idx': batch_idx,
                'point_count': len(points_data),
                'min_lon': min_lon,
                'max_lon': max_lon,
                'min_lat': min_lat,
                'max_lat': max_lat,
                'total_images': 0,
                'has_coverage': False
            }
        
        print(f'  ✓ 找到 {total_count} 幅影像')
        
        # 统计不同云覆盖阈值下的影像数量
        coverage_info = {
            'batch_idx': batch_idx,
            'point_count': len(points_data),
            'min_lon': min_lon,
            'max_lon': max_lon,
            'min_lat': min_lat,
            'max_lat': max_lat,
            'total_images': total_count,
            'has_coverage': True
        }
        
        print(f'  影像质量分析:')
        for threshold in CLOUD_THRESHOLDS:
            filtered = s2_all.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', threshold))
            count = filtered.size().getInfo()
            coverage_info[f'cloud_lt_{threshold}'] = count
            print(f'    云覆盖<{threshold}%: {count} 幅 ({count/total_count*100:.1f}%)')
        
        # 获取云覆盖率列表并计算统计量
        cloud_list = s2_all.aggregate_array('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        
        coverage_info['min_cloud'] = min(cloud_list)
        coverage_info['max_cloud'] = max(cloud_list)
        coverage_info['mean_cloud'] = np.mean(cloud_list)
        coverage_info['median_cloud'] = np.median(cloud_list)
        
        print(f'  云覆盖率统计:')
        print(f'    最小值: {coverage_info["min_cloud"]:.1f}%')
        print(f'    最大值: {coverage_info["max_cloud"]:.1f}%')
        print(f'    平均值: {coverage_info["mean_cloud"]:.1f}%')
        print(f'    中位数: {coverage_info["median_cloud"]:.1f}%')
        
        # 评估数据可用性
        if coverage_info['cloud_lt_20'] > 0:
            print(f'  ✓ 数据质量: 优秀 (有{coverage_info["cloud_lt_20"]}幅低云影像)')
        elif coverage_info['cloud_lt_30'] > 0:
            print(f'  ⚠ 数据质量: 良好 (建议使用云覆盖<30%的影像)')
        elif coverage_info['cloud_lt_50'] > 0:
            print(f'  ⚠ 数据质量: 中等 (建议使用云覆盖<50%的影像)')
        else:
            print(f'  ✗ 数据质量: 较差 (所有影像云覆盖都>50%)')
        
        return coverage_info
        
    except Exception as e:
        print(f'  ✗ 检查失败: {str(e)}')
        return {
            'batch_idx': batch_idx,
            'point_count': len(points_data),
            'has_coverage': False,
            'error': str(e)
        }


def generate_coverage_visualization(coverage_df):
    """
    生成覆盖率可视化图表
    入参:
    - coverage_df (DataFrame): 覆盖率数据框
    方法:
    - 绘制影像数量分布图
    - 绘制云覆盖率统计图
    - 绘制空间分布地图
    出参:
    - 保存图表到文件
    """
    print('\n生成可视化图表...')
    
    # 过滤有覆盖的批次
    df_valid = coverage_df[coverage_df['has_coverage'] == True].copy()
    
    if len(df_valid) == 0:
        print('  ⚠ 没有有效数据，跳过可视化')
        return
    
    # 创建图表
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 影像数量分布
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(df_valid['batch_idx'], df_valid['total_images'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('批次编号', fontsize=11)
    ax1.set_ylabel('影像数量', fontsize=11)
    ax1.set_title('各批次总影像数量', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 不同云覆盖阈值下的影像数量
    ax2 = plt.subplot(2, 3, 2)
    thresholds_to_plot = [10, 20, 30, 50]
    for threshold in thresholds_to_plot:
        col = f'cloud_lt_{threshold}'
        if col in df_valid.columns:
            ax2.plot(df_valid['batch_idx'], df_valid[col], 
                    marker='o', label=f'<{threshold}%', linewidth=2, markersize=4)
    ax2.set_xlabel('批次编号', fontsize=11)
    ax2.set_ylabel('影像数量', fontsize=11)
    ax2.set_title('不同云覆盖阈值下的可用影像', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 平均云覆盖率分布
    ax3 = plt.subplot(2, 3, 3)
    if 'mean_cloud' in df_valid.columns:
        colors = ['green' if x < 20 else 'orange' if x < 50 else 'red' 
                 for x in df_valid['mean_cloud']]
        ax3.bar(df_valid['batch_idx'], df_valid['mean_cloud'], color=colors, alpha=0.7)
        ax3.axhline(y=20, color='green', linestyle='--', linewidth=2, label='优秀(<20%)')
        ax3.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='可用(<50%)')
        ax3.set_xlabel('批次编号', fontsize=11)
        ax3.set_ylabel('平均云覆盖率 (%)', fontsize=11)
        ax3.set_title('各批次平均云覆盖率', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 云覆盖率统计箱线图
    ax4 = plt.subplot(2, 3, 4)
    if all(col in df_valid.columns for col in ['min_cloud', 'median_cloud', 'max_cloud']):
        cloud_data = [df_valid['min_cloud'], df_valid['median_cloud'], df_valid['max_cloud']]
        ax4.boxplot(cloud_data, labels=['最小值', '中位数', '最大值'])
        ax4.set_ylabel('云覆盖率 (%)', fontsize=11)
        ax4.set_title('云覆盖率分布箱线图', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 数据质量分类饼图
    ax5 = plt.subplot(2, 3, 5)
    quality_counts = {
        '优秀(<20%)': sum(df_valid['cloud_lt_20'] > 0),
        '良好(20-30%)': sum((df_valid['cloud_lt_30'] > 0) & (df_valid['cloud_lt_20'] == 0)),
        '中等(30-50%)': sum((df_valid['cloud_lt_50'] > 0) & (df_valid['cloud_lt_30'] == 0)),
        '较差(>50%)': sum(df_valid['cloud_lt_50'] == 0)
    }
    colors_pie = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    ax5.pie(quality_counts.values(), labels=quality_counts.keys(), autopct='%1.1f%%',
           colors=colors_pie, startangle=90)
    ax5.set_title('批次数据质量分布', fontsize=12, fontweight='bold')
    
    # 6. 空间分布地图
    ax6 = plt.subplot(2, 3, 6)
    if all(col in df_valid.columns for col in ['min_lon', 'max_lon', 'min_lat', 'max_lat']):
        # 计算每个批次的中心点
        center_lons = (df_valid['min_lon'] + df_valid['max_lon']) / 2
        center_lats = (df_valid['min_lat'] + df_valid['max_lat']) / 2
        
        # 根据云覆盖<20%的影像数量着色
        scatter = ax6.scatter(center_lons, center_lats, 
                            c=df_valid['cloud_lt_20'], 
                            s=100, alpha=0.6, cmap='RdYlGn',
                            edgecolors='black', linewidth=0.5)
        
        # 绘制边界框
        for idx, row in df_valid.iterrows():
            rect_lons = [row['min_lon'], row['max_lon'], row['max_lon'], row['min_lon'], row['min_lon']]
            rect_lats = [row['min_lat'], row['min_lat'], row['max_lat'], row['max_lat'], row['min_lat']]
            ax6.plot(rect_lons, rect_lats, 'gray', alpha=0.3, linewidth=0.5)
        
        plt.colorbar(scatter, ax=ax6, label='云覆盖<20%影像数')
        ax6.set_xlabel('经度', fontsize=11)
        ax6.set_ylabel('纬度', fontsize=11)
        ax6.set_title('批次空间分布与影像质量', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_REPORT.replace('.csv', '_可视化.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ 已保存: {output_path}')


def main():
    """
    主函数
    """
    print('=' * 80)
    print('Sentinel-2 影像覆盖率检查脚本')
    print('=' * 80)
    print(f'输入文件: {INPUT_CSV}')
    print(f'时间范围: {START_DATE} 至 {END_DATE}')
    print(f'批次大小: {BATCH_SIZE}个点/批')
    print(f'检查阈值: {CLOUD_THRESHOLDS}%')
    print('=' * 80)
    
    # 读取CSV
    print('\n[1/3] 读取输入CSV文件...')
    try:
        try:
            df = pd.read_csv(INPUT_CSV, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(INPUT_CSV, encoding='gbk')
        
        print(f'✓ 成功读取 {len(df)} 个点')
        print(f'  列名: {df.columns.tolist()}')
    except Exception as e:
        print(f'✗ 读取CSV失败: {str(e)}')
        sys.exit(1)
    
    # 分批检查
    print(f'\n[2/3] 开始批次检查...')
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    all_coverage_info = []
    
    for i in range(0, len(df), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE + 1
        df_batch = df.iloc[i:i+BATCH_SIZE]
        
        print(f'\n========== 批次 {batch_idx}/{total_batches} (共{len(df_batch)}个点) ==========')

        # 读取坐标（自动识别格式）
        print(f'  读取坐标...', end=' ')
        points_data = []

        # 检查列名，判断坐标格式
        if 'lon' in df_batch.columns and 'lat' in df_batch.columns:
            # WGS84经纬度格式
            print('(WGS84格式) ', end='')
            for idx, row in df_batch.iterrows():
                lon = row['lon']
                lat = row['lat']
                points_data.append((lon, lat))
        elif 'N' in df_batch.columns and 'E' in df_batch.columns:
            # CGCS2000投影坐标格式
            print('(CGCS2000格式，需转换) ', end='')
            for idx, row in df_batch.iterrows():
                northing = row['N']
                easting = row['E']
                lon, lat = convert_coordinates(northing, easting)
                points_data.append((lon, lat))
        else:
            print('✗')
            print(f'  错误: 无法识别坐标列，需要 (lon, lat) 或 (N, E)')
            print(f'  当前列名: {df_batch.columns.tolist()}')
            continue

        print('✓')
        
        # 检查覆盖率
        coverage_info = check_batch_coverage(points_data, batch_idx)
        all_coverage_info.append(coverage_info)
    
    # 保存结果
    print(f'\n[3/3] 保存结果...')
    coverage_df = pd.DataFrame(all_coverage_info)
    coverage_df.to_csv(OUTPUT_REPORT, index=False, encoding='utf-8')
    print(f'✓ 覆盖率报告已保存: {OUTPUT_REPORT}')
    
    # 生成可视化
    generate_coverage_visualization(coverage_df)
    
    # 汇总统计
    print('\n' + '=' * 80)
    print('覆盖率检查汇总')
    print('=' * 80)
    
    total_batches_checked = len(coverage_df)
    batches_with_coverage = sum(coverage_df['has_coverage'])
    batches_without_coverage = total_batches_checked - batches_with_coverage
    
    print(f'检查批次总数: {total_batches_checked}')
    print(f'有影像覆盖: {batches_with_coverage} ({batches_with_coverage/total_batches_checked*100:.1f}%)')
    print(f'无影像覆盖: {batches_without_coverage} ({batches_without_coverage/total_batches_checked*100:.1f}%)')
    
    # 有效批次的统计
    df_valid = coverage_df[coverage_df['has_coverage'] == True]
    
    if len(df_valid) > 0:
        print(f'\n影像质量统计 (基于{len(df_valid)}个有效批次):')
        print(f'  平均总影像数: {df_valid["total_images"].mean():.1f}')
        
        for threshold in CLOUD_THRESHOLDS:
            col = f'cloud_lt_{threshold}'
            if col in df_valid.columns:
                avg_count = df_valid[col].mean()
                batches_with_good = sum(df_valid[col] > 0)
                print(f'  云覆盖<{threshold}%: 平均{avg_count:.1f}幅, {batches_with_good}个批次有可用影像')
        
        if 'mean_cloud' in df_valid.columns:
            print(f'\n云覆盖率统计:')
            print(f'  最低平均云覆盖: {df_valid["mean_cloud"].min():.1f}%')
            print(f'  最高平均云覆盖: {df_valid["mean_cloud"].max():.1f}%')
            print(f'  总体平均云覆盖: {df_valid["mean_cloud"].mean():.1f}%')
        
        # 质量评估
        excellent = sum(df_valid['cloud_lt_20'] > 0)
        good = sum((df_valid['cloud_lt_30'] > 0) & (df_valid['cloud_lt_20'] == 0))
        medium = sum((df_valid['cloud_lt_50'] > 0) & (df_valid['cloud_lt_30'] == 0))
        poor = sum(df_valid['cloud_lt_50'] == 0)
        
        print(f'\n数据质量分类:')
        print(f'  优秀 (云<20%): {excellent} 批次 ({excellent/len(df_valid)*100:.1f}%)')
        print(f'  良好 (云20-30%): {good} 批次 ({good/len(df_valid)*100:.1f}%)')
        print(f'  中等 (云30-50%): {medium} 批次 ({medium/len(df_valid)*100:.1f}%)')
        print(f'  较差 (云>50%): {poor} 批次 ({poor/len(df_valid)*100:.1f}%)')
        
        print(f'\n建议:')
        if excellent / len(df_valid) > 0.7:
            print('  ✓ 数据质量优秀，建议使用云覆盖<20%阈值进行采样')
        elif (excellent + good) / len(df_valid) > 0.7:
            print('  ⚠ 数据质量良好，建议使用云覆盖<30%阈值进行采样')
        elif (excellent + good + medium) / len(df_valid) > 0.7:
            print('  ⚠ 数据质量中等，建议使用云覆盖<50%阈值进行采样')
        else:
            print('  ✗ 数据质量较差，建议考虑更换时间范围或区域')
    
    print('=' * 80)
    print('检查完成！')
    print('=' * 80)


if __name__ == '__main__':
    main()

