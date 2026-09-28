import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

"""
入参:
- 无命令行参数，文件路径在代码中指定
方法:
- 读取CSV文件中的water_mask列数据
- 根据经纬度坐标重构为2D栅格数组
- 使用matplotlib可视化水体掩膜分布
出参:
- 可视化图像显示和保存
"""


def load_watermask_data(csv_path):
    """
    入参:
    - csv_path (str/Path): CSV文件路径
    方法:
    - 使用pandas读取CSV文件
    - 提取longitude, latitude和water_mask列
    出参:
    - df (DataFrame): 包含坐标和water_mask的数据框
    """
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"数据行数: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 检查必需的列是否存在
    required_cols = ['longitude', 'latitude', 'water_mask']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    return df[['longitude', 'latitude', 'water_mask']]


def reconstruct_2d_array(df):
    """
    入参:
    - df (DataFrame): 包含longitude, latitude, water_mask的数据框
    方法:
    - 获取唯一的经纬度值并排序
    - 计算2D数组的行列数
    - 将1D数据映射到2D栅格位置
    出参:
    - watermask_2d (ndarray): 重构的2D水体掩膜数组
    - extent (tuple): 地理范围(xmin, xmax, ymin, ymax)
    """
    print("正在重构2D数组...")
    
    # 获取唯一的经纬度值
    unique_lons = np.sort(df['longitude'].unique())
    unique_lats = np.sort(df['latitude'].unique())
    
    height = len(unique_lats)
    width = len(unique_lons)
    
    print(f"图像尺寸: {width} x {height}")
    
    # 初始化2D数组，使用NaN填充
    watermask_2d = np.full((height, width), np.nan)
    
    # 创建经纬度到索引的映射
    lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}
    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
    
    # 填充2D数组
    for _, row in df.iterrows():
        lon_idx = lon_to_idx[row['longitude']]
        # 纬度需要翻转，因为图像坐标系从上到下，纬度从下到上
        lat_idx = height - 1 - lat_to_idx[row['latitude']]
        watermask_2d[lat_idx, lon_idx] = row['water_mask']
    
    # 计算地理范围
    extent = (unique_lons.min(), unique_lons.max(), 
              unique_lats.min(), unique_lats.max())
    
    return watermask_2d, extent


def visualize_watermask(watermask_2d, extent, output_path=None):
    """
    入参:
    - watermask_2d (ndarray): 2D水体掩膜数组
    - extent (tuple): 地理范围(xmin, xmax, ymin, ymax)
    - output_path (str/Path, optional): 输出图像保存路径
    方法:
    - 创建matplotlib图形对象
    - 使用二值颜色映射显示水体掩膜
    - 添加坐标轴标签和标题
    - 保存并显示图像
    出参:
    - 无返回值，直接显示和保存图像
    """
    print("正在创建可视化图像...")
    
    # 统计水体像素数量
    total_pixels = np.sum(~np.isnan(watermask_2d))
    water_pixels = np.nansum(watermask_2d)
    water_percentage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    print(f"总像素数: {int(total_pixels)}")
    print(f"水体像素数: {int(water_pixels)}")
    print(f"水体占比: {water_percentage:.2f}%")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 显示水体掩膜，0为陆地（黑色），1为水体（蓝色）
    im = ax.imshow(watermask_2d, cmap='Blues', extent=extent, 
                   interpolation='nearest', vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='Water Mask')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Land', 'Water'])
    
    # 设置标题和标签
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Water Mask Visualization\nWater Coverage: {water_percentage:.2f}%', fontsize=14)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {output_path}")
    
    # 显示图像
    plt.show()
    print("可视化完成！")


def main():
    """
    入参:
    - 无
    方法:
    - 定义输入输出路径
    - 调用数据加载函数
    - 调用2D重构函数
    - 调用可视化函数
    出参:
    - 无
    """
    # 设置文件路径
    csv_path = Path(r"E:\1代码\模型\gee\栅格与表互相转换\处理结果\step2_output_with_indices.csv")
    output_path = csv_path.parent / "watermask_visualization.png"
    
    # 加载数据
    df = load_watermask_data(csv_path)
    
    # 重构2D数组
    watermask_2d, extent = reconstruct_2d_array(df)
    
    # 可视化
    visualize_watermask(watermask_2d, extent, output_path)


if __name__ == "__main__":
    main()

