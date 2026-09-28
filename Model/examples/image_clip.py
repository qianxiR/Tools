import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager

def display_grid_crops(image_folder, grid_size=(4, 4)):
    """
    从指定文件夹中读取所有图像，显示渔网裁剪示意图和按照渔网网格裁剪的图像
    
    参数:
    image_folder: 图像文件夹路径
    grid_size: 渔网网格大小，默认为(4, 4)
    """
    # 设置中文字体，解决乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 检查路径是否存在
    if not os.path.exists(image_folder):
        print(f"错误: 路径 {image_folder} 不存在!")
        return
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    
    if not image_files:
        print(f"错误: 在 {image_folder} 中没有找到图像文件!")
        return
    
    print(f"找到 {len(image_files)} 张图像，开始处理...")
    
    for i, img_file in enumerate(image_files):
        print(f"正在处理图像 {i+1}/{len(image_files)}: {img_file}")
        
        # 读取图像
        img_path = os.path.join(image_folder, img_file)
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 获取原始尺寸
        height, width = original_image.shape[:2]
        
        # 创建带有渔网网格的图像副本
        grid_image = original_image.copy()
        
        # 计算网格线的位置和每个网格单元的大小
        rows, cols = grid_size
        row_height = height // rows
        col_width = width // cols
        
        # 绘制水平线 - 使用黑色且更粗的线条
        for r in range(1, rows):
            y = r * row_height
            cv2.line(grid_image, (0, y), (width, y), (0, 0, 0), 20)  # 黑色，线宽8
        
        # 绘制垂直线 - 使用黑色且更粗的线条
        for c in range(1, cols):
            x = c * col_width
            cv2.line(grid_image, (x, 0), (x, height), (0, 0, 0), 20)  # 黑色，线宽8
        
        # 创建一个大图来显示所有裁剪的图像
        # 每行显示cols个裁剪图像，总共rows行
        fig, axes = plt.subplots(rows + 1, cols, figsize=(cols * 4, (rows + 1) * 4))
        
        # 在第一行显示原始图像和网格图像
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'原始图像: {img_file}\n尺寸: {width}x{height}')
        axes[0, 0].axis('off')
        
        # 创建一个带有所有裁剪区域标记的网格图像
        all_crops_grid = grid_image.copy()
        
        # 对每个网格单元进行裁剪
        for r in range(rows):
            for c in range(cols):
                # 计算当前网格单元的边界
                cell_x1 = c * col_width
                cell_y1 = r * row_height
                cell_x2 = (c + 1) * col_width if c < cols - 1 else width
                cell_y2 = (r + 1) * row_height if r < rows - 1 else height
                
                # 在所有裁剪图像上标记当前网格单元
                cv2.rectangle(all_crops_grid, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 0, 0), 3)
                
                # 裁剪当前网格单元
                cropped_image = original_image[cell_y1:cell_y2, cell_x1:cell_x2]
                
                # 显示裁剪后的图像
                axes[r + 1, c].imshow(cropped_image)
                axes[r + 1, c].set_title(f'裁剪 ({c},{r})\n坐标: ({cell_x1},{cell_y1})')
                axes[r + 1, c].axis('off')
        
        # 显示带有所有裁剪区域标记的网格图像
        axes[0, 1].imshow(all_crops_grid)
        axes[0, 1].set_title(f'渔网裁剪示意图\n网格大小: {grid_size[0]}x{grid_size[1]}')
        axes[0, 1].axis('off')
        
        # 隐藏第一行其余的子图
        for c in range(2, cols):
            axes[0, c].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"grid_crops_{img_file.split('.')[0]}.png", dpi=300)
        plt.show()
        print(f"已显示图像 {i+1} 的渔网裁剪，并保存为 'grid_crops_{img_file.split('.')[0]}.png'")
    
    print("所有图像处理完成！")

# 主程序
if __name__ == "__main__":
    # 设置图像文件夹路径
    image_folder = r"D:\VScode workbase\DATA\uavid_seg\uavid_train\seq1\Images"
    
    # 处理目录中的所有图像，添加4x4的渔网网格，直接按照渔网分割裁剪
    display_grid_crops(image_folder, grid_size=(4, 4))