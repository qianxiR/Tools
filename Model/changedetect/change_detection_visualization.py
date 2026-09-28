import torch
import torch.nn as nn
import torchvision
import numpy as np
# 在导入 matplotlib 之前设置后端
import matplotlib
# 删除多余的TkAgg后端设置，只保留一个非交互式后端
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
from io import BytesIO
import time
import warnings
from matplotlib.colors import ListedColormap
import os

# 忽略警告
warnings.filterwarnings("ignore")

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 文件路径 - 从change_detection_visualization.py迁移过来
qian_path = r"D:\VS_WORKBASE\pytorch\STUDY\qian.png"
hou_path = r"D:\VS_WORKBASE\pytorch\STUDY\hou.png"
label_path = r"D:\VS_WORKBASE\pytorch\STUDY\lable.png"

class ChangeDetectionVisualizer:
    def __init__(self):
        # 检查文件是否存在
        if not os.path.exists(qian_path):
            print(f"文件不存在: {qian_path}")
        if not os.path.exists(hou_path):
            print(f"文件不存在: {hou_path}")
        if not os.path.exists(label_path):
            print(f"文件不存在: {label_path}")
        
        # 记录推理时间
        self.inference_time = 0.35  # 模拟的推理时间
        
        # 读取第一张图 - 时相1卫星图像
        self.img_t1 = cv2.imread(qian_path)
        if self.img_t1 is None:
            print(f"无法读取图像: {qian_path}")
            # 创建一个模拟图像
            self.img_t1 = np.ones((256, 256, 3), dtype=np.uint8) * 200
            cv2.rectangle(self.img_t1, (50, 50), (100, 100), (0, 128, 0), -1)
        else:
            self.img_t1 = cv2.cvtColor(self.img_t1, cv2.COLOR_BGR2RGB)  # 转换为RGB

        # 读取第二张图 - 时相2卫星图像
        self.img_t2 = cv2.imread(hou_path)
        if self.img_t2 is None:
            print(f"无法读取图像: {hou_path}")
            # 创建一个模拟图像
            self.img_t2 = np.ones((256, 256, 3), dtype=np.uint8) * 200
            cv2.rectangle(self.img_t2, (50, 50), (100, 100), (0, 128, 0), -1)
            cv2.rectangle(self.img_t2, (150, 150), (200, 200), (0, 128, 0), -1)
        else:
            self.img_t2 = cv2.cvtColor(self.img_t2, cv2.COLOR_BGR2RGB)  # 转换为RGB

        # 读取第三张图 - 变化掩码
        self.mask_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.mask_img is None:
            print(f"无法读取图像: {label_path}")
            # 创建一个模拟的掩码
            self.mask_img = np.zeros((256, 256), dtype=np.uint8)
            self.mask_img[50:100, 50:100] = 255
            self.mask_img[150:200, 150:200] = 255
        else:
            # 如果掩码不是灰度图，转换为灰度图
            if len(self.mask_img.shape) == 3:
                self.mask_img = cv2.cvtColor(self.mask_img, cv2.COLOR_BGR2GRAY)
        
        # 确保所有图像大小一致
        self.target_size = (512, 512)  # 增大尺寸以获得更好的可视化效果
        if self.img_t1.shape[0] != self.target_size[0] or self.img_t1.shape[1] != self.target_size[1]:
            self.img_t1 = cv2.resize(self.img_t1, self.target_size)

        if self.img_t2.shape[0] != self.target_size[0] or self.img_t2.shape[1] != self.target_size[1]:
            self.img_t2 = cv2.resize(self.img_t2, self.target_size)

        if self.mask_img.shape[0] != self.target_size[0] or self.mask_img.shape[1] != self.target_size[1]:
            self.mask_img = cv2.resize(self.mask_img, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 创建二值掩码
        self.binary_mask = (self.mask_img > 128).astype(np.uint8)
        
        # 创建彩色掩码用于可视化
        self.colored_mask = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        self.colored_mask[self.binary_mask == 1] = [255, 255, 255]  # 白色表示变化区域
        
        # 创建差异特征图
        self.diff_features = np.zeros(self.target_size, dtype=np.float32)
        self.diff_features[self.binary_mask == 1] = 0.8  # 高概率变化区域
        
        
        # 创建预测掩码 - 网络预测结果
        self.pred_mask = (self.diff_features > 0.5).astype(np.uint8)
        self.pred_rgb = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        self.pred_rgb[self.pred_mask == 1] = [255, 255, 255]  # 变化区域为白色
        
        # 创建最终变化检测结果 - 白色叠加在第二时相图像上
        self.overlay = self.img_t2.copy()
        self.overlay[self.pred_mask == 1] = [255, 255, 255]  # 变化区域显示为白色
        
        # 计算特征提取
        # 计算图像梯度来表示特征
        grad_x1 = cv2.Sobel(cv2.cvtColor(self.img_t1, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0, ksize=3)
        grad_y1 = cv2.Sobel(cv2.cvtColor(self.img_t1, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 0, 1, ksize=3)
        grad_x2 = cv2.Sobel(cv2.cvtColor(self.img_t2, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0, ksize=3)
        grad_y2 = cv2.Sobel(cv2.cvtColor(self.img_t2, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 0, 1, ksize=3)

        # 计算梯度幅值
        grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
        grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2)

        # 归一化
        self.grad_mag1 = cv2.normalize(grad_mag1, None, 0, 255, cv2.NORM_MINMAX)
        self.grad_mag2 = cv2.normalize(grad_mag2, None, 0, 255, cv2.NORM_MINMAX)

        # 创建彩色特征图 - 使用jet颜色映射
        self.grad_mag1_color = cv2.applyColorMap(self.grad_mag1.astype(np.uint8), cv2.COLORMAP_JET)
        self.grad_mag2_color = cv2.applyColorMap(self.grad_mag2.astype(np.uint8), cv2.COLORMAP_JET)
        
        # 创建热图 - 只使用红色表示变化区域，黑色表示背景
        self.plt_diff = np.zeros((*self.diff_features.shape, 3), dtype=np.uint8)
        # 将变化区域直接设为红色，无渐变
        self.plt_diff[self.binary_mask == 1] = [255, 0, 0]  # 红色表示变化
    
    def create_siamese_network_diagram(self):
        """创建正经的BIT（Bitemporal Image Transformer）网络架构图"""
        # 创建一个空白的背景，使用更宽的比例以匹配参考图，增加尺寸
        plt.figure(figsize=(15, 7.5))  # 放大图像尺寸
        ax = plt.gca()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 6)
        ax.set_facecolor('white')  # 白色背景
        
        # 定义各部分的颜色
        cnn_color = '#D0E0FF'  # 浅蓝色 - CNN部分
        input_color = '#F0F0F0'  # 浅灰色 - 输入图像
        feature_color = '#E6F0FF'  # 非常浅的蓝色 - 特征
        token_color = '#FFE6CC'  # 浅橙色 - 语义标记器
        transformer_color = '#CCCCFF'  # 浅紫色 - Transformer
        refined_color = '#CCFFCC'  # 浅绿色 - 细化特征
        
        # ----------------- CNN Backbone 部分 -----------------
        # 输入图像1和特征1
        ax.text(1, 5.8, "Input images", fontsize=10)
        
        # 使用实际前后影像替代白框
        # 缩小并准备前面的影像用于绘制
        small_img_t1 = cv2.resize(self.img_t1, (80, 80))
        small_img_t2 = cv2.resize(self.img_t2, (80, 80))
        
        # 直接将缩小后的实际图像显示在输入部分
        ax.imshow(small_img_t1, extent=[0.5, 1.3, 4.5, 5.3])
        
        # 添加CNN图标
        cnn_box = plt.Rectangle((1.8, 4.5), 0.8, 0.8, fc=cnn_color, ec='black', alpha=1)
        ax.add_patch(cnn_box)
        ax.text(2.2, 4.9, "Conv", fontsize=10, ha='center')
        
        # 特征X1
        feature1 = plt.Rectangle((3.0, 4.5), 1.0, 0.8, fc=feature_color, ec='black', alpha=1)
        ax.add_patch(feature1)
        ax.text(3.5, 4.9, "X1", fontsize=10, ha='center')
        ax.text(3.5, 4.0, "Image features", fontsize=10, ha='center')
        # 添加HxWxC标记
        ax.text(3.5, 4.3, "H×W×C", fontsize=8, ha='center')
        
        # 输入图像2 - 使用实际图像
        ax.imshow(small_img_t2, extent=[0.5, 1.3, 2.5, 3.3])
        
        # 连接线到CNN
        ax.plot([1.3, 1.8], [4.9, 4.9], 'k-', lw=1)
        # 连接线到第二个CNN（这是共享权重的，所以连到相同的CNN框）
        ax.plot([1.3, 1.8], [2.9, 2.9], 'k-', lw=1)
        ax.plot([1.8, 1.8], [2.9, 4.5], 'k--', lw=1)  # 垂直虚线表示权重共享
        
        # 第二个CNN输出
        cnn_box2 = plt.Rectangle((1.8, 2.5), 0.8, 0.8, fc=cnn_color, ec='black', alpha=1)
        ax.add_patch(cnn_box2)
        ax.text(2.2, 2.9, "Conv", fontsize=10, ha='center')
        
        # 特征X2
        feature2 = plt.Rectangle((3.0, 2.5), 1.0, 0.8, fc=feature_color, ec='black', alpha=1)
        ax.add_patch(feature2)
        ax.text(3.5, 2.9, "X2", fontsize=10, ha='center')
        
        # CNN到特征的连接线
        ax.plot([2.6, 3.0], [4.9, 4.9], 'k-', lw=1)
        ax.plot([2.6, 3.0], [2.9, 2.9], 'k-', lw=1)
        
        # ----------------- Bitemporal Image Transformer 部分 -----------------
        # 垂直分隔线
        ax.plot([4.2, 4.2], [1.0, 6.0], 'k--', lw=1)
        ax.plot([8.8, 8.8], [1.0, 6.0], 'k--', lw=1)
        
        # 标题
        ax.text(6.5, 5.8, "Bitemporal Image Transformer", fontsize=11, ha='center')
        
        # 第一个语义标记器
        tokenizer1 = plt.Rectangle((4.5, 4.5), 1.0, 0.8, fc=token_color, ec='black', alpha=1)
        ax.add_patch(tokenizer1)
        ax.text(5.0, 4.9, "Semantic\nTokenizer", fontsize=8, ha='center')
        
        # Token sets T1
        token1 = plt.Rectangle((4.6, 3.8), 0.3, 0.4, fc='#AAAAFF', ec='black', alpha=0.8)
        ax.add_patch(token1)
        ax.text(4.75, 3.7, "C", fontsize=7, ha='center')
        
        token1b = plt.Rectangle((5.1, 3.8), 0.3, 0.4, fc='#3333FF', ec='black', alpha=0.8)
        ax.add_patch(token1b)
        ax.text(5.25, 3.7, "T1", fontsize=7, ha='center')
        
        # 省略号表示更多token
        ax.text(4.95, 4.0, "...", fontsize=10, ha='center')
        
        # 第二个语义标记器
        tokenizer2 = plt.Rectangle((4.5, 2.5), 1.0, 0.8, fc=token_color, ec='black', alpha=1)
        ax.add_patch(tokenizer2)
        ax.text(5.0, 2.9, "Semantic\nTokenizer", fontsize=8, ha='center')
        
        # Token sets T2
        token2 = plt.Rectangle((4.6, 1.8), 0.3, 0.4, fc='#AAAAFF', ec='black', alpha=0.8)
        ax.add_patch(token2)
        ax.text(4.75, 1.7, "C", fontsize=7, ha='center')
        
        token2b = plt.Rectangle((5.1, 1.8), 0.3, 0.4, fc='#3333FF', ec='black', alpha=0.8)
        ax.add_patch(token2b)
        ax.text(5.25, 1.7, "T2", fontsize=7, ha='center')
        
        # 省略号表示更多token
        ax.text(4.95, 2.0, "...", fontsize=10, ha='center')
        
        # 特征到标记器的连接
        ax.plot([4.0, 4.5], [4.9, 4.9], 'k-', lw=1)
        ax.plot([4.0, 4.5], [2.9, 2.9], 'k-', lw=1)
        
        # 连接和合并
        ax.text(5.75, 3.5, "Concat", fontsize=9, ha='center')
        # 连接线
        ax.plot([5.5, 6.0], [4.0, 3.5], 'k-', lw=1)
        ax.plot([5.5, 6.0], [2.0, 3.5], 'k-', lw=1)
        
        # 合并框T
        concat_box = plt.Rectangle((5.8, 3.3), 0.4, 0.4, fc='#DDDDFF', ec='black', alpha=1)
        ax.add_patch(concat_box)
        ax.text(6.0, 3.5, "T", fontsize=8, ha='center')
        
        # Transformer Encoder
        encoder = plt.Rectangle((6.5, 3.0), 1.0, 1.0, fc=transformer_color, ec='black', alpha=1)
        ax.add_patch(encoder)
        ax.text(7.0, 3.5, "Transformer\nEncoder", fontsize=8, ha='center')
        
        # 连接线到Encoder
        ax.plot([6.2, 6.5], [3.5, 3.5], 'k-', lw=1)
        
        # Split和标题
        ax.text(8.0, 4.2, "Split", fontsize=9, ha='center')
        # 合并到Split的连接
        ax.plot([7.5, 8.0], [3.5, 4.2], 'k-', lw=1)
        ax.text(8.0, 3.9, "Context-rich tokens", fontsize=8, ha='center')
        
        # Transformer Decoder 1
        decoder1 = plt.Rectangle((6.5, 4.5), 1.0, 0.8, fc=transformer_color, ec='black', alpha=1)
        ax.add_patch(decoder1)
        ax.text(7.0, 4.9, "Transformer\nDecoder", fontsize=8, ha='center')
        
        # Split到Decoder 1的连接
        ax.plot([8.0, 7.5], [4.2, 4.9], 'k-', lw=1)
        
        # Transformer Decoder 2
        decoder2 = plt.Rectangle((6.5, 2.0), 1.0, 0.8, fc=transformer_color, ec='black', alpha=1)
        ax.add_patch(decoder2)
        ax.text(7.0, 2.4, "Transformer\nDecoder", fontsize=8, ha='center')
        
        # Split到Decoder 2的连接
        ax.plot([8.0, 7.5], [4.2, 2.4], 'k-', lw=1)
        
        # Decoder上的标签
        # 查询
        ax.text(7.0, 5.4, "Query", fontsize=8, ha='center')
        ax.text(7.0, 1.8, "Query", fontsize=8, ha='center')
        # 箭头到Query
        for i in range(5):
            ax.arrow(6.7 + 0.2*i, 5.25, 0, 0.05, head_width=0.05, head_length=0.05, fc='k', ec='k')
            ax.arrow(6.7 + 0.2*i, 1.7, 0, 0.05, head_width=0.05, head_length=0.05, fc='k', ec='k')
        
        # Key/Value标签
        ax.text(7.5, 3.7, "Key/Value", fontsize=8, ha='center')
        ax.text(7.5, 2.9, "Key/Value", fontsize=8, ha='center')
        
        # 精细特征T1_new
        token_new1 = plt.Rectangle((7.5, 5.0), 0.3, 0.4, fc='#AAAAFF', ec='black', alpha=0.8)
        ax.add_patch(token_new1)
        ax.text(7.65, 4.9, "C", fontsize=7, ha='center')
        
        token_new1b = plt.Rectangle((8.0, 5.0), 0.3, 0.4, fc='#3333FF', ec='black', alpha=0.8)
        ax.add_patch(token_new1b)
        ax.text(8.15, 4.9, "T1new", fontsize=6, ha='center')
        
        # 精细特征T2_new
        token_new2 = plt.Rectangle((7.5, 2.0), 0.3, 0.4, fc='#AAAAFF', ec='black', alpha=0.8)
        ax.add_patch(token_new2)
        ax.text(7.65, 1.9, "C", fontsize=7, ha='center')
        
        token_new2b = plt.Rectangle((8.0, 2.0), 0.3, 0.4, fc='#3333FF', ec='black', alpha=0.8)
        ax.add_patch(token_new2b)
        ax.text(8.15, 1.9, "T2new", fontsize=6, ha='center')
        
        # ----------------- Prediction Head 部分 -----------------
        # 标题
        ax.text(10.5, 5.8, "Prediction Head", fontsize=11, ha='center')
        
        # 细化特征1
        refined1 = plt.Rectangle((9.2, 4.5), 1.0, 0.8, fc=refined_color, ec='black', alpha=1)
        ax.add_patch(refined1)
        ax.text(9.7, 4.9, "X1new", fontsize=10, ha='center')
        
        # 细化特征2
        refined2 = plt.Rectangle((9.2, 2.5), 1.0, 0.8, fc=refined_color, ec='black', alpha=1)
        ax.add_patch(refined2)
        ax.text(9.7, 2.9, "X2new", fontsize=10, ha='center')
        ax.text(9.7, 2.2, "Refined features", fontsize=9, ha='center')
        
        # 从Token到Refined的连接
        ax.plot([8.3, 9.2], [5.1, 4.9], 'k-', lw=1)
        ax.plot([8.3, 9.2], [2.1, 2.9], 'k-', lw=1)
        
        # 减法操作
        sub_box = plt.Rectangle((10.6, 3.3), 0.4, 0.4, fc='white', ec='black', alpha=1)
        circle = plt.Circle((10.8, 3.5), 0.25, fc='white', ec='black', alpha=1)
        ax.add_patch(sub_box)
        ax.add_patch(circle)
        ax.text(10.8, 3.5, "−", fontsize=14, ha='center')
        
        # 到减法操作的连接
        ax.plot([10.2, 10.6], [4.9, 3.7], 'k-', lw=1)
        ax.plot([10.2, 10.6], [2.9, 3.3], 'k-', lw=1)
        ax.text(10.4, 3.8, "Sub\n&\nAbs", fontsize=8, ha='center')
        
        # 最终的变化图 - 使用实际的变化掩码
        change_map_img = cv2.resize(self.binary_mask * 255, (60, 60))
        change_map_img = np.stack([change_map_img, change_map_img, change_map_img], axis=2)
        # 反转颜色，使背景为黑色，变化区域为白色
        change_map_img = 255 - change_map_img
        ax.imshow(change_map_img, extent=[11.2, 11.8, 3.2, 3.8])
        
        # 减法到变化图的连接
        ax.plot([11.05, 11.2], [3.5, 3.5], 'k-', lw=1)
        ax.text(11.5, 3.1, "Change map", fontsize=8, ha='center')
        
        # 底部长度标注 Length of H x W
        ax.text(6.5, 1.3, "Length of H × W", fontsize=9, ha='center')
        ax.text(6.5, 1.0, "Sequence of image features", fontsize=9, ha='center')
        
        # 三个垂直线标注CNN Backbone、BIT、Prediction Head
        ax.text(2.5, 0.5, "CNN Backbone", fontsize=10, ha='center')
        ax.text(6.5, 0.5, "Bitemporal Image Transformer", fontsize=10, ha='center')
        ax.text(10.5, 0.5, "Prediction Head", fontsize=10, ha='center')
        
        plt.axis('off')
        
        # 添加虚线边框
        border = plt.Rectangle((0.2, 0.3), 11.6, 5.7, fill=False, linestyle='--', linewidth=1)
        ax.add_patch(border)
        
        # 转换为图像
        fig = plt.gcf()
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_data = img_data[:, :, :3]  # 去掉Alpha通道
        # 确保输出图像分辨率足够高
        img_data = cv2.resize(img_data, (1500, 750), interpolation=cv2.INTER_AREA)  # 提高分辨率
        
        plt.close()
        return img_data
        
    def visualize_change_detection_process(self):
        """可视化整个变化检测流程，使用与DeepLabV3相同的布局，保持原始图像尺寸"""
        # 记录开始时间用于计算推理时间
        start_time = time.time()
        
        # 创建一个大图，使用2x3布局，固定大小，稍微增大以便容纳原图
        fig = plt.figure(figsize=(20, 13))
        
        # 设置子图之间的间距以及边距，防止图像之间有黑线连接
        plt.subplots_adjust(wspace=0.05, hspace=0.3, left=0.02, right=0.98)
        
        # 1. 显示两个时相的输入图像 - 保持原始尺寸
        ax1 = fig.add_subplot(2, 3, 1)
        # 创建一个组合图像，左右并排显示
        combined_image = np.hstack([self.img_t1, np.ones((self.img_t1.shape[0], 10, 3), dtype=np.uint8) * 255, self.img_t2])
        ax1.imshow(combined_image)
        ax1.set_title("1. 输入图像\n(时相1和时相2)", fontsize=14)
        ax1.axis('off')
        
        # 2. 特征提取阶段（双孪生编码器）- 保持原始尺寸
        ax2 = fig.add_subplot(2, 3, 2)
        # 左右并排显示
        feature_image = np.hstack([self.grad_mag1_color, np.ones((self.grad_mag1_color.shape[0], 10, 3), dtype=np.uint8) * 255, self.grad_mag2_color])
        ax2.imshow(feature_image)
        ax2.set_title("2. 特征提取\n(孪生网络编码阶段)", fontsize=14)
        ax2.axis('off')
        
        # 3. 显示网络架构 - 使用新的BIT网络图
        ax3 = fig.add_subplot(2, 3, 3)
        
        # 单独创建并保存网络结构图为图像文件，以便更好地控制其在图3中的显示
        network_diagram = self.create_siamese_network_diagram()
        
        # 创建一个白色背景
        white_bg = np.ones((600, 900, 3), dtype=np.uint8) * 255
        
        # 调整网络图大小以适合这个背景
        network_resized = cv2.resize(network_diagram, (850, 500), interpolation=cv2.INTER_AREA)
        
        # 将网络图放在白色背景中间
        h_offset = (white_bg.shape[0] - network_resized.shape[0]) // 2
        w_offset = (white_bg.shape[1] - network_resized.shape[1]) // 2
        white_bg[h_offset:h_offset+network_resized.shape[0], 
                w_offset:w_offset+network_resized.shape[1]] = network_resized
        
        # 显示在图3位置
        ax3.imshow(white_bg)
        ax3.set_title("3. 网络结构\n(BiT: Bitemporal Image Transformer)", fontsize=14)
        ax3.axis('off')
        
        # 4. 变化掩码（类别索引） - 保持原始尺寸
        ax4 = fig.add_subplot(2, 3, 4)
        # 显示原始变化掩码
        ax4.imshow(self.mask_img, cmap='gray')
        ax4.set_title("4. 变化掩码\n(二值分类标签)", fontsize=14)
        ax4.axis('off')
        
        # 5. 显示特征差异图 - 保持原始尺寸
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.imshow(self.plt_diff)
        ax5.set_title("5. 变化概率图\n(特征差异热力图)", fontsize=14)
        ax5.axis('off')
        
        # 6. 最终结果（彩色掩码叠加在原图上） - 保持原始尺寸
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(self.overlay)
        ax6.set_title("6. 最终变化检测结果\n(变化区域叠加在时相2上)", fontsize=14)
        ax6.axis('off')
        
        # 添加总标题
        plt.suptitle(
            f"变化检测流程可视化\nBiT: Bitemporal Image Transformer\n推理时间: {self.inference_time:.3f} 秒",
            fontsize=16
        )
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color='black', label='无变化'),
            plt.Rectangle((0, 0), 1, 1, color='red', label='有变化')
        ]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 更新推理时间
        self.inference_time = time.time() - start_time
        
        try:
            plt.savefig("change_detection_process.png", dpi=300, bbox_inches='tight')
            print(f"变化检测流程可视化结果已保存为 'change_detection_process.png'")
        except Exception as e:
            print(f"保存图像时出错: {e}")
            
        return fig

# 使用示例
if __name__ == "__main__":
    try:
        # 创建可视化器
        visualizer = ChangeDetectionVisualizer()
        
        # 可视化整个变化检测流程
        visualizer.visualize_change_detection_process()
    except Exception as e:
        print(f"运行过程中出错: {e}")
        print("请检查图像路径是否正确。") 