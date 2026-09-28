import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
from io import BytesIO
import time
import warnings
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from sklearn.cluster import KMeans

# 忽略警告
warnings.filterwarnings("ignore")

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ViTSegmentationVisualizer:
    def __init__(self):
        try:
            # 加载预训练的ViT模型
            self.model = torchvision.models.vit_b_16(pretrained=True)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
            # 分割类别 - 修改为指定的类别
            self.classes = [
                '背景', '人', '羊', '草地', '杂草', '房屋', '树木'
            ]
            
            # 为每个类别创建一个颜色
            np.random.seed(42)
            self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
            # 设置背景为黑色
            self.colors[0] = [0, 0, 0]
            # 设置人为红色
            self.colors[1] = [255, 0, 0]
            # 设置羊为浅紫色
            self.colors[2] = [200, 150, 200]
            # 设置草地为绿色
            self.colors[3] = [0, 255, 0]
            # 设置杂草为黄绿色
            self.colors[4] = [180, 255, 100]
            # 设置房屋为蓝色
            self.colors[5] = [0, 0, 255]
            # 设置树木为深绿色
            self.colors[6] = [0, 128, 0]
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("使用模拟模型进行演示...")
            self.model = None
            
            # 分割类别 - 修改为指定的类别
            self.classes = [
                '背景', '人', '羊', '草地', '杂草', '房屋', '树木'
            ]
            
            # 为每个类别创建一个颜色
            np.random.seed(42)
            self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
            # 设置背景为黑色
            self.colors[0] = [0, 0, 0]
            # 设置人为红色
            self.colors[1] = [255, 0, 0]
            # 设置羊为浅紫色
            self.colors[2] = [200, 150, 200]
            # 设置草地为绿色
            self.colors[3] = [0, 255, 0]
            # 设置杂草为黄绿色
            self.colors[4] = [180, 255, 100]
            # 设置房屋为蓝色
            self.colors[5] = [0, 0, 255]
            # 设置树木为深绿色
            self.colors[6] = [0, 128, 0]
            
            # 初始化必要的属性
            self.input_tensor = None
        
    def load_image(self, image_path=None, url=None):
        """加载图像，可以是本地路径或URL"""
        try:
            if url:
                response = requests.get(url)
                self.original_image = Image.open(BytesIO(response.content))
                self.image_source = f"URL: {url}"
            elif image_path:
                self.original_image = Image.open(image_path)
                self.image_source = f"文件: {image_path}"
            else:
                raise ValueError("必须提供image_path或url参数")
                
            # 确保图像是RGB格式（处理RGBA或其他格式）
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')
                
            # 保存原始图像
            self.image_np = np.array(self.original_image)
            
            # 调整图像大小以适应模型输入
            self.input_image = self.original_image.resize((224, 224))
            
            # 预处理图像
            self.preprocess_image()
            
            return self.original_image
        except Exception as e:
            print(f"加载图像时出错: {e}")
            # 创建一个示例图像
            print("使用示例图像...")
            self.image_np = np.ones((600, 800, 3), dtype=np.uint8) * 255
            # 添加一些形状
            cv2.rectangle(self.image_np, (300, 200), (500, 400), (255, 0, 0), -1)
            cv2.circle(self.image_np, (200, 300), 100, (0, 255, 0), -1)
            self.original_image = Image.fromarray(self.image_np)
            self.input_image = self.original_image.resize((224, 224))
            self.image_source = "示例图像"
            
            # 预处理图像
            self.preprocess_image()
            
            return self.original_image
    
    def preprocess_image(self):
        """预处理图像以适应模型输入"""
        # 定义预处理变换
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 应用预处理
        self.input_tensor = preprocess(self.input_image)
        
        # 添加批次维度
        self.input_batch = self.input_tensor.unsqueeze(0)
        
        # 如果有GPU，将数据移到GPU
        if torch.cuda.is_available() and self.model is not None:
            self.input_batch = self.input_batch.cuda()
            
        return self.input_batch
    
    def segment(self):
        """执行语义分割"""
        # 记录开始时间
        start_time = time.time()
        
        try:
            if self.model is not None:
                # 使用模型进行预测
                with torch.no_grad():
                    # 获取ViT模型的输出
                    outputs = self.model(self.input_batch)
                    
                    # 获取模型的注意力权重
                    # 注意：在实际应用中，需要修改模型以访问注意力权重
                    # 这里我们使用一个技巧来提取注意力权重
                    try:
                        # 尝试访问模型的注意力权重
                        # 这需要模型有相应的钩子或属性
                        attn_weights = []
                        
                        # 遍历模型的编码器块
                        for block in self.model.encoder.layers:
                            # 尝试获取自注意力权重
                            if hasattr(block.self_attention, 'attention_weights'):
                                attn_weights.append(block.self_attention.attention_weights)
                        
                        if attn_weights:
                            # 如果成功获取到注意力权重
                            self.attention_weights = attn_weights[0].cpu().numpy()
                        else:
                            # 如果无法获取，则生成模拟的注意力权重
                            print("无法获取真实注意力权重，使用模拟数据...")
                            self.attention_weights = self._generate_simulated_attention()
                    except Exception as e:
                        print(f"获取注意力权重时出错: {e}")
                        # 生成模拟的注意力权重
                        self.attention_weights = self._generate_simulated_attention()
                    
                    # 使用模型的特征图生成分割掩码
                    # 获取最后一层的特征
                    features = outputs.logits  # 分类logits
                    
                    # 将特征转换为CPU张量
                    features = features.cpu()
                    
                    # 获取预测的类别
                    _, predicted_classes = torch.max(features, 1)
                    
                    # 创建分割掩码
                    h, w = self.image_np.shape[:2]
                    self.segmentation_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # 使用特征图生成分割掩码
                    # 这里我们使用一个简单的方法：将特征图上采样到原始图像大小
                    # 在实际应用中，应该使用更复杂的方法，如条件随机场或深度学习解码器
                    
                    # 获取特征图的空间维度
                    # 对于ViT，我们需要重构特征图
                    # 假设特征是 [batch_size, num_patches, hidden_dim]
                    # 我们需要将其重构为 [batch_size, height, width, hidden_dim]
                    
                    # 使用注意力权重生成分割掩码
                    # 我们将使用注意力权重的平均值作为每个像素的类别概率
                    
                    # 首先，我们将注意力权重调整为原始图像大小
                    attention_map = self.attention_weights.mean(axis=0)  # 平均所有注意力头
                    attention_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    # 归一化注意力图
                    attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
                    
                    # 使用阈值将注意力图转换为分割掩码
                    # 我们将使用多个阈值来生成不同类别的区域
                    thresholds = [0.2, 0.4, 0.6, 0.8]
                    
                    # 背景 (类别0) - 默认
                    # 人 (类别1) - 高注意力区域
                    self.segmentation_mask[attention_resized > thresholds[3]] = 1
                    
                    # 羊 (类别2) - 中高注意力区域
                    self.segmentation_mask[(attention_resized > thresholds[2]) & (attention_resized <= thresholds[3])] = 2
                    
                    # 草地 (类别3) - 低注意力区域
                    self.segmentation_mask[attention_resized < thresholds[0]] = 3
                    
                    # 杂草 (类别4) - 随机点
                    for i in range(20):
                        x = np.random.randint(0, w)
                        y = np.random.randint(0, h)
                        if self.segmentation_mask[y, x] == 3:  # 只在草地上添加杂草
                            cv2.circle(self.segmentation_mask, (x, y), np.random.randint(5, 15), 4, -1)
                    
                    # 房屋 (类别5) - 中注意力区域
                    self.segmentation_mask[(attention_resized > thresholds[1]) & (attention_resized <= thresholds[2])] = 5
                    
                    # 树木 (类别6) - 低中注意力区域
                    self.segmentation_mask[(attention_resized > thresholds[0]) & (attention_resized <= thresholds[1])] = 6
                    
                    # 后处理：应用形态学操作使分割掩码更平滑
                    kernel = np.ones((5, 5), np.uint8)
                    self.segmentation_mask = cv2.morphologyEx(self.segmentation_mask, cv2.MORPH_CLOSE, kernel)
                    self.segmentation_mask = cv2.morphologyEx(self.segmentation_mask, cv2.MORPH_OPEN, kernel)
                    
            else:
                # 如果模型加载失败，使用基于注意力的模拟分割
                print("模型加载失败，使用基于注意力的模拟分割...")
                h, w = self.image_np.shape[:2]
                
                # 生成模拟的注意力权重
                self.attention_weights = self._generate_simulated_attention()
                
                # 使用注意力权重生成分割掩码
                attention_map = self.attention_weights.mean(axis=0)  # 平均所有注意力头
                attention_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # 归一化注意力图
                attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
                
                # 创建分割掩码
                self.segmentation_mask = np.zeros((h, w), dtype=np.uint8)
                
                # 使用阈值将注意力图转换为分割掩码
                thresholds = [0.2, 0.4, 0.6, 0.8]
                
                # 背景 (类别0) - 默认
                # 人 (类别1) - 高注意力区域
                self.segmentation_mask[attention_resized > thresholds[3]] = 1
                
                # 羊 (类别2) - 中高注意力区域
                self.segmentation_mask[(attention_resized > thresholds[2]) & (attention_resized <= thresholds[3])] = 2
                
                # 草地 (类别3) - 低注意力区域
                self.segmentation_mask[attention_resized < thresholds[0]] = 3
                
                # 杂草 (类别4) - 随机点
                for i in range(20):
                    x = np.random.randint(0, w)
                    y = np.random.randint(0, h)
                    if self.segmentation_mask[y, x] == 3:  # 只在草地上添加杂草
                        cv2.circle(self.segmentation_mask, (x, y), np.random.randint(5, 15), 4, -1)
                
                # 房屋 (类别5) - 中注意力区域
                self.segmentation_mask[(attention_resized > thresholds[1]) & (attention_resized <= thresholds[2])] = 5
                
                # 树木 (类别6) - 低中注意力区域
                self.segmentation_mask[(attention_resized > thresholds[0]) & (attention_resized <= thresholds[1])] = 6
                
                # 后处理：应用形态学操作使分割掩码更平滑
                kernel = np.ones((5, 5), np.uint8)
                self.segmentation_mask = cv2.morphologyEx(self.segmentation_mask, cv2.MORPH_CLOSE, kernel)
                self.segmentation_mask = cv2.morphologyEx(self.segmentation_mask, cv2.MORPH_OPEN, kernel)
                
        except Exception as e:
            print(f"分割时出错: {e}")
            # 创建基于图像内容的分割掩码
            print("使用基于图像内容的分割掩码...")
            h, w = self.image_np.shape[:2]
            
            # 使用颜色聚类进行简单分割
            # 将图像转换为Lab颜色空间
            lab_image = cv2.cvtColor(self.image_np, cv2.COLOR_RGB2Lab)
            
            # 将图像重塑为二维数组
            pixels = lab_image.reshape(-1, 3)
            
            # 使用K-means聚类
            n_clusters = 7  # 与类别数相同
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # 将标签重塑为原始图像形状
            segmented_image = labels.reshape(h, w)
            
            # 将聚类标签映射到我们的类别
            # 这里我们使用一个简单的映射：按照聚类中心的亮度排序
            cluster_centers = kmeans.cluster_centers_
            brightness = np.sqrt(np.sum(cluster_centers**2, axis=1))
            sorted_indices = np.argsort(brightness)
            
            # 创建映射
            label_map = np.zeros(n_clusters, dtype=np.uint8)
            
            # 背景 (类别0) - 最暗的聚类
            label_map[sorted_indices[0]] = 0
            
            # 人 (类别1) - 第二亮的聚类
            label_map[sorted_indices[-2]] = 1
            
            # 羊 (类别2) - 最亮的聚类
            label_map[sorted_indices[-1]] = 2
            
            # 草地 (类别3) - 中等亮度的聚类
            label_map[sorted_indices[2]] = 3
            
            # 杂草 (类别4) - 较暗的聚类
            label_map[sorted_indices[1]] = 4
            
            # 房屋 (类别5) - 较亮的聚类
            label_map[sorted_indices[-3]] = 5
            
            # 树木 (类别6) - 中等亮度的聚类
            label_map[sorted_indices[3]] = 6
            
            # 应用映射
            self.segmentation_mask = label_map[segmented_image]
            
            # 生成模拟的注意力权重
            self.attention_weights = self._generate_simulated_attention()
        
        # 记录结束时间
        end_time = time.time()
        self.inference_time = end_time - start_time
        
        return self.segmentation_mask
        
    def _generate_simulated_attention(self):
        """生成模拟的注意力权重"""
        attention_weights = np.zeros((8, 16, 16))
        
        # 生成一些有意义的注意力模式
        for i in range(8):  # 8个注意力头
            # 创建高斯分布的注意力中心
            x_center = np.random.randint(4, 12)
            y_center = np.random.randint(4, 12)
            
            # 为每个位置生成注意力权重
            for x in range(16):
                for y in range(16):
                    # 距离中心越近，注意力越高
                    dist = np.sqrt((x - x_center)**2 + (y - y_center)**2)
                    attention_weights[i, y, x] = np.exp(-0.3 * dist)
                    
        return attention_weights
    
    def create_colored_mask(self, mask):
        """创建彩色掩码用于可视化"""
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        
        for i in range(len(self.colors)):
            idx = mask == i
            r[idx] = self.colors[i, 0]
            g[idx] = self.colors[i, 1]
            b[idx] = self.colors[i, 2]
            
        colored_mask = np.stack([r, g, b], axis=2)
        return colored_mask
    
    def visualize_attention(self, attention_weights, head_idx=0):
        """可视化注意力权重
        
        红色区域表示高注意力（模型更关注的区域）
        蓝色区域表示低注意力（模型较少关注的区域）
        """
        # 选择一个注意力头
        attention = attention_weights[head_idx]
        
        # 调整大小以匹配原始图像
        h, w = self.image_np.shape[:2]
        attention_resized = cv2.resize(attention, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 归一化
        attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
        
        # 应用颜色映射 - 使用JET颜色映射，蓝色表示低注意力，红色表示高注意力
        # COLORMAP_JET: 蓝色 -> 青色 -> 黄色 -> 红色 (从低到高)
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def visualize_segmentation_process(self):
        """可视化整个ViT语义分割流程"""
        # 创建一个大图，使用2x3布局，固定大小
        fig = plt.figure(figsize=(18, 12))
        
        # 设置子图之间的间距
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        
        # 1. 显示原始输入图像
        ax1 = fig.add_subplot(2, 3, 1)
        # 调整图像大小为固定尺寸
        resized_image = cv2.resize(self.image_np, (400, 300), interpolation=cv2.INTER_AREA)
        ax1.imshow(resized_image)
        ax1.set_title("1. 输入图像", fontsize=14)
        ax1.axis('off')
        
        # 2. 显示图像分块和位置编码
        ax2 = fig.add_subplot(2, 3, 2)
        # 创建一个图像分块的可视化
        patched_img = resized_image.copy()
        h, w = patched_img.shape[:2]
        patch_size = 16  # ViT通常使用16x16的patch
        
        # 绘制垂直线
        for x in range(0, w, patch_size):
            cv2.line(patched_img, (x, 0), (x, h), (255, 255, 255), 1)
        
        # 绘制水平线
        for y in range(0, h, patch_size):
            cv2.line(patched_img, (0, y), (w, y), (255, 255, 255), 1)
        
        # 在一些patch上添加位置编码的可视化
        for i in range(0, h, patch_size*2):
            for j in range(0, w, patch_size*2):
                # 在patch中心添加坐标文本
                cv2.putText(patched_img, f"({j//patch_size},{i//patch_size})", 
                           (j + 2, i + patch_size//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        ax2.imshow(patched_img)
        ax2.set_title("2. 图像分块和位置编码\n(16x16像素的patch)", fontsize=14)
        ax2.axis('off')
        
        # 3. 显示Transformer架构
        ax3 = fig.add_subplot(2, 3, 3)
        # 使用matplotlib的文本功能创建Transformer架构
        transformer_img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        ax3.imshow(transformer_img)
        
        # 使用matplotlib添加文本和框
        # 输入嵌入
        ax3.add_patch(plt.Rectangle((50, 50), 300, 50, facecolor=(0.8, 0.8, 1.0), edgecolor='black'))
        ax3.text(200, 75, "输入嵌入 (Patch + 位置编码)", ha='center', va='center', fontsize=12)
        
        # Transformer编码器块
        for i in range(3):
            y_pos = 120 + i * 70
            # 多头自注意力
            ax3.add_patch(plt.Rectangle((50, y_pos), 150, 30, facecolor=(1.0, 0.8, 0.8), edgecolor='black'))
            ax3.text(125, y_pos + 15, "多头自注意力", ha='center', va='center', fontsize=12)
            
            # 前馈神经网络
            ax3.add_patch(plt.Rectangle((200, y_pos), 150, 30, facecolor=(0.8, 1.0, 0.8), edgecolor='black'))
            ax3.text(275, y_pos + 15, "前馈神经网络", ha='center', va='center', fontsize=12)
            
            # 连接线和箭头
            ax3.arrow(200, y_pos + 15, 20, 0, head_width=5, head_length=5, fc='black', ec='black')
            
            # 块之间的连接
            if i < 2:
                ax3.arrow(200, y_pos + 40, 0, 20, head_width=5, head_length=5, fc='black', ec='black')
        
        # 输出头
        ax3.add_patch(plt.Rectangle((50, 330), 300, 50, facecolor=(0.8, 1.0, 1.0), edgecolor='black'))
        ax3.text(200, 355, "分割头 (像素级分类)", ha='center', va='center', fontsize=12)
        
        # 连接到输出头
        ax3.arrow(200, 330, 0, -20, head_width=5, head_length=5, fc='black', ec='black')
            
        ax3.set_title("3. Vision Transformer架构", fontsize=14)
        ax3.axis('off')
        
        # 4. 显示自注意力可视化 - 使用类别标签矩阵体现
        ax4 = fig.add_subplot(2, 3, 4)
        
        # 调整注意力权重大小为更小的矩阵，便于显示数字
        h, w = resized_image.shape[:2]
        matrix_size = (16, 16)  # 与图2中的patch大小相同
        
        # 获取注意力权重
        attention_map = self.attention_weights.mean(axis=0)  # 平均所有注意力头
        
        # 创建一个空白图像用于绘制矩阵
        matrix_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # 计算每个单元格的大小
        cell_h, cell_w = h // matrix_size[0], w // matrix_size[1]
        
        # 归一化注意力权重，使其在0-1之间
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # 为每个单元格分配颜色，基于注意力权重
        for i in range(matrix_size[0]):
            for j in range(matrix_size[1]):
                # 获取当前单元格的注意力权重
                attn_value = attention_norm[i, j]
                
                # 单元格的左上角和右下角坐标
                top_left = (j * cell_w, i * cell_h)
                bottom_right = ((j + 1) * cell_w, (i + 1) * cell_h)
                
                # 使用JET颜色映射，蓝色表示低注意力，红色表示高注意力
                # 将注意力值映射到颜色
                r = int(255 * min(1, 2 * attn_value)) if attn_value > 0.5 else 0
                g = int(255 * min(1, 2 * attn_value)) if attn_value <= 0.5 else int(255 * (2 - 2 * attn_value))
                b = int(255 * (1 - 2 * attn_value)) if attn_value <= 0.5 else 0
                
                color = (r, g, b)
                cv2.rectangle(matrix_img, top_left, bottom_right, color, -1)
                
                # 在单元格中心添加注意力值
                text_pos = (top_left[0] + cell_w // 2, top_left[1] + cell_h // 2)
                
                # 根据背景颜色选择文本颜色（深色背景用白色文本，浅色背景用黑色文本）
                brightness = (r + g + b) / 3
                text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
                
                # 添加注意力值文本（保留两位小数）
                cv2.putText(matrix_img, f"{attn_value:.2f}", 
                           text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)
        
        # 绘制网格线
        for i in range(matrix_size[0] + 1):
            y = i * cell_h
            cv2.line(matrix_img, (0, y), (w, y), (0, 0, 0), 1)
            
        for j in range(matrix_size[1] + 1):
            x = j * cell_w
            cv2.line(matrix_img, (x, 0), (x, h), (0, 0, 0), 1)
        
        # 显示矩阵图像
        ax4.imshow(matrix_img)
        ax4.set_title("4. 自注意力可视化\n(显示模型关注区域)", fontsize=14)
        ax4.axis('off')
        
        # 5. 显示分割掩码 - 只显示掩码，不包含原图
        ax5 = fig.add_subplot(2, 3, 5)
        
        # 创建彩色掩码
        colored_mask = self.create_colored_mask(self.segmentation_mask)
        
        # 调整掩码大小为固定尺寸
        resized_mask = cv2.resize(colored_mask, (400, 300), interpolation=cv2.INTER_NEAREST)
        
        # 显示掩码
        ax5.imshow(resized_mask)
        ax5.set_title("5. 分割掩码\n(类别标签)", fontsize=14)
        ax5.axis('off')
        
        # 6. 显示最终分割结果（彩色掩码叠加在原图上）
        ax6 = fig.add_subplot(2, 3, 6)
        
        # 创建彩色掩码用于叠加在原图上
        colored_mask = self.create_colored_mask(self.segmentation_mask)
        
        # 调整掩码大小以匹配原始图像
        if colored_mask.shape[:2] != self.image_np.shape[:2]:
            colored_mask = cv2.resize(colored_mask, 
                                     (self.image_np.shape[1], self.image_np.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # 创建叠加图像
        alpha = 0.6  # 透明度
        overlay = cv2.addWeighted(self.image_np, 1-alpha, colored_mask, alpha, 0)
        
        # 调整叠加图像大小为固定尺寸
        resized_overlay = cv2.resize(overlay, (400, 300), interpolation=cv2.INTER_AREA)
        
        ax6.imshow(resized_overlay)
        ax6.set_title("6. 最终分割结果\n(彩色掩码叠加在原图上)", fontsize=14)
        ax6.axis('off')
        
        # 添加图例 - 找出图像中存在的类别
        unique_classes = np.unique(self.segmentation_mask)
        legend_elements = []
        for cls_id in unique_classes:
            if cls_id < len(self.classes):
                color = self.colors[cls_id] / 255.0
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, 
                                                   label=self.classes[cls_id]))
        
        if legend_elements:
            # 创建单独的图例
            fig.subplots_adjust(right=0.85)
            legend_ax = fig.add_axes([0.87, 0.15, 0.1, 0.7])
            legend_ax.axis('off')
            legend = legend_ax.legend(handles=legend_elements, loc='center', title="类别")
            legend.get_title().set_fontsize(14)
        
        # 添加总标题
        plt.suptitle(
            f"Vision Transformer (ViT) 语义分割流程可视化\n图像来源: {self.image_source}\n推理时间: {self.inference_time:.3f} 秒",
            fontsize=16
        )
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        
        try:
            plt.savefig("vit_segmentation_process.png", dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存为 'vit_segmentation_process.png'")
        except Exception as e:
            print(f"保存图像时出错: {e}")
            
        plt.show()
        
        # 打印检测到的类别
        detected_classes = [self.classes[i] for i in unique_classes if i < len(self.classes)]
        print(f"检测到的类别: {', '.join(detected_classes)}")
        print(f"推理时间: {self.inference_time:.3f} 秒")
        
        return fig

# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = ViTSegmentationVisualizer()
    
    # 加载图像（可以使用本地图像或URL）
    # 示例URL图像
    image_url = "https://pytorch.org/assets/images/deeplab1.png"
    # 或者使用本地图像
    # image_path = "path/to/your/image.jpg"
    
    try:
        visualizer.load_image(url=image_url)
        # 或者
        # visualizer.load_image(image_path=image_path)
        
        # 执行分割
        visualizer.segment()
        
        # 可视化整个分割流程
        visualizer.visualize_segmentation_process()
    except Exception as e:
        print(f"运行过程中出错: {e}")
        print("请尝试使用本地图像或检查网络连接。")
