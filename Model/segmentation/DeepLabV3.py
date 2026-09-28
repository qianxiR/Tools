import torch
import torch.nn as nn
import torchvision
import numpy as np
# 在导入 matplotlib 之前设置后端
import matplotlib
matplotlib.use('TkAgg')  # 使用Tk后端
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
from io import BytesIO
import time
import warnings
from matplotlib.colors import ListedColormap

# 忽略警告
warnings.filterwarnings("ignore")

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SegmentationVisualizer:
    def __init__(self):
        try:
            # 加载预训练的分割模型 (DeepLabV3 with ResNet50 backbone)
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
            # COCO数据集的类别
            self.classes = [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
                'train', 'tvmonitor'
            ]
            
            # 创建彩色映射用于可视化 - 使用固定的颜色方案而不是随机颜色
            self.colors = np.array([
                [0, 0, 0],        # 背景 - 黑色
                [128, 0, 0],      # 飞机 - 深红色
                [0, 128, 0],      # 自行车 - 深绿色
                [128, 128, 0],    # 鸟 - 橄榄色
                [0, 0, 128],      # 船 - 深蓝色
                [128, 0, 128],    # 瓶子 - 紫色
                [0, 128, 128],    # 公交车 - 青色
                [128, 128, 128],  # 汽车 - 灰色
                [64, 0, 0],       # 猫 - 棕红色
                [192, 0, 0],      # 椅子 - 红色
                [64, 128, 0],     # 牛 - 深绿色
                [192, 128, 0],    # 餐桌 - 橙色
                [64, 0, 128],     # 狗 - 紫色
                [192, 0, 128],    # 马 - 粉色
                [64, 128, 128],   # 摩托车 - 浅青色
                [192, 128, 128],  # 人 - 浅红色
                [0, 64, 0],       # 盆栽植物 - 深绿色
                [128, 64, 0],     # 羊 - 棕色
                [0, 192, 0],      # 沙发 - 绿色
                [128, 192, 0],    # 火车 - 黄绿色
                [0, 64, 128]      # 电视显示器 - 蓝色
            ], dtype=np.uint8)
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("使用模拟模型进行演示...")
            self.model = None
            self.classes = [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
                'train', 'tvmonitor'
            ]
            # 使用相同的固定颜色方案
            self.colors = np.array([
                [0, 0, 0],        # 背景 - 黑色
                [128, 0, 0],      # 飞机 - 深红色
                [0, 128, 0],      # 自行车 - 深绿色
                [128, 128, 0],    # 鸟 - 橄榄色
                [0, 0, 128],      # 船 - 深蓝色
                [128, 0, 128],    # 瓶子 - 紫色
                [0, 128, 128],    # 公交车 - 青色
                [128, 128, 128],  # 汽车 - 灰色
                [64, 0, 0],       # 猫 - 棕红色
                [192, 0, 0],      # 椅子 - 红色
                [64, 128, 0],     # 牛 - 深绿色
                [192, 128, 0],    # 餐桌 - 橙色
                [64, 0, 128],     # 狗 - 紫色
                [192, 0, 128],    # 马 - 粉色
                [64, 128, 128],   # 摩托车 - 浅青色
                [192, 128, 128],  # 人 - 浅红色
                [0, 64, 0],       # 盆栽植物 - 深绿色
                [128, 64, 0],     # 羊 - 棕色
                [0, 192, 0],      # 沙发 - 绿色
                [128, 192, 0],    # 火车 - 黄绿色
                [0, 64, 128]      # 电视显示器 - 蓝色
            ], dtype=np.uint8)
        
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
            self.input_image = self.original_image.resize((513, 513))
            
            return self.original_image
        except Exception as e:
            print(f"加载图像时出错: {e}")
            # 创建一个示例图像
            print("使用示例图像...")
            self.image_np = np.ones((300, 400, 3), dtype=np.uint8) * 255
            # 添加一些形状
            cv2.rectangle(self.image_np, (100, 50), (300, 200), (255, 0, 0), -1)
            cv2.circle(self.image_np, (200, 150), 50, (0, 255, 0), -1)
            self.original_image = Image.fromarray(self.image_np)
            self.input_image = self.original_image.resize((513, 513))
            self.image_source = "示例图像"
            return self.original_image
    
    def preprocess_image(self):
        """预处理图像以适应模型输入"""
        # 转换为张量
        self.input_tensor = torchvision.transforms.functional.to_tensor(self.input_image)
        # 标准化
        self.input_tensor = torchvision.transforms.functional.normalize(
            self.input_tensor, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        # 添加批次维度
        self.input_batch = self.input_tensor.unsqueeze(0)
        
        # 如果有GPU，将数据移到GPU
        if torch.cuda.is_available():
            self.input_batch = self.input_batch.cuda()
            
        return self.input_batch
    
    def segment(self):
        """执行图像分割"""
        # 记录开始时间
        start_time = time.time()
        
        try:
            if self.model is not None:
                # 预处理图像
                self.preprocess_image()
                
                # 使用模型进行预测
                with torch.no_grad():
                    output = self.model(self.input_batch)['out'][0]
                
                # 获取分割结果
                self.output_predictions = output.argmax(0).cpu().numpy()
            else:
                # 如果模型加载失败，创建一些模拟的分割结果
                print("使用模拟分割结果...")
                self.output_predictions = np.zeros((513, 513), dtype=np.uint8)
                # 添加一些模拟的分割区域
                self.output_predictions[100:250, 150:350] = 15  # 人
                self.output_predictions[300:400, 200:300] = 10  # 椅子
                self.output_predictions[50:150, 50:150] = 17    # 羊
        except Exception as e:
            print(f"分割时出错: {e}")
            # 创建一些模拟的分割结果
            print("使用模拟分割结果...")
            self.output_predictions = np.zeros((513, 513), dtype=np.uint8)
            # 添加一些模拟的分割区域
            self.output_predictions[100:250, 150:350] = 15  # 人
            self.output_predictions[300:400, 200:300] = 10  # 椅子
            self.output_predictions[50:150, 50:150] = 17    # 羊
        
        # 记录结束时间
        end_time = time.time()
        self.inference_time = end_time - start_time
        
        return self.output_predictions
    
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
    
    def visualize_segmentation_process(self):
        """可视化整个图像分割流程，使用与场景分类相同的布局"""
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
        
        # 2. 显示预处理后的图像
        ax2 = fig.add_subplot(2, 3, 2)
        # 将标准化后的张量转换回图像
        preprocessed_img = self.input_tensor.cpu().numpy().transpose(1, 2, 0)
        # 反标准化
        preprocessed_img = preprocessed_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        preprocessed_img = np.clip(preprocessed_img, 0, 1)
        # 调整大小为固定尺寸
        preprocessed_img = cv2.resize(preprocessed_img, (400, 300), interpolation=cv2.INTER_AREA)
        ax2.imshow(preprocessed_img)
        ax2.set_title("2. 预处理后的图像\n(调整大小和标准化)", fontsize=14)
        ax2.axis('off')
        
        # 3. 显示编码器-解码器结构
        ax3 = fig.add_subplot(2, 3, 3)
        try:
            # 创建一个空白的背景
            ax3.set_xlim(0, 10)
            ax3.set_ylim(0, 10)
            ax3.set_facecolor('#f0f0f0')  # 浅灰色背景
            
            # 绘制编码器部分 (左侧)
            encoder_box1 = plt.Rectangle((1, 7), 2, 2, fc='#6666FF', ec='black', alpha=0.8)
            encoder_box2 = plt.Rectangle((1.5, 4), 1.5, 1.5, fc='#8888FF', ec='black', alpha=0.8)
            encoder_box3 = plt.Rectangle((1.75, 2), 1, 1, fc='#AAAAFF', ec='black', alpha=0.8)
            
            # 绘制解码器部分 (右侧)
            decoder_box1 = plt.Rectangle((7, 7), 2, 2, fc='#6666FF', ec='black', alpha=0.8)
            decoder_box2 = plt.Rectangle((7, 4), 1.5, 1.5, fc='#8888FF', ec='black', alpha=0.8)
            decoder_box3 = plt.Rectangle((7.25, 2), 1, 1, fc='#AAAAFF', ec='black', alpha=0.8)
            
            # 添加所有矩形到图中
            ax3.add_patch(encoder_box1)
            ax3.add_patch(encoder_box2)
            ax3.add_patch(encoder_box3)
            ax3.add_patch(decoder_box1)
            ax3.add_patch(decoder_box2)
            ax3.add_patch(decoder_box3)
            
            # 绘制连接线 - 垂直连接
            ax3.plot([2, 2], [7, 5.5], 'k-', lw=1.5)
            ax3.plot([2.25, 2.25], [4, 3], 'k-', lw=1.5)
            ax3.plot([8, 8], [7, 5.5], 'k-', lw=1.5)
            ax3.plot([7.75, 7.75], [4, 3], 'k-', lw=1.5)
            
            # 绘制跳跃连接 - 水平连接
            ax3.plot([3, 7], [8], 'k-', lw=1.5)
            ax3.plot([3, 7], [4.75], 'k-', lw=1.5)
            ax3.plot([2.75, 7.25], [2.5], 'k-', lw=1.5)
            
            # 添加箭头
            ax3.arrow(6.5, 8, 0.3, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
            ax3.arrow(6.5, 4.75, 0.3, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
            ax3.arrow(6.75, 2.5, 0.3, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
            
            # 添加文本标签
            ax3.text(2, 9.5, "编码器", fontsize=12, ha='center')
            ax3.text(8, 9.5, "解码器", fontsize=12, ha='center')
            ax3.text(5, 8.3, "跳跃连接", fontsize=10, ha='center')
            
            # 添加标题
            ax3.text(5, 6, "编码器-解码器网络结构", fontsize=12, ha='center')
            
        except Exception as e:
            print(f"创建编码器-解码器示意图时出错: {e}")
            ax3.text(0.5, 0.5, "编码器-解码器结构", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=20)
            
        ax3.set_title("3. 编码器-解码器结构", fontsize=14)
        ax3.axis('off')
        
        # 4. 显示分割掩码（类别索引）
        ax4 = fig.add_subplot(2, 3, 4)
        # 创建自定义颜色映射
        cmap = ListedColormap(self.colors / 255.0)
        
        # 调整预测掩码大小为固定尺寸
        resized_mask = cv2.resize(self.output_predictions, (400, 300), interpolation=cv2.INTER_NEAREST)
        
        ax4.imshow(resized_mask, cmap=cmap, interpolation='nearest')
        ax4.set_title("4. 分割掩码\n(类别索引)", fontsize=14)
        ax4.axis('off')
        
        # 5. 显示最终分割结果（彩色掩码）
        ax5 = fig.add_subplot(2, 3, 5)
        
        # 创建彩色掩码
        colored_mask = self.create_colored_mask(self.output_predictions)
        
        # 调整掩码大小为固定尺寸
        resized_colored_mask = cv2.resize(colored_mask, (400, 300), interpolation=cv2.INTER_NEAREST)
        
        ax5.imshow(resized_colored_mask)
        ax5.set_title("5. 分割掩码\n(彩色表示)", fontsize=14)
        ax5.axis('off')
        
        # 6. 显示最终分割结果（彩色掩码叠加在原图上）
        ax6 = fig.add_subplot(2, 3, 6)
        
        # 确保原图是3通道
        if len(self.image_np.shape) > 3:
            image_for_overlay = self.image_np[:, :, :3]
        else:
            image_for_overlay = self.image_np
        
        # 调整掩码大小以匹配原始图像
        if colored_mask.shape[:2] != image_for_overlay.shape[:2]:
            colored_mask = cv2.resize(colored_mask, 
                                     (image_for_overlay.shape[1], image_for_overlay.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # 创建叠加图像
        alpha = 0.6  # 透明度
        overlay = cv2.addWeighted(image_for_overlay, 1-alpha, colored_mask, alpha, 0)
        
        # 调整叠加图像大小为固定尺寸
        resized_overlay = cv2.resize(overlay, (400, 300), interpolation=cv2.INTER_AREA)
        
        ax6.imshow(resized_overlay)
        ax6.set_title("6. 最终分割结果\n(彩色掩码叠加在原图上)", fontsize=14)
        ax6.axis('off')
        
        # 添加总标题
        plt.suptitle(
            f"图像分割流程可视化\n图像来源: {self.image_source}\n推理时间: {self.inference_time:.3f} 秒",
            fontsize=16
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 添加图例 - 使用单独的图例而不是修改布局
        # 找出图像中存在的类别
        unique_classes = np.unique(self.output_predictions)
        legend_elements = []
        for cls_id in unique_classes:
            if cls_id < len(self.classes):
                color = self.colors[cls_id] / 255.0
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, 
                                                   label=self.classes[cls_id]))
        
        if legend_elements:
            # 在图像下方添加图例
            fig.legend(handles=legend_elements, loc='lower center', 
                      bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(legend_elements)))
        
        try:
            plt.savefig("segmentation_process.png", dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存为 'segmentation_process.png'")
        except Exception as e:
            print(f"保存图像时出错: {e}")
            
        # 打印检测到的类别
        detected_classes = [self.classes[i] for i in unique_classes if i < len(self.classes)]
        print(f"检测到的类别: {', '.join(detected_classes)}")
        print(f"推理时间: {self.inference_time:.3f} 秒")
        
        return fig

# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = SegmentationVisualizer()
    
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
