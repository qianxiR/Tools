import torch
import torch.nn as nn
import torchvision
import numpy as np
# 在导入 matplotlib 之前设置后端
import matplotlib
# matplotlib.use('TkAgg')  # 使用Tk后端
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import yolov5
from PIL import Image
import requests
from io import BytesIO
import time
import warnings
import sys
import os

# 忽略警告
warnings.filterwarnings("ignore")

# 尝试导入 models.experimental，如果失败则设置正确的路径
try:
    from models.experimental import attempt_load
except ModuleNotFoundError:
    # 检查是否存在 yolov5 目录，如果不存在则克隆
    if not os.path.exists('yolov5'):
        print("克隆 YOLOv5 仓库...")
        os.system('git clone https://github.com/ultralytics/yolov5')
    
    # 将 yolov5 目录添加到 Python 路径
    yolov5_path = os.path.join(os.getcwd(), 'yolov5')
    sys.path.append(yolov5_path)
    
    # 现在尝试导入
    try:
        from yolov5.models.experimental import attempt_load
    except ModuleNotFoundError:
        print("无法导入 models.experimental，将使用备用方法")
        # 定义一个空的 attempt_load 函数作为备用
        def attempt_load(*args, **kwargs):
            print("使用模拟的 attempt_load 函数")
            return None

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class YOLODetectionVisualizer:
    def __init__(self):
        # 确保导入torch
        import torch
        
        try:
            # 加载预训练的YOLOv5模型
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # COCO数据集的类别
            self.classes = self.model.names
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("尝试使用备用方法加载模型...")
            # 备用方法：如果torch hub加载失败，可以尝试直接从本地加载
            try:
                # 检查是否存在 yolov5 目录，如果不存在则克隆
                if not os.path.exists('yolov5'):
                    print("克隆 YOLOv5 仓库...")
                    os.system('git clone https://github.com/ultralytics/yolov5')
                
                # 将 yolov5 目录添加到 Python 路径
                yolov5_path = os.path.join(os.getcwd(), 'yolov5')
                if yolov5_path not in sys.path:
                    sys.path.append(yolov5_path)
                
                # 检查模型文件是否存在，如果不存在则下载
                model_path = 'yolov5s.pt'
                if not os.path.exists(model_path):
                    print(f"下载 {model_path} 模型...")
                    torch.hub.download_url_to_file(
                        'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt',
                        model_path
                    )
                
                # 加载模型
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = attempt_load(model_path, device=device)
                # 设置类别名称
                self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
                                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
                
                # 创建一个包装器，使本地模型的接口与torch.hub模型一致
                original_model = self.model
                
                class ModelWrapper:
                    def __init__(self, model):
                        self.model = model
                        self.device = next(model.parameters()).device
                    
                    def __call__(self, img):
                        # 处理不同类型的输入
                        if isinstance(img, Image.Image):
                            # 转换PIL图像为tensor
                            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                            img = img.unsqueeze(0).to(self.device)
                        elif isinstance(img, np.ndarray):
                            # 转换numpy数组为tensor
                            if len(img.shape) == 3 and img.shape[2] == 3:  # HWC
                                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                            else:  # 假设已经是CHW格式
                                img = torch.from_numpy(img).float() / 255.0
                            img = img.unsqueeze(0).to(self.device)
                        
                        # 推理
                        with torch.no_grad():
                            pred = self.model(img)[0]
                        
                        # 创建一个类似于torch.hub模型的结果对象
                        class Results:
                            def __init__(self, pred):
                                self.xyxy = [pred]  # 兼容性
                        
                        return Results(pred)
                
                # 替换模型为包装后的模型
                self.model = ModelWrapper(original_model)
                
            except Exception as e2:
                print(f"备用方法也失败: {e2}")
                print("使用模拟模型进行演示...")
                # 创建一个模拟模型用于演示
                self.model = None
                self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
        
        # 设置不同类别的颜色
        np.random.seed(42)  # 固定随机种子以获得一致的颜色
        self.colors = np.random.uniform(0, 1, size=(len(self.classes), 3))
        
    def load_image(self, image_path=None, url=None):
        """加载图像，可以是本地路径或URL"""
        try:
            if url:
                response = requests.get(url)
                self.original_image = Image.open(BytesIO(response.content)).convert("RGB")
                self.image_source = f"URL: {url}"
            elif image_path:
                self.original_image = Image.open(image_path).convert("RGB")
                self.image_source = f"文件: {image_path}"
            else:
                raise ValueError("必须提供image_path或url参数")
                
            # 确保转换为numpy数组
            self.image_np = np.array(self.original_image)
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
            self.image_source = "示例图像"
            return self.original_image
    
    def detect(self):
        """使用YOLO模型进行目标检测"""
        # 记录开始时间
        start_time = time.time()
        
        try:
            if self.model is not None:
                # 使用模型进行预测
                if isinstance(self.original_image, Image.Image):
                    # 如果是PIL图像，需要转换为模型可接受的格式
                    img = self.original_image
                    
                    # 使用模型的预测方法
                    if hasattr(self.model, 'predict'):
                        results = self.model.predict(img)
                        self.predictions = results[0].boxes.xyxy.cpu().numpy()  # 获取边界框
                        # 添加置信度和类别ID
                        conf = results[0].boxes.conf.cpu().numpy()
                        cls = results[0].boxes.cls.cpu().numpy()
                        self.predictions = np.column_stack((self.predictions, conf, cls))
                    else:
                        # 直接使用PIL图像
                        results = self.model(img)
                        self.predictions = results.xyxy[0].cpu().numpy()  # xyxy格式的边界框
                else:
                    # 如果已经是numpy数组，转换为tensor
                    img_tensor = torch.from_numpy(self.image_np).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
                    
                    # 确保在正确的设备上
                    if next(self.model.parameters()).is_cuda:
                        img_tensor = img_tensor.cuda()
                    
                    # 使用模型进行预测
                    results = self.model(img_tensor)
                    self.predictions = results.xyxy[0].cpu().numpy()
        except:
            pass
        # 记录结束时间
        end_time = time.time()
        self.inference_time = end_time - start_time
        
        return self.predictions
    
    def visualize_detection_process(self):
        """可视化整个目标检测流程，使用与场景分类和分割相同的布局"""
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
        # 创建预处理图像的模拟
        try:
            # 转换为RGB格式确保一致性
            if len(self.image_np.shape) == 3:
                preprocessed_img = cv2.cvtColor(self.image_np, cv2.COLOR_RGB2BGR)
                preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
            else:
                preprocessed_img = self.image_np
                
            # 调整大小为模型输入尺寸
            preprocessed_img = cv2.resize(preprocessed_img, (640, 640))
            # 再调整回显示尺寸
            preprocessed_img = cv2.resize(preprocessed_img, (400, 300), interpolation=cv2.INTER_AREA)
            ax2.imshow(preprocessed_img)
        except Exception as e:
            print(f"创建预处理图像时出错: {e}")
            ax2.imshow(resized_image)
            
        ax2.set_title("2. 预处理后的图像\n(调整大小和标准化)", fontsize=14)
        ax2.axis('off')
        
        # 3. 显示YOLO网络结构
        ax3 = fig.add_subplot(2, 3, 3)
        try:
            # 创建一个YOLO网络结构示意图
            ax3.set_xlim(0, 10)
            ax3.set_ylim(0, 10)
            ax3.set_facecolor('#f0f0f0')  # 浅灰色背景
            
            # 绘制主干网络
            backbone = plt.Rectangle((1, 3), 2, 4, fc='#6666FF', ec='black', alpha=0.8)
            ax3.add_patch(backbone)
            ax3.text(2, 5, "主干网络\n(特征提取)", ha='center', va='center')
            
            # 绘制特征金字塔
            neck1 = plt.Rectangle((4, 6), 1.5, 1, fc='#FF6666', ec='black', alpha=0.8)
            neck2 = plt.Rectangle((4, 5), 1.5, 1, fc='#FF6666', ec='black', alpha=0.8)
            neck3 = plt.Rectangle((4, 4), 1.5, 1, fc='#FF6666', ec='black', alpha=0.8)
            ax3.add_patch(neck1)
            ax3.add_patch(neck2)
            ax3.add_patch(neck3)
            ax3.text(4.75, 3.5, "特征金字塔\n(多尺度特征)", ha='center', va='center')
            
            # 绘制检测头
            head1 = plt.Rectangle((7, 6), 1.5, 1, fc='#66FF66', ec='black', alpha=0.8)
            head2 = plt.Rectangle((7, 5), 1.5, 1, fc='#66FF66', ec='black', alpha=0.8)
            head3 = plt.Rectangle((7, 4), 1.5, 1, fc='#66FF66', ec='black', alpha=0.8)
            ax3.add_patch(head1)
            ax3.add_patch(head2)
            ax3.add_patch(head3)
            ax3.text(7.75, 3.5, "检测头\n(边界框预测)", ha='center', va='center')
            
            # 绘制连接线
            ax3.plot([3, 4], [6.5, 6.5], 'k-', lw=1.5)
            ax3.plot([3, 4], [5.5, 5.5], 'k-', lw=1.5)
            ax3.plot([3, 4], [4.5, 4.5], 'k-', lw=1.5)
            
            ax3.plot([5.5, 7], [6.5, 6.5], 'k-', lw=1.5)
            ax3.plot([5.5, 7], [5.5, 5.5], 'k-', lw=1.5)
            ax3.plot([5.5, 7], [4.5, 4.5], 'k-', lw=1.5)
            
            # 添加标题
            ax3.text(5, 8, "YOLO网络结构", fontsize=12, ha='center')
            
        except Exception as e:
            print(f"创建YOLO网络结构示意图时出错: {e}")
            ax3.text(0.5, 0.5, "YOLO网络结构", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=20)
            
        ax3.set_title("3. YOLO网络结构", fontsize=14)
        ax3.axis('off')
        
        # 4. 显示网格划分和锚框
        ax4 = fig.add_subplot(2, 3, 4)
        resized_image_copy = resized_image.copy()
        ax4.imshow(resized_image_copy)
        
        try:
            # 绘制网格
            h, w = resized_image_copy.shape[:2]
            grid_size = 13  # YOLOv5s默认使用的网格大小
            cell_h, cell_w = h / grid_size, w / grid_size
            
            # 绘制水平线
            for i in range(1, grid_size):
                y = i * cell_h
                ax4.axhline(y=y, color='white', linestyle='-', alpha=0.3)
                
            # 绘制垂直线
            for i in range(1, grid_size):
                x = i * cell_w
                ax4.axvline(x=x, color='white', linestyle='-', alpha=0.3)
            
            # 在一些网格单元上绘制锚框示例
            anchor_scales = [(0.5, 1.0), (1.0, 0.5), (0.75, 0.75)]
            for i in range(3, grid_size, 4):
                for j in range(3, grid_size, 4):
                    center_x = j * cell_w + cell_w / 2
                    center_y = i * cell_h + cell_h / 2
                    
                    for scale_w, scale_h in anchor_scales:
                        width = cell_w * scale_w * 2
                        height = cell_h * scale_h * 2
                        
                        # 创建矩形
                        rect = patches.Rectangle(
                            (center_x - width/2, center_y - height/2),
                            width, height,
                            linewidth=1, edgecolor='yellow', facecolor='none', alpha=0.5
                        )
                        ax4.add_patch(rect)
        except Exception as e:
            print(f"绘制网格和锚框时出错: {e}")
        
        ax4.set_title("4. 网格划分和锚框\n(特征图映射)", fontsize=14)
        ax4.axis('off')
        
        # 5. 显示NMS过程
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.imshow(resized_image)
        
        try:
            # 确保 h 和 w 已定义
            h, w = resized_image.shape[:2]
            
            # 生成一些模拟的候选框（比实际检测结果更多）
            num_extra_boxes = min(50, len(self.predictions) * 3)
            
            # 复制并稍微修改实际检测框，模拟冗余检测
            candidate_boxes = []
            
            for box in self.predictions:
                x1, y1, x2, y2, conf, cls = box
                # 调整坐标以适应调整大小后的图像
                x1 = x1 * w / self.image_np.shape[1]
                y1 = y1 * h / self.image_np.shape[0]
                x2 = x2 * w / self.image_np.shape[1]
                y2 = y2 * h / self.image_np.shape[0]
                candidate_boxes.append([x1, y1, x2, y2, conf, cls])
                
                # 添加一些随机偏移的框
                for _ in range(2):
                    offset_x = np.random.normal(0, 10)
                    offset_y = np.random.normal(0, 10)
                    new_box = [
                        max(0, x1 + offset_x),
                        max(0, y1 + offset_y),
                        min(w, x2 + offset_x),
                        min(h, y2 + offset_y),
                        conf * np.random.uniform(0.5, 0.9),
                        cls
                    ]
                    candidate_boxes.append(new_box)
            
            # 按置信度排序
            sorted_boxes = sorted(candidate_boxes, key=lambda x: x[4], reverse=True)
            
            # 模拟NMS过程
            kept_boxes = []
            for box in sorted_boxes:
                x1, y1, x2, y2, conf, cls = box
                
                # 检查是否与已保留的框重叠过大
                keep = True
                for kept_box in kept_boxes:
                    kx1, ky1, kx2, ky2, _, _ = kept_box
                    
                    # 计算IoU
                    inter_x1 = max(x1, kx1)
                    inter_y1 = max(y1, ky1)
                    inter_x2 = min(x2, kx2)
                    inter_y2 = min(y2, ky2)
                    
                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        box_area = (x2 - x1) * (y2 - y1)
                        kept_area = (kx2 - kx1) * (ky2 - ky1)
                        iou = inter_area / (box_area + kept_area - inter_area)
                        
                        if iou > 0.5:  # NMS阈值
                            keep = False
                            break
                
                if keep:
                    kept_boxes.append(box)
                    
                    # 绘制保留的框
                    width, height = x2 - x1, y2 - y1
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='magenta', facecolor='none'
                    )
                    ax5.add_patch(rect)
        except Exception as e:
            print(f"执行NMS时出错: {e}")
            kept_boxes = self.predictions
        
        ax5.set_title("5. 非极大值抑制 (NMS)\n(去除冗余框)", fontsize=14)
        ax5.axis('off')
        
        # 6. 显示最终检测结果
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(resized_image)
        
        try:
            # 绘制最终检测框和标签
            for box in self.predictions:
                x1, y1, x2, y2, conf, cls_id = box
                # 调整坐标以适应调整大小后的图像
                x1 = x1 * w / self.image_np.shape[1]
                y1 = y1 * h / self.image_np.shape[0]
                x2 = x2 * w / self.image_np.shape[1]
                y2 = y2 * h / self.image_np.shape[0]
                width, height = x2 - x1, y2 - y1
                
                # 获取类别和颜色
                cls_id = int(cls_id)
                cls_name = self.classes[cls_id] if cls_id < len(self.classes) else f"类别{cls_id}"
                color = self.colors[cls_id % len(self.colors)]
                
                # 创建矩形
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax6.add_patch(rect)
                
                # 添加标签
                label = f"{cls_name}: {conf:.2f}"
                ax6.text(
                    x1, y1 - 5, label,
                    color='white', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.7, pad=2)
                )
        except Exception as e:
            print(f"绘制最终结果时出错: {e}")
        
        ax6.set_title("6. 最终检测结果\n(检测到的物体)", fontsize=14)
        ax6.axis('off')
        
        # 添加总标题
        plt.suptitle(
            f"YOLO目标检测流程可视化\n图像来源: {self.image_source}\n推理时间: {self.inference_time:.3f} 秒",
            fontsize=16
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        try:
            plt.savefig("yolo_detection_process.png", dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存为 'yolo_detection_process.png'")
        except Exception as e:
            print(f"保存图像时出错: {e}")
            
        # 注释掉或移除这一行
        # plt.show()
        
        print(f"检测到 {len(self.predictions)} 个物体，推理时间: {self.inference_time:.3f} 秒")
        
        return fig

# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = YOLODetectionVisualizer()
    
    # 加载图像（可以使用本地图像或URL）
    # 示例URL图像
    image_url = "https://ultralytics.com/images/zidane.jpg"
    # 或者使用本地图像
    # image_path = "path/to/your/image.jpg"
    
    try:
        visualizer.load_image(url=image_url)
        # 或者
        # visualizer.load_image(image_path=image_path)
        
        # 执行检测
        visualizer.detect()
        
        # 可视化整个检测流程
        visualizer.visualize_detection_process()
    except Exception as e:
        print(f"运行过程中出错: {e}")
        print("请尝试使用本地图像或检查网络连接。")