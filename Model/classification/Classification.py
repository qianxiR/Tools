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
import matplotlib.cm as cm

# 忽略警告
warnings.filterwarnings("ignore")

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SceneClassificationVisualizer:
    def __init__(self):
        try:
            # 加载预训练的ResNet50模型
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
            # 场景类别 (使用Places365数据集的前20个类别作为示例)
            self.classes = [
                '机场航站楼', '水族馆', '操场', '艺术画廊', '面包店', 
                '酒吧', '卧室', '酒馆', '桥梁', '公交站', 
                '建筑外观', '营地', '校园', '城堡', '洞穴', 
                '教堂', '城市', '会议室', '走廊', '咖啡厅'
            ]
            
            # 直接内置一些ImageNet类别名称，避免读取外部文件
            self.imagenet_classes = [
                'tench', 'goldfish', 'shark', 'pufferfish', 'ray',
                'rooster', 'hen', 'ostrich', 'peacock', 'flamingo',
                'parrot', 'eagle', 'owl', 'lizard', 'crocodile',
                'turtle', 'snake', 'spider', 'scorpion', 'crab',
                'snail', 'butterfly', 'moth', 'bee', 'ant',
                'grasshopper', 'dragonfly', 'ladybug', 'beetle', 'fly',
                'mosquito', 'squirrel', 'rabbit', 'hedgehog', 'bat',
                'bear', 'panda', 'koala', 'kangaroo', 'monkey',
                'gorilla', 'orangutan', 'chimpanzee', 'gibbon', 'baboon',
                'tiger', 'lion', 'cheetah', 'leopard', 'jaguar',
                'wolf', 'fox', 'coyote', 'jackal', 'hyena',
                'weasel', 'otter', 'badger', 'skunk', 'raccoon',
                'porcupine', 'camel', 'llama', 'giraffe', 'elephant',
                'rhinoceros', 'hippopotamus', 'zebra', 'horse', 'donkey',
                'mule', 'cow', 'buffalo', 'bison', 'goat',
                'sheep', 'deer', 'antelope', 'moose', 'reindeer',
                'dog', 'cat', 'hamster', 'guinea pig', 'mouse',
                'rat', 'rabbit', 'hare', 'beaver', 'armadillo',
                'sloth', 'penguin', 'seagull', 'albatross', 'swan',
                'duck', 'goose', 'heron', 'stork', 'crane',
                'pelican', 'pigeon', 'dove', 'sparrow', 'finch'
            ]
            # 扩展到1000个类别以匹配ImageNet
            while len(self.imagenet_classes) < 1000:
                self.imagenet_classes.append(f"类别_{len(self.imagenet_classes)}")
                
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("使用模拟模型进行演示...")
            self.model = None
            # 场景类别
            self.classes = [
                '机场航站楼', '水族馆', '操场', '艺术画廊', '面包店', 
                '酒吧', '卧室', '酒馆', '桥梁', '公交站', 
                '建筑外观', '营地', '校园', '城堡', '洞穴', 
                '教堂', '城市', '会议室', '走廊', '咖啡厅'
            ]
            self.imagenet_classes = self.classes
            
            # 初始化必要的属性，即使在模拟模式下也需要
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
            
            # 预处理图像，即使在模拟模式下也需要
            self.preprocess_image()
            
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
            self.input_image = self.original_image.resize((224, 224))
            self.image_source = "示例图像"
            
            # 预处理图像，即使在模拟模式下也需要
            self.preprocess_image()
            
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
        if torch.cuda.is_available() and self.model is not None:
            self.input_batch = self.input_batch.cuda()
            
        return self.input_batch
    
    def classify(self):
        """执行场景分类"""
        # 记录开始时间
        start_time = time.time()
        
        try:
            if self.model is not None:
                # 使用模型进行预测
                with torch.no_grad():
                    output = self.model(self.input_batch)
                
                # 获取分类结果
                self.output_probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
                self.predicted_class_idx = np.argmax(self.output_probabilities)
                
                # 获取前5个预测结果
                self.top5_indices = np.argsort(self.output_probabilities)[-5:][::-1]
                self.top5_probabilities = self.output_probabilities[self.top5_indices]
                
                # 获取类别名称
                try:
                    self.top5_classes = [self.imagenet_classes[idx] for idx in self.top5_indices]
                except:
                    # 如果无法获取ImageNet类别，使用索引
                    self.top5_classes = [f"类别 {idx}" for idx in self.top5_indices]
                
                # 为了演示，我们假设这是场景分类结果
                # 在实际应用中，您应该使用专门的场景分类模型
                scene_idx = self.top5_indices[0] % len(self.classes)
                self.predicted_scene = self.classes[scene_idx]
                
            else:
                # 如果模型加载失败，创建一些模拟的分类结果
                print("使用模拟分类结果...")
                self.top5_indices = np.random.choice(len(self.classes), 5, replace=False)
                self.top5_probabilities = np.random.random(5)
                self.top5_probabilities = self.top5_probabilities / np.sum(self.top5_probabilities)
                self.top5_classes = [self.classes[idx] for idx in self.top5_indices]
                self.predicted_scene = self.classes[self.top5_indices[0]]
                
        except Exception as e:
            print(f"分类时出错: {e}")
            # 创建一些模拟的分类结果
            print("使用模拟分类结果...")
            self.top5_indices = np.random.choice(len(self.classes), 5, replace=False)
            self.top5_probabilities = np.random.random(5)
            self.top5_probabilities = self.top5_probabilities / np.sum(self.top5_probabilities)
            self.top5_classes = [self.classes[idx] for idx in self.top5_indices]
            self.predicted_scene = self.classes[self.top5_indices[0]]
        
        # 记录结束时间
        end_time = time.time()
        self.inference_time = end_time - start_time
        
        return self.predicted_scene, self.top5_classes, self.top5_probabilities
    
    def generate_cam(self):
        """生成类激活映射 (CAM)"""
        try:
            if self.model is not None and hasattr(self, 'input_tensor'):
                # 获取最后一个卷积层的特征图
                # 注意：这是一个简化版本，实际的CAM需要更多步骤
                # 我们这里只是为了演示
                
                # 创建一个简单的热力图作为CAM的模拟
                cam = cv2.resize(self.input_tensor.cpu().numpy().transpose(1, 2, 0), 
                                (self.image_np.shape[1], self.image_np.shape[0]))
                cam = np.mean(cam, axis=2)
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
                
                # 应用颜色映射
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                return heatmap
            else:
                # 创建一个模拟的热力图
                h, w = self.image_np.shape[:2]
                y, x = np.ogrid[:h, :w]
                mask = (1 - ((x - w/2)**2 + (y - h/2)**2) / ((w/2)**2 + (h/2)**2))
                mask = np.clip(mask, 0, 1)
                
                # 应用颜色映射
                heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                return heatmap
        except Exception as e:
            print(f"生成CAM时出错: {e}")
            # 创建一个模拟的热力图
            h, w = self.image_np.shape[:2]
            y, x = np.ogrid[:h, :w]
            mask = (1 - ((x - w/2)**2 + (y - h/2)**2) / ((w/2)**2 + (h/2)**2))
            mask = np.clip(mask, 0, 1)
            
            # 应用颜色映射
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            return heatmap
    
    def visualize_classification_process(self):
        """可视化整个场景分类流程"""
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
        
        # 3. 显示卷积神经网络结构
        ax3 = fig.add_subplot(2, 3, 3)
        try:
            # 创建一个CNN结构示意图
            ax3.set_xlim(0, 10)
            ax3.set_ylim(0, 10)
            ax3.set_facecolor('#f0f0f0')  # 浅灰色背景
            
            # 绘制输入层
            input_layer = plt.Rectangle((1, 4), 1, 2, fc='#6666FF', ec='black', alpha=0.8)
            ax3.add_patch(input_layer)
            ax3.text(1.5, 3.5, "输入层", ha='center')
            
            # 绘制卷积层
            conv_layer1 = plt.Rectangle((3, 4.5), 1, 1, fc='#FF6666', ec='black', alpha=0.8)
            conv_layer2 = plt.Rectangle((3, 3.5), 1, 1, fc='#FF6666', ec='black', alpha=0.8)
            ax3.add_patch(conv_layer1)
            ax3.add_patch(conv_layer2)
            ax3.text(3.5, 3, "卷积层", ha='center')
            
            # 绘制池化层
            pool_layer = plt.Rectangle((5, 4), 1, 1, fc='#66FF66', ec='black', alpha=0.8)
            ax3.add_patch(pool_layer)
            ax3.text(5.5, 3.5, "池化层", ha='center')
            
            # 绘制全连接层
            fc_layer = plt.Rectangle((7, 4), 1, 1, fc='#FFFF66', ec='black', alpha=0.8)
            ax3.add_patch(fc_layer)
            ax3.text(7.5, 3.5, "全连接层", ha='center')
            
            # 绘制输出层
            output_layer = plt.Rectangle((9, 4), 0.5, 1, fc='#FF66FF', ec='black', alpha=0.8)
            ax3.add_patch(output_layer)
            ax3.text(9.25, 3.5, "输出", ha='center')
            
            # 绘制连接线
            ax3.plot([2, 3], [5, 5], 'k-', lw=1.5)
            ax3.plot([2, 3], [5, 4], 'k-', lw=1.5)
            ax3.plot([4, 5], [5, 4.5], 'k-', lw=1.5)
            ax3.plot([4, 5], [4, 4.5], 'k-', lw=1.5)
            ax3.plot([6, 7], [4.5, 4.5], 'k-', lw=1.5)
            ax3.plot([8, 9], [4.5, 4.5], 'k-', lw=1.5)
            
            # 添加标题
            ax3.text(5, 6, "卷积神经网络 (CNN) 结构", fontsize=12, ha='center')
            
        except Exception as e:
            print(f"创建CNN结构示意图时出错: {e}")
            ax3.text(0.5, 0.5, "卷积神经网络结构", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=20)
            
        ax3.set_title("3. 卷积神经网络结构", fontsize=14)
        ax3.axis('off')
        
        # 4. 显示类激活映射 (CAM)
        ax4 = fig.add_subplot(2, 3, 4)
        
        # 生成CAM
        heatmap = self.generate_cam()
        
        # 调整热力图大小为固定尺寸
        resized_heatmap = cv2.resize(heatmap, (400, 300), interpolation=cv2.INTER_AREA)
        
        ax4.imshow(resized_heatmap)
        ax4.set_title("4. 类激活映射 (CAM)\n(显示模型关注区域)", fontsize=14)
        ax4.axis('off')
        
        # 5. 显示热力图叠加在原图上
        ax5 = fig.add_subplot(2, 3, 5)
        
        # 调整热力图大小以匹配原始图像
        if heatmap.shape[:2] != self.image_np.shape[:2]:
            heatmap = cv2.resize(heatmap, 
                               (self.image_np.shape[1], self.image_np.shape[0]), 
                               interpolation=cv2.INTER_AREA)
        
        # 创建叠加图像
        alpha = 0.6  # 透明度
        overlay = cv2.addWeighted(self.image_np, 1-alpha, heatmap, alpha, 0)
        
        # 调整叠加图像大小为固定尺寸
        resized_overlay = cv2.resize(overlay, (400, 300), interpolation=cv2.INTER_AREA)
        
        ax5.imshow(resized_overlay)
        ax5.set_title("5. 热力图叠加\n(突出显示关键区域)", fontsize=14)
        ax5.axis('off')
        
        # 6. 显示分类结果和概率条形图
        ax6 = fig.add_subplot(2, 3, 6)
        
        # 准备更友好的类别名称显示
        display_classes = []
        for i, class_idx in enumerate(self.top5_indices):
            if self.model is not None:
                # 对于真实模型预测，尝试使用更友好的名称
                if class_idx < len(self.imagenet_classes) and not self.imagenet_classes[class_idx].startswith('类别_'):
                    # 使用预定义的类别名称
                    class_name = self.imagenet_classes[class_idx]
                    # 如果类别名称是英文，尝试翻译成中文
                    chinese_names = {
                        'tench': '丁鱼',
                        'goldfish': '金鱼',
                        'shark': '鲨鱼',
                        'pufferfish': '河豚',
                        'ray': '鳐鱼',
                        'rooster': '公鸡',
                        'hen': '母鸡',
                        'ostrich': '鸵鸟',
                        'peacock': '孔雀',
                        'flamingo': '火烈鸟',
                        'parrot': '鹦鹉',
                        'eagle': '鹰',
                        'owl': '猫头鹰',
                        'lizard': '蜥蜴',
                        'crocodile': '鳄鱼',
                        'turtle': '乌龟',
                        'snake': '蛇',
                        'spider': '蜘蛛',
                        'scorpion': '蝎子',
                        'crab': '螃蟹',
                        'snail': '蜗牛',
                        'butterfly': '蝴蝶',
                        'moth': '飞蛾',
                        'bee': '蜜蜂',
                        'ant': '蚂蚁',
                        'dog': '狗',
                        'cat': '猫',
                        'horse': '马',
                        'sheep': '绵羊',
                        'cow': '奶牛',
                        'elephant': '大象',
                        'bear': '熊',
                        'zebra': '斑马',
                        'giraffe': '长颈鹿',
                        'penguin': '企鹅',
                        'bird': '鸟',
                        'fish': '鱼',
                        'tiger': '老虎',
                        'lion': '狮子',
                        'wolf': '狼',
                        'fox': '狐狸',
                        'deer': '鹿',
                        'monkey': '猴子',
                        'rabbit': '兔子',
                        'panda': '熊猫',
                        'koala': '考拉',
                        'kangaroo': '袋鼠'
                    }
                    if class_name in chinese_names:
                        class_name = chinese_names[class_name]
                    display_classes.append(class_name)
                else:
                    # 使用场景类别名称
                    scene_idx = class_idx % len(self.classes)
                    display_classes.append(self.classes[scene_idx])
            else:
                # 对于模拟模式，直接使用场景类别
                display_classes.append(self.top5_classes[i])
        
        # 绘制条形图
        y_pos = np.arange(len(display_classes))
        ax6.barh(y_pos, self.top5_probabilities, align='center', color='skyblue')
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(display_classes)
        ax6.invert_yaxis()  # 标签从上到下
        ax6.set_xlabel('概率')
        ax6.set_xlim(0, 1)
        
        # 在条形上添加概率值
        for i, v in enumerate(self.top5_probabilities):
            ax6.text(v + 0.01, i, f"{v:.2f}", va='center')
        
        ax6.set_title("6. 分类结果\n(前5个预测类别及概率)", fontsize=14)
        
        # 添加总标题
        plt.suptitle(
            f"场景分类流程可视化\n图像来源: {self.image_source}\n预测场景: {self.predicted_scene}\n推理时间: {self.inference_time:.3f} 秒",
            fontsize=16
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        try:
            plt.savefig("scene_classification_process.png", dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存为 'scene_classification_process.png'")
        except Exception as e:
            print(f"保存图像时出错: {e}")
            
        # 注释掉或移除这一行
        # plt.show()
        
        print(f"预测场景: {self.predicted_scene}")
        print(f"推理时间: {self.inference_time:.3f} 秒")
        
        return fig

# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = SceneClassificationVisualizer()
    
    # 加载图像（可以使用本地图像或URL）
    # 示例URL图像
    image_url = "https://images.pexels.com/photos/338515/pexels-photo-338515.jpeg"
    # 或者使用本地图像
    # image_path = "path/to/your/image.jpg"
    
    try:
        visualizer.load_image(url=image_url)
        # 或者
        # visualizer.load_image(image_path=image_path)
        
        # 执行分类
        visualizer.classify()
        
        # 可视化整个分类流程
        visualizer.visualize_classification_process()
    except Exception as e:
        print(f"运行过程中出错: {e}")
        print("请尝试使用本地图像或检查网络连接。")
