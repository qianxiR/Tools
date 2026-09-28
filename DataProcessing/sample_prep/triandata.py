"""
RGB图像和水体掩膜配对数据集生成工具

入参:
- image_path (str): RGB图像路径（三通道）
- mask_path (str): 水体掩膜路径（单通道）
- output_dir (str): 数据集输出根目录
- window_size (int): 滑动窗口大小
- stride (int): 滑动步长
- train_ratio (float): 训练集比例

方法:
① 读取RGB图像和掩膜图像
② 使用滑动窗口同步提取图像-掩膜配对块
③ 按照指定比例划分训练集和测试集
④ 对训练集进行数据增强（旋转）
⑤ 按照目录结构保存：images/和masks/子目录

出参:
- 训练集和测试集的RGB图像-掩膜配对数据
"""

import os
import numpy as np
from PIL import Image
import rasterio
from sklearn.model_selection import train_test_split

# 全局配置参数
IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "栅格与表互相转换", "处理结果", "step4_RGB.png")
MASK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "栅格与表互相转换", "处理结果", "step3_water_mask.png")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_train")
WINDOW_SIZE = 128
STRIDE = 4
TRAIN_RATIO = 0.7


def sliding_window_paired(image, mask, window_size=128, stride=4):
    """
    滑动窗口提取图像-掩膜配对块
    
    入参:
    - image (ndarray): RGB图像矩阵 (H x W x 3)
    - mask (ndarray): 掩膜图像矩阵 (H x W)
    - window_size (int): 窗口大小
    - stride (int): 滑动步长
    
    方法:
    - 在两个图像上同步滑动窗口
    - 保证每个图像块与掩膜块一一对应
    - 支持RGB三通道图像
    
    出参:
    - image_patches (list): RGB图像块列表
    - mask_patches (list): 掩膜块列表
    """
    image_patches = []
    mask_patches = []
    h, w = image.shape[:2]
    
    # 确保图像和掩膜的高宽一致
    assert image.shape[:2] == mask.shape[:2], "图像和掩膜的高宽必须一致"
    
    # 滑动窗口提取
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            image_patch = image[y:y+window_size, x:x+window_size]
            mask_patch = mask[y:y+window_size, x:x+window_size]
            image_patches.append(image_patch)
            mask_patches.append(mask_patch)
    
    return image_patches, mask_patches


def augment_paired_patches(image_patch, mask_patch, add_noise=False):
    """
    对配对的RGB图像-掩膜块进行数据增强
    
    入参:
    - image_patch (ndarray): RGB图像块 (H x W x 3)
    - mask_patch (ndarray): 掩膜块 (H x W)
    - add_noise (bool): 是否添加噪声
    
    方法:
    - 对RGB图像和掩膜同步进行旋转（0/90/180/270度）
    - 可选：对RGB图像添加噪声（掩膜不加噪声）
    
    出参:
    - aug_image_patches (list): 增强后的RGB图像块列表
    - aug_mask_patches (list): 增强后的掩膜块列表
    """
    aug_image_patches = []
    aug_mask_patches = []
    
    # 旋转增强：0/90/180/270度
    for k in range(4):
        rotated_image = np.rot90(image_patch, k)
        rotated_mask = np.rot90(mask_patch, k)
        aug_image_patches.append(rotated_image)
        aug_mask_patches.append(rotated_mask)
        
        # 可选：对RGB图像添加噪声（掩膜保持不变）
        if add_noise:
            noise = np.random.normal(0, 0.05, rotated_image.shape)
            noisy_image = np.clip(rotated_image / 255.0 + noise, 0, 1) * 255
            aug_image_patches.append(noisy_image.astype(np.uint8))
            aug_mask_patches.append(rotated_mask)
    
    return aug_image_patches, aug_mask_patches


def save_paired_patches(image_patches, mask_patches, save_dir, prefix):
    """
    保存配对的RGB图像-掩膜块
    
    入参:
    - image_patches (list): RGB图像块列表 (每个元素为 H x W x 3)
    - mask_patches (list): 掩膜块列表 (每个元素为 H x W)
    - save_dir (str): 保存根目录
    - prefix (str): 文件名前缀（train或test）
    
    方法:
    - 创建images和masks子目录
    - 保存配对数据，文件名一致
    - 图像保存为RGB格式，掩膜保存为灰度格式
    
    出参: 无（直接保存文件）
    """
    image_dir = os.path.join(save_dir, "images")
    mask_dir = os.path.join(save_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    for idx, (img_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
        # 保存RGB图像块
        img = Image.fromarray(img_patch.astype(np.uint8), mode='RGB')
        img.save(os.path.join(image_dir, f"{prefix}_patch_{idx:06d}.png"))
        
        # 保存掩膜块（灰度）
        mask = Image.fromarray(mask_patch.astype(np.uint8), mode='L')
        mask.save(os.path.join(mask_dir, f"{prefix}_patch_{idx:06d}.png"))


def generate_training_dataset(image_path, mask_path, output_dir, 
                             window_size=None, stride=None, train_ratio=None):
    """
    生成训练集和测试集（支持外部调用）
    
    入参:
    - image_path (str): RGB图像路径
    - mask_path (str): 掩膜图像路径
    - output_dir (str): 输出目录
    - window_size (int): 窗口大小，None时使用全局默认值
    - stride (int): 步长，None时使用全局默认值
    - train_ratio (float): 训练集比例，None时使用全局默认值
    
    方法:
    ① 读取RGB图像和掩膜图像
    ② 滑动窗口提取配对块
    ③ 划分训练集和测试集
    ④ 训练集数据增强（旋转）
    ⑤ 保存到指定目录结构
    
    出参: 无（直接保存文件）
    """
    # 使用全局默认值（如果未提供参数）
    if window_size is None:
        window_size = WINDOW_SIZE
    if stride is None:
        stride = STRIDE
    if train_ratio is None:
        train_ratio = TRAIN_RATIO
    
    print("=" * 70)
    print("RGB图像与水体掩膜配对数据集生成工具")
    print("=" * 70)
    
    # ① 读取图像和掩膜
    print(f"\n【步骤1】读取图像")
    print(f"  RGB图像: {image_path}")
    print(f"  掩膜图像: {mask_path}")
    
    # 判断文件格式并加载图像和掩膜
    # 支持GeoTIFF和普通图像格式
    def load_image(path):
        """根据文件扩展名选择加载方式"""
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.tif', '.tiff']:
            # 使用rasterio读取GeoTIFF
            with rasterio.open(path) as src:
                data = src.read()  # 形状为 (bands, height, width)
                # 处理多波段图像（选择RGB波段或所有波段）
                if data.shape[0] == 1:
                    # 单波段
                    return data[0]
                elif data.shape[0] >= 3:
                    # 多波段，取前3个波段作为RGB
                    return np.transpose(data[:3], (1, 2, 0))  # 转为 (H, W, 3)
                else:
                    return data[0]
        else:
            # 使用PIL读取普通图像
            return np.array(Image.open(path).convert('RGB'))
    
    image = load_image(image_path)
    mask = load_image(mask_path)
    
    # 确保图像是RGB格式 (H, W, 3)
    if len(image.shape) == 2:
        # 灰度图像，转换为RGB
        image = np.stack([image] * 3, axis=-1)
    
    # 确保掩膜是单通道 (H, W)
    if len(mask.shape) == 3:
        # 多通道掩膜，取第一个通道
        mask = mask[:, :, 0]
    
    # 归一化到0-255范围
    if image.max() > 255:
        image = (image / image.max() * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    if mask.max() > 255:
        mask = (mask / mask.max() * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    print(f"  图像尺寸: {image.shape}")
    print(f"  掩膜尺寸: {mask.shape}")
    print(f"  图像值范围: [{image.min()}, {image.max()}]")
    print(f"  掩膜值范围: [{mask.min()}, {mask.max()}]")
    
    # ② 滑动窗口提取配对块
    print(f"\n【步骤2】滑动窗口提取")
    print(f"  窗口大小: {window_size}x{window_size}")
    print(f"  步长: {stride}")
    
    image_patches, mask_patches = sliding_window_paired(
        image, mask, 
        window_size=window_size, 
        stride=stride
    )
    
    print(f"  ✓ 提取总块数: {len(image_patches)}")
    
    # ③ 划分训练集和测试集
    print(f"\n【步骤3】划分训练集和测试集")
    print(f"  训练集比例: {train_ratio*100:.0f}%")
    print(f"  测试集比例: {(1-train_ratio)*100:.0f}%")
    
    train_img, test_img, train_mask, test_mask = train_test_split(
        image_patches, 
        mask_patches, 
        train_size=train_ratio, 
        random_state=42
    )
    
    print(f"  ✓ 训练集: {len(train_img)} 对")
    print(f"  ✓ 测试集: {len(test_img)} 对")
    
    # ④ 训练集数据增强（旋转4个方向）
    print(f"\n【步骤4】训练集数据增强")
    print(f"  增强方式: 旋转（0°/90°/180°/270°）")
    
    all_train_img_aug = []
    all_train_mask_aug = []
    
    for img_patch, mask_patch in zip(train_img, train_mask):
        aug_imgs, aug_masks = augment_paired_patches(
            img_patch, mask_patch, 
            add_noise=False
        )
        all_train_img_aug.extend(aug_imgs)
        all_train_mask_aug.extend(aug_masks)
    
    print(f"  ✓ 增强后训练集: {len(all_train_img_aug)} 对")
    print(f"  增强倍数: {len(all_train_img_aug) / len(train_img):.1f}x")
    
    # ⑤ 保存数据集
    print(f"\n【步骤5】保存数据集")
    print(f"  输出目录: {output_dir}")
    
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    # 保存训练集（增强后）
    print(f"  正在保存训练集...")
    save_paired_patches(all_train_img_aug, all_train_mask_aug, train_dir, "train")
    print(f"    ✓ 训练集图像: {train_dir}/images/")
    print(f"    ✓ 训练集掩膜: {train_dir}/masks/")
    
    # 保存测试集（未增强）
    print(f"  正在保存测试集...")
    save_paired_patches(test_img, test_mask, test_dir, "test")
    print(f"    ✓ 测试集图像: {test_dir}/images/")
    print(f"    ✓ 测试集掩膜: {test_dir}/masks/")
    
    # 最终统计
    print(f"\n【完成】数据集生成完成")
    print(f"=" * 70)
    print(f"训练集: {len(all_train_img_aug)} 对（已增强）")
    print(f"测试集: {len(test_img)} 对（未增强）")
    print(f"\n目录结构:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/  ({len(all_train_img_aug)} 张)")
    print(f"    │   └── masks/   ({len(all_train_mask_aug)} 张)")
    print(f"    └── test/")
    print(f"        ├── images/  ({len(test_img)} 张)")
    print(f"        └── masks/   ({len(test_mask)} 张)")
    print(f"=" * 70)


def main():
    """
    主函数 - 使用全局配置参数生成训练集和测试集
    
    入参: 无（使用全局配置）
    
    方法:
    - 调用generate_training_dataset函数
    - 使用全局默认参数
    
    出参: 无（直接保存文件）
    """
    generate_training_dataset(
        image_path=IMAGE_PATH,
        mask_path=MASK_PATH,
        output_dir=OUTPUT_DIR,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        train_ratio=TRAIN_RATIO
    )


if __name__ == "__main__":
    main()