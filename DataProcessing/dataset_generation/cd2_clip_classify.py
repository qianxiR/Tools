#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变化区域分类工具

功能：使用 RemoteCLIP 对 T1/T2 变化区域进行零样本分类
输出：regions.json 中的 classification 字段
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import math
import argparse
import importlib.util
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# 导入类别提示词
from prompt_classify import get_voted_prompts


def imread_unicode(path):
    """支持中文路径的图片读取"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


# 6大类别颜色映射 (BGR 格式，用于 cv2.rectangle)
CLASS_COLORS = {
    "building":    (0, 0, 255),       # 红色
    "highway":     (0, 165, 255),     # 橙色
    "vegetation":  (0, 255, 0),       # 绿色
    "farmland":    (0, 255, 255),     # 黄色
    "bareland":    (0, 128, 255),     # 蓝橙
    "water":       (255, 0, 0),       # 蓝色
}
DEFAULT_COLOR = (200, 200, 200)  # 灰色（未知类别兜底）


def load_model(model_name: str = "ViT-B-32", checkpoint_path: str = None, device: str = "cuda"):
    """
    加载 RemoteCLIP 模型

    入参:
    - model_name (str): 模型名称
    - checkpoint_path (str): 模型权重路径
    - device (str): 设备

    出参:
    - model, preprocess, tokenizer
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("请先安装 open_clip_torch: pip install open_clip_torch")

    if checkpoint_path is None:
        # 根据 model_name 搜索对应的 checkpoint 文件
        checkpoint_filename = f"RemoteCLIP-{model_name}.pt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, f"../checkpoints/{checkpoint_filename}"),
            os.path.join(script_dir, f"checkpoints/{checkpoint_filename}"),
            f"checkpoints/{checkpoint_filename}",
            f"../checkpoints/{checkpoint_filename}",
            f"../../checkpoints/{checkpoint_filename}",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break

        if checkpoint_path is None:
            raise FileNotFoundError(f"找不到模型权重文件 {checkpoint_filename}，请使用 --checkpoint 参数指定路径")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型权重文件不存在: {checkpoint_path}")

    checkpoint_path = os.path.abspath(checkpoint_path)

    # 这里直接把 checkpoint 交给 open_clip 加载，而不是先随机初始化再手动 load_state_dict。
    # 这样做可以复用 open_clip 的标准权重装载流程，并消除“随机初始化”误导日志。
    print(f"Loading RemoteCLIP: {model_name}")
    print(f"Using checkpoint: {checkpoint_path}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=checkpoint_path,
        device=device,
    )
    model = model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer


def classify_image(model, preprocess, tokenizer, image_path: str,
                   text_prompts: List[str], class_labels: List[str], device: str = "cuda") -> Dict:
    """
    对图像进行语义分类

    入参:
    - model: RemoteCLIP 模型
    - preprocess: 图像预处理函数
    - tokenizer: 文本 tokenizer
    - image_path (str): 图像路径
    - text_prompts (List[str]): 文本提示词
    - class_labels (List[str]): 类别标签
    - device (str): 设备

    出参:
    - Dict: 分类结果
    """
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
    text_tokens = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    predicted_idx = int(similarities.argmax())
    predicted_class = class_labels[predicted_idx]
    confidence = float(similarities[predicted_idx])

    all_scores = {class_labels[i]: float(similarities[i]) for i in range(len(class_labels))}

    return {"predicted_class": predicted_class, "confidence": confidence, "all_scores": all_scores}


def classify_image_voted(model, preprocess, tokenizer, image_path: str,
                         all_prompts: List[str], prompt_to_class: List[str],
                         device: str = "cuda") -> Dict:
    """
    多提示词投票分类：每个大类用多个同义词提示词，同类取 max 相似度后 argmax

    入参:
    - model: RemoteCLIP 模型
    - preprocess: 图像预处理函数
    - tokenizer: 文本 tokenizer
    - image_path (str): 图像路径
    - all_prompts (List[str]): 展平的所有同义词提示词
    - prompt_to_class (List[str]): 每个提示词对应的大类标签
    - device (str): 设备

    方法:
    - 对所有提示词计算相似度 → 按大类分组取 max → argmax 选大类

    出参:
    - Dict: {predicted_class, confidence, all_scores (按大类)}
    """
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
    text_tokens = tokenizer(all_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    # 按大类聚合：取每个大类的最大相似度
    class_max_scores = {}
    for i, cls in enumerate(prompt_to_class):
        score = float(similarities[i])
        if cls not in class_max_scores or score > class_max_scores[cls]:
            class_max_scores[cls] = score

    # 选择 Top-1（最高分）
    predicted_class = max(class_max_scores, key=class_max_scores.get)
    confidence = class_max_scores[predicted_class]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_scores": class_max_scores
    }


def find_original_image(split_dir: str, subdir: str, name: str) -> Optional[str]:
    """
    在数据集中查找原始 T1/T2 图像路径

    入参:
    - split_dir (str): split 目录路径
    - subdir (str): 子目录名 (t1 或 t2)
    - name (str): 图像名（不含扩展名）

    出参:
    - Optional[str]: 图像路径，未找到则返回 None
    """
    for ext in ["png", "jpg", "jpeg", "tif"]:
        path = os.path.join(split_dir, subdir, f"{name}.{ext}")
        if os.path.exists(path):
            return path
    return None


def _do_classify(model, preprocess, tokenizer, image_path: str,
                 text_prompts: List[str], class_labels: List[str], device: str,
                 use_voted: bool, all_prompts: List[str], prompt_to_class: List[str]) -> Dict:
    """
    统一分类入口，根据 use_voted 选择普通分类或投票分类

    入参:
    - model, preprocess, tokenizer, image_path, device: 模型组件
    - text_prompts, class_labels: 普通分类参数
    - use_voted: 是否使用投票分类
    - all_prompts, prompt_to_class: 投票分类参数

    出参:
    - Dict: 分类结果
    """
    if use_voted:
        return classify_image_voted(model, preprocess, tokenizer, image_path,
                                    all_prompts, prompt_to_class, device)
    else:
        return classify_image(model, preprocess, tokenizer, image_path,
                             text_prompts, class_labels, device)


def compute_spatial_location(bbox: List[int], image_width: int, image_height: int) -> str:
    """
    基于 bbox 中心点相对于图像中心的方向，计算变化区域的空间方位

    入参:
    - bbox (List[int]): [x, y, w, h] 区域边界框（绝对坐标）
    - image_width (int): 图像宽度（像素）
    - image_height (int): 图像高度（像素）

    方法:
    - 计算 bbox 中心点相对于图像中心的偏移 (dx, dy)
    - y 轴翻转：图像坐标 y 向下，地理方向 y 向上
    - 使用 atan2 将偏移角度映射到 8 方向扇区（每 45° 一个）
    - 偏移距离 < 10% 最大维度时判定为 center

    出参:
    - return (str): 方位字符串 (east/northeast/north/northwest/west/southwest/south/southeast/center)
    """
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    img_cx = image_width / 2
    img_cy = image_height / 2

    dx = cx - img_cx
    dy = -(cy - img_cy)  # y 轴翻转：图像坐标 y 向下，地理方向 y 向上

    threshold = max(image_width, image_height) * 0.1
    if abs(dx) < threshold and abs(dy) < threshold:
        return "center"

    angle = math.atan2(dy, dx)
    sector = round(angle / (math.pi / 4)) % 8
    directions = ["east", "northeast", "north", "northwest",
                  "west", "southwest", "south", "southeast"]
    return directions[sector]


def load_region_metadata(regions_dir: str, original_name: str) -> Optional[Dict]:
    """
    从 Step 1 输出的 regions.json 中读取区域元数据，保留所有字段

    入参:
    - regions_dir (str): regions 目录路径
    - original_name (str): 原始图像名称

    出参:
    - Optional[Dict]: 元数据字典，未找到则返回 None
    """
    json_path = os.path.join(regions_dir, original_name, "regions.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 直接返回完整的 regions 数据，保留所有字段
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load metadata {json_path}: {e}")
        return None


def process_single_region(region_dir: str, region_name: str, model, preprocess, tokenizer,
                          text_prompts: List[str], class_labels: List[str], device: str = "cuda",
                          crop_bbox: List[int] = None,
                          region_id: int = None,
                          use_voted: bool = False, all_prompts: List[str] = None,
                          prompt_to_class: List[str] = None) -> Optional[Dict]:
    """
    处理单个变化区域

    入参:
    - region_dir (str): 区域目录路径
    - region_name (str): 区域名称
    - model, preprocess, tokenizer: RemoteCLIP 组件
    - text_prompts (List[str]): 文本提示词
    - class_labels (List[str]): 类别标签
    - device (str): 设备
    - crop_bbox (List[int]): 裁剪区域在原图中的绝对坐标 [x, y, w, h]
    - use_voted (bool): 是否使用投票分类
    - all_prompts, prompt_to_class: 投票分类参数

    出参:
    - Optional[Dict]: 区域分类结果
    """
    t1_path = os.path.join(region_dir, "t1", f"{region_name}.png")
    t2_path = os.path.join(region_dir, "t2", f"{region_name}.png")
    mask_path = os.path.join(region_dir, "label", f"{region_name}.png")

    if not all(os.path.exists(p) for p in [t1_path, t2_path, mask_path]):
        return None

    try:
        result_t1 = _do_classify(model, preprocess, tokenizer, t1_path,
                                 text_prompts, class_labels, device, use_voted, all_prompts, prompt_to_class)
        result_t2 = _do_classify(model, preprocess, tokenizer, t2_path,
                                 text_prompts, class_labels, device, use_voted, all_prompts, prompt_to_class)

        from change_text_templates import generate_dataset_text
        if not crop_bbox:
            return None
        mask = imread_unicode(mask_path)
        h, w = mask.shape[:2] if mask is not None else (0, 0)

        record = generate_dataset_text(
            image_name=f"{region_name}.png",
            t1_class=result_t1["predicted_class"],
            t2_class=result_t2["predicted_class"],
            confidence_t1=result_t1["confidence"],
            confidence_t2=result_t2["confidence"],
            change_bbox=crop_bbox,
            patch_size=[h, w]
        )
        record["region_name"] = region_name
        record["is_global"] = False
        if region_id is not None:
            record["region_id"] = region_id
        return record

    except Exception as e:
        print(f"Error processing {region_name}: {e}")
        return None


def group_regions_by_original_image(regions_dir: str) -> Dict[str, List[str]]:
    """
    按原始图像分组区域

    入参:
    - regions_dir (str): regions 目录路径

    出参:
    - Dict[str, List[str]]: {原图名: [区域列表]}
    """
    region_dirs = sorted([d for d in os.listdir(regions_dir) if os.path.isdir(os.path.join(regions_dir, d))])
    groups = {}

    for region_name in region_dirs:
        parts = region_name.rsplit('_', 1)
        if len(parts) == 2:
            original_name = parts[0]
            if original_name not in groups:
                groups[original_name] = []
            groups[original_name].append(region_name)

    return groups


def process_dataset(dataset_path: str, split: str, model, preprocess, tokenizer,
                    text_prompts: List[str], class_labels: List[str], device: str = "cuda",
                    all_prompts: List[str] = None, prompt_to_class: List[str] = None,
                    use_voted: bool = False):
    """
    处理整个数据集：仅局部变化区域分类（region_id > 0）

    入参:
    - dataset_path (str): 数据集根目录
    - split (str): 数据集分割 (train/val/test)
    - model: RemoteCLIP 模型
    - preprocess: 图像预处理函数
    - tokenizer: 文本 tokenizer
    - text_prompts (List[str]): 文本提示词
    - class_labels (List[str]): 类别标签
    - device (str): 设备
    """
    regions_dir = os.path.join(dataset_path, split, "regions")
    split_dir = os.path.join(dataset_path, split)

    if not os.path.exists(regions_dir):
        print(f"Error: Regions directory not found: {regions_dir}")
        print("Please run extract_regions.py first to extract change regions.")
        return

    groups = group_regions_by_original_image(regions_dir)

    print(f"\n{'='*60}")
    print(f"Classify Regions - {split}")
    print(f"{'='*60}")
    print(f"Samples: {len(groups)}, Regions: {sum(len(v) for v in groups.values())}")
    print(f"{'='*60}\n")

    stats = {"total_original": len(groups), "total_regions": sum(len(v) for v in groups.values()),
             "samples_processed": 0, "local_processed": 0, "failed": 0}

    for original_name, region_names in tqdm(groups.items(), desc=f"Processing {split}"):
        sample_dir = os.path.join(regions_dir, original_name)
        regions_json_path = os.path.join(sample_dir, "regions.json")

        if not os.path.exists(regions_json_path):
            stats["failed"] += 1
            continue

        try:
            with open(regions_json_path, 'r', encoding='utf-8') as f:
                regions_json = json.load(f)
        except Exception as e:
            print(f"Error loading {regions_json_path}: {e}")
            stats["failed"] += 1
            continue

        for region in regions_json.get("regions", []):
            region_id = int(region.get("region_id", 0) or 0)
            if region_id <= 0:
                # 移除背景区域可能遗留的分类字段，避免后续流程误用
                region.pop("classification", None)
                continue

            region_name = region.get("region_name", "")
            region_dir = os.path.join(regions_dir, region_name)

            t1_path = os.path.join(region_dir, "t1", f"{region_name}.png")
            t2_path = os.path.join(region_dir, "t2", f"{region_name}.png")

            if not os.path.exists(t1_path) or not os.path.exists(t2_path):
                continue

            try:
                result_t1 = _do_classify(model, preprocess, tokenizer, t1_path,
                                        text_prompts, class_labels, device, use_voted, all_prompts, prompt_to_class)
                result_t2 = _do_classify(model, preprocess, tokenizer, t2_path,
                                        text_prompts, class_labels, device, use_voted, all_prompts, prompt_to_class)

                region["classification"] = {
                    "clip": {
                        "t1_class": result_t1.get("predicted_class", ""),
                        "t2_class": result_t2.get("predicted_class", ""),
                        "confidence_t1": result_t1.get("confidence", 0),
                        "confidence_t2": result_t2.get("confidence", 0),
                    }
                }
                stats["local_processed"] += 1

            except Exception as e:
                stats["failed"] += 1
                print(f"Error classifying {region_name}: {e}")

        with open(regions_json_path, 'w', encoding='utf-8') as f:
            json.dump(regions_json, f, ensure_ascii=False, indent=2)
        stats["samples_processed"] += 1

    print(f"\n{'='*60}")
    print(f"Classify Complete - {split}")
    print(f"{'='*60}")
    print(f"Samples:  {stats['total_original']}")
    print(f"Regions:  {stats['local_processed']} classified, {stats['failed']} failed")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Classify change regions and generate text descriptions")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"],
                       help="Dataset split to process (default: train)")
    parser.add_argument("--model", type=str, default="ViT-B-32", choices=["RN50", "ViT-B-32", "ViT-L-14"],
                       help="Model name (default: ViT-B-32)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (default: cuda)")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return 1

    model, preprocess, tokenizer = load_model(args.model, args.checkpoint, args.device)

    # 获取类别提示词（6类基础版本）
    all_prompts, prompt_to_class = get_voted_prompts()
    class_labels = list(dict.fromkeys(prompt_to_class))
    print(f"Classes ({len(class_labels)}): {class_labels}")
    for cls in class_labels:
        prompts = [p for p, c in zip(all_prompts, prompt_to_class) if c == cls]
        print(f"  {cls}: {prompts}")

    if args.split == "all":
        for split in ["train", "val", "test"]:
            process_dataset(args.dataset_path, split, model, preprocess, tokenizer,
                          None, class_labels, args.device,
                          all_prompts=all_prompts, prompt_to_class=prompt_to_class, use_voted=True)
    else:
        process_dataset(args.dataset_path, args.split, model, preprocess, tokenizer,
                       None, class_labels, args.device,
                       all_prompts=all_prompts, prompt_to_class=prompt_to_class, use_voted=True)

    print("\n" + "=" * 60)
    print("All classification completed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
