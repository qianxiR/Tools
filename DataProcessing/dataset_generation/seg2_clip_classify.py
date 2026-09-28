#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单时相分割区域分类工具（SEG 模式 Step seg2）

功能：使用 RemoteCLIP 对分割区域进行零样本分类（单时相）
输入：seg1 输出的 regions/{sample}/{region}/image/ 目录
输出：regions.json 中的 classification.clip 字段（单类别）

与 CD 模式的区别：
- 只对单张 image 分类（非 T1/T2 双分类）
- classification.clip 输出 class + confidence（非 t1_class/t2_class）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import argparse
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from prompt_classify import get_voted_prompts


def imread_unicode(path):
    """
    入参:
        path (str): 图像文件路径
    方法:
        使用 np.fromfile + cv2.imdecode 读取，兼容中文路径
    出参:
        np.ndarray | None: 解码成功的图像数组，失败返回 None
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def load_model(model_name: str = "ViT-B-32", checkpoint_path: str = None, device: str = "cuda"):
    """
    入参:
        model_name (str): 模型架构名称
        checkpoint_path (str): 模型权重路径
        device (str): 计算设备
    方法:
        通过 open_clip 加载 RemoteCLIP 预训练权重
    出参:
        tuple: (model, preprocess, tokenizer)
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("pip install open_clip_torch")

    if checkpoint_path is None:
        checkpoint_filename = f"RemoteCLIP-{model_name}.pt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, f"../checkpoints/{checkpoint_filename}"),
            os.path.join(script_dir, f"checkpoints/{checkpoint_filename}"),
            f"checkpoints/{checkpoint_filename}",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        if checkpoint_path is None:
            raise FileNotFoundError(f"找不到 {checkpoint_filename}，请用 --checkpoint 指定")

    print(f"Loading RemoteCLIP: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=os.path.abspath(checkpoint_path), device=device,
    )
    model = model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def classify_image_voted(model, preprocess, tokenizer, image_path: str,
                         all_prompts: List[str], prompt_to_class: List[str],
                         device: str = "cuda") -> Dict:
    """
    入参:
        model, preprocess, tokenizer: RemoteCLIP 组件
        image_path (str): 待分类图像路径
        all_prompts (List[str]): 展平的同义词提示词
        prompt_to_class (List[str]): 每个提示词对应的大类标签
        device (str): 计算设备
    方法:
        对所有提示词计算图像-文本相似度，按大类分组取 max 后 argmax 选类别
    出参:
        Dict: {predicted_class, confidence, all_scores}
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

    class_max_scores = {}
    for i, cls in enumerate(prompt_to_class):
        score = float(similarities[i])
        if cls not in class_max_scores or score > class_max_scores[cls]:
            class_max_scores[cls] = score

    predicted_class = max(class_max_scores, key=class_max_scores.get)
    confidence = class_max_scores[predicted_class]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_scores": class_max_scores,
    }


def group_regions_by_original_image(regions_dir: str) -> Dict[str, List[str]]:
    """
    入参:
        regions_dir (str): regions 目录路径
    方法:
        扫描子目录，按原始图像名（rsplit '_' 前缀）分组
    出参:
        Dict[str, List[str]]: {原图名: [区域目录名列表]}
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
                    all_prompts: List[str], prompt_to_class: List[str], device: str = "cuda"):
    """
    入参:
        dataset_path (str): 数据集根目录
        split (str): 数据集划分
        model, preprocess, tokenizer: RemoteCLIP 组件
        all_prompts, prompt_to_class: 投票分类提示词配置
        device (str): 计算设备
    方法:
        遍历 regions.json 中 region_id > 0 的区域，读取 image/ 子图做 CLIP 分类，
        结果写入 classification.clip（单类别字段）
    出参:
        None（结果写回 regions.json）
    """
    regions_dir = os.path.join(dataset_path, split, "regions")

    if not os.path.exists(regions_dir):
        print(f"Error: Regions directory not found: {regions_dir}")
        print("Please run seg1_extract_regions.py first.")
        return

    groups = group_regions_by_original_image(regions_dir)

    print(f"\n{'='*60}")
    print(f"SEG CLIP Classify - {split}")
    print(f"{'='*60}")
    print(f"Samples: {len(groups)}, Regions: {sum(len(v) for v in groups.values())}")
    print(f"{'='*60}\n")

    stats = {"total_original": len(groups), "local_processed": 0, "failed": 0}

    for original_name, region_names in tqdm(groups.items(), desc=f"Processing {split}"):
        regions_json_path = os.path.join(regions_dir, original_name, "regions.json")

        if not os.path.exists(regions_json_path):
            stats["failed"] += 1
            continue

        with open(regions_json_path, 'r', encoding='utf-8') as f:
            regions_json = json.load(f)

        for region in regions_json.get("regions", []):
            region_id = int(region.get("region_id", 0) or 0)
            if region_id <= 0:
                region.pop("classification", None)
                continue

            region_name = region.get("region_name", "")
            image_path = os.path.join(regions_dir, region_name, "image", f"{region_name}.png")

            if not os.path.exists(image_path):
                stats["failed"] += 1
                continue

            try:
                result = classify_image_voted(model, preprocess, tokenizer, image_path,
                                             all_prompts, prompt_to_class, device)

                region["classification"] = {
                    "clip": {
                        "class": result["predicted_class"],
                        "confidence": result["confidence"],
                    }
                }
                stats["local_processed"] += 1

            except Exception as e:
                stats["failed"] += 1
                print(f"Error classifying {region_name}: {e}")

        with open(regions_json_path, 'w', encoding='utf-8') as f:
            json.dump(regions_json, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"SEG CLIP Classify Complete - {split}")
    print(f"{'='*60}")
    print(f"Regions: {stats['local_processed']} classified, {stats['failed']} failed")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SEG mode: CLIP classify segmentation regions")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test", "all"],
                       help="Dataset split (default: train)")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                       choices=["RN50", "ViT-B-32", "ViT-L-14"],
                       help="RemoteCLIP model name (default: ViT-B-32)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Device (default: cuda)")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return 1

    model, preprocess, tokenizer = load_model(args.model, args.checkpoint, args.device)

    all_prompts, prompt_to_class = get_voted_prompts()
    class_labels = list(dict.fromkeys(prompt_to_class))
    print(f"Classes ({len(class_labels)}): {class_labels}")

    if args.split == "all":
        for split in ["train", "val", "test"]:
            process_dataset(args.dataset_path, split, model, preprocess, tokenizer,
                          all_prompts, prompt_to_class, args.device)
    else:
        process_dataset(args.dataset_path, args.split, model, preprocess, tokenizer,
                       all_prompts, prompt_to_class, args.device)

    print("\n" + "=" * 60)
    print("SEG CLIP classification completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
