#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEG Step 1: 实例提取与类别关联

从语义分割掩码中提取每个独立区域（实例），推断类别，裁剪原始图像子图。
输出 instances.json。

用法:
    python tools/seg1_extract_instances.py \
        --data_root "E:/xzkjxm/dataes/WHU_20" \
        --split test

    # 或处理单张图像
    python tools/seg1_extract_instances.py \
        --image image.png \
        --seg_mask mask.png \
        --output_dir output/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from cd_utils import (
    imread_unicode, imwrite_unicode,
    parse_classes, build_palette, rgb_to_label,
    save_instances, append_process_log,
    DEFAULT_CLASSES,
)


def extract_instances_from_label(label_map: np.ndarray, cls_name_map: Dict[int, List[str]],
                                 min_area: int = 4) -> List[Dict]:
    """
    从多类 label map 中提取所有实例（每个类别的每个连通域为一个实例）
    """
    instances = []
    instance_id = 0

    for cls_id in range(1, max(cls_name_map.keys()) + 1):
        if cls_id not in cls_name_map:
            continue

        binary = (label_map == cls_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

        for comp_id in range(1, num_labels):
            comp_mask = (labels == comp_id).astype(np.uint8)
            area = int(comp_mask.sum())

            if area < min_area:
                continue

            coords = np.where(comp_mask > 0)
            y_min, y_max = int(coords[0].min()), int(coords[0].max())
            x_min, x_max = int(coords[1].min()), int(coords[1].max())

            instance_id += 1
            instances.append({
                "instance_id": instance_id,
                "category": cls_name_map[cls_id][0],
                "cls_id": cls_id,
                "bbox": [x_min, y_min, x_max + 1, y_max + 1],
                "area": area,
                "keep": True,
                "clip_score": None,
                "qwen_result": None,
            })

    return instances


def crop_instance_image(img: np.ndarray, instance: Dict,
                        instance_dir: str, instance_name: str) -> bool:
    """裁剪并保存实例对应的原始图像子图"""
    x1, y1, x2, y2 = instance["bbox"]
    crop = img[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return False

    sub_img = os.path.join(instance_dir, "image")
    os.makedirs(sub_img, exist_ok=True)
    imwrite_unicode(os.path.join(sub_img, f"{instance_name}.png"), crop)
    return True


def process_single_image(image_path: str, seg_mask_path: str,
                         output_dir: str, sample_name: str,
                         cls_name_map: Dict, palette: np.ndarray,
                         min_area: int = 4) -> Optional[Dict]:
    """处理单张分割掩码，提取实例并保存"""
    img = imread_unicode(image_path)
    mask_img = imread_unicode(seg_mask_path)

    if img is None or mask_img is None:
        return None

    # RGB mask → label map
    if len(mask_img.shape) == 2:
        unique = np.unique(mask_img)
        if len(unique) <= 2 and set(unique.tolist()).issubset({0, 255}):
            label_map = (mask_img > 127).astype(np.uint8)
        else:
            label_map = mask_img.copy()
    else:
        label_map = rgb_to_label(mask_img, palette)

    # 确保图像尺寸一致
    h, w = label_map.shape[:2]
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))

    # 提取实例
    instances = extract_instances_from_label(label_map, cls_name_map, min_area)

    if not instances:
        return {
            "image_name": sample_name,
            "num_instances": 0,
            "instances": [],
        }

    # 裁剪并保存每个实例
    instances_dir = os.path.join(output_dir, "instances")
    for inst in instances:
        inst_name = f"{sample_name}_{inst['instance_id']:03d}"
        inst_dir = os.path.join(instances_dir, inst_name)

        success = crop_instance_image(img, inst, inst_dir, inst_name)
        if success:
            inst["region_name"] = inst_name
            inst["files"] = {
                "image": f"instances/{inst_name}/image/{inst_name}.png",
            }
        else:
            inst["region_name"] = inst_name
            inst["files"] = None

    for inst in instances:
        inst.pop("cls_id", None)

    return {
        "image_name": sample_name,
        "num_instances": len(instances),
        "instances": instances,
    }


def process_dataset(data_root: str, split: str, cls_name_map: Dict,
                    palette: np.ndarray, min_area: int = 4):
    """
    处理整个数据集的 seg_mask，批量提取实例

    目录结构:
    {data_root}/{split}/
    ├── image/       ← 原始图像
    └── seg_mask/    ← 推理输出的分割掩码 (RGB PNG)
    """
    split_dir = os.path.join(data_root, split)
    image_dir = os.path.join(split_dir, "image")
    mask_dir = os.path.join(split_dir, "seg_mask")
    output_dir = split_dir

    if not os.path.isdir(mask_dir):
        print(f"Error: seg_mask 目录不存在: {mask_dir}")
        print("请先运行 infer_batch_seg.py 生成分割掩码")
        return

    # 搜索所有掩码文件
    mask_files = sorted(
        list(Path(mask_dir).glob("*.png")) +
        list(Path(mask_dir).glob("*.jpg")) +
        list(Path(mask_dir).glob("*.tif"))
    )

    if not mask_files:
        print(f"Warning: {mask_dir} 中没有找到掩码文件")
        return

    print(f"\n{'='*60}")
    print(f"SEG Step 1: 实例提取 - {split}")
    print(f"{'='*60}")
    print(f"样本数: {len(mask_files)}")
    print(f"输出: {output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "process_log.json")

    total_instances = 0
    processed = 0

    for mask_path in tqdm(mask_files, desc=f"Extracting {split}"):
        sample_name = mask_path.stem

        # 查找原始图像（兼容不同扩展名）
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            candidate = os.path.join(image_dir, sample_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            continue

        result = process_single_image(
            image_path, str(mask_path),
            output_dir, sample_name, cls_name_map, palette, min_area
        )

        if result is not None:
            instances_json_path = os.path.join(output_dir, f"{sample_name}_instances.json")
            save_instances(result, instances_json_path)
            total_instances += result["num_instances"]
            processed += 1

    append_process_log(log_path, "step1_extract", f"Extracted {total_instances} instances from {processed} samples", {
        "total_samples": len(mask_files),
        "processed": processed,
        "total_instances": total_instances,
    })

    print(f"\n{'='*60}")
    print(f"SEG Step 1 完成")
    print(f"{'='*60}")
    print(f"样本: {processed}/{len(mask_files)}")
    print(f"实例: {total_instances}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SEG Step 1: 从分割掩码提取实例")
    parser.add_argument("--data_root", type=str, help="数据集根目录 (batch 模式)")
    parser.add_argument("--split", type=str, default="test", help="数据划分")
    parser.add_argument("--classes", type=str, default=None, help="类别文件路径")
    parser.add_argument("--min_area", type=int, default=4, help="最小实例面积")

    # 单图模式
    parser.add_argument("--image", type=str, help="原始图像路径 (单图模式)")
    parser.add_argument("--seg_mask", type=str, help="分割掩码路径 (单图模式)")
    parser.add_argument("--output_dir", type=str, help="输出目录 (单图模式)")

    args = parser.parse_args()

    cls_name_map, num_cls = parse_classes(args.classes or DEFAULT_CLASSES)
    palette = build_palette(num_cls, cls_name_map)
    palette[0] = [255, 255, 255]  # background → 白色

    if args.image and args.seg_mask:
        output_dir = args.output_dir or "output_seg_postprocess"
        os.makedirs(output_dir, exist_ok=True)
        sample_name = Path(args.image).stem
        result = process_single_image(
            args.image, args.seg_mask,
            output_dir, sample_name, cls_name_map, palette, args.min_area
        )
        if result:
            save_instances(result, os.path.join(output_dir, f"{sample_name}_instances.json"))
            print(f"提取 {result['num_instances']} 个实例")
    elif args.data_root:
        if args.split == "all":
            for sp in ["train", "val", "test"]:
                process_dataset(args.data_root, sp, cls_name_map, palette, args.min_area)
        else:
            process_dataset(args.data_root, args.split, cls_name_map, palette, args.min_area)
    else:
        parser.error("请指定 --data_root (批量模式) 或 --image/--seg_mask (单图模式)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
