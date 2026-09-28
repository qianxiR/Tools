#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: 实例提取与类别关联

从变化检测掩码中提取每个独立变化区域（实例），推断类别，裁剪 T1/T2 子图。
输出 instances.json。

用法:
    python tools/cd1_extract_instances.py \
        --data_root "E:/xzkjxm/dataes/CLCD-CD" \
        --split test

    # 或处理单张图像
    python tools/cd1_extract_instances.py \
        --t1 image_A.png --t2 image_B.png \
        --change_mask change_mask.png \
        --output_dir output/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from cd_utils import (
    imread_unicode, imwrite_unicode,
    parse_classes, build_palette, rgb_to_label,
    get_cls_display_name, save_instances, append_process_log,
    DEFAULT_CLASSES,
)


def extract_instances_from_label(label_map: np.ndarray, cls_name_map: Dict[int, List[str]],
                                 min_area: int = 4) -> List[Dict]:
    """
    从多类 label map 中提取所有实例（每个类别的每个连通域为一个实例）

    入参:
    - label_map: (H, W) uint8，每像素=类别 ID
    - cls_name_map: {cls_id: [synonyms]}
    - min_area: 最小实例面积（像素），低于此值的连通域被丢弃

    出参:
    - List[Dict]: 每个实例的信息
    """
    h, w = label_map.shape[:2]
    instances = []
    instance_id = 0

    for cls_id in range(1, max(cls_name_map.keys()) + 1):
        if cls_id not in cls_name_map:
            continue

        # 当前类别的二值掩码
        binary = (label_map == cls_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        # 连通域分析
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

        for comp_id in range(1, num_labels):
            comp_mask = (labels == comp_id).astype(np.uint8)
            area = int(comp_mask.sum())

            if area < min_area:
                continue

            # 计算 bbox [x1, y1, x2, y2]
            coords = np.where(comp_mask > 0)
            y_min, y_max = int(coords[0].min()), int(coords[0].max())
            x_min, x_max = int(coords[1].min()), int(coords[1].max())

            instance_id += 1
            instances.append({
                "instance_id": instance_id,
                "category": cls_name_map[cls_id][0],  # 主类别名
                "cls_id": cls_id,
                "bbox": [x_min, y_min, x_max + 1, y_max + 1],
                "area": area,
                "keep": True,
                "clip_score": None,
                "qwen_result": None,
            })

    return instances


def crop_instance_images(t1_img: np.ndarray, t2_img: np.ndarray,
                         change_mask_vis: np.ndarray, instance: Dict,
                         instance_dir: str, instance_name: str) -> bool:
    """
    裁剪并保存实例对应的 T1/T2/mask 子图
    """
    x1, y1, x2, y2 = instance["bbox"]

    t1_crop = t1_img[y1:y2, x1:x2].copy()
    t2_crop = t2_img[y1:y2, x1:x2].copy()
    mask_crop = change_mask_vis[y1:y2, x1:x2].copy()

    if t1_crop.size == 0 or t2_crop.size == 0:
        return False

    # 创建子目录
    sub_t1 = os.path.join(instance_dir, "t1")
    sub_t2 = os.path.join(instance_dir, "t2")
    sub_label = os.path.join(instance_dir, "label")
    for d in [sub_t1, sub_t2, sub_label]:
        os.makedirs(d, exist_ok=True)

    imwrite_unicode(os.path.join(sub_t1, f"{instance_name}.png"), t1_crop)
    imwrite_unicode(os.path.join(sub_t2, f"{instance_name}.png"), t2_crop)
    imwrite_unicode(os.path.join(sub_label, f"{instance_name}.png"), mask_crop)

    return True


def process_single_image(t1_path: str, t2_path: str, change_mask_path: str,
                         output_dir: str, sample_name: str,
                         cls_name_map: Dict, palette: np.ndarray,
                         min_area: int = 4) -> Dict:
    """
    处理单张变化掩码，提取实例并保存

    入参:
    - t1_path: T1 图像路径
    - t2_path: T2 图像路径
    - change_mask_path: 变化掩码路径 (RGB PNG 或 灰度 label PNG)
    - output_dir: 输出目录
    - sample_name: 样本名称
    - cls_name_map: 类别映射
    - palette: 调色板
    - min_area: 最小实例面积

    出参:
    - Dict: instances 数据
    """
    t1_img = imread_unicode(t1_path)
    t2_img = imread_unicode(t2_path)
    mask_img = imread_unicode(change_mask_path)

    if t1_img is None or t2_img is None or mask_img is None:
        return None

    # 将 mask 转为 RGB（用于可视化）
    if len(mask_img.shape) == 2:
        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        # 如果是灰度，判断是 label map 还是二值 mask
        unique = np.unique(mask_img)
        if len(unique) <= 2 and set(unique.tolist()).issubset({0, 255}):
            # 二值 mask → 全部标记为 cls_id=1 (building 默认)
            label_map = (mask_img > 127).astype(np.uint8)
        else:
            label_map = mask_img.copy()
    else:
        mask_rgb = mask_img
        # RGB mask → 反查 label
        label_map = rgb_to_label(mask_img, palette)

    # 确保图像尺寸一致
    h, w = label_map.shape[:2]
    if t1_img.shape[:2] != (h, w):
        t1_img = cv2.resize(t1_img, (w, h))
    if t2_img.shape[:2] != (h, w):
        t2_img = cv2.resize(t2_img, (w, h))
    if mask_rgb.shape[:2] != (h, w):
        mask_rgb = cv2.resize(mask_rgb, (w, h))

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

        success = crop_instance_images(t1_img, t2_img, mask_rgb, inst, inst_dir, inst_name)
        if success:
            inst["region_name"] = inst_name
            inst["files"] = {
                "t1": f"instances/{inst_name}/t1/{inst_name}.png",
                "t2": f"instances/{inst_name}/t2/{inst_name}.png",
                "label": f"instances/{inst_name}/label/{inst_name}.png",
            }
        else:
            inst["region_name"] = inst_name
            inst["files"] = None

    # 清理不需要在 JSON 中输出的内部字段
    for inst in instances:
        inst.pop("cls_id", None)

    result = {
        "image_name": sample_name,
        "num_instances": len(instances),
        "instances": instances,
    }

    return result


def process_dataset(data_root: str, split: str, cls_name_map: Dict,
                    palette: np.ndarray, min_area: int = 4):
    """
    处理整个数据集的 change_mask，批量提取实例

    目录结构要求:
    {data_root}/{split}/
    ├── t1/           ← T1 图像
    ├── t2/           ← T2 图像
    └── change_mask/  ← 推理输出的变化掩码 (RGB PNG)
    """
    split_dir = os.path.join(data_root, split)
    t1_dir = os.path.join(split_dir, "t1")
    t2_dir = os.path.join(split_dir, "t2")
    mask_dir = os.path.join(split_dir, "change_mask")
    output_dir = split_dir

    if not os.path.isdir(mask_dir):
        print(f"Error: change_mask 目录不存在: {mask_dir}")
        print("请先运行 infer_batch_cd.py 生成变化掩码")
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
    print(f"Step 1: 实例提取 - {split}")
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
        fname = mask_path.name

        t1_path = os.path.join(t1_dir, fname)
        t2_path = os.path.join(t2_dir, fname)

        # 尝试不同扩展名
        if not os.path.exists(t1_path):
            for ext in ['.png', '.jpg', '.tif']:
                alt = os.path.join(t1_dir, sample_name + ext)
                if os.path.exists(alt):
                    t1_path = alt
                    break
        if not os.path.exists(t2_path):
            for ext in ['.png', '.jpg', '.tif']:
                alt = os.path.join(t2_dir, sample_name + ext)
                if os.path.exists(alt):
                    t2_path = alt
                    break

        if not os.path.exists(t1_path) or not os.path.exists(t2_path):
            continue

        result = process_single_image(
            str(t1_path), str(t2_path), str(mask_path),
            output_dir, sample_name, cls_name_map, palette, min_area
        )

        if result is not None:
            # 保存每个样本的 instances.json
            instances_json_path = os.path.join(output_dir, f"{sample_name}_instances.json")
            save_instances(result, instances_json_path)
            total_instances += result["num_instances"]
            processed += 1

    # 追加日志
    append_process_log(log_path, "step1_extract", f"Extracted {total_instances} instances from {processed} samples", {
        "total_samples": len(mask_files),
        "processed": processed,
        "total_instances": total_instances,
    })

    print(f"\n{'='*60}")
    print(f"Step 1 完成")
    print(f"{'='*60}")
    print(f"样本: {processed}/{len(mask_files)}")
    print(f"实例: {total_instances}")
    print(f"{'='*60}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Step 1: 从变化掩码提取实例")
    parser.add_argument("--data_root", type=str, help="数据集根目录 (batch 模式)")
    parser.add_argument("--split", type=str, default="test", help="数据划分")
    parser.add_argument("--classes", type=str, default=None, help="类别文件路径")
    parser.add_argument("--min_area", type=int, default=4, help="最小实例面积")

    # 单图模式
    parser.add_argument("--t1", type=str, help="T1 图像路径 (单图模式)")
    parser.add_argument("--t2", type=str, help="T2 图像路径 (单图模式)")
    parser.add_argument("--change_mask", type=str, help="变化掩码路径 (单图模式)")
    parser.add_argument("--output_dir", type=str, help="输出目录 (单图模式)")

    args = parser.parse_args()

    # 类别
    cls_name_map, num_cls = parse_classes(args.classes or DEFAULT_CLASSES)
    palette = build_palette(num_cls, cls_name_map)
    palette[0] = [255, 255, 255]  # background → 白色

    if args.t1 and args.t2 and args.change_mask:
        # 单图模式
        output_dir = args.output_dir or "output_cd_postprocess"
        os.makedirs(output_dir, exist_ok=True)
        sample_name = Path(args.t1).stem
        result = process_single_image(
            args.t1, args.t2, args.change_mask,
            output_dir, sample_name, cls_name_map, palette, args.min_area
        )
        if result:
            save_instances(result, os.path.join(output_dir, f"{sample_name}_instances.json"))
            print(f"提取 {result['num_instances']} 个实例")
    elif args.data_root:
        # 批量模式
        if args.split == "all":
            for sp in ["train", "val", "test"]:
                process_dataset(args.data_root, sp, cls_name_map, palette, args.min_area)
        else:
            process_dataset(args.data_root, args.split, cls_name_map, palette, args.min_area)
    else:
        parser.error("请指定 --data_root (批量模式) 或 --t1/--t2/--change_mask (单图模式)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
