#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEG Step 4: 重建最终分割掩码

根据 instances.json 中的 keep 字段，从原始掩码中剔除误检实例，
直接覆盖 seg_mask 目录中的原始文件。

用法:
    python tools/seg4_rebuild_mask.py \
        --data_root "E:/xzkjxm/dataes/WHU_20" \
        --split test
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
from PIL import Image

from cd_utils import (
    imread_unicode, imwrite_unicode,
    parse_classes, build_palette, rgb_to_label,
    load_instances, append_process_log,
    DEFAULT_CLASSES,
)


def _get_cls_name_map():
    """获取默认类别映射（内部使用）"""
    cls_name_map, _ = parse_classes(DEFAULT_CLASSES)
    return cls_name_map


def mask_to_overlay(img_np: np.ndarray, mask: np.ndarray,
                    palette: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """原图 + 彩色掩码半透明叠加"""
    img_f = img_np.astype(np.float32)
    color_mask = palette[mask].astype(np.float32)
    region = (mask > 0)[:, :, np.newaxis].astype(np.float32)
    out = img_f * (1 - alpha * region) + color_mask * alpha * region
    return out.astype(np.uint8)


def rebuild_single_mask(instances_path: str, mask_path: str,
                        palette: np.ndarray) -> Dict:
    """
    根据 instances.json 的 keep 字段，对单个掩码文件去除误检实例，原地覆盖保存。

    入参:
    - instances_path: instances.json 路径
    - mask_path: 原始掩码路径（RGB PNG），会被原地覆盖
    - palette: 调色板

    出参:
    - Dict: 统计信息 {removed_instances, removed_area}
    """
    data = load_instances(instances_path)
    if data is None:
        return {"error": 1}

    instances = data.get("instances", [])
    if not instances:
        return {"no_instances": 1}

    # 读取原始掩码 → label map
    mask_img = imread_unicode(mask_path)
    if mask_img is None:
        return {"error": 1}

    if len(mask_img.shape) == 2:
        label_map = mask_img.copy()
    else:
        label_map = rgb_to_label(mask_img, palette)

    h, w = label_map.shape[:2]

    # 创建 remove_mask: 标记需要删除的像素
    remove_mask = np.zeros((h, w), dtype=np.uint8)

    # 预构建类别名→ID映射
    cls_name_map_inv = {}
    for idx, syns in _get_cls_name_map().items():
        for s in syns:
            cls_name_map_inv[s] = idx

    removed_count = 0
    removed_area = 0

    for inst in instances:
        if inst.get("keep", True):
            continue

        bbox = inst.get("bbox")
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        # 钳制到图像范围
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 标记该实例的 bbox 区域中属于该类别的像素为删除
        category = inst.get("category", "")
        target_cls_id = cls_name_map_inv.get(category, -1)
        if target_cls_id > 0:
            region = label_map[y1:y2, x1:x2]
            match = (region == target_cls_id)
            # 进一步用连通域精确匹配（避免删除同类其他实例）
            if match.any():
                match_uint8 = match.astype(np.uint8)
                n_labels, labels_comp = cv2.connectedComponents(match_uint8, connectivity=8)
                target_area = inst.get("area", 0)
                best_label = 0
                best_diff = float('inf')
                for lbl in range(1, n_labels):
                    comp_area = int((labels_comp == lbl).sum())
                    diff = abs(comp_area - target_area)
                    if diff < best_diff:
                        best_diff = diff
                        best_label = lbl

                if best_label > 0:
                    precise_mask = (labels_comp == best_label)
                    remove_mask[y1:y2, x1:x2][precise_mask] = 1

        removed_count += 1
        removed_area += inst.get("area", 0)

    if removed_count == 0:
        return {"removed_instances": 0, "removed_area": 0}

    # 从 label_map 中删除标记的像素
    final_label = label_map.copy()
    final_label[remove_mask > 0] = 0  # 置为 background

    # 渲染 RGB 并原地覆盖
    final_rgb = palette[final_label]
    imwrite_unicode(mask_path, final_rgb)

    return {
        "removed_instances": removed_count,
        "removed_area": removed_area,
    }


def process_dataset(data_root: str, split: str, cls_name_map: Dict, palette: np.ndarray):
    """
    批量重建分割掩码 — 处理 seg_mask 目录

    目录结构:
    {data_root}/{split}/
    ├── image/       ← 原始图像
    ├── seg_mask/    ← 分割掩码（会被原地覆盖）
    └── instances/       ← 后处理实例裁剪子图
    """
    split_dir = os.path.join(data_root, split)
    postprocess_dir = split_dir

    # 查找 instances.json
    json_files = sorted(Path(postprocess_dir).glob("*_instances.json"))
    if not json_files:
        print("Warning: 未找到 instances.json")
        return

    # 需要处理的掩码目录
    mask_dir = os.path.join(split_dir, "seg_mask")
    if not os.path.isdir(mask_dir):
        print(f"Warning: seg_mask 目录不存在: {mask_dir}")
        return

    # 可选的 overlay 目录
    overlay_dir = os.path.join(split_dir, "overlay")
    image_dir = os.path.join(split_dir, "image")
    # 兼容 img/ 目录名
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(split_dir, "img")

    has_overlay = os.path.isdir(overlay_dir)

    print(f"\n{'='*60}")
    print(f"SEG Step 4: 重建最终掩码 - {split}")
    print(f"{'='*60}")
    print(f"样本数: {len(json_files)}")
    print(f"掩码目录: {mask_dir}")
    print(f"{'='*60}\n")

    log_path = os.path.join(postprocess_dir, "process_log.json")
    stats = {"total_removed": 0, "total_removed_area": 0, "total_files_processed": 0}

    for jf in tqdm(json_files, desc=f"Rebuilding {split}"):
        sample_name = jf.stem.replace("_instances", "")

        # 查找该样本的掩码文件
        mask_path = None
        for ext in ['.png', '.jpg', '.tif']:
            candidate = os.path.join(mask_dir, sample_name + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            continue

        # 执行重建并覆盖
        result = rebuild_single_mask(str(jf), mask_path, palette)
        stats["total_removed"] += result.get("removed_instances", 0)
        stats["total_removed_area"] += result.get("removed_area", 0)

        has_changes = result.get("removed_instances", 0) > 0
        if has_changes:
            stats["total_files_processed"] += 1

        # 重新生成 overlay（仅在有变更且 overlay 目录存在时）
        if has_changes and has_overlay:
            # 查找原始图像
            img = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                candidate = os.path.join(image_dir, sample_name + ext)
                if os.path.exists(candidate):
                    img = imread_unicode(candidate)
                    break

            if img is not None:
                # 读取重建后的掩码
                rebuilt_mask = imread_unicode(mask_path)
                if rebuilt_mask is not None:
                    rebuilt_label = rgb_to_label(rebuilt_mask, palette)

                    h, w = img.shape[:2]
                    ov = img[:h, :w].copy()
                    ov = mask_to_overlay(ov, rebuilt_label, palette)

                    # 查找原始 overlay 文件名并覆盖
                    ov_path = None
                    for ext in ['.png', '.jpg', '.tif']:
                        candidate = os.path.join(overlay_dir, sample_name + ext)
                        if os.path.exists(candidate):
                            ov_path = candidate
                            break
                    if ov_path is None:
                        ov_path = os.path.join(overlay_dir, sample_name + ".png")

                    Image.fromarray(ov).save(ov_path)

    append_process_log(log_path, "step4_rebuild",
                       f"Removed {stats['total_removed']} instances from {stats['total_files_processed']} files", stats)

    print(f"\n{'='*60}")
    print(f"SEG Step 4 完成")
    print(f"{'='*60}")
    print(f"删除实例: {stats['total_removed']}")
    print(f"删除面积: {stats['total_removed_area']} px")
    print(f"修改文件: {stats['total_files_processed']} 个")
    print(f"覆盖目录: {mask_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SEG Step 4: 重建最终掩码")
    parser.add_argument("--data_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--split", type=str, default="test", help="数据划分")
    parser.add_argument("--classes", type=str, default=None, help="类别文件路径")

    args = parser.parse_args()

    cls_name_map, num_cls = parse_classes(args.classes or DEFAULT_CLASSES)
    palette = build_palette(num_cls, cls_name_map)
    palette[0] = [255, 255, 255]  # background → 白色

    if args.split == "all":
        for sp in ["train", "val", "test"]:
            process_dataset(args.data_root, sp, cls_name_map, palette)
    else:
        process_dataset(args.data_root, args.split, cls_name_map, palette)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
