#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 metadata.json 转换为 COCO 目标检测格式

功能：
1. 读取所有 split 的 metadata.json
2. 转换为标准 COCO 格式
3. 支持 OBB（旋转框）和 segmentation（多边形）

输出：
- annotations.json — 合并所有 split 的 COCO 格式标注文件

COCO 格式说明：
{
  "images": [{"id": 1, "file_name": "001.jpg", "width": 512, "height": 512}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, w, h],
      "area": 1234,
      "segmentation": [[x1,y1,x2,y2,...]],
      "iscrowd": 0
    }
  ],
  "categories": [{"id": 1, "name": "building"}]
}
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


# 类别映射 (与 CLIP 分类一致)
CLASS_NAMES = [
    "background",
    "building",
    "highway",
    "vegetation",
    "farmland",
    "bareland",
    "water",
]

# 创建类别 ID 映射
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_metadata(split_dir: str, split_name: str) -> Optional[Dict]:
    """
    加载单个 split 的 metadata.json

    入参:
    - split_dir (str): split 目录路径
    - split_name (str): split 名称

    出参:
    - Optional[Dict]: metadata 数据
    """
    meta_path = os.path.join(split_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_to_coco(dataset_path: str, splits: List[str], mode: str = "cd") -> Dict:
    """
    将 metadata.json 转换为 COCO 格式

    入参:
    - dataset_path (str): 数据集根目录
    - splits (List[str]): 要处理的 split 列表
    - mode (str): cd 或 seg，决定分类字段提取逻辑

    出参:
    - Dict: COCO 格式数据
    """
    images = []
    annotations = []
    categories = []

    # 创建类别列表
    for class_name in CLASS_NAMES:
        if class_name == "background":
            continue
        categories.append({
            "id": CLASS_TO_ID[class_name],
            "name": class_name,
            "supercategory": "object",
        })

    image_id = 1
    annotation_id = 1

    for split in splits:
        split_dir = os.path.join(dataset_path, split)
        metadata = load_metadata(split_dir, split)

        if metadata is None:
            continue

        # metadata 是数组格式
        samples = metadata if isinstance(metadata, list) else metadata.get("samples", [])

        for sample in samples:
            image_name = sample.get("image_name", "")
            if not image_name:
                continue

            image_size = sample.get("image_size", {})
            img_w = image_size.get("w", 0)
            img_h = image_size.get("h", 0)

            # 添加 image 记录
            images.append({
                "id": image_id,
                "file_name": f"{image_name}.png",
                "width": img_w,
                "height": img_h,
            })

            # 处理区域标注
            regions = sample.get("regions", [])
            for region in regions:
                region_id = region.get("region_id", 0)
                if int(region_id) <= 0:
                    continue

                # 获取分类信息
                classification = region.get("classification", {})
                class_name = ""

                if mode == "seg":
                    # SEG 模式：从 final/llm/clip 提取单类别
                    for source in ("final", "llm", "clip"):
                        payload = classification.get(source, {})
                        if isinstance(payload, dict):
                            cls = str(payload.get("class", "")).strip().lower()
                            if cls:
                                class_name = cls
                                break
                else:
                    # CD 模式：优先 t2_class，其次 t1_class
                    t1_class = classification.get("t1_class", "")
                    t2_class = classification.get("t2_class", "")
                    class_name = t2_class or t1_class
                if not class_name or class_name == "background":
                    continue

                category_id = CLASS_TO_ID.get(class_name)
                if category_id is None or category_id == 0:
                    continue

                # 获取 OBB 信息（转换为轴向 bbox）
                obb = region.get("obb", {})
                cx_norm = obb.get("cx", 0.5)
                cy_norm = obb.get("cy", 0.5)
                w_norm = obb.get("w", 0)
                h_norm = obb.get("h", 0)

                # 转换为像素坐标
                cx_px = int(cx_norm * img_w)
                cy_px = int(cy_norm * img_h)
                w_px = int(w_norm * img_w)
                h_px = int(h_norm * img_h)

                # 计算轴向 bbox (左上角 x, y, 宽度, 高度)
                bbox_x = int(cx_px - w_px / 2)
                bbox_y = int(cy_px - h_px / 2)

                # 获取面积
                area = region.get("area_pixels", 0)
                if area <= 0:
                    area = w_px * h_px

                # 获取 segmentation（COCO 格式需要嵌套数组）
                segmentation = region.get("segmentation", [])
                if segmentation:
                    # 将扁平数组转换为嵌套数组
                    segmentation = [segmentation]
                else:
                    # 使用 approx_poly 构建 segmentation
                    approx_poly = region.get("approx_poly", [])
                    if approx_poly:
                        segmentation = [[coord for pt in approx_poly for coord in pt]]
                    else:
                        # 使用 OBB 的四个顶点
                        import numpy as np
                        import cv2
                        angle = obb.get("angle", 0.0)
                        rw_px, rh_px = w_px, h_px
                        angle_cv = angle
                        if rw_px < rh_px:
                            rw_px, rh_px = rh_px, rw_px
                            angle_cv += 90.0
                        rect = ((cx_px, cy_px), (rw_px, rh_px), angle_cv)
                        box_pts = cv2.boxPoints(rect).astype(int).tolist()
                        segmentation = [[coord for pt in box_pts for coord in pt]]

                # 添加 annotation 记录
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [bbox_x, bbox_y, w_px, h_px],
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }

                # 添加 OBB 信息（扩展字段）
                annotation["obb"] = {
                    "cx": cx_norm,
                    "cy": cy_norm,
                    "w": w_norm,
                    "h": h_norm,
                    "angle": obb.get("angle", 0.0),
                }

                annotations.append(annotation)
                annotation_id += 1

            image_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert metadata.json to COCO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset_path", "-d", type=str, required=True,
                       help="Dataset root directory")
    parser.add_argument("--split", "-s", type=str, default="all",
                       choices=["train", "val", "test", "all"],
                       help="Dataset split to convert (default: all)")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                       help="Output directory (default: dataset_path)")
    parser.add_argument("--mode", type=str, default="cd",
                       choices=["cd", "seg"],
                       help="Mode: cd=change detection, seg=segmentation (default: cd)")

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return 1

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.dataset_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]

    print("=" * 70)
    print("Converting metadata.json to COCO format")
    print("=" * 70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Splits: {splits_to_process}")
    print(f"Mode: {args.mode}")
    print("=" * 70)

    # 转换所有 splits
    coco_data = convert_to_coco(args.dataset_path, splits_to_process, args.mode)

    # 保存合并的 COCO 文件
    output_path = output_dir / "annotations.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved: {output_path}")
    print(f"   Images: {len(coco_data['images'])}")
    print(f"   Annotations: {len(coco_data['annotations'])}")
    print(f"   Categories: {len(coco_data['categories'])}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
