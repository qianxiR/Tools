#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单时相分割区域提取工具（SEG 模式 Step seg1）

功能：基于 label mask 的连通域分析，提取分割区域子图与 OBB 坐标
输入：{split}/image/ + {split}/label/
输出：regions/{sample}/ 目录结构

与 CD 模式的区别：
- 输入为单时相 image（非 T1/T2 双时相）
- files 字段为 {"image": "...", "label": "..."}（非 t1/t2）
- bbox 标注图只生成 image_bbox.png + label_bbox.png（无 t2）
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import math
import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm


def imread_unicode(path):
    """
    入参:
        path (str): 图像文件路径
    方法:
        使用 np.fromfile 读取文件字节流，通过 cv2.imdecode 解码，兼容中文路径
    出参:
        np.ndarray | None: 解码成功的图像数组，失败返回 None
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def imwrite_unicode(path, img):
    """
    入参:
        path (str): 目标保存路径
        img (np.ndarray): 待保存图像
    方法:
        根据文件扩展名选择编码格式，使用 tofile 写入以支持中文路径
    出参:
        bool: 是否成功保存
    """
    try:
        ext = Path(path).suffix.lower()
        if ext == '.png':
            encoded = cv2.imencode('.png', img)[1]
        elif ext in ['.jpg', '.jpeg']:
            encoded = cv2.imencode('.jpg', img)[1]
        else:
            encoded = cv2.imencode('.png', img)[1]
        encoded.tofile(path)
        return True
    except Exception:
        return False


def extract_connected_components(mask: np.ndarray, min_area: int = 2) -> List[dict]:
    """
    提取 mask 中的所有连通域，返回 YOLO-OBB 风格的旋转框标注

    入参:
    - mask (np.ndarray): 二值 mask (0或255)
    - min_area (int): 最小保留面积（像素数），低于此值的连通域被丢弃

    方法:
    - 二值化阈值127 → 8连通标记 → cv2.minAreaRect 提取最小外接旋转矩形
    - 归一化 cx/cy/w/h 至 [0,1]，angle 统一为 YOLO-OBB 约定（w>=h, angle∈[-90,90)）
    - 使用 cv2.approxPolyDP 提取简化多边形顶点（epsilon=0.5%周长）

    出参:
    - List[dict]: 每个连通域的信息字典，包含:
        - label_id: 连通域标记 ID
        - area: 前景像素面积
        - crop_bbox: 外接轴向 bbox (x, y, w, h) 像素坐标
        - rotated_rect: (cx, cy, w, h, angle) 像素级旋转框
        - obb_norm: (cx, cy, w, h, angle) 归一化 YOLO-OBB 格式
        - approx_poly: 多边形近似顶点
        - num_approx_points: 顶点数量
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    img_h, img_w = mask.shape[:2]
    components = []

    for i in range(1, num_labels):
        instance = (labels == i).astype(np.uint8)
        area = int(np.sum(instance))

        if area < min_area:
            continue

        points = cv2.findNonZero(instance)
        if points is None:
            continue

        rotated_rect = cv2.minAreaRect(points)

        box_points = cv2.boxPoints(rotated_rect)
        xs = box_points[:, 0]
        ys = box_points[:, 1]
        x_min, y_min = int(xs.min()), int(ys.min())
        x_max, y_max = int(xs.max()), int(ys.max())
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_w, x_max), min(img_h, y_max)
        crop_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        cx, cy, rw, rh, angle = _normalize_rotated_rect(rotated_rect, img_w, img_h)

        contours, _ = cv2.findContours(instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        approx_poly = []
        num_approx_points = 0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            epsilon = perimeter * 0.005
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_poly = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            num_approx_points = len(approx_poly)

        components.append({
            "label_id": i,
            "area": area,
            "crop_bbox": crop_bbox,
            "rotated_rect": rotated_rect,
            "obb_norm": (cx, cy, rw, rh, angle),
            "approx_poly": approx_poly,
            "num_approx_points": num_approx_points,
        })

    components.sort(key=lambda c: c["rotated_rect"][0][0])
    return components


def _normalize_rotated_rect(rect, img_w: int, img_h: int) -> Tuple[float, float, float, float, float]:
    """
    将 OpenCV RotatedRect 统一为 YOLO-OBB 约定并归一化

    入参:
    - rect: cv2.minAreaRect 返回的 ((cx, cy), (w, h), angle)
    - img_w, img_h: 图像宽高，用于归一化

    出参:
    - (cx, cy, w, h, angle): 归一化后的 YOLO-OBB 参数
    """
    (cx, cy), (rw, rh), angle = rect

    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0

    cx /= img_w
    cy /= img_h
    rw /= img_w
    rh /= img_h

    return (round(cx, 6), round(cy, 6), round(rw, 6), round(rh, 6), round(angle, 2))


def crop_by_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    从图像中裁剪 bbox 范围内的区域

    入参:
    - image (np.ndarray): 输入图像
    - bbox (Tuple[int, int, int, int]): 边界框 (x, y, w, h)

    出参:
    - np.ndarray: 裁剪后的图像
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    return image[y1:y2, x1:x2].copy()


def compute_spatial_location(bbox: Tuple[int, int, int, int], image_width: int, image_height: int) -> str:
    """
    基于 bbox 中心点相对于图像中心的方向，计算空间方位

    入参:
    - bbox: 区域边界框 (x, y, w, h)
    - image_width, image_height: 原图宽高

    方法:
    - 计算 bbox 中心点与图像中心点的相对偏移
    - y 轴翻转到地理方向坐标系
    - 8 个等角扇区输出英文方位

    出参:
    - str: 方位字符串 (east/northeast/north/...)
    """
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    img_cx = image_width / 2
    img_cy = image_height / 2

    dx = cx - img_cx
    dy = -(cy - img_cy)

    threshold = max(image_width, image_height) * 0.1
    if abs(dx) < threshold and abs(dy) < threshold:
        return "center"

    angle = math.atan2(dy, dx)
    sector = round(angle / (math.pi / 4)) % 8
    directions = ["east", "northeast", "north", "northwest",
                  "west", "southwest", "south", "southeast"]
    return directions[sector]


def save_region_images(img: np.ndarray, mask: np.ndarray,
                       bbox: Tuple[int, int, int, int], region_name: str,
                       output_base_dir: str) -> bool:
    """
    保存单个分割区域的 image/mask 图像

    入参:
    - img (np.ndarray): 原始图像
    - mask (np.ndarray): 完整 mask
    - bbox: 前景边界框 (x, y, w, h)
    - region_name (str): 区域名称
    - output_base_dir (str): 输出基础目录

    出参:
    - bool: 是否成功保存
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False

    img_region = crop_by_bbox(img, bbox)

    img_h, img_w = mask.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    mask_region = mask[y1:y2, x1:x2]

    if img_region.size == 0 or mask_region.size == 0:
        return False

    region_dir = os.path.join(output_base_dir, region_name)
    image_dir = os.path.join(region_dir, "image")
    label_dir = os.path.join(region_dir, "label")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    imwrite_unicode(os.path.join(image_dir, f"{region_name}.png"), img_region)
    imwrite_unicode(os.path.join(label_dir, f"{region_name}.png"), mask_region)

    return True


def process_single_sample(image_path: str, mask_path: str, output_dir: str) -> dict:
    """
    处理单个样本：提取所有前景分割区域

    入参:
    - image_path (str): 单时相图像路径
    - mask_path (str): 分割 mask 路径
    - output_dir (str): 输出目录

    方法:
    - 读取 image + mask，对 mask 做连通域分析
    - 每个连通域裁剪 image/mask 子图并保存
    - 生成 regions.json（单时相结构，files 字段为 image/label）
    - 生成带编号的 bbox 标注可视化图

    出参:
    - dict: 统计信息
    """
    img = imread_unicode(image_path)
    mask = imread_unicode(mask_path)

    if img is None or mask is None:
        return {"error": 1}

    img_h, img_w = mask.shape[:2]
    components = extract_connected_components(mask)

    if len(components) == 0:
        return {"no_change": 1}

    basename = Path(image_path).stem
    total_image_pixels = img.shape[0] * img.shape[1]
    total_fg_pixels = sum(comp["area"] for comp in components)
    saved_count = 0
    region_counter = 1
    all_bboxes = []

    for idx, comp in enumerate(components, start=1):
        region_name = f"{basename}_{region_counter:03d}"
        success = save_region_images(img, mask, comp["crop_bbox"], region_name, output_dir)
        if success:
            saved_count += 1
            region_counter += 1
            all_bboxes.append({
                "region_id": saved_count,
                "region_name": region_name,
                "crop_bbox": comp["crop_bbox"],
                "rotated_rect": comp["rotated_rect"],
                "obb_norm": comp["obb_norm"],
                "area": comp["area"],
                "location": compute_spatial_location(comp["crop_bbox"], img_w, img_h),
                "approx_poly": comp["approx_poly"],
                "num_approx_points": comp["num_approx_points"],
            })

    # 样本汇总目录：含 regions.json + 标注可视化图
    sample_dir = os.path.join(output_dir, basename)
    os.makedirs(sample_dir, exist_ok=True)

    # 背景区域 _000：保存完整 image + mask
    bg_name = f"{basename}_000"
    bg_dir = os.path.join(output_dir, bg_name)
    os.makedirs(os.path.join(bg_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(bg_dir, "label"), exist_ok=True)
    imwrite_unicode(os.path.join(bg_dir, "image", f"{bg_name}.png"), img)
    imwrite_unicode(os.path.join(bg_dir, "label", f"{bg_name}.png"), mask)

    # 生成带编号的 bbox 标注可视化图
    if all_bboxes and img is not None:
        img_annotated = img.copy()
        label_annotated = mask.copy()
        if len(label_annotated.shape) == 2:
            label_annotated = cv2.cvtColor(label_annotated, cv2.COLOR_GRAY2BGR)

        for idx, item in enumerate(all_bboxes, start=1):
            poly_points = item["approx_poly"]
            if not poly_points or len(poly_points) < 3:
                continue

            poly_points_np = np.array(poly_points, dtype=np.int32).reshape(-1, 1, 2)
            cx = int(np.mean([pt[0] for pt in poly_points]))
            cy = int(np.mean([pt[1] for pt in poly_points]))

            cv2.polylines(img_annotated, [poly_points_np], True, (0, 0, 255), 2)
            cv2.polylines(label_annotated, [poly_points_np], True, (0, 0, 255), 2)

            for pt in poly_points:
                cv2.circle(img_annotated, tuple(pt), 3, (0, 0, 255), -1)
                cv2.circle(label_annotated, tuple(pt), 3, (0, 0, 255), -1)

            cv2.putText(img_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            cv2.putText(label_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(label_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        imwrite_unicode(os.path.join(sample_dir, "image_bbox.png"), img_annotated)
        imwrite_unicode(os.path.join(sample_dir, "label_bbox.png"), label_annotated)

    # 保存原始 mask
    if mask is not None:
        cv2.imwrite(os.path.join(sample_dir, "original_label.png"), mask)

    # 背景区域记录（region_id=0，表示完整图像）
    background_region = {
        "region_id": 0,
        "region_name": f"{basename}_000",
        "location": "full_image",
        "area_pixels": total_image_pixels - total_fg_pixels,
        "obb": {
            "class_id": 0,
            "cx": 0.5,
            "cy": 0.5,
            "w": 1.0,
            "h": 1.0,
            "angle": 0.0,
        },
        "files": {
            "image": f"{basename}_000/image/{basename}_000.png",
            "label": f"{basename}_000/label/{basename}_000.png",
        },
        "classification": None,
    }

    def compute_obb_in_original(item: dict, img_w: int, img_h: int) -> dict:
        rotated_rect = item["rotated_rect"]
        cx_px, cy_px = rotated_rect[0]
        rw_px, rh_px = rotated_rect[1]
        angle = rotated_rect[2]

        return {
            "class_id": 1,
            "cx": round(cx_px / img_w, 6),
            "cy": round(cy_px / img_h, 6),
            "w": round(rw_px / img_w, 6),
            "h": round(rh_px / img_h, 6),
            "angle": round(angle, 2),
        }

    regions_json_data = {
        "image_name": basename,
        "image_size": {"h": img_h, "w": img_w},
        "num_regions": len(all_bboxes),
        "total_image_pixels": total_image_pixels,
        "total_fg_pixels": total_fg_pixels,
        "regions": [
            background_region,
        ] + [
            {
                "region_id": item["region_id"],
                "region_name": item["region_name"],
                "location": item["location"],
                "area_pixels": item["area"],
                "obb": compute_obb_in_original(item, img_w, img_h),
                "approx_poly": item["approx_poly"],
                "num_approx_points": item["num_approx_points"],
                "segmentation": [coord for pt in item["approx_poly"] for coord in pt],
                "files": {
                    "image": f"{item['region_name']}/image/{item['region_name']}.png",
                    "label": f"{item['region_name']}/label/{item['region_name']}.png",
                },
                "classification": None,
            }
            for item in all_bboxes
        ],
    }
    with open(os.path.join(sample_dir, "regions.json"), "w", encoding="utf-8") as f:
        json.dump(regions_json_data, f, ensure_ascii=False, indent=2)

    return {"total_regions": len(components), "saved_regions": saved_count}


def process_dataset(dataset_path: str, split: str):
    """
    处理整个数据集

    入参:
    - dataset_path (str): 数据集根目录
    - split (str): 数据集分割 (train/val/test)

    方法:
    - 扫描 {split}/image/ 下的图像文件
    - 逐样本调用 process_single_sample 提取区域
    """
    image_dir = os.path.join(dataset_path, split, "image")
    mask_dir = os.path.join(dataset_path, split, "label")
    output_dir = os.path.join(dataset_path, split, "regions")

    image_files = sorted(
        list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
        + list(Path(image_dir).glob("*.tif")) + list(Path(image_dir).glob("*.tiff"))
    )

    print(f"\n{'='*60}")
    print(f"SEG Extract Regions - {split} split")
    print(f"{'='*60}")
    print(f"Samples: {len(image_files)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    stats = {"total_samples": len(image_files), "total_regions": 0, "saved_regions": 0, "no_fg": 0, "error": 0}

    for image_path in tqdm(image_files, desc=f"Processing {split}"):
        basename = image_path.stem
        mask_path = os.path.join(mask_dir, image_path.name)

        try:
            result = process_single_sample(str(image_path), mask_path, output_dir)

            if "error" in result:
                stats["error"] += 1
            elif "no_change" in result:
                stats["no_fg"] += 1
            else:
                stats["total_regions"] += result["total_regions"]
                stats["saved_regions"] += result["saved_regions"]

        except Exception as e:
            stats["error"] += 1
            print(f"\nError processing {basename}: {e}")

    print(f"\n{'='*60}")
    print(f"SEG Extract Complete - {split}")
    print(f"{'='*60}")
    print(f"Samples:   {stats['total_samples']}")
    print(f"Regions:   {stats['total_regions']} ({stats['saved_regions']} saved)")
    print(f"No foreground: {stats['no_fg']}, Errors: {stats['error']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SEG mode: Extract foreground segmentation regions")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="all",
                       choices=["train", "val", "test", "all"],
                       help="Dataset split to process (default: all)")

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return 1

    if args.split == "all":
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(args.dataset_path, split)
            if os.path.isdir(split_dir):
                process_dataset(args.dataset_path, split)
    else:
        process_dataset(args.dataset_path, args.split)

    print("\n" + "=" * 60)
    print("SEG extraction completed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
