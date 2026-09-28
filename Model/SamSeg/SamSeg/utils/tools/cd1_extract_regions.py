#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变化区域提取工具

功能：基于 label mask 的连通域分析，提取变化区域子图与 OBB 坐标
输出：regions/{sample}/ 目录结构
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


def imread_unicode(path):
    """支持中文路径的图片读取"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def imwrite_unicode(path, img):
    """支持中文路径的图片写入"""
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
import numpy as np
from tqdm import tqdm


def extract_connected_components(mask: np.ndarray, min_area: int = 2) -> List[dict]:
    """
    提取 mask 中的所有连通域，返回 YOLO-OBB 风格的旋转框标注

    入参:
    - mask (np.ndarray): 二值 mask (0或255)
    - min_area (int): 最小保留面积（像素数），低于此值的连通域被丢弃，默认 2 以移除 1×1 噪声

    方法:
    - 二值化阈值127 → 8连通标记 → cv2.minAreaRect 提取最小外接旋转矩形
    - 裁剪范围由旋转矩形的四个顶点外接轴向 bbox 决定，保证完整覆盖旋转框
    - 归一化 cx/cy/w/h 至 [0,1]，angle 统一为 YOLO-OBB 约定（w>=h, angle∈[-90,90)）
    - 使用 cv2.approxPolyDP 提取简化多边形顶点（epsilon=2%周长）

    出参:
    - List[dict]: 每个连通域的信息字典，包含:
        - label_id: 连通域标记 ID
        - area: 前景像素面积
        - crop_bbox: 旋转框外接轴向 bbox (x, y, w, h) 像素坐标，供裁剪使用
        - rotated_rect: (cx, cy, w, h, angle) 像素级旋转框（OpenCV 约定）
        - obb_norm: (cx, cy, w, h, angle) 归一化 YOLO-OBB 格式
        - approx_poly: 多边形近似顶点 [[x1,y1], [x2,y2], ...]
        - num_approx_points: 顶点数量
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 连通域分析（8连通）
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    img_h, img_w = mask.shape[:2]
    components = []

    for i in range(1, num_labels):
        instance = (labels == i).astype(np.uint8)
        area = int(np.sum(instance))

        # 过滤面积过小的噪声连通域
        if area < min_area:
            continue

        # 前景像素坐标
        points = cv2.findNonZero(instance)
        if points is None:
            continue

        # 最小外接旋转矩形：用于 YOLO-OBB 标注
        rotated_rect = cv2.minAreaRect(points)

        # 旋转矩形的四个顶点 → 外接轴向 bbox，保证完整覆盖旋转框
        box_points = cv2.boxPoints(rotated_rect)
        xs = box_points[:, 0]
        ys = box_points[:, 1]
        x_min, y_min = int(xs.min()), int(ys.min())
        x_max, y_max = int(xs.max()), int(ys.max())
        # 钳制到图像边界内
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_w, x_max), min(img_h, y_max)
        crop_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        cx, cy, rw, rh, angle = _normalize_rotated_rect(rotated_rect, img_w, img_h)

        # 提取多边形近似顶点
        contours, _ = cv2.findContours(instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        approx_poly = []
        num_approx_points = 0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            epsilon = perimeter * 0.005  # 更精细的近似（原0.02）
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_poly = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            num_approx_points = len(approx_poly)

        components.append({
            "label_id": i,
            "area": area,
            "crop_bbox": crop_bbox,
            "rotated_rect": rotated_rect,
            "obb_norm": (cx, cy, rw, rh, angle),
            "obb_original": (cx * img_w, cy * img_h, rw * img_w, rh * img_h, angle),
            "approx_poly": approx_poly,
            "num_approx_points": num_approx_points,
        })

    # 按旋转矩形中心 x 坐标排序，保持处理顺序稳定
    components.sort(key=lambda c: c["rotated_rect"][0][0])
    return components


def _normalize_rotated_rect(rect: cv2.RotatedRect, img_w: int, img_h: int) -> Tuple[float, float, float, float, float]:
    """
    将 OpenCV RotatedRect 统一为 YOLO-OBB 约定并归一化

    YOLO-OBB 约定: w >= h，angle 范围 [-90, 90)
    OpenCV 约定: angle ∈ (-90, 0]，width 可能小于 height

    入参:
    - rect: cv2.minAreaRect 返回的 ((cx, cy), (w, h), angle)
    - img_w, img_h: 图像宽高，用于归一化

    出参:
    - (cx, cy, w, h, angle): 归一化后的 YOLO-OBB 参数
    """
    (cx, cy), (rw, rh), angle = rect

    # OpenCV angle ∈ (-90, 0]；若 rw < rh，交换并调整角度以保持 w >= h
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0

    # 归一化坐标和尺寸
    cx /= img_w
    cy /= img_h
    rw /= img_w
    rh /= img_h

    return (round(cx, 6), round(cy, 6), round(rw, 6), round(rh, 6), round(angle, 2))


def crop_by_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    从图像中裁剪 bbox 范围内的区域，保留全部原始像素

    入参:
    - image (np.ndarray): 输入图像 (H, W, 3)
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
    基于 bbox 中心点相对于图像中心的方向，计算变化区域的空间方位

    入参:
    - bbox (Tuple[int, int, int, int]): 区域边界框 (x, y, w, h)，使用原图绝对像素坐标
    - image_width (int): 原图宽度（像素）
    - image_height (int): 原图高度（像素）

    方法:
    - 先计算 bbox 中心点与图像中心点的相对偏移，避免使用左上角导致方位判断失真
    - 将图像坐标系的 y 轴翻转到地理方向坐标系，使“上方区域”稳定映射为 north
    - 使用 8 个等角扇区输出英文方位，若区域过于接近图像中心则统一记为 center

    出参:
    - str: 英文方位字符串，取值为 east/northeast/north/northwest/west/southwest/south/southeast/center
    """
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    img_cx = image_width / 2
    img_cy = image_height / 2

    # 使用中心点偏移而不是左上角偏移，避免大框区域的方位判断被裁剪尺寸干扰。
    dx = cx - img_cx
    dy = -(cy - img_cy)

    # 中心附近的小范围变化统一记为 center，减少边界抖动导致的方向跳变。
    threshold = max(image_width, image_height) * 0.1
    if abs(dx) < threshold and abs(dy) < threshold:
        return "center"

    angle = math.atan2(dy, dx)
    sector = round(angle / (math.pi / 4)) % 8
    directions = ["east", "northeast", "north", "northwest",
                  "west", "southwest", "south", "southeast"]
    return directions[sector]


def save_region_images(t1_img: np.ndarray, t2_img: np.ndarray, mask: np.ndarray,
                       bbox: Tuple[int, int, int, int], region_name: str,
                       output_base_dir: str) -> bool:
    """
    保存单个变化区域的 T1/T2/mask 图像
    T1/T2 仅保留前景像素（背景置黑），裁剪范围与前景边界框一致

    入参:
    - t1_img (np.ndarray): T1 图像
    - t2_img (np.ndarray): T2 图像
    - mask (np.ndarray): 完整 mask
    - bbox (Tuple[int, int, int, int]): 前景边界框 (x, y, w, h)
    - region_name (str): 区域名称
    - output_base_dir (str): 输出基础目录

    出参:
    - bool: 是否成功保存
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False

    # 裁剪 bbox 范围内的 T1/T2（保留全部像素）
    t1_region = crop_by_bbox(t1_img, bbox)
    t2_region = crop_by_bbox(t2_img, bbox)

    # 提取边界框范围内的 mask
    img_h, img_w = mask.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    mask_region = mask[y1:y2, x1:x2]

    if t1_region.size == 0 or t2_region.size == 0 or mask_region.size == 0:
        return False

    # 保存
    region_dir = os.path.join(output_base_dir, region_name)
    t1_dir = os.path.join(region_dir, "t1")
    t2_dir = os.path.join(region_dir, "t2")
    label_dir = os.path.join(region_dir, "label")

    os.makedirs(t1_dir, exist_ok=True)
    os.makedirs(t2_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    imwrite_unicode(os.path.join(t1_dir, f"{region_name}.png"), t1_region)
    imwrite_unicode(os.path.join(t2_dir, f"{region_name}.png"), t2_region)
    imwrite_unicode(os.path.join(label_dir, f"{region_name}.png"), mask_region)

    return True


def process_single_sample(t1_path: str, t2_path: str, mask_path: str, output_dir: str) -> dict:
    """
    处理单个样本，提取所有前景变化区域

    入参:
    - t1_path (str): T1 图像路径
    - t2_path (str): T2 图像路径
    - mask_path (str): mask 路径
    - output_dir (str): 输出目录

    出参:
    - dict: 统计信息
    """
    t1_img = imread_unicode(t1_path)
    t2_img = imread_unicode(t2_path)
    mask = imread_unicode(mask_path)

    if t1_img is None or t2_img is None or mask is None:
        return {"error": 1}

    img_h, img_w = mask.shape[:2]
    components = extract_connected_components(mask)

    if len(components) == 0:
        return {"no_change": 1}

    basename = Path(t1_path).stem
    total_image_pixels = t1_img.shape[0] * t1_img.shape[1]
    total_change_pixels = sum(comp["area"] for comp in components)
    saved_count = 0
    region_counter = 1
    all_bboxes = []

    for idx, comp in enumerate(components, start=1):
        region_name = f"{basename}_{region_counter:03d}"
        success = save_region_images(t1_img, t2_img, mask, comp["crop_bbox"], region_name, output_dir)
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

    sample_dir = os.path.join(output_dir, basename)
    os.makedirs(sample_dir, exist_ok=True)

    bg_name = f"{basename}_000"
    bg_dir = os.path.join(output_dir, bg_name)
    os.makedirs(os.path.join(bg_dir, "t1"), exist_ok=True)
    os.makedirs(os.path.join(bg_dir, "t2"), exist_ok=True)
    os.makedirs(os.path.join(bg_dir, "label"), exist_ok=True)
    imwrite_unicode(os.path.join(bg_dir, "t1", f"{bg_name}.png"), t1_img)
    imwrite_unicode(os.path.join(bg_dir, "t2", f"{bg_name}.png"), t2_img)
    imwrite_unicode(os.path.join(bg_dir, "label", f"{bg_name}.png"), mask)

    if all_bboxes and t1_img is not None and t2_img is not None:
        t1_annotated = t1_img.copy()
        t2_annotated = t2_img.copy()
        label_annotated = mask.copy()
        if len(label_annotated.shape) == 2:
            label_annotated = cv2.cvtColor(label_annotated, cv2.COLOR_GRAY2BGR)

        for idx, item in enumerate(all_bboxes, start=1):
            # 使用多边形顶点绘制点标注
            poly_points = item["approx_poly"]
            if not poly_points or len(poly_points) < 3:
                continue

            poly_points_np = np.array(poly_points, dtype=np.int32).reshape(-1, 1, 2)

            # 计算多边形中心用于标注类别编号
            cx = int(np.mean([pt[0] for pt in poly_points]))
            cy = int(np.mean([pt[1] for pt in poly_points]))

            # 绘制红色多边形轮廓线
            cv2.polylines(t1_annotated, [poly_points_np], True, (0, 0, 255), 2)
            cv2.polylines(t2_annotated, [poly_points_np], True, (0, 0, 255), 2)
            cv2.polylines(label_annotated, [poly_points_np], True, (0, 0, 255), 2)

            # 绘制多边形顶点（红色圆点）
            for pt in poly_points:
                cv2.circle(t1_annotated, tuple(pt), 3, (0, 0, 255), -1)
                cv2.circle(t2_annotated, tuple(pt), 3, (0, 0, 255), -1)
                cv2.circle(label_annotated, tuple(pt), 3, (0, 0, 255), -1)

            # 绘制类别编号（白色文字+黑色边框）
            cv2.putText(t1_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(t1_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            cv2.putText(t2_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(t2_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            cv2.putText(label_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(label_annotated, str(idx), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        imwrite_unicode(os.path.join(sample_dir, "t1_bbox.png"), t1_annotated)
        imwrite_unicode(os.path.join(sample_dir, "t2_bbox.png"), t2_annotated)
        imwrite_unicode(os.path.join(sample_dir, "label_bbox.png"), label_annotated)

    if mask is not None:
        cv2.imwrite(os.path.join(sample_dir, "original_label.png"), mask)

    background_region = {
        "region_id": 0,
        "region_name": f"{basename}_000",
        "location": "full_image",
        "area_pixels": total_image_pixels - total_change_pixels,
        "obb": {
            "class_id": 0,
            "cx": 0.5,
            "cy": 0.5,
            "w": 1.0,
            "h": 1.0,
            "angle": 0.0,
        },
        "files": {
            "t1": f"{basename}_000/t1/{basename}_000.png",
            "t2": f"{basename}_000/t2/{basename}_000.png",
            "label": f"{basename}_000/label/{basename}_000.png",
        },
        "classification": None,
    }

    def compute_obb_in_original(item: dict, img_w: int, img_h: int) -> dict:
        rotated_rect = item["rotated_rect"]
        cx_px, cy_px = rotated_rect[0]
        rw_px, rh_px = rotated_rect[1]
        angle = rotated_rect[2]

        cx_norm = round(cx_px / img_w, 6)
        cy_norm = round(cy_px / img_h, 6)
        w_norm = round(rw_px / img_w, 6)
        h_norm = round(rh_px / img_h, 6)

        return {
            "class_id": 1,
            "cx": cx_norm,
            "cy": cy_norm,
            "w": w_norm,
            "h": h_norm,
            "angle": round(angle, 2),
        }

    regions_json_data = {
        "image_name": basename,
        "image_size": {"h": img_h, "w": img_w},
        "num_regions": len(all_bboxes),
        "total_image_pixels": total_image_pixels,
        "total_change_pixels": total_change_pixels,
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
                "segmentation": [coord for pt in item["approx_poly"] for coord in pt],  # COCO格式: [x1,y1,x2,y2,...]
                "files": {
                    "t1": f"{item['region_name']}/t1/{item['region_name']}.png",
                    "t2": f"{item['region_name']}/t2/{item['region_name']}.png",
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
    """
    t1_dir = os.path.join(dataset_path, split, "t1")
    t2_dir = os.path.join(dataset_path, split, "t2")
    mask_dir = os.path.join(dataset_path, split, "label")
    output_dir = os.path.join(dataset_path, split, "regions")

    t1_files = sorted(list(Path(t1_dir).glob("*.png")) + list(Path(t1_dir).glob("*.jpg")))

    print(f"\n{'='*60}")
    print(f"Extract Regions - {split} split")
    print(f"{'='*60}")
    print(f"Samples: {len(t1_files)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    stats = {"total_samples": len(t1_files), "total_regions": 0, "saved_regions": 0, "no_change": 0, "error": 0}

    for t1_path in tqdm(t1_files, desc=f"Processing {split}"):
        basename = t1_path.stem
        t2_path = os.path.join(t2_dir, t1_path.name)
        mask_path = os.path.join(mask_dir, t1_path.name)

        try:
            result = process_single_sample(str(t1_path), t2_path, mask_path, output_dir)

            if "error" in result:
                stats["error"] += 1
            elif "no_change" in result:
                stats["no_change"] += 1
            else:
                stats["total_regions"] += result["total_regions"]
                stats["saved_regions"] += result["saved_regions"]

        except Exception as e:
            stats["error"] += 1
            print(f"\nError processing {basename}: {e}")

    print(f"\n{'='*60}")
    print(f"Extract Complete - {split}")
    print(f"{'='*60}")
    print(f"Samples:   {stats['total_samples']}")
    print(f"Regions:   {stats['total_regions']} ({stats['saved_regions']} saved)")
    print(f"No change: {stats['no_change']}, Errors: {stats['error']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Extract foreground change regions")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"],
                       help="Dataset split to process (default: all)")

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return 1

    if args.split == "all":
        for split in ["train", "val", "test"]:
            process_dataset(args.dataset_path, split)
    else:
        process_dataset(args.dataset_path, args.split)

    print("\n" + "=" * 60)
    print("All extraction completed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
