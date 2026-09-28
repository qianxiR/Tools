# -*- coding: utf-8 -*-
"""
连通域分析与几何提取工具
从二值掩码中提取连通域，生成多边形坐标和 OBB 旋转框标注。

入参:
- mask: 二值掩码 (H, W) numpy 数组

方法:
- extract_connected_components: 连通域分析，提取每个区域的 OBB 和多边形
- polygon_to_coords: 多边形近似，返回简化顶点坐标
- compute_obb: 从点集计算 OBB 旋转框并归一化为 YOLO-OBB 格式
- normalize_rotated_rect: OpenCV RotatedRect → YOLO-OBB 归一化
- compute_spatial_location: 基于 bbox 中心计算空间方位

出参:
- 各函数返回值见签名
"""

import math
import cv2
import numpy as np
from typing import List, Tuple, Dict


def extract_connected_components(mask: np.ndarray, min_area: int = 2) -> List[dict]:
    """
    入参:
    - mask (np.ndarray): 二值 mask (0或255)，支持灰度或彩色
    - min_area (int): 最小保留面积（像素数），低于此值的连通域被丢弃，默认 2 以移除 1×1 噪声

    方法:
    - 二值化阈值127 → 8连通标记 → cv2.minAreaRect 提取最小外接旋转矩形
    - 旋转矩形四顶点外接轴向 bbox，保证完整覆盖旋转框，钳制到图像边界内
    - 归一化 cx/cy/w/h 至 [0,1]，angle 统一为 YOLO-OBB 约定（w>=h, angle∈[-90,90)）
    - cv2.approxPolyDP 提取简化多边形顶点（epsilon=0.5%周长）

    出参:
    - return (List[dict]): 每个连通域的信息字典，包含:
        - label_id: 连通域标记 ID
        - area: 前景像素面积
        - crop_bbox: 旋转框外接轴向 bbox (x, y, w, h) 像素坐标
        - rotated_rect: ((cx, cy), (w, h), angle) 像素级旋转框（OpenCV 约定）
        - obb_norm: (cx, cy, w, h, angle) 归一化 YOLO-OBB 格式
        - approx_poly: [[x1,y1], [x2,y2], ...] 多边形近似顶点
        - num_approx_points: 顶点数量
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 8连通标记
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

        # 最小外接旋转矩形：用于 YOLO-OBB 标注
        rotated_rect = cv2.minAreaRect(points)

        # 旋转矩形四顶点 → 外接轴向 bbox，保证完整覆盖旋转框
        box_points = cv2.boxPoints(rotated_rect)
        xs, ys = box_points[:, 0], box_points[:, 1]
        x_min, y_min = max(0, int(xs.min())), max(0, int(ys.min()))
        x_max, y_max = min(img_w, int(xs.max())), min(img_h, int(ys.max()))
        crop_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        obb_norm = normalize_rotated_rect(rotated_rect, img_w, img_h)

        # 多边形近似顶点
        approx_poly = polygon_to_coords(instance, epsilon_ratio=0.005)

        components.append({
            "label_id": i,
            "area": area,
            "crop_bbox": crop_bbox,
            "rotated_rect": rotated_rect,
            "obb_norm": obb_norm,
            "approx_poly": approx_poly,
            "num_approx_points": len(approx_poly),
        })

    # 按旋转矩形中心 x 坐标排序，保持处理顺序稳定
    components.sort(key=lambda c: c["rotated_rect"][0][0])
    return components


def polygon_to_coords(binary_mask: np.ndarray, epsilon_ratio: float = 0.005) -> List[List[int]]:
    """
    入参:
    - binary_mask (np.ndarray): 单连通域二值掩码 (0或255)，uint8
    - epsilon_ratio (float): Douglas-Peucker 简化比率，越小越精细，默认 0.005

    方法:
    - findContours 提取最大外轮廓 → approxPolyDP 简化 → 返回整数顶点坐标
    - 若存在多个轮廓，仅保留面积最大的（对应主连通域）

    出参:
    - return (List[List[int]]): 多边形顶点 [[x1,y1], [x2,y2], ...]，空列表表示无有效轮廓
    """
    if len(binary_mask.shape) == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # 取面积最大的轮廓（对应主连通域）
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = perimeter * epsilon_ratio
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(pt[0][0]), int(pt[0][1])] for pt in approx]


def compute_obb(points: np.ndarray, img_w: int, img_h: int) -> Dict[str, float]:
    """
    入参:
    - points (np.ndarray): 前景像素坐标，shape (N, 1, 2)，cv2.findNonZero 格式
    - img_w (int): 图像宽度（像素）
    - img_h (int): 图像高度（像素）

    方法:
    - cv2.minAreaRect 提取最小外接旋转矩形
    - 调用 normalize_rotated_rect 归一化为 YOLO-OBB 约定 (w>=h, angle∈[-90,90))

    出参:
    - return (dict): {cx, cy, w, h, angle} 归一化 OBB 参数，cx/cy/w/h ∈ [0,1]
    """
    rotated_rect = cv2.minAreaRect(points)
    cx, cy, w, h, angle = normalize_rotated_rect(rotated_rect, img_w, img_h)
    return {"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle}


def normalize_rotated_rect(rect: cv2.RotatedRect, img_w: int, img_h: int) -> Tuple[float, float, float, float, float]:
    """
    入参:
    - rect: cv2.minAreaRect 返回的 ((cx, cy), (w, h), angle)
    - img_w (int): 图像宽度，用于归一化
    - img_h (int): 图像高度，用于归一化

    方法:
    - OpenCV 约定: angle ∈ (-90, 0]，width 可能小于 height
    - YOLO-OBB 约定: w >= h，angle 范围 [-90, 90)
    - 若 rw < rh，交换宽高并 angle += 90 以满足 w >= h
    - cx/cy/w/h 除以图像宽高归一化到 [0,1]

    出参:
    - return (tuple): (cx, cy, w, h, angle) 归一化后的 YOLO-OBB 参数，保留 6 位小数
    """
    (cx, cy), (rw, rh), angle = rect

    # OpenCV angle ∈ (-90, 0]；若 rw < rh，交换并调整角度以保持 w >= h
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0

    cx /= img_w
    cy /= img_h
    rw /= img_w
    rh /= img_h

    return (round(cx, 6), round(cy, 6), round(rw, 6), round(rh, 6), round(angle, 2))


def compute_spatial_location(bbox: Tuple[int, int, int, int], image_width: int, image_height: int) -> str:
    """
    入参:
    - bbox (Tuple[int, int, int, int]): 区域边界框 (x, y, w, h)，绝对像素坐标
    - image_width (int): 原图宽度（像素）
    - image_height (int): 原图高度（像素）

    方法:
    - 计算 bbox 中心点与图像中心点的相对偏移，避免左上角导致方位判断失真
    - y 轴翻转到地理方向坐标系，使"上方区域"稳定映射为 north
    - 8 个等角扇区输出英文方位，中心附近统一记为 center

    出参:
    - return (str): 英文方位，取值 east/northeast/north/northwest/west/southwest/south/southeast/center
    """
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    img_cx = image_width / 2
    img_cy = image_height / 2

    # 使用中心点偏移，避免大框区域的方位判断被裁剪尺寸干扰
    dx = cx - img_cx
    dy = -(cy - img_cy)

    # 中心附近小范围变化统一记为 center，减少边界抖动
    threshold = max(image_width, image_height) * 0.1
    if abs(dx) < threshold and abs(dy) < threshold:
        return "center"

    angle = math.atan2(dy, dx)
    sector = round(angle / (math.pi / 4)) % 8
    directions = ["east", "northeast", "north", "northwest",
                  "west", "southwest", "south", "southeast"]
    return directions[sector]
