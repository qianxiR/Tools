#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变化检测后处理公共工具模块

提供 Unicode 路径读写、类别映射、调色板等公共能力
供 cd1~cd4 及 cd_pipeline 调用
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ─── 从项目根目录的 infer.py 导入统一类别 & 颜色定义 ───
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from infer import DEFAULT_CLASSES, NAME_COLOR, DEFAULT_PALETTE  # noqa: E402


# ═══════════════════════════════════════════════════════════════
#  图像 I/O（Unicode 路径兼容）
# ═══════════════════════════════════════════════════════════════

def imread_unicode(path: str) -> Optional[np.ndarray]:
    """支持中文路径的图片读取，对 GeoTIFF/TIFF 使用 rasterio，其他格式使用 OpenCV"""
    ext = Path(path).suffix.lower()
    # GeoTIFF / TIFF: 必须使用 rasterio（避免 OpenCV GeoTIFF 警告）
    if ext in ('.tif', '.tiff'):
        import rasterio
        with rasterio.open(path) as src:
            # rasterio 读取为 (bands, H, W)，转为 OpenCV 约定 (H, W, C)
            data = src.read()  # (bands, H, W) uint8/uint16/float32
            if data.shape[0] == 1:
                # 灰度 → (H, W)
                return data[0]
            elif data.shape[0] >= 3:
                # RGB → BGR (OpenCV 约定)
                img = np.transpose(data[:3], (1, 2, 0))
                return img[:, :, ::-1].copy()
            else:
                return np.transpose(data, (1, 2, 0))
    # PNG / JPG 等其他格式: 使用 OpenCV
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """支持中文路径的图片写入"""
    try:
        ext = Path(path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            encoded = cv2.imencode('.jpg', img)[1]
        else:
            encoded = cv2.imencode('.png', img)[1]
        encoded.tofile(path)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
#  类别与调色板
# ═══════════════════════════════════════════════════════════════

def parse_classes(classes_input) -> Tuple[Dict[int, List[str]], int]:
    """
    解析类别定义，返回 cls_name_map 和 num_cls

    入参:
    - classes_input: 类别文件路径(str) 或 类别列表(list)

    出参:
    - cls_name_map: {cls_id: [synonym1, synonym2, ...]}
    - num_cls: 类别总数（含 background）
    """
    if isinstance(classes_input, str) and os.path.isfile(classes_input):
        with open(classes_input, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
    elif isinstance(classes_input, list):
        lines = classes_input
    else:
        lines = DEFAULT_CLASSES

    cls_name_map = {}
    for idx, line in enumerate(lines):
        synonyms = [s.strip().lower() for s in str(line).strip().split(",")]
        cls_name_map[idx] = synonyms

    return cls_name_map, len(lines)


def build_palette(num_cls: int, cls_name_map: Dict[int, List[str]]) -> np.ndarray:
    """
    构建 (num_cls, 3) uint8 调色板
    """
    palette = np.zeros((num_cls, 3), dtype=np.uint8)
    default_ci = 0
    for cls_id in range(num_cls):
        if cls_id == 0:
            continue
        synonyms = cls_name_map.get(cls_id, [str(cls_id)])
        matched = False
        for syn in synonyms:
            if syn in NAME_COLOR:
                palette[cls_id] = NAME_COLOR[syn]
                matched = True
                break
        if not matched:
            palette[cls_id] = DEFAULT_PALETTE[default_ci % len(DEFAULT_PALETTE)]
            default_ci += 1
    return palette


def rgb_to_label(rgb_mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    RGB 彩色掩码 → class index 标签图

    对每个像素找 palette 中最近的颜色，返回对应 index
    """
    h, w = rgb_mask.shape[:2]
    pixels = rgb_mask.reshape(-1, 3).astype(np.float32)
    palette_f = palette.astype(np.float32)
    # 计算距离
    dists = np.sum((pixels[:, None, :] - palette_f[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1).astype(np.uint8)
    return labels.reshape(h, w)


def get_cls_display_name(cls_name_map: Dict[int, List[str]]) -> Dict[int, str]:
    """返回 {cls_id: 显示名(第一个同义词)}"""
    return {idx: syns[0] for idx, syns in cls_name_map.items()}


# ═══════════════════════════════════════════════════════════════
#  instances.json 读写
# ═══════════════════════════════════════════════════════════════

def load_instances(json_path: str) -> Optional[Dict]:
    """读取 instances.json"""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def save_instances(data: Dict, json_path: str):
    """保存 instances.json"""
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_process_log(log_path: str, step: str, message: str, data: dict = None):
    """追加过程日志"""
    from datetime import datetime
    log = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "message": message,
    }
    if data:
        log["data"] = data

    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append(log)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
