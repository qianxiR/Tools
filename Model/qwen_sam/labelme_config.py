"""Load QwenSam labels, paths, training and inference settings from config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml
from PIL import Image

# QWENSAM_CONFIG 环境变量可覆盖默认配置文件，支持多套数据集配置并行（如 0822 全景）。
CONFIG_PATH = Path(os.environ.get("QWENSAM_CONFIG", str(Path(__file__).with_name("config.yaml"))))


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """
    入参:
    - path: YAML 配置文件路径。

    方法:
    - 以 UTF-8 安全加载 YAML，并校验必需顶层 section。

    出参:
    - dict: 完整配置字典。
    """
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"classes", "model", "architecture", "dataset", "training", "inference"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    return config


CONFIG = load_config()

# 兼容导出：业务代码继续使用原常量名，但所有值均来自 config.yaml。
LABELME_CLASS_NAMES = tuple(CONFIG["classes"]["names"])
LABELME_NUM_CLASSES = len(LABELME_CLASS_NAMES)
LABELME_STRING_TO_CLASS_ID = {str(key): int(value) for key, value in CONFIG["classes"]["label_mapping"].items()}
LABELME_PALETTE_RGB = {int(key): tuple(value) for key, value in CONFIG["classes"]["palette_rgb"].items()}

MODEL_CONFIG = dict(CONFIG["model"])
MODEL_CONFIG.update({
    "num_classes": LABELME_NUM_CLASSES,
    "edge_channels": int(CONFIG["architecture"]["edge_channels"]),
    "spatial_gate": bool(CONFIG["architecture"]["spatial_gate"]),
})
DATASET_CONFIG = dict(CONFIG["dataset"])
DATASET_CONFIG["splits"] = {name: tuple(ids) for name, ids in DATASET_CONFIG["splits"].items()}

TRAIN_CONFIG = {
    "out": CONFIG["training"]["output_dir"],
    "steps": int(CONFIG["training"]["steps"]),
    "batch_size": int(CONFIG["training"]["batch_size"]),
    "acc": int(CONFIG["training"]["gradient_accumulation"]),
    "lr": float(CONFIG["training"]["learning_rate"]),
    "wd": float(CONFIG["training"]["weight_decay"]),
    "val_every": int(CONFIG["training"]["validation_interval"]),
    "log_every": int(CONFIG["training"]["log_interval"]),
    "device": CONFIG["training"]["device"],
    "edge_loss_weight": float(CONFIG["training"]["edge_loss_weight"]),
    "spatial_gate": bool(CONFIG["architecture"]["spatial_gate"]),
    "seed": int(CONFIG["training"]["seed"]),
}
INFERENCE_CONFIG = {
    "run_dir": CONFIG["inference"]["run_dir"],
    "splits": tuple(CONFIG["inference"]["splits"]),
    "checkpoint": CONFIG["inference"]["checkpoint"],
    "device": CONFIG["inference"]["device"],
    "output_dir_name": CONFIG["inference"]["output_dir_name"],
    "smooth_kernel": int(CONFIG["inference"]["smooth_kernel"]),
    "min_area": int(CONFIG["inference"]["postprocess_area_threshold"]),
    "max_hole_area": int(CONFIG["inference"]["postprocess_area_threshold"]),
}


def validate_class_map(class_map: np.ndarray) -> np.ndarray:
    """入参: HxW 整数类别图；方法: 校验维度和值域；出参: uint8 连续类别图。"""
    arr = np.asarray(class_map)
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Expected integer HxW class map, got {arr.shape} {arr.dtype}")
    if np.any((arr < 0) | (arr >= LABELME_NUM_CLASSES)):
        raise ValueError(f"Unknown class IDs: {np.unique(arr).tolist()}")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def class_map_to_rgb(class_map: np.ndarray) -> np.ndarray:
    """入参: HxW 类别图；方法: 调色板查表；出参: HxWx3 RGB 图。"""
    ids = validate_class_map(class_map)
    palette = np.asarray([LABELME_PALETTE_RGB[index] for index in range(LABELME_NUM_CLASSES)], dtype=np.uint8)
    return palette[ids]


def save_class_map(class_map: np.ndarray, path: str | Path) -> None:
    """入参: 类别图和 PNG 路径；方法: 单通道保存；出参: None。"""
    Image.fromarray(validate_class_map(class_map), mode="L").save(path)


def load_class_map(path: str | Path) -> np.ndarray:
    """入参: 单通道类别 PNG；方法: 原样读取并校验；出参: uint8 类别图。"""
    with Image.open(path) as image:
        return validate_class_map(np.asarray(image))


def class_percentages(class_map: np.ndarray) -> Mapping[str, float]:
    """入参: 类别图；方法: 统计各类比例；出参: 类别名称到百分比映射。"""
    ids = validate_class_map(class_map)
    return {name: float((ids == index).mean() * 100) for index, name in enumerate(LABELME_CLASS_NAMES)}
