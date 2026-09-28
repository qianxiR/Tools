# -*- coding: utf-8 -*-
from __future__ import annotations

"""
检测结果演示可视化（原 viis.py 迁移修正版）。

做什么: 读取一张管网 CCTV 图像，把 0-1000 归一化 bbox_2d 检测结果按类别配色画框并标注
       类别代码，窗口展示的同时打印逐类统计。
为什么: 快速目检一组检测结果（模型输出或人工标注）的空间位置与类别是否合理；
       正式的逐图对比可视化走 visualize_bbox_prediction.py / evaluate_detection.py。
迁移修正: 原版 (F:/管网/数据_筛选500/viis.py) 引用未定义变量 image_clue 且 color_map
       存在重复 ZW 键；本版改为命令行传入图像路径，检测结果默认内置演示数据，
       亦可用 --detections 传入 <obj>[...]</obj> 文本或 JSON 数组文件，画框/统计逻辑不变。
用法: python scripts/visualize_detection_demo.py --image <图像路径> [--detections <JSON 文件>] [--save <输出路径>]
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 内置演示数据（0-1000 归一化 xyxy + 类别代码），与原版一致
DEMO_DETECTIONS = [
    {"bbox_2d": [353, 103, 747, 353], "label": "TL"},
    {"bbox_2d": [263, 303, 847, 995], "label": "CJ"},
    {"bbox_2d": [436, 313, 511, 381], "label": "ZW"},
    {"bbox_2d": [692, 486, 719, 511], "label": "ZW"},
    {"bbox_2d": [533, 513, 603, 539], "label": "ZW"},
]

# 类别配色（BGR 供 cv2 使用）；12 类全覆盖，与 prompt JSON 的 label_codes 一致
COLOR_MAP = {
    "TL": (255, 0, 0),      # 接口材料脱落
    "CJ": (0, 255, 0),      # 沉积
    "ZW": (0, 0, 255),      # 障碍物
    "BX": (255, 255, 0),    # 变形
    "CR": (255, 0, 255),    # 管壁裂纹
    "FS": (0, 255, 255),    # 腐蚀
    "JG": (128, 0, 128),    # 结垢
    "PL": (255, 128, 0),    # 破裂
    "SG": (0, 128, 255),    # 树根
    "TJ": (128, 255, 0),    # 脱节
    "CK": (255, 0, 128),    # 错口
    "AJ": (128, 128, 255),  # 支管暗接
}

LABEL_NAMES = {
    "TL": "接口材料脱落", "CJ": "管底沉积物", "ZW": "异物阻塞",
}


def parse_args() -> argparse.Namespace:
    """入参: 无。方法: 解析 --image/--detections/--save 命令行参数。出参: argparse.Namespace。"""
    parser = argparse.ArgumentParser(description="Visualize 0-1000 normalized bbox detections on one pipe CCTV image.")
    parser.add_argument("--image", required=True, help="输入图像路径（jpg/png/tif 等）")
    parser.add_argument(
        "--detections", default=None,
        help="可选检测结果 JSON 文件（<obj>[...]</obj> 文本或 [{\"bbox_2d\":[...],\"label\":\"..\"}] 数组）；省略时用内置演示数据",
    )
    parser.add_argument("--save", default=None, help="可选输出保存路径（省略时仅窗口展示）")
    return parser.parse_args()


def load_detections(path: str | None) -> list[dict]:
    """入参: detections JSON 文件路径或 None。
    方法: None 时返回内置演示数据；否则读文件文本，剥离 <obj></obj> 包裹后 json.loads，
          仅保留含合法 bbox_2d 的条目（兼容模型原始输出格式）。
    出参: [{"bbox_2d":[x1,y1,x2,y2], "label":<code>}] 列表。"""
    if path is None:
        return DEMO_DETECTIONS
    text = Path(path).read_text(encoding="utf-8").strip()
    text = re.sub(r"</?obj>", "", text)
    data = json.loads(text)
    return [
        {"bbox_2d": item["bbox_2d"], "label": str(item.get("label", "object"))}
        for item in data
        if isinstance(item, dict) and isinstance(item.get("bbox_2d"), list) and len(item["bbox_2d"]) == 4
    ]


def draw_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """入参: BGR 图像数组、0-1000 归一化检测结果列表。
    方法: 逐框把归一化坐标反算到像素（x*w/1000、y*h/1000），按类别颜色画矩形与类别代码标签。
    出参: 标注后的 BGR 图像副本（不修改原图）。"""
    img_copy = image_bgr.copy()
    h, w = img_copy.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox_2d"]
        x1_px, y1_px = int(x1 * w / 1000), int(y1 * h / 1000)
        x2_px, y2_px = int(x2 * w / 1000), int(y2 * h / 1000)
        color = COLOR_MAP.get(det["label"], (255, 255, 255))
        cv2.rectangle(img_copy, (x1_px, y1_px), (x2_px, y2_px), color, 3)
        cv2.putText(img_copy, det["label"], (x1_px, max(0, y1_px - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img_copy


def print_stats(detections: list[dict]) -> None:
    """入参: 检测结果列表。方法: 按类别代码计数并打印统计（含已知类别中文名）。出参: 无。"""
    print("检测到的缺陷统计：")
    print("=" * 50)
    label_counts: dict[str, int] = {}
    for det in detections:
        label_counts[det["label"]] = label_counts.get(det["label"], 0) + 1
    for label, count in label_counts.items():
        print(f"{label} ({LABEL_NAMES.get(label, '未知')}): {count} 个")
    print(f"\n总计: {len(detections)} 个缺陷")


def main() -> None:
    """入参: 无（配置来自命令行）。方法: 读图 → 载入检测 → 画框展示/保存 → 打印统计。出参: None。"""
    args = parse_args()
    detections = load_detections(args.detections)
    image_rgb = np.array(Image.open(args.image).convert("RGB"))
    image_bgr = image_rgb[:, :, ::-1].copy()  # RGB→BGR 供 cv2 绘制
    drawn = draw_detections(image_bgr, detections)
    if args.save:
        Image.fromarray(drawn[:, :, ::-1]).save(args.save)
        print(f"已保存: {args.save}")
    plt.figure(figsize=(20, 12))
    plt.imshow(drawn[:, :, ::-1])
    plt.axis("off")
    plt.title("管道缺陷检测结果", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print_stats(detections)


if __name__ == "__main__":
    main()
