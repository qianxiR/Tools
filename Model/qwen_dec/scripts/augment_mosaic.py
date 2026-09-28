"""
Mosaic 4 图拼接数据增强（YOLO 格式）。

做什么:
    从训练集随机选 4 张图拼成 1 张 2×2 网格图，YOLO 标注同步变换。
    输出到独立目录（不污染原始数据），生成增强后的 split 清单。

为什么:
    训练集 42% 是单目标图，模型学不到多目标检测。
    Mosaic 把 4 张图拼成 1 张，直接制造多目标场景（平均 8 框/图），
    是 YOLOv4/v5 中效果最显著的检测增强（COCO +5% mAP）。

输出:
    数据_筛选500_aug/images/<id>_mosaic.<ext>   增强图像
    数据_筛选500_aug/labels/<id>_mosaic.txt      增强标注（YOLO cxcywh）
    数据_筛选500_aug/split/train_mosaic.txt       增强清单（相对路径）

用法:
    python scripts/augment_mosaic.py                    # 默认生成 342 张（1:1）
    python scripts/augment_mosaic.py --num 684          # 生成 684 张（2:1）
    python scripts/augment_mosaic.py --num 342 --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
DATA_ROOT = Path(r"F:/管网/数据_筛选500")
IMG_DIR = DATA_ROOT / "images"
LABEL_DIR = DATA_ROOT / "labels"
SPLIT_TRAIN = DATA_ROOT / "split" / "train.txt"

OUT_ROOT = DATA_ROOT.parent / "数据_筛选500_aug"
OUT_IMG_DIR = OUT_ROOT / "images"
OUT_LABEL_DIR = OUT_ROOT / "labels"
OUT_SPLIT = OUT_ROOT / "split" / "train_mosaic.txt"

MOSAIC_SIZE = 640  # 输出图像边长（正方形），YOLO Mosaic 标准


def load_train_list() -> List[str]:
    """入参: 无; 方法: 读 split/train.txt; 出参: 相对路径列表（如 images/xxx.png）。"""
    lines = SPLIT_TRAIN.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def load_yolo_label(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """入参: YOLO 标注文件; 方法: 解析 cls xc yc w h; 出参: [(cls, xc, yc, w, h), ...]。"""
    targets = []
    if not txt_path.exists():
        return targets
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            targets.append((cls, xc, yc, w, h))
        except ValueError:
            continue
    return targets


def make_one_mosaic(
    four_images: List[Tuple[str, Image.Image, List]],
    idx: int,
    out_img_dir: Path,
    out_label_dir: Path,
) -> str:
    """入参: 4 个 (rel_path, PIL, targets) 元组 + 序号 + 输出目录。
    方法: 2×2 拼接 4 张图到 MOSAIC_SIZE×MOSAIC_SIZE，YOLO 坐标同步变换。
    出参: 增强图像的相对路径（images/xxx_mosaic.ext）。"""
    cell = MOSAIC_SIZE // 2  # 每个格子 320×320
    canvas = Image.new("RGB", (MOSAIC_SIZE, MOSAIC_SIZE), (0, 0, 0))

    # 4 个格子的左上角偏移
    offsets = [(0, 0), (cell, 0), (0, cell), (cell, cell)]

    all_targets = []
    ext = ".png"

    for i, (rel_path, pil, targets) in enumerate(four_images):
        ox, oy = offsets[i]
        # resize 到 cell×cell（不保宽高比，Mosaic 标准做法）
        pil_resized = pil.convert("RGB").resize((cell, cell), Image.LANCZOS)
        canvas.paste(pil_resized, (ox, oy))
        ext = Path(rel_path).suffix  # 取最后一个的扩展名

        # 变换 YOLO 坐标：原归一化 cxcywh → 拼接后归一化 cxcywh
        for cls, xc, yc, w, h in targets:
            # 原图中的归一化坐标 → cell 内的归一化坐标
            # 拼接后整图 MOSAIC_SIZE，cell 占一半
            new_xc = (ox / MOSAIC_SIZE) + (xc * cell / MOSAIC_SIZE)
            new_yc = (oy / MOSAIC_SIZE) + (yc * cell / MOSAIC_SIZE)
            new_w = w * cell / MOSAIC_SIZE
            new_h = h * cell / MOSAIC_SIZE
            # 裁剪到 [0,1]
            new_xc = max(0.0, min(1.0, new_xc))
            new_yc = max(0.0, min(1.0, new_yc))
            new_w = max(0.001, min(1.0, new_w))
            new_h = max(0.001, min(1.0, new_h))
            all_targets.append((cls, new_xc, new_yc, new_w, new_h))

    # 生成唯一 ID
    first_id = Path(four_images[0][0]).stem
    out_id = f"{first_id}_mosaic{idx}"
    out_rel = f"images/{out_id}{ext}"
    out_img_path = out_img_dir / f"{out_id}{ext}"
    out_label_path = out_label_dir / f"{out_id}.txt"

    # 保存图像
    canvas.save(out_img_path, quality=95)

    # 保存 YOLO 标注
    lines = []
    for cls, xc, yc, w, h in all_targets:
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    out_label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out_rel


def main():
    """入参: --num（生成数量）--seed（随机种子）; 方法: 随机选 4 图拼接; 出参: 无。"""
    parser = argparse.ArgumentParser(description="Mosaic 4-image augmentation for YOLO detection.")
    parser.add_argument("--num", type=int, default=342, help="生成增强样本数量（默认 342 = 1:1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_list = load_train_list()
    print(f"[mosaic] 原始训练集: {len(train_list)} 张")

    # 创建输出目录
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)

    # 预加载所有训练图（避免重复 IO）
    print("[mosaic] 预加载训练图...")
    cache = {}  # rel_path -> (PIL, targets)
    for rel in train_list:
        img_path = DATA_ROOT / rel
        stem = img_path.stem
        label_path = LABEL_DIR / f"{stem}.txt"
        if not img_path.exists():
            continue
        pil = Image.open(img_path)
        targets = load_yolo_label(label_path)
        cache[rel] = (pil, targets)
    valid = list(cache.keys())
    print(f"[mosaic] 有效图: {len(valid)} 张")

    # 生成 Mosaic 样本
    print(f"[mosaic] 生成 {args.num} 张 Mosaic 样本...")
    out_lines = []
    total_boxes = 0
    for i in range(args.num):
        # 随机选 4 张不同的图
        chosen = random.sample(valid, min(4, len(valid)))
        four = [(rel, cache[rel][0], cache[rel][1]) for rel in chosen]
        out_rel = make_one_mosaic(four, i, OUT_IMG_DIR, OUT_LABEL_DIR)
        out_lines.append(out_rel)
        total_boxes += len(four[0][2]) + len(four[1][2]) + len(four[2][2]) + len(four[3][2])

    # 写 split 清单
    OUT_SPLIT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # 统计
    print(f"\n[mosaic] DONE")
    print(f"  生成样本: {len(out_lines)} 张")
    print(f"  总框数: {total_boxes} (平均 {total_boxes/len(out_lines):.1f} 框/图)")
    print(f"  输出目录: {OUT_ROOT}")
    print(f"  split 清单: {OUT_SPLIT}")
    print(f"\n  增强前: {len(valid)} 张, 平均 2.0 框/图")
    print(f"  增强后: {len(valid) + len(out_lines)} 张, 平均 {total_boxes/len(out_lines):.1f} 框/图")
    print(f"\n  下一步: 合并原始+增强后跑转换器生成 JSONL")


if __name__ == "__main__":
    main()
