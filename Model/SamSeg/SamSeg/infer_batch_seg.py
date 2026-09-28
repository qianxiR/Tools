r"""
SegEarth-OV3 批量语义分割推理脚本（多轮迭代优化版）
对 WHU_20 等数据集批量输出 seg_mask / overlay

优化策略（与 infer_seg.py 同步）:
  1. 多轮迭代推理 — 分割后涂黑残差区域，再跑一轮，提升覆盖率
  2. 自适应阈值衰减
  3. Contextual Padding + 高斯融合（大图滑动窗口时生效）
  4. background 不发文本提示，同义词增强匹配
  5. 统一颜色映射（build_palette）

python infer_batch_seg.py --data_root "E:\xzkjxm\dataes\CD_20" --split val
python infer_batch_seg.py --data_root "E:\xzkjxm\dataes\CD_20" --split all
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from infer import (
    DEFAULT_CLASSES,
    load_model, load_classes, multipass_inference, postprocess, build_palette,
)


def mask_to_rgb(mask, palette):
    """将灰度掩码 (H,W) 渲染为 RGB 彩色图 (H,W,3)"""
    return palette[mask]


def mask_to_overlay(img, mask, palette, alpha=0.5):
    """原图 + 彩色掩码半透明叠加"""
    img_np = np.array(img).astype(np.float32)
    color_mask = palette[mask].astype(np.float32)
    region = (mask > 0)[:, :, np.newaxis].astype(np.float32)
    out = img_np * (1 - alpha * region) + color_mask * alpha * region
    return out.astype(np.uint8)


def process_one_image(processor, device, query_words, query_idx_tensor,
                      num_cls, num_queries, img, prob,
                      max_passes, coverage_target, min_area, max_hole, smooth_kernel=3):
    """
    对单张图像执行多轮迭代推理 + 后处理
    返回 seg_mask (numpy, H, W)
    """
    # 多轮迭代推理
    seg_pred = multipass_inference(
        processor, query_words, query_idx_tensor, num_cls, num_queries,
        device, img, prob=prob,
        max_passes=max_passes, coverage_target=coverage_target,
    )

    # 后处理
    seg_mask = postprocess(seg_pred, num_cls, min_area, max_hole, smooth_kernel)

    return seg_mask


def main():
    parser = argparse.ArgumentParser(description="SegEarth-OV3 批量语义分割推理（多轮迭代）")
    parser.add_argument("--data_root", type=str, default="E:/xzkjxm/dataes/WHU_20",
                        help="数据集根目录，输出直接写入该目录下")
    parser.add_argument("--split", type=str, default="all",
                        help="处理哪个划分: train / val / test / all")
    parser.add_argument("--classes", type=str, default=None,
                        help="类别文件路径，每行一类，同义词逗号分隔")
    parser.add_argument("--ckpt", type=str, default="sam3/sam3.pt")
    parser.add_argument("--bpe", type=str, default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--max_passes", type=int, default=10,
                        help="多轮迭代最大轮数（默认3）")
    parser.add_argument("--coverage_target", type=float, default=0.95,
                        help="目标覆盖率（默认0.95）")
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--max_hole", type=int, default=256)
    parser.add_argument("--smooth_kernel", type=int, default=3,
                        help="形态学平滑核半径（0=关闭，默认3）")
    args = parser.parse_args()

    # 类别：跳过 class 0 (background)
    if args.classes:
        query_words, query_idx, num_cls, num_queries = load_classes(args.classes)
        name_list = open(args.classes).readlines()
    else:
        query_words, query_idx = [], []
        for idx, line in enumerate(DEFAULT_CLASSES):
            if idx == 0:
                continue  # background 不发提示
            synonyms = [w.strip() for w in line.split(",")]
            query_words.extend(synonyms)
            query_idx.extend([idx] * len(synonyms))
        num_cls = len(DEFAULT_CLASSES)
        num_queries = len(query_words)
        name_list = DEFAULT_CLASSES
    query_idx_tensor = torch.Tensor(query_idx).to(torch.int64)

    # 类别名映射
    cls_name_map = {}
    for idx, line in enumerate(name_list):
        synonyms = [s.strip().lower() for s in str(line).strip().split(",")]
        cls_name_map[idx] = synonyms

    # 构建调色板（使用公共 build_palette，背景设为白色）
    seg_palette = build_palette(num_cls, cls_name_map)
    seg_palette[0] = [255, 255, 255]  # background → 白色

    # 模型
    print(f"[1/3] 加载模型: {args.ckpt}")
    processor, device = load_model(args.ckpt, args.bpe, args.conf)
    query_idx_tensor = query_idx_tensor.to(device)
    print(f"      类别数: {num_cls}, 查询数: {num_queries}, 多轮迭代: max={args.max_passes}")

    # 确定要处理的 split
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    # 统计总数（支持 png / tif / jpg）
    img_exts = {"*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"}
    total_images = 0
    for sp in splits:
        img_dir = Path(args.data_root) / sp / "image"
        if img_dir.exists():
            for ext in img_exts:
                total_images += len(list(img_dir.glob(ext)))
    print(f"      待处理: {total_images} 张图像 (splits: {', '.join(splits)})")

    # 批量推理
    print(f"[2/3] 批量推理中...")
    Image.MAX_IMAGE_PIXELS = None
    done = 0
    t_start = time.time()
    for sp in splits:
        img_dir = Path(args.data_root) / sp / "image"
        if not img_dir.exists():
            print(f"      跳过 {sp}: 目录不存在")
            continue

        # 创建输出目录
        base = Path(args.data_root) / sp
        out_mask = base / "seg_mask"
        out_ov = base / "overlay"
        for d in [out_mask, out_ov]:
            d.mkdir(parents=True, exist_ok=True)

        # 收集所有图片文件并排序
        all_files = []
        for ext in img_exts:
            all_files.extend(img_dir.glob(ext))
        all_files = sorted(all_files, key=lambda p: p.name)

        for img_path in all_files:
            fname = img_path.name
            done += 1

            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            seg_mask = process_one_image(
                processor, device, query_words, query_idx_tensor,
                num_cls, num_queries, image, args.prob,
                args.max_passes, args.coverage_target,
                args.min_area, args.max_hole, args.smooth_kernel)

            # 保存为 PNG（统一输出 .png 后缀）
            out_name = Path(fname).stem + ".png"
            Image.fromarray(mask_to_rgb(seg_mask, seg_palette)).save(str(out_mask / out_name))

            # 保存叠加图
            overlay = mask_to_overlay(image, seg_mask, seg_palette)
            Image.fromarray(overlay).save(str(out_ov / out_name))

            elapsed = time.time() - t_start
            avg = elapsed / done
            eta = avg * (total_images - done)
            print(f"\r      [{done}/{total_images}] {sp}/{fname}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", end="", flush=True)
    print()

    # 统计
    print(f"[3/3] 完成! 共处理 {done} 张图像")
    print(f"      输出目录: {Path(args.data_root).resolve()}")
    for sp in splits:
        for sub in ["seg_mask", "overlay"]:
            d = Path(args.data_root) / sp / sub
            if d.exists():
                n = len(list(d.glob("*.png")))
                print(f"      {sp}/{sub}: {n} 张")


if __name__ == "__main__":
    main()
