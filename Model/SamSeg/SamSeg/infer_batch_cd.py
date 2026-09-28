r"""
SegEarth-OV3 批量推理脚本 — 语义分割 + 变化检测（多轮迭代优化版）
对 WHU-CD_20 等数据集批量输出 t1_mask / t2_mask / change_mask

优化策略（与 infer_cd.py 同步）:
  1. 多轮迭代推理 — 每个时相独立多轮，涂黑残差区域再分割
  2. 自适应阈值衰减
  3. Contextual Padding + 高斯融合（大图滑动窗口时生效）
  4. background 不发文本提示，同义词增强匹配
  5. 统一颜色映射（build_palette）

python infer_batch_cd.py --data_root "E:\xzkjxm\dataes\CD-CD_20" --split val
python infer_batch_cd.py --data_root "E:\xzkjxm\dataes\CD-CD_20" --split all
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
    compute_instance_change_map,
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


def process_one_pair(processor, device, query_words, query_idx_tensor,
                     num_cls, num_queries, img_t1, img_t2, prob,
                     max_passes, coverage_target, min_area, max_hole,
                     smooth_kernel=3, iou_threshold=0.1):
    """
    对一对 (t1, t2) 图像执行多轮迭代推理 + 实例级变化检测
    返回 t1_mask, t2_mask, change_mask (numpy, H, W)
    """
    w1, h1 = img_t1.size
    w2, h2 = img_t2.size
    target_w, target_h = max(w1, w2), max(h1, h2)
    if (w1, h1) != (target_w, target_h):
        img_t1 = img_t1.resize((target_w, target_h), Image.BILINEAR)
    if (w2, h2) != (target_w, target_h):
        img_t2 = img_t2.resize((target_w, target_h), Image.BILINEAR)

    # --- T1 多轮迭代推理 ---
    seg_t1 = multipass_inference(
        processor, query_words, query_idx_tensor, num_cls, num_queries,
        device, img_t1, prob=prob,
        max_passes=max_passes, coverage_target=coverage_target,
    )

    # --- T2 多轮迭代推理 ---
    seg_t2 = multipass_inference(
        processor, query_words, query_idx_tensor, num_cls, num_queries,
        device, img_t2, prob=prob,
        max_passes=max_passes, coverage_target=coverage_target,
    )

    # --- 后处理（语义分割） ---
    t1_mask = postprocess(seg_t1, num_cls, min_area, max_hole, smooth_kernel)
    t2_mask = postprocess(seg_t2, num_cls, min_area, max_hole, smooth_kernel)

    # --- 实例级变化检测（IoU 匹配） ---
    change_pred = compute_instance_change_map(
        t1_mask, t2_mask, num_cls,
        iou_threshold=iou_threshold,
        min_area=max(min_area, 4),
    )
    change_mask = postprocess(change_pred, num_cls, min_area, max_hole, smooth_kernel)

    return t1_mask, t2_mask, change_mask, img_t1, img_t2


def main():
    parser = argparse.ArgumentParser(description="SegEarth-OV3 批量推理（多轮迭代）")
    parser.add_argument("--data_root", type=str, default="E:/xzkjxm/dataes/WHU-CD_20",
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
                        help="多轮迭代最大轮数（默认10）")
    parser.add_argument("--coverage_target", type=float, default=0.95,
                        help="目标覆盖率（默认0.95）")
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--max_hole", type=int, default=256)
    parser.add_argument("--smooth_kernel", type=int, default=3,
                        help="形态学平滑核半径（0=关闭，默认3）")
    parser.add_argument("--iou_threshold", type=float, default=0.0,
                        help="掩码 IoU 阈值，默认 0（有重叠即匹配）")
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
    # 变化检测用同样的调色板
    chg_palette = seg_palette.copy()

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

    # 统计总数
    img_exts = {"*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"}
    total_pairs = 0
    for sp in splits:
        t1_dir = Path(args.data_root) / sp / "t1"
        if t1_dir.exists():
            for ext in img_exts:
                total_pairs += len(list(t1_dir.glob(ext)))
    print(f"      待处理: {total_pairs} 对图像 (splits: {', '.join(splits)})")

    # 批量推理
    print(f"[2/3] 批量推理中...")
    done = 0
    t_start = time.time()
    for sp in splits:
        t1_dir = Path(args.data_root) / sp / "t1"
        t2_dir = Path(args.data_root) / sp / "t2"
        if not t1_dir.exists():
            print(f"      跳过 {sp}: 目录不存在")
            continue

        # 创建输出目录
        base = Path(args.data_root) / sp
        out_t1 = base / "t1_mask"
        out_t2 = base / "t2_mask"
        out_cd = base / "change_mask"
        out_ov = base / "overlay"
        for d in [out_t1, out_t2, out_cd, out_ov]:
            d.mkdir(parents=True, exist_ok=True)

        # 按文件名排序（支持 png/tif/jpg 等多种格式）
        t1_files = []
        for ext in img_exts:
            t1_files.extend(t1_dir.glob(ext))
        t1_files = sorted(t1_files, key=lambda p: p.name)

        for t1_path in t1_files:
            fname = t1_path.name
            stem = t1_path.stem
            # 在 t2 目录中查找同名文件（兼容不同扩展名）
            t2_path = None
            for ext in img_exts:
                candidate = t2_dir / f"{stem}{ext[1:]}"  # ext 去掉 * 前缀
                if candidate.exists():
                    t2_path = candidate
                    break
            if t2_path is None:
                print(f"      警告: {stem} 缺少 t2，跳过")
                continue

            done += 1
            Image.MAX_IMAGE_PIXELS = None
            img_t1 = Image.open(t1_path).convert("RGB")
            img_t2 = Image.open(t2_path).convert("RGB")

            t1_mask, t2_mask, change_mask, img_t1_r, img_t2_r = process_one_pair(
                processor, device, query_words, query_idx_tensor,
                num_cls, num_queries, img_t1, img_t2, args.prob,
                args.max_passes, args.coverage_target,
                args.min_area, args.max_hole, args.smooth_kernel,
                args.iou_threshold)

            # 保存 RGB 彩色渲染（统一输出 .png 后缀）
            out_name = stem + ".png"
            Image.fromarray(mask_to_rgb(t1_mask, seg_palette)).save(str(out_t1 / out_name))
            Image.fromarray(mask_to_rgb(t2_mask, seg_palette)).save(str(out_t2 / out_name))
            Image.fromarray(mask_to_rgb(change_mask, chg_palette)).save(str(out_cd / out_name))

            # 保存叠加图: t1_seg | t2_seg | change_overlay
            ov_t1 = mask_to_overlay(img_t1_r, t1_mask, seg_palette)
            ov_t2 = mask_to_overlay(img_t2_r, t2_mask, seg_palette)
            ov_cd = mask_to_overlay(img_t2_r, change_mask, chg_palette, alpha=0.4)
            composite = np.concatenate([ov_t1, ov_t2, ov_cd], axis=1)
            Image.fromarray(composite).save(str(out_ov / out_name))

            elapsed = time.time() - t_start
            avg = elapsed / done
            eta = avg * (total_pairs - done)
            print(f"\r      [{done}/{total_pairs}] {sp}/{fname}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", end="", flush=True)
    print()

    # 统计
    print(f"[3/3] 完成! 共处理 {done} 对图像")
    print(f"      输出目录: {Path(args.data_root).resolve()}")
    for sp in splits:
        for sub in ["t1_mask", "t2_mask", "change_mask", "overlay"]:
            d = Path(args.data_root) / sp / sub
            if d.exists():
                n = len(list(d.glob("*.png")))
                print(f"      {sp}/{sub}: {n} 张")


if __name__ == "__main__":
    main()
