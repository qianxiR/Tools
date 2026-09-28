"""
SegEarth-OV3 语义分割推理脚本（多轮迭代优化版）
调用 infer.py 底层模块执行推理 + 后处理 + 可视化

优化策略（来自 Remote SAMsing 论文）:
  1. 多轮迭代推理 — 分割后涂黑残差区域，再跑一轮，提升覆盖率
  2. 自适应阈值衰减 — 每轮阈值 ×0.7，从严格到宽松
  3. Contextual Padding + 高斯融合（大图滑动窗口时生效）

python infer_seg.py --image "resources/image.png" 

"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from infer import (
    DEFAULT_CLASSES,
    load_model, load_classes, multipass_inference, postprocess, build_palette,
)


def main():
    parser = argparse.ArgumentParser(description="SegEarth-OV3 语义分割推理（多轮迭代）")
    parser.add_argument("--image", type=str, default="resources/image.png")
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--output", type=str, default="seg_pred.png")
    parser.add_argument("--ckpt", type=str, default="sam3/sam3.pt")
    parser.add_argument("--bpe", type=str, default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--max_passes", type=int, default=10,
                        help="多轮迭代最大轮数（默认3）")
    parser.add_argument("--coverage_target", type=float, default=0.95,
                        help="目标覆盖率，达到后停止迭代（默认0.95）")
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--max_hole", type=int, default=256)
    parser.add_argument("--smooth_kernel", type=int, default=3,
                        help="形态学平滑核半径（0=关闭，默认3）")
    parser.add_argument("--target_class", type=str, default=None,
                        help="指定输出的类别名(逗号分隔)，如 'building,road'")
    args = parser.parse_args()

    # 类别：跳过 class 0 (background)，不发文本提示
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

    # 模型
    print(f"[1/4] 加载模型: {args.ckpt}")
    processor, device = load_model(args.ckpt, args.bpe, args.conf)
    query_idx_tensor = query_idx_tensor.to(device)
    print(f"      类别数: {num_cls}, 查询数: {num_queries}")

    # 图片
    print(f"[2/4] 读取图片: {args.image}")
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    print(f"      尺寸: {w}x{h}")

    # 多轮迭代推理
    print(f"[3/4] 多轮迭代推理 (max={args.max_passes}, target={args.coverage_target:.0%})...")
    seg_pred_np = multipass_inference(
        processor, query_words, query_idx_tensor, num_cls, num_queries,
        device, image, prob=args.prob,
        max_passes=args.max_passes, coverage_target=args.coverage_target,
    )

    # 后处理
    seg_pred_np = postprocess(seg_pred_np, num_cls, args.min_area, args.max_hole, args.smooth_kernel)

    # 构建类别名映射
    cls_name_map = {}
    for idx, line in enumerate(name_list):
        synonyms = [s.strip().lower() for s in str(line).strip().split(",")]
        cls_name_map[idx] = synonyms
    cls_display = {idx: syns[0] for idx, syns in cls_name_map.items()}

    # 颜色映射（使用公共 build_palette，背景设为白色）
    palette = build_palette(num_cls, cls_name_map)
    palette[0] = [255, 255, 255]  # background → 白色
    color_mask = palette[seg_pred_np]

    # 保存
    img_stem = Path(args.image).stem
    out_dir = Path("output_seg") / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[4/4] 保存结果:")

    # ① 多类掩码（彩色分割图，单独保存）
    seg_color_path = str(out_dir / "seg_color_mask.png")
    Image.fromarray(color_mask).save(seg_color_path)
    print(f"      color_mask: {seg_color_path}")

    # ② 多类分割索引图（每个像素=类别ID，用于后续分析）
    seg_label_path = str(out_dir / "seg_label.png")
    Image.fromarray(seg_pred_np.astype(np.uint8)).save(seg_label_path)
    print(f"      label_map: {seg_label_path}")

    # ③ 主结果图：原图 + Overlay
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(image); ax[0].axis("off"); ax[0].set_title("Input")
    ax[1].imshow(image); ax[1].imshow(color_mask, alpha=0.5); ax[1].axis("off"); ax[1].set_title("Overlay (Multipass)")
    # 图例：只显示结果中实际出现的类别（同义词只显示首个名称）
    present_cls = sorted(set(seg_pred_np[seg_pred_np > 0].tolist()))
    legend_items = [Patch(facecolor=palette[i] / 255, label=cls_display.get(i, str(i)))
                    for i in present_cls]
    if legend_items:
        ax[1].legend(handles=legend_items, loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    result_path = str(out_dir / "seg_pred.png")
    plt.savefig(result_path, bbox_inches="tight", dpi=150)
    print(f"      result: {result_path}")

    # 输出各类别掩码（每个同义词类只保存一个掩码，用首个同义词命名；只保存结果中实际存在的类别）
    if args.target_class:
        target_cls_ids = []
        for name in [n.strip() for n in args.target_class.split(",") if n.strip()]:
            for idx, synonyms in cls_name_map.items():
                if name.lower() in synonyms:
                    target_cls_ids.append(idx)
                    break
    else:
        # 只保存结果中实际出现的类别
        target_cls_ids = sorted(set(seg_pred_np[seg_pred_np > 0].tolist()))
    for cls_id in target_cls_ids:
        display_name = cls_display.get(cls_id, str(cls_id))
        t_mask = (seg_pred_np == cls_id).astype(np.uint8) * 255
        mask_path = str(out_dir / f"cls_{display_name}_mask.png")
        Image.fromarray(t_mask).save(mask_path)
        print(f"      class '{display_name}'(id={cls_id}): {mask_path}")

    print("完成！")


if __name__ == "__main__":
    main()
