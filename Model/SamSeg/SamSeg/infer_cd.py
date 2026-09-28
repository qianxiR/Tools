"""
SegEarth-OV3 变化检测推理脚本（多轮迭代优化版）
调用 infer.py 底层模块执行双时相推理 + 变化检测 + 后处理 + 可视化

优化策略（来自 Remote SAMsing 论文）:
  1. 多轮迭代推理 — 每个时相独立多轮，涂黑残差区域再分割
  2. 自适应阈值衰减
  3. Contextual Padding + 高斯融合（大图滑动窗口时生效）

python infer_cd.py --t1 "resources/1.png" --t2 "resources/2.png" 

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
    compute_instance_change_map,
)


def main():
    parser = argparse.ArgumentParser(description="SegEarth-OV3 变化检测推理（多轮迭代）")
    parser.add_argument("--t1", type=str, default="resources/1.png")
    parser.add_argument("--t2", type=str, default="resources/2.png")
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
    parser.add_argument("--iou_threshold", type=float, default=0.0,
                        help="掩码 IoU 阈值，默认 0（有重叠即匹配）")
    parser.add_argument("--target_class", type=str, default=None,
                        help="指定输出的类别名(逗号分隔)")
    args = parser.parse_args()

    # 类别：跳过 class 0 (background)
    if args.classes:
        query_words, query_idx, num_cls, num_queries = load_classes(args.classes)
        name_list = open(args.classes).readlines()
    else:
        query_words, query_idx = [], []
        for idx, line in enumerate(DEFAULT_CLASSES):
            if idx == 0:
                continue
            synonyms = [w.strip() for w in line.split(",")]
            query_words.extend(synonyms)
            query_idx.extend([idx] * len(synonyms))
        num_cls = len(DEFAULT_CLASSES)
        num_queries = len(query_words)
        name_list = DEFAULT_CLASSES
    query_idx_tensor = torch.Tensor(query_idx).to(torch.int64)

    print(f"[1/6] 加载模型: {args.ckpt}")
    processor, device = load_model(args.ckpt, args.bpe, args.conf)
    query_idx_tensor = query_idx_tensor.to(device)

    img_t1 = Image.open(args.t1).convert("RGB")
    img_t2 = Image.open(args.t2).convert("RGB")
    w1, h1 = img_t1.size
    w2, h2 = img_t2.size
    target_w, target_h = max(w1, w2), max(h1, h2)
    if (w1, h1) != (w2, h2):
        print(f"      T1={w1}x{h1}, T2={w2}x{h2}, resize到 {target_w}x{target_h}")
        if (w1, h1) != (target_w, target_h):
            img_t1 = img_t1.resize((target_w, target_h), Image.BILINEAR)
        if (w2, h2) != (target_w, target_h):
            img_t2 = img_t2.resize((target_w, target_h), Image.BILINEAR)
    print(f"      尺寸: {target_w}x{target_h}, 类别: {num_cls}")

    # T1 多轮迭代推理
    print(f"[2/5] T1 多轮迭代推理 (max={args.max_passes})...")
    seg_t1 = multipass_inference(
        processor, query_words, query_idx_tensor, num_cls, num_queries,
        device, img_t1, prob=args.prob,
        max_passes=args.max_passes, coverage_target=args.coverage_target,
    )
    t1_mask = postprocess(seg_t1, num_cls, args.min_area, args.max_hole, args.smooth_kernel)

    # T2 多轮迭代推理
    print(f"[3/5] T2 多轮迭代推理 (max={args.max_passes})...")
    seg_t2 = multipass_inference(
        processor, query_words, query_idx_tensor, num_cls, num_queries,
        device, img_t2, prob=args.prob,
        max_passes=args.max_passes, coverage_target=args.coverage_target,
    )
    t2_mask = postprocess(seg_t2, num_cls, args.min_area, args.max_hole, args.smooth_kernel)

    # 实例级变化检测（IoU 匹配）
    print("[4/5] 实例级变化检测...")
    change_pred = compute_instance_change_map(
        t1_mask, t2_mask, num_cls,
        iou_threshold=args.iou_threshold,
        min_area=max(args.min_area, 4),
    )
    change_mask = postprocess(change_pred, num_cls, args.min_area, args.max_hole, args.smooth_kernel)

    # 保存结果 — 仅保留 t1_mask / t2_mask / change_mask / overlay
    print("[5/5] 保存结果...")
    t1_stem = Path(args.t1).stem
    t2_stem = Path(args.t2).stem
    out_dir = Path("output_cd") / f"{t1_stem}_{t2_stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 类别名映射 & 调色板
    cls_name_map = {}
    for idx, line in enumerate(name_list):
        synonyms = [s.strip().lower() for s in str(line).strip().split(",")]
        cls_name_map[idx] = synonyms
    palette = build_palette(num_cls, cls_name_map)
    palette[0] = [255, 255, 255]  # background → 白色

    # ① t1_mask
    t1_color = palette[t1_mask]
    Image.fromarray(t1_color).save(str(out_dir / "t1_mask.png"))
    print(f"      t1_mask: {out_dir / 't1_mask.png'}")

    # ② t2_mask
    t2_color = palette[t2_mask]
    Image.fromarray(t2_color).save(str(out_dir / "t2_mask.png"))
    print(f"      t2_mask: {out_dir / 't2_mask.png'}")

    # ③ change_mask
    cd_color = palette[change_mask]
    Image.fromarray(cd_color).save(str(out_dir / "change_mask.png"))
    print(f"      change_mask: {out_dir / 'change_mask.png'}")

    # ④ overlay（T1+变化 | T2+变化 并排）
    alpha = 0.4
    img_t1_f = np.array(img_t1).astype(np.float32)
    img_t2_f = np.array(img_t2).astype(np.float32)
    cd_f = cd_color.astype(np.float32)
    change_region = (change_mask > 0)[:, :, np.newaxis].astype(np.float32)
    ov_t1 = (img_t1_f * (1 - alpha * change_region) + cd_f * alpha * change_region).astype(np.uint8)
    ov_t2 = (img_t2_f * (1 - alpha * change_region) + cd_f * alpha * change_region).astype(np.uint8)
    overlay = np.concatenate([ov_t1, ov_t2], axis=1)
    Image.fromarray(overlay).save(str(out_dir / "overlay.png"))
    print(f"      overlay: {out_dir / 'overlay.png'}")

    print("完成！")


if __name__ == "__main__":
    main()
