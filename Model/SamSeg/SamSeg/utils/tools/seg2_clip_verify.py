#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEG Step 2: RemoteCLIP 语义一致性验证

对每个分割实例，计算原始图像与指定类别文本的相似度（clip_score），
低于阈值则标记 keep=false。

用法:
    python tools/seg2_clip_verify.py \
        --data_root "E:/xzkjxm/dataes/WHU_20" \
        --split test \
        --clip_threshold 0.5
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from cd_utils import (
    parse_classes, save_instances, load_instances,
    append_process_log, DEFAULT_CLASSES,
)

# 直接复用 CD 的 CLIP 函数
from cd2_clip_verify import (
    load_remoteclip_model, build_category_prompts, compute_clip_score,
)


def process_single_sample(instances_path: str, postprocess_dir: str,
                          model, preprocess, tokenizer,
                          category_prompts: Dict[str, List[str]],
                          all_categories: List[str],
                          clip_threshold: float,
                          device: str = "cuda") -> Dict:
    """处理单个样本的 instances.json，进行 CLIP 验证"""
    data = load_instances(instances_path)
    if data is None:
        return {"error": 1}

    instances = data.get("instances", [])
    if not instances:
        return {"no_instances": 1}

    # 在 CLIP 验证之前，先删除面积过小的实例（宽或高 < 10px）
    MIN_DIM = 10
    before_count = len(instances)
    def _is_large_enough(inst):
        if inst.get("files") is None:
            return True
        bbox = inst.get("bbox")
        if bbox is None:
            return True
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return w >= MIN_DIM and h >= MIN_DIM

    instances[:] = [inst for inst in instances if _is_large_enough(inst)]
    removed_small = before_count - len(instances)

    verified = 0
    filtered = 0
    # removed_small 单独统计，不计入 filtered

    for inst in instances:
        if inst.get("files") is None:
            continue

        # SEG: 对原始图像子图做 CLIP 验证（CD 是对 T2）
        img_path = os.path.join(postprocess_dir, inst["files"]["image"])
        if not os.path.exists(img_path):
            continue

        category = inst.get("category", "")
        result = compute_clip_score(
            model, preprocess, tokenizer,
            img_path, category, category_prompts, all_categories, device
        )

        inst["clip_score"] = result["clip_score"]
        inst["relative_score"] = result["relative_score"]
        inst["predicted_class"] = result["predicted_class"]
        inst["is_top1"] = result["is_top1"]
        inst["rank"] = result["rank"]

        inst["keep"] = result["relative_score"] >= clip_threshold
        verified += 1
        if not inst["keep"]:
            filtered += 1

    save_instances(data, instances_path)
    return {"verified": verified, "filtered": filtered}


def process_dataset(data_root: str, split: str,
                    model, preprocess, tokenizer,
                    cls_name_map: Dict, clip_threshold: float,
                    device: str = "cuda"):
    """批量处理数据集"""
    postprocess_dir = os.path.join(data_root, split)

    json_files = sorted(Path(postprocess_dir).glob("*_instances.json"))

    if not json_files:
        print(f"Warning: 未找到 instances.json 文件")
        return

    category_prompts = build_category_prompts(cls_name_map)
    cls_display = {idx: syns[0] for idx, syns in cls_name_map.items()}
    all_categories = [cls_display[i] for i in range(1, max(cls_name_map.keys()) + 1) if i in cls_display]

    print(f"\n{'='*60}")
    print(f"SEG Step 2: RemoteCLIP 语义验证 - {split}")
    print(f"{'='*60}")
    print(f"样本数: {len(json_files)}")
    print(f"CLIP 阈值: {clip_threshold}")
    print(f"类别: {all_categories}")
    print(f"{'='*60}\n")

    log_path = os.path.join(postprocess_dir, "process_log.json")
    total_verified = 0
    total_filtered = 0

    for jf in tqdm(json_files, desc=f"CLIP verify {split}"):
        result = process_single_sample(
            str(jf), postprocess_dir,
            model, preprocess, tokenizer,
            category_prompts, all_categories,
            clip_threshold, device
        )
        total_verified += result.get("verified", 0)
        total_filtered += result.get("filtered", 0)

    append_process_log(log_path, "step2_clip_verify",
                       f"CLIP verified {total_verified}, filtered {total_filtered}", {
        "total_verified": total_verified,
        "total_filtered": total_filtered,
        "clip_threshold": clip_threshold,
    })

    print(f"\n{'='*60}")
    print(f"SEG Step 2 完成")
    print(f"{'='*60}")
    print(f"验证: {total_verified} 个实例")
    print(f"过滤: {total_filtered} 个 (clip_score < {clip_threshold})")
    print(f"保留: {total_verified - total_filtered} 个")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SEG Step 2: RemoteCLIP 语义验证")
    parser.add_argument("--data_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--split", type=str, default="test", help="数据划分")
    parser.add_argument("--classes", type=str, default=None, help="类别文件路径")
    parser.add_argument("--clip_threshold", type=float, default=0.9,
                        help="CLIP 相对分数阈值 (relative_score=目标类别分/最高类别分, <阈值则过滤)")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32",
                        choices=["RN50", "ViT-B-32", "ViT-L-14"], help="RemoteCLIP 模型")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型权重路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()

    cls_name_map, num_cls = parse_classes(args.classes or DEFAULT_CLASSES)
    model, preprocess, tokenizer = load_remoteclip_model(args.clip_model, args.checkpoint, args.device)

    if args.split == "all":
        for sp in ["train", "val", "test"]:
            process_dataset(args.data_root, sp, model, preprocess, tokenizer,
                          cls_name_map, args.clip_threshold, args.device)
    else:
        process_dataset(args.data_root, args.split, model, preprocess, tokenizer,
                       cls_name_map, args.clip_threshold, args.device)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
