#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: RemoteCLIP 语义一致性验证

对每个变化实例，计算 T2 图像与指定类别文本的相似度（clip_score），
低于阈值则标记 keep=false。

用法:
    python tools/cd2_clip_verify.py \
        --data_root "E:/xzkjxm/dataes/CLCD-CD" \
        --split test \
        --clip_threshold 0.5
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from cd_utils import (
    imread_unicode, parse_classes, save_instances, load_instances,
    append_process_log, DEFAULT_CLASSES,
)


def load_remoteclip_model(model_name: str = "ViT-B-32",
                          checkpoint_path: str = None,
                          device: str = "cuda"):
    """加载 RemoteCLIP 模型"""
    try:
        import open_clip
    except ImportError:
        raise ImportError("请先安装 open_clip_torch: pip install open_clip_torch")

    if checkpoint_path is None:
        checkpoint_filename = f"RemoteCLIP-{model_name}.pt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, f"../checkpoints/{checkpoint_filename}"),
            os.path.join(script_dir, f"checkpoints/{checkpoint_filename}"),
            f"checkpoints/{checkpoint_filename}",
            f"../checkpoints/{checkpoint_filename}",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break

        if checkpoint_path is None:
            raise FileNotFoundError(f"找不到模型权重 {checkpoint_filename}")

    print(f"Loading RemoteCLIP: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=os.path.abspath(checkpoint_path), device=device
    )
    model = model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer


def build_category_prompts(cls_name_map: Dict[int, List[str]]) -> Dict[str, List[str]]:
    """
    构建每个类别的同义词提示词列表

    出参: {category_name: [prompt1, prompt2, ...]}
    """
    prompts = {}
    for cls_id, synonyms in cls_name_map.items():
        if cls_id == 0:
            continue
        main_name = synonyms[0]
        # RemoteCLIP 训练时使用的 prompt 模板
        prompts[main_name] = [f"a photo of {s}" for s in synonyms]
    return prompts


def compute_clip_score(model, preprocess, tokenizer,
                       image_path: str, category: str,
                       category_prompts: Dict[str, List[str]],
                       all_categories: List[str],
                       device: str = "cuda") -> Dict:
    """
    计算图像与各类别文本的 CLIP 相似度，返回绝对分数和相对排名信息

    方法:
    - 对每个类别（含同义词）编码文本特征
    - 编码图像特征
    - 按同义词分组取 max → 各类别得分
    - 计算 clip_score (目标类别绝对分) 和 relative_score (目标类别得分 / 最高得分)
    - 判定目标类别是否为 Top-1 预测

    出参:
    - Dict: {
        "clip_score": float,          # 目标类别绝对相似度
        "relative_score": float,      # 目标类别 / 最高类别 (0~1, >0.5 说明接近或等于最高)
        "predicted_class": str,       # CLIP 预测的类别
        "is_top1": bool,              # 目标类别是否为 Top-1
        "rank": int,                  # 目标类别排名 (1=最高)
      }
    """
    import torch
    from PIL import Image

    img = imread_unicode(image_path)
    if img is None:
        return {"clip_score": 0.0, "relative_score": 0.0, "predicted_class": "", "is_top1": False, "rank": 999}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    # 构建所有类别的提示词
    all_prompts = []
    prompt_to_cat = []
    for cat in all_categories:
        prompts = category_prompts.get(cat, [f"a photo of {cat}"])
        for p in prompts:
            all_prompts.append(p)
            prompt_to_cat.append(cat)

    text_tokens = tokenizer(all_prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    # 按类别分组取 max
    cat_scores = {}
    for i, cat in enumerate(prompt_to_cat):
        score = float(similarities[i])
        if cat not in cat_scores or score > cat_scores[cat]:
            cat_scores[cat] = score

    # 目标类别得分
    target_score = cat_scores.get(category, 0.0)

    # 排名
    sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
    predicted_class = sorted_cats[0][0]
    max_score = sorted_cats[0][1]
    rank = next(i + 1 for i, (c, _) in enumerate(sorted_cats) if c == category)
    is_top1 = (rank == 1)
    relative_score = target_score / max_score if max_score > 0 else 0.0

    return {
        "clip_score": round(float(target_score), 4),
        "relative_score": round(float(relative_score), 4),
        "predicted_class": predicted_class,
        "is_top1": is_top1,
        "rank": rank,
    }
    target_scores = [float(similarities[i]) for i, c in enumerate(prompt_to_cat) if c == category]
    return max(target_scores) if target_scores else 0.0


def process_single_sample(instances_path: str, postprocess_dir: str,
                          model, preprocess, tokenizer,
                          category_prompts: Dict[str, List[str]],
                          all_categories: List[str],
                          clip_threshold: float,
                          device: str = "cuda") -> Dict:
    """
    处理单个样本的 instances.json，进行 CLIP 验证

    入参:
    - instances_path: instances.json 路径
    - postprocess_dir: split 目录
    - model, preprocess, tokenizer: RemoteCLIP 组件
    - category_prompts: 类别提示词
    - all_categories: 所有类别列表
    - clip_threshold: CLIP 相对分数阈值 (relative_score)
    - device: 设备

    出参:
    - Dict: 统计信息
    """
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
            return True  # 保留无文件的记录
        bbox = inst.get("bbox")
        if bbox is None:
            return True
        # bbox 格式: [x1, y1, x2, y2]
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

        t2_path = os.path.join(postprocess_dir, inst["files"]["t2"])
        if not os.path.exists(t2_path):
            continue

        category = inst.get("category", "")
        result = compute_clip_score(
            model, preprocess, tokenizer,
            t2_path, category, category_prompts, all_categories, device
        )

        inst["clip_score"] = result["clip_score"]
        inst["relative_score"] = result["relative_score"]
        inst["predicted_class"] = result["predicted_class"]
        inst["is_top1"] = result["is_top1"]
        inst["rank"] = result["rank"]

        # 过滤逻辑: 相对分数低于阈值 → 视为误检
        # relative_score = 目标类别得分 / 最高类别得分
        # > 0.9 说明目标类别接近或等于最高分（正确分类）
        # < 0.9 说明 CLIP 认为该区域属于其他类别
        inst["keep"] = result["relative_score"] >= clip_threshold
        verified += 1
        if not inst["keep"]:
            filtered += 1

    # 保存更新后的 instances.json
    save_instances(data, instances_path)

    return {"verified": verified, "filtered": filtered}


def process_dataset(data_root: str, split: str,
                    model, preprocess, tokenizer,
                    cls_name_map: Dict, clip_threshold: float,
                    device: str = "cuda"):
    """批量处理数据集"""
    postprocess_dir = os.path.join(data_root, split)

    # 查找所有 instances.json
    json_files = sorted(Path(postprocess_dir).glob("*_instances.json"))

    if not json_files:
        print(f"Warning: 未找到 instances.json 文件")
        return

    # 构建提示词
    category_prompts = build_category_prompts(cls_name_map)
    cls_display = {idx: syns[0] for idx, syns in cls_name_map.items()}
    all_categories = [cls_display[i] for i in range(1, max(cls_name_map.keys()) + 1) if i in cls_display]

    print(f"\n{'='*60}")
    print(f"Step 2: RemoteCLIP 语义验证 - {split}")
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
    print(f"Step 2 完成")
    print(f"{'='*60}")
    print(f"验证: {total_verified} 个实例")
    print(f"过滤: {total_filtered} 个 (clip_score < {clip_threshold})")
    print(f"保留: {total_verified - total_filtered} 个")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Step 2: RemoteCLIP 语义验证")
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
