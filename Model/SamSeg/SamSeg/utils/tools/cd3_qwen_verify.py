#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: Qwen-VL 视觉语义复核

对每个变化实例，调用 Qwen-VL 大模型进行二次判别。
Qwen-VL 结果优先级高于 RemoteCLIP。

用法:
    python tools/cd3_qwen_verify.py \
        --data_root "E:/xzkjxm/dataes/CLCD-CD" \
        --split test \
        --verify_mode uncertain

环境变量:
    DASHSCOPE_API_KEY — 阿里云 DashScope API Key
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import base64
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm

from cd_utils import (
    imread_unicode, parse_classes, load_instances, save_instances,
    append_process_log, DEFAULT_CLASSES,
)


def encode_image_base64(image_path: str) -> Optional[str]:
    """将图片编码为 base64 字符串"""
    img = imread_unicode(image_path)
    if img is None:
        return None
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')


def call_qwen_vl(api_key: str, t1_base64: str, t2_base64: str,
                 category: str, clip_score: float,
                 model_name: str = "qwen-vl-plus",
                 timeout: int = 60) -> Dict:
    """
    调用 Qwen-VL API 进行二次判别

    出参:
    - {"keep": bool, "reason": str}
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai: pip install openai")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    prompt_text = f"""You are a remote sensing change detection expert.

Category: {category}
RemoteCLIP score: {clip_score}

Examine the T1 (before) and T2 (after) images of a detected change region.
Determine whether this region truly belongs to the specified category "{category}" and whether the detected change is valid.

Return ONLY a JSON object (no other text):
{{"keep": true/false, "reason": "your brief explanation"}}"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{t1_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{t2_base64}"}},
                {"type": "text", "text": prompt_text},
            ]
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        timeout=timeout,
    )

    result_text = response.choices[0].message.content.strip()

    # 解析 JSON
    try:
        # 尝试直接解析
        result = json.loads(result_text)
    except json.JSONDecodeError:
        # 尝试从 markdown code block 中提取
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            # 尝试找到第一个 { ... }
            brace_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if brace_match:
                result = json.loads(brace_match.group(0))
            else:
                return {"keep": True, "reason": f"Failed to parse Qwen response: {result_text[:200]}"}

    return {
        "keep": bool(result.get("keep", True)),
        "reason": str(result.get("reason", "")),
    }


def process_single_sample(instances_path: str, postprocess_dir: str,
                          api_key: str, clip_threshold: float,
                          verify_mode: str = "uncertain",
                          model_name: str = "qwen-vl-plus") -> Dict:
    """
    处理单个样本的 instances.json，进行 Qwen-VL 验证
    """
    data = load_instances(instances_path)
    if data is None:
        return {"error": 1}

    instances = data.get("instances", [])
    if not instances:
        return {"no_instances": 1}

    verified = 0
    keep_changed = 0

    for inst in instances:
        if inst.get("files") is None:
            continue

        clip_score = inst.get("clip_score") or 0.0

        # 根据 verify_mode 决定是否调用 Qwen-VL
        if verify_mode == "uncertain" and clip_score >= clip_threshold:
            # 高置信度实例跳过 Qwen-VL 验证
            inst["qwen_result"] = {"keep": True, "reason": "Skipped (high CLIP score)"}
            continue

        t1_path = os.path.join(postprocess_dir, inst["files"]["t1"])
        t2_path = os.path.join(postprocess_dir, inst["files"]["t2"])

        if not os.path.exists(t1_path) or not os.path.exists(t2_path):
            continue

        t1_b64 = encode_image_base64(t1_path)
        t2_b64 = encode_image_base64(t2_path)

        if t1_b64 is None or t2_b64 is None:
            continue

        try:
            qwen_result = call_qwen_vl(
                api_key, t1_b64, t2_b64,
                inst.get("category", ""), clip_score,
                model_name=model_name
            )

            old_keep = inst.get("keep", True)
            inst["qwen_result"] = qwen_result
            # Qwen-VL 结果覆盖 CLIP 判定
            inst["keep"] = qwen_result["keep"]

            if old_keep != inst["keep"]:
                keep_changed += 1

            verified += 1

        except Exception as e:
            inst["qwen_result"] = {"keep": inst.get("keep", True), "reason": f"Qwen-VL error: {str(e)}"}
            print(f"\n  Warning: Qwen-VL failed for instance {inst.get('instance_id')}: {e}")

    # 保存更新后的 instances.json
    save_instances(data, instances_path)

    return {"verified": verified, "keep_changed": keep_changed}


def process_dataset(data_root: str, split: str,
                    api_key: str, clip_threshold: float,
                    verify_mode: str = "uncertain",
                    model_name: str = "qwen-vl-plus"):
    """批量处理数据集"""
    postprocess_dir = os.path.join(data_root, split)

    json_files = sorted(Path(postprocess_dir).glob("*_instances.json"))
    if not json_files:
        print(f"Warning: 未找到 instances.json")
        return

    print(f"\n{'='*60}")
    print(f"Step 3: Qwen-VL 二次判别 - {split}")
    print(f"{'='*60}")
    print(f"样本数: {len(json_files)}")
    print(f"验证模式: {verify_mode}")
    print(f"模型: {model_name}")
    print(f"{'='*60}\n")

    log_path = os.path.join(postprocess_dir, "process_log.json")
    total_verified = 0
    total_changed = 0

    for jf in tqdm(json_files, desc=f"Qwen-VL {split}"):
        result = process_single_sample(
            str(jf), postprocess_dir,
            api_key, clip_threshold,
            verify_mode, model_name
        )
        total_verified += result.get("verified", 0)
        total_changed += result.get("keep_changed", 0)

    append_process_log(log_path, "step3_qwen_verify",
                       f"Qwen-VL verified {total_verified}, changed {total_changed}", {
        "total_verified": total_verified,
        "keep_changed": total_changed,
        "verify_mode": verify_mode,
        "model": model_name,
    })

    print(f"\n{'='*60}")
    print(f"Step 3 完成")
    print(f"{'='*60}")
    print(f"验证: {total_verified} 个实例")
    print(f"判定变更: {total_changed} 个")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Step 3: Qwen-VL 二次判别")
    parser.add_argument("--data_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--split", type=str, default="test", help="数据划分")
    parser.add_argument("--clip_threshold", type=float, default=0.5, help="CLIP 阈值（用于 uncertain 模式）")
    parser.add_argument("--verify_mode", type=str, default="uncertain",
                        choices=["all", "uncertain"], help="验证模式: all=全部, uncertain=仅低 CLIP 分")
    parser.add_argument("--model_name", type=str, default="qwen-vl-plus", help="Qwen-VL 模型名")
    parser.add_argument("--api_key", type=str, default=None, help="DashScope API Key (或用环境变量)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: 请设置 DASHSCOPE_API_KEY 环境变量或使用 --api_key 参数")
        return 1

    if args.split == "all":
        for sp in ["train", "val", "test"]:
            process_dataset(args.data_root, sp, api_key, args.clip_threshold,
                          args.verify_mode, args.model_name)
    else:
        process_dataset(args.data_root, args.split, api_key, args.clip_threshold,
                       args.verify_mode, args.model_name)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
