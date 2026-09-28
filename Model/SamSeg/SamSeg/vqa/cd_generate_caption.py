#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CD Caption 生成工具

功能：基于 T1/T2 分割掩码 + 原图，调用 VLM 生成变化检测描述

输入：{data_root}/{split}/{t1,t2,t1_mask,t2_mask,change_mask}/*.png
输出：{data_root}/{split}/caption_{split}.json

流程：
1. AI 根据 T1/T2 分割掩码（带颜色定义）识别哪些类别发生了变化
2. AI 结合原图理解变化的具体情况
3. 按模板生成 caption + 结构化分析

用法：
python vqa/cd_generate_caption.py --data_root "E:/xzkjxm/dataes/WHU-CD_20" --split test
"""
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import json
import re
import base64
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import numpy as np
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════
#  颜色定义 — 与 infer.py NAME_COLOR 一致
# ═══════════════════════════════════════════════════════════════
COLOR_MAP = {
    "background":  (255, 255, 255),
    "building":    (255, 0, 0),
    "road":        (128, 0, 128),
    "water":       (0, 0, 255),
    "bareland":    (139, 90, 43),
    "vegetation":  (0, 128, 0),
    "farmland":    (0, 100, 0),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """
    入参:
        path: 图像路径
    方法:
        np.fromfile + cv2.imdecode，兼容中文路径
    出参:
        Optional[np.ndarray]: 图像数组或 None
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return image
    except Exception:
        return None


def encode_image_to_base64(image_path: Path, resize_max_dim: int = 768) -> str:
    """
    入参:
        image_path: 图像路径
        resize_max_dim: 最长边限制
    方法:
        读取并缩放图像，编码为 JPEG base64 data URL
    出参:
        str: base64 data URL
    """
    image = imread_unicode(image_path)
    if image is not None:
        height, width = image.shape[:2]
        if max(height, width) > resize_max_dim:
            scale = resize_max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            encoded = base64.b64encode(buffer).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".tif": "image/tiff"}
    return f"data:{mime_map.get(suffix, 'image/png')};base64,{encoded}"


def get_message_content(message) -> str:
    """
    入参:
        message: OpenAI 返回的 message 对象
    方法:
        兼容推理模型的 reasoning 字段
    出参:
        str: 提取的文本内容
    """
    content = message.content or getattr(message, "reasoning", None) or ""
    if content.startswith("Thinking Process"):
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Answer:") or line.startswith("回答:"):
                return "\n".join(lines[i + 1:]).strip()
    return content


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    入参:
        text: 模型返回的原始文本
    方法:
        从 markdown 代码块或普通文本中提取 JSON 对象
    出参:
        Dict[str, Any]: 解析后的 JSON 对象
    """
    content = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)
    else:
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            content = content[first_brace:last_brace + 1]
    return json.loads(content)


def build_color_table_text() -> str:
    """生成颜色-类别映射表文本"""
    lines = ["| Color | RGB | Class |"]
    lines.append("|-------|-----|-------|")
    for cls, rgb in COLOR_MAP.items():
        lines.append(f"| {cls} | ({rgb[0]},{rgb[1]},{rgb[2]}) | {cls} |")
    return "\n".join(lines)


def find_matching_files(split_dir: Path, subdir: str, stem: str) -> List[Path]:
    """
    入参:
        split_dir: split 目录
        subdir: 子目录名
        stem: 文件名（不含扩展名）
    方法:
        查找匹配的图像文件
    出参:
        List[Path]: 匹配的文件路径列表
    """
    directory = split_dir / subdir
    if not directory.exists():
        return []
    results = []
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            results.append(candidate)
    return results


def get_all_stems(split_dir: Path, subdir: str) -> List[str]:
    """
    入参:
        split_dir: split 目录
        subdir: 子目录名
    方法:
        获取目录下所有图像文件的 stem（不含扩展名）
    出参:
        List[str]: stem 列表
    """
    directory = split_dir / subdir
    if not directory.exists():
        return []
    stems = []
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            stems.append(f.stem)
    return stems


def generate_caption_for_sample(
    client: OpenAI,
    model_name: str,
    split_dir: Path,
    stem: str,
) -> Optional[Dict[str, Any]]:
    """
    入参:
        client: OpenAI 客户端
        model_name: 模型名称
        split_dir: split 目录
        stem: 文件名 stem
    方法:
        1. 查找 T1/T2 原图 + T1/T2 分割掩码
        2. 发送给 VLM：先看掩码识别变化类别，再看原图理解细节
        3. 生成 caption + 结构化分析
    出参:
        Optional[Dict]: caption 结果；失败返回 None
    """
    # 查找各路图像
    t1_files = find_matching_files(split_dir, "t1", stem)
    t2_files = find_matching_files(split_dir, "t2", stem)
    t1_mask_files = find_matching_files(split_dir, "t1_mask", stem)
    t2_mask_files = find_matching_files(split_dir, "t2_mask", stem)

    t1_path = t1_files[0] if t1_files else None
    t2_path = t2_files[0] if t2_files else None
    t1_mask_path = t1_mask_files[0] if t1_mask_files else None
    t2_mask_path = t2_mask_files[0] if t2_mask_files else None

    if not t1_path or not t2_path:
        return None

    color_table = build_color_table_text()

    system_prompt = (
        "You are an expert in remote sensing change detection analysis.\n"
        "You will receive segmentation masks and original images for bi-temporal analysis.\n\n"
        "## Color-to-Class Mapping (Segmentation Masks)\n"
        f"{color_table}\n\n"
        "## Analysis Steps\n"
        "1. Compare T1 and T2 segmentation masks to identify which categories changed\n"
        "2. Compare the two masks to locate change regions\n"
        "3. Examine original T1/T2 images to understand specific change details\n\n"
        "## Caption Template (MUST follow exactly)\n"
        "\"The bi-temporal remote sensing images show that [changed object] is located "
        "[direction] of [reference object], with [surrounding object] distributed nearby. "
        "Compared with the earlier observation, [changed object] has undergone a noticeable change, "
        "while the surrounding area remains relatively stable.\"\n\n"
        "## Output Format\n"
        "Return raw JSON only:\n"
        "{\n"
        '  "caption": "caption following the template above",\n'
        '  "changed_categories": ["list of categories that changed"],\n'
        '  "change_regions": [{"location": "direction", "from_class": "class", "to_class": "class"}],\n'
        '  "num_change_regions": number,\n'
        '  "t1_class_counts": {"class": count},\n'
        '  "t2_class_counts": {"class": count},\n'
        '  "change_type": "expansion or shrinkage or renewal or transformation",\n'
        '  "surrounding_categories": ["nearby unchanged categories"],\n'
        '  "reference_objects": ["reference objects with locations"]\n'
        "}\n\n"
        "Use exact lowercase class names: building, road, water, bareland, vegetation, farmland.\n"
        "Use spatial directions: north, south, east, west, northeast, northwest, southeast, southwest, center."
    )

    user_content: List[Dict[str, Any]] = []

    # Step 1: 分割掩码（让 AI 通过对比 T1/T2 掩码识别变化类别）
    if t1_mask_path:
        user_content.append({"type": "text", "text": "T1 segmentation mask:"})
        user_content.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(t1_mask_path)}})
    if t2_mask_path:
        user_content.append({"type": "text", "text": "T2 segmentation mask:"})
        user_content.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(t2_mask_path)}})

    # Step 2: 原图（让 AI 理解变化细节）
    user_content.append({"type": "text", "text": "Original T1 image:"})
    user_content.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(t1_path)}})
    user_content.append({"type": "text", "text": "Original T2 image:"})
    user_content.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(t2_path)}})

    user_content.append({"type": "text", "text": "Analyze the changes and generate the caption following the template."})

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
        extra_body={"disable_search": True},
    )

    text = get_message_content(response.choices[0].message).strip()
    result = extract_json_object(text)

    # 验证必要字段
    if "caption" not in result:
        raise ValueError("Missing 'caption' field in VLM response")

    result["image_name"] = stem
    result["t1"] = f"t1/{t1_path.name}"
    result["t2"] = f"t2/{t2_path.name}"
    if t1_mask_path:
        result["t1_mask"] = f"t1_mask/{t1_mask_path.name}"
    if t2_mask_path:
        result["t2_mask"] = f"t2_mask/{t2_mask_path.name}"

    return result


def process_split(
    data_root: str,
    split: str,
    client: OpenAI,
    model_name: str,
    max_workers: int = 1,
) -> None:
    """
    入参:
        data_root: 数据集根目录
        split: 数据集划分
        client: OpenAI 客户端
        model_name: 模型名称
        max_workers: 最大并行数
    方法:
        遍历所有样本，生成 caption，即时写入 caption_{split}.json
    出参:
        None
    """
    split_dir = Path(data_root) / split
    if not split_dir.exists():
        print(f"Skip: {split_dir} not found")
        return

    # 从 t1 目录获取文件列表
    stems = get_all_stems(split_dir, "t1")
    if not stems:
        print(f"Skip: no images in {split_dir / 't1'}")
        return

    output_path = split_dir / f"caption_{split}.json"

    print(f"\n{'=' * 60}")
    print(f"CD Caption Generation - {split}")
    print(f"Samples: {len(stems)}")
    print(f"{'=' * 60}")

    results: List[Dict[str, Any]] = []
    success_count = 0
    print_lock = threading.Lock()
    results_lock = threading.Lock()

    def flush():
        with results_lock:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    def _process_one(idx, stem):
        with print_lock:
            print(f"[{idx}/{len(stems)}] {stem}")
        try:
            result = generate_caption_for_sample(client, model_name, split_dir, stem)
            if result is None:
                with print_lock:
                    print(f"   SKIP {stem}: missing T1/T2 images")
                return None
            with print_lock:
                print(f"   OK {stem}: {result.get('change_type', 'unknown')}")
            return result
        except Exception as e:
            with print_lock:
                print(f"   FAIL {stem}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, i, stem): stem
            for i, stem in enumerate(stems, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                with results_lock:
                    results.append(result)
                    success_count += 1
            flush()

    results.sort(key=lambda x: str(x.get("image_name", "")))
    flush()

    print(f"\n{'=' * 60}")
    print(f"CD Caption Complete - {split}")
    print(f"Success: {success_count}/{len(stems)}")
    print(f"Saved: {output_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="CD Caption Generation")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"], help="Split")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--model", type=str, default="qwen-vl-plus", help="VLM model")
    parser.add_argument("--base_url", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API base URL")
    parser.add_argument("--max_workers", type=int, default=1, help="Workers")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("API Key required. Set DASHSCOPE_API_KEY or pass --api_key.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"Model: {args.model}, API: {args.base_url}")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        split_dir = Path(args.data_root) / split
        if not split_dir.is_dir():
            continue
        process_split(args.data_root, split, client, args.model, args.max_workers)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
