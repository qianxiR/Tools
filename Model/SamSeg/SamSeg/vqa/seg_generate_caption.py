#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEG Caption 生成工具

功能：基于分割掩码 + 原图，调用 VLM 生成语义分割描述

输入：{data_root}/{split}/{image,seg_mask}/*.png
输出：{data_root}/{split}/caption_{split}.json

流程：
1. AI 根据 seg_mask（带颜色定义）识别图像中有哪些类别
2. AI 结合原图理解场景细节
3. 按模板生成 caption + 结构化分析

用法：
python vqa/seg_generate_caption.py --data_root "E:/xzkjxm/dataes/WHU_20" --split test
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


def find_matching_file(split_dir: Path, subdir: str, stem: str) -> Optional[Path]:
    """
    入参:
        split_dir: split 目录
        subdir: 子目录名
        stem: 文件名（不含扩展名）
    方法:
        查找匹配的图像文件（取第一个匹配）
    出参:
        Optional[Path]: 匹配的文件路径或 None
    """
    directory = split_dir / subdir
    if not directory.exists():
        return None
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_all_stems(split_dir: Path, subdir: str) -> List[str]:
    """
    入参:
        split_dir: split 目录
        subdir: 子目录名
    方法:
        获取目录下所有图像文件的 stem
    出参:
        List[str]: stem 列表
    """
    directory = split_dir / subdir
    if not directory.exists():
        return []
    return sorted(f.stem for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


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
        1. 查找原图 + 分割掩码
        2. 发送给 VLM：先看掩码识别类别，再看原图理解细节
        3. 生成 caption + 结构化分析
    出参:
        Optional[Dict]: caption 结果；失败返回 None
    """
    image_path = find_matching_file(split_dir, "image", stem)
    seg_mask_path = find_matching_file(split_dir, "seg_mask", stem)

    if not image_path:
        return None

    color_table = build_color_table_text()

    system_prompt = (
        "You are an expert in remote sensing image segmentation analysis.\n"
        "You will receive a segmentation mask and the original image.\n\n"
        "## Color-to-Class Mapping (Segmentation Mask)\n"
        f"{color_table}\n\n"
        "## Analysis Steps\n"
        "1. Examine the segmentation mask to identify which land-cover categories are present\n"
        "2. Determine the spatial distribution and relative location of each category\n"
        "3. Examine the original image to understand the scene details\n"
        "4. Classify the overall urban scene type\n\n"
        "## Caption Template (MUST follow exactly)\n"
        "\"The remote sensing image shows that [target object] is located "
        "[direction] of [reference object], with [surrounding object] distributed nearby. "
        "Together, these land-cover elements form the spatial structure of the scene.\"\n\n"
        "## Output Format\n"
        "Return raw JSON only:\n"
        "{\n"
        '  "caption": "caption following the template above",\n'
        '  "present_categories": ["list of categories present"],\n'
        '  "category_details": [{"category": "name", "location": "direction", "description": "brief desc"}],\n'
        '  "num_regions": total_number_of_regions,\n'
        '  "class_counts": {"category": count},\n'
        '  "scene_type": "one of the scene types below",\n'
        '  "surrounding_categories": ["nearby categories"]\n'
        "}\n\n"
        "Scene types: high_density_urban, low_density_urban, residential_area, commercial_area, "
        "industrial_area, transportation_area, green_space_area, water_area, "
        "urban_rural_transition, mixed_functional_area\n\n"
        "Use exact lowercase class names: building, road, water, bareland, vegetation, farmland.\n"
        "Use spatial directions: north, south, east, west, northeast, northwest, southeast, southwest, center."
    )

    user_content: List[Dict[str, Any]] = []

    # Step 1: 分割掩码（让 AI 识别类别）
    if seg_mask_path:
        user_content.append({"type": "text", "text": "Segmentation mask:"})
        user_content.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(seg_mask_path)}})

    # Step 2: 原图（让 AI 理解细节）
    user_content.append({"type": "text", "text": "Original image:"})
    user_content.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(image_path)}})

    user_content.append({"type": "text", "text": "Analyze the scene and generate the caption following the template."})

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

    if "caption" not in result:
        raise ValueError("Missing 'caption' field in VLM response")

    result["image_name"] = stem
    result["image"] = f"image/{image_path.name}"
    if seg_mask_path:
        result["seg_mask"] = f"seg_mask/{seg_mask_path.name}"

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

    # 从 image 目录获取文件列表
    stems = get_all_stems(split_dir, "image")
    if not stems:
        print(f"Skip: no images in {split_dir / 'image'}")
        return

    output_path = split_dir / f"caption_{split}.json"

    print(f"\n{'=' * 60}")
    print(f"SEG Caption Generation - {split}")
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
                    print(f"   SKIP {stem}: missing original image")
                return None
            with print_lock:
                print(f"   OK {stem}: {result.get('scene_type', 'unknown')}")
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
    print(f"SEG Caption Complete - {split}")
    print(f"Success: {success_count}/{len(stems)}")
    print(f"Saved: {output_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="SEG Caption Generation")
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
