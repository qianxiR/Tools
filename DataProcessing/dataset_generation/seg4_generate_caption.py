#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEG 区域描述 + metadata 汇总工具（SEG 模式 Step seg4）

功能：
1. 遍历每个样本的 regions.json
2. 对 clip 与 llm 分类不一致的局部 patch 调用 VLM 修正
3. 调用 VLM 为每个样本生成场景描述（caption/feature/summary）
4. 汇总生成 split 级别的 metadata.json

输入：seg1-3 输出的 regions.json（含 classification.clip/llm）
输出：{split}/metadata.json

与 CD 模式的区别：
- 无 t1/t2 双时相关联，files 字段为 image/label
- classification 输出 class（非 t1_class/t2_class）
- metadata 记录包含 caption/feature_analysis/summary 场景描述
"""
import os
import json
import re
import base64
import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import numpy as np
from openai import OpenAI


VALID_CLASSES = {
    "building", "highway", "vegetation", "farmland", "bareland", "water",
    "bare_land",
}

CLASS_ALIASES = {
    "bareland": "bareland", "bare_land": "bareland",
    "barren_land": "bareland", "barrenland": "bareland",
    "road": "highway", "roads": "highway",
}


def get_message_content(message) -> str:
    """
    入参:
        message: OpenAI 返回的 message 对象
    方法:
        兼容推理模型的 reasoning 字段，统一提取可解析文本
    出参:
        str: 文本内容
    """
    content = message.content or getattr(message, "reasoning", None) or ""
    if content.startswith("Thinking Process"):
        lines = content.split("\n")
        for idx, line in enumerate(lines):
            if line.startswith("Answer:") or line.startswith("回答:"):
                return "\n".join(lines[idx + 1:]).strip()
    return content


def normalize_class_label(value: Any) -> str:
    """
    入参:
        value: 原始类别值
    方法:
        统一映射到标准类别名
    出参:
        str: 标准化类别名；无效返回空字符串
    """
    if value is None:
        return ""
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    label = CLASS_ALIASES.get(label, label)
    if label not in VALID_CLASSES:
        return ""
    return label


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """
    入参:
        path (Path): 图像路径
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
        image_path (Path): 图像路径
        resize_max_dim (int): 最大边长限制
    方法:
        读取图像并按最长边缩放，编码为 JPEG base64 data URL
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


def find_image_path(split_dir: Path, subdir: str, name: str) -> str:
    """
    入参:
        split_dir (Path): split 目录
        subdir (str): 子目录名 (image/label)
        name (str): 文件名（不含扩展名）
    方法:
        尝试常见遥感图像扩展名
    出参:
        str: 相对路径；未找到返回空字符串
    """
    for ext in ["png", "jpg", "jpeg", "tif", "tiff"]:
        candidate = split_dir / subdir / f"{name}.{ext}"
        if candidate.exists():
            return f"{subdir}/{name}.{ext}"
    return ""


def resolve_region_class(region: Dict[str, Any]) -> str:
    """
    入参:
        region (Dict): 单个 region 结构
    方法:
        按优先级 final > llm > clip 提取最终类别
    出参:
        str: 类别名；无有效分类返回空字符串
    """
    classification = region.get("classification") or {}
    for source in ("final", "llm", "clip"):
        payload = classification.get(source) or {}
        cls = normalize_class_label(payload.get("class"))
        if cls:
            return cls
    return ""


def should_refine_region(region: Dict[str, Any]) -> bool:
    """
    入参:
        region (Dict): 单个 region 结构
    方法:
        检查 clip 与 llm 的 class 是否不一致
    出参:
        bool: 是否需要 VLM 修正
    """
    region_id = int(region.get("region_id", 0) or 0)
    if region_id <= 0:
        return False

    classification = region.get("classification") or {}
    llm = classification.get("llm") or {}
    clip = classification.get("clip") or {}

    llm_cls = normalize_class_label(llm.get("class"))
    clip_cls = normalize_class_label(clip.get("class"))

    return bool(llm_cls and clip_cls and llm_cls != clip_cls)


def refine_region_with_patch(
    client: OpenAI,
    model_name: str,
    regions_json_path: Path,
    region: Dict[str, Any],
) -> Optional[str]:
    """
    入参:
        client: OpenAI 客户端
        model_name: 模型名称
        regions_json_path: regions.json 路径
        region: 单个 region 结构
    方法:
        读取局部 image patch，调用 VLM 做单类别判定
    出参:
        Optional[str]: 修正后的类别名；不需要修正返回 None
    """
    if not should_refine_region(region):
        return None

    regions_root = regions_json_path.parent.parent
    files = region.get("files") or {}
    image_path = regions_root / str(files.get("image", "")).strip()

    if not image_path.exists():
        return None

    classification = region.get("classification") or {}
    llm = classification.get("llm") or {}
    clip = classification.get("clip") or {}
    location = str(region.get("location", "")).strip().lower()

    system_prompt = (
        "You classify a single local patch from a remote sensing segmentation image.\n"
        "Choose the most plausible land-cover class from this exact set only:\n"
        "building, highway, vegetation, farmland, bareland, water.\n"
        "Return raw JSON only: {\"class\": \"...\"}."
    )

    user_content = [
        {
            "type": "text",
            "text": (
                f"Region location: {location or 'unknown'}\n"
                f"LLM classified as: {llm.get('class', 'unknown')}\n"
                f"CLIP classified as: {clip.get('class', 'unknown')}\n"
                "Use the cropped patch as main evidence.\n"
                "Output only the final JSON."
            ),
        },
        {"type": "text", "text": "Local patch:"},
        {"type": "image_url", "image_url": {"url": encode_image_to_base64(image_path)}},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=256,
        extra_body={"disable_search": True},
    )

    text = get_message_content(response.choices[0].message).strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1:
            text = text[first:last + 1]

    result = json.loads(text)
    cls = normalize_class_label(result.get("class"))
    if not cls:
        raise ValueError("Patch refinement returned invalid class")
    return cls


def generate_caption(
    client: OpenAI,
    model_name: str,
    image_path: Path,
    region_facts: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    入参:
        client: OpenAI 客户端
        model_name: 模型名称
        image_path (Path): 完整图像路径
        region_facts (List[Dict]): 区域摘要 [{region_id, location, class}, ...]
    方法:
        发送完整图像 + 区域摘要给 VLM，生成 caption/feature/summary
    出参:
        Dict[str, str]: {caption, feature_analysis, summary}
    """
    facts_str = json.dumps(region_facts, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are a remote sensing image analysis expert.\n"
        "Analyze the provided satellite/aerial image and the structured region facts.\n"
        "Provide directional information from an overall perspective.\n"
        "Use descriptive terms like 'large area', 'small portion' for proportions.\n"
        "Do not give exact percentages.\n"
    )

    user_content = [
        {
            "type": "text",
            "text": (
                f"Region facts:\n```json\n{facts_str}\n```\n\n"
                "Respond in this exact format:\n"
                "<caption>One comprehensive sentence describing the entire image "
                "and its main features (total-subtotal structure)</caption>\n"
                "<feature>List the features you can see, with locations "
                "(e.g., 'building cluster in the northern section')</feature>\n"
                "<summary>A detailed paragraph summarizing the image content, "
                "combining visual elements and identified features. "
                "THIS SECTION IS REQUIRED.</summary>"
            ),
        },
        {"type": "image_url", "image_url": {"url": encode_image_to_base64(image_path)}},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1000,
        extra_body={"disable_search": True},
    )

    text = get_message_content(response.choices[0].message)

    caption = ""
    if "<caption>" in text and "</caption>" in text:
        caption = text.split("<caption>")[1].split("</caption>")[0].strip()

    feature = ""
    if "<feature>" in text and "</feature>" in text:
        feature = text.split("<feature>")[1].split("</feature>")[0].strip()

    summary = ""
    if "<summary>" in text and "</summary>" in text:
        summary = text.split("<summary>")[1].split("</summary>")[0].strip()

    return {"caption": caption, "feature_analysis": feature, "summary": summary}


def build_metadata_record(
    dataset_path: Path,
    split: str,
    regions_json_path: Path,
    client: Optional[OpenAI],
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """
    入参:
        dataset_path: 数据集根目录
        split: 数据划分名称
        regions_json_path: regions.json 路径
        client: OpenAI 客户端
        model_name: 模型名称
    方法:
        1. 读取 regions.json，对不一致区域做 VLM 修正
        2. 构建区域事实摘要，调用 VLM 生成场景描述
        3. 汇总为 metadata 记录
    出参:
        Optional[Dict]: 单条 metadata 记录
    """
    with open(regions_json_path, 'r', encoding='utf-8') as f:
        regions_data = json.load(f)

    # 修正不一致区域
    refined_count = 0
    if client is not None:
        for region in regions_data.get("regions", []):
            refined_cls = refine_region_with_patch(client, model_name, regions_json_path, region)
            if refined_cls is not None:
                classification = region.setdefault("classification", {})
                llm = classification.setdefault("llm", {})
                llm["class"] = refined_cls
                llm["refined_from_patch"] = True
                llm["refined_model"] = model_name
                refined_count += 1

        if refined_count > 0:
            with open(regions_json_path, 'w', encoding='utf-8') as f:
                json.dump(regions_data, f, ensure_ascii=False, indent=2)

    # 构建 final 分类
    for region in regions_data.get("regions", []):
        classification = region.setdefault("classification", {})
        llm_cls = normalize_class_label((classification.get("llm") or {}).get("class"))
        clip_cls = normalize_class_label((classification.get("clip") or {}).get("class"))
        final_cls = llm_cls or clip_cls
        if final_cls:
            classification["final"] = {"class": final_cls}

    split_dir = dataset_path / split
    sample_name = regions_json_path.parent.name
    image_size = regions_data.get("image_size", {})

    # 区域事实摘要（用于 caption 生成）
    region_facts = []
    for region in regions_data.get("regions", []):
        region_id = int(region.get("region_id", 0) or 0)
        if region_id <= 0:
            continue
        region_facts.append({
            "region_id": region_id,
            "location": region.get("location", ""),
            "class": resolve_region_class(region),
        })

    # 生成场景描述
    caption_data = {}
    if client is not None and region_facts:
        bg_region = next((r for r in regions_data.get("regions", []) if int(r.get("region_id", 0) or 0) == 0), None)
        if bg_region:
            files = bg_region.get("files") or {}
            img_path = (regions_json_path.parent.parent / str(files.get("image", "")).strip())
            if img_path.exists():
                caption_data = generate_caption(client, model_name, img_path, region_facts)

    record = {
        "image_name": regions_data.get("image_name", sample_name),
        "image_size": image_size,
        "image": find_image_path(split_dir, "image", sample_name),
        "label": find_image_path(split_dir, "label", sample_name),
        "regions": regions_data.get("regions", []),
        "_seg4_refined": refined_count,
    }
    if caption_data:
        record["caption"] = caption_data.get("caption", "")
        record["feature_analysis"] = caption_data.get("feature_analysis", "")
        record["summary"] = caption_data.get("summary", "")

    return record


def process_split(
    dataset_path: str,
    split: str,
    client: Optional[OpenAI],
    model_name: str,
    max_workers: int = 1,
) -> None:
    """
    入参:
        dataset_path (str): 数据集根目录
        split (str): 数据划分名称
        client: OpenAI 客户端
        model_name (str): 模型名称
        max_workers (int): 最大并行数
    方法:
        多线程处理所有样本，即时写入 metadata.json
    出参:
        None
    """
    dataset_root = Path(dataset_path)
    split_dir = dataset_root / split
    regions_root = split_dir / "regions"
    output_path = split_dir / "metadata.json"

    if not regions_root.exists():
        print(f"Missing: {regions_root}")
        return

    regions_json_files = sorted(glob.glob(str(regions_root / "*" / "regions.json")))
    if not regions_json_files:
        print(f"No regions.json in {regions_root}")
        return

    print(f"\n{'='*60}")
    print(f"SEG Caption + Metadata - {split}")
    print(f"Samples: {len(regions_json_files)}")
    print(f"{'='*60}\n")

    metadata: List[Dict[str, Any]] = []
    print_lock = threading.Lock()
    results_lock = threading.Lock()

    def flush():
        with results_lock:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_one, i, dataset_root, split, Path(p),
                client, model_name, print_lock, len(regions_json_files),
            ): p
            for i, p in enumerate(regions_json_files, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            with results_lock:
                result.pop("_seg4_refined", None)
                metadata.append(result)
            flush()

    metadata.sort(key=lambda item: str(item.get("image_name", "")))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_path} ({len(metadata)} samples)")


def _process_one(idx, dataset_root, split, rj_path, client, model_name, print_lock, total):
    """单样本处理封装"""
    sample_name = rj_path.parent.name
    with print_lock:
        print(f"[{idx}/{total}] {sample_name}")
    try:
        record = build_metadata_record(dataset_root, split, rj_path, client, model_name)
        with print_lock:
            n_regions = len([r for r in (record.get("regions", [])) if int(r.get("region_id", 0) or 0) > 0])
            print(f"  OK: {n_regions} regions")
        return record
    except Exception as e:
        with print_lock:
            print(f"  FAIL: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="SEG mode: Generate captions and build metadata")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test", "all"], help="Split")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--model", type=str, default="qwen-vl-plus", help="VLM model")
    parser.add_argument("--base_url", type=str,
                       default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                       help="API base URL")
    parser.add_argument("--max_workers", type=int, default=1, help="Workers")
    parser.add_argument("--batch_size", type=int, default=10, help="Kept for compatibility")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("未检测到 API Key，请设置 DASHSCOPE_API_KEY 或传入 --api_key。")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        split_dir = Path(args.dataset_path) / split
        if not split_dir.is_dir():
            continue
        process_split(args.dataset_path, split, client, args.model, args.max_workers)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
