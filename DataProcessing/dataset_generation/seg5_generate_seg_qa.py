#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEG QA 生成工具（SEG 模式 Step seg5）

功能：基于 metadata.json 调用 LLM 生成分割场景问答对
输入：{split}/metadata.json（seg4 输出）
输出：{split}/qa_{split}.json

QA 类型：
- qa (yes/no): 是否存在某类目标、某位置是否为某类
- mcq: 目标位置、目标类别、数量
- caption: 局部描述、整体描述

与 CD 模式的区别：
- 无变化检测 QA（CtW/CfW/CN 语义完全不同）
- QA 聚焦于场景理解：what/where/how many/describe
- 区域事实只有 class（无 t1_class/t2_class）
"""
import os
import json
import re
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import numpy as np
from openai import OpenAI


def get_message_content(message) -> str:
    """
    入参:
        message: OpenAI 返回的 message 对象
    方法:
        兼容推理模型的 reasoning 字段
    出参:
        str: 提取的文本内容
    """
    content = message.content or getattr(message, 'reasoning', None) or ""
    if content.startswith("Thinking Process"):
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Answer:") or line.startswith("回答:"):
                content = "\n".join(lines[i+1:]).strip()
                break
    return content


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


def normalize_qa_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    入参:
        result (Dict): 模型返回的原始 JSON
    方法:
        统一兼容 qa 列表 + mcq/caption 分栏结构
    出参:
        List[Dict]: 归一化后的问答列表
    """
    qa_items: List[Dict[str, Any]] = []

    primary_items = result.get("qa", [])
    if isinstance(primary_items, list):
        qa_items.extend(item for item in primary_items if isinstance(item, dict))

    for field_name, expected_type in (("mcq", "mcq"), ("caption", "caption")):
        field_items = result.get(field_name, [])
        if not isinstance(field_items, list):
            continue
        for item in field_items:
            if not isinstance(item, dict):
                continue
            item.setdefault("type", expected_type)
            qa_items.append(item)

    return qa_items


def normalize_mcq_options(options: Any) -> Dict[str, str]:
    """
    入参:
        options (Any): 模型返回的选项（dict/list/str）
    方法:
        统一转换为 A-D 字典
    出参:
        Dict[str, str]: 标准化选项
    """
    option_keys = ["A", "B", "C", "D"]

    if isinstance(options, dict):
        normalized = {str(k).strip().upper(): str(v).strip() for k, v in options.items() if str(v).strip()}
    elif isinstance(options, list):
        normalized = {}
        for idx, v in enumerate(options[:4]):
            text = str(v).strip()
            if text:
                normalized[option_keys[idx]] = text
    elif isinstance(options, str):
        normalized = {}
        for line in options.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip().upper()
            v = v.strip()
            if k in option_keys and v:
                normalized[k] = v
    else:
        normalized = {}

    missing = [k for k in option_keys if not normalized.get(k)]
    if missing:
        raise ValueError(f"MCQ missing options: {', '.join(missing)}")

    return {k: normalized[k] for k in option_keys}


def validate_qa_items(qa_items: List[Dict[str, Any]]) -> None:
    """
    入参:
        qa_items (List[Dict]): 问答列表
    方法:
        检查至少覆盖 qa/mcq/caption 三类，校验每条格式
    出参:
        None（校验失败抛 ValueError）
    """
    total = len([item for item in qa_items if isinstance(item, dict)])
    if total < 10:
        raise ValueError(f"Need at least 10 questions, got {total}")

    present_types = {str(item.get("type", "")).strip().lower() for item in qa_items if isinstance(item, dict)}
    missing = sorted({"qa", "mcq", "caption"} - present_types)
    if missing:
        raise ValueError(f"Missing types: {', '.join(missing)}")

    for item in qa_items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip().lower()
        answer = str(item.get("answer", "")).strip()

        if not str(item.get("question", "")).strip():
            raise ValueError("Empty question")

        if item_type == "qa":
            if answer.lower() not in {"yes", "no"}:
                raise ValueError("QA answer must be yes/no")
            item["answer"] = answer.lower()

        elif item_type == "mcq":
            item["options"] = normalize_mcq_options(item.get("options"))
            if answer.upper() not in {"A", "B", "C", "D"}:
                raise ValueError("MCQ answer must be A/B/C/D")
            item["answer"] = answer.upper()

        elif item_type == "caption":
            if not answer:
                raise ValueError("Caption answer cannot be empty")


def extract_region_facts(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    入参:
        sample (Dict): metadata.json 中的单条记录
    方法:
        从 regions 提取 region_id + location + class
    出参:
        List[Dict]: [{region_id, location, class}, ...]
    """
    items = []
    for region in sample.get("regions", []):
        region_id = int(region.get("region_id", 0) or 0)
        if region_id <= 0:
            continue

        classification = region.get("classification") or {}
        final_cls = ""
        for source in ("final", "llm", "clip"):
            payload = classification.get(source) or {}
            cls = str(payload.get("class", "")).strip().lower()
            if cls:
                final_cls = cls
                break

        location = str(region.get("location", "")).strip().lower()
        if not location or not final_cls:
            continue

        items.append({
            "region_id": region_id,
            "location": location,
            "class": final_cls,
        })

    return items


def build_multimodal_user_content(
    sample_dir: Path,
    input_json: str,
) -> List[Dict[str, Any]]:
    """
    入参:
        sample_dir (Path): 当前样本的 regions 目录
        input_json (str): 结构化区域事实 JSON
    方法:
        构建 image_bbox.png + label_bbox.png + 区域事实的多模态请求
    出参:
        List[Dict]: OpenAI 多模态 user content 列表
    """
    content = [
        {
            "type": "text",
            "text": (
                "Generate a Q&A dataset for this segmentation sample.\n"
                "The numbered polygons correspond to region_id values in the JSON.\n"
                "Use the bbox images for visual grounding.\n"
                "Use the structured JSON as the only authority for location and class.\n"
                "Do not mention region ids in questions or answers.\n"
                "Do not generate change-related questions (this is segmentation, not change detection)."
            ),
        }
    ]

    for image_name, desc in (
        ("image_bbox.png", "Image with numbered region polygons:"),
        ("label_bbox.png", "Label image with numbered regions:"),
    ):
        image_path = sample_dir / image_name
        if image_path.exists():
            content.append({"type": "text", "text": desc})
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_image_to_base64(image_path)},
            })

    content.append({
        "type": "text",
        "text": f"Region facts:\n```json\n{input_json}\n```",
    })

    return content


def generate_qa_with_llm(
    client: OpenAI,
    model_name: str,
    sample: Dict[str, Any],
    split_dir: Path,
) -> Dict[str, Any]:
    """
    入参:
        client: OpenAI 客户端
        model_name (str): 模型名称
        sample (Dict): metadata 单条记录
        split_dir (Path): split 目录
    方法:
        1. 加载 prompt 文件 + bbox 图
        2. 调用 LLM 生成 QA，最多重试 3 次
        3. 失败时发送 repair prompt
    出参:
        Dict: QA 记录
    """
    script_dir = Path(__file__).parent
    prompt_path = script_dir / "prompt_seg_qa.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    region_facts = extract_region_facts(sample)
    num_regions = len(region_facts)

    # 预计算 per-class count，确保计数问题答案确定性
    from collections import Counter
    class_counts = dict(Counter(r["class"] for r in region_facts))

    input_data = {
        "image": sample.get("image", ""),
        "label": sample.get("label", ""),
        "regions": region_facts,
        "num_regions": num_regions,
        "class_counts": class_counts,
    }

    input_json = json.dumps(input_data, ensure_ascii=False, indent=2)
    sample_name = str(sample.get("image_name", "")).strip()
    sample_dir = split_dir / "regions" / sample_name
    human_content = build_multimodal_user_content(sample_dir, input_json)

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_content},
    ]
    messages = list(base_messages)
    result: Optional[Dict[str, Any]] = None
    qa_pairs: List[Dict[str, Any]] = []

    image_name = sample.get("image_name", "")

    for attempt in range(3):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=2048,
            extra_body={"disable_search": True},
        )

        json_string = get_message_content(response.choices[0].message).strip()
        if not json_string:
            error_message = "Empty response"
        else:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_string, re.DOTALL)
            if match:
                json_string = match.group(1)
            else:
                first_brace = json_string.find("{")
                last_brace = json_string.rfind("}")
                if first_brace != -1 and last_brace != -1:
                    json_string = json_string[first_brace:last_brace + 1]

            try:
                result = json.loads(json_string)
                result.setdefault("image", sample.get("image", ""))
                result.setdefault("label", sample.get("label", ""))

                qa_pairs = normalize_qa_items(result)
                validate_qa_items(qa_pairs)
                result["qa"] = qa_pairs

                result.pop("mcq", None)
                result.pop("caption", None)
                break
            except Exception as exc:
                error_message = str(exc)

        if attempt == 2:
            raise ValueError(error_message)

        repair_prompt = (
            "Your previous JSON failed validation.\n"
            f"Error: {error_message}\n"
            "Regenerate the full JSON. Generate at least 10 questions:\n"
            "- 3+ QA (yes/no): about object presence, location, and urban scene\n"
            "- 4+ MCQ: about location, class, count, and scene_type\n"
            "- 3+ Caption: local, scene_understanding, and overall descriptions\n"
            "Return raw JSON only."
        )
        messages = base_messages + [
            {"role": "assistant", "content": json_string},
            {"role": "user", "content": repair_prompt},
        ]

    if result is None:
        raise ValueError("LLM failed to produce valid QA JSON")

    for i, pair in enumerate(qa_pairs, 1):
        if isinstance(pair, dict) and not pair.get("id"):
            pair["id"] = f"{image_name}_q{i:03d}"

    return result


def process_single_sample(
    idx: int, total: int, sample: Dict[str, Any],
    split_dir: Path, client: OpenAI, model_name: str,
    print_lock: threading.Lock,
) -> Dict[str, Any]:
    """
    入参:
        idx, total: 样本序号/总数
        sample: metadata 单条记录
        split_dir: split 目录
        client: OpenAI 客户端
        model_name: 模型名称
        print_lock: 打印锁
    方法:
        提取区域事实，调用 LLM 生成 QA
    出参:
        Dict: 处理结果
    """
    image_name = sample.get("image_name", "")
    region_facts = extract_region_facts(sample)

    with print_lock:
        print(f"[{idx}/{total}] {image_name}: {len(region_facts)} regions")

    if not region_facts:
        with print_lock:
            print(f"   FAIL {image_name}: no valid region facts")
        return {"image_name": image_name, "qa": None, "error": "no facts", "success": False}

    try:
        qa_result = generate_qa_with_llm(client, model_name, sample, split_dir)
        num_q = len(qa_result.get("qa", []))
        with print_lock:
            print(f"   OK {image_name}: {num_q} Q&A")
        return {"image_name": image_name, "qa": qa_result, "success": True, "qa_pairs": num_q}
    except Exception as e:
        with print_lock:
            print(f"   FAIL {image_name}: {e}")
        return {"image_name": image_name, "qa": None, "error": str(e), "success": False}


def process_split(
    dataset_path: str, split: str,
    client: OpenAI, model_name: str, max_workers: int = 4,
) -> None:
    """
    入参:
        dataset_path (str): 数据集根目录
        split (str): 数据集划分
        client: OpenAI 客户端
        model_name (str): 模型名称
        max_workers (int): 最大并行数
    方法:
        读取 metadata.json，多线程生成 QA，即时写入 qa_{split}.json
    出参:
        None
    """
    split_dir = Path(dataset_path) / split
    metadata_path = split_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"Error: metadata.json not found: {metadata_path}")
        print("Please run seg4 first.")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    output_path = split_dir / f"qa_{split}.json"
    print(f"Loaded {len(metadata)} samples")

    qa_results: List[Dict[str, Any]] = []
    success_count = 0
    total_qa_pairs = 0
    print_lock = threading.Lock()
    results_lock = threading.Lock()

    def flush():
        with results_lock:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(qa_results, f, ensure_ascii=False, indent=2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_sample,
                i, len(metadata), sample, split_dir, client, model_name, print_lock,
            ): i
            for i, sample in enumerate(metadata, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            with results_lock:
                qa_results.append({
                    "image_name": result["image_name"],
                    "qa": result.get("qa"),
                    "error": result.get("error"),
                })
            if result["success"]:
                success_count += 1
                total_qa_pairs += result.get("qa_pairs", 0)
            flush()

    print(f"\n{'='*60}")
    print(f"SEG QA Complete - {split}")
    print(f"Success: {success_count}/{len(qa_results)}, Pairs: {total_qa_pairs}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SEG mode: Generate QA dataset (Step seg5)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test", "all"], help="Split")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--model", type=str, default="qwen-vl-plus", help="VLM model")
    parser.add_argument("--base_url", type=str,
                       default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                       help="API base URL")
    parser.add_argument("--max_workers", type=int, default=1, help="Workers")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    if args.split == "all":
        for split in ["train", "val", "test"]:
            process_split(args.dataset_path, split, client, args.model, args.max_workers)
    else:
        process_split(args.dataset_path, args.split, client, args.model, args.max_workers)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
