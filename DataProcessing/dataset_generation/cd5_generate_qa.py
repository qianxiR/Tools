#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问答对生成工具

功能：基于 metadata.json 调用 LLM 生成变化检测问答对
输出：{split}/qa_{split}.json



"""
import os
import json
import re
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import numpy as np
from openai import OpenAI


def get_message_content(message) -> str:
    """
    获取消息内容，兼容推理模型的 reasoning 字段

    入参:
        message: OpenAI 返回的 message 对象

    出参:
        str: 提取的文本内容
    """
    content = message.content or getattr(message, 'reasoning', None) or ""

    # 处理推理模型的格式，提取实际回答
    if content.startswith("Thinking Process"):
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Answer:") or line.startswith("回答:"):
                content = "\n".join(lines[i+1:]).strip()
                break

    return content


def normalize_qa_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    入参:
        result (Dict[str, Any]): 模型返回的原始 JSON 对象

    方法:
        1. 统一兼容顶层 `qa` 列表输出，以及少数模型返回 `qa`/`mcq`/`caption` 分栏结构的情况
        2. 仅保留字典类型问答项，避免异常结构进入下游流程
        3. 按原始顺序拼接，保证问题类型覆盖同时尽量保留模型生成顺序

    出参:
        List[Dict[str, Any]]: 归一化后的问答列表
    """
    qa_items: List[Dict[str, Any]] = []

    # 优先使用标准结构，减少对正常返回结果的额外干预。
    primary_items = result.get("qa", [])
    if isinstance(primary_items, list):
        qa_items.extend(item for item in primary_items if isinstance(item, dict))

    # 兼容按类型拆分返回的情况，避免提示词更严格后模型切换输出风格导致流程中断。
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


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """
    入参:
        path (Path): 本地图像路径

    方法:
        使用 np.fromfile + cv2.imdecode 读取图像，保证中文路径下也能稳定打开 bbox 标注图。

    出参:
        Optional[np.ndarray]: 成功时返回图像数组，失败时返回 None
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
        image_path (Path): 本地图像路径
        resize_max_dim (int): 图像最长边限制，默认 768

    方法:
        1. 优先读取本地图像并按最长边缩放，控制多模态请求体积
        2. 统一编码为 JPEG data URL，兼顾兼容性与上传开销
        3. 若图像无法解码，则回退到原始二进制文件直传，避免因编码失败中断 QA 生成

    出参:
        str: 可直接传给多模态接口的 data URL
    """
    image = imread_unicode(image_path)
    if image is not None:
        height, width = image.shape[:2]
        if max(height, width) > resize_max_dim:
            scale = resize_max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError(f"Failed to encode image: {image_path}")
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime_type = mime_map.get(suffix, "image/png")
    return f"data:{mime_type};base64,{encoded}"


def normalize_mcq_options(options: Any) -> Dict[str, str]:
    """
    入参:
        options (Any): 模型返回的原始选项结构，可能为 dict / list / str

    方法:
        1. 将选择题选项统一转换为 A-D 字典，减少模型输出风格差异对下游的影响
        2. 仅保留非空文本，保证评测侧能稳定读取标准化选项
        3. 若无法构成完整四选项，则直接报错，防止无效 MCQ 样本进入数据集

    出参:
        Dict[str, str]: 标准化后的 A-D 选项字典
    """
    option_keys = ["A", "B", "C", "D"]

    if isinstance(options, dict):
        normalized = {
            str(key).strip().upper(): str(value).strip()
            for key, value in options.items()
            if str(value).strip()
        }
    elif isinstance(options, list):
        normalized = {}
        for idx, value in enumerate(options[:4]):
            text = str(value).strip()
            if text:
                normalized[option_keys[idx]] = text
    elif isinstance(options, str):
        normalized = {}
        lines = [line.strip() for line in options.splitlines() if line.strip()]
        for line in lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            if key in option_keys and value:
                normalized[key] = value
    else:
        normalized = {}

    missing = [key for key in option_keys if not normalized.get(key)]
    if missing:
        raise ValueError(f"MCQ options must contain A-D, missing: {', '.join(missing)}")

    return {key: normalized[key] for key in option_keys}


def classify_mcq_question(question: str) -> str:
    """
    入参:
        question (str): MCQ 题干文本

    方法:
        将 MCQ 题型收敛为四类：方位问题、变化问题、方位变化问题、变化类型问题

    出参:
        str: `location` / `change` / `location_change` / `change_type`
    """
    clean = str(question).strip().lower()

    if "type of urban change" in clean or "type of change" in clean:
        return "change_type"

    if "location-change pair" in clean or "both location and change" in clean or "matches both location and change" in clean:
        return "location_change"

    if clean.startswith("where "):
        return "location"

    if clean.startswith("which change") or clean.startswith("what change") or "change happened" in clean or "land cover change" in clean:
        return "change"

    raise ValueError(f"Unsupported MCQ question style: {question}")


def classify_qa_question(question: str) -> str:
    """
    入参:
        question (str): QA 题干文本

    方法:
        将 yes/no 问题映射到 CN、CtW、CfW 及扩展槽位（urban expansion 等）。

    出参:
        str: `cn` / `ctw` / `cfw` / `urban`
    """
    clean = str(question).strip().lower()

    if clean in {"did any visible change occur?", "did any visible change occur"}:
        return "cn"

    if clean.startswith("did any region become "):
        return "ctw"

    if clean.startswith("did any changed region start as "):
        return "cfw"

    # 城市变化理解扩展 QA
    if "construction" in clean or "urban" in clean or "expansion" in clean or "demolition" in clean:
        return "urban"

    raise ValueError(f"Unsupported QA question style: {question}")


def classify_caption_question(question: str) -> str:
    """
    入参:
        question (str): Caption 题干文本

    方法:
        将 caption 分为五类：Level1(城市概况)/Level2(变化类型)/Level3(分割级)/local/overall

    出参:
        str: `level1_global` / `level2_change_type` / `level3_mask` / `local` / `overall`
    """
    clean = str(question).strip().lower()

    # Level 1: Global urban description
    if "urban change pattern" in clean or "overall urban" in clean:
        return "level1_global"

    # Level 2: Change type significance
    if "type of urban change" in clean or "what does it signify" in clean or "what type of" in clean and "change" in clean:
        return "level2_change_type"

    # Level 3: Mask-based detailed description
    if "specific change region" in clean or "change regions and their" in clean or "characteristics" in clean and "region" in clean:
        return "level3_mask"

    # Local description (含具体方位词)
    location_keywords = (
        "north", "south", "east", "west",
        "northeast", "northwest", "southeast", "southwest",
        "center", "central", "middle",
        "upper", "lower", "left", "right",
    )
    if any(kw in clean for kw in location_keywords):
        return "local"

    # Overall summary
    if any(kw in clean for kw in ("overall", "entire", "whole", "all change", "pattern", "summary", "comprehensive")):
        return "overall"

    return "local"


def resolve_region_classes(region: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    入参:
        region (Dict[str, Any]): 单个 region 结构

    方法:
        1. 优先使用 `classification.final`，因为它通常代表最终确认结果
        2. 若 final 不存在，则回退到 `classification.llm`，再回退到 `classification.clip`
        3. 同时返回来源标签，便于在提示词中说明当前使用的是哪一路分类结果

    出参:
        Tuple[str, str, str]: (t1_class, t2_class, source_name)
    """
    classification = region.get("classification") or {}
    candidates = [
        ("final", classification.get("final") or {}),
        ("llm", classification.get("llm") or {}),
        ("clip", classification.get("clip") or {}),
    ]

    for source_name, payload in candidates:
        t1_class = str(payload.get("t1_class", "")).strip().lower()
        t2_class = str(payload.get("t2_class", "")).strip().lower()
        if t1_class and t2_class:
            return t1_class, t2_class, source_name

    t1_class = str(region.get("t1_class", "")).strip().lower()
    t2_class = str(region.get("t2_class", "")).strip().lower()
    if t1_class and t2_class:
        return t1_class, t2_class, "region"

    return "", "", ""


def ensure_required_types(qa_items: List[Dict[str, Any]]) -> None:
    """
    入参:
        qa_items (List[Dict[str, Any]]): 归一化后的问答列表

    方法:
        检查结果是否覆盖 `qa`、`mcq`、`caption` 三类问题。
        同时校验每类题型的 answer_format 与答案格式，并保证先完成核心 8 题覆盖，再允许扩展题追加。

    出参:
        None
    """
    present_types = {
        str(item.get("type", "")).strip().lower()
        for item in qa_items
        if isinstance(item, dict)
    }
    total_items = len([item for item in qa_items if isinstance(item, dict)])
    if total_items < 12:
        raise ValueError(f"Q&A count must be at least 12, got {total_items}")

    required_types = {"qa", "mcq", "caption"}
    missing_types = sorted(required_types - present_types)
    if missing_types:
        raise ValueError(f"Missing required QA types: {', '.join(missing_types)}")

    type_counts = {"qa": 0, "mcq": 0, "caption": 0}

    for item in qa_items:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).strip().lower()
        answer_format = str(item.get("answer_format", "")).strip().lower()
        answer = str(item.get("answer", "")).strip()
        if item_type in type_counts:
            type_counts[item_type] += 1

        if not str(item.get("question", "")).strip():
            raise ValueError("Question text cannot be empty")

        if item_type == "qa":
            if answer_format != "yes_no":
                raise ValueError("QA items must use answer_format='yes_no'")
            normalized_answer = answer.lower()
            if normalized_answer not in {"yes", "no"}:
                raise ValueError("QA answers must be exactly 'yes' or 'no'")
            item["answer"] = normalized_answer

        elif item_type == "mcq":
            if answer_format != "mcq":
                raise ValueError("MCQ items must use answer_format='mcq'")
            item["options"] = normalize_mcq_options(item.get("options"))
            normalized_answer = answer.upper()
            if normalized_answer not in {"A", "B", "C", "D"}:
                raise ValueError("MCQ answers must be exactly one of A/B/C/D")
            item["answer"] = normalized_answer

        elif item_type == "caption":
            if answer_format != "text":
                raise ValueError("Caption items must use answer_format='text'")
            if not answer:
                raise ValueError("Caption answer cannot be empty")


def extract_region_change_items(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    入参:
        sample (Dict[str, Any]): metadata.json 中的单条记录

    方法:
        从 regions 列表中提取可用于 QA 生成的结构化变化事实，保留：
        1. region_id：与 bbox 标注图中的编号对齐
        2. location：区域空间方位
        3. t1_class / t2_class：现有分类字段中的类别结果
        4. class_source：当前类别来自 final / llm / clip / region 哪一路
        这样做是为了让 Step 5 同时利用已有分类结果与带编号的 bbox 图，而不是只依赖纯文本描述。

    出参:
        List[Dict[str, Any]]: 归一化后的区域变化列表
    """
    items: List[Dict[str, Any]] = []

    for region in sample.get("regions", []):
        region_id = int(region.get("region_id", 0) or 0)
        if region_id <= 0:
            continue

        t1_class, t2_class, class_source = resolve_region_classes(region)
        location = str(region.get("location", "")).strip().lower()

        if not location or not t1_class or not t2_class:
            continue

        items.append({
            "region_id": region_id,
            "location": location,
            "t1_class": t1_class,
            "t2_class": t2_class,
            "class_source": class_source,
        })

    return items


def build_multimodal_user_content(
    sample_dir: Path,
    input_json: str,
) -> List[Dict[str, Any]]:
    """
    入参:
        sample_dir (Path): 当前样本的 regions 目录
        input_json (str): 发给模型的结构化变化 JSON

    方法:
        1. 尝试附加 t1/t2/label 三张带编号 bbox 图，让模型理解区域的空间布局与局部外观
        2. 使用文本说明明确 region_id 与图中编号一一对应，避免模型无法对齐结构化字段与图像
        3. 将结构化分类结果一并发送，作为类别名称的权威来源

    出参:
        List[Dict[str, Any]]: OpenAI 多模态 user content 列表
    """
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Generate a Q&A dataset for this change detection sample.\n"
                "The numbered polygons in the images correspond to region_id values in the structured JSON.\n"
                "Use the bbox images to understand visible layout and local changes.\n"
                "Use the structured JSON as the only authority for location, t1_class, and t2_class.\n"
                "Do not revise, correct, replace, or invent any location or class from the images.\n"
                "The images are only for visual grounding and local appearance understanding.\n"
                "Do not generate MCQ questions about counting changed areas.\n"
                "If you generate counting questions, you MUST use t1_class_counts / t2_class_counts as the authoritative source.\n"
                "Do not mention region ids or numbered regions in the final questions or answers."
            ),
        }
    ]

    for image_name, description in (
        ("t1_bbox.png", "T1 image with numbered change regions:"),
        ("t2_bbox.png", "T2 image with numbered change regions:"),
        ("label_bbox.png", "Label image with numbered change regions:"),
    ):
        image_path = sample_dir / image_name
        if not image_path.exists():
            continue
        content.append({"type": "text", "text": description})
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_to_base64(image_path)},
        })

    content.append({
        "type": "text",
        "text": f"Structured change facts:\n```json\n{input_json}\n```",
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
        model_name: 模型名称
        sample (Dict): metadata.json 中的单条记录（含 image_name, regions 等）
        split_dir (Path): 当前 split 目录

    方法:
        1. 加载提示词文件与 bbox 图，构建多模态请求
        2. 调用 LLM 生成 QA JSON，最多重试 3 次
        3. 失败时拼接 repair prompt 引导模型修正

    出参:
        Dict: QA 数据集记录（含 t1/t2/label/qa 列表）
    """
    script_dir = Path(__file__).parent
    prompt_candidates = [
        script_dir / "prompt_cd_qa.md",
    ]
    prompt_file = next((p for p in prompt_candidates if p.exists()), None)
    if prompt_file is None:
        searched = "\n".join([f"- {p}" for p in prompt_candidates])
        raise FileNotFoundError(f"Prompt file not found. Searched:\n{searched}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    region_changes = extract_region_change_items(sample)
    num_regions = len(region_changes)

    # 预计算 per-class count，确保计数问题答案确定性
    from collections import Counter
    t1_class_counts = dict(Counter(r["t1_class"] for r in region_changes))
    t2_class_counts = dict(Counter(r["t2_class"] for r in region_changes))

    input_data = {
        "t1": sample.get("t1", ""),
        "t2": sample.get("t2", ""),
        "label": sample.get("label", ""),
        "regions": region_changes,
        "num_regions": num_regions,
        "t1_class_counts": t1_class_counts,
        "t2_class_counts": t2_class_counts,
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
            extra_body={"disable_search": True}
        )

        json_string = get_message_content(response.choices[0].message).strip()
        if not json_string:
            error_message = "Model returned empty content"
        else:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_string, re.DOTALL)
            if match:
                json_string = match.group(1)
            else:
                # 尝试直接找到第一个 { 到最后一个 } 的 JSON 对象
                first_brace = json_string.find("{")
                last_brace = json_string.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_string = json_string[first_brace:last_brace + 1]

            try:
                result = json.loads(json_string)
                result.setdefault("t1", sample.get("t1", ""))
                result.setdefault("t2", sample.get("t2", ""))
                result.setdefault("label", sample.get("label", ""))

                qa_pairs = normalize_qa_items(result)
                ensure_required_types(qa_pairs)
                result["qa"] = qa_pairs

                # 删除兼容性中间字段，统一下游消费结构，避免重复读取。
                result.pop("mcq", None)
                result.pop("caption", None)
                break
            except Exception as exc:
                error_message = str(exc)

        if attempt == 2:
            raise ValueError(error_message)

        repair_prompt = (
            "Your previous JSON failed validation.\n"
            f"Validation error: {error_message}\n"
            "Regenerate the full JSON from scratch and follow every rule exactly.\n"
            "Use a two-stage generation flow.\n"
            "Stage 1: first generate the core 12 questions.\n"
            "- 3 QA yes/no questions covering CN, CtW, and CfW\n"
            "- 4 MCQ questions covering location, change, location_change, and change_type\n"
            "- 5 caption questions covering Level1 global, Level2 change type, Level3 mask-based, local, and overall\n"
            "Stage 2: after the core 12, append a few extra high-value non-duplicate questions if more facts are available.\n"
            "So the final total must be at least 12, not capped at 12.\n"
            "Use these QA templates exactly:\n"
            '- CN: "Did any visible change occur?"\n'
            '- CtW: "Did any region become {class}?"\n'
            '- CfW: "Did any changed region start as {class}?"\n'
            '- change_type MCQ: "What type of urban change is shown in this image pair?"\n'
            "Return raw JSON only."
        )
        messages = base_messages + [
            {"role": "assistant", "content": json_string if 'json_string' in locals() else ""},
            {"role": "user", "content": repair_prompt},
        ]

    if result is None:
        raise ValueError("Model failed to produce valid QA JSON")

    image_name = sample.get("image_name", "")
    for i, pair in enumerate(qa_pairs, 1):
        if not isinstance(pair, dict):
            continue
        if pair.get("id"):
            continue
        pair["id"] = f"{image_name}_q{i:03d}"

    return result


def process_single_sample(
    idx: int,
    total: int,
    sample: Dict[str, Any],
    split_dir: Path,
    client: OpenAI,
    model_name: str,
    print_lock: threading.Lock,
) -> Dict[str, Any]:
    """
    入参:
        idx (int): 样本序号
        total (int): 总样本数
        sample (Dict): metadata 中的单条记录
        split_dir (Path): 当前 split 目录
        client: OpenAI 客户端
        model_name (str): 模型名称
        print_lock (threading.Lock): 打印锁

    方法:
        提取区域变化事实，若无有效区域则直接标记失败；否则调用 LLM 生成 QA 并返回结果

    出参:
        Dict: 处理结果（含 image_name, qa, success, qa_pairs 等）
    """
    image_name = sample.get("image_name", "")

    region_changes = extract_region_change_items(sample)

    with print_lock:
        print(f"[{idx}/{total}] {image_name}: {len(region_changes)} regions")

    if not region_changes:
        with print_lock:
            print(f"   FAIL {image_name}: no valid region change facts")
        return {
            "image_name": image_name,
            "qa": None,
            "error": "no valid region change facts",
            "success": False,
            "qa_pairs": 0,
        }

    try:
        num_regions = len(region_changes)
        qa_result = generate_qa_with_llm(client, model_name, sample, split_dir)

        num_q = len(qa_result.get("qa", []))
        with print_lock:
            print(f"   OK {image_name}: {num_q} Q&A (regions={num_regions})")

        return {
            "image_name": image_name,
            "qa": qa_result,
            "success": True,
            "qa_pairs": num_q,
        }

    except Exception as e:
        with print_lock:
            print(f"   FAIL {image_name}: {e}")
        return {
            "image_name": image_name,
            "qa": None,
            "error": str(e),
            "success": False,
            "qa_pairs": 0,
        }


def process_split(
    dataset_path: str,
    split: str,
    client: OpenAI,
    model_name: str,
    max_workers: int = 4,
) -> None:
    """
    入参:
        dataset_path (str): 数据集根目录
        split (str): 数据集划分 (train/val/test)
        client: OpenAI 客户端
        model_name (str): 模型名称
        max_workers (int): 最大并行数

    方法:
        读取 metadata.json，多线程调用 LLM 生成 QA，每条结果即时追加写入 qa_{split}.json

    出参:
        None（结果保存到文件）
    """
    split_dir = Path(dataset_path) / split
    metadata_path = split_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"Error: metadata.json not found: {metadata_path}")
        print("Please run Step 4 first.")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    output_path = split_dir / f"qa_{split}.json"
    print(f"Loaded {len(metadata)} samples")
    print(f"Parallel workers: {max_workers}")

    qa_results: List[Dict[str, Any]] = []
    success_count = 0
    total_qa_pairs = 0
    print_lock = threading.Lock()
    results_lock = threading.Lock()

    def _flush_results():
        with results_lock:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(qa_results, f, ensure_ascii=False, indent=2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_sample,
                i, len(metadata), sample, split_dir, client, model_name, print_lock
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
                total_qa_pairs += result["qa_pairs"]
            _flush_results()

    print(f"\n{'='*60}")
    print(f"QA Complete - {split}")
    print(f"{'='*60}")
    print(f"Saved: {output_path}")
    print(f"Success: {success_count}/{len(qa_results)}, Pairs: {total_qa_pairs}")
    print(f"{'='*60}")


def main():
    """
    入参:
        无（从命令行读取参数）

    方法:
        解析命令行参数（dataset_path / split / model / base_url / max_workers），调用 process_split 生成 QA 数据集

    出参:
        int: 0 成功，1 失败
    """
    parser = argparse.ArgumentParser(description="Generate Q&A dataset from metadata.json (Step 5)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"], help="Dataset split")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API Key（DashScope API Key）")
    parser.add_argument("--model", type=str, default="qwen-vl-plus",
                        help="VLM model name")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API base URL")
    parser.add_argument("--max_workers", type=int, default=1, help="Max parallel workers")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"Model: {args.model}, API: {args.base_url}")

    if args.split == "all":
        for split in ["train", "val", "test"]:
            print(f"\n{'=' * 60}")
            process_split(args.dataset_path, split, client, args.model, args.max_workers)
    else:
        process_split(args.dataset_path, args.split, client, args.model, args.max_workers)

    print(f"\n{'='*60}")
    print("Done!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
