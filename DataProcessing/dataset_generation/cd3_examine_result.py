#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM 分类工具（Step 3）

功能：调用 VLM 对变化区域进行分类并打分
输出：regions.json 新增 classification.llm 字段

用法:
    python -m Tools.DataProcessing.dataset_generation.main --mode cd --dataset_path "E:/xzkjxm/dataes/WHU-CD_20" --split test --step cd3
"""
import os
import json
import re
import glob
import base64
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import numpy as np
from openai import OpenAI

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'


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


VALID_CLASSES = {
    "building", "highway", "vegetation", "farmland", "bare_land", "water",
}

CLASS_ALIASES = {
    "bareland": "bare_land", "barren_land": "bare_land", "barrenland": "bare_land",
    "desert": "bare_land",
    "road": "highway", "roads": "highway",
}


def normalize_class_label(value: Any) -> str:
    """
    规范化类别标签

    入参:
        value: 原始类别值

    出参:
        str: 规范化后的类别名（小写英文）
    """
    if value is None:
        return ""
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    label = CLASS_ALIASES.get(label, label)
    return label


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """
    支持中文路径的图片读取

    入参:
        path: 图片路径

    出参:
        Optional[np.ndarray]: 图片数据或 None
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def encode_image_to_base64(image_path: Path, resize_max_dim: int = 768) -> str:
    """
    编码图像为 base64

    入参:
        image_path: 图像路径
        resize_max_dim: 最大边长，超过则压缩

    出参:
        str: base64 编码的 data URL
    """
    ext = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".tif": "image/tiff", ".tiff": "image/tiff"
    }
    mime_type = mime_map.get(ext, "image/png")

    img = imread_unicode(image_path)
    if img is not None:
        h, w = img.shape[:2]
        if max(h, w) > resize_max_dim:
            scale = resize_max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        encoded = base64.b64encode(buffer).decode("utf-8")
        mime_type = "image/jpeg"
    else:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def load_json(path: Path) -> Dict[str, Any]:
    """
    加载 JSON 文件

    入参:
        path: 文件路径

    出参:
        Dict: JSON 数据
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """
    保存 JSON 文件

    入参:
        path: 文件路径
        data: 数据
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_system_prompt() -> str:
    """
    加载系统提示词

    出参:
        str: 系统提示词内容
    """
    script_dir = Path(__file__).parent
    prompt_path = script_dir / "prompt_cd_examine_result.md"
    return prompt_path.read_text(encoding="utf-8").strip()


def extract_json_payload(response_text: str) -> Any:
    """
    从响应中提取 JSON 数据（兼容截断响应）

    入参:
        response_text: VLM 响应文本

    出参:
        Any: 解析后的 JSON 数据

    异常:
        ValueError: 无法解析 JSON 时抛出
    """
    text = response_text.strip()

    # 优先匹配完整的 markdown 代码块（含闭合 ```）
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # 处理截断响应：剥离开头的 ```json 但没有闭合 ```
        truncated_match = re.match(r"```(?:json)?\s*(.*)", text, re.DOTALL)
        if truncated_match:
            text = truncated_match.group(1).strip()

    # 直接解析完整 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试包裹数组
    try:
        text_clean = text.strip("[]").strip()
        text_clean = "[" + text_clean + "]"
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    # 尝试直接解析
    if text.strip().startswith("["):
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

    # 截断 JSON 数组修复：找到最后一个完整的 JSON 对象，截断并闭合数组
    if "[" in text:
        last_obj_end = text.rfind("}")
        if last_obj_end > 0:
            truncated = text[:last_obj_end + 1] + "]"
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"无法解析 JSON: {text[:200]}")


def call_vlm_classify(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    t1_bbox_path: Path,
    t2_bbox_path: Path,
) -> List[Dict[str, Any]]:
    """
    调用 VLM 进行分类（已废弃，保留兼容性）

    入参:
        client: OpenAI 客户端
        model_name: 模型名称
        system_prompt: 系统提示词
        t1_bbox_path: T1 带边界框图像路径
        t2_bbox_path: T2 带边界框图像路径

    出参:
        List[Dict[str, Any]]: 分类结果列表
    """
    content_parts = []

    # 添加 T1 图像
    if t1_bbox_path.exists():
        content_parts.append({
            "type": "text",
            "text": "T1 image with numbered region bboxes (red numbers):"
        })
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": encode_image_to_base64(t1_bbox_path)}
        })

    # 添加 T2 图像
    if t2_bbox_path.exists():
        content_parts.append({
            "type": "text",
            "text": "T2 image with numbered region bboxes (red numbers):"
        })
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": encode_image_to_base64(t2_bbox_path)}
        })

    # 添加指令
    content_parts.append({
        "type": "text",
        "text": "\nPlease classify all numbered regions (region_id > 0) according to the Category System."
    })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
        extra_body={"disable_search": True}
    )

    response_text = get_message_content(response.choices[0].message)
    result = extract_json_payload(response_text)

    if isinstance(result, dict):
        result = result.get("regions", result.get("final_output", []))

    if not isinstance(result, list):
        raise ValueError(f"VLM 返回格式错误，期望 JSON 数组: {type(result)}")

    return result


def call_vlm_classify_batch(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    samples: List[Dict[str, Any]],
    batch_size: int = 10,
) -> List[Tuple[str, Optional[str], List[Dict[str, Any]]]]:
    """
    批量调用 VLM 进行分类（优化版：复用对话上下文）

    入参:
        client: OpenAI 客户端
        model_name: 模型名称
        system_prompt: 系统提示词
        samples: 样本列表，每项包含 {sample_name, t1_bbox_path, t2_bbox_path, valid_ids}
        batch_size: 每批次处理的样本数（默认10，之后重开对话）

    出参:
        List[Tuple[str, Optional[str], List[Dict[str, Any]]]]:
        返回 [(sample_name, error_message, results), ...]。
    """
    results = []
    # 分批处理
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start + batch_size]
        total_in_batch = len(batch)

        # 初始化对话（系统提示只设置一次）
        messages = [{"role": "system", "content": system_prompt}]

        for idx, sample in enumerate(batch, 1):
            sample_name = sample["sample_name"]
            print(f"  Processing [{idx}/{total_in_batch}]: {sample_name}...", end="", flush=True)
            t1_bbox_path = sample["t1_bbox_path"]
            t2_bbox_path = sample["t2_bbox_path"]
            valid_ids = sample["valid_ids"]

            # 构建用户消息（仅图片，无额外文字）
            content_parts = []

            if t1_bbox_path.exists():
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": encode_image_to_base64(t1_bbox_path)}
                })

            if t2_bbox_path.exists():
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": encode_image_to_base64(t2_bbox_path)}
                })

            # 添加到对话历史
            messages.append({"role": "user", "content": content_parts})

            # 调用 API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=4096,
                extra_body={"disable_search": True}
            )

            response_text = get_message_content(response.choices[0].message)
            result = extract_json_payload(response_text)

            # 解析结果
            error_message: Optional[str] = None
            classifications = []

            if isinstance(result, dict):
                result = result.get("regions", result.get("final_output", []))
                if isinstance(result, list):
                    classifications = result
                else:
                    error_message = "Invalid response format"
            elif isinstance(result, list):
                classifications = result
            else:
                error_message = "Invalid response format"

            results.append((sample_name, error_message, classifications))

            # 完成标记
            status = f"FAIL ({error_message})" if error_message else f"OK ({len(classifications)} regions)"
            print(f" [90m→[0m {status}")

            # 将助手回复添加到对话历史（保持上下文）
            messages.append({"role": "assistant", "content": response_text})

    return results


def normalize_llm_result(
    raw: List[Dict[str, Any]],
    valid_ids: set,
) -> List[Dict[str, Any]]:
    """
    规范化 LLM 分类结果

    入参:
        raw: LLM 原始结果
        valid_ids: 有效的 region_id 集合

    出参:
        List[Dict]: 规范化后的结果
    """
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        # 提取 region_id
        rid_raw = item.get("region_id")
        rid = rid_raw if isinstance(rid_raw, int) else None
        if rid is None and rid_raw:
            m = re.search(r"\d+", str(rid_raw))
            if m:
                rid = int(m.group(0))

        if rid is None or rid not in valid_ids:
            continue

        # 规范化类别
        t1_class = normalize_class_label(item.get("t1_class"))
        t2_class = normalize_class_label(item.get("t2_class"))

        # 验证类别有效性
        if t1_class and t1_class not in VALID_CLASSES:
            t1_class = ""
        if t2_class and t2_class not in VALID_CLASSES:
            t2_class = ""

        cleaned.append({
            "region_id": rid,
            "t1_class": t1_class,
            "t2_class": t2_class,
        })

    return cleaned


def apply_llm_classification(
    llm_results: List[Dict[str, Any]],
    regions_data: Dict[str, Any],
) -> Tuple[int, int]:
    """
    应用 LLM 分类结果到 regions.json

    入参:
        llm_results: LLM 分类结果
        regions_data: regions 数据

    出参:
        Tuple[int, int]: (处理区域数, 成功分类数)
    """
    by_id = {r["region_id"]: r for r in llm_results}

    processed = 0
    classified = 0

    for region in regions_data.get("regions", []):
        region_id = region.get("region_id")
        if region_id is None or int(region_id) <= 0:
            continue

        processed += 1

        llm_result = by_id.get(region_id)
        if not llm_result:
            continue

        # 确保 classification 字段存在
        if region.get("classification") is None:
            region["classification"] = {}

        # 添加 llm 分类
        region["classification"]["llm"] = {
            "t1_class": llm_result.get("t1_class"),
            "t2_class": llm_result.get("t2_class"),
        }

        if llm_result.get("t1_class") and llm_result.get("t2_class"):
            classified += 1

    return processed, classified


def process_single_sample(
    idx: int,
    total: int,
    regions_json_path: Path,
    split_dir: Path,
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    print_lock: threading.Lock,
) -> Dict[str, Any]:
    """
    处理单个样本

    入参:
        idx: 样本索引
        total: 总样本数
        regions_json_path: regions.json 路径
        split_dir: split 目录
        client: OpenAI 客户端
        model_name: 模型名称
        system_prompt: 系统提示词
        print_lock: 打印锁

    出参:
        Dict: 处理结果统计
    """
    sample_name = regions_json_path.parent.name
    sample_dir = regions_json_path.parent

    with print_lock:
        print(f"[{idx}/{total}] {sample_name}")

    try:
        regions_data = load_json(regions_json_path)

        # 收集有效的 region_id
        valid_ids = set()
        for r in regions_data.get("regions", []):
            rid = r.get("region_id")
            if rid and int(rid) > 0:
                valid_ids.add(int(rid))

        if not valid_ids:
            with print_lock:
                print(f"  - failed: no valid regions")
            return {"processed": 0, "classified": 0, "failed": 1}

        # 查找 bbox 图像
        t1_bbox_path = sample_dir / "t1_bbox.png"
        t2_bbox_path = sample_dir / "t2_bbox.png"

        if not t1_bbox_path.exists() or not t2_bbox_path.exists():
            with print_lock:
                print(f"  - failed: bbox images not found")
            return {"processed": 0, "classified": 0, "failed": 1}

        # 调用 VLM 分类
        llm_results = call_vlm_classify(
            client, model_name, system_prompt, t1_bbox_path, t2_bbox_path
        )

        # 规范化结果
        normalized = normalize_llm_result(llm_results, valid_ids)

        # 应用分类结果
        processed, classified = apply_llm_classification(normalized, regions_data)

        # 保存
        save_json(regions_json_path, regions_data)

        with print_lock:
            print(f"  - regions: {processed}, classified: {classified}")

        return {"processed": 1, "classified": classified, "failed": 0}

    except Exception as e:
        with print_lock:
            print(f"  - failed: {e}")
        return {"processed": 0, "classified": 0, "failed": 1}


def process_split(
    dataset_path: str,
    split: str,
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    batch_size: int = 10,
) -> None:
    """
    处理 split（批量模式，复用对话上下文）

    入参:
        dataset_path: 数据集路径
        split: split 名称
        client: OpenAI 客户端
        model_name: 模型名称
        system_prompt: 系统提示词
        batch_size: 每批次处理的样本数（默认10，之后重开对话）
    """
    split_dir = Path(dataset_path) / split
    regions_root = split_dir / "regions"

    if not regions_root.exists():
        print(f"⚠️  缺少 regions 目录: {regions_root}")
        return

    regions_json_files = sorted(glob.glob(str(regions_root / "*" / "regions.json")))
    if not regions_json_files:
        print(f"⚠️  未找到 regions.json: {regions_root}/*/regions.json")
        return

    print(f"\n{'=' * 68}")
    print(f"Step 3: VLM Classification - {split} split")
    print(f"Samples: {len(regions_json_files)}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size} (conversation resets after each batch)")
    print(f"{'=' * 68}")

    stats = {"total": len(regions_json_files), "processed": 0, "classified": 0, "failed": 0}

    # 收集所有样本数据
    samples_to_process = []
    for regions_json_path in regions_json_files:
        regions_json_path = Path(regions_json_path)
        sample_name = regions_json_path.parent.name
        sample_dir = regions_json_path.parent

        try:
            regions_data = load_json(regions_json_path)

            # 收集有效的 region_id
            valid_ids = set()
            for r in regions_data.get("regions", []):
                rid = r.get("region_id")
                if rid and int(rid) > 0:
                    valid_ids.add(int(rid))

            if not valid_ids:
                print(f"  ⚠️  {sample_name}: no valid regions")
                stats["failed"] += 1
                continue

            # 查找 bbox 图像
            t1_bbox_path = sample_dir / "t1_bbox.png"
            t2_bbox_path = sample_dir / "t2_bbox.png"

            if not t1_bbox_path.exists() or not t2_bbox_path.exists():
                print(f"  ⚠️  {sample_name}: bbox images not found")
                stats["failed"] += 1
                continue

            samples_to_process.append({
                "sample_name": sample_name,
                "regions_json_path": regions_json_path,
                "regions_data": regions_data,
                "t1_bbox_path": t1_bbox_path,
                "t2_bbox_path": t2_bbox_path,
                "valid_ids": valid_ids,
            })

        except Exception as e:
            print(f"  ⚠️  {sample_name}: failed to load - {e}")
            stats["failed"] += 1

    print(f"Valid samples: {len(samples_to_process)}")

    # 批量处理
    for batch_start in range(0, len(samples_to_process), batch_size):
        batch = samples_to_process[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(samples_to_process) + batch_size - 1) // batch_size

        print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} samples) ---")

        # 准备批量调用的数据
        batch_samples = []
        for s in batch:
            batch_samples.append({
                "sample_name": s["sample_name"],
                "t1_bbox_path": s["t1_bbox_path"],
                "t2_bbox_path": s["t2_bbox_path"],
                "valid_ids": s["valid_ids"],
            })

        try:
            # 批量调用 VLM（复用对话上下文）
            batch_results = call_vlm_classify_batch(
                client, model_name, system_prompt, batch_samples, batch_size=len(batch)
            )

            # 处理结果
            for sample_name, error_message, classifications in batch_results:
                # 找到对应的原始样本数据
                original_sample = next(s for s in batch if s["sample_name"] == sample_name)
                regions_json_path = original_sample["regions_json_path"]
                regions_data = original_sample["regions_data"]
                valid_ids = original_sample["valid_ids"]

                if error_message:
                    print(f"  ✗ {sample_name}: {error_message}")
                    stats["failed"] += 1
                    continue

                # 规范化结果
                normalized = normalize_llm_result(classifications, valid_ids)

                # 应用分类结果
                processed, classified = apply_llm_classification(normalized, regions_data)

                # 保存
                save_json(regions_json_path, regions_data)
                stats["processed"] += 1
                stats["classified"] += classified

        except Exception as e:
            print(f"  ✗ Batch {batch_num} failed: {e}")
            for s in batch:
                stats["failed"] += 1

    print(f"\nSummary ({split})")
    print(f"  total:      {stats['total']}")
    print(f"  processed:  {stats['processed']}")
    print(f"  failed:     {stats['failed']}")
    print(f"  classified: {stats['classified']} regions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: VLM Classification with scoring")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"], help="Dataset split")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API Key（云端 API 需要）")
    parser.add_argument("--model", type=str, default="qwen-vl-plus",
                        help="VLM model name（默认: qwen-vl-plus）")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API base URL（默认 DashScope 兼容接口）")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for conversation context (default: 10)")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("未检测到 DASHSCOPE_API_KEY，请先设置环境变量或传入 --api_key。")
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    system_prompt = load_system_prompt()

    print(f"🤖 VLM model: {args.model}")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        split_dir = Path(args.dataset_path) / split
        if not split_dir.exists():
            continue
        process_split(args.dataset_path, split, client, args.model, system_prompt, args.batch_size)


if __name__ == "__main__":
    main()
