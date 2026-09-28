#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单时相分割区域 VLM 分类修正工具（SEG 模式 Step seg3）

功能：调用 VLM 对分割区域进行分类确认/修正
输入：seg1 输出的 regions/{sample}/ + seg2 写入的 classification.clip
输出：regions.json 新增 classification.llm 字段（单类别）

与 CD 模式的区别：
- 只发送单张 image_bbox.png（非 T1+T2 双图）
- classification.llm 输出 class（非 t1_class/t2_class）
- prompt 聚焦于"识别目标类别"而非"判断变化类型"
"""
import os
import json
import re
import glob
import base64
import argparse
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
    "bare_land": "bareland", "barren_land": "bareland", "barrenland": "bareland",
    "desert": "bareland",
    "road": "highway", "roads": "highway",
}


def get_message_content(message) -> str:
    """
    入参:
        message: OpenAI 返回的 message 对象
    方法:
        兼容推理模型的 reasoning 字段，提取实际文本
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


def normalize_class_label(value: Any) -> str:
    """
    入参:
        value: 原始类别值
    方法:
        统一为小写英文，通过别名映射表标准化
    出参:
        str: 规范化后的类别名
    """
    if value is None:
        return ""
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    label = CLASS_ALIASES.get(label, label)
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
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
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
    img = imread_unicode(image_path)
    if img is not None:
        h, w = img.shape[:2]
        if max(h, w) > resize_max_dim:
            scale = resize_max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".tif": "image/tiff"}
    return f"data:{mime_map.get(ext, 'image/png')};base64,{encoded}"


def load_system_prompt() -> str:
    """
    加载 SEG 分类 prompt 文件

    出参:
        str: 系统提示词内容
    """
    script_dir = Path(__file__).parent
    prompt_path = script_dir / "prompt_seg_classify.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def extract_json_payload(response_text: str) -> Any:
    """
    入参:
        response_text (str): VLM 响应文本
    方法:
        尝试从 markdown 代码块、直接 JSON、截断 JSON 等多种格式中提取 JSON
    出参:
        Any: 解析后的 JSON 数据
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
        return json.loads("[" + text.strip("[]").strip() + "]")
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
    image_bbox_path: Path,
) -> List[Dict[str, Any]]:
    """
    入参:
        client: OpenAI 客户端
        model_name (str): VLM 模型名称
        system_prompt (str): 系统提示词
        image_bbox_path (Path): 带编号 bbox 标注图路径
    方法:
        发送单张 image_bbox.png 给 VLM，要求分类所有编号区域
    出参:
        List[Dict[str, Any]]: 分类结果列表 [{region_id, class}, ...]
    """
    content_parts = []

    if image_bbox_path.exists():
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": encode_image_to_base64(image_bbox_path)},
        })

    content_parts.append({
        "type": "text",
        "text": "\nClassify all numbered regions (region_id > 0) according to the Category System.",
    })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
        extra_body={"disable_search": True},
    )

    response_text = get_message_content(response.choices[0].message)
    result = extract_json_payload(response_text)

    if isinstance(result, dict):
        result = result.get("regions", result.get("final_output", []))

    if not isinstance(result, list):
        raise ValueError(f"VLM 返回格式错误，期望 JSON 数组: {type(result)}")

    return result


def normalize_llm_result(raw: List[Dict[str, Any]], valid_ids: set) -> List[Dict[str, Any]]:
    """
    入参:
        raw (List[Dict]): VLM 原始分类结果
        valid_ids (set): 有效的 region_id 集合
    方法:
        提取 region_id，规范化 class 标签，过滤无效条目
    出参:
        List[Dict]: [{region_id, class}, ...]
    """
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        rid_raw = item.get("region_id")
        rid = rid_raw if isinstance(rid_raw, int) else None
        if rid is None and rid_raw:
            m = re.search(r"\d+", str(rid_raw))
            if m:
                rid = int(m.group(0))

        if rid is None or rid not in valid_ids:
            continue

        cls = normalize_class_label(item.get("class"))
        if cls and cls not in VALID_CLASSES:
            cls = ""

        cleaned.append({"region_id": rid, "class": cls})

    return cleaned


def apply_llm_classification(
    llm_results: List[Dict[str, Any]],
    regions_data: Dict[str, Any],
) -> tuple:
    """
    入参:
        llm_results (List[Dict]): VLM 分类结果
        regions_data (Dict): regions.json 数据
    方法:
        将 VLM 分类结果写入对应 region 的 classification.llm 字段
    出参:
        tuple: (处理区域数, 成功分类数)
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

        if region.get("classification") is None:
            region["classification"] = {}

        region["classification"]["llm"] = {
            "class": llm_result.get("class"),
        }

        if llm_result.get("class"):
            classified += 1

    return processed, classified


def process_split(
    dataset_path: str,
    split: str,
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    max_workers: int = 4,
) -> None:
    """
    入参:
        dataset_path (str): 数据集根目录
        split (str): 数据集划分
        client: OpenAI 客户端
        model_name (str): VLM 模型名称
        system_prompt (str): 系统提示词
        max_workers (int): 最大并行数
    方法:
        多线程遍历所有 regions.json，调用 VLM 分类后写回
    出参:
        None（结果写回 regions.json）
    """
    split_dir = Path(dataset_path) / split
    regions_root = split_dir / "regions"

    if not regions_root.exists():
        print(f"Missing regions directory: {regions_root}")
        return

    regions_json_files = sorted(glob.glob(str(regions_root / "*" / "regions.json")))
    if not regions_json_files:
        print(f"No regions.json found in {regions_root}")
        return

    print(f"\n{'='*60}")
    print(f"SEG VLM Classify - {split}")
    print(f"Samples: {len(regions_json_files)}, Model: {model_name}")
    print(f"{'='*60}\n")

    stats = {"total": len(regions_json_files), "processed": 0, "classified": 0, "failed": 0}
    print_lock = threading.Lock()

    def _process_one(idx, rj_path):
        rj_path = Path(rj_path)
        sample_name = rj_path.parent.name
        sample_dir = rj_path.parent

        with print_lock:
            print(f"[{idx}/{len(regions_json_files)}] {sample_name}")

        try:
            with open(rj_path, 'r', encoding='utf-8') as f:
                regions_data = json.load(f)

            valid_ids = set()
            for r in regions_data.get("regions", []):
                rid = r.get("region_id")
                if rid and int(rid) > 0:
                    valid_ids.add(int(rid))

            if not valid_ids:
                with print_lock:
                    print(f"  SKIP: no valid regions")
                return {"processed": 0, "classified": 0, "failed": 1}

            image_bbox_path = sample_dir / "image_bbox.png"
            if not image_bbox_path.exists():
                with print_lock:
                    print(f"  SKIP: image_bbox.png not found")
                return {"processed": 0, "classified": 0, "failed": 1}

            llm_results = call_vlm_classify(client, model_name, system_prompt, image_bbox_path)
            normalized = normalize_llm_result(llm_results, valid_ids)
            processed, classified = apply_llm_classification(normalized, regions_data)

            with open(rj_path, 'w', encoding='utf-8') as f:
                json.dump(regions_data, f, ensure_ascii=False, indent=2)

            with print_lock:
                print(f"  OK: {processed} regions, {classified} classified")

            return {"processed": 1, "classified": classified, "failed": 0}

        except Exception as e:
            with print_lock:
                print(f"  FAIL: {e}")
            return {"processed": 0, "classified": 0, "failed": 1}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, i, rj_path): i
            for i, rj_path in enumerate(regions_json_files, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            stats["processed"] += result["processed"]
            stats["classified"] += result["classified"]
            stats["failed"] += result["failed"]

    print(f"\n{'='*60}")
    print(f"SEG VLM Classify Complete - {split}")
    print(f"Processed: {stats['processed']}, Failed: {stats['failed']}")
    print(f"Classified regions: {stats['classified']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SEG mode: VLM classify segmentation regions")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test", "all"], help="Dataset split")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--model", type=str, default="qwen-vl-plus", help="VLM model name")
    parser.add_argument("--base_url", type=str,
                       default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                       help="API base URL")
    parser.add_argument("--max_workers", type=int, default=1, help="Max parallel workers")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("未检测到 API Key，请设置 DASHSCOPE_API_KEY 或传入 --api_key。")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    system_prompt = load_system_prompt()

    print(f"VLM model: {args.model}")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        split_dir = Path(args.dataset_path) / split
        if not split_dir.exists():
            continue
        process_split(args.dataset_path, split, client, args.model, system_prompt, args.max_workers)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
