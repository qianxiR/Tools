#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEG QA 生成工具

功能：基于 caption_{split}.json 中的结构化分析，调用 LLM 生成分割场景 QA
输入：{data_root}/{split}/caption_{split}.json（seg_generate_caption.py 输出）
输出：{data_root}/{split}/qa_{split}.json

流程：
1. 读取 Step 1 生成的 caption + 结构化分析
2. 用 prompt_seg_qa.md 生成 QA 对
3. QA 完全基于 caption 中的信息，不依赖图像

用法：
python vqa/seg_generate_seg_qa.py --data_root "E:/xzkjxm/dataes/WHU_20" --split test
"""
import os
import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    content = message.content or getattr(message, "reasoning", None) or ""
    if content.startswith("Thinking Process"):
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Answer:") or line.startswith("回答:"):
                return "\n".join(lines[i + 1:]).strip()
    return content


def normalize_qa_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    入参:
        result: 模型返回的原始 JSON
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
        options: 模型返回的选项（dict/list/str）
    方法:
        统一转换为 A-D 字典
    出参:
        Dict[str, str]: 标准化选项
    """
    option_keys = ["A", "B", "C", "D"]

    if isinstance(options, dict):
        normalized = {str(k).strip().upper(): str(v).strip()
                      for k, v in options.items() if str(v).strip()}
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
        qa_items: 问答列表
    方法:
        检查至少覆盖 qa/mcq/caption 三类，校验每条格式
    出参:
        None（校验失败抛 ValueError）
    """
    total = len([item for item in qa_items if isinstance(item, dict)])
    if total < 10:
        raise ValueError(f"Need at least 10 questions, got {total}")

    present_types = {str(item.get("type", "")).strip().lower()
                     for item in qa_items if isinstance(item, dict)}
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
            item["answer_format"] = "yes_no"

        elif item_type == "mcq":
            item["options"] = normalize_mcq_options(item.get("options"))
            if answer.upper() not in {"A", "B", "C", "D"}:
                raise ValueError("MCQ answer must be A/B/C/D")
            item["answer"] = answer.upper()
            item["answer_format"] = "mcq"

        elif item_type == "caption":
            if not answer:
                raise ValueError("Caption answer cannot be empty")
            item["answer_format"] = "text"


def generate_qa_for_sample(
    client: OpenAI,
    model_name: str,
    caption_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    入参:
        client: OpenAI 客户端
        model_name: 模型名称
        caption_data: Step 1 生成的 caption + 结构化分析
    方法:
        1. 加载 prompt_seg_qa.md
        2. 将 caption 结构化数据发给 LLM
        3. 生成 QA，最多重试 3 次
    出参:
        Dict: QA 记录
    """
    script_dir = Path(__file__).parent
    prompt_file = script_dir / "prompt_seg_qa.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # 构建结构化输入
    input_data = {
        "caption": caption_data.get("caption", ""),
        "present_categories": caption_data.get("present_categories", []),
        "category_details": caption_data.get("category_details", []),
        "num_regions": caption_data.get("num_regions", 0),
        "class_counts": caption_data.get("class_counts", {}),
        "scene_type": caption_data.get("scene_type", ""),
        "surrounding_categories": caption_data.get("surrounding_categories", []),
    }

    input_json = json.dumps(input_data, ensure_ascii=False, indent=2)

    user_content = (
        f"Generate Q&A pairs based on this segmentation analysis:\n\n"
        f"```json\n{input_json}\n```\n\n"
        f"Follow the prompt instructions. Return raw JSON only."
    )

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    messages = list(base_messages)
    result: Optional[Dict[str, Any]] = None
    qa_pairs: List[Dict[str, Any]] = []

    image_name = caption_data.get("image_name", "")

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
            "Regenerate with at least 10 questions:\n"
            "- 3+ QA (yes/no): about object presence, location, scene\n"
            "- 4+ MCQ: location, class, count, scene_type\n"
            "- 3+ Caption: local, scene_understanding, overall\n"
            "Return raw JSON only."
        )
        messages = base_messages + [
            {"role": "assistant", "content": json_string},
            {"role": "user", "content": repair_prompt},
        ]

    if result is None:
        raise ValueError("LLM failed to produce valid QA JSON")

    # 补充 ID
    for i, pair in enumerate(qa_pairs, 1):
        if isinstance(pair, dict) and not pair.get("id"):
            pair["id"] = f"{image_name}_q{i:03d}"

    # 保留 caption 元信息
    result["image_name"] = image_name
    result["caption"] = caption_data.get("caption", "")
    result["image"] = caption_data.get("image", "")

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
        读取 caption_{split}.json，生成 QA，写入 qa_{split}.json
    出参:
        None
    """
    split_dir = Path(data_root) / split
    caption_path = split_dir / f"caption_{split}.json"

    if not caption_path.exists():
        print(f"Error: {caption_path} not found. Run seg_generate_caption.py first.")
        return

    with open(caption_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)

    output_path = split_dir / f"qa_{split}.json"

    print(f"\n{'=' * 60}")
    print(f"SEG QA Generation - {split}")
    print(f"Samples: {len(caption_data)}")
    print(f"{'=' * 60}")

    qa_results: List[Dict[str, Any]] = []
    success_count = 0
    total_qa_pairs = 0
    print_lock = threading.Lock()
    results_lock = threading.Lock()

    def flush():
        with results_lock:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(qa_results, f, ensure_ascii=False, indent=2)

    def _process_one(idx, sample):
        image_name = sample.get("image_name", "")
        with print_lock:
            print(f"[{idx}/{len(caption_data)}] {image_name}")

        try:
            qa_result = generate_qa_for_sample(client, model_name, sample)
            num_q = len(qa_result.get("qa", []))
            with print_lock:
                print(f"   OK {image_name}: {num_q} Q&A")
            return {"success": True, "qa": qa_result, "qa_pairs": num_q,
                    "image_name": image_name}
        except Exception as e:
            with print_lock:
                print(f"   FAIL {image_name}: {e}")
            return {"success": False, "qa": None, "error": str(e),
                    "image_name": image_name}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, i, sample): i
            for i, sample in enumerate(caption_data, 1)
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

    print(f"\n{'=' * 60}")
    print(f"SEG QA Complete - {split}")
    print(f"Success: {success_count}/{len(qa_results)}, Pairs: {total_qa_pairs}")
    print(f"Saved: {output_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="SEG QA Generation from Caption")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"], help="Split")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--model", type=str, default="qwen-vl-plus",
                        help="LLM model (text-only is OK)")
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
