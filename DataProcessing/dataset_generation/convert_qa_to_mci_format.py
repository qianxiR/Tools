#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCI 格式转换工具

功能：合并 train/val/test 的 QA 文件 + metadata.json，转换为 train_release 范式格式
输出：
- {dataset_name}.json — 合并后的 QA 数据集
- metadata.json — 合并后的元数据
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_options(pair: Dict[str, Any]) -> tuple[list[Any], str]:
    """
    入参:
    - pair (Dict[str, Any]): 单条 QA 对象
    方法:
    - 兼容 dict/list/string 三种 options 结构
    - 同时生成 options_list 与 options_str，便于下游统一使用
    出参:
    - tuple[list[Any], str]: (options_list, options_str)
    """
    raw_options = pair.get("options", [])

    if isinstance(raw_options, dict):
        ordered_keys = sorted(raw_options.keys())
        options_list = [{str(key): str(raw_options[key]).strip()} for key in ordered_keys if str(raw_options[key]).strip()]
        options_str = "\n".join(
            f"{key}. {str(raw_options[key]).strip()}" for key in ordered_keys if str(raw_options[key]).strip()
        )
        return options_list, options_str

    if isinstance(raw_options, list):
        cleaned = [str(item).strip() for item in raw_options if str(item).strip()]
        return cleaned, "\n".join(cleaned)

    if isinstance(raw_options, str) and raw_options.strip():
        cleaned = [line.strip() for line in raw_options.splitlines() if line.strip()]
        return cleaned, "\n".join(cleaned)

    return [], ""


def select_split_file(dataset_path: Path, split: str) -> Optional[Path]:
    """
    入参:
    - dataset_path (Path): 数据集根目录
    - split (str): train/val/test
    方法:
    - 在 split 目录下按命名候选匹配 QA 文件
    出参:
    - Optional[Path]: 命中的文件路径，找不到则返回 None
    """
    split_dir = dataset_path / split
    candidates = [
        split_dir / f"qa_{split}.json",
        split_dir / f"QA_{split}.json",
    ]
    return next((item for item in candidates if item.exists()), None)


def infer_task_from_question(question: str) -> str:
    """
    入参:
    - question (str): 单条 QA 的问题文本
    方法:
    - 按问题类型进行规则匹配
    出参:
    - str: 任务类型标签（Description/CN/CtW/CfW/IN/DN/Unknown）
    """
    text = re.sub(r"\s+", " ", question.strip().lower())
    if not text:
        return "Unknown"

    # Description
    if text.startswith("provide a comprehensive description") or text.startswith("describe this image pair"):
        return "Description"

    # CN (Change or Not)
    if (text.startswith("based on the comparison") and "detectable land cover transformations involving" in text) or \
       (text.startswith("have the regions of") and "changed" in text):
        return "CN"

    # CtW (Change to What)
    if (text.startswith("what specific land cover class") and "transformed into" in text) or \
       (text.startswith("what have the regions of") and "changed to" in text):
        return "CtW"

    # CfW (Change from What)
    if (text.startswith("what was the original land cover classification") and "currently identified as" in text) or \
       (text.startswith("what did the") and "change from" in text):
        return "CfW"

    # IN (Increase or Not)
    if (text.startswith("when comparing the areal extent") and "net expansion" in text) or \
       (text.startswith("have the regions of") and "increased" in text):
        return "IN"

    # DN (Decrease or Not)
    if (text.startswith("when comparing the areal extent") and "net reduction" in text) or \
       (text.startswith("have the regions of") and "decreased" in text):
        return "DN"

    return "Unknown"


def build_release_rows(sample: Dict[str, Any], split: str, task_name: str, mode: str = "cd") -> List[Dict[str, Any]]:
    """
    入参:
    - sample (Dict[str, Any]): qa_{split}.json 的单条记录
    - split (str): 数据划分名称
    - task_name (str): 输出 task 字段值
    - mode (str): cd 或 seg
    方法:
    - 将单条 image 记录展开为多条 release 样本，每个问答对一条
    - CD 模式使用 t1/t2/label；SEG 模式使用 image/label
    出参:
    - List[Dict[str, Any]]: release 样本列表
    """
    qa_obj = sample.get("qa")
    if not isinstance(qa_obj, dict):
        return []
    image_name = str(sample.get("image_name", "")).strip()

    if mode == "seg":
        image_path = str(qa_obj.get("image", "")).strip()
        label = str(qa_obj.get("label", "")).strip()
        if not image_path:
            return []
        pre_image = image_path.replace("/", "\\")
        post_image = image_path.replace("/", "\\")
    else:
        t1 = str(qa_obj.get("t1", "")).strip()
        t2 = str(qa_obj.get("t2", "")).strip()
        label = str(qa_obj.get("label", "")).strip()
        if not t1 or not t2:
            return []
        pre_image = t1.replace("/", "\\")
        post_image = t2.replace("/", "\\")

    qa_pairs = qa_obj.get("qa", [])
    rows: List[Dict[str, Any]] = []
    if isinstance(qa_pairs, list):
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                continue
            question = str(pair.get("question", "")).strip()
            answer = str(pair.get("answer", "")).strip()
            qa_id = str(pair.get("id", "")).strip()
            qa_type = str(pair.get("type", "")).strip()
            answer_format = str(pair.get("answer_format", "")).strip()
            options_list, options_str = normalize_options(pair)
            if not question or not answer:
                continue
            task_value = infer_task_from_question(question) if task_name == "AUTO" else task_name
            rows.append(
                {
                    "image_name": image_name,
                    "qa_id": qa_id,
                    "pre_image_path": pre_image,
                    "post_image_path": post_image,
                    "post_image_type": "Optical",
                    "ground_truth": label,
                    "ground_truth_option": "",
                    "options_list": options_list,
                    "options_str": options_str,
                    "prompts": question,
                    "training_answer": answer,
                    "type": qa_type,
                    "answer_format": answer_format,
                    "task": task_value,
                    "split": split,
                }
            )
    return rows


def convert_split(dataset_path: Path, split: str, output_rows: List[Dict[str, Any]], task_name: str, mode: str = "cd") -> None:
    """
    入参:
    - dataset_path (Path): 数据集根目录
    - split (str): train/val/test
    - output_rows (List[Dict]): 聚合结果容器
    - task_name (str): 输出 task 字段值
    - mode (str): cd 或 seg
    方法:
    - 读取 split QA 文件并转换追加到总结果
    出参:
    - None
    """
    split_file = select_split_file(dataset_path, split)
    if split_file is None:
        print(f"{split}: qa file not found")
        return

    with split_file.open("r", encoding="utf-8") as file:
        records = json.load(file)

    converted = 0
    for sample in records:
        rows = build_release_rows(sample, split, task_name, mode)
        output_rows.extend(rows)
        converted += len(rows)
    print(f"{split}: converted={converted}")


def merge_metadata_files(dataset_path: Path, splits: List[str]) -> None:
    """
    入参:
    - dataset_path (Path): 数据集根目录
    - splits (List[str]): split列表 (train/val/test)
    方法:
    - 合并各split的metadata.json
    - 添加split字段标识来源
    - 保存到数据集根目录
    出参:
    - None
    """
    merged_metadata: List[Dict[str, Any]] = []

    for split in splits:
        metadata_path = dataset_path / split / "metadata.json"
        if not metadata_path.exists():
            print(f"{split}: metadata.json not found")
            continue

        with metadata_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        # 为每条记录添加split标识
        for record in records:
            record["_split"] = split
            merged_metadata.append(record)

        print(f"{split}: merged={len(records)} metadata records")

    if merged_metadata:
        output_path = dataset_path / "metadata.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(merged_metadata, f, ensure_ascii=False, indent=2)
        print(f"saved: {output_path} (total={len(merged_metadata)} records)")
    else:
        print("no metadata records to merge")


def main() -> int:
    """
    入参:
    - 无（从命令行参数读取）
    方法:
    - 合并 train/val/test 的 QA 文件
    - 合并 metadata.json
    - 输出 train_release 范式列表数据
    出参:
    - int: 0 成功
    """
    parser = argparse.ArgumentParser(description="Convert QA files to train_release style format")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--output", type=str, default=None, help="Output file name")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma separated splits")
    parser.add_argument("--task", type=str, default="AUTO", help="Task field value or AUTO")
    parser.add_argument("--mode", type=str, default="cd",
                       choices=["cd", "seg"], help="Mode: cd or seg (default: cd)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    dataset_name = dataset_path.name
    split_list = [item.strip() for item in args.splits.split(",") if item.strip()]

    merged_rows: List[Dict[str, Any]] = []

    print(f"Mode: {args.mode}, Single-turn QA")
    for split in split_list:
        convert_split(dataset_path, split, merged_rows, args.task, args.mode)

    # 确定输出文件名
    if args.output:
        output_name = args.output
    else:
        output_name = f"{dataset_name}.json"

    output_path = dataset_path / output_name
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(merged_rows, file, ensure_ascii=False, indent=2)

    print(f"saved: {output_path}")
    print(f"samples: {len(merged_rows)}")

    # 合并metadata.json到根目录
    print("\n" + "=" * 50)
    print("Merging metadata files...")
    print("=" * 50)
    merge_metadata_files(dataset_path, split_list)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
