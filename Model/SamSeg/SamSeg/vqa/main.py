#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQA 数据生成主入口

两步流水线：caption 生成 → QA 生成
通过 --mode cd / seg 区分变化检测与语义分割任务。

CD 用法:
    python vqa/main.py --mode cd --data_root "E:/xzkjxm/dataes/CD-CD_20" --split val
    python vqa/main.py --mode cd --data_root "E:/xzkjxm/dataes/CD-CD_20" --split all

SEG 用法:
    python vqa/main.py --mode seg --data_root "E:/xzkjxm/dataes/CD_20" --split val
    python vqa/main.py --mode seg --data_root "E:/xzkjxm/dataes/CD_20" --split all

可选参数:
    --skip_caption   跳过 Step 1（caption 已生成时使用）
    --skip_qa        跳过 Step 2（仅生成 caption）
    --model          VLM/LLM 模型名称（默认 qwen-vl-plus）
    --max_workers    并行数（默认 1）
"""

import argparse
import os
import sys
import time
from pathlib import Path
from openai import OpenAI


def get_api_key(args) -> str:
    """
    入参:
        args: 命令行参数
    方法:
        按优先级获取 API Key：命令行 > 环境变量
    出参:
        str: API Key
    """
    key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        print("ERROR: API Key required. Set DASHSCOPE_API_KEY or pass --api_key.")
        sys.exit(1)
    return key


def resolve_splits(data_root: str, split_arg: str, mode: str) -> list:
    """
    入参:
        data_root: 数据集根目录
        split_arg: split 参数（train/val/test/all）
        mode: cd 或 seg
    方法:
        将 "all" 展开为 train/val/test，跳过不存在的目录
    出参:
        list: 有效的 split 列表
    """
    candidates = ["train", "val", "test"] if split_arg == "all" else [split_arg]
    valid = []
    for sp in candidates:
        sp_dir = Path(data_root) / sp
        if sp_dir.is_dir():
            valid.append(sp)
        else:
            print(f"  Skip: {sp_dir} not found")
    return valid


def run_cd_pipeline(args, client, splits):
    """
    入参:
        args: 命令行参数
        client: OpenAI 客户端
        splits: 有效的 split 列表
    方法:
        CD 模式两步流水线：
        Step 1 — cd_generate_caption: T1/T2 mask + 原图 → caption
        Step 2 — cd_generate_qa: caption → QA
    出参:
        None
    """
    from cd_generate_caption import process_split as caption_step
    from cd_generate_qa import process_split as qa_step

    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"  CD Pipeline — {split}")
        print(f"{'=' * 60}")

        # Step 1: Caption
        if not args.skip_caption:
            print(f"\n--- Step 1: Caption Generation ---")
            t0 = time.time()
            caption_step(args.data_root, split, client, args.model, args.max_workers)
            print(f"  Time: {time.time() - t0:.1f}s")
        else:
            caption_path = Path(args.data_root) / split / f"caption_{split}.json"
            print(f"\n  Skip caption (use existing: {caption_path})")

        # Step 2: QA
        if not args.skip_qa:
            print(f"\n--- Step 2: QA Generation ---")
            t0 = time.time()
            qa_step(args.data_root, split, client, args.model, args.max_workers)
            print(f"  Time: {time.time() - t0:.1f}s")
        else:
            print(f"\n  Skip QA generation")


def run_seg_pipeline(args, client, splits):
    """
    入参:
        args: 命令行参数
        client: OpenAI 客户端
        splits: 有效的 split 列表
    方法:
        SEG 模式两步流水线：
        Step 1 — seg_generate_caption: seg_mask + 原图 → caption
        Step 2 — seg_generate_seg_qa: caption → QA
    出参:
        None
    """
    from seg_generate_caption import process_split as caption_step
    from seg_generate_seg_qa import process_split as qa_step

    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"  SEG Pipeline — {split}")
        print(f"{'=' * 60}")

        # Step 1: Caption
        if not args.skip_caption:
            print(f"\n--- Step 1: Caption Generation ---")
            t0 = time.time()
            caption_step(args.data_root, split, client, args.model, args.max_workers)
            print(f"  Time: {time.time() - t0:.1f}s")
        else:
            caption_path = Path(args.data_root) / split / f"caption_{split}.json"
            print(f"\n  Skip caption (use existing: {caption_path})")

        # Step 2: QA
        if not args.skip_qa:
            print(f"\n--- Step 2: QA Generation ---")
            t0 = time.time()
            qa_step(args.data_root, split, client, args.model, args.max_workers)
            print(f"  Time: {time.time() - t0:.1f}s")
        else:
            print(f"\n  Skip QA generation")


def main():
    parser = argparse.ArgumentParser(
        description="VQA Data Generation Pipeline (Caption → QA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # CD — full pipeline
  python vqa/main.py --mode cd --data_root "E:/xzkjxm/dataes/CD-CD_20" --split val

  # SEG — full pipeline
  python vqa/main.py --mode seg --data_root "E:/xzkjxm/dataes/CD_20" --split val

  # Only generate QA (caption already exists)
  python vqa/main.py --mode cd --data_root "E:/xzkjxm/dataes/CD-CD_20" --split val --skip_caption

  # Only generate caption
  python vqa/main.py --mode seg --data_root "E:/xzkjxm/dataes/CD_20" --split test --skip_qa
""",
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=["cd", "seg"],
                        help="Task mode: cd=change detection, seg=semantic segmentation")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Dataset root directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="Dataset split (default: test)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API Key (or set DASHSCOPE_API_KEY)")
    parser.add_argument("--model", type=str, default="qwen-vl-plus",
                        help="VLM/LLM model name (default: qwen-vl-plus)")
    parser.add_argument("--base_url", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API base URL")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Max parallel workers (default: 1)")
    parser.add_argument("--skip_caption", action="store_true",
                        help="Skip Step 1 (caption already generated)")
    parser.add_argument("--skip_qa", action="store_true",
                        help="Skip Step 2 (only generate caption)")

    args = parser.parse_args()

    # 添加 vqa 目录到 sys.path，以便 import 子模块
    vqa_dir = str(Path(__file__).parent.resolve())
    if vqa_dir not in sys.path:
        sys.path.insert(0, vqa_dir)

    api_key = get_api_key(args)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    splits = resolve_splits(args.data_root, args.split, args.mode)
    if not splits:
        print("No valid splits found. Check --data_root and --split.")
        return 1

    print("=" * 60)
    print(f"  VQA Pipeline — mode={args.mode}")
    print(f"  Data: {args.data_root}")
    print(f"  Splits: {', '.join(splits)}")
    print(f"  Model: {args.model}")
    print(f"  Workers: {args.max_workers}")
    print(f"  Steps: {'caption' if not args.skip_caption else ''}"
          f"{' → ' if not args.skip_caption and not args.skip_qa else ''}"
          f"{'qa' if not args.skip_qa else ''}")
    print("=" * 60)

    t_start = time.time()

    if args.mode == "cd":
        run_cd_pipeline(args, client, splits)
    else:
        run_seg_pipeline(args, client, splits)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  All done! Total time: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
