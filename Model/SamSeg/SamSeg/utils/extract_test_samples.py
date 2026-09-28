#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试样本提取工具（支持 CD / SEG 模式）

CD 用法:
python utils/extract_test_samples.py --mode cd --source "E:/xzkjxm/dataes/CD/cdzcq_cd" --dest "E:/xzkjxm/dataes/WHU-CD_20" --count 20

SEG 用法:
python utils/extract_test_samples.py --mode seg --source "E:/xzkjxm/dataes/CD/cdzcq_seg" --dest "E:/xzkjxm/dataes/WHU_20" --count 20
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def extract_samples(source_path: str, dest_path: str, count: int, splits: list, mode: str = "cd") -> None:
    """
    入参:
    - source_path (str): 源数据集根目录
    - dest_path (str): 目标数据集根目录
    - count (int): 每个split提取的样本数量
    - splits (list): 要处理的数据集划分列表
    - mode (str): cd 或 seg，决定子目录结构

    方法:
    - CD 模式: 从 t1/ 获取文件名，复制 t1/t2，有 label 则一并复制
    - SEG 模式: 从 image/ 获取文件名，复制 image，有 label 则一并复制
    - 如果源路径和目标路径相同，只输出样本列表

    出参:
    - return (None): 无返回值，直接写文件
    """
    source_root = Path(source_path).resolve()
    dest_root = Path(dest_path).resolve()

    same_path = (source_root == dest_root)
    if same_path:
        print("源路径和目标路径相同，将只输出样本列表而不复制文件")

    # CD: t1/t2; SEG: image（label 按需检测）
    if mode == "seg":
        scan_dir = "image"
        core_subdirs = ["image"]
    else:
        scan_dir = "t1"
        core_subdirs = ["t1", "t2"]

    for split in splits:
        split_source = source_root / split
        split_dest = dest_root / split

        if not split_source.exists():
            print(f"Skip: {split} 目录不存在")
            continue

        scan_path = split_source / scan_dir
        if not scan_path.exists():
            print(f"Skip: {split}/{scan_dir} 目录不存在")
            continue

        # 检测是否有 label 目录，有则加入复制列表
        subdirs = list(core_subdirs)
        has_label = (split_source / "label").is_dir()
        if has_label:
            subdirs.append("label")

        all_files = sorted([
            f for f in os.listdir(scan_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ])

        if len(all_files) < count:
            print(f"Warning: {split} 只有 {len(all_files)} 个样本，将全部复制")
            selected = all_files
        else:
            selected = random.sample(all_files, count)

        label_info = " (含 label)" if has_label else ""
        print(f"\n{split}: 提取 {len(selected)} 个样本{label_info}")

        for subdir in subdirs:
            (split_dest / subdir).mkdir(parents=True, exist_ok=True)

        for filename in selected:
            for subdir in subdirs:
                src_file = split_source / subdir / filename
                dst_file = split_dest / subdir / filename

                if src_file.exists():
                    if same_path:
                        continue
                    if src_file.resolve() == dst_file.resolve():
                        continue
                    try:
                        shutil.copy2(src_file, dst_file)
                    except PermissionError:
                        print(f"  Skip (permission): {split}/{subdir}/{filename}")
                else:
                    # 文件不存在时静默跳过
                    pass

        print(f"  Done: {[f'{s}/{len(selected)}' for s in subdirs]}")

    print(f"\n提取完成! 目标目录: {dest_root}")


def main():
    parser = argparse.ArgumentParser(
        description="测试样本提取工具 (CD/SEG)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--mode", type=str, default="cd",
                       choices=["cd", "seg"],
                       help="模式: cd=变化检测(t1/t2), seg=语义分割(image) (default: cd)")
    parser.add_argument("--source", type=str, required=True,
                       help="源数据集路径")
    parser.add_argument("--dest", type=str, required=True,
                       help="目标数据集路径")
    parser.add_argument("--count", type=int, default=20,
                       help="每个split提取的样本数量 (default: 20)")
    parser.add_argument("--splits", type=str, default="train,val,test",
                       help="要处理的split，逗号分隔 (default: train,val,test)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (default: 42)")

    args = parser.parse_args()

    source_resolved = Path(args.source).resolve()
    dest_resolved = Path(args.dest).resolve()

    random.seed(args.seed)
    splits = [s.strip() for s in args.splits.split(",")]

    print("=" * 60)
    print(f"测试样本提取工具 (mode={args.mode})")
    print("=" * 60)
    print(f"源目录: {args.source}")
    print(f"目标目录: {args.dest}")

    if source_resolved == dest_resolved:
        print("警告: 源目录和目标目录相同! 只显示样本列表，不复制文件。")

    print(f"样本数量: {args.count} (per split)")
    print(f"数据划分: {', '.join(splits)}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)

    extract_samples(args.source, args.dest, args.count, splits, args.mode)

    # 打印最终结构
    core_subdirs = ["image"] if args.mode == "seg" else ["t1", "t2"]
    print("\n目标目录结构:")
    for split in splits:
        split_path = Path(args.dest) / split
        if split_path.exists():
            for subdir in core_subdirs + ["label"]:
                subdir_path = split_path / subdir
                if subdir_path.exists():
                    file_count = len([f for f in os.listdir(subdir_path)
                                     if f.lower().endswith((".png", ".jpg", ".tif"))])
                    print(f"  {split}/{subdir}: {file_count} files")

    return 0


if __name__ == "__main__":
    exit(main())
