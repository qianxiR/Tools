#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变化检测后处理主流程 — 一键串联 Step1→Step2→Step3→Step4

用法:
    # 完整流程
    python tools/cd_pipeline.py --data_root "E:/xzkjxm/dataes/CD-CD_20" --split all

    # 跳过 Qwen-VL (仅 CLIP 过滤)
    python tools/cd_pipeline.py --data_root "E:/xzkjxm/dataes/CD-CD_20" --split all --skip_step 3


前置条件:
    需先运行 infer_batch_cd.py 生成 change_mask/ 目录:
    python infer_batch_cd.py --data_root "E:/xzkjxm/dataes/CD-CD_20" --split all

"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import time
import shlex
import shutil
import glob
import subprocess

# 将 tools/ 目录加入 sys.path，确保可以 import 同目录模块
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def run_step(step_name: str, cmd: str):
    """运行一个步骤并打印状态"""
    print(f"\n{'#'*60}")
    print(f"# {step_name}")
    print(f"# {'#'*60}")
    print(f"CMD: {cmd}\n")

    t0 = time.time()
    # Windows 下用 shlex + subprocess.run 避免引号解析问题
    if sys.platform == 'win32':
        # shlex 在 posix=False 模式下能正确处理 Windows 引号
        parts = shlex.split(cmd, posix=False)
        # 去掉每个部分首尾可能残留的引号（shlex posix=False 保留引号）
        parts = [p.strip('"') for p in parts]
        ret = subprocess.call(parts)
    else:
        ret = subprocess.call(cmd, shell=True)
    elapsed = time.time() - t0

    if ret != 0:
        print(f"\n⚠ {step_name} 失败 (exit code: {ret}, 耗时 {elapsed:.1f}s)")
        return False

    print(f"\n✓ {step_name} 完成 ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="变化检测后处理主流程")
    parser.add_argument("--data_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--split", type=str, default="test", help="数据划分 (train/val/test/all)")
    parser.add_argument("--classes", type=str, default=None, help="类别文件路径")
    parser.add_argument("--min_area", type=int, default=4, help="Step1 最小实例面积")
    parser.add_argument("--clip_threshold", type=float, default=0.9,
                        help="Step2 CLIP 相对分数阈值 (relative_score=目标类别分/最高类别分)")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32",
                        choices=["RN50", "ViT-B-32", "ViT-L-14"], help="RemoteCLIP 模型")
    parser.add_argument("--checkpoint", type=str, default=None, help="CLIP 模型权重路径")
    parser.add_argument("--verify_mode", type=str, default="uncertain",
                        choices=["all", "uncertain"], help="Step3 Qwen-VL 验证模式")
    parser.add_argument("--qwen_model", type=str, default="qwen-vl-plus", help="Qwen-VL 模型名")
    parser.add_argument("--skip_step", type=int, nargs="*", default=[],
                        help="跳过的步骤 (如 --skip_step 3 跳过 Qwen-VL)")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--python", type=str, default=None, help="Python 解释器路径")

    args = parser.parse_args()

    # 确定 Python 解释器
    python = args.python or sys.executable
    tools_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建公共参数
    common_args = f'--data_root "{args.data_root}" --split {args.split}'
    if args.classes:
        common_args += f' --classes "{args.classes}"'

    steps = []

    # Step 1: 实例提取
    if 1 not in args.skip_step:
        cmd = f'"{python}" "{os.path.join(tools_dir, "cd1_extract_instances.py")}" {common_args} --min_area {args.min_area}'
        steps.append(("Step 1: 实例提取", cmd))

    # Step 2: CLIP 语义验证
    if 2 not in args.skip_step:
        cmd = f'"{python}" "{os.path.join(tools_dir, "cd2_clip_verify.py")}" {common_args} --clip_threshold {args.clip_threshold} --clip_model {args.clip_model} --device {args.device}'
        if args.checkpoint:
            cmd += f' --checkpoint "{args.checkpoint}"'
        steps.append(("Step 2: RemoteCLIP 语义验证", cmd))

    # Step 3: Qwen-VL 二次判别
    if 3 not in args.skip_step:
        cmd = f'"{python}" "{os.path.join(tools_dir, "cd3_qwen_verify.py")}" {common_args} --clip_threshold {args.clip_threshold} --verify_mode {args.verify_mode} --model_name {args.qwen_model}'
        steps.append(("Step 3: Qwen-VL 二次判别", cmd))

    # Step 4: 重建掩码
    if 4 not in args.skip_step:
        cmd = f'"{python}" "{os.path.join(tools_dir, "cd4_rebuild_mask.py")}" {common_args}'
        steps.append(("Step 4: 重建最终掩码", cmd))

    # 打印计划
    print(f"\n{'='*60}")
    print(f"变化检测后处理流水线")
    print(f"{'='*60}")
    print(f"数据集: {args.data_root}")
    print(f"划分: {args.split}")
    print(f"CLIP 阈值: {args.clip_threshold}")
    print(f"验证模式: {args.verify_mode}")
    print(f"步骤: {len(steps)} 个 (跳过: {args.skip_step or '无'})")
    print(f"{'='*60}")

    # 执行
    t_total = time.time()
    for i, (name, cmd) in enumerate(steps):
        success = run_step(name, cmd)
        if not success:
            print(f"\n⚠ 流水线在 {name} 中断")
            return 1

    elapsed_total = time.time() - t_total

    # 清理中间文件：instances/ 目录、*_instances.json、process_log.json
    splits_to_clean = ["train", "val", "test"] if args.split == "all" else [args.split]
    for sp in splits_to_clean:
        split_dir = os.path.join(args.data_root, sp)
        if not os.path.isdir(split_dir):
            continue
        # 删除 instances/ 目录
        instances_dir = os.path.join(split_dir, "instances")
        if os.path.isdir(instances_dir):
            shutil.rmtree(instances_dir)
            print(f"  已删除: {instances_dir}")
        # 删除 *_instances.json
        for jf in glob.glob(os.path.join(split_dir, "*_instances.json")):
            os.remove(jf)
            print(f"  已删除: {jf}")
        # 删除 process_log.json
        log_file = os.path.join(split_dir, "process_log.json")
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"  已删除: {log_file}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"✓ 流水线全部完成!")
    print(f"{'='*60}")
    print(f"总耗时: {elapsed_total:.1f}s")
    print(f"输出目录: {os.path.join(args.data_root, args.split)}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
