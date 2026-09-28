#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图文数据集生成工具 - 主入口

模式:
  cd  — 变化检测（CD）模式，T1/T2 双时相 + 变化 mask
  seg — 语义分割（SEG）模式，单时相 image + 分割 mask

CD 流程:
cd1. 提取变化区域（cd1_extract_regions.py）
cd2. RemoteCLIP 分类（cd2_clip_classify.py）
cd3. VLM 分类修正（cd3_examine_result.py）
cd4. 局部 patch 修正 llm 并汇总 metadata（cd4_generate_caption.py，需要 API）
cd5. 生成 QA 数据集（cd5_generate_qa.py）
  mci. 转换为 MCI 格式（convert_qa_to_mci_format.py）
  coco. 转换为 COCO 格式（convert_to_coco.py）

SEG 流程:
seg1. 提取分割区域（seg1_extract_regions.py）
seg2. RemoteCLIP 分类（seg2_clip_classify.py）
seg3. VLM 分类修正（seg3_vlm_classify.py）
seg4. 描述生成 + metadata 汇总（seg4_generate_caption.py，需要 API）
seg5. 生成 QA 数据集（seg5_generate_seg_qa.py）
  mci. 转换为 MCI 格式（convert_qa_to_mci_format.py）
  coco. 转换为 COCO 格式（convert_to_coco.py）

使用方法:

# ==================== CD 模式 ====================
python -m Tools.DataProcessing.dataset_generation.main --mode cd --dataset_path "E:/xzkjxm/dataes/WHU-CD_20" --split all --step all
python -m Tools.DataProcessing.dataset_generation.main --dataset_path "E:/xzkjxm/dataes/WHU-CD_20" --split val --step cd1
python -m Tools.DataProcessing.dataset_generation.main --dataset_path "E:/xzkjxm/dataes/WHU-CD_20" --split all --step cd123
python -m Tools.DataProcessing.dataset_generation.main --dataset_path "E:/xzkjxm/dataes/WHU-CD_20" --split all --step cd45
python -m Tools.DataProcessing.dataset_generation.main --dataset_path "E:/xzkjxm/dataes/WHU-CD_20" --split all --step coco

# ==================== SEG 模式 ====================
python -m Tools.DataProcessing.dataset_generation.main --mode seg --dataset_path "E:/xzkjxm/dataes/WHU_20" --split all --step all
python -m Tools.DataProcessing.dataset_generation.main --mode seg --dataset_path "E:/xzkjxm/dataes/WHU_20" --split test --step seg1
python -m Tools.DataProcessing.dataset_generation.main --mode seg --dataset_path "E:/xzkjxm/dataes/WHU_20" --split all --step seg123
python -m Tools.DataProcessing.dataset_generation.main --mode seg --dataset_path "E:/xzkjxm/dataes/WHU_20" --split all --step seg45
python -m Tools.DataProcessing.dataset_generation.main --mode seg --dataset_path "E:/xzkjxm/dataes/WHU_20" --split all --step coco

"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="图文数据集生成工具（CD模式: Step cd1-cd5, SEG模式: Step seg1-5 + 共用 mci/coco）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
none
        """
    )

    parser.add_argument("--mode", type=str, default="cd",
                       choices=["cd", "seg"],
                       help="模式: cd=变化检测, seg=语义分割 (default: cd)")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="数据集根目录")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test", "all"],
                       help="数据集分割 (default: train)")
    parser.add_argument("--step", type=str, default="all",
                       help="处理步骤 (CD: all/cd1-cd5/cd123/cd45/mci/coco; SEG: all/seg1-seg5/seg123/seg45/mci/coco)")

    # Step 2 参数：RemoteCLIP 分类
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        choices=["RN50", "ViT-B-32", "ViT-L-14"],
                        help="RemoteCLIP 模型名称 (default: ViT-B-32)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="RemoteCLIP 模型权重路径")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="设备 (default: cuda)")

    # Step 3 参数：VLM 审查修正（cd3_examine_result.py）
    parser.add_argument("--vlm_model", type=str, default="qwen-vl-plus",
                        help="VLM 模型名称 (Step 3, 默认: qwen-vl-plus)")

    # Step 4 参数：VLM 生成描述
    parser.add_argument("--desc_model", type=str, default="qwen-vl-plus",
                        help="描述生成模型 (Step 4, 默认: qwen-vl-plus)")

    # Step 5 参数：QA 生成（cd5_generate_qa.py）
    parser.add_argument("--qa_model", type=str, default="qwen-vl-plus",
                        help="QA 生成模型 (Step 5, 默认: qwen-vl-plus)")

    parser.add_argument("--api_key", type=str, default=None,
                        help="API Key（云端 API 需要）")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API base URL（默认 DashScope 兼容接口）")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for conversation context (Step 3/4, default: 10)")

    # 共用参数：MCI 转换（convert_qa_to_mci_format.py）
    parser.add_argument("--mci_task", type=str, default="auto",
                        choices=["auto", "change", "description", "locate", "bbox", "count"],
                        help="MCI 任务类型 (Step mci, default: auto)")

    args = parser.parse_args()

    # 显示 API 配置
    print(f"\n Mode: {args.mode.upper()}")
    print(f" API: DashScope / OpenAI compatible")
    print(f" base_url: {args.base_url}")

    if args.mode == "cd":
        print(f" Step3 model: {args.vlm_model}")
        print(f" Step4 model: {args.desc_model}")
        print(f" Step5 model: {args.qa_model}")
    else:
        print(f" Step3 model: {args.vlm_model}")
        print(f" Step4 model: {args.desc_model}")
        print(f" Step5 model: {args.qa_model}")

    if not os.path.isabs(args.dataset_path):
        args.dataset_path = os.path.abspath(args.dataset_path)
        print(f"📂 使用相对路径，自动转为绝对路径: {args.dataset_path}")

    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset path not found: {args.dataset_path}")
        return 1

    module_dir = os.path.dirname(os.path.abspath(__file__))

    # 根据 mode 构建 step_map
    if args.mode == "cd":
        cd_step_choices = {"all", "cd1", "cd2", "cd3", "cd4", "cd5", "mci", "coco", "cd123", "cd45"}
        if args.step not in cd_step_choices:
            print(f"Error: CD mode step must be one of {sorted(cd_step_choices)}")
            return 1
        step_map = {
            "all": ["cd1", "cd2", "cd3", "cd4", "cd5", "mci", "coco"],
            "cd1": ["cd1"],
            "cd2": ["cd2"],
            "cd3": ["cd3"],
            "cd4": ["cd4"],
            "cd5": ["cd5"],
            "mci": ["mci"],
            "coco": ["coco"],
            "cd123": ["cd1", "cd2", "cd3"],
            "cd45": ["cd4", "cd5"],
        }
    else:
        seg_step_choices = {"all", "seg1", "seg2", "seg3", "seg4", "seg5", "mci", "coco", "seg123", "seg45"}
        if args.step not in seg_step_choices:
            print(f"Error: SEG mode step must be one of {sorted(seg_step_choices)}")
            return 1
        step_map = {
            "all": ["seg1", "seg2", "seg3", "seg4", "seg5", "mci", "coco"],
            "seg1": ["seg1"],
            "seg2": ["seg2"],
            "seg3": ["seg3"],
            "seg4": ["seg4"],
            "seg5": ["seg5"],
            "mci": ["mci"],
            "coco": ["coco"],
            "seg123": ["seg1", "seg2", "seg3"],
            "seg45": ["seg4", "seg5"],
        }
    steps_to_run = step_map.get(args.step) or [int(args.step)]

    def _get_api_key():
        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("\n❌ Error: 未检测到 DASHSCOPE_API_KEY。")
            print("   请先执行: $env:DASHSCOPE_API_KEY=\"你的Key\"")
            print("   或在命令中传入: --api_key \"你的Key\"")
            return None
        return api_key

    # ==================== SEG Step seg1: 提取分割区域 ====================
    if "seg1" in steps_to_run:
        print("\n" + "=" * 70)
        print(" SEG Step seg1: 提取分割区域")
        print("=" * 70)

        cmd = [sys.executable, os.path.join(module_dir, "seg1_extract_regions.py"),
               "--dataset_path", args.dataset_path, "--split", args.split]
        print(f" Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n Error: 提取分割区域失败")
            return 1
        print(" SEG Step seg1 完成！")

    # ==================== SEG Step seg2: RemoteCLIP 分类 ====================
    if "seg2" in steps_to_run:
        print("\n" + "=" * 70)
        print(" SEG Step seg2: RemoteCLIP 分类")
        print("=" * 70)

        cmd = [sys.executable, os.path.join(module_dir, "seg2_clip_classify.py"),
               "--dataset_path", args.dataset_path, "--split", args.split,
               "--model", args.model, "--device", args.device]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        print(f" Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n Error: SEG CLIP 分类失败")
            return 1
        print(" SEG Step seg2 完成！")

    # ==================== SEG Step seg3: VLM 分类修正 ====================
    if "seg3" in steps_to_run:
        print("\n" + "=" * 70)
        print(" SEG Step seg3: VLM 分类修正")
        print("=" * 70)

        api_key = _get_api_key()
        if not api_key:
            return 1

        cmd = [sys.executable, os.path.join(module_dir, "seg3_vlm_classify.py"),
               "--dataset_path", args.dataset_path, "--split", args.split,
               "--model", args.vlm_model, "--base_url", args.base_url]
        if args.api_key:
            cmd.extend(["--api_key", args.api_key])
        print(f" Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n Error: SEG VLM 分类修正失败")
            return 1
        print(" SEG Step seg3 完成！")

    # ==================== SEG Step seg4: 描述生成 + metadata ====================
    if "seg4" in steps_to_run:
        print("\n" + "=" * 70)
        print(" SEG Step seg4: 描述生成 + metadata 汇总")
        print("=" * 70)

        api_key = _get_api_key()
        if not api_key:
            return 1

        cmd = [sys.executable, os.path.join(module_dir, "seg4_generate_caption.py"),
               "--dataset_path", args.dataset_path, "--split", args.split,
               "--model", args.desc_model, "--base_url", args.base_url,
               "--batch_size", str(args.batch_size)]
        if args.api_key:
            cmd.extend(["--api_key", args.api_key])
        print(f" Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n Error: SEG 描述生成失败")
            return 1
        print(" SEG Step seg4 完成！")

    # ==================== SEG Step seg5: QA 生成 ====================
    if "seg5" in steps_to_run:
        print("\n" + "=" * 70)
        print(" SEG Step seg5: 生成分割 QA 数据集")
        print("=" * 70)

        api_key = _get_api_key()
        if not api_key:
            return 1

        cmd = [sys.executable, os.path.join(module_dir, "seg5_generate_seg_qa.py"),
               "--dataset_path", args.dataset_path, "--split", args.split,
               "--model", args.qa_model, "--base_url", args.base_url]
        if args.api_key:
            cmd.extend(["--api_key", args.api_key])
        print(f" Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n Error: SEG QA 生成失败")
            return 1
        print(" SEG Step seg5 完成！")

        # 清理中间产物
        print("\n" + "=" * 70)
        print(" 清理中间产物")
        print("=" * 70)

        splits_to_clean = ["train", "val", "test"] if args.split == "all" else [args.split]
        for split in splits_to_clean:
            split_dir = os.path.join(args.dataset_path, split)
            if not os.path.isdir(split_dir):
                continue

            qa_src = os.path.join(split_dir, f"qa_{split}.json")
            qa_dst = os.path.join(split_dir, f"QA_{split}.json")
            if os.path.exists(qa_src):
                os.rename(qa_src, qa_dst)
                print(f"  {split}/qa_{split}.json -> QA_{split}.json")

        print(" 清理完成！")

    # ==================== CD Step cd1: 提取变化区域 ====================
    if "cd1" in steps_to_run:
        print("\n" + "=" * 70)
        print("🔍 Step cd1: 提取变化区域")
        print("=" * 70)

        cmd = [sys.executable, os.path.join(module_dir, "cd1_extract_regions.py"),
               "--dataset_path", args.dataset_path, "--split", args.split]
        print(f"📂 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n❌ Error: 提取变化区域失败")
            return 1
        print("✅ Step cd1 完成！")

    # ==================== CD Step cd2: RemoteCLIP 分类 ====================
    if "cd2" in steps_to_run:
        print("\n" + "=" * 70)
        print("📝 Step cd2: RemoteCLIP 分类")
        print("=" * 70)

        cmd = [sys.executable, os.path.join(module_dir, "cd2_clip_classify.py"),
               "--dataset_path", args.dataset_path, "--split", args.split,
               "--model", args.model, "--device", args.device]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        print(f"📂 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n❌ Error: RemoteCLIP 分类失败")
            return 1
        print("✅ Step cd2 完成！")

    # ==================== CD Step cd3: VLM 审查修正分类 + 重绘图 ====================
    if "cd3" in steps_to_run:
        print("\n" + "=" * 70)
        print("🔍 Step cd3: VLM 审查修正分类 + 重绘图")
        print("=" * 70)

        api_key = _get_api_key()
        if not api_key:
            return 1

        cmd = [sys.executable, os.path.join(module_dir, "cd3_examine_result.py"),
               "--dataset_path", args.dataset_path, "--split", args.split,
               "--model", args.vlm_model, "--base_url", args.base_url,
               "--batch_size", str(args.batch_size)]
        if args.api_key:
            cmd.extend(["--api_key", args.api_key])
        print(f"📂 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n❌ Error: VLM 分类修正失败")
            return 1
        print("✅ Step cd3 完成！")

    # ==================== CD Step cd4: 局部 patch 修正 llm + metadata 汇总 ====================
    if "cd4" in steps_to_run:
        print("\n" + "=" * 70)
        print("🧠 Step cd4: 局部 patch 修正 llm + metadata 汇总")
        print("=" * 70)

        api_key = _get_api_key()
        if not api_key:
            return 1

        cmd = [sys.executable, os.path.join(module_dir, "cd4_generate_caption.py"),
               "--dataset_path", args.dataset_path, "--split", args.split,
               "--model", args.desc_model, "--base_url", args.base_url,
               "--batch_size", str(args.batch_size)]
        if args.api_key:
            cmd.extend(["--api_key", args.api_key])
        print(f"📂 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n❌ Error: Step cd4 局部 patch 修正失败")
            return 1
        print("✅ Step cd4 完成！")

    # ==================== CD Step cd5: QA 生成 + 清理 ====================
    if "cd5" in steps_to_run:
        print("\n" + "=" * 70)
        print("❓ Step cd5: 生成问答数据集")
        print("=" * 70)

        api_key = _get_api_key()
        if not api_key:
            return 1

        cmd = [sys.executable, os.path.join(module_dir, "cd5_generate_qa.py"),
               "--dataset_path", args.dataset_path, "--split", args.split,
               "--model", args.qa_model, "--base_url", args.base_url]
        if args.api_key:
            cmd.extend(["--api_key", args.api_key])
        print(f"📂 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n❌ Error: 生成数据集失败")
            return 1
        print("✅ 数据生成完成！")

        # 清理中间产物，保留 metadata.json + QA_{split}.json
        print("\n" + "=" * 70)
        print("🧹 清理中间产物")
        print("=" * 70)

        splits_to_clean = ["train", "val", "test"] if args.split == "all" else [args.split]

        for split in splits_to_clean:
            split_dir = os.path.join(args.dataset_path, split)
            if not os.path.isdir(split_dir):
                continue

            # 单轮 QA 模式：重命名 qa_{split}.json → QA_{split}.json
            qa_src = os.path.join(split_dir, f"qa_{split}.json")
            qa_dst = os.path.join(split_dir, f"QA_{split}.json")
            if os.path.exists(qa_src):
                os.rename(qa_src, qa_dst)
                print(f"  ✅ {split}/qa_{split}.json → QA_{split}.json")

            # 删除中间文件（保留 regions 目录供后续使用）
            intermediate = [
                os.path.join(split_dir, "text"),
                os.path.join(split_dir, f"caption_{split}.json"),
            ]
            for path in intermediate:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"  🗑️  删除目录: {os.path.relpath(path, args.dataset_path)}")
                elif os.path.isfile(path):
                    os.remove(path)
                    print(f"  🗑️  删除文件: {os.path.relpath(path, args.dataset_path)}")

        print("✅ 清理完成！")

    # ==================== 共用 Step mci: MCI 格式转换 ====================
    if "mci" in steps_to_run:
        print("\n" + "=" * 70)
        print("🧩 Step mci: 转换为 MCI 格式")
        print("   模式: 单轮 QA")
        print("=" * 70)

        cmd = [sys.executable, os.path.join(module_dir, "convert_qa_to_mci_format.py"),
               "--dataset_path", args.dataset_path, "--task", args.mci_task,
               "--mode", args.mode]

        print(f"📂 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n❌ Error: 转换 MCI 格式失败")
            return 1
        print("✅ Step mci 完成！")

    # ==================== 共用 Step coco: COCO 格式转换 ====================
    if "coco" in steps_to_run:
        print("\n" + "=" * 70)
        print("📋 Step coco: 转换为 COCO 格式")
        print("=" * 70)

        cmd = [sys.executable, os.path.join(module_dir, "convert_to_coco.py"),
               "--dataset_path", args.dataset_path, "--split", args.split,
               "--mode", args.mode]

        print(f"📂 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n❌ Error: 转换 COCO 格式失败")
            return 1
        print("✅ Step coco 完成！")

    # ==================== 全部完成 ====================
    print("\n" + "=" * 70)
    print("🎉 全部处理完成！")
    print("=" * 70)
    print(f"✅ 已完成步骤: {steps_to_run}")
    print(f"📁 数据集路径: {args.dataset_path}")
    print(f"📊 数据集划分: {args.split}")

    splits_final = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits_final:
        meta_path = os.path.join(args.dataset_path, s, "metadata.json")
        if os.path.exists(meta_path):
            print(f"📄 metadata: {os.path.relpath(meta_path, args.dataset_path)}")

        qa_path = os.path.join(args.dataset_path, s, f"QA_{s}.json")
        if os.path.exists(qa_path):
            print(f"📄 QA 数据: {os.path.relpath(qa_path, args.dataset_path)}")

    # COCO 文件
    if "coco" in steps_to_run:
        coco_path = os.path.join(args.dataset_path, "coco_annotations.json")
        if os.path.exists(coco_path):
            print(f"📄 COCO 标注: {os.path.relpath(coco_path, args.dataset_path)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
