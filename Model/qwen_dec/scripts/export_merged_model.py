# -*- coding: utf-8 -*-
"""v10 LoRA 合并导出（L0 轻量化：消除推理期 Qwen 侧 LoRA 分支计算，产出单权重部署形态）。

做什么: 加载基座 + best_adapter（语言/视觉塔 LoRA + external_feature_adapter.pt），
       先恢复外部链权重，再把外层 PEFT LoRA（Qwen 语言 decoder + 原生视觉塔）merge_and_unload
       焊进基座导出 merged_qwen\；SigLIP2 内层 LoRA 保留原协议（86M 模型 LoRA 开销可忽略，
       external_feature_adapter.pt 加载路径不变），连同 FPN/融合链一起复制到导出目录。
为什么: 推理期未合并的 LoRA 每层多两次低秩矩阵乘；合并后等价单权重，提速约 5-10%，
       且 Qwen 侧部署不再依赖 peft。约束解码（白名单 mask）与 QUANT=int4 量化加载与合并形态正交。
用法: python scripts\export_merged_model.py --adapter F:\管网\runs\...\train\best_adapter --out F:\管网\runs\...\merged
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from model.qwen_feature_injection import FeatureInjectedQwen3VLForConditionalGeneration, EXTERNAL_ADAPTER_FILE


def parse_args() -> argparse.Namespace:
    """入参: 无。方法: 解析 --adapter（训练 best_adapter 目录）、--model（基座）、--out（导出根目录）。出参: Namespace。"""
    parser = argparse.ArgumentParser(description="Merge v10 Qwen-side LoRA into standalone deployment weights.")
    parser.add_argument("--adapter", required=True, help="train/best_adapter 目录（含 LoRA 与 external_feature_adapter.pt）")
    parser.add_argument("--model", default=r"F:/pre/Qwen3-VL-4B-Instruct", help="基座 Qwen3-VL 目录")
    parser.add_argument("--siglip", default=r"F:/pre/siglip2-base-patch16-224", help="SigLIP2 Base 目录（部署时仍需）")
    parser.add_argument("--out", required=True, help="导出根目录（merged_qwen\\ + external_feature_adapter.pt）")
    return parser.parse_args()


def main() -> None:
    """入参: 无（命令行配置）。方法: 加载→恢复外部链→合并 Qwen 侧 LoRA→导出与协议文件复制。
    出参: 无（产物落盘并打印部署说明）。"""
    args = parse_args()
    adapter = str(Path(args.adapter).resolve()).replace("\\", "/")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[merge] base: {args.model}\n[merge] adapter: {adapter}")
    base = FeatureInjectedQwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cpu",
    )
    base.bind_external_extractor(args.siglip, torch.bfloat16)

    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(base, adapter)
    feature_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    # 先恢复外部链（SigLIP LoRA/FPN/projector），再合并——合并只动 Qwen 侧，外部协议不变
    feature_model.load_external_adapter(adapter, strict=True)
    merged = peft_model.merge_and_unload()

    qwen_dir = out_root / "merged_qwen"
    print(f"[merge] exporting merged Qwen -> {qwen_dir}")
    merged.save_pretrained(qwen_dir, safe_serialization=True)

    shutil.copy2(Path(adapter) / EXTERNAL_ADAPTER_FILE, out_root / EXTERNAL_ADAPTER_FILE)
    print(f"[merge] copied {EXTERNAL_ADAPTER_FILE} -> {out_root}")
    print("[merge] done. 部署：MODEL_PATH=<merged_qwen> + 原生 SigLIP2 目录 + 本目录 external_feature_adapter.pt；"
          "约束解码与 QUANT=int4 照常可用。")


if __name__ == "__main__":
    main()