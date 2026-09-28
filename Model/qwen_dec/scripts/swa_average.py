"""
SWA (Stochastic Weight Averaging) for LoRA + external adapter.

做什么:
    加载多个 checkpoint 的 adapter 权重做算术平均，生成一个"平均版"adapter。
    LoRA 权重是线性相加关系（W = W0 + BA），平均多个 BA 等价于平均多个 LoRA 增量，
    数学上是合法的（不需要重新训练）。

为什么:
    训练 F1 在 epoch 8-14 之间剧烈震荡（0.39-0.49），峰值是"撞到"的。
    多个 epoch 权重平均能平滑训练轨迹，对抗震荡，稳定提升泛化。

用法:
    python scripts/swa_average.py
    # 默认平均 best_adapter + checkpoint-308（F1 最高的两个）
    # 输出到 output/pipe_defect_lora_baseline_restore/swa_adapter
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def average_safetensors(paths: list[Path], out_path: Path) -> None:
    """入参: 多个 safetensors 文件路径 + 输出路径; 方法: 逐 key 算术平均; 出参: 无（写出新文件）。"""
    print(f"[swa] averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  - {p}")

    # 累加所有 state_dict
    accum: dict[str, torch.Tensor] = None
    for path in paths:
        sd = load_file(str(path))
        if accum is None:
            accum = {k: v.clone().float() for k, v in sd.items()}
        else:
            # 校验 key 一致
            assert set(sd.keys()) == set(accum.keys()), f"key mismatch in {path}"
            for k in accum:
                accum[k] += sd[k].float()

    # 求平均
    n = len(paths)
    for k in accum:
        accum[k] /= n

    # 转回原 dtype（用第一个 key 的 dtype 作为参考；LoRA 权重通常都是同一 dtype）
    ref_keys = list(load_file(str(paths[0])).keys())
    ref_dtype = load_file(str(paths[0]))[ref_keys[0]].dtype
    accum = {k: v.to(ref_dtype) for k, v in accum.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(accum, str(out_path))
    print(f"[swa] saved averaged weights to {out_path} (n={n}, dtype={ref_dtype})")


def average_external_adapter(paths: list[Path], out_path: Path) -> None:
    """入参: 多个 external_feature_adapter.pt 路径 + 输出路径; 方法: 逐 key 算术平均; 出参: 无。"""
    print(f"[swa] averaging {len(paths)} external_feature_adapter.pt:")
    accum: dict[str, torch.Tensor] = None
    for path in paths:
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        # 过滤非 tensor（metadata dict）
        sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
        if accum is None:
            accum = {k: v.clone().float() for k, v in sd.items()}
        else:
            assert set(sd.keys()) == set(accum.keys()), f"key mismatch in {path}"
            for k in accum:
                accum[k] += sd[k]
    n = len(paths)
    for k in accum:
        accum[k] = (accum[k] / n).to(torch.bfloat16)  # external 用 bf16
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(accum, str(out_path))
    print(f"[swa] saved averaged external adapter to {out_path} (n={n})")


def main():
    """入参: --ckpt-dirs（多个 checkpoint 目录）; 方法: 平均 LoRA + external; 出参: 无。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dirs", nargs="+", required=True,
        help="checkpoint 目录列表（每个应含 adapter_model.safetensors + external_feature_adapter.pt）",
    )
    parser.add_argument("--out-dir", required=True, help="输出目录")
    args = parser.parse_args()

    ckpt_dirs = [Path(d) for d in args.ckpt_dirs]
    out_dir = Path(args.out_dir)

    # 校验所有 checkpoint 完整
    for d in ckpt_dirs:
        assert (d / "adapter_model.safetensors").exists(), f"missing adapter_model.safetensors in {d}"
        assert (d / "external_feature_adapter.pt").exists(), f"missing external_feature_adapter.pt in {d}"

    # 平均 LoRA 权重
    average_safetensors(
        [d / "adapter_model.safetensors" for d in ckpt_dirs],
        out_dir / "adapter_model.safetensors",
    )

    # 平均 external adapter
    average_external_adapter(
        [d / "external_feature_adapter.pt" for d in ckpt_dirs],
        out_dir / "external_feature_adapter.pt",
    )

    # 拷贝非权重文件（config、tokenizer 等）从第一个 checkpoint
    ref = ckpt_dirs[0]
    for item in ref.iterdir():
        if item.name in ("adapter_model.safetensors", "external_feature_adapter.pt", "optimizer.pt",
                         "rng_state.pth", "scheduler.pt", "trainer_state.json", "training_args.bin"):
            continue
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)
            print(f"[swa] copied {item.name}")

    print(f"\n[swa] DONE. averaged adapter at: {out_dir}")
    print(f"[swa] evaluate with:")
    print(f"  python evaluate_detection.py --adapter {out_dir} --test-jsonl ...")


if __name__ == "__main__":
    main()
