"""Stack per-sample visualizations of the ablation configs into one comparison image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def parse_runs(arguments: list[str]) -> list[tuple[str, Path, float]]:
    """
    入参:
    - arguments: 形如 "name=run_dir" 的 CLI 参数列表。

    方法:
    - 拆分配置名与实验目录，读取 best_val_metrics.json 的 foreground_miou
      作为标签中的量化指标，缺失指标时以 nan 占位。

    出参:
    - list[tuple[str, Path, float]]: (配置名, 实验目录, fg_mIoU) 列表。
    """
    runs = []
    for argument in arguments:
        name, _, directory = argument.partition("=")
        metrics_path = Path(directory) / "best_val_metrics.json"
        if metrics_path.is_file():
            miou = float(json.loads(metrics_path.read_text(encoding="utf-8"))["foreground_miou"])
        else:
            miou = float("nan")
        runs.append((name, Path(directory), miou))
    return runs


def label_strip(width: int, text: str) -> Image.Image:
    """
    入参:
    - width: 与下方可视化图等宽的条带宽度。
    - text: 配置名与指标组成的标签文本。

    方法:
    - 生成 44px 深色条带并左对齐绘制白色文本，便于在堆叠图中定位配置。

    出参:
    - Image.Image: 带文本的标签条带。
    """
    strip = Image.new("RGB", (width, 44), (24, 24, 24))
    draw = ImageDraw.Draw(strip)
    draw.text((16, 8), text, fill=(255, 255, 255), font=ImageFont.truetype(FONT_PATH, 26))
    return strip


def main() -> None:
    """
    入参:
    - CLI 参数：--runs name=dir 列表（顺序即自上而下堆叠顺序）、--view 可视化
      图名（默认 mask_overlay）、--out-dir 输出目录。

    方法:
    - 取全部实验 VIS 目录的样本交集，逐样本将各配置的同一视图加标签条带后
    垂直堆叠为单张对比图，样本按编号数值排序。

    出参:
    - None: 对比图写入 --out-dir，命名 sample<N>_<view>.compare.png。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="name=run_dir 列表，顺序即自上而下堆叠顺序")
    parser.add_argument("--view", default="mask_overlay", choices=["mask", "mask_overlay", "edge_lines", "edge_overlay"])
    parser.add_argument("--out-dir", default=r"F:\岩石\runs\exp_calss_edge\vis_compare")
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_sets = [{path.name for path in (run_dir / "VIS").iterdir() if path.is_dir()} for _, run_dir, _ in runs]
    common_samples = sorted(set.intersection(*sample_sets), key=int)
    for sample in common_samples:
        blocks = []
        for name, run_dir, miou in runs:
            view = Image.open(run_dir / "VIS" / sample / f"{args.view}.png").convert("RGB")
            label = f"{name}  fg_mIoU={miou:.4f}" if miou == miou else name
            blocks.append(label_strip(view.width, label))
            blocks.append(view)
        canvas = Image.new("RGB", (blocks[0].width, sum(block.height for block in blocks)), (0, 0, 0))
        offset = 0
        for block in blocks:
            canvas.paste(block, (0, offset))
            offset += block.height
        output_path = out_dir / f"sample{sample}_{args.view}.compare.png"
        canvas.save(output_path)
        print(f"sample{sample}: stacked {len(runs)} configs -> {output_path}", flush=True)
    print(f"DONE: {len(common_samples)} comparison images under {out_dir}")


if __name__ == "__main__":
    main()
