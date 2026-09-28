"""Overlay LabelMe polygon annotations onto their source images for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (255, 64, 64),
    (64, 160, 255),
    (64, 200, 64),
    (255, 200, 0),
    (200, 64, 255),
    (64, 255, 200),
    (255, 128, 64),
    (128, 64, 255),
    (255, 255, 64),
    (64, 128, 255),
]


def assign_colors(annotation: dict) -> dict[str, tuple[int, int, int]]:
    """入参: LabelMe json；方法: 按 shape 出现顺序为每个 label 分配调色板颜色；出参: label 到 RGB 映射。"""
    colors = {}
    for shape in annotation.get("shapes", []):
        label = shape["label"]
        if label not in colors:
            colors[label] = PALETTE[len(colors) % len(PALETTE)]
    return colors


def find_image(source_root: Path, stem: str, annotation: dict) -> Path:
    """入参: 源目录、json 文件名主干、json 内容；方法: 优先 imagePath 字段，其次同主干 tif/jpg；出参: 影像路径。"""
    image_path = annotation.get("imagePath")
    candidates = []
    if image_path:
        candidates.append(source_root / Path(image_path).name)
    candidates += [source_root / f"{stem}.tif", source_root / f"{stem}.jpg"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(source_root.glob(f"{stem}.*"))
    matches = [path for path in matches if path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}]
    if not matches:
        raise FileNotFoundError(f"No source image found for {stem} in {source_root}")
    return matches[0]


def render_sample(source_root: Path, json_path: Path, out_dir: Path, alpha: float) -> Path:
    """入参: 源目录、标注 json、输出目录、叠加透明度；方法: 栅格化 polygon 半透明叠加到影像并绘制轮廓与图例；出参: 输出 PNG 路径。"""
    annotation = json.loads(json_path.read_text(encoding="utf-8"))
    stem = json_path.stem
    image_file = find_image(source_root, stem, annotation)
    with Image.open(image_file) as source:
        image = np.asarray(source.convert("RGB")).copy()
    colors = assign_colors(annotation)
    overlay = image.copy()
    for shape in annotation.get("shapes", []):
        points = [tuple(point) for point in shape["points"]]
        color = colors[shape["label"]]
        canvas = Image.new("RGB", (overlay.shape[1], overlay.shape[0]), color)
        mask = Image.new("L", (overlay.shape[1], overlay.shape[0]), 0)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        mask_array = np.asarray(mask)[..., None].astype(bool)
        overlay[mask_array[..., 0]] = color
    blended = (image.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    canvas = Image.fromarray(blended)
    draw = ImageDraw.Draw(canvas)
    for shape in annotation.get("shapes", []):
        draw.line([tuple(point) for point in shape["points"]] + [tuple(shape["points"][0])], fill=colors[shape["label"]], width=3)
    legend_height = 34 * len(colors) + 20
    legend = Image.new("RGB", (blended.shape[1], legend_height), (255, 255, 255))
    legend_draw = ImageDraw.Draw(legend)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    for index, (label, color) in enumerate(colors.items()):
        top = 10 + index * 34
        legend_draw.rectangle((20, top, 52, top + 24), fill=color, outline=(0, 0, 0), width=1)
        legend_draw.text((64, top - 4), f"{label}", fill=(0, 0, 0), font=font)
    result = np.vstack([np.asarray(canvas), np.asarray(legend)])
    out_path = out_dir / f"{stem}_标注可视化.png"
    Image.fromarray(result).save(out_path)
    return out_path


def main() -> None:
    """
    入参:
    - CLI: --source-root 标注与影像目录、--out-dir 输出目录、--alpha 叠加透明度。

    方法:
    - 遍历源目录全部 json，逐个将 polygon 叠加到对应影像并保存可视化。

    出参:
    - 无；每份标注输出一张 {stem}_标注可视化.png。
    """
    parser = argparse.ArgumentParser(description="Overlay LabelMe annotations onto source images")
    parser.add_argument("--source-root", default=r"F:\岩石\0822标注\全景", help="标注 json 与影像所在目录")
    parser.add_argument("--out-dir", default=None, help="输出目录，默认源目录下的 可视化 子目录")
    parser.add_argument("--alpha", type=float, default=0.45, help="polygon 填充叠加透明度")
    args = parser.parse_args()
    source_root = Path(args.source_root)
    out_dir = Path(args.out_dir) if args.out_dir else source_root / "可视化"
    out_dir.mkdir(parents=True, exist_ok=True)
    for json_path in sorted(source_root.glob("*.json")):
        out_path = render_sample(source_root, json_path, out_dir, args.alpha)
        print(f"{json_path.name} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
