"""Render tiled class maps to RGB pseudo-color using the config palette."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from labelme_config import class_map_to_rgb, load_class_map

DEFAULT_LABEL_DIR = r"F:\岩石\0822数据集\infer\label"


def render_tiles(label_dir: Path, out_dir: Path) -> int:
    """
    入参:
    - label_dir: 类别图切片目录（uint8 类别索引 PNG）。
    - out_dir: RGB 可视化输出目录。

    方法:
    - 对每张类别图执行 labelme_config.class_map_to_rgb 调色板查表
      （颜色来自 config.yaml classes.palette_rgb），与类别图同名保存。

    出参:
    - int: 渲染的切片数量。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for label_path in sorted(label_dir.glob("*.png")):
        rgb = class_map_to_rgb(load_class_map(label_path))
        Image.fromarray(rgb, mode="RGB").save(out_dir / label_path.name)
        rendered += 1
    return rendered


def main() -> None:
    """
    入参:
    - CLI: --label-dir 类别图目录，--out-dir 输出目录。

    方法:
    - 扫描类别图目录全部 PNG，按 config 调色板渲染 RGB 后写入输出目录。

    出参:
    - 无；RGB 可视化写入 --out-dir。
    """
    parser = argparse.ArgumentParser(description="Render class maps to RGB with the config palette")
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR, help="类别图切片目录")
    parser.add_argument("--out-dir", default=None, help="RGB 输出目录，默认 label 目录的兄弟目录 label_rgb")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label dir: {label_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else label_dir.with_name("label_rgb")
    rendered = render_tiles(label_dir, out_dir)
    print(f"{rendered} tiles rendered -> {out_dir}")


if __name__ == "__main__":
    main()