"""Prepare paired image/label dataset: split 7:1:2, fishnet 256 tiles, palette visualization.

入参(全局):
- --input-dir: 含 image\\ 与 label\\ 子目录的配对文件夹（同名配对；label 侧优先同名
  LabelMe json 并运行时栅格化，无 json 时回退 0..N-1 类别图）。
- --output-root: 数据集输出（train/val/test 子目录 + jsonl + manifest）。
- --vis-dir: 可视化输出（每个输入图一个文件夹，四图）。
- --seed: 随机打乱种子，默认 1337。

方法:
- 按文件名排序 + 固定种子打乱，按总张数整数分配 train/val/test（73 用
  round(0.7n) / round(0.2n)，val 取剩余）；每张图按 split 步长渔网切片
  （train 256/192 重叠并过滤全背景，val/test 256 无重叠全保留），标签同步
  生成逐类 3×3 形态学梯度边缘；并行生成该图的调色板可视化四图。

出参:
- 无；数据集与可视化写入 --output-root/--vis-dir。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from labelme_config import LABELME_CLASS_NAMES, LABELME_NUM_CLASSES, LABELME_PALETTE_RGB, class_map_to_rgb, save_class_map
from prepare_labelme_dataset import TILE_SIZE, generate_class_edge_labels, generate_edge_label, pad_tile, rasterize_labelme, tile_starts

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def add_legend(image: np.ndarray, class_names: tuple | list) -> np.ndarray:
    """入参: RGB 预测图与类别名；方法: 底部追加 64px 白色条绘制调色板色块与名称；出参: 带图例 RGB。"""
    height, width = image.shape[:2]
    canvas = np.full((height + 64, width, 3), 255, dtype=np.uint8)
    canvas[:height] = image
    legend_image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(legend_image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    x = 20
    for class_id, name in enumerate(class_names[:LABELME_NUM_CLASSES]):
        color = LABELME_PALETTE_RGB[class_id]
        text = f"{class_id}: {name}"
        draw.rectangle((x, height + 14, x + 24, height + 38), fill=color, outline=(0, 0, 0), width=1)
        draw.text((x + 32, height + 12), text, fill=(0, 0, 0), font=font)
        x += 32 + draw.textbbox((0, 0), text, font=font)[2] + 28
    return np.asarray(legend_image)


def render_class_edges(edge_map: np.ndarray) -> np.ndarray:
    """入参: [K,H,W] 类别边缘图；方法: 按调色板渲染彩色线；出参: HxWx3 RGB。"""
    height, width = edge_map.shape[1:]
    rendered = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id in range(min(edge_map.shape[0], LABELME_NUM_CLASSES - 1)):
        rendered[edge_map[class_id] > 0] = LABELME_PALETTE_RGB[class_id + 1]
    return rendered


def blend_edges(rgb: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
    """入参: RGB 原图与类别边缘图；方法: 边缘以调色板颜色覆盖到原图；出参: 叠加图。"""
    overlay = rgb.copy()
    for class_id in range(min(edge_map.shape[0], LABELME_NUM_CLASSES - 1)):
        overlay[edge_map[class_id] > 0] = LABELME_PALETTE_RGB[class_id + 1]
    return overlay


def load_class_map(path: Path) -> np.ndarray:
    """入参: 标签路径；方法: json 走 LabelMe 栅格化（矢量权威），png 直接读类别图；出参: 0..N-1 类别图。"""
    if path.suffix.lower() == ".json":
        return rasterize_labelme(json.loads(path.read_text(encoding="utf-8")))
    with Image.open(path) as label_file:
        return np.asarray(label_file.convert("L"))


def find_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """入参: 输入根目录；方法: 按同名配对扫描 image\\ 与 label\\ 子目录；label 侧优先同名 json（LabelMe），无则回退类别图；出参: (影像, 标签) 路径列表。"""
    image_dir = input_dir / "image"
    label_dir = input_dir / "label"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expect image\\ and label\\ subdirs under {input_dir}")
    image_names = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS]
    pairs = []
    for image_path in sorted(image_names):
        label_path = label_dir / f"{image_path.stem}.json"
        if not label_path.is_file():
            label_path = next((label_dir / f"{image_path.stem}{suffix}" for suffix in IMAGE_EXTS if (label_dir / f"{image_path.stem}{suffix}").is_file()), None)
        if label_path is None:
            print(f"[warn] no paired label for {image_path.name}")
            continue
        pairs.append((image_path, label_path))
    return pairs


def split_assignments(count: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    """
    入参:
    - count: 总张数。
    - seed: 随机打乱种子。

    方法:
    - 固定种子打乱后分配：count 为 3 的倍数时严格均分（train=val=test=count/3，
      保证验证集存在）；否则按 7:1:2 取整分配，并保底至少 1 张验证（从训练集
      挪用），避免小样本下验证集为空。

    出参:
    - tuple: (train, val, test) 索引列表。
    """
    rng = np.random.default_rng(seed)
    indices = list(range(count))
    rng.shuffle(indices)
    if count >= 3 and count % 3 == 0:
        third = count // 3
        return indices[:third], indices[third : 2 * third], indices[2 * third :]
    n_train = int(round(count * 0.7))
    n_test = int(round(count * 0.2))
    n_val = count - n_train - n_test
    if n_val < 1 and n_train > 1:
        n_train -= 1
        n_val = 1
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


def split_tiles(image: np.ndarray, class_map: np.ndarray, edge_maps: np.ndarray, stem: str, split: str,
                output_root: Path, records: list[dict]) -> None:
    """
    入参: 影像/类别图/逐类边缘、样本名、split、输出根与记录容器。

    方法: train 用 192 步长重叠切片并过滤全背景，val/test 用 256 无重叠全保留；
    影像/类别/边缘逐类及并集同步保存，tile_origin 记录原图坐标。

    出参: 无（records 原地追加）。
    """
    stride = TILE_SIZE if split in ("val", "test") else int(TILE_SIZE * 0.75)
    image_dir = output_root / split / "image"
    label_dir = output_root / split / "label"
    boundary_dir = output_root / split / "boundary"
    class_boundary_dirs = [output_root / split / f"boundary_class{class_id}" for class_id in range(1, LABELME_NUM_CLASSES)]
    for directory in (image_dir, label_dir, boundary_dir, *class_boundary_dirs):
        directory.mkdir(parents=True, exist_ok=True)
    union_edge = np.any(edge_maps > 0, axis=0).astype(np.uint8) * 255
    for top in tile_starts(image.shape[0], stride):
        for left in tile_starts(image.shape[1], stride):
            label_tile = pad_tile(class_map, top, left, 0)
            if split == "train" and not np.any(label_tile):
                continue
            tile_name = f"{stem}_{top:05d}_{left:05d}.png"
            Image.fromarray(pad_tile(image, top, left, 0), mode="RGB").save(image_dir / tile_name)
            save_class_map(label_tile, label_dir / tile_name)
            Image.fromarray(pad_tile(union_edge, top, left, 0), mode="L").save(boundary_dir / tile_name)
            for class_id in range(1, LABELME_NUM_CLASSES):
                Image.fromarray(pad_tile(edge_maps[class_id - 1], top, left, 0), mode="L").save(class_boundary_dirs[class_id - 1] / tile_name)
            records.append({
                "image": f"{split}/image/{tile_name}",
                "mask": f"{split}/label/{tile_name}",
                "boundary": f"{split}/boundary/{tile_name}",
                "boundaries": [f"{split}/boundary_class{class_id}/{tile_name}" for class_id in range(1, LABELME_NUM_CLASSES)],
                "source_sample": stem,
                "tile_origin": [left, top],
            })


def render_vis(image: np.ndarray, class_map: np.ndarray, stem: str, vis_dir: Path) -> None:
    """入参: 整图、类别图、样本名、可视化根；方法: 输出每图一文件夹四图（带图例）；出参: 无。"""
    out = vis_dir / stem
    out.mkdir(parents=True, exist_ok=True)
    mask = class_map_to_rgb(class_map)
    edge_maps = generate_class_edge_labels(class_map)
    edge_rgb = render_class_edges(edge_maps)
    mask_overlay = (0.55 * image.astype(np.float32) + 0.45 * mask.astype(np.float32)).clip(0, 255).astype(np.uint8)
    edge_overlay = blend_edges(image, edge_maps)
    cv2.imencode(".png", cv2.cvtColor(add_legend(mask, LABELME_CLASS_NAMES), cv2.COLOR_RGB2BGR))[1].tofile(out / "mask.png")
    cv2.imencode(".png", cv2.cvtColor(add_legend(mask_overlay, LABELME_CLASS_NAMES), cv2.COLOR_RGB2BGR))[1].tofile(out / "mask_overlay.png")
    cv2.imencode(".png", cv2.cvtColor(edge_rgb, cv2.COLOR_RGB2BGR))[1].tofile(out / "edge_lines.png")
    cv2.imencode(".png", cv2.cvtColor(edge_overlay, cv2.COLOR_RGB2BGR))[1].tofile(out / "edge_overlay.png")
    print(f"[vis] {stem} -> {out}")


def main() -> None:
    """入参: CLI；方法: 配对扫描→7:1:2 划分→逐图切片与可视化→写 jsonl/manifest；出参: 无。"""
    parser = argparse.ArgumentParser(description="Prepare paired image/label dataset with 7:1:2 split")
    parser.add_argument("--input-dir", required=True, help="含 image\\ 与 label\\ 同名配对子目录")
    parser.add_argument("--output-root", required=True, help="数据集输出根（train/val/test）")
    parser.add_argument("--vis-dir", required=True, help="可视化输出根（每图一文件夹）")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    vis_dir = Path(args.vis_dir)
    pairs = find_pairs(Path(args.input_dir))
    if not pairs:
        raise RuntimeError(f"No paired image/label found under {args.input_dir}")
    train_idx, val_idx, test_idx = split_assignments(len(pairs), args.seed)
    print(f"[split] {len(pairs)} images -> train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} (seed {args.seed})")

    records = {split: [] for split in ("train", "val", "test")}
    for split, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for index in indices:
            image_path, label_path = pairs[index]
            stem = image_path.stem
            with Image.open(image_path) as image_file:
                image = np.asarray(image_file.convert("RGB"))
            class_map = np.asarray(load_class_map(label_path))
            if image.shape[:2] != class_map.shape:
                raise ValueError(f"{stem} image/label size mismatch: {image.shape[:2]} vs {class_map.shape}")
            if np.unique(class_map).max() >= LABELME_NUM_CLASSES:
                raise ValueError(f"{stem} label has class id out of range 0..{LABELME_NUM_CLASSES - 1}")
            edge_maps = generate_class_edge_labels(class_map)
            split_tiles(image, class_map, edge_maps, stem, split, output_root, records[split])
            render_vis(image, class_map, stem, vis_dir)

    for split, recs in records.items():
        with (output_root / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for record in recs:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{split}] tiles={len(recs)}")
    manifest = {
        "source_root": str(Path(args.input_dir)),
        "output_root": str(output_root),
        "task": "paired_7_1_2_lithology",
        "tile_size": TILE_SIZE,
        "train_stride": int(TILE_SIZE * 0.75),
        "val_test_stride": TILE_SIZE,
        "filter_empty_background": True,
        "auxiliary_target": "per_class_morphological_gradient_union",
        "auxiliary_channels": LABELME_NUM_CLASSES - 1,
        "splits": {split: {"images": len(recs_src), "tiles": len(records[split])}
                   for split, recs_src in [("train", train_idx), ("val", val_idx), ("test", test_idx)]},
        "label_mapping": {name: class_id for name, class_id in zip(LABELME_CLASS_NAMES, range(LABELME_NUM_CLASSES))},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
