"""Prepare tiled lithology data from the 0820 LabelMe polygon annotations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from labelme_config import (
    DATASET_CONFIG,
    LABELME_NUM_CLASSES,
    LABELME_STRING_TO_CLASS_ID,
    save_class_map,
)

SOURCE_ROOT = Path(DATASET_CONFIG["source_root"])
LABEL_DIR = SOURCE_ROOT / "label"
IMAGE_DIR = Path(DATASET_CONFIG.get("image_dir", SOURCE_ROOT / "拼图"))
OUTPUT_ROOT = Path(DATASET_CONFIG["output_root"])
TILE_SIZE = DATASET_CONFIG["tile_size"]
OVERLAP_RATIO = DATASET_CONFIG["overlap_ratio"]
STRIDE = int(TILE_SIZE * (1.0 - OVERLAP_RATIO))
# 推理期（val 整图拼接与 infer 无标注）使用无重叠 256 网格，训练保留 25% 重叠
INFER_STRIDE = DATASET_CONFIG["infer_stride"]
SPLITS = DATASET_CONFIG["splits"]
EDGE_KERNEL_SIZE = DATASET_CONFIG["edge_kernel_size"]


def tile_starts(length: int, stride: int = STRIDE) -> list[int]:
    """
    入参:
    - length: 原始图像在一个方向的像素长度。
    - stride: 切片步长；训练切片为 192（25% 重叠），推理切片为 256（无重叠）。

    方法:
    - 以给定步长生成裁剪起点，边缘不足一个 tile 的区域由 padding 补齐。

    出参:
    - list[int]: 覆盖整个方向的裁剪起点。
    """
    return list(range(0, max(1, length), stride))


def pad_tile(array: np.ndarray, top: int, left: int, fill_value: int) -> np.ndarray:
    """
    入参:
    - array: 原始图像、类别或边缘数组。
    - top/left: 裁剪左上角坐标。
    - fill_value: 右边和上边补齐使用的值。

    方法:
    - 截取至图像边界后，以常量补齐到固定 TILE_SIZE，图像背景为黑色、标签背景为 0。

    出参:
    - np.ndarray: 固定 TILE_SIZE x TILE_SIZE 的裁剪结果。
    """
    tile = array[top : top + TILE_SIZE, left : left + TILE_SIZE]
    height, width = tile.shape[:2]
    pad_bottom = TILE_SIZE - height
    pad_right = TILE_SIZE - width
    if array.ndim == 3:
        padding = ((0, pad_bottom), (0, pad_right), (0, 0))
    else:
        padding = ((0, pad_bottom), (0, pad_right))
    return np.pad(tile, padding, mode="constant", constant_values=fill_value)


def rasterize_labelme(annotation: dict) -> np.ndarray:
    """
    入参:
    - annotation: LabelMe JSON 解析结果，含 imageHeight/imageWidth/shapes。

    方法:
    - 以 LabelMe 绘制顺序（后绘覆盖先绘）将 polygon 逐个填充到单通道类别图，
      未知标签字符串直接报错，避免静默丢类。

    出参:
    - np.ndarray: HxW uint8 类别索引图，值域 0 到 4。
    """
    height, width = int(annotation["imageHeight"]), int(annotation["imageWidth"])
    canvas = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    for shape in annotation.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            raise ValueError(f"Unsupported LabelMe shape_type: {shape.get('shape_type')}")
        label = shape["label"]
        if label not in LABELME_STRING_TO_CLASS_ID:
            raise ValueError(f"Unknown LabelMe label string: {label}")
        drawer.polygon([tuple(point) for point in shape["points"]], fill=LABELME_STRING_TO_CLASS_ID[label])
    return np.asarray(canvas, dtype=np.uint8)


def generate_class_edge_labels(class_map: np.ndarray, kernel_size: int = EDGE_KERNEL_SIZE) -> np.ndarray:
    """
    入参:
    - class_map: HxW uint8 类别索引图，0 为背景，1..4 为前景类别。
    - kernel_size: 形态学梯度核边长，必须为正奇数。

    方法:
    - 对每个前景类别分别执行膨胀−腐蚀形态学梯度，不把类别边缘提前合并；
      输出是 multi-label 边缘，类别交界处允许多个通道同时为正。

    出参:
    - np.ndarray: [4,H,W] uint8 类别边缘，255 为对应类别边缘。
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edge_count = LABELME_NUM_CLASSES - 1
    class_edges = np.zeros((edge_count, *class_map.shape), dtype=np.uint8)
    for class_id in range(1, LABELME_NUM_CLASSES):
        binary = (class_map == class_id).astype(np.uint8) * 255
        class_edges[class_id - 1] = cv2.absdiff(cv2.dilate(binary, kernel), cv2.erode(binary, kernel))
    return class_edges


def generate_edge_label(class_map: np.ndarray, kernel_size: int = EDGE_KERNEL_SIZE) -> np.ndarray:
    """
    入参:
    - class_map: HxW uint8 类别索引图。
    - kernel_size: 形态学梯度核边长。

    方法:
    - 调用类别分通道边缘提取，再沿通道取并集，保留旧单通道边缘接口。

    出参:
    - np.ndarray: HxW uint8 union edge，255 为任一前景类别边缘。
    """
    return np.any(generate_class_edge_labels(class_map, kernel_size) > 0, axis=0).astype(np.uint8) * 255


def find_image(sample_id: int) -> Path:
    """
    入参:
    - sample_id: 样本编号，对应 拼图/N.jpg。

    方法:
    - 在 IMAGE_DIR 内查找同名 jpg，缺失时报错。

    出参:
    - Path: 原始图像路径。
    """
    image_path = IMAGE_DIR / f"{sample_id}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing source image: {image_path}")
    return image_path


def prepare_labeled_sample(sample_id: int, split: str) -> list[dict]:
    """
    入参:
    - sample_id: 标注样本编号（train 为 2/3，val 为 4）。
    - split: 输出划分名称；train 使用 192 步长（25% 重叠），val 使用 256 步长
      （无重叠，与整图推理网格一致）。

    方法:
    - 栅格化 LabelMe polygon 为类别图并生成形态学梯度边缘图，按划分步长
      同步裁剪图像/类别/边缘为 256x256 PNG；train 写盘前过滤类别完全为
      背景的切片，val 保留全部切片以保证整图拼接覆盖完整。

    出参:
    - list[dict]: 当前样本生成的切片记录。
    """
    stride = STRIDE if split == "train" else INFER_STRIDE
    annotation = json.loads((LABEL_DIR / f"{sample_id}.json").read_text(encoding="utf-8"))
    with Image.open(find_image(sample_id)) as image_file:
        image = np.asarray(image_file.convert("RGB"))
    class_map = rasterize_labelme(annotation)
    if image.shape[:2] != class_map.shape:
        raise ValueError(f"sample{sample_id} image/label shape mismatch: {image.shape[:2]} vs {class_map.shape}")
    edge_map = generate_edge_label(class_map)
    class_edge_maps = generate_class_edge_labels(class_map)

    image_dir = OUTPUT_ROOT / split / "image"
    label_dir = OUTPUT_ROOT / split / "label"
    boundary_dir = OUTPUT_ROOT / split / "boundary"
    class_boundary_dirs = [OUTPUT_ROOT / split / f"boundary_class{class_id}" for class_id in range(1, 5)]
    for directory in (image_dir, label_dir, boundary_dir, *class_boundary_dirs):
        directory.mkdir(parents=True, exist_ok=True)
    save_class_map(class_map, OUTPUT_ROOT / f"sample{sample_id}_class.png")
    Image.fromarray(edge_map, mode="L").save(OUTPUT_ROOT / f"sample{sample_id}_edge.png")

    records = []
    for top in tile_starts(image.shape[0], stride):
        for left in tile_starts(image.shape[1], stride):
            tile_name = f"sample{sample_id}_{top:05d}_{left:05d}.png"
            image_tile = pad_tile(image, top, left, 0)
            label_tile = pad_tile(class_map, top, left, 0)
            boundary_tile = pad_tile(edge_map, top, left, 0)
            class_boundary_tiles = [pad_tile(class_edge_maps[class_id - 1], top, left, 0) for class_id in range(1, 5)]
            if split == "train" and not np.any(label_tile):
                continue
            Image.fromarray(image_tile, mode="RGB").save(image_dir / tile_name)
            Image.fromarray(label_tile, mode="L").save(label_dir / tile_name)
            Image.fromarray(boundary_tile, mode="L").save(boundary_dir / tile_name)
            for class_id, class_boundary_tile in enumerate(class_boundary_tiles, start=1):
                Image.fromarray(class_boundary_tile, mode="L").save(class_boundary_dirs[class_id - 1] / tile_name)
            records.append({
                "image": f"{split}/image/{tile_name}",
                "mask": f"{split}/label/{tile_name}",
                "boundary": f"{split}/boundary/{tile_name}",
                "boundaries": [f"{split}/boundary_class{class_id}/{tile_name}" for class_id in range(1, 5)],
                "source_sample": f"sample{sample_id}",
                "tile_origin": [left, top],
            })
    return records


def prepare_infer_sample(sample_id: int, split: str) -> list[dict]:
    """
    入参:
    - sample_id: 未标注样本编号（infer 为 1-26）。
    - split: 输出划分名称，固定为 infer。

    方法:
    - 仅按无重叠 256 网格裁剪图像（无类别/边缘真值），保留全部切片供整图推理拼接。

    出参:
    - list[dict]: 当前样本生成的切片记录，不含 mask/boundary 字段。
    """
    with Image.open(find_image(sample_id)) as image_file:
        image = np.asarray(image_file.convert("RGB"))
    image_dir = OUTPUT_ROOT / split / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for top in tile_starts(image.shape[0], INFER_STRIDE):
        for left in tile_starts(image.shape[1], INFER_STRIDE):
            tile_name = f"sample{sample_id}_{top:05d}_{left:05d}.png"
            Image.fromarray(pad_tile(image, top, left, 0), mode="RGB").save(image_dir / tile_name)
            records.append({
                "image": f"{split}/image/{tile_name}",
                "source_sample": f"sample{sample_id}",
                "tile_origin": [left, top],
            })
    return records


def main() -> None:
    """
    入参:
    - 无；使用脚本顶部的源目录、输出目录和划分配置。

    方法:
    - 按划分清理并重建输出目录，标注样本产出图像/类别/边缘三联切片，
      未标注样本仅产出图像切片，最后写 JSONL 与 manifest。

    出参:
    - 无；数据集写入 OUTPUT_ROOT。
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, sample_ids in SPLITS.items():
        shutil.rmtree(OUTPUT_ROOT / split, ignore_errors=True)
        records = []
        for sample_id in sample_ids:
            if split == "infer":
                records.extend(prepare_infer_sample(sample_id, split))
            else:
                records.extend(prepare_labeled_sample(sample_id, split))
        with (OUTPUT_ROOT / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = len(records)
    manifest = {
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "task": "labelme_lithology_edge_constrained_segmentation",
        "tile_size": TILE_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "stride": STRIDE,
        "filter_empty_background": True,
        "edge_method": "per_class_morphological_gradient_union",
        "edge_channels": 4,
        "edge_kernel_size": EDGE_KERNEL_SIZE,
        "infer_stride": INFER_STRIDE,
        "splits": {split: {"samples": list(sample_ids), "tiles": counts[split]} for split, sample_ids in SPLITS.items()},
        "label_mapping": {name: class_id for name, class_id in LABELME_STRING_TO_CLASS_ID.items()},
    }
    with (OUTPUT_ROOT / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
