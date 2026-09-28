"""Oversample slope-deposit (poji) tiles with dihedral augmentation against class scarcity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from labelme_config import DATASET_CONFIG, LABELME_STRING_TO_CLASS_ID

POJI_ID = LABELME_STRING_TO_CLASS_ID["poji"]
# 8 个二面体变换（水平翻转 x 90 度整数倍旋转）去掉恒等，得到 7 个增强变体
VARIANTS = [(flip, turns) for flip in (True, False) for turns in range(4)][1:]
AUG_SUFFIX = "_aug{index}"
DATASET_ROOT = Path(DATASET_CONFIG["output_root"])


def load_image(path: Path, mode: str) -> np.ndarray:
    """
    入参:
    - path: 切片 PNG 路径。
    - mode: "L" 单通道类别/边缘或 "RGB" 图像。

    方法:
    - 以指定模式读取并转为连续内存数组，保证后续几何变换安全。

    出参:
    - np.ndarray: 图像数组（2D 或 HxWx3）。
    """
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert(mode), dtype=np.uint8))


def dihedral_transform(array: np.ndarray, flip: bool, turns: int) -> np.ndarray:
    """
    入参:
    - array: 2D 类别/边缘图、HxWx3 RGB 图像或 KxHxW 边缘通道图。
    - flip: 是否先水平翻转（沿宽度轴）。
    - turns: 90 度逆时针旋转次数，0 到 3。

    方法:
    - 按数组维度定位空间轴：2D 与 HxWx3 空间轴在前两维，KxHxW 空间轴在
      后两维；对空间轴同步执行翻转与旋转，通道维不参与变换。

    出参:
    - np.ndarray: 与输入同形状的几何变换结果。
    """
    if array.ndim == 3 and array.shape[0] <= 4 and array.shape[0] < array.shape[2]:
        transformed = np.flip(array, axis=2) if flip else array
        return np.ascontiguousarray(np.rot90(transformed, turns, axes=(1, 2)))
    transformed = np.flip(array, axis=1) if flip else array
    return np.ascontiguousarray(np.rot90(transformed, turns))


def remove_previous_augments(records: list[dict]) -> list[dict]:
    """
    入参:
    - records: train.jsonl 解析后的完整记录列表。

    方法:
    - 过滤掉 image 文件名含增强后缀的记录，并删除各切片目录中遗留的
      增强文件，保证脚本可重复执行且过采样倍数不叠加。

    出参:
    - list[dict]: 仅含原始切片的记录列表。
    """
    for directory in (DATASET_ROOT / "train").iterdir():
        if directory.is_dir():
            for stale in directory.glob("*_aug*.png"):
                stale.unlink()
    return [record for record in records if "_aug" not in Path(record["image"]).stem]


def augment_poji_tiles(records: list[dict]) -> tuple[list[dict], dict]:
    """
    入参:
    - records: 仅含原始切片的训练记录列表。

    方法:
    - 读取每条记录的类别切片并统计坡积像素，对含坡积的切片将图像、
      类别、union 边缘与四通道类别边缘同步做 7 个二面体变换，写为新
      文件并复制记录，使坡积切片采样频率提升 8 倍。

    出参:
    - tuple: (原始记录 + 增强记录列表, 坡积统计 dict)。
    """
    augmented = list(records)
    statistics = {"poji_tiles": 0, "poji_pixels": 0, "generated_tiles": 0}
    for record in records:
        label_path = DATASET_ROOT / record["mask"]
        label = load_image(label_path, "L")
        poji_pixels = int((label == POJI_ID).sum())
        if poji_pixels == 0:
            continue
        statistics["poji_tiles"] += 1
        statistics["poji_pixels"] += poji_pixels
        image = load_image(DATASET_ROOT / record["image"], "RGB")
        boundary = load_image(DATASET_ROOT / record["boundary"], "L")
        boundaries = np.stack([load_image(DATASET_ROOT / path, "L") for path in record["boundaries"]], axis=0)
        stem = Path(record["image"]).stem
        for index, (flip, turns) in enumerate(VARIANTS, start=1):
            suffix = AUG_SUFFIX.format(index=index)
            names = {
                "image": f"{stem}{suffix}.png",
                "mask": f"{stem}{suffix}.png",
                "boundary": f"{stem}{suffix}.png",
                "boundaries": [f"{stem}{suffix}.png" for _ in record["boundaries"]],
            }
            Image.fromarray(dihedral_transform(image, flip, turns), mode="RGB").save(DATASET_ROOT / "train" / "image" / names["image"])
            Image.fromarray(dihedral_transform(label, flip, turns), mode="L").save(DATASET_ROOT / "train" / "label" / names["mask"])
            Image.fromarray(dihedral_transform(boundary, flip, turns), mode="L").save(DATASET_ROOT / "train" / "boundary" / names["boundary"])
            for class_id, class_boundary_path in enumerate(record["boundaries"]):
                channel = dihedral_transform(boundaries[class_id], flip, turns)
                Image.fromarray(channel, mode="L").save(DATASET_ROOT / "train" / f"boundary_class{class_id + 1}" / names["boundaries"][class_id])
            augmented.append({
                "image": f"train/image/{names['image']}",
                "mask": f"train/label/{names['mask']}",
                "mask_merged": f"train/label/{names['mask']}",
                "boundary": f"train/boundary/{names['boundary']}",
                "boundaries": [f"train/boundary_class{class_id + 1}/{names['boundaries'][class_id]}" for class_id in range(4)],
                "prompt": record["prompt"],
                "source_sample": record["source_sample"],
                "tile_origin": record["tile_origin"],
            })
            statistics["generated_tiles"] += 1
    return augmented, statistics


def main() -> None:
    """
    入参:
    - 无；数据集根目录与坡积类别映射来自 config.yaml。

    方法:
    - 读取 train.jsonl，幂等清理旧增强产物，对含坡积切片生成 7 个二面体
      增强副本并追加记录，最后整体回写 train.jsonl 并打印过采样统计。

    出参:
    - None: 增强切片与更新后的 train.jsonl 写入数据集目录。
    """
    jsonl_path = DATASET_ROOT / "train.jsonl"
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    original_count = len(records)
    records = remove_previous_augments(records)
    augmented, statistics = augment_poji_tiles(records)
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in augmented:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"original_tiles={len(records)}, poji_tiles={statistics['poji_tiles']}, "
        f"poji_pixels={statistics['poji_pixels']}, generated_aug_tiles={statistics['generated_tiles']}, "
        f"total_tiles={len(augmented)} (removed {original_count - len(records)} stale aug records)"
    )


if __name__ == "__main__":
    main()
