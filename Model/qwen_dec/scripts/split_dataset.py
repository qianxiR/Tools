# -*- coding: utf-8 -*-
from __future__ import annotations

"""
数据集划分脚本 (7:1:2 = train:val:test, 随机种子 42)。

做什么: 将 <DATASET_ROOT>/images/ 下样本按固定种子随机打乱后按比例切分, 输出清单到 split/。
为什么: 提供可复现的 train/val/test 划分, 供 YOLO/Ultralytics 训练配置与 Qwen3-VL JSONL
      转换器 (convert_yolo_to_qwenvl.py) 直接引用。
迁移说明: 原版位于 F:/管网/数据_筛选500/split_dataset.py（以脚本所在目录为根）；
      迁入 scripts/ 后根目录参数化为 DATASET_ROOT 环境变量（默认 F:/管网/数据_筛选3000），
      划分逻辑与种子保持不变，清单内容与原版逐字节一致。
用法: python scripts/split_dataset.py   （或 $env:DATASET_ROOT="F:/管网/数据_筛选500"; python scripts/split_dataset.py）
"""
import os
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RATIOS = (0.7, 0.1, 0.2)   # train, val, test, 和为 1
SEED = 42
DATA_ROOT = Path(os.environ.get("DATASET_ROOT", r"F:/管网/数据_筛选3000"))


def load_samples(img_dir: Path) -> list[Path]:
    """
    入参:
        img_dir (Path): 图像目录.
    方法:
        枚举目录下所有受支持扩展名的图像, 按文件名排序.
        排序使打乱前的初始顺序确定, 从而种子相同时划分结果完全可复现.
    出参:
        samples (list[Path]): 按文件名排序后的图像路径列表.
    """
    return sorted(
        (p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS),
        key=lambda p: p.name,
    )


def split_samples(
    samples: list[Path],
    ratios: tuple[float, float, float] = RATIOS,
    seed: int = SEED,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    入参:
        samples (list[Path]): 全量样本 (已排序).
        ratios (tuple): (train, val, test) 比例, 和为 1.
        seed (int): 随机种子.
    方法:
        固定种子后对样本索引做 random.shuffle 打乱; 先按 test、val 比例四舍五入取整数个数,
        train 取剩余, 保证三集无重叠且合计等于总数.
    出参:
        train, val, test (list[Path]): 三个子集样本列表.
    """
    n = len(samples)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)

    n_test = round(n * ratios[2])
    n_val = round(n * ratios[1])
    n_train = n - n_test - n_val                # 取余避免浮点误差导致总数不闭合

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]
    return train, val, test


def write_split(out_path: Path, samples: list[Path]) -> None:
    """
    入参:
        out_path (Path): 清单文件输出路径 (如 split/train.txt).
        samples (list[Path]): 子集样本路径列表.
    方法:
        每行写一条相对路径 "images/文件名", 用正斜杠分隔以保证 Windows/Linux 跨平台兼容;
        清单统一使用 LF 换行 (与训练框架读取约定一致).
    出参: 无 (文件已写入磁盘).
    """
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for p in samples:
            f.write(f"images/{p.name}\n")


def main() -> None:
    """
    入参: 无 (路径由 DATASET_ROOT 环境变量或脚本默认值给出).
    方法:
        以 DATA_ROOT 为根, 收集 images/ 样本, 按 7:1:2 与种子 42 切分,
        写出 split/train.txt、val.txt、test.txt, 并打印各集数量与实际比例.
    出参: 无 (向标准输出打印统计).
    """
    img_dir = DATA_ROOT / "images"
    out_dir = DATA_ROOT / "split"
    out_dir.mkdir(exist_ok=True)

    samples = load_samples(img_dir)
    train, val, test = split_samples(samples)

    write_split(out_dir / "train.txt", train)
    write_split(out_dir / "val.txt", val)
    write_split(out_dir / "test.txt", test)

    n = len(samples)
    print(f"数据集根目录: {DATA_ROOT}")
    print(f"总计样本: {n}")
    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")
    print(
        f"比例: {len(train)/n:.3f} : {len(val)/n:.3f} : {len(test)/n:.3f}  "
        f"(随机种子={SEED})"
    )
    print(f"清单已输出至: {out_dir}")


if __name__ == "__main__":
    main()
