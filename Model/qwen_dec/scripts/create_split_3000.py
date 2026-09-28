"""
为数据_筛选3000 创建 train/val/test 切分。

做什么:
    按 80/7/13 比例随机切分 3000 张图为 train/val/test，
    排除数据_筛选500 的 test 集（保证历史评估可比，避免数据泄露）。

为什么:
    3000 数据集缺少 split 目录，训练需要标准的 train/val/test 切分。
    500 的 test 集有 84/98 张在 3000 里，必须排除防止泄露。

输出:
    F:/管网/数据_筛选3000/split/train.txt
    F:/管网/数据_筛选3000/split/val.txt
    F:/管网/数据_筛选3000/split/test.txt
"""
from __future__ import annotations

import random
from pathlib import Path

DATA_ROOT = Path(r"F:/管网/数据_筛选3000")
IMG_DIR = DATA_ROOT / "images"
SPLIT_DIR = DATA_ROOT / "split"

# 500 的 test 集（排除项）
DS500_ROOT = Path(r"F:/管网/数据_筛选500")
DS500_TEST = DS500_ROOT / "split" / "test.txt"

# 500 的 val 集（也排除，保持一致性）
DS500_VAL = DS500_ROOT / "split" / "val.txt"

SEED = 42
VAL_RATIO = 0.1
TEST_RATIO = 0.2


def load_excluded_stems() -> set:
    """入参: 无; 方法: 读 500 的 test 和 val 清单，提取 stem 集合; 出参: 排除的 stem set。"""
    excluded = set()
    for split_file in [DS500_TEST, DS500_VAL]:
        if not split_file.exists():
            continue
        for line in split_file.read_text(encoding="utf-8").splitlines():
            rel = line.strip()
            if rel:
                stem = Path(rel).stem
                excluded.add(stem)
    return excluded


def main():
    """入参: 无; 方法: 列出所有图像，排除 500 的 test/val，按比例随机切分; 出参: 无。"""
    random.seed(SEED)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    excluded = load_excluded_stems()
    print(f"[split] 排除 500 的 test+val stem: {len(excluded)} 个")

    # 列出所有图像（排除 500 的 test/val）
    all_images = []
    skipped = 0
    for img_path in sorted(IMG_DIR.iterdir()):
        if not img_path.is_file():
            continue
        stem = img_path.stem
        if stem in excluded:
            skipped += 1
            continue
        ext = img_path.suffix
        all_images.append(f"images/{stem}{ext}")

    print(f"[split] 总图像: {len(all_images)} (排除 {skipped} 张 500 的 test/val)")

    # 洗牌后按比例切分
    random.shuffle(all_images)
    n = len(all_images)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)

    test_list = all_images[:n_test]
    val_list = all_images[n_test:n_test + n_val]
    train_list = all_images[n_test + n_val:]

    # 写 split 文件
    for name, data in [("train", train_list), ("val", val_list), ("test", test_list)]:
        out = SPLIT_DIR / f"{name}.txt"
        out.write_text("\n".join(data) + "\n", encoding="utf-8")
        print(f"[{name}] {len(data)} images -> {out}")

    print(f"\n[split] DONE: train={len(train_list)} val={len(val_list)} test={len(test_list)}")


if __name__ == "__main__":
    main()
