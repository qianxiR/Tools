# -*- coding: utf-8 -*-
"""
数据集工具
提供遥感影像数据集划分、渔网分割（同步多文件夹裁剪）等工具。

入参:
- file_list: 文件名列表
- source_folders: 源文件夹列表
- window_size / step_size: 裁剪窗口尺寸与步长

方法:
- split_dataset: 按比例随机划分数据集
- get_common_files: 求多文件夹交集文件名
- split_image_grid: 单图像渔网分割
- process_synchronized_folders: 多文件夹同步渔网分割
- create_folder_structure: 创建 train/val/test 标准目录

出参:
- 各函数返回值见签名
"""

import os
import shutil
import random
from pathlib import Path
import cv2


def split_dataset(file_list, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    入参:
    - file_list (list): 文件名列表
    - train_ratio (float): 训练集比例
    - val_ratio (float): 验证集比例
    - test_ratio (float): 测试集比例
    - seed (int): 随机种子，保证可重复性

    方法:
    - 固定 seed 随机打乱后按比例切分

    出参:
    - return (tuple): (train_files, val_files, test_files)
    """
    random.seed(seed)
    shuffled = file_list.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return shuffled[:t], shuffled[t:t + v], shuffled[t + v:]


def get_common_files(folders, prefix_filter=None):
    """
    入参:
    - folders (list[str]): 文件夹路径列表
    - prefix_filter (list[str]|None): 文件名白名单，如 ['1.png', '2.png']

    方法:
    - 求所有文件夹中 PNG 文件名的交集

    出参:
    - return (list[str]): 排序后的公共文件名列表
    """
    file_sets = []
    for folder in folders:
        if not os.path.exists(folder):
            return []
        pngs = set(f for f in os.listdir(folder) if f.lower().endswith('.png'))
        file_sets.append(pngs)
    common = set.intersection(*file_sets) if file_sets else set()
    if prefix_filter is not None:
        common = common & set(prefix_filter)
    return sorted(common)


def split_image_grid(image_path, output_folder, base_filename,
                     window_size=256, step_size=128):
    """
    入参:
    - image_path (str): 输入图像路径
    - output_folder (str): 输出文件夹
    - base_filename (str): 输出文件名前缀
    - window_size (int): 窗口边长
    - step_size (int): 步长（< window_size 时产生重叠）

    方法:
    - 滑动窗口逐块裁剪并保存为 {base}_{row:03d}_{col:03d}.png

    出参:
    - return (int): 生成的片段数
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0
    h, w = image.shape[:2]
    rows = (h - window_size) // step_size + 1
    cols = (w - window_size) // step_size + 1
    count = 0
    for r in range(rows):
        for c in range(cols):
            y, x = r * step_size, c * step_size
            if y + window_size <= h and x + window_size <= w:
                patch = image[y:y + window_size, x:x + window_size]
                path = os.path.join(output_folder, f"{base_filename}_{r:03d}_{c:03d}.png")
                cv2.imwrite(path, patch)
                count += 1
    return count


def process_synchronized_folders(source_folders, output_folders,
                                 window_size=256, step_size=256,
                                 file_filter=None):
    """
    入参:
    - source_folders (list[str]): 源文件夹列表（如 ['t1', 't2', 'label', 'edge']）
    - output_folders (list[str]): 对应输出文件夹列表
    - window_size (int): 窗口边长
    - step_size (int): 步长
    - file_filter (list[str]|None): 文件名白名单

    方法:
    - 求所有源文件夹交集文件名
    - 对每个文件同步裁剪，确保输出文件名一致

    出参:
    - return (dict): {source_folder: clip_count} 每个文件夹生成的片段数
    """
    for d in output_folders:
        os.makedirs(d, exist_ok=True)

    common = get_common_files(source_folders, file_filter)
    if not common:
        return {}

    stats = {s: 0 for s in source_folders}
    for fname in common:
        base = Path(fname).stem
        for src, dst in zip(source_folders, output_folders):
            n = split_image_grid(os.path.join(src, fname), dst, base,
                                window_size, step_size)
            stats[src] += n
    return stats


def create_folder_structure(base_dir, subsets=('train', 'val', 'test'),
                            categories=('t1', 't2', 'label', 'edge')):
    """
    入参:
    - base_dir (str): 数据集根目录
    - subsets (tuple): 划分名称
    - categories (tuple): 每个划分下的子类别

    方法:
    - 创建 {base_dir}/{subset}/{category}/ 目录结构

    出参:
    - return (list[str]): 创建的所有目录路径
    """
    created = []
    for s in subsets:
        for c in categories:
            p = os.path.join(base_dir, s, c)
            os.makedirs(p, exist_ok=True)
            created.append(p)
    return created


def copy_files_to_splits(file_splits, source_folders, target_dir,
                         folder_mapping=None):
    """
    入参:
    - file_splits (dict): {'train': [...], 'val': [...], 'test': [...]}
    - source_folders (list[str]): 源文件夹列表
    - target_dir (str): 目标根目录
    - folder_mapping (dict|None): 源文件夹名 → 目标子文件夹名的映射

    方法:
    - 将源文件复制到 target_dir/{split}/{mapped_subfolder}/

    出参:
    - return (int): 总复制文件数
    """
    if folder_mapping is None:
        folder_mapping = {f: f for f in source_folders}

    total = 0
    for split_name, files in file_splits.items():
        for fname in files:
            for src in source_folders:
                src_path = os.path.join(src, fname)
                dst_sub = folder_mapping.get(src, src)
                dst_path = os.path.join(target_dir, split_name, dst_sub, fname)
                if os.path.exists(src_path):
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    total += 1
    return total
