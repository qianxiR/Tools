"""Dataset and losses for the minimal SAM3-Adapter lithology segmentation network."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from labelme_config import MODEL_CONFIG, load_class_map, validate_class_map

IMG_SIZE = 336
AUX_CHANNELS = int(MODEL_CONFIG["edge_channels"])


class GeologyTileDataset(Dataset):
    """
    入参:
    - records: JSONL 解析后的记录列表，含 image/mask 相对路径。
    - data_root: 数据根目录，用于拼接相对路径。
    - augment: 训练模式是否执行同步几何增强。

    方法:
    - 读取 256 tile 图像与 0-4 类别标签，中心缩放到 IMG_SIZE；
      标签使用最近邻插值，图像使用 SAM3 归一化 (mean=std=0.5)。

    出参:
    - __getitem__ 返回 (image [3,S,S] float, class_map [S,S] long,
      edge_map [4,S,S] float, image_path str)。
    """

    def __init__(self, records: list[dict], data_root: str, augment: bool = False):
        self.augment = augment
        self.samples = []
        for record in records:
            image_path = Path(data_root) / record["image"]
            mask_path = Path(data_root) / record.get("mask", record.get("mask_merged"))
            boundary_paths = record.get("boundaries")
            if boundary_paths:
                boundary_paths = [
                    str(Path(data_root) / path) if not Path(path).is_absolute() else str(Path(path))
                    for path in boundary_paths
                ]
            else:
                boundary_path = record.get("boundary")
                boundary_paths = [
                    str(Path(data_root) / boundary_path) if boundary_path and not Path(boundary_path).is_absolute() else str(Path(boundary_path))
                ] if boundary_path else []
            self.samples.append((str(image_path), str(mask_path), boundary_paths))

    def __len__(self):
        return len(self.samples)

    def _augment_pair(self, image: np.ndarray, class_map: np.ndarray, edge_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        入参:
        - image: HxWx3 uint8 RGB 图像。
        - class_map: HxW uint8 类别索引图。
        - edge_map: KxHxW uint8 类别边缘图。

        方法:
        - 对训练样本同步执行水平/垂直翻转和 90 度整数倍旋转，确保图像、类别
          标签和四通道边缘标签保持同一几何变换。

        出参:
        - tuple[np.ndarray, np.ndarray, np.ndarray]: 增强后的图像、类别图和边缘图。
        """
        if np.random.random() < 0.5:
            image, class_map, edge_map = cv2.flip(image, 1), cv2.flip(class_map, 1), cv2.flip(edge_map, 1)
        if np.random.random() < 0.5:
            image, class_map, edge_map = cv2.flip(image, 0), cv2.flip(class_map, 0), cv2.flip(edge_map, 0)
        turns = np.random.randint(0, 4)
        if turns:
            image, class_map = np.rot90(image, turns), np.rot90(class_map, turns)
            edge_map = np.rot90(edge_map, turns, axes=(1, 2))
        return np.ascontiguousarray(image), validate_class_map(np.ascontiguousarray(class_map)), np.ascontiguousarray(edge_map)

    def __getitem__(self, index: int):
        """
        入参:
        - index: 样本索引。

        方法:
        - 加载配对图像/类别标签，缩放到 SAM3 输入尺寸；训练模式时执行标签同步增强，
          最后执行 SAM3 归一化；存在 boundary 标注时同步返回二值边缘目标。

        出参:
        - tuple: 图像张量、long 类别图、边缘目标（或全零占位）、原始图像路径。
        """
        image_path, mask_path, boundary_paths = self.samples[index]
        raw = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        class_map = load_class_map(mask_path)
        boundary = np.zeros((AUX_CHANNELS, class_map.shape[0], class_map.shape[1]), dtype=np.uint8)
        for class_id, boundary_path in enumerate(boundary_paths[:AUX_CHANNELS]):
            with Image.open(boundary_path) as boundary_file:
                boundary[class_id] = (np.asarray(boundary_file.convert("L"), dtype=np.uint8) > 127).astype(np.uint8)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        class_map = cv2.resize(class_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        boundary = np.stack([
            cv2.resize(channel, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            for channel in boundary
        ], axis=0)
        if self.augment:
            image, class_map, boundary = self._augment_pair(image, class_map, boundary)
        else:
            class_map = validate_class_map(class_map)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0).sub_(0.5).div_(0.5)
        boundary_tensor = torch.from_numpy(boundary.astype(np.float32))
        return tensor, torch.from_numpy(class_map.astype(np.int64)), boundary_tensor, image_path


def collate_keep_list(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    boundaries = torch.stack([item[2] for item in batch])
    paths = [item[3] for item in batch]
    return images, labels, boundaries, paths


def foreground_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    入参:
    - logits: [B,C,H,W] 类别 logits。
    - target: [B,H,W] long 类别索引。

    方法:
    - 对四个前景类别计算 soft Dice 并取均值，背景不计入，
      约束前景召回，缓解背景支配。

    出参:
    - torch.Tensor: 前景平均 Dice 损失。
    """
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(probs.dtype)
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice[1:].mean()


def lovasz_gradient(sorted_foreground: torch.Tensor) -> torch.Tensor:
    """
    入参:
    - sorted_foreground: 按预测误差降序排列的二值真值向量。

    方法:
    - 根据 Jaccard 集合函数的 Lovasz 扩展计算每个排序位置的梯度权重。

    出参:
    - torch.Tensor: 与输入同长度的 Lovasz 梯度权重。
    """
    foreground_count = sorted_foreground.sum()
    intersection = foreground_count - sorted_foreground.float().cumsum(0)
    union = foreground_count + (1.0 - sorted_foreground.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if jaccard.numel() > 1:
        jaccard[1:] -= jaccard[:-1].clone()
    return jaccard


def foreground_lovasz_softmax_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    入参:
    - logits: [B,C,H,W] 五分类 logits。
    - target: [B,H,W] long 类别索引，类别 0 为背景。

    方法:
    - 对当前 batch 中存在的每个前景类别按像素误差排序并计算 Lovasz-Softmax，
      直接优化前景 IoU 的代理目标；背景不参与该项损失。

    出参:
    - torch.Tensor: 当前 batch 的前景 Lovasz-Softmax 损失标量。
    """
    probabilities = torch.softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    labels = target.reshape(-1)
    losses = []
    for class_id in range(1, logits.shape[1]):
        foreground = (labels == class_id).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probabilities[:, class_id]).abs()
        errors_sorted, order = torch.sort(errors, descending=True)
        foreground_sorted = foreground[order]
        losses.append(torch.dot(errors_sorted, lovasz_gradient(foreground_sorted)))
    if not losses:
        return probabilities.sum() * 0.0
    return torch.stack(losses).mean()


def edge_bce_dice_loss(edge_logits: torch.Tensor, edge_target: torch.Tensor, pos_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """
    入参:
    - edge_logits: [B,K,H,W] 多标签类别边缘 logits。
    - edge_target: [B,K,H,W] float 二值类别边缘真值。
    - pos_weight: [K] 每个边缘类别的 BCE 正类权重；为空时使用 1。

    方法:
    - 对每个类别边缘独立执行 BCEWithLogits 与 soft Dice，不在边缘通道间
      做 softmax；边界处允许多个类别通道同时为正。

    出参:
    - tuple: (标量损失, {bce、dice、pos_fraction} 诊断 dict)。
    """
    if edge_logits.ndim != 4 or edge_target.shape != edge_logits.shape:
        raise ValueError(f"Expected edge logits/target [B,K,H,W], got {edge_logits.shape} and {edge_target.shape}")
    weights = pos_weight.to(edge_logits.device).reshape(1, -1, 1, 1) if pos_weight is not None else None
    bce = F.binary_cross_entropy_with_logits(edge_logits, edge_target, pos_weight=weights)
    probs = torch.sigmoid(edge_logits)
    intersection = (probs * edge_target).sum(dim=(0, 2, 3))
    denominator = probs.sum(dim=(0, 2, 3)) + edge_target.sum(dim=(0, 2, 3))
    per_class = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    active = edge_target.sum(dim=(0, 2, 3)) > 0
    dice_loss = (1.0 - per_class[active]).mean() if active.any() else probs.sum() * 0.0
    loss = 0.5 * bce + 0.5 * dice_loss
    return loss, {"bce": float(bce.item()), "dice": float(dice_loss.item()), "pos_fraction": edge_target.mean(dim=(0, 2, 3)).detach().cpu().tolist()}
