"""
VOC 检测数据加载器（LandSlideDataSet）。

数据集结构:
    LandSlideDataSet/
        images/      *.tif  (3 波段 RGB, uint8)
        Annotations/ *.xml  (Pascal VOC, 绝对像素 xyxy, 单类别 slide)

约定:
    - 仅有 image、无对应 xml 的图像 → 视为负样本（空标注），DETR 匹配到 background。
    - 图像预处理与现有 read_img_mask 完全一致: 长边 resize 到 max_dim + center-pad 到正方形。
      这样训练时输入与 SAM2 image encoder 的 1024×1024 约定对齐。
    - bbox 同步做相同 scale + center-pad 变换, 再归一化为 cxcywh ∈ [0,1]
      (以 max_dim×max_dim 正方形为基准)。
    - 返回的元信息 (scale, pad) 供推理时把预测框还原到原图坐标。
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------
# 图像读取（与 utils.util_functions._pil_read_rgb 一致，避免引入对私有函数的依赖）
# ----------------------------------------------------------------------------
def _pil_read_rgb(path: str) -> np.ndarray:
    """入参: 图像路径。方法: PIL 打开并转 RGB。出参: [H,W,3] uint8 RGB。"""
    with Image.open(path) as im:
        im.info.pop("icc_profile", None)
        return np.array(im.convert("RGB"))


# ----------------------------------------------------------------------------
# VOC XML 解析
# ----------------------------------------------------------------------------
def parse_voc_xml(xml_path: str) -> dict:
    """解析 Pascal VOC XML，提取尺寸与目标框。

    入参:
    - xml_path: XML 文件路径

    方法:
    - 解析 size(宽/高/通道) 与每个 object 的 name + bndbox(xmin,ymin,xmax,ymax)

    出参:
    - {"width": int, "height": int, "objects": [(name, xmin, ymin, xmax, ymax), ...]}
      若文件不存在或解析失败，返回空 objects（视为负样本）。
    """
    if not os.path.isfile(xml_path):
        return {"width": 0, "height": 0, "objects": []}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text) if size is not None and size.find("width") is not None else 0
    height = int(size.find("height").text) if size is not None and size.find("height") is not None else 0

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip() if obj.find("name") is not None else ""
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        # 过滤掉退化框（ xmax<=xmin 或 ymax<=ymin ）
        if xmax > xmin and ymax > ymin:
            objects.append((name, xmin, ymin, xmax, ymax))
    return {"width": width, "height": height, "objects": objects}


# ----------------------------------------------------------------------------
# bbox 变换：原图 xyxy → 处理后正方形内 cxcywh (归一化)
# ----------------------------------------------------------------------------
def transform_boxes_xyxy(objects, scale: float, pad_left: int, pad_top: int, max_dim: int):
    """对原图绝对像素 xyxy 做与图像相同的 scale + center-pad 变换，
    并归一化为 cxcywh ∈ [0,1]（基准为 max_dim×max_dim 正方形）。

    入参:
    - objects:   [(name, xmin, ymin, xmax, ymax), ...] 原图坐标
    - scale:     长边缩放系数 (max_dim / max(h,w))
    - pad_left:  水平中心 padding 左侧像素数
    - pad_top:   垂直中心 padding 上侧像素数
    - max_dim:   目标正方形边长

    方法:
    - 先按 scale 缩放，再加 pad 偏移，得到处理后图内的 xyxy（像素坐标）
    - 裁剪到 [0, max_dim] 有效区，过滤掉完全越界或退化的框
    - 归一化: 除以 max_dim → cxcywh

    出参:
    - boxes:  np.ndarray [K, 4] 归一化 cxcywh
    - labels: np.ndarray [K] int64 (类别索引)
    - name_to_idx: 类别名→索引映射（首次出现即登记）
    """
    boxes = []
    labels = []
    for (name, xmin, ymin, xmax, ymax) in objects:
        x1 = xmin * scale + pad_left
        y1 = ymin * scale + pad_top
        x2 = xmax * scale + pad_left
        y2 = ymax * scale + pad_top
        # 裁剪到有效正方形区域内
        x1 = max(0.0, min(x1, max_dim))
        y1 = max(0.0, min(y1, max_dim))
        x2 = max(0.0, min(x2, max_dim))
        y2 = max(0.0, min(y2, max_dim))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue  # 变换后退化，丢弃
        cx = (x1 + x2) / 2.0 / max_dim
        cy = (y1 + y2) / 2.0 / max_dim
        w = (x2 - x1) / max_dim
        h = (y2 - y1) / max_dim
        boxes.append([cx, cy, w, h])
        labels.append(name)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=object)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class VoCDetectDataset(Dataset):
    """LandSlideDataSet 检测数据集。

    入参:
    - data_dir:     数据集根目录（含 images/ 与 Annotations/）
    - class_names:  类别列表（如 ["slide"]），决定 label 索引；None 则按出现顺序动态登记
    - max_dim:      处理后正方形边长（与 SAM2 输入约定一致，默认 1024）
    - split_names:  可选，仅加载指定文件名（无扩展名）列表；用于划分 train/val
    - return_neg:   是否包含无标注（负样本）图像，默认 True

    方法:
    - 扫描 images/*.tif，配对 Annotations/<name>.xml（无则为负样本）
    - __getitem__: 读图 → resize+pad → 读 xml → 同步变换 bbox → 归一化 cxcywh
    - 返回 dict: image[3,max_dim,max_dim] float32 / boxes[K,4] / labels[K] /
                 meta{原始尺寸, scale, pad, 图路径}

    出参:
    - sample dict（由 collate_voc_detect 批处理）
    """

    def __init__(
        self,
        data_dir: str,
        class_names: Optional[List[str]] = None,
        max_dim: int = 1024,
        split_names: Optional[List[str]] = None,
        return_neg: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.ann_dir = self.data_dir / "Annotations"
        self.max_dim = max_dim
        self.return_neg = return_neg

        # 类别名 → 索引（背景类 id=num_classes，由检测头负责；此处只管前景）
        if class_names is None:
            class_names = self._scan_class_names()
        self.class_names = list(class_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.class_names)}

        # 扫描图像并配对标注
        self.samples = []  # list of (img_path, xml_path_or_None)
        for img_path in sorted(self.img_dir.glob("*.tif")):
            stem = img_path.stem
            if split_names is not None and stem not in split_names:
                continue
            xml_path = self.ann_dir / f"{stem}.xml"
            has_ann = xml_path.is_file()
            if not has_ann and not return_neg:
                continue  # 跳过负样本
            self.samples.append((str(img_path), str(xml_path) if has_ann else None))

    def _scan_class_names(self) -> List[str]:
        """扫描所有 XML 登记类别名（按首次出现顺序），保证数据集自洽。"""
        names: List[str] = []
        seen = set()
        for xml_path in sorted(self.ann_dir.glob("*.xml")):
            ann = parse_voc_xml(str(xml_path))
            for (name, *_rest) in ann["objects"]:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names if names else ["object"]

    def __len__(self) -> int:
        return len(self.samples)

    def _read_and_pad(self, img_path: str):
        """读图并 resize+center-pad 到 max_dim×max_dim。

        入参: img_path
        方法: 长边 resize 到 max_dim（保比例）→ center-pad 到正方形
        出参: (img_pad[max_dim,max_dim,3] uint8 RGB, scale, pad_left, pad_top, orig_h, orig_w)
        """
        img = _pil_read_rgb(img_path)
        h, w = img.shape[:2]
        assert h > 0 and w > 0, f"Empty image: {img_path}"

        scale = float(self.max_dim) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

        pad_w = self.max_dim - new_w
        pad_h = self.max_dim - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                 borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = np.ascontiguousarray(img)
        return img, scale, pad_left, pad_top, h, w

    def __getitem__(self, idx: int) -> dict:
        img_path, xml_path = self.samples[idx]

        img, scale, pad_left, pad_top, orig_h, orig_w = self._read_and_pad(img_path)

        # 读标注（负样本 → 空标注）
        if xml_path is not None:
            ann = parse_voc_xml(xml_path)
            boxes_xywh, label_names = transform_boxes_xyxy(
                ann["objects"], scale, pad_left, pad_top, self.max_dim
            )
        else:
            boxes_xywh = np.zeros((0, 4), dtype=np.float32)
            label_names = np.zeros((0,), dtype=object)

        labels = np.array(
            [self.name_to_idx[n] for n in label_names], dtype=np.int64
        )

        # 图像归一化到 [0,1] float32, CHW
        img_f = img.astype(np.float32) / 255.0
        img_chw = np.ascontiguousarray(img_f.transpose(2, 0, 1))

        return {
            "image": torch.from_numpy(img_chw),                       # [3, max_dim, max_dim]
            "boxes": torch.from_numpy(boxes_xywh.astype(np.float32)), # [K, 4] cxcywh 归一化
            "labels": torch.from_numpy(labels),                        # [K] int64
            "meta": {
                "image_path": img_path,
                "orig_h": orig_h,
                "orig_w": orig_w,
                "scale": scale,
                "pad_left": pad_left,
                "pad_top": pad_top,
                "max_dim": self.max_dim,
            },
        }


# ----------------------------------------------------------------------------
# collate: 变长目标列表 → 批处理
# ----------------------------------------------------------------------------
def collate_voc_detect(batch: List[dict]):
    """DETR 风格批处理：图像堆叠，目标保持变长列表。

    入参: List[sample dict]
    方法: 图像张量 stack 为 [B,3,max_dim,max_dim]；boxes/labels 拆为 List[Tensor]，
          每个元素 [K_i,4] / [K_i]；meta 保留为 list。
    出参: (images[B,3,H,W], targets[List[dict{boxes,labels}]], metas[List[dict]])
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [
        {"boxes": b["boxes"], "labels": b["labels"]} for b in batch
    ]
    metas = [b["meta"] for b in batch]
    return images, targets, metas
