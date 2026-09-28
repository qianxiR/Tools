# -*- coding: utf-8 -*-
"""
影像变换与增强工具
提供遥感影像的归一化、缩放、翻转、裁剪等变换类，兼容变化检测和分割任务。

入参:
- image: numpy 数组 (H, W, C)
- label: numpy 数组 (H, W) 或 (H, W, C)
- mean/std: 归一化参数

方法:
- 各 Transform 类的 __call__ 接口：统一 (image, label) → (image, label)
- ComposedTransforms: 将多个变换串联为管道
- BCDTransforms.get_transform_pipelines: 生成训练/验证变换管道

出参:
- 变换后的 (image, label) 元组
"""

import random
import numpy as np
import cv2
import torch
from typing import Callable, Sequence, Union, Optional


class ScaleTransform:
    """固定尺寸缩放"""
    def __init__(self, width, height):
        self.width, self.height = width, height

    def __call__(self, img, label):
        img = cv2.resize(img, (self.width, self.height))
        label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return img, label


class ResizeTransform:
    """保持宽高比的缩放"""
    def __init__(self, min_size, max_size=None, strict=False):
        self.min_size = (min_size,) if not isinstance(min_size, (list, tuple)) else min_size
        self.max_size = max_size
        self.strict = strict

    def get_size(self, h, w):
        if self.strict:
            return (self.max_size, self.min_size[0]) if w < h else (self.min_size[0], self.max_size)
        size = random.choice(self.min_size)
        if self.max_size and max(w, h) / min(w, h) * size > self.max_size:
            size = int(self.max_size * min(w, h) / max(w, h))
        ow = size if w < h else int(size * w / h)
        oh = int(size * h / w) if w < h else size
        return oh, ow

    def __call__(self, img, label):
        oh, ow = self.get_size(*img.shape[:2])
        img = cv2.resize(img, (ow, oh))
        label = cv2.resize(label, (ow, oh), interpolation=cv2.INTER_NEAREST)
        return img, label


class RandomCropResizeTransform:
    """随机裁剪后缩放回原始尺寸"""
    def __init__(self, crop_area):
        self.crop_area = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            y1, x1 = random.randint(0, self.crop_area), random.randint(0, self.crop_area)
            img = cv2.resize(img[y1:h - y1, x1:w - x1], (w, h))
            label = cv2.resize(label[y1:h - y1, x1:w - x1], (w, h), interpolation=cv2.INTER_NEAREST)
        return img, label


class RandomFlipTransform:
    """随机水平/垂直翻转（各 50% 概率）"""
    def __call__(self, img, label):
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
            label = cv2.flip(label, 0)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)
        return img, label


class NormalizeTransform:
    """归一化：/255 → (x - mean) / std"""
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, -1)

    def __call__(self, img, label):
        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return img, label


class RandomExchangeTransform:
    """随机交换前后时相图像（变化检测专用）"""
    def __call__(self, img, label):
        if random.random() < 0.5:
            img = np.concatenate((img[:, :, 3:6], img[:, :, 0:3]), axis=2)
        return img, label


class ToTensorTransform:
    """HWC uint8 → CHW float32 张量"""
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, img, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            img = cv2.resize(img, (w, h))
            label = cv2.resize(label, (int(w / self.scale), int(h / self.scale)),
                               interpolation=cv2.INTER_NEAREST)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)) if img.ndim == 3 else torch.from_numpy(img)
        label_t = torch.LongTensor(np.array(label, dtype=np.int8)).unsqueeze(0)
        return img_t, label_t


class ComposedTransforms:
    """串联多个变换的管道"""
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, img, label=None):
        for t in self.transforms_list:
            img, label = t(img, label)
        return img, label


class TransformBuilder:
    """
    变换管道构建器

    入参:
    - 无（通过类方法链式配置）

    方法:
    - normalize / scale / resize / random_crop / random_flip / random_exchange / to_tensor:
      各变换的工厂方法
    - compose: 串联已配置的变换列表
    - build_train_val: 一步生成训练+验证管道

    出参:
    - ComposedTransforms 实例
    """
    DEFAULT_MEAN = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    DEFAULT_STD = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    @staticmethod
    def normalize(mean=None, std=None):
        return NormalizeTransform(mean or TransformBuilder.DEFAULT_MEAN,
                                  std or TransformBuilder.DEFAULT_STD)

    @staticmethod
    def scale(width, height):
        return ScaleTransform(width, height)

    @staticmethod
    def resize(min_size, max_size=None, strict=False):
        return ResizeTransform(min_size, max_size, strict)

    @staticmethod
    def random_crop(crop_area):
        return RandomCropResizeTransform(crop_area)

    @staticmethod
    def random_flip():
        return RandomFlipTransform()

    @staticmethod
    def random_exchange():
        return RandomExchangeTransform()

    @staticmethod
    def to_tensor(scale=1):
        return ToTensorTransform(scale)

    @staticmethod
    def compose(transforms_list):
        return ComposedTransforms(transforms_list)

    @staticmethod
    def build_train_val(in_width, in_height, mean=None, std=None):
        """
        入参:
        - in_width (int): 输入宽度
        - in_height (int): 输入高度
        - mean (list): 归一化均值
        - std (list): 归一化标准差

        方法:
        - 生成标准训练管道（归一化+缩放+裁剪+翻转+交换+张量）
        - 和验证管道（归一化+缩放+张量）

        出参:
        - return (tuple): (train_transform, val_transform)
        """
        m = mean or TransformBuilder.DEFAULT_MEAN
        s = std or TransformBuilder.DEFAULT_STD
        crop_area = int(7.0 / 224.0 * in_width)

        train = ComposedTransforms([
            NormalizeTransform(m, s),
            ScaleTransform(in_width, in_height),
            RandomCropResizeTransform(crop_area),
            RandomFlipTransform(),
            RandomExchangeTransform(),
            ToTensorTransform(),
        ])
        val = ComposedTransforms([
            NormalizeTransform(m, s),
            ScaleTransform(in_width, in_height),
            ToTensorTransform(),
        ])
        return train, val
