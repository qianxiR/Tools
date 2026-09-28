#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变化检测分类记录生成工具
"""


def generate_dataset_text(
    image_name: str,
    t1_class: str,
    t2_class: str,
    confidence_t1: float | None = None,
    confidence_t2: float | None = None,
    change_bbox: list | None = None,
    patch_size: list | None = None,
    area_pixels: int | None = None
) -> dict:
    """
    生成分类记录

    入参:
    - image_name: 图像文件名
    - t1_class: T1 时刻类别
    - t2_class: T2 时刻类别
    - confidence_t1: T1 置信度
    - confidence_t2: T2 置信度
    - change_bbox: 边界框 [x, y, w, h]
    - patch_size: patch 尺寸 [h, w]
    - area_pixels: 像素面积

    出参:
    - dict: 分类记录
    """
    record: dict[str, str | float | list | int] = {
        "image_t1": image_name,
        "image_t2": image_name,
        "mask": image_name,
        "t1_class": t1_class,
        "t2_class": t2_class,
    }

    if confidence_t1 is not None:
        record["confidence_t1"] = round(confidence_t1, 4)
    if confidence_t2 is not None:
        record["confidence_t2"] = round(confidence_t2, 4)
    if change_bbox is not None:
        record["change_bbox"] = change_bbox
    if patch_size is not None:
        record["patch_size"] = patch_size
    if area_pixels is not None:
        record["area_pixels"] = area_pixels

    return record
