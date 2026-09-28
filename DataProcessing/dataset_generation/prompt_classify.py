#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遥感图像地物类别文本提示词定义

用于 RemoteCLIP 零样本分类的文本提示词

支持的类别：6类基础版本
- building: 建筑物
- highway: 交通运输
- vegetation: 植被
- farmland: 农田
- bare_land: 裸土
- water: 水域
"""

VOTED_PROMPTS = {
    "building": [
        "a remote sensing image of building",
        "a remote sensing image of house",
        "a remote sensing image of roof",
    ],
    "highway": [
        "a remote sensing image of highway",
        "a remote sensing image of road",
        "a remote sensing image of street",
    ],
    "vegetation": [
        "a remote sensing image of vegetation",
        "a remote sensing image of forest",
        "a remote sensing image of grass",
    ],
    "farmland": [
        "a remote sensing image of farmland",
        "a remote sensing image of cropland",
        "a remote sensing image of agricultural field",
    ],
    "bareland": [
        "a remote sensing image of bare land",
        "a remote sensing image of barren land",
        "a remote sensing image of soil",
    ],
    "water": [
        "a remote sensing image of water",
        "a remote sensing image of river",
    ],
}

CLASSES = ["building", "highway", "vegetation", "farmland", "bareland", "water"]


def get_voted_prompts() -> tuple:
    """
    获取多提示词投票配置：每个大类对应多个同义词提示词

    出参:
    - tuple: (all_prompts, prompt_to_class)
      - all_prompts (list): 展平的所有提示词
      - prompt_to_class (list): 每个提示词对应的大类标签
    """
    all_prompts = []
    prompt_to_class = []
    for key in CLASSES:
        prompts = VOTED_PROMPTS.get(key, [f"a remote sensing image of {key}"])
        all_prompts.extend(prompts)
        prompt_to_class.extend([key] * len(prompts))
    return all_prompts, prompt_to_class


if __name__ == "__main__":
    all_prompts, prompt_to_class = get_voted_prompts()
    class_labels = list(dict.fromkeys(prompt_to_class))
    print(f"Classes: {class_labels}")
    for cls in class_labels:
        prompts = [p for p, c in zip(all_prompts, prompt_to_class) if c == cls]
        print(f"  {cls}: {prompts}")
