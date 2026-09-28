"""Shared prompt loading for dataset conversion, training, and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_full_prompt(prompt_path: Path) -> str:
    """入参: prompt_path 为 UTF-8 prompt JSON 路径，必须包含非空 full_prompt。
    方法: 读取 JSON 并提取唯一生效的 full_prompt，禁止静默回退到其他任务提示。
    出参: 去除首尾空白的 prompt 文本；文件不存在、JSON 无效或字段为空时抛出异常。"""
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    prompt = str(data.get("full_prompt", "")).strip()
    if not prompt:
        raise ValueError(f"empty full_prompt in prompt file: {prompt_path}")
    return prompt


def load_label_codes(prompt_path: Path) -> List[str]:
    """入参: prompt_path 为 UTF-8 prompt JSON 路径，必须包含非空且不重复的 label_codes。
    方法: 保持配置顺序读取类别代码并校验每项为非空字符串，作为训练与评估的唯一类别顺序。
    出参: 类别代码列表；字段缺失、存在空值或重复值时抛出 ValueError。"""
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    label_codes = [str(code).strip() for code in data.get("label_codes", [])]
    if not label_codes or any(not code for code in label_codes):
        raise ValueError(f"empty label_codes in prompt file: {prompt_path}")
    if len(label_codes) != len(set(label_codes)):
        raise ValueError(f"duplicate label_codes in prompt file: {prompt_path}")
    return label_codes
