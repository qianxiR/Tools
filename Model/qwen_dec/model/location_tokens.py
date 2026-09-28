from __future__ import annotations

import json
import math
import re
from numbers import Real
from typing import Any, Dict, List, Sequence


LOCATION_BIN_COUNT = 1000
COORDINATE_SCALE = 1000
LOCATION_TOKEN_PATTERN = re.compile(r"^<loc_(\d{1,3})>$")


def location_tokens() -> List[str]:
    """入参: 无。
    方法: 按量化 bin 升序生成 Falcon 风格位置词表，确保训练和推理追加顺序一致。
    出参: `<loc_0>` 至 `<loc_999>` 的 1000 个 token。"""
    return [f"<loc_{index}>" for index in range(LOCATION_BIN_COUNT)]


def quantize_coordinate(value: int | float) -> int:
    """入参: 0-1000 归一化坐标，必须是有限实数。
    方法: 按 round(value * 999 / 1000) 映射到 1000 个离散位置 bin。
    出参: [0,999] 整数 bin；越界或非有限值抛出 ValueError。"""
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"coordinate must be a finite number, got {value!r}.")
    numeric_value = float(value)
    if not 0 <= numeric_value <= COORDINATE_SCALE:
        raise ValueError(f"coordinate must be in [0, {COORDINATE_SCALE}], got {value!r}.")
    return round(numeric_value * (LOCATION_BIN_COUNT - 1) / COORDINATE_SCALE)


def dequantize_bin(index: int) -> int:
    """入参: [0,999] 整数位置 bin。
    方法: 按 round(bin * 1000 / 999) 还原到工程统一的 0-1000 坐标域。
    出参: [0,1000] 归一化整数坐标；非法 bin 抛出 ValueError。"""
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < LOCATION_BIN_COUNT:
        raise ValueError(f"location bin must be an integer in [0, {LOCATION_BIN_COUNT - 1}], got {index!r}.")
    return round(index * COORDINATE_SCALE / (LOCATION_BIN_COUNT - 1))


def coordinate_to_token(value: int | float) -> str:
    """入参: 0-1000 归一化坐标。
    方法: 量化坐标并套用 Falcon `<loc_bin>` 表达。
    出参: 单个合法 location token。"""
    return f"<loc_{quantize_coordinate(value)}>"


def token_to_coordinate(token: str) -> int:
    """入参: `<loc_0>` 至 `<loc_999>` 字符串。
    方法: 严格解析 token 中的 bin，并反量化到 0-1000 坐标域。
    出参: 归一化整数坐标；格式或范围非法时抛出 ValueError。"""
    match = LOCATION_TOKEN_PATTERN.fullmatch(token) if isinstance(token, str) else None
    if match is None:
        raise ValueError(f"invalid location token: {token!r}.")
    return dequantize_bin(int(match.group(1)))


def encode_bbox(box: Sequence[int | float]) -> List[str]:
    """入参: 四元素 0-1000 xyxy 坐标序列。
    方法: 保持 xyxy 顺序，逐坐标量化为 Falcon location token。
    出参: 四个 location token；元素数不为四或坐标非法时抛出 ValueError。"""
    if isinstance(box, (str, bytes)) or len(box) != 4:
        raise ValueError(f"bbox must contain four coordinates, got {box!r}.")
    return [coordinate_to_token(value) for value in box]


def decode_bbox(box: Sequence[Any], allow_legacy_numeric: bool = True) -> List[int] | None:
    """入参: 四元素 bbox 和是否兼容旧数字坐标。
    方法: location token 反量化；兼容模式下也接受全数字 0-1000 bbox，混合或畸形格式视为无效。
    出参: 四元素归一化整数 xyxy；无法解析时返回 None。"""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if all(isinstance(value, str) for value in box):
        if not all(LOCATION_TOKEN_PATTERN.fullmatch(value) for value in box):
            return None
        return [token_to_coordinate(value) for value in box]
    numeric = all(isinstance(value, Real) and not isinstance(value, bool) for value in box)
    if not allow_legacy_numeric or not numeric:
        return None
    values = [float(value) for value in box]
    if not all(math.isfinite(value) and 0 <= value <= COORDINATE_SCALE for value in values):
        return None
    return [round(value) for value in values]


def canonicalize_targets(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """入参: 多目标字典列表，bbox 可为旧数字坐标或 location token。
    方法: 校验每个目标并把 bbox 统一编码为四个 location token，同时保留 label 等其他字段。
    出参: 新目标列表；目标结构或 bbox 非法时抛出 ValueError。"""
    if not isinstance(targets, list):
        raise ValueError("assistant target must be a JSON array.")
    canonical: List[Dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict) or "bbox_2d" not in target:
            raise ValueError(f"each target must be an object containing bbox_2d, got {target!r}.")
        decoded = decode_bbox(target["bbox_2d"], allow_legacy_numeric=True)
        if decoded is None:
            raise ValueError(f"invalid bbox_2d: {target['bbox_2d']!r}.")
        canonical_target = dict(target)
        canonical_target["bbox_2d"] = encode_bbox(decoded)
        canonical.append(canonical_target)
    return canonical


def canonicalize_assistant_text(text: str) -> str:
    """入参: `<obj>...</obj>` 包裹或裸 JSON 多目标数组。
    方法: 剥离现有包装、解析 JSON、统一 bbox 为 location token，再紧凑序列化并恢复 `<obj>` 包装。
    出参: 规范化的 `<obj>[...]</obj>` 文本；JSON 或目标结构非法时抛出 ValueError。"""
    body = str(text).replace("<obj>", "").replace("</obj>", "").strip()
    targets = json.loads(body)
    canonical = canonicalize_targets(targets)
    return f"<obj>{json.dumps(canonical, ensure_ascii=False, separators=(',', ':'))}</obj>"


def add_location_tokens(tokenizer: Any) -> List[int]:
    """入参: Hugging Face tokenizer。
    方法: 以普通 added token 按固定顺序注册 1000 个位置 token，并校验每个 token 独占一个唯一词表 ID。
    出参: 与 `<loc_0>` 至 `<loc_999>` 同序的 token ID 列表；注册结果不一致时抛出 RuntimeError。"""
    tokens = location_tokens()
    tokenizer.add_tokens(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(token_ids) != LOCATION_BIN_COUNT or len(set(token_ids)) != LOCATION_BIN_COUNT:
        raise RuntimeError("location tokens were not mapped to 1000 unique token IDs.")
    for token, token_id in zip(tokens, token_ids):
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if encoded != [token_id]:
            raise RuntimeError(f"location token {token} is not encoded as one token: {encoded}.")
    return token_ids
