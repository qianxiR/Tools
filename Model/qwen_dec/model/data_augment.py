"""
训练数据增强（水平翻转）。

做什么:
    对训练样本以概率 p 做水平翻转，同时同步变换 assistant 段的 bbox 坐标。
    翻转后的 PIL Image 直接替换 record["image"]（不再走路径），保证 Qwen ViT
    和 SigLIP2 外部视觉链路看到同一张翻转图。

    为什么:
        训练集仅 342 张图，多目标样本稀少。水平翻转等效翻倍数据量，且对管道 CCTV
        图像安全（左右镜像不破坏缺陷语义）。预期能改善多目标场景的漏检（Recall）。

    用法:
        from model.data_augment import maybe_augment_record
        record = maybe_augment_record(record, p=0.5)  # 训练时调用；val/test 不调用
"""
from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List

from PIL import Image

from model.location_tokens import decode_bbox


def _parse_assistant(value: str) -> List[Dict[str, Any]]:
    """入参: assistant 段文本（<obj>[...]</obj> 包裹或裸 JSON 数组）。
    方法: 剥离 <obj></obj> 包裹后 json.loads 为目标列表。
    出参: 目标 dict 列表（每个含 bbox_2d 和 label）；空数组时返回 []。"""
    body = value.replace("<obj>", "").replace("</obj>", "").strip()
    if not body:
        return []
    return json.loads(body)


def _format_assistant(targets: List[Dict[str, Any]]) -> str:
    """入参: 目标 dict 列表。
    方法: 紧凑 JSON 序列化（与转换器一致：separators=(",",":")、ensure_ascii=False），<obj> 包裹。
    出参: '<obj>[{"bbox_2d":[...],"label":"..."},...]</obj>' 字符串。"""
    body = json.dumps(targets, ensure_ascii=False, separators=(",", ":"))
    return f"<obj>{body}</obj>"


def flip_bbox(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """入参: 目标列表，bbox 可为旧 Falcon location token 或裸整数坐标（均由 decode_bbox 兼容解析）。
    方法: 先反量化/解析到 0-1000 裸整数，执行水平翻转 [x1,y1,x2,y2]→[1000-x2,y1,1000-x1,y2]，
          输出统一为裸整数（与当前裸整数训练链一致，不再量化回 location token）。
    出参: bbox 为四个 0-1000 裸整数的新目标列表；非法 bbox 抛出 ValueError。"""
    out: List[Dict[str, Any]] = []
    for t in targets:
        box = decode_bbox(t.get("bbox_2d"), allow_legacy_numeric=True)
        if box is None:
            raise ValueError(f"invalid bbox_2d for horizontal flip: {t.get('bbox_2d')!r}.")
        x1, y1, x2, y2 = box
        new_t = dict(t)
        new_t["bbox_2d"] = [1000 - x2, y1, 1000 - x1, y2]
        out.append(new_t)
    return out


def maybe_augment_record(record: Dict[str, Any], p: float = 0.5) -> Dict[str, Any]:
    """入参: 原始 record（{"id","image","conversations":[{user},{assistant}]}），p 翻转概率。
    方法:
      - 以概率 p 决定是否翻转；不翻转时返回原 record（不拷贝，避免无谓开销）。
      - 翻转时：
        1) 从 record["image"] 读图（支持路径或 PIL），转 RGB 后 FLIP_LEFT_RIGHT；
        2) 解析 assistant 段 bbox，应用 flip_bbox 变换；
        3) 深拷贝 record，替换两个字段：
           - conversations[1]["value"] = 翻转后的 <obj>[...]</obj> 文本
           - image = 翻转后的 PIL.Image 对象（不再是路径）
        这样下游 process_vision_info（Qwen ViT）和 encode_external_images（SigLIP2）
        两条链路都拿到同一张翻转图，避免训练分布不一致。
    出参: 增强（或原样）record。"""
    if random.random() >= p:
        return record

    # 1) 读图 + 翻转
    img_field = record["image"]
    if isinstance(img_field, Image.Image):
        pil = img_field.convert("RGB")
    else:
        pil = Image.open(img_field).convert("RGB")
    flipped = pil.transpose(Image.FLIP_LEFT_RIGHT)

    # 2) 翻转 bbox
    targets = _parse_assistant(record["conversations"][1]["value"])
    flipped_targets = flip_bbox(targets)
    new_assistant = _format_assistant(flipped_targets)

    # 3) 深拷贝 record 并替换两个字段
    new_record = copy.deepcopy(record)
    new_record["image"] = flipped  # PIL 对象，下游两链路都接受
    new_record["conversations"][1]["value"] = new_assistant
    # 标记增强用于调试（不影响训练逻辑）
    new_record["_augmented"] = "hflip"
    return new_record
