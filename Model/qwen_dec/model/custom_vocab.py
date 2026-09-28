"""
自建词表：坐标原子 token + 类别 token。

做什么:
    为自训练 TransformerDecoder 构建确定性词表——把 0..1000 每个整数各注册为 1 个
    原子 token（如 547 是 1 个 token，而非 BPE 的 "5"/"47"），类别代码各 1 个 token。

为什么:
    替换 Qwen3-VL 基座后，需要一套与自训练 decoder 匹配的词表。确定性构造
    （不用频次过滤）保证所有坐标/类别都必需、无歧义；4 个特殊 token 与 Change3D
    保持一致的 ignore_index=0 约定（CrossEntropyLoss 自动忽略 <pad>）。

用法:
    from model.custom_vocab import build_word_map, save_word_map, load_word_map
    word_map = build_word_map(label_codes=["AJ","BX",...])
    encode_targets(word_map, [{"bbox_2d":[547,749,1000,1000],"label":"CJ"}])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# 坐标归一化范围（与转换器、推理反归一化严格互逆）
COORD_SCALE: int = 1000
# 序列容量上限：1 + 13*5 + 1 = 67（max=13 目标），留余量到 72
MAX_TARGETS_PER_IMAGE: int = 13
TOKENS_PER_TARGET: int = 5  # x1 y1 x2 y2 LABEL
# 固定特殊 token 索引（与 Change3D 约定一致，便于 CrossEntropyLoss(ignore_index=0)）
PAD_INDEX: int = 0
START_INDEX: int = 1
END_INDEX: int = 2
UNK_INDEX: int = 3

PAD_TOKEN: str = "<pad>"
START_TOKEN: str = "<start>"
END_TOKEN: str = "<end>"
UNK_TOKEN: str = "<unk>"


def build_word_map(label_codes: Sequence[str]) -> Dict[str, int]:
    """入参: label_codes 类别代码列表（顺序即 id 顺序）。
    方法: 确定性构造词表——4 个特殊 token + 0..1000 整数原子 token + 类别 token。
          不做频次过滤（每个整数和类别都必需，不存在"高频/低频"问题）。
    出参: word -> id 映射 dict。"""
    if len(set(label_codes)) != len(label_codes):
        raise ValueError(f"label_codes must be unique, got {list(label_codes)}")

    word_map: Dict[str, int] = {
        PAD_TOKEN: PAD_INDEX,
        START_TOKEN: START_INDEX,
        END_TOKEN: END_INDEX,
        UNK_TOKEN: UNK_INDEX,
    }
    # 0..1000 各自作为原子 token
    for value in range(COORD_SCALE + 1):
        word_map[str(value)] = len(word_map)
    # 类别代码
    for code in label_codes:
        if code in word_map:
            raise ValueError(f"label code {code!r} collides with an existing vocab token")
        word_map[code] = len(word_map)
    return word_map


def save_word_map(word_map: Dict[str, int], save_path: str | Path) -> None:
    """入参: word_map 与保存路径。
    方法: 以 UTF-8 缩进 JSON 落盘（ensure_ascii=False，便于直接查看）。
    出参: 无。"""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(word_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_word_map(load_path: str | Path) -> Dict[str, int]:
    """入参: 词表 JSON 路径。
    方法: 读入并做最小完整性校验（4 个特殊 token 索引正确）。
    出参: word -> id 映射 dict。"""
    data = json.loads(Path(load_path).read_text(encoding="utf-8"))
    for token, expected in (
        (PAD_TOKEN, PAD_INDEX),
        (START_TOKEN, START_INDEX),
        (END_TOKEN, END_INDEX),
        (UNK_TOKEN, UNK_INDEX),
    ):
        if data.get(token) != expected:
            raise ValueError(
                f"corrupt word_map: {token} expected id {expected}, got {data.get(token)}"
            )
    return data


def rev_word_map(word_map: Dict[str, int]) -> Dict[int, str]:
    """入参: word_map。方法: 翻转。出参: id -> word 映射。"""
    return {v: k for k, v in word_map.items()}


def _sort_targets_by_y1(targets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """入参: 目标列表。
    方法: 按 bbox_2d 的 y1（第二个坐标）升序排序，与现有 Qwen 链训练/推理约定一致。
    出参: 排序后的目标列表。"""
    return sorted(
        targets,
        key=lambda t: t.get("bbox_2d", [0, 0, 0, 0])[1] if len(t.get("bbox_2d", [])) >= 4 else 0,
    )


def encode_targets(
    word_map: Dict[str, int],
    targets: Sequence[Dict[str, Any]],
    max_len: int | None = None,
) -> Tuple[List[int], int]:
    """入参: word_map、目标列表、可选序列容量上限（None=MAX_LEN 默认）。
    方法: 把目标列表编码为 token id 序列：
          <start> [x1 y1 x2 y2 LABEL]* </end>，按 y1 升序；其余位置在 encode_caption 里补 <pad>。
          坐标取裸整数 0-1000；超出范围的坐标 clamp 后取 id（缺失则 <unk>）。
    出参: (token id 列表【不含 padding】，真实长度【含 start/end】)。
    注意: 调用方负责 padding。max_len 仅用于截断保护。"""
    if max_len is None:
        max_len = 1 + MAX_TARGETS_PER_IMAGE * TOKENS_PER_TARGET + 1

    ordered = _sort_targets_by_y1(targets)
    ids: List[int] = [word_map[START_TOKEN]]
    for t in ordered:
        box = t.get("bbox_2d")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        for value in box:
            try:
                iv = int(round(float(value)))
            except (TypeError, ValueError):
                ids.append(word_map[UNK_TOKEN])
                continue
            iv = max(0, min(COORD_SCALE, iv))
            ids.append(word_map.get(str(iv), word_map[UNK_TOKEN]))
        label = str(t.get("label", ""))
        ids.append(word_map.get(label, word_map[UNK_TOKEN]))

    ids.append(word_map[END_TOKEN])
    # 截断保护：保证末尾仍是 <end>
    if len(ids) > max_len:
        ids = ids[: max_len - 1] + [word_map[END_TOKEN]]
    return ids, len(ids)


def encode_caption(
    word_map: Dict[str, int],
    targets: Sequence[Dict[str, Any]],
    max_len: int | None = None,
) -> Tuple[List[int], int]:
    """入参: word_map、目标列表、定长序列容量。
    方法: encode_targets 后右侧补 <pad> 到 max_len（与 Change3D 原始链一致）。
          真实长度 caplen 含 start/end；pad 部分在 loss 里被 ignore_index=0 忽略。
    出参: (定长 token id 列表, caplen)。"""
    ids, caplen = encode_targets(word_map, targets, max_len=max_len)
    if max_len is None:
        max_len = 1 + MAX_TARGETS_PER_IMAGE * TOKENS_PER_TARGET + 1
    while len(ids) < max_len:
        ids.append(word_map[PAD_TOKEN])
    return ids, caplen


def parse_token_sequence(
    token_ids: Sequence[int],
    rev_map: Dict[int, str],
    word_map: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    """入参: 模型解码出的 token id 序列（含/不含特殊 token 均可）。
    方法:
      1) 过滤掉 <start>/<end>/<pad>/<unk>；
      2) 剩余 token 按 5 个一组切分 [x1,y1,x2,y2,LABEL]，末尾残组丢弃；
      3) 每组前 4 个必须是 0-1000 整数 token，第 5 个必须是类别 token（合法类别码）。
    出参: [{"bbox_2d":[x1,y1,x2,y2],"label":"XX"}, ...] 列表；解析失败返回空列表。
    注意: 仅保留合法类别码，避免把噪声 token 误判为目标。"""
    if word_map is not None:
        skip_ids = {
            word_map[PAD_TOKEN], word_map[START_TOKEN],
            word_map[END_TOKEN], word_map[UNK_TOKEN],
        }
    else:
        # 退化：用已知固定索引
        skip_ids = {PAD_INDEX, START_INDEX, END_INDEX, UNK_INDEX}

    valid: List[int] = [tid for tid in token_ids if tid not in skip_ids]

    # 推断合法类别码集合（id -> "XX" 且非整数串）
    label_strs: set[str] = set()
    for w in rev_map.values():
        if w and not w.lstrip("-").isdigit():
            label_strs.add(w)

    targets: List[Dict[str, Any]] = []
    for i in range(0, len(valid) - TOKENS_PER_TARGET + 1, TOKENS_PER_TARGET):
        group = valid[i : i + TOKENS_PER_TARGET]
        coords = []
        ok = True
        for tid in group[:4]:
            w = rev_map.get(int(tid))
            if w is None or not w.isdigit():
                ok = False
                break
            coords.append(int(w))
        if not ok:
            continue
        label_word = rev_map.get(int(group[4]))
        if label_word is None or label_word not in label_strs:
            continue
        x1, y1, x2, y2 = coords
        # 合法性：x2 > x1, y2 > y1, 且都在 0-1000（防御性）
        if x2 <= x1 or y2 <= y1:
            continue
        targets.append({"bbox_2d": [x1, y1, x2, y2], "label": label_word})
    return targets


def tokens_to_text(token_ids: Sequence[int], rev_map: Dict[int, str]) -> str:
    """入参: token id 序列。方法: 反查为可读文本。出参: 空格分隔的 token 字符串（调试用）。"""
    return " ".join(rev_map.get(int(tid), "<?>") for tid in token_ids)


def verify_roundtrip(label_codes: Sequence[str]) -> None:
    """入参: 类别码列表。
    方法: 端到端往返测试：构造若干样本目标 → encode → parse → 校验还原一致。
    出参: 无；不一致时抛出 AssertionError。"""
    wm = build_word_map(label_codes)
    rev = rev_word_map(wm)
    samples = [
        [{"bbox_2d": [547, 749, 1000, 1000], "label": "CJ"}],
        [
            {"bbox_2d": [120, 450, 380, 619], "label": "CR"},
            {"bbox_2d": [500, 100, 700, 300], "label": "SG"},  # y1=100 应排在前面
        ],
        [],  # 空图
    ]
    for idx, sample in enumerate(samples):
        ids, _ = encode_targets(wm, sample)
        parsed = parse_token_sequence(ids, rev, wm)
        expected = _sort_targets_by_y1(sample)
        assert len(parsed) == len(expected), (
            f"sample {idx}: count mismatch {len(parsed)} vs {len(expected)}"
        )
        for got, exp in zip(parsed, expected):
            assert got["label"] == exp["label"], f"label mismatch: {got} vs {exp}"
            for a, b in zip(got["bbox_2d"], exp["bbox_2d"]):
                assert int(a) == int(b), f"coord mismatch: {got} vs {exp}"
    print(f"[verify_roundtrip] OK: {len(samples)} samples, vocab_size={len(wm)}")
