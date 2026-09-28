"""
VOC → Qwen3-VL 对话格式转换（LandSlideDataSet 滑坡目标检测）。

做什么:
    遍历 images/*.tif，配对 Annotations/<stem>.xml，把 Pascal VOC 的绝对像素 xyxy
    归一化为 0-1000 相对坐标裸整数并组装成对话 JSONL。

为什么:
    Qwen3-VL 检测输出约定:
      - 坐标 = 0-1000 相对归一化裸整数（x 以原图 width 为基准，y 以原图 height 为基准），不是绝对像素。
      - 输出 = 以 <obj> 包裹的 JSON 数组，单个目标为 {"bbox_2d":[x1,y1,x2,y2],"label":"..."}。
      - 当前工程沿用基座原生数字 token，不注册 location token，也不使用 <box>/<|box_start|> 等包裹格式。

输出文件:
    H:/西藏遥感Agent/dataes/LandSlideDataSet/landslide_qwenvl.jsonl
    H:/西藏遥感Agent/dataes/LandSlideDataSet/landslide_qwenvl_train.jsonl
    H:/西藏遥感Agent/dataes/LandSlideDataSet/landslide_qwenvl_val.jsonl
    H:/西藏遥感Agent/dataes/LandSlideDataSet/landslide_qwenvl_test.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 复用现有 VOC 解析逻辑（不修改原文件，直接 import 其纯函数）
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data"))
from voc_detect import parse_voc_xml  # noqa: E402

import xml.etree.ElementTree as ET  # noqa: E402
from PIL import Image  # noqa: E402

# ----------------------------------------------------------------------------
# 路径常量
# ----------------------------------------------------------------------------
DATA_ROOT = Path(r"H:/西藏遥感Agent/dataes/LandSlideDataSet")
IMG_DIR = DATA_ROOT / "images"
ANN_DIR = DATA_ROOT / "Annotations"
HERE = PROJECT_ROOT

OUT_DIR = DATA_ROOT
OUT_ALL = OUT_DIR / "landslide_qwenvl.jsonl"
OUT_TRAIN = OUT_DIR / "landslide_qwenvl_train.jsonl"
OUT_VAL = OUT_DIR / "landslide_qwenvl_val.jsonl"
OUT_TEST = OUT_DIR / "landslide_qwenvl_test.jsonl"
PROMPT_JSON = HERE / "data" / "detection_prompt.json"

LABEL_NAME = "landslide"      # 单类别（滑坡）
COORD_SCALE = 1000            # Qwen3-VL 相对坐标上界
SPLIT_RATIO = 0.8             # 训练占比
VAL_TEST_RATIO = 0.5          # 剩余 holdout 中验证集占比；另一半作为测试集
SEED = 42

DEFAULT_USER_PROMPT = (
    "Detect landslide targets in mountainous scenes. Output a valid JSON array wrapped by <obj> and </obj>. "
    'Each target must be {"bbox_2d":[x1,y1,x2,y2],"label":"landslide"}, where every coordinate is an '
    "integer normalized to 0-1000. If no target exists, output <obj>[]</obj>."
)


def load_user_prompt() -> str:
    """入参: 无 | 方法: 优先读取 data/detection_prompt.json 中的 full_prompt，缺失时使用默认滑坡检测提示 | 出参: 转换阶段 user prompt"""
    if PROMPT_JSON.exists():
        try:
            data = json.loads(PROMPT_JSON.read_text(encoding="utf-8"))
            prompt = str(data.get("full_prompt", "")).strip()
            if prompt:
                return prompt
        except Exception:
            pass
    return DEFAULT_USER_PROMPT


USER_PROMPT = load_user_prompt()


def write_jsonl(path: Path, data: List[Dict]) -> None:
    """入参: path 输出 JSONL 路径，data 样本字典列表。
    方法: 按 UTF-8 逐行写出紧凑 JSON，保留中文路径与提示词原文。
    出参: 无；目标文件被覆盖为当前数据切分结果。"""
    with path.open("w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_samples(samples: List[Dict], rng: random.Random) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """入参: samples 全量样本，rng 固定种子随机数。
    方法: 先按 SPLIT_RATIO 切出训练集，再把剩余 holdout 按 VAL_TEST_RATIO 拆成 val/test；
          当 holdout 数量足够时保证 val/test 均非空。
    出参: (train_split, val_split, test_split)。"""
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * SPLIT_RATIO)
    holdout = shuffled[n_train:]
    n_val = int(len(holdout) * VAL_TEST_RATIO)
    if len(holdout) > 1:
        n_val = max(1, min(n_val, len(holdout) - 1))
    train_split = shuffled[:n_train]
    val_split = holdout[:n_val]
    test_split = holdout[n_val:]
    return train_split, val_split, test_split


# ----------------------------------------------------------------------------
# 图像尺寸读取
# ----------------------------------------------------------------------------
def read_image_size(img_path: Path) -> Optional[tuple]:
    """入参: img_path 图像路径。方法: PIL 打开读宽高（不加载像素，省内存）。
    出参: (width, height) 元组；读取失败返回 None。"""
    try:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 坐标归一化：绝对像素 xyxy → 0-1000 相对坐标
# ----------------------------------------------------------------------------
def normalize_bbox_to_1000(xmin: float, ymin: float, xmax: float, ymax: float,
                           width: int, height: int) -> Optional[List[int]]:
    """入参: VOC 绝对像素 xyxy + 原图宽高。
    方法: x 坐标以 width、y 坐标以 height 分别归一到 [0,1000]，与推理端
          x/1000*width、y/1000*height 的反归一化公式严格互逆。
    出参: [x1,y1,x2,y2] 整数列表；退化框（xmax<=xmin 或 ymax<=ymin）返回 None。"""
    if xmax <= xmin or ymax <= ymin:
        return None
    if width <= 0 or height <= 0:
        return None

    x1 = max(0, min(COORD_SCALE, round(xmin / width * COORD_SCALE)))
    y1 = max(0, min(COORD_SCALE, round(ymin / height * COORD_SCALE)))
    x2 = max(0, min(COORD_SCALE, round(xmax / width * COORD_SCALE)))
    y2 = max(0, min(COORD_SCALE, round(ymax / height * COORD_SCALE)))
    return [x1, y1, x2, y2]


# ----------------------------------------------------------------------------
# 单样本组装
# ----------------------------------------------------------------------------
def build_sample(stem: str, img_path: Path, xml_path: Optional[Path]) -> Optional[Dict]:
    """入参: stem 文件名主干，img_path 图像路径，xml_path 标注路径（负样本为 None）。
    方法:
      - 读图像尺寸（失败则跳过）。
      - 有 xml: 解析每个 object，归一化 bbox，过滤退化框，组装 JSON 数组。
      - 无 xml: assistant 文本为 "[]"（负样本，等价"未检测到滑坡"）。
      - user 文本以 "<image>\n" 占位图，后接固定提示词。
    出参: {"id","image","conversations":[{"from":"user"},{"from":"assistant"}]}；
          图像读取失败返回 None。"""
    size = read_image_size(img_path)
    if size is None:
        return None
    width, height = size

    if xml_path is not None and xml_path.is_file():
        ann = parse_voc_xml(str(xml_path))
        targets: List[Dict] = []
        for (name, xmin, ymin, xmax, ymax) in ann["objects"]:
            box = normalize_bbox_to_1000(xmin, ymin, xmax, ymax, width, height)
            if box is None:
                continue
            targets.append({"bbox_2d": box, "label": LABEL_NAME})
        body = json.dumps(targets, ensure_ascii=False, separators=(",", ":"))
    else:
        body = "[]"
    assistant_text = f"<obj>{body}</obj>"

    return {
        "id": stem,
        "image": str(img_path),
        "conversations": [
            {"from": "user", "value": f"<image>\n{USER_PROMPT}"},
            {"from": "assistant", "value": assistant_text},
        ],
    }


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    """入参: 无。
    方法: 扫描全部 *.tif 配对 xml，生成全量 JSONL 与 train/val/test 切分文件；
          固定种子保证切分可复现，test 从原 holdout 中拆出用于最终评估。
    出参: 无；写出 landslide_qwenvl*.jsonl 并打印切分统计。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    samples: List[Dict] = []
    pos_count, neg_count, total_boxes = 0, 0, 0
    for img_path in sorted(IMG_DIR.glob("*.tif")):
        stem = img_path.stem
        xml_path = ANN_DIR / f"{stem}.xml"
        has_ann = xml_path.is_file()
        sample = build_sample(stem, img_path, xml_path if has_ann else None)
        if sample is None:
            print(f"[WARN] skip unreadable image: {img_path}")
            continue
        samples.append(sample)
        if has_ann and sample["conversations"][1]["value"] != "<obj>[]</obj>":
            pos_count += 1
            body = sample["conversations"][1]["value"].removeprefix("<obj>").removesuffix("</obj>")
            total_boxes += len(json.loads(body))
        else:
            neg_count += 1

    if not samples:
        raise RuntimeError(f"no *.tif images found under {IMG_DIR}")

    train_split, val_split, test_split = split_samples(samples, rng)

    write_jsonl(OUT_ALL, samples)
    write_jsonl(OUT_TRAIN, train_split)
    write_jsonl(OUT_VAL, val_split)
    write_jsonl(OUT_TEST, test_split)

    # ---- 统计与抽样目检 ----
    print("=" * 60)
    print(f"total samples: {len(samples)}  (pos {pos_count} / neg {neg_count})")
    if pos_count:
        print(f"total boxes: {total_boxes}  avg/pos: {total_boxes / pos_count:.2f}")
    else:
        print("no positive samples")
    print(f"split: train={len(train_split)}  val={len(val_split)}  test={len(test_split)}")
    print("output:")
    print(f"  {OUT_ALL}")
    print(f"  {OUT_TRAIN}")
    print(f"  {OUT_VAL}")
    print(f"  {OUT_TEST}")
    print("-" * 60)
    print("first 2 samples:")
    for rec in samples[:2]:
        print(json.dumps(rec, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
