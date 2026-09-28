"""
YOLO → Qwen3-VL 对话格式转换（管网缺陷目标检测，数据_筛选3000）。

做什么:
    按 split/{train,val,test}.txt 给定的图像清单，配对 labels/<stem>.txt（标准 YOLO 标注），
    把 YOLO 的归一化 cxcywh 转成 0-1000 相对 xyxy 裸整数并组装对话 JSONL。

为什么:
    - 源数据是 YOLO 格式（类别ID  x_center  y_center  width  height，全 0-1 归一化），
      与既有 VOC 转换器不同，需独立脚本。
    - YOLO 坐标已按 x/原图宽、y/原图高 分别归一化，与 Qwen3-VL 的 0-1000 相对坐标定义同维，
      故 cxcywh 可直接乘 1000 得到 xyxy，无需读图像尺寸做像素换算（尺寸只用于校验可读性）。
    - Qwen3-VL 检测输出约定：assistant 段输出以 <obj> 包裹的 JSON 数组
      <obj>[{"bbox_2d":[x1,y1,x2,y2],"label":"<code>"}]</obj>；坐标为 0-1000 裸整数，
      label 用类别代码（AJ/BX/…），不使用 location token 或 <box>/<|box_start|> 等坐标包裹格式。

类别:
    labels/classes.txt 行序即 YOLO 类别 ID（0 基），共 12 类管网 CCTV 缺陷代码。
    label 直接使用类别代码（AJ/BX/CJ…），prompt 清单里的英文释义由
    data/pipe_defect_prompt.json 的 code_to_en 维护，本脚本不重复映射。
    12 类：AJ(支管暗接) BX(变形) CJ(沉积) CK(错口) CR(管壁裂纹) FS(腐蚀)
           JG(结垢) PL(破裂) SG(树根) TJ(脱节) TL(接口材料脱落) ZW(障碍物)。

输出文件（写入数据集根目录）:
    F:/管网/数据_筛选3000/pipe_qwenvl.jsonl            全量
    F:/管网/数据_筛选3000/pipe_qwenvl_train.jsonl      训练
    F:/管网/数据_筛选3000/pipe_qwenvl_val.jsonl        验证
    F:/管网/数据_筛选3000/pipe_qwenvl_test.jsonl       测试
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from model.prompt_loader import load_full_prompt  # noqa: E402

# ----------------------------------------------------------------------------
# 路径常量
# ----------------------------------------------------------------------------
HERE = PROJECT_ROOT

DATA_ROOT = Path(os.environ.get("DATASET_ROOT", r"F:/管网/数据_筛选3000"))
IMG_DIR = DATA_ROOT / "images"
LABEL_DIR = DATA_ROOT / "labels"
CLASSES_FILE = LABEL_DIR / "classes.txt"
SPLIT_DIR = DATA_ROOT / "split"

OUT_DIR = DATA_ROOT
OUT_ALL = OUT_DIR / "pipe_qwenvl.jsonl"
OUT_TRAIN = OUT_DIR / "pipe_qwenvl_train.jsonl"
OUT_VAL = OUT_DIR / "pipe_qwenvl_val.jsonl"
OUT_TEST = OUT_DIR / "pipe_qwenvl_test.jsonl"
PROMPT_JSON = os.environ.get("PROMPT_JSON", "")
TRAIN_PROMPT_JSON = Path(os.environ.get(
    "TRAIN_PROMPT_JSON",
    PROMPT_JSON or str(HERE / "data" / "pipe_defect_3000_prompt_train_en.json"),
))

COORD_SCALE = 1000            # Qwen3-VL 相对坐标上界
# label 直接使用 YOLO 类别代码（classes.txt 行序），prompt 清单的英文释义由
# data/pipe_defect_3000_prompt_rawint.json 的 code_to_en 维护，本脚本不重复映射。

USER_PROMPT = load_full_prompt(TRAIN_PROMPT_JSON)


def write_jsonl(path: Path, data: List[Dict]) -> None:
    """入参: path 输出 JSONL 路径，data 样本字典列表。
    方法: 按 UTF-8 逐行写出紧凑 JSON，保留中文路径与提示词原文。
    出参: 无；目标文件被覆盖为当前数据切分结果。"""
    with path.open("w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# 类别表：读 classes.txt，按行序映射为中文名
# ----------------------------------------------------------------------------
def load_class_names() -> List[str]:
    """入参: 无 | 方法: 逐行读 classes.txt，按行序（即 YOLO 类别 ID 0 基）原样取类别代码。
    出参: 类别代码列表（AJ/BX/CJ…），下标 = YOLO class_id，直接作为 label。"""
    names: List[str] = []
    for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines():
        code = line.strip()
        if code:
            names.append(code)
    if not names:
        raise RuntimeError(f"empty classes.txt: {CLASSES_FILE}")
    return names


# ----------------------------------------------------------------------------
# 图像尺寸读取（仅校验可读性，不参与坐标换算）
# ----------------------------------------------------------------------------
def read_image_size(img_path: Path) -> Optional[tuple]:
    """入参: img_path 图像路径。方法: PIL 打开读宽高（不加载像素）。
    出参: (width, height)；读取失败返回 None。"""
    try:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 坐标换算：YOLO 归一化 cxcywh → Qwen3-VL 0-1000 xyxy
# ----------------------------------------------------------------------------
def yolo_to_bbox_1000(xc: float, yc: float, w: float, h: float) -> Optional[List[int]]:
    """入参: YOLO 归一化中心点 (xc,yc) 与宽高 (w,h)，均 0-1（x 维以宽、y 维以高为基准）。
    方法: 直接乘 COORD_SCALE 得 0-1000 xyxy，夹到 [0,1000]，过滤退化框。
          因 YOLO 已按轴向归一化，与 Qwen3-VL 0-1000 相对坐标同维，无需原图尺寸。
    出参: [x1,y1,x2,y2] 整数列表；退化框（x2<=x1 或 y2<=y1）返回 None。"""
    x1 = max(0, min(COORD_SCALE, round((xc - w / 2) * COORD_SCALE)))
    y1 = max(0, min(COORD_SCALE, round((yc - h / 2) * COORD_SCALE)))
    x2 = max(0, min(COORD_SCALE, round((xc + w / 2) * COORD_SCALE)))
    y2 = max(0, min(COORD_SCALE, round((yc + h / 2) * COORD_SCALE)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_yolo_label(txt_path: Path, class_names: List[str]) -> List[Dict]:
    """入参: txt_path YOLO 标注文件，class_names 类别表（下标=class_id）。
    方法: 逐行解析 `cls xc yc w h`，换算为 0-1000 裸整数 bbox，跳过非标准行与退化框。
    出参: 目标字典列表（保持原顺序）。"""
    targets: List[Dict] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        box = yolo_to_bbox_1000(xc, yc, w, h)
        if box is None:
            continue
        label = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        targets.append({"bbox_2d": box, "label": label})
    return targets


# ----------------------------------------------------------------------------
# 单样本组装
# ----------------------------------------------------------------------------
def build_sample(img_path: Path, class_names: List[str], data_root: Path = None) -> Optional[Dict]:
    """入参: img_path 图像路径，class_names 类别表，data_root 数据根目录（默认原始 DATA_ROOT）。
    方法: 校验图像可读；配对 <data_root>/labels/<stem>.txt，解析目标得 JSON 数组并以 <obj> 包裹。
    出参: {"id","image","conversations":[{user},{assistant}]}；图像不可读返回 None。"""
    root = data_root or DATA_ROOT
    label_dir = root / "labels"
    if read_image_size(img_path) is None:
        return None
    stem = img_path.stem
    txt_path = label_dir / f"{stem}.txt"
    if txt_path.is_file():
        targets = parse_yolo_label(txt_path, class_names)
        body = json.dumps(targets, ensure_ascii=False, separators=(",", ":"))
    else:
        body = "[]"
    # <obj> 包裹：与推理/评估解析端约定一致，assistant 段整体为 <obj>[...]</obj>。
    assistant_text = f"<obj>{body}</obj>"

    return {
        "id": stem,
        "image": str(img_path),
        "conversations": [
            {"from": "user", "value": f"<image>\n{USER_PROMPT}"},
            {"from": "assistant", "value": assistant_text},
        ],
    }


def assistant_targets(value: str) -> List[Dict]:
    """入参: assistant 段文本（<obj>[...]</obj> 包裹或裸 JSON 数组）。
    方法: 剥离 <obj></obj> 包裹后 json.loads 为目标列表，与 build_sample 的包裹互逆。
    出参: 目标 dict 列表；value 为 "<obj>[]</obj>" 时返回空列表。"""
    body = value.replace("<obj>", "").replace("</obj>", "")
    return json.loads(body)


# ----------------------------------------------------------------------------
# 切分清单读取：split/<name>.txt 每行 images/<name>.<ext>
# ----------------------------------------------------------------------------
def load_split(split_name: str, data_root: Path = None) -> List[Path]:
    """入参: split_name，data_root（默认原始 DATA_ROOT）。
    方法: 读 <data_root>/split/<name>.txt，每行相对 data_root 的图像路径。
    出参: 存在的图像绝对路径列表（跳过缺失项并告警）。"""
    root = data_root or DATA_ROOT
    split_dir = root / "split"
    f = split_dir / f"{split_name}.txt"
    paths: List[Path] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        p = root / rel  # 形如 images/xxx.png
        if p.is_file():
            paths.append(p)
        else:
            print(f"[WARN] split {split_name} missing image: {rel}")
    return paths


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def convert_split(name: str, class_names: List[str], data_root: Path = None) -> tuple[List[Dict], Dict[str, int]]:
    """入参: name 切分名，class_names 类别表，data_root（默认原始 DATA_ROOT）。
    方法: 读 split 清单逐图组装样本，统计各类别框数与图像数。
    出参: (samples, class_box_counts)。"""
    root = data_root or DATA_ROOT
    img_paths = load_split(name, root)
    samples: List[Dict] = []
    box_counts: Dict[str, int] = {c: 0 for c in class_names}
    for img_path in sorted(img_paths):
        sample = build_sample(img_path, class_names, root)
        if sample is None:
            print(f"[WARN] skip unreadable image: {img_path}")
            continue
        samples.append(sample)
        for tgt in assistant_targets(sample["conversations"][1]["value"]):
            lbl = tgt.get("label")
            if lbl in box_counts:
                box_counts[lbl] += 1
    return samples, box_counts


def main() -> None:
    """入参: 无。
    方法: 读 classes.txt 与三份 split 清单，分别生成 train/val/test JSONL 及全量 JSONL，
          打印各类别框数、最大目标数/图、最长 assistant 文本（用于核对 MAX_LENGTH）。
    出参: 无；写出 pipe_qwenvl*.jsonl 并打印统计。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names()
    print(f"classes ({len(class_names)}): {class_names}")

    all_samples: List[Dict] = []
    total_box_counts: Dict[str, int] = {c: 0 for c in class_names}

    # 检测 Mosaic 增强数据是否存在，存在则合并到 train
    AUG_ROOT = DATA_ROOT.parent / f"{DATA_ROOT.name}_aug"
    use_mosaic = (AUG_ROOT / "split" / "train_mosaic.txt").exists()
    if use_mosaic:
        print(f"[mosaic] 检测到增强数据: {AUG_ROOT}")

    for name, out_path in [("train", OUT_TRAIN), ("val", OUT_VAL), ("test", OUT_TEST)]:
        samples, counts = convert_split(name, class_names)
        # train 额外合并 Mosaic 增强样本
        if name == "train" and use_mosaic:
            aug_samples, aug_counts = convert_split("train_mosaic", class_names, AUG_ROOT)
            samples = samples + aug_samples
            for c, n in aug_counts.items():
                counts[c] += n
            print(f"[mosaic] +{len(aug_samples)} mosaic samples merged into train")
        write_jsonl(out_path, samples)
        all_samples += samples
        for c, n in counts.items():
            total_box_counts[c] += n
        n_boxes = sum(counts.values())
        print(f"[{name}] images={len(samples)} boxes={n_boxes} -> {out_path}")

    write_jsonl(OUT_ALL, all_samples)

    # 统计：最大目标数/图、最长 assistant 文本长度（粗估 token 上限参考）
    max_objs = 0
    max_assist_len = 0
    for rec in all_samples:
        objs = assistant_targets(rec["conversations"][1]["value"])
        max_objs = max(max_objs, len(objs))
        max_assist_len = max(max_assist_len, len(rec["conversations"][1]["value"]))

    print("=" * 60)
    print(f"total samples: {len(all_samples)}  total boxes: {sum(total_box_counts.values())}")
    print("per-class box counts:")
    for c, n in total_box_counts.items():
        print(f"  {c}: {n}")
    print(f"max objects/image: {max_objs}")
    print(f"max assistant text len (chars): {max_assist_len}")
    print("output:")
    print(f"  {OUT_ALL}")
    print(f"  {OUT_TRAIN}")
    print(f"  {OUT_VAL}")
    print(f"  {OUT_TEST}")
    print("-" * 60)
    print("first 2 samples:")
    for rec in all_samples[:2]:
        print(json.dumps(rec, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
