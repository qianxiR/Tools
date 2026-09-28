#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQA 可视化工具 — 将分割/变化检测推理结果与 QA 问答合并为对比图

SEG 布局（六列）: 原图 | 分割掩码 | 叠加图 | QA面板 | MCQ面板 | Caption面板
CD  布局（六列）: T1 | T2 | 变化掩码 | QA面板 | MCQ面板 | Caption面板

用法:
python utils/visualize_vqa.py --mode seg --data_root "E:/xzkjxm/dataes/CD_20" --split all
python utils/visualize_vqa.py --mode cd --data_root "E:/xzkjxm/dataes/CD-CD_20" --split all
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════
#  常量配置
# ═══════════════════════════════════════════════════════════════
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMG_H = 512
PANEL_W = 480
PAD = 10
FONT_BODY = 12
FONT_TITLE = 13
LABEL_H = 22
GAP = 2

# 各模式图像列定义: (子目录名, 标签名)
SEG_COLUMNS = [("image", "Image"), ("seg_mask", "Seg Mask"), ("overlay", "Overlay")]
CD_COLUMNS = [("t1", "T1"), ("t2", "T2"), ("change_mask", "Change")]

# 问答类型视觉样式
TYPE_STYLE = {
    "qa":      {"bg": (52, 152, 219), "label": "QA"},
    "mcq":     {"bg": (155, 89, 182), "label": "MCQ"},
    "caption": {"bg": (46, 204, 113), "label": "Caption"},
}


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def _font(size: int) -> ImageFont.FreeTypeFont:
    """
    入参: size — 字体像素大小
    方法: 按优先级尝试 Windows 字体路径，最终回退 PIL 默认
    出参: 可用字体实例
    """
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def _find_img(split_dir: Path, subdir: str, stem: str) -> Optional[Path]:
    """
    入参: split_dir — split 目录, subdir — 子目录名, stem — 文件名(无后缀)
    方法: 在 split_dir/subdir/ 中查找 stem 匹配的图像文件
    出参: 图像路径或 None
    """
    d = split_dir / subdir
    if not d.is_dir():
        return None
    for ext in IMAGE_EXT:
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _load_resize(path: Path, target_h: int) -> Image.Image:
    """
    入参: path — 图像路径, target_h — 目标高度
    方法: 加载 RGB 图像并等比缩放到目标高度
    出参: 缩放后的 PIL Image
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if h != target_h:
        img = img.resize((max(1, int(w * target_h / h)), target_h), Image.LANCZOS)
    return img


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int, y: int,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
    max_w: int,
) -> int:
    """
    入参: draw — PIL 绘图对象, text — 文本, x/y — 起点坐标,
          font — 字体, fill — 文字颜色, max_w — 最大行宽
    方法: 逐字符测量宽度，超出 max_w 则换行
    出参: 绘制完成后的 y 坐标
    """
    line_h = FONT_BODY + 5
    line = ""
    for ch in text:
        test = line + ch
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] > max_w:
            draw.text((x, y), line, fill=fill, font=font)
            y += line_h
            line = ch
        else:
            line = test
    if line:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h
    return y


def _add_label(img: Image.Image, label: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """
    入参: img — 图像列, label — 列标题, font — 标题字体
    方法: 在图像顶部添加深色标签栏
    出参: 带标签的图像
    """
    w, h = img.size
    canvas = Image.new("RGB", (w, h + LABEL_H), (50, 50, 55))
    canvas.paste(img, (0, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 3), label, fill=(240, 240, 240), font=font)
    return canvas


def _hconcat(*imgs: Image.Image) -> Image.Image:
    """
    入参: imgs — 待拼接的图像列表
    方法: 取最大高度，水平拼接，底部对齐
    出参: 拼接后的图像
    """
    max_h = max(i.size[1] for i in imgs)
    total_w = sum(i.size[0] for i in imgs)
    canvas = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for img in imgs:
        canvas.paste(img, (x, max_h - img.size[1]))
        x += img.size[0]
    return canvas


# ═══════════════════════════════════════════════════════════════
#  问答面板渲染
# ═══════════════════════════════════════════════════════════════

def _render_panel(
    items: List[Dict[str, Any]],
    qtype: str,
    height: int,
) -> Image.Image:
    """
    入参:
        items — 该类型的问答列表（已过滤为单一 type）
        qtype — 问答类型: qa / mcq / caption
        height — 面板总高度（与图像列一致）
    方法:
        按 qtype 决定渲染策略：
        - qa: 显示问题 + Yes/No 答案
        - mcq: 显示问题 + A-D 选项（正确答案标红加 ✓）
        - caption: 显示问题 + 描述性答案
    出参:
        PIL.Image — 单类型问答面板
    """
    font = _font(FONT_BODY)
    title_font = _font(FONT_TITLE)
    style = TYPE_STYLE[qtype]
    content_w = PANEL_W - 2 * PAD
    line_h = FONT_BODY + 5

    panel = Image.new("RGB", (PANEL_W, height), (248, 248, 248))
    draw = ImageDraw.Draw(panel)

    y = PAD

    # 类型标题标签
    badge_w = len(style["label"]) * 9 + 20
    draw.rectangle([PAD, y, PAD + badge_w, y + FONT_TITLE + 8], fill=style["bg"])
    draw.text((PAD + 10, y + 2), style["label"], fill=(255, 255, 255), font=title_font)
    y += FONT_TITLE + 16

    for idx, item in enumerate(items, 1):
        question = str(item.get("question", ""))
        answer = str(item.get("answer", ""))

        # 剩余空间不足时截断
        if y + line_h * 2 > height - PAD:
            draw.text((PAD + 8, y), f"... +{len(items) - idx + 1} more",
                      fill=(160, 160, 160), font=font)
            break

        # Q序号 + 问题文本（自动换行）
        y = _draw_wrapped(draw, f"Q{idx}: {question}",
                          PAD + 4, y, font, (30, 30, 30), content_w - 4)

        if qtype == "mcq":
            # MCQ: 显示 A-D 四个选项，正确答案标红加前缀
            options = item.get("options", {})
            for key in ["A", "B", "C", "D"]:
                opt_text = str(options.get(key, ""))
                is_correct = (answer.upper() == key)
                color = (220, 50, 50) if is_correct else (100, 100, 100)
                prefix = "✓ " if is_correct else "  "
                y = _draw_wrapped(
                    draw, f"{prefix}{key}: {opt_text}",
                    PAD + 20, y, font, color, content_w - 20,
                )
        else:
            # QA / Caption: 显示答案（红色高亮）
            y = _draw_wrapped(
                draw, f"→ {answer}",
                PAD + 20, y, font, (200, 50, 50), content_w - 20,
            )

        y += 8

    return panel


# ═══════════════════════════════════════════════════════════════
#  主可视化逻辑
# ═══════════════════════════════════════════════════════════════

def visualize(
    data_root: str,
    split: str,
    output_dir: Path,
    mode: str,
    max_samples: int,
) -> None:
    """
    入参:
        data_root — 数据集根目录
        split — 数据划分 (train/val/test)
        output_dir — 可视化输出目录
        mode — seg 或 cd
        max_samples — 最多处理样本数（0=全部）
    方法:
        读取 qa_{split}.json，对每个样本：
        1. 加载图像列（SEG: image/seg_mask/overlay, CD: t1/t2/change_mask）
        2. 按 type 拆分问答为 qa/mcq/caption 三组
        3. 分别渲染三组问答面板
        4. 水平拼接所有列并保存
    出参:
        None（直接写文件）
    """
    split_dir = Path(data_root) / split
    qa_path = split_dir / f"qa_{split}.json"

    if not qa_path.exists():
        print(f"ERROR: {qa_path} 不存在，请先运行 VQA pipeline")
        return

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # 过滤有效样本（qa 字段存在且无 error）
    valid = [s for s in qa_data if s.get("qa") and not s.get("error")]
    if max_samples > 0:
        valid = valid[:max_samples]

    if not valid:
        print("没有有效样本")
        return

    # 根据模式选择图像列定义
    columns = SEG_COLUMNS if mode == "seg" else CD_COLUMNS
    title_font = _font(FONT_TITLE)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"模式: {mode.upper()}, 样本数: {len(valid)}, 输出: {output_dir}")

    for idx, sample in enumerate(valid, 1):
        qa_record = sample["qa"]
        stem = str(qa_record.get("image_name", sample.get("image_name", f"sample_{idx}")))
        qa_items = qa_record.get("qa", [])

        # 按 type 拆分问答
        grouped: Dict[str, List[Dict]] = {"qa": [], "mcq": [], "caption": []}
        for item in qa_items:
            t = str(item.get("type", "")).strip().lower()
            if t in grouped:
                grouped[t].append(item)

        # 加载图像列（缺失时用灰色占位）
        img_cols = []
        for subdir, label in columns:
            p = _find_img(split_dir, subdir, stem)
            if p:
                img = _load_resize(p, IMG_H)
            else:
                img = Image.new("RGB", (120, IMG_H), (210, 210, 210))
            img_cols.append(_add_label(img, label, title_font))

        # 渲染三种问答面板
        panel_cols = []
        for qtype in ["qa", "mcq", "caption"]:
            panel = _render_panel(grouped[qtype], qtype, IMG_H)
            panel_cols.append(_add_label(panel, TYPE_STYLE[qtype]["label"], title_font))

        # 列间分隔线 + 水平拼接
        total_h = IMG_H + LABEL_H
        gap_col = Image.new("RGB", (GAP, total_h), (180, 180, 180))
        all_cols = []
        for col in img_cols + panel_cols:
            all_cols.append(col)
            all_cols.append(gap_col)

        composite = _hconcat(*all_cols[:-1])  # 末尾多余 gap 去掉

        out_path = output_dir / f"{stem}.png"
        composite.save(str(out_path))
        print(f"  [{idx}/{len(valid)}] {stem}")

    print(f"完成: {len(valid)} 张可视化 → {output_dir}")


def main():
    """
    入参: 命令行参数（mode, data_root, split, output, max_samples）
    方法: 解析参数，支持 split=all 时自动遍历 train/val/test 三个划分
    出参: None
    """
    parser = argparse.ArgumentParser(
        description="VQA 可视化工具 (SEG/CD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["seg", "cd"],
                        help="模式: seg=语义分割, cd=变化检测")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据集根目录")
    parser.add_argument("--split", type=str, default="test",
                        help="数据划分 (train/val/test/all, default: test)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认: {data_root}/{split}/vis_vqa）")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最多处理样本数（0=全部）")
    args = parser.parse_args()

    # split=all 时展开为三个标准划分，其余情况保持原值
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for sp in splits:
        output = Path(args.output) if args.output else Path(args.data_root) / sp / "vis_vqa"
        visualize(args.data_root, sp, output, args.mode, args.max_samples)


if __name__ == "__main__":
    main()
