"""将 0-1000 归一化 bbox 预测结果绘制到原图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_PREDICTION = (
    '<obj>[{"bbox_2d":[240,200,450,750],"label":"CJ"},'
    '{"bbox_2d":[500,150,950,800],"label":"CJ"},'
    '{"bbox_2d":[300,600,700,900],"label":"CJ"},'
    '{"bbox_2d":[240,200,450,750],"label":"FS"},'
    '{"bbox_2d":[500,150,950,800],"label":"FS"},'
    '{"bbox_2d":[240,200,450,750],"label":"JG"},'
    '{"bbox_2d":[500,150,950,800],"label":"JG"},'
    '{"bbox_2d":[300,100,700,600],"label":"BX"}]</obj>'
)

LABEL_COLORS = {
    "AJ": (230, 25, 75),
    "BX": (245, 130, 48),
    "CJ": (0, 130, 200),
    "CK": (145, 30, 180),
    "CR": (70, 240, 240),
    "FS": (240, 50, 230),
    "JG": (210, 245, 60),
    "PL": (250, 190, 190),
    "SG": (0, 128, 128),
    "TJ": (170, 110, 40),
    "TL": (128, 128, 128),
    "ZW": (0, 0, 128),
}
DEFAULT_COLOR = (255, 255, 255)
COORDINATE_SCALE = 1000


def parse_prediction_text(text: str) -> list[dict[str, Any]]:
    """入参: 包含 `<obj>...</obj>` 的预测文本或裸 JSON 数组。
    方法: 去除包装标签，解析 JSON，并校验每个目标的 bbox_2d 与 label 字段。
    出参: 目标字典列表；格式错误或坐标越界时抛出 ValueError。"""
    body = text.replace("<obj>", "").replace("</obj>", "").strip()
    try:
        targets = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError(f"prediction is not valid JSON: {error}") from error
    if not isinstance(targets, list):
        raise ValueError("prediction JSON must be an array.")
    normalized: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        if not isinstance(target, dict):
            raise ValueError(f"target {index} must be an object.")
        box = target.get("bbox_2d")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"target {index} bbox_2d must contain four coordinates.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
            raise ValueError(f"target {index} bbox_2d must contain numeric coordinates.")
        if any(value < 0 or value > COORDINATE_SCALE for value in box):
            raise ValueError(f"target {index} bbox_2d coordinates must be in [0, 1000].")
        if box[0] > box[2] or box[1] > box[3]:
            raise ValueError(f"target {index} bbox_2d must use xyxy order.")
        normalized.append({"bbox_2d": [round(value) for value in box], "label": str(target.get("label", "object"))})
    return normalized


def load_targets(prediction_file: Path | None, prediction_text: str | None) -> list[dict[str, Any]]:
    """入参: 可选 prediction_file、prediction_text；两者不能同时提供。
    方法: 优先读取文件，其次读取命令行文本，均未提供时使用脚本内置示例预测。
    出参: 已校验的归一化目标列表。"""
    if prediction_file is not None and prediction_text is not None:
        raise ValueError("use either --prediction-file or --prediction-text, not both.")
    if prediction_file is not None:
        return parse_prediction_text(prediction_file.read_text(encoding="utf-8"))
    return parse_prediction_text(prediction_text if prediction_text is not None else DEFAULT_PREDICTION)


def normalized_to_pixels(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    """入参: 0-1000 归一化 xyxy 框及原图宽高。
    方法: 按图像宽高分别反归一化，并将结果限制在图像边界内。
    出参: 原图像素坐标 `(x1, y1, x2, y2)`。"""
    return (
        round(box[0] * width / COORDINATE_SCALE),
        round(box[1] * height / COORDINATE_SCALE),
        round(box[2] * width / COORDINATE_SCALE),
        round(box[3] * height / COORDINATE_SCALE),
    )


def load_font(size: int) -> ImageFont.ImageFont:
    """入参: 字号像素值。
    方法: 优先加载常见 Windows 字体，找不到时回退 Pillow 默认字体。
    出参: 可供 ImageDraw 使用的字体对象。"""
    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_prediction(image_path: Path, output_path: Path, targets: list[dict[str, Any]]) -> None:
    """入参: 原图路径、输出 PNG 路径和归一化预测目标列表。
    方法: 将每个 bbox 反归一化后绘制彩色边框、序号、类别和归一化坐标；重叠框使用不同颜色区分。
    出参: 无；输出目录不存在时自动创建并写出 PNG。"""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = load_font(max(18, round(min(width, height) / 55)))
    small_font = load_font(max(14, round(min(width, height) / 75)))
    stroke = max(3, round(min(width, height) / 300))
    for index, target in enumerate(targets, start=1):
        normalized_box = target["bbox_2d"]
        pixel_box = normalized_to_pixels(normalized_box, width, height)
        label = target["label"]
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)
        draw.rectangle(pixel_box, outline=color, width=stroke)
        title = f"#{index} {label}"
        coordinate_text = f"[{','.join(str(value) for value in normalized_box)}]"
        title_box = draw.textbbox((0, 0), title, font=font)
        coordinate_box = draw.textbbox((0, 0), coordinate_text, font=small_font)
        text_width = max(title_box[2] - title_box[0], coordinate_box[2] - coordinate_box[0]) + 12
        text_height = title_box[3] - title_box[1] + coordinate_box[3] - coordinate_box[1] + 12
        text_x = max(0, min(pixel_box[0], width - text_width))
        text_y = pixel_box[1] - text_height - 3
        if text_y < 0:
            text_y = min(pixel_box[1] + 3, height - text_height)
        draw.rectangle((text_x, text_y, text_x + text_width, text_y + text_height), fill=color)
        draw.text((text_x + 6, text_y + 3), title, fill=(0, 0, 0), font=font)
        draw.text((text_x + 6, text_y + 3 + title_box[3] - title_box[1]), coordinate_text, fill=(0, 0, 0), font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    """入参: 命令行中的原图、输出路径及可选预测来源。
    方法: 解析预测，执行归一化坐标到像素坐标的可视化。
    出参: 无；完成后打印输出文件路径。"""
    parser = argparse.ArgumentParser(description="Visualize normalized bbox predictions on an image.")
    parser.add_argument("--image", type=Path, required=True, help="原图路径")
    parser.add_argument("--output", type=Path, default=None, help="输出 PNG 路径，默认在原图旁生成 *_bbox_vis.png")
    parser.add_argument("--prediction-file", type=Path, default=None, help="包含 <obj>...</obj> 预测文本的 UTF-8 文件")
    parser.add_argument("--prediction-text", default=None, help="直接传入 <obj>...</obj> 预测文本")
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")
    targets = load_targets(args.prediction_file, args.prediction_text)
    output_path = args.output or args.image.with_name(f"{args.image.stem}_bbox_vis.png")
    draw_prediction(args.image, output_path, targets)
    print(f"visualization: {output_path.resolve()}")


if __name__ == "__main__":
    main()
