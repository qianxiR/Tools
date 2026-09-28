# -*- coding: utf-8 -*-
"""QwenDec 目标检测数据集英文动作提示批量生成与写回脚本。
从现有 QwenVL JSONL 检测样本中读取类别，调用 DashScope Qwen 生成“Detect xx targets in xx scenes”式英文动作提示，并同步更新训练/验证/全量数据。"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
from pathlib import Path
from time import sleep
from typing import Any

from openai import OpenAI
from PIL import Image


HERE = Path(__file__).resolve().parents[1]
DATA_DIR = HERE / "data"
DATASET_ROOT = Path(r"H:/西藏遥感Agent/dataes/LandSlideDataSet")
DEFAULT_JSONL_FILES = [
    DATASET_ROOT / "landslide_qwenvl.jsonl",
    DATASET_ROOT / "landslide_qwenvl_train.jsonl",
    DATASET_ROOT / "landslide_qwenvl_val.jsonl",
]
PROMPT_JSON = DATA_DIR / "detection_prompt.json"
DEFAULT_MODEL = "qwen-vl-plus"
DEFAULT_LABEL = "slide"
DEFAULT_TARGET_CN = "滑坡"
DEFAULT_SCENE_CN = "山地场景"
DEFAULT_TARGET_EN = "landslide"
DEFAULT_SCENE_EN = "mountainous scenes"


def parse_args() -> argparse.Namespace:
    """入参: 命令行参数 | 方法: 解析 JSONL 路径、目标类别、场景、Qwen 模型和是否只预览 | 出参: argparse.Namespace"""
    parser = argparse.ArgumentParser(description="Generate and inject detection action prompts for QwenDec JSONL datasets.")
    parser.add_argument("--jsonl", action="append", type=Path, default=None, help="需要更新的 JSONL；可重复传入。")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="检测输出 JSON 中使用的英文 label。")
    parser.add_argument("--target-cn", default=DEFAULT_TARGET_CN, help="中文目标名称，例如 滑坡。")
    parser.add_argument("--scene-cn", default=DEFAULT_SCENE_CN, help="中文场景名称，例如 山地场景。")
    parser.add_argument("--target-en", default=DEFAULT_TARGET_EN, help="英文目标名称，例如 landslide。")
    parser.add_argument("--scene-en", default=DEFAULT_SCENE_EN, help="英文场景名称，例如 mountainous scenes。")
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", DEFAULT_MODEL), help="DashScope 兼容 OpenAI 模型名。")
    parser.add_argument("--temperature", type=float, default=0.2, help="Qwen 生成温度。")
    parser.add_argument("--sample-images", type=int, default=0, help="传给 Qwen 的样例图片数量；0 表示只用文本元信息。")
    parser.add_argument("--dry-run", action="store_true", help="只打印将写入的 prompt，不修改 JSONL。")
    parser.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份文件。")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """入参: JSONL 路径 | 方法: 逐行解析非空 JSON 记录 | 出参: 样本字典列表"""
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]], backup: bool):
    """入参: 输出路径、样本列表、是否备份 | 方法: 可选复制 .bak 后按 UTF-8 JSONL 写回 | 出参: 无"""
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_labels(records: list[dict[str, Any]]) -> list[str]:
    """入参: 样本列表 | 方法: 从 assistant JSON 中提取 bbox 目标 label 并去重排序 | 出参: label 字符串列表"""
    labels = set()
    for record in records:
        conversations = record.get("conversations", [])
        if len(conversations) < 2:
            continue
        text = conversations[1].get("value", "[]")
        for item in parse_assistant_targets(text):
            label = str(item.get("label", "")).strip()
            if label:
                labels.add(label)
    return sorted(labels)


def parse_assistant_targets(text: str) -> list[dict[str, Any]]:
    """入参: assistant 文本 | 方法: 解析 JSON 数组，失败时返回空列表 | 出参: 目标字典列表"""
    try:
        data = json.loads(text)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def sample_image_paths(records: list[dict[str, Any]], limit: int) -> list[Path]:
    """入参: 样本列表、最多图片数 | 方法: 按记录顺序选取存在的图片路径 | 出参: Path 列表"""
    paths = []
    for record in records:
        path = Path(str(record.get("image", "")))
        if path.exists():
            paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def resize_image(image: Image.Image, max_dim: int = 768) -> Image.Image:
    """入参: PIL 图像和最长边 | 方法: 等比例缩小大图，避免 API 请求体过大 | 出参: PIL 图像"""
    width, height = image.size
    if max(width, height) <= max_dim:
        return image
    scale = max_dim / float(max(width, height))
    return image.resize((int(round(width * scale)), int(round(height * scale))), Image.LANCZOS)


def image_to_base64_url(path: Path) -> str:
    """入参: 图片路径 | 方法: 读取图片、转 RGB JPEG、编码为 OpenAI image_url data URL | 出参: data URL 字符串"""
    image = Image.open(path).convert("RGB")
    image = resize_image(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_user_content(image_paths: list[Path], text: str) -> list[dict[str, Any]]:
    """入参: 图片路径列表、文本提示 | 方法: 构造 DashScope 兼容 OpenAI 多模态 content | 出参: content 列表"""
    content = []
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_to_base64_url(path)}})
    content.append({"type": "text", "text": text})
    return content


def extract_json_object(text: str) -> dict[str, Any]:
    """入参: 模型响应文本 | 方法: 去除 markdown 围栏并解析 JSON 对象 | 出参: JSON 字典"""
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean).strip()
    return json.loads(clean)


def qwen_client() -> OpenAI | None:
    """入参: 无 | 方法: 从 DASHSCOPE_API_KEY 创建 OpenAI 兼容客户端 | 出参: OpenAI 客户端或 None"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def generate_action_prompt_with_qwen(
    client: OpenAI | None,
    model: str,
    scene_cn: str,
    target_cn: str,
    scene_en: str,
    target_en: str,
    label: str,
    image_paths: list[Path],
    temperature: float,
) -> tuple[str, str]:
    """入参: Qwen 客户端、模型、中英文场景、中英文目标、label、样例图、温度 | 方法: 调用 Qwen 生成英文动作提示，失败时回退模板 | 出参: (动作提示, 来源)"""
    fallback = f"Detect {target_en} targets in {scene_en}"
    if client is None:
        return fallback, "template_no_api_key"
    task = (
        "You are a remote sensing object detection dataset engineer. "
        "Generate one concise English action prompt for an object detection dataset.\n"
        f"Scene in Chinese: {scene_cn}\n"
        f"Target in Chinese: {target_cn}\n"
        f"Scene in English: {scene_en}\n"
        f"Target in English: {target_en}\n"
        f"Output label: {label}\n"
        "Requirements:\n"
        "1. Describe only the detection action the model should perform.\n"
        "2. Use a concise wording like \"Detect <target> targets in <scene>\".\n"
        "3. Do not include coordinate format, JSON output rules, or training instructions.\n"
        "4. The action_prompt must be English only.\n"
        '5. Return strictly JSON: {"action_prompt":"..."}'
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": build_user_content(image_paths, task)}],
                max_tokens=256,
                temperature=temperature,
                extra_body={"disable_search": True},
            )
            data = extract_json_object(response.choices[0].message.content or "{}")
            action_prompt = str(data.get("action_prompt", "")).strip(" 。\n\t")
            if action_prompt:
                return action_prompt, f"qwen:{model}"
        except Exception as exc:
            print(f"[QWEN] action prompt attempt {attempt + 1}/3 failed: {exc}")
            if attempt < 2:
                sleep(5)
    return fallback, "template_after_qwen_failed"


def build_detection_prompt(action_prompt: str, label: str) -> str:
    """入参: 英文动作提示和输出 label | 方法: 拼接模型训练所需的检测动作与 JSON 输出约束 | 出参: 完整英文 user prompt"""
    return (
        f"{action_prompt}. Output a JSON array. "
        f"Each target must be {{\"bbox_2d\":[\"<loc_qx1>\",\"<loc_qy1>\",\"<loc_qx2>\",\"<loc_qy2>\"],\"label\":\"{label}\"}}, "
        "where each q value is a 0-999 bin quantized from the normalized 0-1000 coordinate. "
        "If no target exists, output []."
    )


def update_records(records: list[dict[str, Any]], full_prompt: str, action_prompt: str, source: str) -> list[dict[str, Any]]:
    """入参: 样本列表、完整提示、动作提示、来源 | 方法: 替换 user 轮文本并附加提示元数据 | 出参: 更新后的样本列表"""
    for record in records:
        conversations = record.setdefault("conversations", [])
        if not conversations:
            conversations.append({"from": "user", "value": ""})
        conversations[0]["from"] = "user"
        conversations[0]["value"] = f"<image>\n{full_prompt}"
        record["detection_action_prompt"] = action_prompt
        record["detection_prompt_source"] = source
    return records


def write_prompt_json(path: Path, payload: dict[str, Any], backup: bool):
    """入参: prompt JSON 路径、内容、是否备份 | 方法: 可选备份后写入缩进 JSON | 出参: 无"""
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    """入参: 命令行参数 | 方法: 生成动作提示、更新 JSONL、写 prompt 元数据 | 出参: 无"""
    args = parse_args()
    jsonl_files = args.jsonl or DEFAULT_JSONL_FILES
    all_records = []
    for path in jsonl_files:
        all_records.extend(read_jsonl(path))
    if not all_records:
        raise RuntimeError("未读取到任何 JSONL 样本。")

    labels = extract_labels(all_records)
    label = args.label or (labels[0] if labels else DEFAULT_LABEL)
    image_paths = sample_image_paths(all_records, args.sample_images)
    action_prompt, source = generate_action_prompt_with_qwen(
        client=qwen_client(),
        model=args.model,
        scene_cn=args.scene_cn,
        target_cn=args.target_cn,
        scene_en=args.scene_en,
        target_en=args.target_en,
        label=label,
        image_paths=image_paths,
        temperature=args.temperature,
    )
    full_prompt = build_detection_prompt(action_prompt, label)
    payload = {
        "action_prompt": action_prompt,
        "full_prompt": full_prompt,
        "label": label,
        "target_cn": args.target_cn,
        "scene_cn": args.scene_cn,
        "target_en": args.target_en,
        "scene_en": args.scene_en,
        "source": source,
        "model": args.model,
        "jsonl_files": [str(path) for path in jsonl_files],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    backup = not args.no_backup
    write_prompt_json(PROMPT_JSON, payload, backup=backup)
    for path in jsonl_files:
        records = read_jsonl(path)
        update_records(records, full_prompt, action_prompt, source)
        write_jsonl(path, records, backup=backup)
        print(f"[OK] updated {path} records={len(records)}")


if __name__ == "__main__":
    main()
