#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
局部 patch 修正与 metadata 汇总工具（Step 4）

功能：
1. 遍历每个样本目录下的 regions.json
2. 对 clip 与 llm 分类不一致的局部 patch 调用 VLM 进行修正
3. 将修正后的结果直接写回 classification.llm
4. 汇总生成 split 级别的 metadata.json

输出：metadata.json
"""
import os
import json
import re
import base64
import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import numpy as np
from openai import OpenAI


VALID_CLASSES = {
    "building", "highway", "vegetation", "farmland", "bare_land", "water",
}

CLASS_ALIASES = {
    "bareland": "bare_land",
    "barren_land": "bare_land",
    "barrenland": "bare_land",
    "road": "highway",
    "roads": "highway",
}


def get_message_content(message) -> str:
    """
    入参:
        message: OpenAI 返回的 message 对象

    方法:
        兼容普通文本返回与带 reasoning 字段的推理模型返回，统一抽取最终可解析文本。

    出参:
        str: 提取后的文本内容
    """
    content = message.content or getattr(message, "reasoning", None) or ""
    if content.startswith("Thinking Process"):
        lines = content.split("\n")
        for idx, line in enumerate(lines):
            if line.startswith("Answer:") or line.startswith("回答:"):
                return "\n".join(lines[idx + 1:]).strip()
    return content


def normalize_class_label(value: Any) -> str:
    """
    入参:
        value: 原始类别值

    方法:
        将不同写法统一映射到当前数据集允许的 6 个标准类别名，避免 llm/clip 输出风格不一致导致误判。

    出参:
        str: 归一化后的类别名；若为空或无效则返回空字符串
    """
    if value is None:
        return ""
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    label = CLASS_ALIASES.get(label, label)
    if label not in VALID_CLASSES:
        return ""
    return label


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """
    入参:
        path: 本地图像路径

    方法:
        使用 np.fromfile + cv2.imdecode 读取图像，保证中文路径下局部 patch 也能正常访问。

    出参:
        Optional[np.ndarray]: 读取成功返回图像数组，失败返回 None
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return image
    except Exception:
        return None


def encode_image_to_base64(image_path: Path, resize_max_dim: int = 512) -> str:
    """
    入参:
        image_path: 局部 patch 图像路径
        resize_max_dim: 图像最长边限制

    方法:
        1. 优先按最长边压缩 patch，减少多模态请求体积
        2. 统一编码为 JPEG data URL，兼顾兼容性与网络开销
        3. 若图像无法解码，则回退到原始二进制文件直传

    出参:
        str: 可直接发给多模态接口的 data URL
    """
    image = imread_unicode(image_path)
    if image is not None:
        height, width = image.shape[:2]
        if max(height, width) > resize_max_dim:
            scale = resize_max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError(f"Failed to encode patch image: {image_path}")
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime_type = mime_map.get(suffix, "image/png")
    return f"data:{mime_type};base64,{encoded}"

def find_image_path(split_dir: Path, subdir: str, name: str) -> str:
    """
    入参:
        split_dir: split 目录路径
        subdir: 图像所在子目录名称
        name: 图像主文件名（不含扩展名）

    方法:
        依次尝试常见遥感图像扩展名，找到后返回相对 split 目录的路径。
        这样做是为了兼容不同数据集图像格式，同时保持 metadata 中路径字段统一。

    出参:
        str: 相对路径；若未找到则返回空字符串
    """
    for ext in ["png", "jpg", "jpeg", "tif", "tiff"]:
        candidate = split_dir / subdir / f"{name}.{ext}"
        if candidate.exists():
            return f"{subdir}/{name}.{ext}"
    return ""


def load_json(path: Path) -> Dict[str, Any]:
    """
    入参:
        path: JSON 文件路径

    方法:
        使用 UTF-8 编码读取 JSON，保证中文路径和中文字段兼容。

    出参:
        Dict[str, Any]: 解析后的 JSON 对象
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """
    入参:
        path: 输出 JSON 文件路径
        data: metadata 记录列表或 regions.json 对象

    方法:
        以缩进格式写出，方便后续人工检查和下游脚本直接读取。

    出参:
        None
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_patch_paths(regions_json_path: Path, region: Dict[str, Any]) -> Dict[str, Path]:
    """
    入参:
        regions_json_path: 当前样本的 regions.json 路径
        region: 单个变化区域结构

    方法:
        结合 Step 1 写入的 files 字段，从 split/regions 根目录解析出 t1/t2/label 局部 patch 路径。

    出参:
        Dict[str, Path]: {"t1": Path, "t2": Path, "label": Path}
    """
    regions_root = regions_json_path.parent.parent
    files = region.get("files") or {}
    return {
        "t1": regions_root / str(files.get("t1", "")).strip(),
        "t2": regions_root / str(files.get("t2", "")).strip(),
        "label": regions_root / str(files.get("label", "")).strip(),
    }


def should_refine_region(region: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    入参:
        region: 单个变化区域结构

    方法:
        1. 仅对 region_id > 0 的有效变化区域执行修正
        2. 仅当 clip 与 llm 在 t1 或 t2 上存在明确且有效的不一致时触发 VLM 局部 patch 修正
        3. 返回双方类别，便于后续直接构造提示信息

    出参:
        Tuple[bool, str, str]: (是否需要修正, llm_pair, clip_pair)
    """
    region_id = int(region.get("region_id", 0) or 0)
    if region_id <= 0:
        return False, "", ""

    classification = region.get("classification") or {}
    llm = classification.get("llm") or {}
    clip = classification.get("clip") or {}

    llm_t1 = normalize_class_label(llm.get("t1_class"))
    llm_t2 = normalize_class_label(llm.get("t2_class"))
    clip_t1 = normalize_class_label(clip.get("t1_class"))
    clip_t2 = normalize_class_label(clip.get("t2_class"))

    if not llm_t1 or not llm_t2 or not clip_t1 or not clip_t2:
        return False, "", ""

    llm_pair = f"{llm_t1} -> {llm_t2}"
    clip_pair = f"{clip_t1} -> {clip_t2}"
    return llm_pair != clip_pair, llm_pair, clip_pair


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    入参:
        text: 模型返回的原始文本

    方法:
        尝试从 markdown 代码块或普通文本中提取 JSON 对象，兼容常见模型输出风格。

    出参:
        Dict[str, Any]: 解析后的 JSON 对象
    """
    content = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)
    else:
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            content = content[first_brace:last_brace + 1]
    return json.loads(content)


def load_prompt_template() -> Tuple[str, str]:
    """
    入参:
        无

    方法:
        从 prompt_cd_refine_region.md 中提取 system prompt 和 user content 模板。
        解析 markdown 中 "### System Prompt" 和 "### User Content" 下方的代码块，
        将其作为 prompt 模板返回，运行时通过 str.format() 填充变量。

    出参:
        Tuple[str, str]: (system_prompt, user_content_template)
    """
    script_dir = Path(__file__).parent
    prompt_path = script_dir / "prompt_cd_refine_region.md"
    content = prompt_path.read_text(encoding="utf-8")

    # 提取 ### System Prompt 下的代码块
    system_match = re.search(
        r"###\s*System\s+Prompt\s*\n```\s*\n(.*?)```", content, re.DOTALL
    )
    system_prompt = system_match.group(1).strip() if system_match else ""

    # 提取 ### User Content 下的代码块
    user_match = re.search(
        r"###\s*User\s+Content\s*\n```\s*\n(.*?)```", content, re.DOTALL
    )
    user_template = user_match.group(1).strip() if user_match else ""

    return system_prompt, user_template


def refine_region_with_patch(
    client: OpenAI,
    model_name: str,
    regions_json_path: Path,
    region: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """
    入参:
        client: OpenAI 客户端
        model_name: 多模态模型名称
        regions_json_path: 当前样本的 regions.json 路径
        region: 单个变化区域结构

    方法:
        1. 读取该 region 的 t1/t2/label 局部 patch
        2. 当 llm 与 clip 不一致时，调用 VLM 只看局部 patch 做二次判定
        3. 仅返回标准化后的 t1/t2 类别，调用方再决定是否覆盖 llm 结果

    出参:
        Optional[Dict[str, str]]: {"t1_class": ..., "t2_class": ...}；若不需要修正或修正失败则返回 None
    """
    should_refine, llm_pair, clip_pair = should_refine_region(region)
    if not should_refine:
        return None

    patch_paths = resolve_patch_paths(regions_json_path, region)
    required_paths = [patch_paths["t1"], patch_paths["t2"]]
    if not all(path.exists() for path in required_paths):
        return None

    # 检查 patch 尺寸，VLM API 要求宽高均 > 10px
    for p in required_paths:
        img_check = imread_unicode(p)
        if img_check is None:
            return None
        h, w = img_check.shape[:2]
        if w <= 10 or h <= 10:
            return None

    classification = region.get("classification") or {}
    llm = classification.get("llm") or {}
    clip = classification.get("clip") or {}
    location = str(region.get("location", "")).strip().lower()

    system_prompt, user_template = load_prompt_template()
    user_text = user_template.format(
        location=location or "unknown",
        llm_t1_class=llm_pair.split(" -> ")[0],
        llm_t2_class=llm_pair.split(" -> ")[1],
        clip_t1_class=clip_pair.split(" -> ")[0],
        clip_t2_class=clip_pair.split(" -> ")[1],
    )

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": user_text},
        {"type": "text", "text": "T1 local patch:"},
        {"type": "image_url", "image_url": {"url": encode_image_to_base64(patch_paths["t1"])}},
        {"type": "text", "text": "T2 local patch:"},
        {"type": "image_url", "image_url": {"url": encode_image_to_base64(patch_paths["t2"])}},
    ]

    if patch_paths["label"].exists():
        user_content.extend([
            {"type": "text", "text": "Label patch:"},
            {"type": "image_url", "image_url": {"url": encode_image_to_base64(patch_paths["label"])}},
        ])

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=256,
        extra_body={"disable_search": True},
    )

    result = extract_json_object(get_message_content(response.choices[0].message))
    t1_class = normalize_class_label(result.get("t1_class"))
    t2_class = normalize_class_label(result.get("t2_class"))
    if not t1_class or not t2_class:
        raise ValueError("Patch refinement returned invalid class labels")

    return {"t1_class": t1_class, "t2_class": t2_class}


def apply_patch_refinement(
    client: Optional[OpenAI],
    model_name: str,
    regions_json_path: Path,
    regions_data: Dict[str, Any],
) -> int:
    """
    入参:
        client: OpenAI 客户端；为空时表示跳过修正流程
        model_name: 多模态模型名称
        regions_json_path: 当前样本的 regions.json 路径
        regions_data: 当前样本完整 regions 数据

    方法:
        1. 遍历所有 region，筛出 llm 与 clip 不一致的局部 patch
        2. 使用 VLM 对局部 patch 做修正，并把修正结果直接覆盖到 classification.llm
        3. 记录修正次数，供 Step 4 日志统计

    出参:
        int: 成功修正的 region 数量
    """
    if client is None:
        return 0

    refined_count = 0
    for region in regions_data.get("regions", []):
        refined = refine_region_with_patch(client, model_name, regions_json_path, region)
        if refined is None:
            continue

        classification = region.setdefault("classification", {})
        llm = classification.setdefault("llm", {})
        llm["t1_class"] = refined["t1_class"]
        llm["t2_class"] = refined["t2_class"]
        llm["refined_from_patch"] = True
        llm["refined_model"] = model_name
        refined_count += 1

    if refined_count > 0:
        save_json(regions_json_path, regions_data)

    return refined_count


def build_metadata_record(
    dataset_path: Path,
    split: str,
    regions_json_path: Path,
    client: Optional[OpenAI],
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """
    入参:
        dataset_path: 数据集根目录
        split: 数据划分名称
        regions_json_path: 单个样本的 regions.json 路径

    方法:
        1. 读取 regions.json
        2. 对 llm 与 clip 不一致的局部 patch 做 VLM 修正，并直接更新到 classification.llm
        3. 基于修正后的 regions 列表构建 metadata 记录

    出参:
        Optional[Dict[str, Any]]: 单条 metadata 记录；读取失败时返回 None
    """
    regions_data = load_json(regions_json_path)
    refined_regions = apply_patch_refinement(client, model_name, regions_json_path, regions_data)
    split_dir = dataset_path / split
    sample_name = regions_json_path.parent.name
    image_size = regions_data.get("image_size", {})
    img_w = int(image_size.get("w", 0) or 0)
    img_h = int(image_size.get("h", 0) or 0)

    record = {
        "image_name": regions_data.get("image_name", sample_name),
        "image_size": image_size,
        "total_pixels": img_w * img_h,
        "t1": find_image_path(split_dir, "t1", sample_name),
        "t2": find_image_path(split_dir, "t2", sample_name),
        "label": find_image_path(split_dir, "label", sample_name),
        "regions": regions_data.get("regions", []),
        "_step4_refined_regions": refined_regions,
    }

    return record


def process_single_sample(
    idx: int,
    total: int,
    dataset_path: Path,
    split: str,
    regions_json_path: Path,
    client: Optional[OpenAI],
    model_name: str,
    print_lock: threading.Lock,
) -> Optional[Dict[str, Any]]:
    """
    入参:
        idx: 当前样本序号
        total: 样本总数
        dataset_path: 数据集根目录
        split: 数据划分名称
        regions_json_path: 单个样本的 regions.json 路径
        client: OpenAI 客户端
        model_name: 多模态模型名称
        print_lock: 控制台打印锁

    方法:
        调用 Step 4 修正 + metadata 构建逻辑生成单样本记录。
        使用打印锁是为了在并行处理时保持日志输出整洁，避免样本信息交错。

    出参:
        Optional[Dict[str, Any]]: metadata 记录；失败时返回 None
    """
    sample_name = regions_json_path.parent.name

    with print_lock:
        print(f"[{idx}/{total}] {sample_name}")

    try:
        record = build_metadata_record(dataset_path, split, regions_json_path, client, model_name)
        with print_lock:
            region_count = len(record.get("regions", [])) if record else 0
            refined_count = int(record.get("_step4_refined_regions", 0)) if record else 0
            print(f"  - regions: {region_count}, refined: {refined_count}")
        return record
    except Exception as e:
        with print_lock:
            print(f"  ✗ {e}")
        return None


def process_split(
    dataset_path: str,
    split: str,
    client: Optional[OpenAI],
    model_name: str,
    max_workers: int = 1,
) -> None:
    """
    入参:
        dataset_path: 数据集根目录
        split: 数据划分名称
        client: OpenAI 客户端
        model_name: 多模态模型名称
        max_workers: 最大并行数

    方法:
        扫描 split 下所有样本的 regions.json，并行执行 patch 修正与 metadata 汇总。
        每拿到一条结果就刷新输出文件，这样中途中断时也能保留已完成部分。

    出参:
        None
    """
    dataset_root = Path(dataset_path)
    split_dir = dataset_root / split
    regions_root = split_dir / "regions"
    output_path = split_dir / "metadata.json"

    if not regions_root.exists():
        print(f"⚠️  目录不存在: {regions_root}")
        return

    regions_json_files = sorted(glob.glob(str(regions_root / "*" / "regions.json")))
    if not regions_json_files:
        print(f"⚠️  未找到 regions.json: {regions_root}")
        return

    print(f"\n{'=' * 68}")
    print(f"Step 4: Build Metadata - {split} split")
    print(f"Samples: {len(regions_json_files)}")
    print(f"{'=' * 68}")

    metadata: List[Dict[str, Any]] = []
    print_lock = threading.Lock()
    results_lock = threading.Lock()

    def flush_results() -> None:
        """
        入参:
            无

        方法:
            将当前已收集的 metadata 列表写入磁盘。
            加锁是为了避免并发写文件时出现内容覆盖或写入竞争。

        出参:
            None
        """
        with results_lock:
            save_json(output_path, metadata)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_sample,
                i,
                len(regions_json_files),
                dataset_root,
                split,
                Path(path),
                client,
                model_name,
                print_lock,
            ): path
            for i, path in enumerate(regions_json_files, 1)
        }

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            with results_lock:
                result.pop("_step4_refined_regions", None)
                metadata.append(result)
            flush_results()

    metadata.sort(key=lambda item: str(item.get("image_name", "")))
    save_json(output_path, metadata)
    print(f"\nSaved: {output_path} ({len(metadata)} samples)")


def main() -> int:
    """
    入参:
        无（从命令行读取参数）

    方法:
        解析参数后，先对 llm/clip 不一致的局部 patch 做 VLM 修正，再汇总生成 metadata.json。

    出参:
        int: 0 表示成功结束
    """
    parser = argparse.ArgumentParser(description="Step 4: Refine llm with local patches and build metadata.json")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集根目录")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"],
                        help="数据集划分 (default: train)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API Key（云端 VLM 修正需要）")
    parser.add_argument("--model", type=str, default="qwen-vl-plus",
                        help="局部 patch 修正模型名称")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API base URL")
    parser.add_argument("--max_workers", type=int, default=1, help="并行数")
    parser.add_argument("--batch_size", type=int, default=10, help="保留兼容参数，不再使用")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("未检测到 DASHSCOPE_API_KEY，请先设置环境变量或传入 --api_key。")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        split_dir = Path(args.dataset_path) / split
        if not split_dir.is_dir():
            continue
        process_split(args.dataset_path, split, client, args.model, args.max_workers)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
