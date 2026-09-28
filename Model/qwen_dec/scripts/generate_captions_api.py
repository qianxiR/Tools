"""
云端 API 为管道缺陷图生成专业描述（gpt-5.6-luna，OpenAI Response 格式）。

目的:
    一次性为所有训练/val/test 图生成"专业描述"文本，存到 JSONL 供后续训练/推理使用。
    训练时把描述作为额外的文本输入注入 Qwen decoder，与 SAM2 视觉特征拼接。

设计原则:
    1. 只看图像，不读 GT bbox（避免信息泄露，让模型描述纯视觉内容）。
    2. 描述聚焦「缺陷类型、大致位置、外观」，禁止精确坐标。
    3. 支持先小批验证（--limit N）再全量生成（不带 --limit）。
    4. 断点续跑：已生成的图跳过，避免重复消耗 API。

输出:
    data/captions.jsonl，每行：{"id": 图像ID, "image": 路径, "caption": 描述文本}

用法:
    # 先验证 5 张
    python scripts/generate_captions_api.py --limit 5
    # 全量生成（489 张）
    python scripts/generate_captions_api.py
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
from PIL import Image

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
API_BASE = "https://newapi.dragon3api.com/v1"
API_KEY = os.environ.get("CAPTION_API_KEY", "")
MODEL_NAME = "gpt-5.6-luna"

DATASET_ROOT = Path(r"F:/管网/数据_筛选500")
ALL_JSONL = DATASET_ROOT / "pipe_qwenvl.jsonl"
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "captions.jsonl"

CAPTION_PROMPT = (
    "Describe the visual appearance of this pipe inspection image objectively. "
    "Focus ONLY on what is directly visible: overall scene (pipe shape, lighting, water level), "
    "surface materials and textures (concrete color, roughness, wetness), "
    "color patches and their location (e.g. 'pale crusty deposit covering the bottom third', "
    "'dark vertical line on the right wall', 'yellow-brown stained area upper-left'), "
    "and any visible structural irregularities (cracks, deformations, holes, protrusions) "
    "described by their physical appearance, NOT by diagnostic category names. "
    "Do NOT use terms like 'defect', 'corrosion', 'crack', 'fracture', 'scale', 'root', 'obstruction', "
    "or any of the standard pipe-defect category names. "
    "Describe shapes, colors, and positions only. "
    "If the image is uniform with nothing notable, say so. "
    "Keep the description concise (under 80 words)."
)

MAX_RETRIES = 3
RETRY_BACKOFF = 5  # 秒


# ----------------------------------------------------------------------------
# 图像编码
# ----------------------------------------------------------------------------
def encode_image_b64(image_path: str) -> str:
    """入参: 图像路径 | 方法: 读取并 base64 编码（自动转 RGB，缩放长边到 1024 省流量） | 出参: base64 字符串"""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        # 长边缩到 1024，保持宽高比，省 API 流量
        w, h = im.size
        if max(w, h) > 1024:
            scale = 1024 / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        import io
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------------
# API 调用
# ----------------------------------------------------------------------------
def call_api(image_b64: str) -> str:
    """入参: base64 图像 | 方法: POST 到 gpt-5.6-luna，OpenAI Response 格式 | 出参: 描述文本"""
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_with_retry(image_path: str) -> str:
    """入参: 图像路径 | 方法: 编码 + 重试调用 | 出参: 描述文本"""
    img_b64 = encode_image_b64(image_path)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return call_api(img_b64)
        except Exception as e:
            last_err = e
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {e}")
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_err}")


# ----------------------------------------------------------------------------
# 断点续跑
# ----------------------------------------------------------------------------
def load_existing_captions() -> Dict[str, str]:
    """入参: 无 | 方法: 读已生成的 captions.jsonl | 出参: {image_path: caption}"""
    if not OUTPUT_FILE.exists():
        return {}
    out = {}
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["image"]] = rec["caption"]
            except Exception:
                continue
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    """入参: --limit N（可选） | 方法: 读全量 JSONL，跳过已生成，调 API 生成描述 | 出参: None"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 张（验证用）")
    args = parser.parse_args()

    records = [json.loads(l) for l in ALL_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"[caption] {len(records)} images to process")

    existing = load_existing_captions()
    print(f"[caption] {len(existing)} already generated, will skip")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 追加模式：保留已有，新增的写到末尾
    with OUTPUT_FILE.open("a", encoding="utf-8") as fout:
        for i, rec in enumerate(records, 1):
            img_path = rec["image"]
            if img_path in existing:
                continue
            try:
                caption = call_with_retry(img_path)
            except Exception as e:
                print(f"[{i}/{len(records)}] FAILED id={rec['id']}: {e}")
                continue
            fout.write(json.dumps({"id": rec["id"], "image": img_path, "caption": caption}, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i}/{len(records)}] OK id={rec['id']}: {caption[:80]}...")

    print(f"[caption] done. output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
