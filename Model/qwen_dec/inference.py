"""
Qwen3-VL 管网缺陷目标检测推理（加载 LoRA adapter，输出 JSON bbox 还原像素并画框）。

做什么:
    以与训练一致的 bf16 加载 Qwen3-VL 基座（叠加 LoRA adapter + external_feature_adapter.pt v10），
    图文混合构造输入（Qwen 原生视觉 token + 256 个 SigLIP2 LoRA 前缀 token），
    解析 0-1000 裸整数 JSON bbox_2d，按原图尺寸还原像素坐标并画框。

为什么:
    - Qwen3-VL processor 不做坐标反归一化，需自行 json.loads 后按 bbox_2d[i]/1000*W|H 还原。
    - 推理 prompt 必须与训练完全一致，保证分布对齐。
    - 输出图便于目检模型是否真正学会缺陷定位。

环境:
    conda activate llamafactory
    运行示例:
        python inference.py --adapter F:\管网\runs\pipe_defect_lora_v10_siglip2_base\train\best_adapter
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor

from model.qwen_feature_injection import EXTERNAL_PREFIX_LENGTH, FeatureInjectedQwen3VLForConditionalGeneration
from model.location_tokens import decode_bbox  # 裸整数/location token 兼容解析；add_location_tokens 已不在裸整数链使用
from model.prompt_loader import load_full_prompt

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    r"F:/pre/Qwen3-VL-4B-Instruct",
)
COORD_SCALE = 1000
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 512 * 28 * 28
LABEL_NAME = "object"
PRED_COLOR = (220, 0, 0)      # 预测框统一红色，与 evaluate_detection.py 的 Pred 颜色一致
SIGLIP_MODEL_PATH = Path(os.environ.get(
    "SIGLIP_MODEL_PATH",
    r"F:/pre/siglip2-base-patch16-224",
))
# Gradio 展示用的画框图临时目录（推理结果图落到这里，URL 回填到助手气泡）
SERVE_DIR = Path(r"F:/管网/runs/infer")
# 推理量化开关：QUANT=int4 时基座以 bitsandbytes NF4 加载（约 2.8GB 驻留），外部链保持 bf16。
QUANT = os.environ.get("QUANT", "").strip().lower()
PROMPT_JSON = os.environ.get("PROMPT_JSON", "")
INFER_PROMPT_JSON = Path(os.environ.get(
    "INFER_PROMPT_JSON",
    PROMPT_JSON or str(Path(__file__).resolve().parent / "data" / "pipe_defect_3000_prompt_infer_en.json"),
))
PROMPT_JSON = INFER_PROMPT_JSON

USER_PROMPT = load_full_prompt(PROMPT_JSON)


# ----------------------------------------------------------------------------
# JSON bbox 解析（容错）
# ----------------------------------------------------------------------------
def normalize_targets(data) -> List[dict]:
    """入参: json.loads 或正则抽取出的任意对象。
    方法: 仅保留 bbox_2d 结构，优先解析当前 0-1000 裸整数坐标；
          同时兼容历史 location token 输出，其他 point、混合类型和畸形数组全部丢弃。
    出参: 合法目标列表；无合法目标返回空列表。"""
    if isinstance(data, dict):
        box = decode_bbox(data.get("bbox_2d"), allow_legacy_numeric=True)
        if box is not None:
            return [{"bbox_2d": box, "label": data.get("label", LABEL_NAME)}]
        return []
    if isinstance(data, list):
        direct_box = decode_bbox(data, allow_legacy_numeric=True)
        if direct_box is not None:
            return [{"bbox_2d": direct_box, "label": LABEL_NAME}]
        targets: List[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            box = decode_bbox(item.get("bbox_2d"), allow_legacy_numeric=True)
            if box is not None:
                targets.append({"bbox_2d": box, "label": item.get("label", LABEL_NAME)})
        return targets
    return []


def parse_bbox_json(text: str) -> List[dict]:
    """入参: 模型输出文本。
    方法:
      1) 剥离 markdown 代码块围栏（```json ... ```）。
      2) 剥离 <obj></obj> 包裹（训练/转换器约定的输出协议），无条件去裸标签后走常规 JSON 解析。
      3) 先整体 json.loads；失败则用贪婪正则抽取最外层 [...] 子串（配对括号，
         覆盖换行缩进的多目标数组；非贪婪会在首个 ] 截断导致解析失败）。
      4) 仍失败则退化：抽取 {"bbox_2d":[...]} 对象与 <points ...> 点位标签。
    出参: 合法目标 dict 列表；解析失败返回空列表（视为无目标）。"""
    text = text.strip()
    # 剥离 ```json / ``` 围栏
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    # 剥离 <obj></obj> 包裹：直接去除裸标签（不引入分支），剩余文本交由后续 JSON 解析
    text = text.replace("<obj>", "").replace("</obj>", "").strip()

    # 整体解析
    try:
        data = json.loads(text)
        targets = normalize_targets(data)
        if targets:
            return targets
    except Exception:
        pass
    # 贪婪抽取最外层数组（最后一个 ] 闭合，覆盖多行缩进）
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            targets = normalize_targets(data)
            if targets:
                return targets
        except Exception:
            pass
    # 退化：逐个抽取对象（兼容散落文本）
    objs = []
    for om in re.finditer(r"\{[^{}]*\"bbox_2d\"[^{}]*\}", text, flags=re.DOTALL):
        try:
            objs.extend(normalize_targets(json.loads(om.group(0))))
        except Exception:
            continue
    return objs


def _normalized_box_iou(box_a: List[float], box_b: List[float]) -> float:
    """入参: 两个 0-1000 归一化 xyxy 框; 方法: 计算交并比; 出参: [0,1] IoU。"""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def apply_class_aware_nms(targets: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    """入参: 归一化 bbox 预测与同类 NMS IoU 阈值。
    方法: 按类别分组，在每组中优先保留面积较大的框并抑制 IoU 达阈值的重复框；无置信度输出时以面积作稳定排序依据。
    出参: 过滤后的预测，保持原始输出顺序。"""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")

    normalized = normalize_targets(targets)
    indexed = []
    for index, target in enumerate(normalized):
        box = target["bbox_2d"]
        if not all(isinstance(value, (int, float)) for value in box):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        indexed.append((index, target))

    kept_indices = set()
    labels = {str(target.get("label", LABEL_NAME)) for _, target in indexed}
    for label in labels:
        candidates = [
            (index, target)
            for index, target in indexed
            if str(target.get("label", LABEL_NAME)) == label
        ]
        candidates.sort(
            key=lambda item: (
                -(item[1]["bbox_2d"][2] - item[1]["bbox_2d"][0])
                * (item[1]["bbox_2d"][3] - item[1]["bbox_2d"][1]),
                item[0],
            )
        )
        group_kept = []
        for index, target in candidates:
            if all(_normalized_box_iou(target["bbox_2d"], kept["bbox_2d"]) < iou_threshold for kept in group_kept):
                group_kept.append(target)
                kept_indices.add(index)

    return [target for index, target in indexed if index in kept_indices]


# ----------------------------------------------------------------------------
# 坐标还原：0-1000 → 原图像素
# ----------------------------------------------------------------------------
def denormalize_box(box: List[int], width: int, height: int) -> List[int]:
    """入参: box=[x1,y1,x2,y2] (0-1000)，原图宽高。
    方法: 按 cookbook 约定，x 除以 1000 乘宽、y 除以 1000 乘高，clamp 到图界。
    出参: [x1_px, y1_px, x2_px, y2_px] 整数。"""
    x1 = max(0, min(width, int(round(box[0] / COORD_SCALE * width))))
    y1 = max(0, min(height, int(round(box[1] / COORD_SCALE * height))))
    x2 = max(0, min(width, int(round(box[2] / COORD_SCALE * width))))
    y2 = max(0, min(height, int(round(box[3] / COORD_SCALE * height))))
    return [x1, y1, x2, y2]


# ----------------------------------------------------------------------------
# 画框
# ----------------------------------------------------------------------------
def draw_boxes(image: Image.Image, targets: List[dict]) -> Image.Image:
    """入参: PIL image, targets=[{"bbox_2d":[...],"label":...}] (0-1000 坐标)。
    方法: 仅可视化 bbox_2d 框——按四角还原像素画红框，框上方叠加红底白字类别标签
          （与评估端 save_visualizations 样式一致，确保类别在复杂背景下清晰可读）。
    出参: 标注后的 PIL image（不修改原图）。"""
    out = image.copy().convert("RGB")
    w, h = out.size
    draw = ImageDraw.Draw(out)
    stroke = max(2, min(w, h) // 400)
    try:
        font = ImageFont.truetype("arial.ttf", max(16, min(w, h) // 40))
    except Exception:
        font = ImageFont.load_default()
    for t in normalize_targets(targets):
        box = t.get("bbox_2d")
        label = str(t.get("label", LABEL_NAME))
        if box and len(box) == 4:
            x1, y1, x2, y2 = denormalize_box(box, w, h)
            draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=stroke)
            tag_h = max(18, min(w, h) // 50)
            tag_w = max(48, len(label) * 10)
            tag_y = max(0, y1 - tag_h)
            draw.rectangle([x1, tag_y, x1 + tag_w, tag_y + tag_h], fill=PRED_COLOR)
            draw.text((x1 + 3, tag_y + 2), label, fill=(255, 255, 255), font=font)
    return out


# ----------------------------------------------------------------------------
# 推理主函数
# ----------------------------------------------------------------------------
def run_inference(
    model, processor: AutoProcessor, image, prompt: str = USER_PROMPT,
    max_new_tokens: int = 512, inject_external: bool = True,
) -> tuple:
    """入参:
      - model, processor
      - image: 图像路径(str) 或 PIL.Image（Gradio 上传直接给 PIL，避免落盘）
      - prompt: 用户提问文本（默认 USER_PROMPT，与训练分布对齐）
      - max_new_tokens: 生成上限
      - inject_external: 是否注入 SigLIP2 视觉前缀（默认 True，与训练分布对齐）；
        False 时不预留前缀槽位、不编码 SigLIP2，走纯原生 Qwen3-VL ViT 路径（原生基线对照实验）。
    方法: 构造图文输入；inject_external=True 时额外预留 16×16 个 SigLIP2 视觉前缀槽位；
          统一调用 model.generate 贪心解码（do_sample=False 保证可复现，不干预 logits——
          解码策略与训练分布完全一致，输出格式由模型自身保证）。
    出参: (raw_text 模型原始输出, width, height)。"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_out = process_vision_info([messages])
    images, videos = vision_out[0], vision_out[1]
    inputs = processor(text=[text], images=images, videos=videos,
                       padding=True, return_tensors="pt").to(model.device)

    # 原生基线（inject_external=False）不预留前缀槽位、不编码 SigLIP2，直接用 Qwen 原生视觉 token。
    if inject_external:
        prefix_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        prefix_ids = torch.full(
            (inputs["input_ids"].shape[0], EXTERNAL_PREFIX_LENGTH),
            prefix_token_id,
            device=inputs["input_ids"].device,
            dtype=inputs["input_ids"].dtype,
        )
        inputs["input_ids"] = torch.cat((prefix_ids, inputs["input_ids"]), dim=1)
        inputs["attention_mask"] = torch.cat((torch.ones_like(prefix_ids), inputs["attention_mask"]), dim=1)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen_kwargs = {}
        if inject_external:
            feature_model = model.get_base_model() if hasattr(model, "get_base_model") else model
            gen_kwargs["external_hidden"] = feature_model.encode_external_images([image])
        # do_sample=False 走贪心解码（检测任务需可复现）。
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **gen_kwargs,
        )
    # 仅取新生成的 token（去掉输入 prompt）
    gen_ids = out_ids[:, prompt_length:]
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # 取宽高：PIL 直接读 size，路径则 Image.open
    if isinstance(image, Image.Image):
        w, h = image.size
    else:
        with Image.open(image) as im:
            w, h = im.size
    return raw_text, w, h


# ----------------------------------------------------------------------------
# 模型加载（启动一次）
# ----------------------------------------------------------------------------
def load_model(adapter: Optional[str] = None):
    """入参: adapter 可选 LoRA 目录。
    方法: 以与训练一致的 bf16 加载 Qwen3-VL 基座并绑定 SigLIP2 LoRA 外部链；adapter 存在(含
          adapter_config.json)时叠加 LoRA 并以 strict 模式恢复 external_feature_adapter.pt
          v10（SigLIP2 LoRA + 纯视觉融合 + projector，缺失即抛错，保证特征注入与训练同分布），
          否则仅基座。路径归一化为正斜杠绝对路径，规避 peft 把反斜杠路径误判为 HF repo id。
    出参: (model, processor)。"""
    adapter_path = str(Path(adapter).resolve()).replace("\\", "/") if adapter else ""
    adapter_config = Path(adapter_path) / "adapter_config.json" if adapter_path else None
    use_lora = adapter_config is not None and adapter_config.is_file()
    print("[startup] Qwen3-VL pipe-defect inference")
    print(f"[startup] base model: {MODEL_PATH}")
    print(f"[startup] external feature: siglip2-base LoRA + MultiScaleFusionDecoder -> 16x16 prefix ({SIGLIP_MODEL_PATH})")
    print(f"[startup] prompt file: {PROMPT_JSON}")
    print(f"[startup] prompt text: {USER_PROMPT}")
    if adapter_path:
        print(f"[startup] adapter arg: {adapter_path}")
    print(f"[startup] adapter loaded: {use_lora}")
    if adapter_path and not use_lora:
        print(f"[WARN] adapter_config.json not found, fallback to base-only: {adapter_path}")
    processor_source = adapter_path if use_lora and (Path(adapter_path) / "tokenizer_config.json").is_file() else MODEL_PATH
    processor = AutoProcessor.from_pretrained(
        processor_source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    print("[load] base model (1-3 min, no output meanwhile)...")
    # 基座统一 bf16 加载：与训练精度严格一致，避免 4bit 量化使权重分布偏移导致检测失效。
    # 词表保持基座原生（裸整数坐标沿用原生数字 token，不注册 location token、不 resize），与训练链一致。
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    ) if QUANT == "int4" else None
    model = FeatureInjectedQwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
        quantization_config=quant_config,
    )
    model.bind_external_extractor(SIGLIP_MODEL_PATH, torch.bfloat16)
    print(f"[load] base precision: bf16 (training-aligned); quantization: {QUANT or 'none'}")
    if use_lora:
        from peft import PeftModel

        print(f"[load] LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        feature_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        # strict=True：外部 adapter 缺失或 v10 元数据不匹配时立即报错，避免训练/推理主干错配。
        external_path = feature_model.load_external_adapter(adapter_path, strict=True)
        metadata = getattr(feature_model, "loaded_external_adapter_metadata", None) or {}
        print(f"[load] external feature adapter: {external_path}")
        if metadata:
            print(
                "[load] external adapter metadata: "
                f"format=v{metadata.get('format_version')}, "
                f"prefix_grid={metadata.get('prefix_grid_size')}, "
                f"backbone={metadata.get('backbone_architecture')}, "
                f"fusion={metadata.get('fusion')}"
            )
    model.eval()
    print(f"[load] vocab_size: {len(processor.tokenizer)} (base native, no location tokens)")
    print("[load] ready.")
    return model, processor


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# TIFF → PNG 自动转换（遥感数据为 .tif，统一转 PNG 便于预览/保存/模型稳定读取）
# ----------------------------------------------------------------------------
def ensure_png(src_path: str) -> Tuple[str, Image.Image]:
    """入参: src_path 上传的图像路径（可能 .tif/.tiff/.png/.jpg...）。
    方法:
      - 若扩展名为 .tif/.tiff：用 PIL 打开转 RGB，存 PNG 到 SERVE_DIR，返回新路径 + PIL。
      - 否则：原样返回路径 + PIL（不转）。
    为什么: .tif 在浏览器预览与跨工具流转不通用，转 PNG 后预览/推理/结果展示统一。
    出参: (display_path, pil_img)，display_path 用于气泡展示与推理。"""
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    lower = src_path.lower()
    if lower.endswith((".tif", ".tiff")):
        pil = Image.open(src_path).convert("RGB")
        stamp = f"{int(time.time() * 1000)}"
        png_path = SERVE_DIR / f"upload_{stamp}.png"
        pil.save(png_path)
        return str(png_path), pil
    return src_path, Image.open(src_path).convert("RGB")


# ----------------------------------------------------------------------------
# 单轮推理 → 助手消息列表（Gradio messages 格式，供 gr.Chatbot 渲染）
# ----------------------------------------------------------------------------
def infer_once(model, processor, pil_img: Image.Image, prompt: str,
               max_new_tokens: int = 512, nms_iou: Optional[float] = None) -> List[dict]:
    """入参: model, processor, PIL 图, 用户提问, 生成 token 上限及可选 NMS IoU 阈值。
    方法: run_inference 取原始文本 → parse_bbox_json 解析；仅在显式指定 nms_iou 时执行 class-aware NMS → draw_boxes 画框图存 SERVE_DIR
          → 组装助手消息列表（画框图消息 + 文本摘要消息）。
    出参: Gradio messages 列表；每条 content 保持单一类型，避免文件 alt_text 被误解析。"""
    t0 = time.time()
    raw_text, w, h = run_inference(model, processor, pil_img, prompt, max_new_tokens)
    targets = parse_bbox_json(raw_text)
    if nms_iou is not None:
        targets = apply_class_aware_nms(targets, nms_iou)
    elapsed = time.time() - t0

    messages: List[dict] = []
    if targets:
        drawn = draw_boxes(pil_img, targets)
        SERVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = f"{int(time.time() * 1000)}"
        out_path = SERVE_DIR / f"result_{stamp}.png"
        drawn.save(out_path)
        # Gradio messages 模式的文件内容使用 {"path": ..., "alt_text": "..."}。
        messages.append({
            "role": "assistant",
            "content": {"path": str(out_path), "alt_text": "Detection result image"},
        })

    # 文本摘要：像素坐标 + 原始输出
    lines = [f"Detected {len(targets)} target(s) in {elapsed:.1f}s (image {w}x{h})."]
    for i, t in enumerate(normalize_targets(targets)):
        if "bbox_2d" in t and len(t["bbox_2d"]) == 4:
            px = denormalize_box(t["bbox_2d"], w, h)
            lines.append(f"  [{i}] label={t.get('label')} norm={t['bbox_2d']} px={px}")
    lines.append("")
    lines.append(f"raw output:\n{raw_text}")
    messages.append({"role": "assistant", "content": "\n".join(lines)})
    return messages


# ----------------------------------------------------------------------------
# Gradio 多模态聊天界面
# ----------------------------------------------------------------------------
def main():
    """入参: 命令行参数 [--adapter <dir>] [--port 7860] [--share] [--max-new-tokens N]。
    方法: 加载模型与 processor → 构建 Gradio Blocks 多模态聊天页 → 绑定提交/清空事件。
    出参: 启动本地 Gradio 服务；函数自身无显式返回值。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public tunnel link")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--nms-iou", type=float, default=None,
        help="optional class-aware NMS IoU threshold; omitted keeps all raw predictions",
    )
    args = ap.parse_args()

    model, processor = load_model(args.adapter)
    max_tokens = args.max_new_tokens
    nms_iou = args.nms_iou

    import gradio as gr

    # 聊天回调：MultimodalTextbox 的 message 形如 {"text":..., "files":[path]}
    def respond(message, history):
        """入参: message={text,files}, history=过往消息列表。
        方法: 取用户上传的首张图，模型固定使用训练同款 USER_PROMPT → infer_once →
              按 Gradio content 单类型约束追加 user/assistant 消息到 history（多轮叠加）。
        出参: (新 history, 清空输入框)。"""
        display_text = (message.get("text") or "").strip()
        files = message.get("files") or []
        if not files:
            history.append({"role": "assistant", "content": "Please upload an image first."})
            return history, {"text": "", "files": []}
        # tif 自动转 png，统一预览/推理路径
        display_path, pil_img = ensure_png(files[0])
        # user 气泡拆成单类型消息，匹配 Gradio 5.50 messages 后处理约束。
        history.append({
            "role": "user",
            "content": {"path": display_path, "alt_text": "Uploaded image"},
        })
        history.append({"role": "user", "content": display_text or USER_PROMPT})
        # assistant 气泡：画框图消息 + 坐标/原始输出文本消息。
        history.extend(infer_once(model, processor, pil_img, USER_PROMPT, max_tokens, nms_iou))
        return history, {"text": "", "files": []}

    def clear_history():
        """入参: 无。
        方法: 返回空对话与空 MultimodalTextbox 值。
        出参: 空 history + 空输入框（清空对话）。"""
        return [], {"text": "", "files": []}

    with gr.Blocks(title="Qwen3-VL Pipe-Defect Detect") as demo:
        gr.Markdown("# Qwen3-VL 管网缺陷检测\n"
                    "上传管道内壁 CCTV 图像，模型输出 JSON bbox 并画框。"
                    "Multi-turn history is kept in the chat.")
        chatbot = gr.Chatbot(type="messages", height=560,
                             allow_tags=True)
        with gr.Row():
            txt = gr.MultimodalTextbox(
                interactive=True, file_types=["image"],
                placeholder="Upload image (click 📎) and type your question, or leave blank for default prompt...",
                scale=9,
            )
            clear_btn = gr.Button("Clear", scale=1)
        txt.submit(respond, [txt, chatbot], [chatbot, txt])
        clear_btn.click(clear_history, None, [chatbot, txt])

    demo.launch(server_port=args.port, share=args.share,
                allowed_paths=[str(SERVE_DIR.resolve())])


if __name__ == "__main__":
    main()