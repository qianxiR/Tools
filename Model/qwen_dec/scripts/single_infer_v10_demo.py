# -*- coding: utf-8 -*-
"""v10 best_adapter 单图 CPU 推理演示（训练占满 GPU 显存时使用，不打断训练）。"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import load_model, run_inference, parse_bbox_json, draw_boxes, SERVE_DIR
from evaluate_detection import normalized_targets_to_pixels, ground_truth_boxes

ADAPTER = r"F:\管网\runs\pipe_defect_lora_v10_siglip2_base_ds500\train\best_adapter"
TEST_JSONL = r"F:\管网\数据_筛选500\pipe_qwenvl_test.jsonl"
MAX_NEW_TOKENS = 180


def main() -> None:
    """入参: 无。方法: 加载 v10 best adapter（强制 CPU，训练占满 GPU 时不打断），取 test 首图推理并对照 GT。
    出参: 打印 raw 输出与解析框；画框图存 SERVE_DIR。"""
    from transformers import AutoProcessor
    from peft import PeftModel
    from model.qwen_feature_injection import FeatureInjectedQwen3VLForConditionalGeneration

    base = FeatureInjectedQwen3VLForConditionalGeneration.from_pretrained(
        r"F:/pre/Qwen3-VL-4B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cpu"},
    )
    base.bind_external_extractor(r"F:/pre/siglip2-base-patch16-224", torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER.replace("\\", "/"))
    # peft 0.18 from_pretrained 会把整模型移到默认 cuda（训练占用时危险），显式拉回 CPU
    model = model.to("cpu")
    feature_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    feature_model.load_external_adapter(ADAPTER, strict=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(ADAPTER.replace("\\", "/"))

    record = json.loads(Path(TEST_JSONL).read_text(encoding="utf-8").splitlines()[0])
    image_path = record["image"]
    with Image.open(image_path) as im:
        w, h = im.size

    raw_text, _, _ = run_inference(model, processor, image_path, max_new_tokens=MAX_NEW_TOKENS)
    preds = parse_bbox_json(raw_text)
    pred_boxes, pred_labels = normalized_targets_to_pixels(preds, w, h)
    gt_boxes, gt_labels = ground_truth_boxes(record, w, h)

    print(f"image: {image_path}")
    print(f"GT  ({len(gt_boxes)}): {list(zip(gt_labels, gt_boxes))}")
    print(f"Pred({len(pred_boxes)}): {list(zip(pred_labels, pred_boxes))}")
    print(f"raw output:\n{raw_text}")

    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SERVE_DIR / "v10_single_infer.png"
    from model.location_tokens import decode_bbox
    from inference import normalize_targets
    targets = normalize_targets(preds)
    for t in targets:
        box = decode_bbox(t.get("bbox_2d"), allow_legacy_numeric=True)
        if box is not None:
            t["bbox_2d"] = box
    draw_boxes(Image.open(image_path), targets).save(out_path)
    print(f"visualization saved: {out_path}")


if __name__ == "__main__":
    main()