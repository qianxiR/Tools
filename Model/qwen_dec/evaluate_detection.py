"""
Qwen3-VL 管网缺陷目标检测评估（在留出 test 集上计算指标）。

做什么:
    加载 inference.py 中同一套模型、LoRA、prompt 与 bbox 解析逻辑，在 test 集逐图推理，
    计算多类别目标检测指标。

为什么:
    训练过程只用 train/val；test 集用于最终泛化评估。评估阶段复用推理端代码，避免
    prompt、坐标反归一化或 LoRA 加载方式与实际部署不一致。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from detection_metrics import (
    evaluate_results as calculate_detection_metrics,
    save_confusion_matrix_json,
    save_confusion_matrix_png,
)
from inference import (
    PROMPT_JSON,
    USER_PROMPT,
    apply_class_aware_nms,
    denormalize_box,
    load_model,
    parse_bbox_json,
    run_inference,
)
from model.prompt_loader import load_label_codes

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", r"F:/管网/数据_筛选3000"))
DEFAULT_TEST_JSONL = Path(os.environ.get(
    "TEST_JSONL",
    str(DATASET_ROOT / "pipe_qwenvl_test.jsonl"),
))
RESULTS_ROOT = Path(r"F:\管网\runs\pipe_defect_lora_v10_siglip2_base\test")
DEFAULT_OUTPUT = Path(os.environ.get(
    "PREDICTIONS_OUTPUT",
    str(RESULTS_ROOT / "test_predictions.jsonl"),
))
DEFAULT_VIS_DIR = Path(os.environ.get(
    "VIS_DIR",
    str(RESULTS_ROOT / "visualizations"),
))
LABEL_CODES = load_label_codes(PROMPT_JSON)

# 迭代解码：第 2+ 轮用此 prompt 追问遗漏目标（不嵌入已检框，保持与训练单轮分布最接近）。
ITERATIVE_FOLLOWUP_PROMPT = (
    "Re-examine the entire image very carefully, including edges, corners, dark areas, "
    "and regions near already-noticeable features. Focus on defects you may have missed: "
    "small, faint, or partially visible ones. Output any additional defects as <obj>[...]</obj>, "
    "or <obj>[]</obj> if you are confident nothing else remains. "
    "Do not repeat objects you already reported."
)
# 跨轮同类 NMS 去重阈值：与评估 IoU 同值（0.5），避免合并重复框人为抬高 F1。
ITERATIVE_MERGE_IOU = 0.5


def parse_args() -> argparse.Namespace:
    """入参: 命令行参数。
    方法: 解析 test JSONL、LoRA adapter、生成长度、IoU 阈值、预测输出路径和可视化目录。
    出参: argparse.Namespace，供 main 调度评估流程。"""
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL pipe-defect detection on test JSONL.")
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vis-dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--confusion-json", type=Path, default=None)
    parser.add_argument("--confusion-png", type=Path, default=None)
    parser.add_argument(
        "--nms-iou", type=float, default=None,
        help="optional class-aware NMS IoU threshold; omitted keeps all raw predictions",
    )
    parser.add_argument(
        "--iterative-rounds", type=int, default=1,
        help="iterative decoding rounds; 1 = single-shot (default, baseline F1=0.4268). "
             "Round 2+ appends a follow-up prompt asking for missed defects; "
             "stops early when a round outputs no new boxes. "
             "Cross-round predictions are merged with class-aware NMS@0.5.",
    )
    parser.add_argument(
        "--no-external", action="store_true",
        help="禁用 SigLIP2 视觉前缀注入，用纯原生 Qwen3-VL ViT token 推理（原生基线对照实验）。"
             "省略时按训练分布注入 256 个 SigLIP2 前缀（需 LoRA adapter 与 external_feature_adapter.pt 同装）。",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """入参: path JSONL 文件路径。
    方法: 逐行读取非空 JSON，保持原始样本顺序用于无置信度预测的 AP 排序。
    出参: 样本字典列表；文件不存在时由 Path.open 抛出明确错误。"""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def normalized_targets_to_pixels(targets: List[Dict[str, Any]], width: int, height: int) -> tuple[List[List[int]], List[str]]:
    """入参: targets=[{"bbox_2d":[...],"label":...}]，原图宽高。
    方法: 使用 inference.denormalize_box 将 0-1000 bbox 还原到像素 xyxy，过滤退化框；
          label 与框同序收集（无 label 回退 "object"），供可视化显示类别不丢。
    出参: (像素框列表 [[x1,y1,x2,y2], ...], 同序 label 列表)。"""
    boxes: List[List[int]] = []
    labels: List[str] = []
    for target in targets:
        box = target.get("bbox_2d")
        if isinstance(box, list) and len(box) == 4:
            px = denormalize_box(box, width, height)
            if px[2] > px[0] and px[3] > px[1]:
                boxes.append(px)
                labels.append(str(target.get("label", "object")))
    return boxes, labels


def ground_truth_boxes(record: Dict[str, Any], width: int, height: int) -> tuple[List[List[int]], List[str]]:
    """入参: JSONL 样本记录，原图宽高。
    方法: 解析 assistant 的 <obj> 包裹 JSON bbox，还原为原图像素坐标并同步收集 label。
    出参: (GT bbox 列表, 同序 GT label 列表)。"""
    answer = record["conversations"][1]["value"]
    targets = parse_bbox_json(answer)
    return normalized_targets_to_pixels(targets, width, height)


def box_iou(box_a: List[int], box_b: List[int]) -> float:
    """入参: box_a/box_b 像素 xyxy。
    方法: 计算交集面积除以并集面积；无交集或退化框返回 0。
    出参: IoU 浮点数，范围 [0,1]。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_image(
    pred_boxes: List[List[int]],
    gt_boxes: List[List[int]],
    pred_labels: List[str],
    gt_labels: List[str],
    iou_threshold: float,
) -> Dict[str, Any]:
    """入参: 单图预测框/类别、GT 框/类别和 IoU 阈值。
    方法: 仅在同类别 GT 中按预测顺序贪心匹配；每个 GT 至多匹配一次，IoU 达标才记为 TP。
    出参: {"tp","fp","fn","ious"}；同框异类预测计 FP，未匹配 GT 计 FN。"""
    matched_gt = set()
    true_positive = 0
    false_positive = 0
    matched_ious: List[float] = []
    for pred_index, pred in enumerate(pred_boxes):
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(gt_boxes):
            if index in matched_gt:
                continue
            if pred_labels[pred_index] != gt_labels[index]:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_iou >= iou_threshold and best_index >= 0:
            matched_gt.add(best_index)
            true_positive += 1
            matched_ious.append(best_iou)
        else:
            false_positive += 1
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": len(gt_boxes) - len(matched_gt),
        "ious": matched_ious,
    }


def precision_recall_f1(tp_count: int, fp_count: int, fn_count: int) -> Dict[str, float]:
    """入参: TP/FP/FN 计数。
    方法: 按目标检测常规定义计算 precision、recall、F1，分母为 0 时返回 0。
    出参: 指标字典。"""
    precision = tp_count / (tp_count + fp_count) if tp_count + fp_count > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if tp_count + fn_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def average_precision(results: List[Dict[str, Any]], iou_threshold: float) -> float:
    """入参: 逐图预测结果，IoU 阈值。
    方法: 将无置信度预测按样本顺序展开，仅匹配同类别 GT，逐个累积 TP/FP 并使用 COCO 风格 precision envelope 积分。
    出参: AP 浮点数；无 GT 时返回 0。"""
    total_gt = sum(len(item["gt_boxes"]) for item in results)
    if total_gt == 0:
        return 0.0

    gt_used = {item["id"]: set() for item in results}
    tp_flags: List[int] = []
    fp_flags: List[int] = []
    for item in results:
        for pred_index, pred in enumerate(item["pred_boxes"]):
            best_iou = 0.0
            best_index = -1
            for index, gt in enumerate(item["gt_boxes"]):
                if index in gt_used[item["id"]]:
                    continue
                if item["pred_labels"][pred_index] != item["gt_labels"][index]:
                    continue
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_iou >= iou_threshold and best_index >= 0:
                gt_used[item["id"]].add(best_index)
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

    if not tp_flags:
        return 0.0

    precisions: List[float] = []
    recalls: List[float] = []
    tp_sum = 0
    fp_sum = 0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        tp_sum += tp_flag
        fp_sum += fp_flag
        precisions.append(tp_sum / (tp_sum + fp_sum))
        recalls.append(tp_sum / total_gt)

    envelope = precisions[:]
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])

    ap = 0.0
    previous_recall = 0.0
    for recall, precision in zip(recalls, envelope):
        if recall > previous_recall:
            ap += (recall - previous_recall) * precision
            previous_recall = recall
    return ap


def evaluate_results(
    results: List[Dict[str, Any]],
    iou_threshold: float,
    class_names: List[str] | None = None,
) -> Dict[str, Any]:
    """入参: 逐图预测、主 IoU 阈值和可选类别顺序。
    方法: 委托 detection_metrics 统一计算全局、逐类指标和混淆矩阵，避免统计口径分叉。
    出参: 可 JSON 序列化的完整测试指标。"""
    return calculate_detection_metrics(results, iou_threshold, class_names)


def write_predictions(path: Path, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    """入参: 输出路径、逐图预测结果、汇总指标。
    方法: 创建父目录后写 JSONL；第一行为 metrics，后续为每图 raw output 和 bbox。
    出参: 无；写出可复查的评估明细。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metrics", "metrics": metrics}, ensure_ascii=False) + "\n")
        for item in results:
            f.write(json.dumps({"type": "prediction", **item}, ensure_ascii=False) + "\n")


def draw_labeled_box(draw: ImageDraw.ImageDraw, box: List[int], label: str, color: tuple[int, int, int], stroke: int) -> None:
    """入参: PIL draw 对象、像素 xyxy 框、标签文本、RGB 颜色、线宽。
    方法: 绘制矩形框与左上角标签；标签背景使用同色填充，保证在遥感影像上可读。
    出参: 无；直接修改传入图像的绘制层。"""
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)
    text_x = x1
    text_y = max(0, y1 - 18)
    draw.rectangle([text_x, text_y, text_x + max(48, len(label) * 8), text_y + 16], fill=color)
    draw.text((text_x + 3, text_y + 2), label, fill=(255, 255, 255))


def save_visualizations(results: List[Dict[str, Any]], vis_dir: Path) -> None:
    """入参: 逐图预测结果列表（含 gt_labels/pred_labels）、可视化输出目录。
    方法: 对每张原图绘制绿色 GT 框与红色预测框，框标签为 GT:<类别>/Pred:<类别>，
          类别取自同序 label 列表（缺失回退 GT/Pred），保证可视化不丢类别；按样本 id 保存 PNG。
    出参: 无；在 vis_dir 下生成逐图检测可视化图片。"""
    vis_dir.mkdir(parents=True, exist_ok=True)
    for item in results:
        with Image.open(item["image"]) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        stroke = max(2, min(canvas.size) // 320)
        for box, label in zip(item["gt_boxes"], item.get("gt_labels", [])):
            draw_labeled_box(draw, box, f"GT:{label}" if label else "GT", (0, 180, 0), stroke)
        for box, label in zip(item["pred_boxes"], item.get("pred_labels", [])):
            draw_labeled_box(draw, box, f"Pred:{label}" if label else "Pred", (220, 0, 0), stroke)
        out_path = vis_dir / f"{item['id']}.png"
        canvas.save(out_path)


def run_iterative_inference(
    model, processor, image, max_rounds: int, max_new_tokens: int,
    inject_external: bool = True,
) -> tuple[str, List[dict]]:
    """入参: model/processor/image（路径或 PIL），max_rounds 迭代轮数，max_new_tokens 单轮生成上限，
          inject_external 是否注入 SigLIP2 前缀（透传给 run_inference，原生基线对照时传 False）。
    方法: 第 1 轮用训练同源 USER_PROMPT，第 2+ 轮用 ITERATIVE_FOLLOWUP_PROMPT 纯文本追问（不告知已检框，
          保持与训练单轮分布最接近）。每轮独立调用 run_inference（inject_external=True 时 SigLIP2 前缀在
          每次 prefill 重新注入，多次调用安全）。任一轮输出空数组即提前终止。所有轮预测拼起来后跨轮同类
          NMS@0.5 去重（与评估 IoU 同阈值，不人为抬高 F1）。
    出参: (合并去重后的 raw_output 跨轮拼接文本, 去重后预测 dict 列表)。"""
    all_targets: List[dict] = []
    raw_outputs: List[str] = []
    prompt = USER_PROMPT
    for round_idx in range(1, max_rounds + 1):
        raw, _, _ = run_inference(model, processor, image, prompt, max_new_tokens, inject_external=inject_external)
        raw_outputs.append(f"[round {round_idx}] {raw}")
        targets = parse_bbox_json(raw)
        all_targets.extend(targets)
        if not targets:  # 输出空数组 → 提前终止
            break
        prompt = ITERATIVE_FOLLOWUP_PROMPT  # 第 2+ 轮换追问 prompt
    merged = apply_class_aware_nms(all_targets, ITERATIVE_MERGE_IOU) if all_targets else []
    return " || ".join(raw_outputs), merged


def run_dataset_evaluation(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """入参: argparse.Namespace 评估参数。
    方法: 加载模型后遍历 test JSONL，使用训练 prompt 推理并收集 GT/pred bbox，再计算指标。
    出参: (逐图结果列表, 汇总指标字典)。"""
    records = read_jsonl(args.test_jsonl)
    model, processor = load_model(args.adapter)
    inject_external = not args.no_external
    if args.no_external:
        print("[startup] --no-external: 纯原生 Qwen3-VL ViT 推理，禁用 SigLIP2 前缀注入")
    results: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        image_path = record["image"]
        with Image.open(image_path) as image:
            width, height = image.size
        if args.iterative_rounds > 1:
            raw_output, pred_targets = run_iterative_inference(
                model, processor, image_path, args.iterative_rounds, args.max_new_tokens,
                inject_external=inject_external,
            )
        else:
            raw_output, _, _ = run_inference(
                model,
                processor,
                image_path,
                USER_PROMPT,
                args.max_new_tokens,
                inject_external=inject_external,
            )
            pred_targets = parse_bbox_json(raw_output)
        if args.nms_iou is not None:
            pred_targets = apply_class_aware_nms(pred_targets, args.nms_iou)
        pred_boxes, pred_labels = normalized_targets_to_pixels(pred_targets, width, height)
        gt_boxes, gt_labels = ground_truth_boxes(record, width, height)
        result = {
            "id": record.get("id", Path(image_path).stem),
            "image": image_path,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "pred_boxes": pred_boxes,
            "pred_labels": pred_labels,
            "raw_output": raw_output,
        }
        results.append(result)
        print(f"[eval] {index}/{len(records)} id={result['id']} gt={len(result['gt_boxes'])} pred={len(result['pred_boxes'])}")
    metrics = evaluate_results(results, args.iou_threshold, LABEL_CODES)
    # 标注本次评估的推理模式，便于多基线 jsonl 对照（True=SigLIP2 前缀注入, False=纯原生 Qwen）。
    metrics["external_prefix"] = not args.no_external
    metrics["adapter"] = args.adapter or "none"
    metrics["iterative_rounds"] = args.iterative_rounds
    return results, metrics


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """入参: 含全局、逐类及混淆矩阵的完整指标。
    方法: 先打印全局指标和逐类定宽表，再按 GT 行、预测列打印原始混淆计数。
    出参: 无；终端形成可直接核对的测试报告。"""
    nested_keys = {"per_class", "confusion_matrix"}
    for key, value in metrics.items():
        if key not in nested_keys:
            print(f"{key}: {value}")
    print("-" * 116)
    print(
        f"{'class':<8}{'GT':>6}{'Pred':>7}{'TP':>6}{'FP':>6}{'FN':>6}"
        f"{'Precision':>12}{'Recall':>10}{'F1':>10}{'AP50':>10}{'mAP50-95':>12}"
    )
    for class_name, values in metrics["per_class"].items():
        print(
            f"{class_name:<8}{values['gt_boxes']:>6}{values['pred_boxes']:>7}"
            f"{values['tp']:>6}{values['fp']:>6}{values['fn']:>6}"
            f"{values['precision']:>12.4f}{values['recall']:>10.4f}{values['f1']:>10.4f}"
            f"{values['ap50']:>10.4f}{values['map50_95']:>12.4f}"
        )
    confusion = metrics["confusion_matrix"]
    labels = confusion["labels"]
    print("-" * 116)
    print("confusion matrix: rows=ground truth, columns=prediction")
    print("GT\\Pred".ljust(12) + "".join(f"{label:>12}" for label in labels))
    for label, row in zip(labels, confusion["matrix"]):
        print(f"{label:<12}" + "".join(f"{value:>12}" for value in row))


def main() -> None:
    """入参: 无。
    方法: 解析参数并执行 test 评估，保存预测、逐图框、逐类指标及 JSON/PNG 混淆矩阵。
    出参: 无；终端与文件同时输出完整类别感知结果。"""
    args = parse_args()
    results, metrics = run_dataset_evaluation(args)
    write_predictions(args.predictions_output, results, metrics)
    save_visualizations(results, args.vis_dir)
    confusion_json = args.confusion_json or args.predictions_output.with_name("confusion_matrix.json")
    confusion_png = args.confusion_png or args.predictions_output.with_name("confusion_matrix.png")
    save_confusion_matrix_json(confusion_json, metrics["confusion_matrix"])
    save_confusion_matrix_png(confusion_png, metrics["confusion_matrix"])
    print("=" * 60)
    print_evaluation_summary(metrics)
    print(f"predictions: {args.predictions_output}")
    print(f"visualizations: {args.vis_dir}")
    print(f"confusion matrix json: {confusion_json}")
    print(f"confusion matrix png: {confusion_png}")
    print("=" * 60)


if __name__ == "__main__":
    main()
