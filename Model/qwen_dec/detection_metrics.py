"""Class-aware detection metrics and confusion-matrix output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont


BACKGROUND_LABEL = "background"


def box_iou(box_a: List[int], box_b: List[int]) -> float:
    """入参: box_a/box_b 为像素 xyxy 框。
    方法: 计算交集面积除以并集面积，退化框或无交集返回 0。
    出参: 范围 [0, 1] 的 IoU。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_image(
    pred_boxes: List[List[int]],
    gt_boxes: List[List[int]],
    pred_labels: List[str],
    gt_labels: List[str],
    iou_threshold: float,
) -> Dict[str, Any]:
    """入参: 单图预测框/类别、GT 框/类别和 IoU 阈值。
    方法: 仅在同类别且未占用的 GT 中贪心选择最大 IoU，达标才记为 TP。
    出参: tp/fp/fn/ious；同框异类预测计 FP，原 GT 计 FN。"""
    matched_gt: set[int] = set()
    matched_ious: List[float] = []
    true_positive = 0
    for pred_index, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt or pred_labels[pred_index] != gt_labels[gt_index]:
                continue
            current_iou = box_iou(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index
        if best_gt_index >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt_index)
            matched_ious.append(best_iou)
            true_positive += 1
    return {
        "tp": true_positive,
        "fp": len(pred_boxes) - true_positive,
        "fn": len(gt_boxes) - len(matched_gt),
        "ious": matched_ious,
    }


def precision_recall_f1(tp_count: int, fp_count: int, fn_count: int) -> Dict[str, float]:
    """入参: TP、FP、FN 计数。
    方法: 按目标检测定义计算 precision、recall、F1，空分母按 0 处理。
    出参: precision/recall/f1 字典。"""
    precision = tp_count / (tp_count + fp_count) if tp_count + fp_count else 0.0
    recall = tp_count / (tp_count + fn_count) if tp_count + fn_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def average_precision(
    results: List[Dict[str, Any]],
    iou_threshold: float,
    target_label: str | None = None,
) -> float:
    """入参: 逐图预测、IoU 阈值和可选目标类别。
    方法: 按保存顺序展开无置信度预测，仅匹配同类别 GT，使用 precision envelope 积分。
    出参: 全类别或指定类别 AP；该类别无 GT 时返回 0。"""
    total_gt = sum(
        1
        for item in results
        for label in item["gt_labels"]
        if target_label is None or label == target_label
    )
    if total_gt == 0:
        return 0.0

    used_gt = [set() for _ in results]
    tp_flags: List[int] = []
    fp_flags: List[int] = []
    for image_index, item in enumerate(results):
        for pred_index, pred_box in enumerate(item["pred_boxes"]):
            pred_label = item["pred_labels"][pred_index]
            if target_label is not None and pred_label != target_label:
                continue
            best_iou = 0.0
            best_gt_index = -1
            for gt_index, gt_box in enumerate(item["gt_boxes"]):
                if gt_index in used_gt[image_index] or pred_label != item["gt_labels"][gt_index]:
                    continue
                current_iou = box_iou(pred_box, gt_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_index = gt_index
            is_match = best_gt_index >= 0 and best_iou >= iou_threshold
            if is_match:
                used_gt[image_index].add(best_gt_index)
            tp_flags.append(int(is_match))
            fp_flags.append(int(not is_match))

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
    ap_value = 0.0
    previous_recall = 0.0
    for recall, precision in zip(recalls, envelope):
        if recall > previous_recall:
            ap_value += (recall - previous_recall) * precision
            previous_recall = recall
    return ap_value


def per_class_metrics(
    results: List[Dict[str, Any]],
    class_names: Sequence[str],
    iou_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """入参: 逐图预测、稳定类别顺序和主 IoU 阈值。
    方法: 每类独立过滤框后执行类别感知匹配，并计算 AP50 与 AP50:95。
    出参: 以类别代码为键的计数及 P/R/F1/AP 指标。"""
    thresholds = [round(0.5 + 0.05 * index, 2) for index in range(10)]
    class_metrics: Dict[str, Dict[str, Any]] = {}
    for class_name in class_names:
        tp_count = 0
        fp_count = 0
        fn_count = 0
        gt_count = 0
        pred_count = 0
        for item in results:
            gt_boxes = [box for box, label in zip(item["gt_boxes"], item["gt_labels"]) if label == class_name]
            pred_boxes = [box for box, label in zip(item["pred_boxes"], item["pred_labels"]) if label == class_name]
            matched = match_image(
                pred_boxes,
                gt_boxes,
                [class_name] * len(pred_boxes),
                [class_name] * len(gt_boxes),
                iou_threshold,
            )
            tp_count += matched["tp"]
            fp_count += matched["fp"]
            fn_count += matched["fn"]
            gt_count += len(gt_boxes)
            pred_count += len(pred_boxes)
        prf = precision_recall_f1(tp_count, fp_count, fn_count)
        ap_values = [average_precision(results, threshold, class_name) for threshold in thresholds]
        class_metrics[class_name] = {
            "gt_boxes": gt_count,
            "pred_boxes": pred_count,
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
            **prf,
            "ap50": average_precision(results, 0.5, class_name),
            "map50_95": sum(ap_values) / len(ap_values),
        }
    return class_metrics


def build_confusion_matrix(
    results: List[Dict[str, Any]],
    class_names: Sequence[str],
    iou_threshold: float,
) -> Dict[str, Any]:
    """入参: 逐图预测、类别顺序和 IoU 阈值。
    方法: 忽略类别按 IoU 贪心配对；错分类进入 GT 行/Pred 列，漏检与冗余预测进入 background。
    出参: 含 labels、轴语义及原始计数矩阵的字典。"""
    labels = list(class_names) + [BACKGROUND_LABEL]
    label_to_index = {label: index for index, label in enumerate(labels)}
    background_index = len(labels) - 1
    matrix = [[0 for _ in labels] for _ in labels]
    for item in results:
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        candidates = sorted(
            (
                (box_iou(pred_box, gt_box), pred_index, gt_index)
                for pred_index, pred_box in enumerate(item["pred_boxes"])
                for gt_index, gt_box in enumerate(item["gt_boxes"])
            ),
            reverse=True,
        )
        for current_iou, pred_index, gt_index in candidates:
            if current_iou < iou_threshold:
                break
            if pred_index in matched_pred or gt_index in matched_gt:
                continue
            gt_class_index = label_to_index.get(item["gt_labels"][gt_index], background_index)
            pred_class_index = label_to_index.get(item["pred_labels"][pred_index], background_index)
            matrix[gt_class_index][pred_class_index] += 1
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
        for gt_index, gt_label in enumerate(item["gt_labels"]):
            if gt_index not in matched_gt:
                matrix[label_to_index.get(gt_label, background_index)][background_index] += 1
        for pred_index, pred_label in enumerate(item["pred_labels"]):
            if pred_index not in matched_pred:
                matrix[background_index][label_to_index.get(pred_label, background_index)] += 1
    return {
        "labels": labels,
        "rows": "ground_truth",
        "columns": "prediction",
        "iou_threshold": iou_threshold,
        "matrix": matrix,
    }


def evaluate_results(
    results: List[Dict[str, Any]],
    iou_threshold: float,
    class_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """入参: 逐图预测、主 IoU 阈值和可选类别顺序。
    方法: 汇总类别感知全局指标、逐类指标、AP 与检测混淆矩阵。
    出参: 可 JSON 序列化的完整测试指标。"""
    resolved_classes = list(dict.fromkeys(class_names or sorted({
        label for item in results for label in item["gt_labels"] + item["pred_labels"]
    })))
    tp_count = 0
    fp_count = 0
    fn_count = 0
    matched_ious: List[float] = []
    for item in results:
        matched = match_image(
            item["pred_boxes"], item["gt_boxes"], item["pred_labels"], item["gt_labels"], iou_threshold,
        )
        tp_count += matched["tp"]
        fp_count += matched["fp"]
        fn_count += matched["fn"]
        matched_ious.extend(matched["ious"])
    prf = precision_recall_f1(tp_count, fp_count, fn_count)
    thresholds = [round(0.5 + 0.05 * index, 2) for index in range(10)]
    ap_values = [average_precision(results, threshold) for threshold in thresholds]
    return {
        "images": len(results),
        "gt_boxes": sum(len(item["gt_boxes"]) for item in results),
        "pred_boxes": sum(len(item["pred_boxes"]) for item in results),
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "iou_threshold": iou_threshold,
        **prf,
        "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "ap50": average_precision(results, 0.5),
        "map50_95": sum(ap_values) / len(ap_values),
        "invalid_bbox_outputs": sum(
            1 for item in results if item["raw_output"].strip() != "[]" and not item["pred_boxes"]
        ),
        "per_class": per_class_metrics(results, resolved_classes, iou_threshold),
        "confusion_matrix": build_confusion_matrix(results, resolved_classes, iou_threshold),
    }


def save_confusion_matrix_json(path: Path, confusion: Dict[str, Any]) -> None:
    """入参: JSON 输出路径和混淆矩阵字典。
    方法: 创建父目录并以 UTF-8 缩进 JSON 保存完整轴语义和计数。
    出参: 无；落盘 confusion_matrix.json。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(confusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_confusion_matrix_png(path: Path, confusion: Dict[str, Any]) -> None:
    """入参: PNG 输出路径和混淆矩阵字典。
    方法: 使用固定网格绘制原始计数；对角线用绿色，非对角线用红色，颜色深浅表示计数。
    出参: 无；落盘可直接查看的混淆矩阵图片。"""
    labels = confusion["labels"]
    display_labels = ["BG" if label == BACKGROUND_LABEL else label for label in labels]
    matrix = confusion["matrix"]
    cell_size = 68
    header_size = 120
    title_height = 44
    width = header_size + cell_size * len(labels)
    height = title_height + header_size + cell_size * len(labels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    title_font = ImageFont.load_default(size=18)
    draw.text((12, 12), "Confusion Matrix (rows=GT, columns=Prediction)", fill=(20, 20, 20), font=title_font)
    maximum = max((value for row in matrix for value in row), default=1) or 1
    origin_y = title_height + header_size
    for index, label in enumerate(display_labels):
        x = header_size + index * cell_size
        y = origin_y + index * cell_size
        draw.text((x + 4, title_height + header_size - 28), label, fill=(20, 20, 20), font=font)
        draw.text((4, y + 24), label, fill=(20, 20, 20), font=font)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            intensity = int(45 + 180 * value / maximum)
            color = (40, intensity, 70) if row_index == column_index else (intensity, 45, 45)
            x1 = header_size + column_index * cell_size
            y1 = origin_y + row_index * cell_size
            draw.rectangle((x1, y1, x1 + cell_size, y1 + cell_size), fill=color, outline=(220, 220, 220))
            draw.text((x1 + 26, y1 + 24), str(value), fill="white", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
