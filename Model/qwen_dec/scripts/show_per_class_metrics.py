"""打印 test_predictions.jsonl 内置的逐类指标与混淆矩阵摘要。

evaluate_detection.py 评估时已通过 detection_metrics.evaluate_results 计算 per_class /
confusion_matrix 并写入 test_predictions.jsonl 首行 metrics。本脚本仅做只读解析与表格展示，
不重复任何匹配/计算，保证与离线 metrics 口径完全一致。
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PREDICTIONS = Path(__file__).resolve().parent.parent / "inference_results" / "test_predictions.jsonl"


def load_metrics(predictions_path: Path) -> dict:
    """入参: predictions_path 为 test_predictions.jsonl 路径。
    方法: 读取首行 type==metrics 的记录并返回其 metrics 字段。
    出参: metrics 字典；文件缺失或首行非 metrics 记录时抛出异常。"""
    if not predictions_path.is_file():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")
    with predictions_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "metrics":
                return record["metrics"]
    raise ValueError(f"no metrics record found in: {predictions_path}")


def print_global(metrics: dict) -> None:
    """入参: metrics 字典。
    方法: 提取并四舍五入展示全局 precision/recall/f1/AP/mAP 及 TP/FP/FN 计数。
    出参: 无。"""
    keys = (
        "images", "gt_boxes", "pred_boxes", "tp", "fp", "fn",
        "precision", "recall", "f1", "mean_matched_iou", "ap50", "map50_95",
        "iou_threshold", "invalid_bbox_outputs",
    )
    print("=" * 70)
    print("全局指标 (Global)")
    print("=" * 70)
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                value = round(value, 4)
            print(f"  {key:<22} {value}")
    print()


def print_per_class(metrics: dict) -> None:
    """入参: metrics 字典。
    方法: 从 per_class 取每类计数与指标，按 f1 降序输出对齐表格。
    出参: 无；per_class 缺失时提示跳过。"""
    per_class = metrics.get("per_class")
    if not per_class:
        print("per_class metrics not found in predictions file.")
        return
    header = f"  {'Class':<8}{'GT':>5}{'Pred':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'P':>8}{'R':>8}{'F1':>8}{'AP50':>8}{'mAP':>8}"
    print("=" * 70)
    print("逐类指标 (Per-class)")
    print("=" * 70)
    print(header)
    print("  " + "-" * (len(header) - 2))
    ordered = sorted(per_class.items(), key=lambda kv: kv[1].get("f1", 0.0), reverse=True)
    for class_name, values in ordered:
        print(
            f"  {class_name:<8}"
            f"{values.get('gt_boxes', 0):>5}"
            f"{values.get('pred_boxes', 0):>6}"
            f"{values.get('tp', 0):>5}"
            f"{values.get('fp', 0):>5}"
            f"{values.get('fn', 0):>5}"
            f"{round(values.get('precision', 0.0), 4):>8}"
            f"{round(values.get('recall', 0.0), 4):>8}"
            f"{round(values.get('f1', 0.0), 4):>8}"
            f"{round(values.get('ap50', 0.0), 4):>8}"
            f"{round(values.get('map50_95', 0.0), 4):>8}"
        )
    print()


def print_confusion_summary(metrics: dict) -> None:
    """入参: metrics 字典。
    方法: 从 confusion_matrix 抽取矩阵与标签，统计对角(正确)/漏检/误检并打印误检最多的组合。
    出参: 无；confusion_matrix 缺失时提示跳过。"""
    confusion = metrics.get("confusion_matrix")
    if not confusion:
        print("confusion_matrix not found in predictions file.")
        return
    labels = confusion.get("labels", [])
    matrix = confusion.get("matrix", [])
    if not labels or not matrix:
        return
    background_index = len(labels) - 1 if labels[-1] == "background" else None
    diagonal = sum(matrix[i][i] for i in range(min(len(labels), len(matrix))))
    off_diagonal = 0
    misses = 0
    false_alarms = 0
    misclass: list[tuple[str, str, int]] = []
    for row_index, row in enumerate(matrix):
        for col_index, count in enumerate(row):
            if row_index == col_index or count == 0:
                continue
            off_diagonal += count
            if background_index is not None and col_index == background_index:
                misses += count
            elif background_index is not None and row_index == background_index:
                false_alarms += count
            else:
                misclass.append((labels[row_index], labels[col_index], count))
    print("=" * 70)
    print("混淆矩阵摘要 (Confusion matrix summary)")
    print("=" * 70)
    print(f"  对角(正确分类):  {diagonal}")
    print(f"  非对角(含漏检/误检): {off_diagonal}")
    if background_index is not None:
        print(f"  漏检(GT→background): {misses}")
        print(f"  误检(background→Pred): {false_alarms}")
    misclass.sort(key=lambda item: item[2], reverse=True)
    if misclass:
        print(f"  最常见混淆(GT→Pred, top 5):")
        for gt_label, pred_label, count in misclass[:5]:
            print(f"    {gt_label} → {pred_label}: {count}")
    print()


def main() -> None:
    """入参: 无。
    方法: 解析命令行 --predictions 参数，加载 metrics 并依次打印全局/逐类/混淆矩阵摘要。
    出参: 无。"""
    import argparse

    parser = argparse.ArgumentParser(description="打印 test_predictions.jsonl 内置逐类指标")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help=f"test_predictions.jsonl 路径(默认 {DEFAULT_PREDICTIONS})",
    )
    args = parser.parse_args()
    metrics = load_metrics(args.predictions)
    print_global(metrics)
    print_per_class(metrics)
    print_confusion_summary(metrics)


if __name__ == "__main__":
    main()
