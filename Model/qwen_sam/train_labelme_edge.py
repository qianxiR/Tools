"""Train the minimal SAM3-Adapter lithology network with cpdc edge refinement."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from labelme_config import (
    LABELME_CLASS_NAMES,
    LABELME_NUM_CLASSES,
    DATASET_CONFIG,
    MODEL_CONFIG,
    TRAIN_CONFIG,
    load_class_map,
)
from models.geology_sam3_dfatg import GeologySam3TextNet
from active_dataset import (
    GeologyTileDataset,
    collate_keep_list,
    edge_bce_dice_loss,
    foreground_dice_loss,
    foreground_lovasz_softmax_loss,
)
from text_embedding import expand_for_batch, load_text_embedding
from autoscan import resolve_weight
from utils.util_functions import multiclass_metrics

IMG_SIZE = MODEL_CONFIG["image_size"]
EDGE_CHANNELS = MODEL_CONFIG["edge_channels"]


def compute_class_weights(records: list[dict], data_root: str) -> torch.Tensor:
    """
    入参:
    - records: 训练集 JSONL 记录。
    - data_root: 数据根目录，用于拼接相对路径。

    方法:
    - 统计五类像素数，前景类使用逆频率平方根权重，背景权重取前景最小值，
      再按前景均值归一化，缓解类别失衡而不让小类主导全部梯度。

    出参:
    - torch.Tensor: [5] 类别权重。
    """
    counts = torch.zeros(LABELME_NUM_CLASSES, dtype=torch.long)
    for record in records:
        class_map = load_class_map(Path(data_root) / record["mask"])
        counts += torch.bincount(torch.from_numpy(class_map.astype(np.int64)).flatten(), minlength=LABELME_NUM_CLASSES)
    print("Train class pixel counts:", dict(zip(LABELME_CLASS_NAMES, counts.tolist())))
    if (counts == 0).any():
        raise RuntimeError(f"Every class must have training pixels, got {counts.tolist()}")
    total = counts.sum().float()
    weights = (total / counts.float()) ** 0.5
    weights[0] = weights[1:].min()
    return weights / weights[1:].mean()


def compute_edge_pos_weights(records: list[dict], data_root: str) -> torch.Tensor:
    """
    入参:
    - records: 训练集 JSONL 记录，每条含四个 boundaries 路径。
    - data_root: 数据根目录。

    方法:
    - 分类别统计边缘正负像素，分别计算 BCE 正类权重并降封到 20，
      多标签边缘不做通道间归一化，保证每个类别边界独立受监督。

    出参:
    - torch.Tensor: [4] 每个类别边缘的正类权重。
    """
    positives = torch.zeros(EDGE_CHANNELS, dtype=torch.float64)
    totals = torch.zeros(EDGE_CHANNELS, dtype=torch.float64)
    for record in records:
        paths = record.get("boundaries")
        if not paths:
            continue
        for class_id, path in enumerate(paths[:EDGE_CHANNELS]):
            with Image.open(Path(data_root) / path) as boundary_file:
                boundary = (np.asarray(boundary_file.convert("L"), dtype=np.uint8) > 127).astype(np.float64)
            positives[class_id] += boundary.sum()
            totals[class_id] += boundary.size
    weights = torch.where(positives > 0, totals / positives.clamp_min(1.0), torch.full_like(positives, 20.0))
    print("Edge positive pixels:", dict(zip(LABELME_CLASS_NAMES[1:], positives.long().tolist())))
    return weights.clamp(max=20.0)


@torch.no_grad()
def validate(model, loader, device, text_hidden=None, text_mask=None) -> dict:
    """
    入参:
    - model: 双头网络（类别 + 四通道类别边缘）。
    - loader: 验证集 DataLoader，提供 [B,4,H,W] 边缘真值。
    - device: 推理设备。

    方法:
    - 聚合类别混淆矩阵，并逐边缘类别统计二值 edge IoU、precision、recall；
      边缘通道独立阈值化，不执行 softmax。

    出参:
    - dict: 类别与四类边缘的汇总指标。
    """
    model.eval()
    total_confusion = torch.zeros((LABELME_NUM_CLASSES, LABELME_NUM_CLASSES), dtype=torch.long)
    edge_inter = torch.zeros(EDGE_CHANNELS, dtype=torch.float64)
    edge_union = torch.zeros(EDGE_CHANNELS, dtype=torch.float64)
    edge_pred_count = torch.zeros(EDGE_CHANNELS, dtype=torch.float64)
    edge_target_count = torch.zeros(EDGE_CHANNELS, dtype=torch.float64)
    for images, labels, boundaries, _ in loader:
        images, labels, boundaries = images.to(device), labels.to(device), boundaries.to(device)
        if text_hidden is not None:
            batch_tokens, batch_mask = expand_for_batch(text_hidden, text_mask, images.shape[0])
            logits, edge_logits = model(images, return_edge=True, text_tokens=batch_tokens, text_mask=batch_mask)
        else:
            logits, edge_logits = model(images, return_edge=True)
        total_confusion += multiclass_metrics(logits.argmax(dim=1).cpu(), labels.cpu(), LABELME_NUM_CLASSES)["confusion"]
        edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
        edge_inter += (edge_pred * boundaries).sum(dim=(0, 2, 3)).cpu().double()
        edge_union += ((edge_pred + boundaries) > 0).float().sum(dim=(0, 2, 3)).cpu().double()
        edge_pred_count += edge_pred.sum(dim=(0, 2, 3)).cpu().double()
        edge_target_count += boundaries.sum(dim=(0, 2, 3)).cpu().double()
    tp = total_confusion.diag().float()
    pred_count = total_confusion.sum(0).float()
    target_count = total_confusion.sum(1).float()
    union = pred_count + target_count - tp
    iou = torch.where(union > 0, tp / union, torch.zeros_like(tp))
    f1 = torch.where(pred_count + target_count > 0, 2.0 * tp / (pred_count + target_count), torch.zeros_like(tp))
    model.train()
    return {
        "per_class_iou": iou.tolist(),
        "per_class_f1": f1.tolist(),
        "foreground_miou": float(iou[1:].mean().item()),
        "macro_f1": float(f1.mean().item()),
        "pixel_accuracy": float(tp.sum().item() / max(1, total_confusion.sum().item())),
        "edge_iou": (edge_inter / edge_union.clamp_min(1.0)).tolist(),
        "edge_precision": (edge_inter / edge_pred_count.clamp_min(1.0)).tolist(),
        "edge_recall": (edge_inter / edge_target_count.clamp_min(1.0)).tolist(),
    }


def create_experiment_dir(configured_out: str | Path) -> tuple[Path, Path]:
    """
    入参:
    - configured_out: 配置指定的实验根目录；目录本身不直接写入训练产物。

    方法:
    - 使用当前时间生成 YYYYMMDD_HHMMSS 实验名，在根目录下创建唯一目录；
      将当前配置和主线训练源码复制到 experiment_sources，保证结果可复现。

    出参:
    - tuple[Path, Path]: 实验输出目录和源码快照目录。
    """
    root = Path(configured_out)
    root.mkdir(parents=True, exist_ok=True)
    experiment_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while experiment_dir.exists():
        experiment_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
        suffix += 1
    experiment_dir.mkdir()
    snapshot_dir = experiment_dir / "experiment_sources"
    snapshot_dir.mkdir()
    source_root = Path(__file__).resolve().parent
    excluded_parts = {"archive", "__pycache__", ".zcode"}
    source_files = [
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
        and not any(part in excluded_parts for part in path.relative_to(source_root).parts)
    ]
    for source_file in source_files:
        relative_path = source_file.relative_to(source_root)
        target = snapshot_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
    return experiment_dir, snapshot_dir


def main():
    config = TRAIN_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATASET_CONFIG["output_root"])
    parser.add_argument("--out", default=config["out"])
    parser.add_argument("--steps", type=int, default=config["steps"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--acc", type=int, default=config["acc"])
    parser.add_argument("--lr", type=float, default=config["lr"])
    parser.add_argument("--wd", type=float, default=config["wd"])
    parser.add_argument("--val-every", type=int, default=config["val_every"])
    parser.add_argument("--log-every", type=int, default=config["log_every"])
    parser.add_argument("--device", default=config["device"])
    parser.add_argument("--edge-loss-weight", type=float, default=config["edge_loss_weight"])
    parser.add_argument("--structure-ablation", choices=["base", "both", "prior", "perception", "fusion"], default="both", help="双模块消融：base=双模块全关（消融基准），both=完整串联，prior=仅边缘先验模块，perception/fusion=仅语义感知模块")
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--text-enabled", action="store_true", default=False, help="启用 Qwen 提示词文本引导（需 --text-embedding 缓存）")
    parser.add_argument("--text-embedding", default=None, help="预计算的 Qwen 文本 embedding 缓存路径（.pt）")
    parser.add_argument("--sam3-path", default=None, help="SAM3 主干权重路径；缺省读配置 model.sam3_checkpoint")
    args = parser.parse_args()
    configured_out = Path(args.out)
    out_dir, snapshot_dir = create_experiment_dir(configured_out)
    print(f"Experiment directory: {out_dir}", flush=True)
    print(f"Experiment sources: {snapshot_dir}", flush=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device

    train_records = [json.loads(line) for line in (Path(args.data_dir) / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    val_records = [json.loads(line) for line in (Path(args.data_dir) / "val.jsonl").read_text(encoding="utf-8").splitlines()]
    class_weights = compute_class_weights(train_records, args.data_dir).to(device)
    edge_pos_weights = compute_edge_pos_weights(train_records, args.data_dir).to(device)
    print(f"Edge pos weights: {edge_pos_weights.tolist()}")
    print(
        f"Training configuration: max_steps={args.steps}, loss=0.6*CE+0.2*Dice+0.2*Lovasz+"
        f"{args.edge_loss_weight}*class_specific_edge_BCE_Dice, edge_channels={EDGE_CHANNELS}, "
        "edge_refinement=Central_PDC(唯一模块), spatial_gate=True"
    )

    train_loader = DataLoader(
        GeologyTileDataset(train_records, args.data_dir, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_keep_list,
    )
    val_loader = DataLoader(
        GeologyTileDataset(val_records, args.data_dir),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_keep_list,
    )
    model = GeologySam3TextNet(
        img_size=IMG_SIZE,
        edge_channels=EDGE_CHANNELS,
        spatial_gate=config["spatial_gate"],
        ablation=args.structure_ablation,
        text_enabled=args.text_enabled,
        text_hidden_size=None,
        sam3_path=args.sam3_path,
    ).to(device)
    args.sam3_path = str(resolve_weight(args.sam3_path, prefer="sam3")) if args.sam3_path else None
    args.text_embedding = str(resolve_weight(args.text_embedding, prefer="embedding")) if args.text_embedding else None
    text_hidden, text_mask = None, None
    if args.text_enabled:
        if not args.text_embedding:
            raise ValueError("--text-enabled requires --text-embedding cache path")
        text_hidden, text_mask, text_hidden_size = load_text_embedding(resolve_weight(args.text_embedding, prefer="embedding"), device)
        print(f"Text-guided enabled: tokens={text_hidden.shape[1]}, hidden_size={text_hidden_size}", flush=True)
        model = GeologySam3TextNet(
            img_size=IMG_SIZE,
            edge_channels=EDGE_CHANNELS,
            spatial_gate=config["spatial_gate"],
            ablation=args.structure_ablation,
            text_enabled=True,
            text_hidden_size=text_hidden_size,
            sam3_path=args.sam3_path,
        ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in params) / 1e6:.2f} M")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total_updates = max(1, math.ceil(args.steps / max(1, args.acc)))
    warmup_updates = max(0, int(0.03 * total_updates))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-8, total_iters=max(1, warmup_updates)),
            CosineAnnealingLR(optimizer, T_max=max(1, total_updates - warmup_updates), eta_min=1e-6),
        ],
        milestones=[max(1, warmup_updates)],
    )
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_score = -1.0
    step = 0
    while step < args.steps:
        for images, labels, boundaries, _ in train_loader:
            if step >= args.steps:
                break
            images, labels, boundaries = images.to(device), labels.to(device), boundaries.to(device)
            if text_hidden is not None:
                batch_tokens, batch_mask = expand_for_batch(text_hidden, text_mask, images.shape[0])
                logits, edge_logits = model(images, return_edge=True, text_tokens=batch_tokens, text_mask=batch_mask)
            else:
                logits, edge_logits = model(images, return_edge=True)
            ce = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=-1)
            dice = foreground_dice_loss(logits, labels)
            lovasz = foreground_lovasz_softmax_loss(logits, labels)
            edge_loss, edge_stats = edge_bce_dice_loss(edge_logits, boundaries, edge_pos_weights)
            loss = (0.6 * ce + 0.2 * dice + 0.2 * lovasz + args.edge_loss_weight * edge_loss) / args.acc
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}: {loss.item()}")
            loss.backward()
            step += 1
            if step % args.acc == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if step % args.log_every == 0:
                print(
                    f"step {step}: CE~{ce.item():.4f} Dice~{dice.item():.4f} "
                    f"EdgeBCE~{edge_stats['bce']:.4f} EdgeDice~{edge_stats['dice']:.4f} "
                    f"EdgePos={','.join(f'{x:.3f}' for x in edge_stats['pos_fraction'])}",
                    flush=True,
                )
                writer.add_scalar("train/ce", ce.item(), step)
                writer.add_scalar("train/dice", dice.item(), step)
                writer.add_scalar("train/edge_bce", edge_stats["bce"], step)
                writer.add_scalar("train/edge_dice", edge_stats["dice"], step)
            if step % args.val_every == 0:
                metrics = validate(model, val_loader, device, text_hidden, text_mask)
                report = " | ".join(
                    f"{name}: IoU={metrics['per_class_iou'][i]:.4f}, F1={metrics['per_class_f1'][i]:.4f}"
                    for i, name in enumerate(LABELME_CLASS_NAMES)
                )
                print(
                    f"[val] step {step}: fg_mIoU={metrics['foreground_miou']:.4f}, "
                    f"pixel_acc={metrics['pixel_accuracy']:.4f}, "
                    f"edge_IoU={','.join(f'{x:.4f}' for x in metrics['edge_iou'])}, "
                    f"edge_P={','.join(f'{x:.4f}' for x in metrics['edge_precision'])}, "
                    f"edge_R={','.join(f'{x:.4f}' for x in metrics['edge_recall'])}\n[val] {report}",
                    flush=True,
                )
                writer.add_scalar("val/foreground_miou", metrics["foreground_miou"], step)
                if metrics["foreground_miou"] > best_score:
                    best_score = metrics["foreground_miou"]
                    checkpoint = {
                        "model": model.state_dict(),
                        "adapter_type": model.backbone.adapter_metadata["adapter_type"],
                        "adapter_metadata": model.backbone.adapter_metadata,
                        "edge_head": True,
                        "edge_channels": EDGE_CHANNELS,
                        "num_classes": LABELME_NUM_CLASSES,
                        "class_names": list(LABELME_CLASS_NAMES),
                        "edge_refinement": "central_pdc",
                        "structure_ablation": model.ablation,
                        "spatial_gate": config["spatial_gate"],
                        "edge_target": "auxiliary_region_binary_255",
                        "edge_loss": "multilabel_bce_dice",
                        "edge_loss_weight": args.edge_loss_weight,
                        "text_enabled": bool(args.text_enabled),
                        "text_hidden_size": int(text_hidden.shape[-1]) if text_hidden is not None else 0,
                    }
                    torch.save(checkpoint, out_dir / "best_model.torch")
                    (out_dir / "best_val_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"New best foreground mIoU: {best_score:.4f}", flush=True)
    torch.save(
        {
            "model": model.state_dict(),
            "adapter_type": model.backbone.adapter_metadata["adapter_type"],
            "adapter_metadata": model.backbone.adapter_metadata,
            "edge_head": True,
            "edge_channels": EDGE_CHANNELS,
            "num_classes": LABELME_NUM_CLASSES,
            "class_names": list(LABELME_CLASS_NAMES),
            "edge_refinement": "central_pdc",
            "structure_ablation": model.ablation,
            "spatial_gate": config["spatial_gate"],
            "edge_target": "auxiliary_region_binary_255",
            "edge_loss": "multilabel_bce_dice",
            "edge_loss_weight": args.edge_loss_weight,
            "text_enabled": bool(args.text_enabled),
            "text_hidden_size": int(text_hidden.shape[-1]) if text_hidden is not None else 0,
        },
        out_dir / "final_model.torch",
    )
    writer.close()
    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
