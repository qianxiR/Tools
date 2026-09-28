"""Inference for the minimal dual-head model with Central PDC edge refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from labelme_config import (
    DATASET_CONFIG,
    INFERENCE_CONFIG,
    MODEL_CONFIG,
    LABELME_CLASS_NAMES,
    LABELME_NUM_CLASSES,
    LABELME_PALETTE_RGB,
    class_map_to_rgb,
    save_class_map,
)
from models.geology_sam3_dfatg import GeologySam3TextNet
from prepare_labelme_dataset import TILE_SIZE
from postprocessing import postprocess_class_map
from text_embedding import expand_for_batch, load_text_embedding
from autoscan import resolve_weight

IMG_SIZE = MODEL_CONFIG["image_size"]


class InferTileDataset(Dataset):
    """
    入参:
    - records: 无标注 JSONL 记录，仅含 image 相对路径。
    - data_root: 数据根目录。

    方法:
    - 读取图像 tile 并缩放到模型输入尺寸，执行 SAM3 归一化，不加载标签。

    出参:
    - __getitem__ 返回 (image [3,S,S] float, image_path str)。
    """

    def __init__(self, records: list[dict], data_root: str):
        self.samples = [str(Path(data_root) / r["image"]) for r in records]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        raw = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0).sub_(0.5).div_(0.5)
        return tensor, image_path


def infer_collate(batch):
    """
    入参:
    - batch: InferTileDataset 样本列表。

    方法:
    - 将图像堆叠，保留路径列表。

    出参:
    - tuple: images、paths。
    """
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def stitch_class_votes(records: list[dict], tile_dir: Path, tile_size: int, num_classes: int) -> np.ndarray:
    """
    入参:
    - records: 含 tile_origin 的切片记录。
    - tile_dir: 逐 tile 类别索引 PNG 目录。
    - tile_size: 切片尺寸。
    - num_classes: 实际类别数（优先取 checkpoint 元数据，兼容与当前配置不同的任务）。

    方法:
    - 对重叠 tile 做类别多数投票；val/infer 无重叠时等价于直接拼接。

    出参:
    - np.ndarray: HxW uint8 类别图。
    """
    height = max(r["tile_origin"][1] + tile_size for r in records)
    width = max(r["tile_origin"][0] + tile_size for r in records)
    votes = np.zeros((height, width, num_classes), dtype=np.uint16)
    for record in records:
        path = tile_dir / Path(record["image"]).name
        tile = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        top, left = record["tile_origin"][1], record["tile_origin"][0]
        region = votes[top : top + tile_size, left : left + tile_size]
        for class_id in range(num_classes):
            region[..., class_id] += tile == class_id
    return votes.argmax(axis=2).astype(np.uint8)


def add_legend(image: np.ndarray, class_names: tuple | list) -> np.ndarray:
    """
    入参:
    - image: HxWx3 RGB 预测图。
    - class_names: 实际类别名（优先取 checkpoint 元数据，长度即实际类别数）。

    方法:
    - 在底部追加 64px 白色条，按统一调色板绘制类别色块与名称。

    出参:
    - np.ndarray: 带图例的 RGB 图。
    """
    height, width = image.shape[:2]
    canvas = np.full((height + 64, width, 3), 255, dtype=np.uint8)
    canvas[:height] = image
    legend_image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(legend_image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    x = 20
    for class_id, name in enumerate(class_names[:LABELME_NUM_CLASSES]):
        color = LABELME_PALETTE_RGB[class_id]
        text = f"{class_id}: {name}"
        draw.rectangle((x, height + 14, x + 24, height + 38), fill=color, outline=(0, 0, 0), width=1)
        draw.text((x + 32, height + 12), text, fill=(0, 0, 0), font=font)
        x += 32 + draw.textbbox((0, 0), text, font=font)[2] + 28
    return np.asarray(legend_image)


def class_edges_from_map(class_map: np.ndarray) -> np.ndarray:
    """
    入参:
    - class_map: HxW uint8 类别图。

    方法:
    - 使用与数据准备完全相同的逐类 3x3 形态学梯度，输出四个类别边缘通道。

    出参:
    - np.ndarray: [4,H,W] uint8 类别边缘图，255 为边缘。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    class_edges = np.zeros((LABELME_NUM_CLASSES - 1, *class_map.shape), dtype=np.uint8)
    for class_id in range(1, LABELME_NUM_CLASSES):
        binary = (class_map == class_id).astype(np.uint8) * 255
        class_edges[class_id - 1] = cv2.absdiff(cv2.dilate(binary, kernel), cv2.erode(binary, kernel))
    return class_edges


def render_class_edges(edge_map: np.ndarray) -> np.ndarray:
    """
    入参:
    - edge_map: [K,H,W] uint8 类别边缘图，非零像素表示对应类别边界。

    方法:
    - 按类别调色板将边缘通道渲染为彩色线条；同一像素多类别重叠时
      使用类别 ID 较大的颜色，保证颜色与掩码调色板一致。

    出参:
    - np.ndarray: HxWx3 RGB 彩色边缘线图，背景为黑色。
    """
    height, width = edge_map.shape[1:]
    rendered = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id in range(min(edge_map.shape[0], LABELME_NUM_CLASSES - 1)):
        rendered[edge_map[class_id] > 0] = LABELME_PALETTE_RGB[class_id + 1]
    return rendered


def blend_edges(rgb: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
    """
    入参:
    - rgb: HxWx3 RGB 原图。
    - edge_map: [K,H,W] uint8 类别边缘图。

    方法:
    - 保留原图作为底图，将每个类别边缘以对应掩码颜色绘制为实线。

    出参:
    - np.ndarray: HxWx3 RGB 彩色边缘叠加图。
    """
    overlay = rgb.copy()
    for class_id in range(min(edge_map.shape[0], LABELME_NUM_CLASSES - 1)):
        overlay[edge_map[class_id] > 0] = LABELME_PALETTE_RGB[class_id + 1]
    return overlay


def run_split(
    model,
    records: list[dict],
    data_dir: Path,
    out_dir: Path,
    device: str,
    labeled: bool,
    smooth_kernel: int,
    min_area: int,
    max_hole_area: int,
    num_classes: int,
    class_names: tuple | list,
    text_hidden=None,
    text_mask=None,
) -> None:
    """
    入参:
    - model: 双头网络（类别 + 类别边缘）。
    - records: 当前样本切片记录。
    - data_dir/out_dir/device: 数据、输出和设备配置。
    - labeled: 是否为含真值的 val 样本。
    - smooth_kernel/min_area/max_hole_area: 类别后处理参数。
    - num_classes/class_names: 实际类别数与类别名（优先取 checkpoint 元数据）。
    - text_hidden/text_mask: Qwen 提示词特征（文本引导权重时传入）。

    方法:
    - 逐 tile 输出类别 argmax 并投票拼接，对 raw 类别图执行仓库已有后处理；
      边缘线由 postprocessed 掩码按真值同款形态学梯度重建，掩码两图由调色板渲染。

    出参:
    - None: 保存可视化图像并打印样本摘要。
    """
    from active_dataset import GeologyTileDataset, collate_keep_list

    sample_name = records[0]["source_sample"]
    dataset = GeologyTileDataset(records, str(data_dir)) if labeled else InferTileDataset(records, str(data_dir))
    collate = collate_keep_list if labeled else infer_collate
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
    work_dir = out_dir.parent / "work"
    tile_class_dir = work_dir / "pred_class_raw"
    tile_class_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            images, paths = (batch[0], batch[-1]) if labeled else batch
            images = images.to(device)
            if text_hidden is not None:
                batch_tokens, batch_mask = expand_for_batch(text_hidden, text_mask, images.shape[0])
                logits = model(images, text_tokens=batch_tokens, text_mask=batch_mask)
            else:
                logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for index, path in enumerate(paths):
                name = Path(path).stem
                tile = cv2.resize(predictions[index], (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_NEAREST)
                save_class_map(tile, tile_class_dir / f"{name}.png")

    class_raw = stitch_class_votes(records, tile_class_dir, TILE_SIZE, num_classes)
    source_root = Path(json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["source_root"])
    stem = sample_name.replace("sample", "")
    image_path = source_root / "拼图" / f"{stem}.jpg"
    if image_path.is_file():
        original = cv2.cvtColor(
            cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
    else:
        # 回退链：全景/单图 {stem}.tif|.png → 配对输入布局 image\{stem}.*
        for suffix in (".tif", ".png"):
            tiff_path = source_root / f"{stem}{suffix}"
            if tiff_path.is_file():
                with Image.open(tiff_path) as image_file:
                    original = np.asarray(image_file.convert("RGB"))
                break
        else:
            for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                pair_path = source_root / "image" / f"{stem}{suffix}"
                if pair_path.is_file():
                    original = cv2.cvtColor(
                        cv2.imdecode(np.fromfile(str(pair_path), dtype=np.uint8), cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2RGB,
                    )
                    break
            else:
                raise FileNotFoundError(f"Missing source image for {sample_name}: {image_path} / {source_root / stem}.tif/.png / {source_root / 'image' / stem}.*")
    class_raw = class_raw[: original.shape[0], : original.shape[1]]
    class_post = postprocess_class_map(class_raw, smooth_kernel, min_area, max_hole_area)

    sample_dir = out_dir / sample_name.replace("sample", "")
    sample_dir.mkdir(parents=True, exist_ok=True)
    mask = class_map_to_rgb(class_post)
    mask_overlay = (0.55 * original.astype(np.float32) + 0.45 * mask.astype(np.float32)).clip(0, 255).astype(np.uint8)
    cv2.imencode(".png", cv2.cvtColor(add_legend(mask, class_names), cv2.COLOR_RGB2BGR))[1].tofile(sample_dir / "mask.png")
    cv2.imencode(".png", cv2.cvtColor(add_legend(mask_overlay, class_names), cv2.COLOR_RGB2BGR))[1].tofile(sample_dir / "mask_overlay.png")
    derived_edges = class_edges_from_map(class_post)
    derived_edge_px = int(np.any(derived_edges > 0, axis=0).sum())
    cv2.imencode(".png", cv2.cvtColor(render_class_edges(derived_edges), cv2.COLOR_RGB2BGR))[1].tofile(sample_dir / "edge_lines.png")
    cv2.imencode(".png", cv2.cvtColor(blend_edges(original, derived_edges), cv2.COLOR_RGB2BGR))[1].tofile(sample_dir / "edge_overlay.png")
    print(
        f"[{sample_name}] raw_fg={int((class_raw > 0).sum())}, post_fg={int((class_post > 0).sum())}, "
        f"edge_px={derived_edge_px}",
        flush=True,
    )


def select_latest_experiment(root: Path) -> Path:
    """
    入参:
    - root: 实验根目录，子目录名称应为时间戳实验名。

    方法:
    - 过滤包含 best_model.torch 的子目录，按目录修改时间选择最新已完成至少
      一次验证的实验，避免读取根目录旧权重。

    出参:
    - Path: 最新有效实验目录；不存在时抛 FileNotFoundError。
    """
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and (path / "best_model.torch").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed experiment found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    """
    入参:
    - CLI 参数：数据目录、实验根目录或时间戳实验目录、数据 split、设备和后处理参数。

    方法:
    - 从实验根目录选择最新有效 checkpoint，加载模型并对指定 split 推理；结果统一写入
      当前实验目录的 VIS 子目录，不覆盖其他实验目录。

    出参:
    - None: 完成预测并保存四类可视化图像。
    """
    config = INFERENCE_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATASET_CONFIG["output_root"])
    parser.add_argument("--splits", nargs="+", default=list(config["splits"]), choices=["train", "val", "test", "infer"])
    parser.add_argument("--run-dir", default=None, help="实验目录（含 best_model.torch）或实验根；缺省自动扫描 runs 下最新实验，再回退配置")
    parser.add_argument("--checkpoint", default=config["checkpoint"])
    parser.add_argument("--device", default=config["device"])
    parser.add_argument("--smooth-kernel", type=int, default=config["smooth_kernel"])
    parser.add_argument("--min-area", type=int, default=config["min_area"])
    parser.add_argument("--max-hole-area", type=int, default=config["max_hole_area"])
    parser.add_argument("--samples", nargs="+", type=int, default=None, help="仅推理指定样本编号（如 4 1 2 3），用于跳过大尺寸全景图")
    parser.add_argument("--text-embedding", default=None, help="Qwen 文本 embedding 缓存路径；checkpoint 启用文本引导时必传")
    args = parser.parse_args()
    if args.run_dir is None:
        from autoscan import resolve_weight  # noqa: F401  保持 autoscan 在 path 中
        from autoscan import scan_deploy
        scan = scan_deploy(Path(__file__).resolve().parent.parent)
        scan_run = scan.get("latest_run")
        args.run_dir = str(scan_run) if scan_run is not None and (scan_run / args.checkpoint).is_file() else str(INFERENCE_CONFIG["run_dir"])

    data_dir = Path(args.data_dir)
    configured_run_dir = Path(args.run_dir)
    run_dir = select_latest_experiment(configured_run_dir) if not (configured_run_dir / args.checkpoint).is_file() else configured_run_dir
    print(f"run_dir = {run_dir}", flush=True)
    device = args.device
    checkpoint = torch.load(run_dir / args.checkpoint, map_location=device)
    # 类别元数据优先读 checkpoint：加载与当前 QWENSAM_CONFIG 不一致的任务权重时自动适配，不再 size mismatch
    num_classes = int(checkpoint.get("num_classes", LABELME_NUM_CLASSES))
    class_names = tuple(checkpoint.get("class_names", LABELME_CLASS_NAMES))[:num_classes]
    text_enabled = bool(checkpoint.get("text_enabled", False))
    text_hidden_size = int(checkpoint.get("text_hidden_size", 0)) or None
    model = GeologySam3TextNet(
        img_size=IMG_SIZE,
        num_classes=num_classes,
        edge_channels=int(checkpoint.get("edge_channels", MODEL_CONFIG["edge_channels"])),
        text_enabled=text_enabled,
        text_hidden_size=text_hidden_size,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    args.sam3_path = str(resolve_weight(args.sam3_path, prefer="sam3")) if args.sam3_path else None
    args.text_embedding = str(resolve_weight(args.text_embedding, prefer="embedding")) if args.text_embedding else None
    text_hidden, text_mask = None, None
    if text_enabled:
        if not args.text_embedding:
            raise ValueError("checkpoint is text-guided but --text-embedding cache path is required")
        text_hidden, text_mask, _ = load_text_embedding(resolve_weight(args.text_embedding, prefer="embedding"), device)
        print(f"Text-guided inference: tokens={text_hidden.shape[1]} hidden_size={text_hidden.shape[-1]}", flush=True)

    out_dir = run_dir / "VIS"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        records = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        if args.samples is not None:
            selected = {f"sample{sample_id}" for sample_id in args.samples}
            records = [record for record in records if record["source_sample"] in selected]
        for sample_name in sorted({record["source_sample"] for record in records}):
            sample_records = [record for record in records if record["source_sample"] == sample_name]
            run_split(model, sample_records, data_dir, out_dir, device, split in {"train", "val", "test"}, args.smooth_kernel, args.min_area, args.max_hole_area, num_classes, class_names, text_hidden, text_mask)


if __name__ == "__main__":
    main()
