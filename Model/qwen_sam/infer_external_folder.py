"""Run compact full-image inference on every image in the external input directory."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from labelme_config import DATASET_CONFIG, INFERENCE_CONFIG, LABELME_CLASS_NAMES, LABELME_NUM_CLASSES, MODEL_CONFIG, class_map_to_rgb
from models.geology_sam3_dfatg import GeologySam3TextNet
from postprocessing import postprocess_class_map
from test_labelme_edge import InferTileDataset, add_legend, blend_edges, infer_collate, render_class_edges, select_latest_experiment, stitch_class_votes
from text_embedding import expand_for_batch, load_text_embedding
from autoscan import resolve_weight

def tile_starts(length: int, tile_size: int) -> list[int]:
    """入参: 长度和切片边长；方法: 无重叠起点；出参: 覆盖完整方向的起点列表。"""
    return list(range(0, max(1, length), tile_size))


def prepare_external_records(image_path: Path, work_root: Path, tile_size: int, stride: int) -> tuple[list[dict], tuple[int, int]]:
    """
    入参:
    - image_path: 外部完整图像。
    - work_root: 临时切片目录。
    - tile_size: 切片边长。
    - stride: 窗口步长；小于 tile_size 即重叠滑动窗口（默认 10% 重叠）。

    方法:
    - 读取完整图，按窗口步长重叠切片（右/下不足补黑），生成兼容
      InferTileDataset 的临时 JSON 记录；重叠区由拼接时的多数投票融合，
      消除 tile 接缝边界伪影。

    出参:
    - tuple: (记录列表, 原图 (height,width))。
    """
    raw = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image_dir = work_root / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    height, width = raw.shape[:2]
    records = []
    for top in tile_starts(height, stride):
        for left in tile_starts(width, stride):
            tile = raw[top : top + tile_size, left : left + tile_size]
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[: tile.shape[0], : tile.shape[1]] = tile
            name = f"{image_path.stem}_{top:05d}_{left:05d}.png"
            cv2.imencode(".png", padded)[1].tofile(image_dir / name)
            records.append({"image": f"image/{name}", "source_sample": image_path.stem, "tile_origin": [left, top]})
    return records, (height, width)


def load_model(run_dir: Path, device: str):
    """入参: 权重目录/设备；方法: 按 checkpoint 元数据恢复模型（含文本引导开关）；出参: eval 模型。"""
    checkpoint = torch.load(run_dir / INFERENCE_CONFIG["checkpoint"], map_location=device)
    text_enabled = bool(checkpoint.get("text_enabled", False))
    text_hidden_size = int(checkpoint.get("text_hidden_size", 0)) or None
    model = GeologySam3TextNet(
        img_size=MODEL_CONFIG["image_size"],
        edge_channels=checkpoint.get("edge_channels", MODEL_CONFIG["edge_channels"]),
        text_enabled=text_enabled,
        text_hidden_size=text_hidden_size,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def memory_safe_postprocess(class_map: np.ndarray) -> np.ndarray:
    """
    入参:
    - class_map: HxW uint8 类别预测图。

    方法:
    - 先执行轻量平滑；对超大外部图不运行逐连通域邻域重分配，避免
      connectedComponents 临时内存峰值导致整图推理中断；使用按类别
      形态学开运算移除小孤岛，再执行受限孔洞填充。

    出参:
    - np.ndarray: 后处理类别图。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INFERENCE_CONFIG["smooth_kernel"], INFERENCE_CONFIG["smooth_kernel"]))
    result = class_map.copy()
    for class_id in range(1, LABELME_NUM_CLASSES):
        binary = (result == class_id).astype(np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        result[(binary > 0) & (opened == 0)] = 0
    return result


def infer_image(image_path: Path, run_dir: Path, model, device: str, stride: int, postprocess_mode: str, text_hidden=None, text_mask=None) -> None:
    """入参: 单图、模型、设备、窗口步长、后处理模式与文本特征；方法: 重叠窗口切片→推理→投票拼接→后处理→保存四图；出参: None。"""
    tile_size = DATASET_CONFIG["tile_size"]
    with tempfile.TemporaryDirectory(prefix="qwensam_") as temp:
        temp_root = Path(temp)
        records, original_size = prepare_external_records(image_path, temp_root, tile_size, stride)
        tile_dir = temp_root / "pred"
        tile_dir.mkdir()
        loader = DataLoader(InferTileDataset(records, str(temp_root)), batch_size=4, shuffle=False, num_workers=0, collate_fn=infer_collate)
        with torch.no_grad():
            for images, paths in loader:
                images = images.to(device)
                if text_hidden is not None:
                    batch_tokens, batch_mask = expand_for_batch(text_hidden, text_mask, images.shape[0])
                    logits = model(images, text_tokens=batch_tokens, text_mask=batch_mask)
                else:
                    logits = model(images)
                predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
                for index, path in enumerate(paths):
                    prediction = cv2.resize(predictions[index], (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
                    cv2.imencode(".png", prediction)[1].tofile(tile_dir / Path(path).name)
        raw_map = stitch_class_votes(records, tile_dir, tile_size, LABELME_NUM_CLASSES)[: original_size[0], : original_size[1]]
        if postprocess_mode == "full":
            post = postprocess_class_map(raw_map, INFERENCE_CONFIG["smooth_kernel"], INFERENCE_CONFIG["min_area"], INFERENCE_CONFIG["max_hole_area"])
        else:
            post = memory_safe_postprocess(raw_map)
    original = cv2.cvtColor(cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = class_map_to_rgb(post)
    edge_map = np.stack([cv2.morphologyEx((post == class_id).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) * 255 for class_id in range(1, LABELME_NUM_CLASSES)], axis=0)
    edge_rgb = render_class_edges(edge_map)
    mask_overlay = (0.55 * original.astype(np.float32) + 0.45 * mask.astype(np.float32)).clip(0, 255).astype(np.uint8)
    edge_overlay = blend_edges(original, edge_map)
    out = run_dir / "VIS" / image_path.stem
    out.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", cv2.cvtColor(edge_rgb, cv2.COLOR_RGB2BGR))[1].tofile(out / "edge_lines.png")
    cv2.imencode(".png", cv2.cvtColor(add_legend(mask, LABELME_CLASS_NAMES), cv2.COLOR_RGB2BGR))[1].tofile(out / "mask.png")
    cv2.imencode(".png", cv2.cvtColor(edge_overlay, cv2.COLOR_RGB2BGR))[1].tofile(out / "edge_overlay.png")
    cv2.imencode(".png", cv2.cvtColor(add_legend(mask_overlay, LABELME_CLASS_NAMES), cv2.COLOR_RGB2BGR))[1].tofile(out / "mask_overlay.png")
    print(f"{image_path.name} -> {out}")


def main() -> None:
    """入参: CLI；方法: 扫描外部目录所有 JPG/PNG/TIF，以默认 10% 重叠窗口+完整后处理逐图推理；出参: vis/<stem>/四图。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=r"F:\岩石\拼图")
    parser.add_argument("--run-dir", default=None, help="实验目录（含 best_model.torch）或实验根；缺省自动扫描 runs 下最新实验，再回退配置")
    parser.add_argument("--device", default=INFERENCE_CONFIG["device"])
    parser.add_argument("--infer-overlap", type=float, default=0.2, help="推理窗口重叠比例，默认 0.2（20% 滑动窗口，投票融合优化边界）")
    parser.add_argument("--postprocess", choices=["full", "light"], default="full", help="默认 full=平滑+去小区域+填洞；light=大图轻量形态学（仅开运算）")
    parser.add_argument("--text-embedding", default=None, help="Qwen 文本 embedding 缓存路径；文本引导 checkpoint 时必传")
    args = parser.parse_args()
    if not 0 <= args.infer_overlap < 1:
        parser.error("--infer-overlap must be in [0, 1)")
    if args.run_dir is None:
        from autoscan import scan_deploy
        scan = scan_deploy(Path(__file__).resolve().parent.parent)
        scan_run = scan.get("latest_run")
        args.run_dir = str(scan_run) if scan_run is not None and (scan_run / INFERENCE_CONFIG["checkpoint"]).is_file() else str(INFERENCE_CONFIG["run_dir"])
    run_dir = select_latest_experiment(Path(args.run_dir)) if not (Path(args.run_dir) / INFERENCE_CONFIG["checkpoint"]).is_file() else Path(args.run_dir)
    print(f"run_dir = {run_dir}", flush=True)
    model = load_model(run_dir, args.device)
    text_enabled = bool(torch.load(run_dir / INFERENCE_CONFIG["checkpoint"], map_location="cpu").get("text_enabled", False))
    args.sam3_path = str(resolve_weight(args.sam3_path, prefer="sam3")) if args.sam3_path else None
    args.text_embedding = str(resolve_weight(args.text_embedding, prefer="embedding")) if args.text_embedding else None
    text_hidden, text_mask = None, None
    if text_enabled:
        if not args.text_embedding:
            raise ValueError("checkpoint is text-guided but --text-embedding cache path is required")
        text_hidden, text_mask, _ = load_text_embedding(resolve_weight(args.text_embedding, prefer="embedding"), args.device)
        print(f"Text-guided inference: tokens={text_hidden.shape[1]} hidden_size={text_hidden.shape[-1]}", flush=True)
    stride = max(1, int(DATASET_CONFIG["tile_size"] * (1.0 - args.infer_overlap)))
    print(f"infer window: tile={DATASET_CONFIG['tile_size']} stride={stride} (overlap={args.infer_overlap:.0%}), postprocess={args.postprocess}", flush=True)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    input_path = Path(args.input_dir)
    if input_path.is_file():
        if input_path.suffix.lower() not in extensions:
            raise RuntimeError(f"Unsupported single image type: {input_path}")
        images = [input_path]
    else:
        images = sorted(path for path in input_path.iterdir() if path.suffix.lower() in extensions)
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir}")
    for image_path in images:
        infer_image(image_path, run_dir, model, args.device, stride, args.postprocess, text_hidden, text_mask)


if __name__ == "__main__":
    main()
