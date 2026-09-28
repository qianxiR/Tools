# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SegEarth-OV3 is a **training-free** open-vocabulary remote sensing segmentation system built on Meta's SAM 3. It performs 2D semantic segmentation, change detection, and 3D semantic segmentation using text prompts — no fine-tuning required.

## Commands

### Standalone Inference (no mmseg/mmcv/mmengine required)

```powershell
# Semantic segmentation
python infer_seg.py --image "resources/image.png"
python infer_seg.py --image "img.tif" --classes "configs/cls_loveda.txt" --conf 0.1 --prob 0.1

# Change detection (bi-temporal)
python infer_cd.py --t1 "resources/1.png" --t2 "resources/2.png" --classes "configs/cls_levir_cd.txt"

# Batch semantic segmentation (entire dataset split)
python infer_batch_seg.py --data_root "E:/xzkjxm/dataes/WHU_20" --split train

# Batch change detection
python infer_batch_cd.py --data_root "E:/xzkjxm/dataes/CLCD-CD" --split test
```

### Post-Processing Pipeline (CD instance filtering)

```powershell
python tools/cd_pipeline.py --data_root "E:/xzkjxm/dataes/CLCD-CD" --split test --clip_threshold 0.9 --clip_model ViT-B-32 --verify_mode uncertain
```

Steps: instance extraction → RemoteCLIP semantic verification → Qwen-VL visual verification (optional, requires `DASHSCOPE_API_KEY`) → mask rebuild. Use `--skip_step 3` to skip Qwen-VL.

### Segmentation Post-Processing Pipeline (SEG instance filtering)

```powershell
python tools/seg_pipeline.py --data_root "E:/xzkjxm/dataes/WHU_20" --split test --clip_threshold 0.9 --clip_model ViT-B-32 --verify_mode uncertain
```

Steps: instance extraction → RemoteCLIP semantic verification → Qwen-VL visual verification (optional, requires `DASHSCOPE_API_KEY`) → mask rebuild. Use `--skip_step 3` to skip Qwen-VL.

### Dataset Evaluation (requires mmseg/mmengine/mmcv)

```powershell
python eval.py ./configs/cfg_DATASET.py
python -m torch.distributed.launch --nproc_per_node=N eval.py ./configs/cfg_DATASET.py --launcher pytorch
```

## Dependencies

**Standalone inference**: PyTorch (CUDA), scikit-image, scipy, matplotlib, Pillow, huggingface_hub, iopath. Optional: rasterio (geospatial output).

**Dataset evaluation** additionally requires: mmcv, mmsegmentation, mmengine.

**Post-processing pipeline** additionally requires: RemoteCLIP checkpoints in `checkpoints/` (ViT-B-32, RN50, ViT-L-14), `dashscope` SDK for Qwen-VL.

SAM 3 checkpoint at `sam3/sam3.pt`, BPE vocab at `sam3/assets/bpe_simple_vocab_16e6.txt.gz`.

No `requirements.txt` or `pyproject.toml` exists — dependencies are inferred from imports.

## Code Architecture

### Inference Module Layering

```
infer.py              — Shared base module (model loading, inference, post-processing, FPN features)
├── infer_seg.py      — Semantic segmentation entry point (from infer import ...)
├── infer_cd.py       — Change detection entry point (from infer import ...)
├── infer_batch_seg.py — Batch semantic segmentation (full dataset splits)
└── infer_batch_cd.py  — Batch change detection (full dataset splits)
```

**`infer.py`** is the single source of truth for:
- `DEFAULT_CLASSES` and `NAME_COLOR` — unified class definitions and color mapping used across all scripts
- `load_model()` — SAM 3 + Sam3Processor, float32 mode
- `load_classes()` — parse class name files (one line/class, synonyms comma-separated, line 0 = background)
- `inference_single_view()` — dual-head fusion on SAM 3 (instance head + semantic head + presence score)
- `slide_inference()` — sliding window with contextual padding + Gaussian weight blending
- `run_inference()` — auto-select single-view or sliding window based on image size
- `multipass_inference()` — iterative multi-pass with adaptive threshold decay (paints segmented areas black, re-infers residuals)
- `aggregate_logits()` — merge synonym queries per class via one-hot + max
- `postprocess()` — morphological smooth → connected component filter → hole fill → instance interior interpolation
- `extract_fpn_features()` / `compute_fpn_similarity()` — for change detection feature comparison

### Inference Pipeline (2D Segmentation)

1. **Class setup**: parse class file or use `DEFAULT_CLASSES`; background (class 0) generates no text prompt
2. **Model loading**: SAM 3 in float32 via `load_model()`
3. **Multi-pass inference**: `multipass_inference()` iteratively:
   - Paint already-segmented pixels black (scene simplification)
   - Run SAM 3 dual-head fusion per synonym query
   - Element-wise max of instance head logits × object score + semantic head logits
   - Multiply by presence score to suppress absent categories
   - Aggregate synonym queries → per-class logits via max
   - Apply adaptive threshold (decays ×0.7 per pass), only fill uncovered pixels
   - Stop at coverage target or convergence
4. **Post-processing**: `postprocess()` — morphological smooth → small component removal → hole fill → interior interpolation
5. **Edge extraction**: `extract_edge()` — morphological edge via dilation - erosion

### Change Detection Pipeline

1. T1 and T2 independently through multi-pass inference
2. **Feature comparison**: FPN cosine similarity → per-class energy map → Otsu threshold
3. **Joint verification**: pixel-level change map validated against segmentation masks
4. Post-processing + semi-transparent red overlay visualization (alpha=0.4)

### Post-Processing Tools Pipeline (`tools/`)

Instance-level false positive filtering for change detection:

| File | Step | Purpose |
|------|------|---------|
| `cd_utils.py` | Shared | Unicode I/O, class mapping utilities |
| `cd1_extract_instances.py` | 1 | Extract connected components, infer category from change mask |
| `cd2_clip_verify.py` | 2 | RemoteCLIP zero-shot verification (relative score threshold) |
| `cd3_qwen_verify.py` | 3 | Qwen-VL visual-semantic verification (requires `DASHSCOPE_API_KEY`) |
| `cd4_rebuild_mask.py` | 4 | Rebuild mask keeping only verified instances |
| `cd_pipeline.py` | Main | Orchestrates steps 1→4 |

Instance-level false positive filtering for semantic segmentation:

| File | Step | Purpose |
|------|------|---------|
| `cd_utils.py` | Shared | Unicode I/O, class mapping utilities (shared with CD pipeline) |
| `seg1_extract_instances.py` | 1 | Extract connected components, infer category from segmentation mask |
| `seg2_clip_verify.py` | 2 | RemoteCLIP zero-shot verification (reuses CD's CLIP functions) |
| `seg3_qwen_verify.py` | 3 | Qwen-VL visual-semantic verification (requires `DASHSCOPE_API_KEY`) |
| `seg4_rebuild_mask.py` | 4 | Rebuild mask keeping only verified instances |
| `seg_pipeline.py` | Main | Orchestrates steps 1→4 |

### Other Key Files

| File | Purpose |
|------|---------|
| `custom_datasets.py` | MMSeg-registered dataset classes (20+ datasets: WHU, LoveDA, LEVIR-CD, etc.) |
| `custom_transforms.py` | `LoadCDImagesFromFile` for bi-temporal change detection transforms |
| `pamr.py` | Plant-Aware Mask Refinement (color-affinity boundary refinement) |
| `eval.py` | mmengine-based evaluation runner (imports segearthov3 models + custom datasets) |
| `sam3/` | SAM 3 model (upstream from Meta) |

### Dataset Directory Layout (expected by batch scripts)

**Segmentation**: `{data_root}/{split}/img/` or `{data_root}/{split}/image/`

**Change detection**: `{data_root}/{split}/t1/` + `{data_root}/{split}/t2/` + optional `{data_root}/{split}/label/`

Batch output goes to `{data_root}/{split}/{seg_mask|t1_mask|t2_mask|change_mask|overlay}/`

Post-processing intermediate files (instances.json, instance crops) are stored directly in `{data_root}/{split}/` alongside the mask output directories — no separate `cd_postprocess`/`seg_postprocess` subdirectory.

## Windows Compatibility

Triton is not available on Windows. The following files have CPU/scipy fallbacks:
- `sam3/perflib/nms.py` — NMS falls back to CPU implementation
- `sam3/perflib/connected_components.py` — connected components falls back to scikit-image
- `sam3/model/edt.py` — Euclidean distance transform uses `scipy.ndimage.distance_transform_edt`

All standalone scripts use float32 (no autocast) to avoid dtype mismatch on Windows.

## Conventions

- Bi-temporal images with different sizes are auto-resized to max dimensions
- Change masks are visualized as semi-transparent red overlay (alpha=0.4) on both T1 and T2
- Class name files: one line per class, synonyms comma-separated (e.g., `tree,forest`); line 0 is always `background`
- `DEFAULT_CLASSES` defines 7 classes: background, building, road, water, bareland, vegetation, farmland
- Chinese file paths are supported via Unicode-aware I/O helpers in `tools/cd_utils.py`
- Console output is in Chinese
