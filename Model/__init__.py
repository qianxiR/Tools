# -*- coding: utf-8 -*-
"""
模型层
模型定义与推理基础设施，覆盖视觉模型与视觉语言模型。
依赖 DataProcessing 层的栅格工具。

子包:
- inference:       推理基础设施（加载、缓存、滑动窗口、平滑预测）
- segmentation:    语义分割模型库（UNet/FPN/PSPNet/DeepLab/HRNet 等，视觉）
- changedetect:    变化检测模型（DSIFN、Detect_siam_unet）与结果可视化（视觉）
- detection:       目标检测（YOLOv5，视觉）
- classification:  场景分类（视觉）
- SamSeg:          SegEarth-OV3: SAM 3 开放词汇分割与变化检测（免训练，文本提示，视觉语言）
- qwen_sam:        SAM3 双头多类别分割（文本融合 + 边缘先验，视觉语言）
- qwen_dec:        Qwen3-VL 目标检测与检测 VL 数据集制作（视觉语言）

注: SamSeg 目录名被 Agent-RSCD 后端按 Tools.Model.SamSeg.SamSeg 路径引用，保持原名不改。
"""

# ==================== 推理基础设施 ====================
from Tools.Model.inference import (
    load_model_cached,
    clear_model_cache,
    estimate_memory_usage,
    adjust_batch_size,
    pad_image_to_multiple,
    create_sliding_windows,
    create_weight_map,
    stitch_result,
    tif_cropping_array,
    stitch_tif_result,
    predict_img_with_smooth_windowing,
    cheap_tiling_prediction,
)

# ==================== SAM 分割（训练免） ====================
from Tools.Model.SamSeg import (
    load_model,
    load_classes,
    build_palette,
    run_inference,
    inference_single_view,
    slide_inference,
    multipass_inference,
    aggregate_logits,
    postprocess,
    extract_edge,
    extract_fpn_features,
    compute_fpn_similarity,
    compute_instance_change_map,
    PAMR,
    LocalAffinity,
    LocalAffinityCopy,
    LocalStDev,
    LocalAffinityAbs,
    DEFAULT_CLASSES,
    NAME_COLOR,
    HAS_RASTERIO,
)

