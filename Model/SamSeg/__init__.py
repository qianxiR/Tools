# -*- coding: utf-8 -*-
"""
SAM 分割模型（SegEarth-OV3）
基于 Meta SAM 3 的训练免开放词汇遥感分割与变化检测。
"""

from Tools.Model.SamSeg.SamSeg import (
    # 模型与配置
    load_model,
    load_classes,
    build_palette,
    # 推理核心
    run_inference,
    inference_single_view,
    slide_inference,
    multipass_inference,
    aggregate_logits,
    # 后处理
    postprocess,
    extract_edge,
    # 变化检测
    extract_fpn_features,
    compute_fpn_similarity,
    compute_instance_change_map,
    # 后处理模块
    PAMR,
    LocalAffinity,
    LocalAffinityCopy,
    LocalStDev,
    LocalAffinityAbs,
    # 常量
    DEFAULT_CLASSES,
    NAME_COLOR,
    HAS_RASTERIO,
)
