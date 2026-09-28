# -*- coding: utf-8 -*-
"""
SegEarth-OV3: 基于 SAM 3 的训练免开放词汇遥感分割系统
提供语义分割、变化检测、实例后处理管线。

核心依赖: torch, sam3 (内置), PIL, scipy. 可选: rasterio, mmseg, dashscope.
"""

from Tools.Model.SamSeg.SamSeg.infer import (
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
    # 常量
    DEFAULT_CLASSES,
    NAME_COLOR,
    HAS_RASTERIO,
)

from Tools.Model.SamSeg.SamSeg.pamr import (
    PAMR,
    LocalAffinity,
    LocalAffinityCopy,
    LocalStDev,
    LocalAffinityAbs,
)
