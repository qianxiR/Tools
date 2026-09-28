# -*- coding: utf-8 -*-
"""
模型推理基础设施
模型加载/缓存、显存估算、滑动窗口推理、平滑预测、TTA增强。
"""

from Tools.Model.inference.model_utils import (
    load_model_cached,
    clear_model_cache,
    estimate_memory_usage,
    adjust_batch_size,
    pad_image_to_multiple,
)
from Tools.Model.inference.sliding_window import (
    create_sliding_windows,
    create_weight_map,
    stitch_result,
    tif_cropping_array,
    stitch_tif_result,
)
from Tools.Model.inference.smooth_predict import (
    predict_img_with_smooth_windowing,
    cheap_tiling_prediction,
)
