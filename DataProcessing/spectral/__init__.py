# -*- coding: utf-8 -*-
"""
光谱指数计算
NDWI / NDVI 等常用遥感光谱指数。
"""

from Tools.DataProcessing.spectral.indices import (
    calculate_ndwi,
    calculate_ndvi,
    stretch_to_255,
    rgb_to_gray,
    compute_indices,
)
