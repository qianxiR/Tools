# -*- coding: utf-8 -*-
"""
后处理层
解译结果的后处理工具集：掩码清理、矢量化、坐标转换。

子包:
- mask_postprocess: 掩码后处理（平滑、连通域、边缘、矢量、坐标、CRF）

说明:
- 掩码后处理自 DataProcessing 迁入；原路径 Tools.DataProcessing.mask_postprocess
  保留转发包，旧导入路径继续有效。
- QwenSam 的多类别掩码后处理（平滑/去小区域/邻域投票/边缘重建）位于
  Model/qwen_sam/postprocessing.py，与其模型配置存在导入耦合，保留原位。
"""

from Tools.PostProcessing.mask_postprocess import (
    smooth_mask,
    remove_small_objects,
    binary_threshold,
    extract_connected_components,
    polygon_to_coords,
    compute_obb,
    normalize_rotated_rect,
    compute_spatial_location,
    distance_transform_boundary,
    canny_edge,
    sobel_edge,
    batch_extract_boundaries,
    mask_to_polygons,
    raster_to_vector,
    polygon_to_line,
    export_vector,
    pixel_to_geo,
    geo_to_pixel,
    transform_polygon_to_geo,
    transform_polygon_to_pixel,
    transform_line_to_geo,
    wkt_to_epsg,
)
