# -*- coding: utf-8 -*-
"""
兼容转发包
掩码后处理已迁移至 Tools.PostProcessing.mask_postprocess；
保留本包使旧的 Tools.DataProcessing.mask_postprocess 导入路径继续有效。
新代码请使用 Tools.PostProcessing.mask_postprocess。
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
