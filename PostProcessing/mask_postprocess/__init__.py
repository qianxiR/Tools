# -*- coding: utf-8 -*-
"""
掩码后处理工具集
掩码平滑、连通域分析、边缘提取、矢量转换、CRF后处理、坐标转换。
"""

from Tools.PostProcessing.mask_postprocess.mask_processing import (
    smooth_mask,
    remove_small_objects,
    binary_threshold,
)
from Tools.PostProcessing.mask_postprocess.region_analysis import (
    extract_connected_components,
    polygon_to_coords,
    compute_obb,
    normalize_rotated_rect,
    compute_spatial_location,
)
from Tools.PostProcessing.mask_postprocess.edge_utils import (
    distance_transform_boundary,
    canny_edge,
    sobel_edge,
    batch_extract_boundaries,
)
from Tools.PostProcessing.mask_postprocess.vector import (
    mask_to_polygons,
    raster_to_vector,
    polygon_to_line,
    export_vector,
)
from Tools.PostProcessing.mask_postprocess.coordinate import (
    pixel_to_geo,
    geo_to_pixel,
    transform_polygon_to_geo,
    transform_polygon_to_pixel,
    transform_line_to_geo,
    wkt_to_epsg,
)
