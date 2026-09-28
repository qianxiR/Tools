# -*- coding: utf-8 -*-
"""
数据处理层
遥感影像通用数据处理工具集，无模型依赖，可独立使用。

子包:
- raster_utils:      GDAL栅格I/O基础工具（加载、波段提取、文件检索、多边形化）
- raster_io:         GeoTIFF读写与地理信息拷贝
- image_ops:         影像增强（8bit转换、光谱指数、直方图匹配）
- image_transforms:  数据增强变换管线（缩放/裁剪/翻转/归一化）
- dataset_utils:     数据集划分、网格切割、同步文件夹处理
- band_utils:        波段映射检测、点位波段值提取
- sample_prep:       分割数据准备工具集（波段组合/裁剪/H5/标签处理）
- mask_postprocess:  兼容转发包（实现已迁至 Tools.PostProcessing.mask_postprocess）
- spectral:          光谱指数计算（NDWI/NDVI）
- visualization:     结果可视化（GeoTIFF保存、热力图、四视图）
- image_utils:       影像基础处理（裁剪、拉伸、波段组合、重采样）
- gee:               Google Earth Engine数据获取（波段提取、格式转换）
- spatial:           GIS空间分析（缓冲区、相交、擦除、最短路径、导出）
"""

# ==================== 栅格 I/O ====================
from Tools.DataProcessing.raster_utils import (
    load_img_by_gdal,
    load_img_by_gdal_geo,
    load_img_by_gdal_info,
    load_img_by_gdal_blocks,
    load_img_bybandlist,
    load_img_normalization,
    load_label,
    load_src,
    get_file,
    find_file,
    geotrans_match,
    polygonize,
    send_message_callback,
    echoRuntime,
    UINT8,
    UINT10,
    UINT16,
)

from Tools.DataProcessing.raster_io import (
    read_geotiff,
    read_img_with_geo,
    write_img_with_geo,
    get_geo_info,
    copy_geo_to_file,
)

# ==================== 影像增强与变换 ====================
from Tools.DataProcessing.image_ops import (
    convert_to_8bit,
    calc_spectral_index,
    histogram_match,
)

from Tools.DataProcessing.image_transforms import (
    ScaleTransform,
    ResizeTransform,
    RandomCropResizeTransform,
    RandomFlipTransform,
    NormalizeTransform,
    RandomExchangeTransform,
    ToTensorTransform,
    ComposedTransforms,
    TransformBuilder,
)

# ==================== 数据集工具 ====================
from Tools.DataProcessing.dataset_utils import (
    split_dataset,
    get_common_files,
    split_image_grid,
    process_synchronized_folders,
    create_folder_structure,
    copy_files_to_splits,
)

# ==================== 波段工具 ====================
from Tools.DataProcessing.band_utils import (
    detect_band_mapping,
    extract_band_values_at_points,
    geotiff_to_csv,
)

# ==================== 掩码后处理 ====================
from Tools.DataProcessing.mask_postprocess import (
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

# ==================== 光谱指数 ====================
from Tools.DataProcessing.spectral import (
    calculate_ndwi,
    calculate_ndvi,
    stretch_to_255,
    rgb_to_gray,
    compute_indices,
)

# ==================== 可视化 ====================
from Tools.DataProcessing.visualization import (
    save_geotiff_mask,
    visualize_quad_view,
    csv_to_heatmap,
    load_watermask_data,
)

# ==================== GEE ====================
from Tools.DataProcessing.gee import (
    resize_tiff_to_500x500,
    convert_to_rgb_255,
    extract_bands_to_csv,
    extract_bands_to_csv_with_mask,
    rgb_255_to_reflectance,
    reflectance_to_rgb_255,
    csv_to_geotiff,
    csv_to_png,
)

# ==================== 空间分析 ====================
from Tools.DataProcessing.spatial import (
    geojson_to_shapely,
    shapely_to_geojson,
    shapely_list_to_feature_collection,
    validate_features,
    compute_bounds,
    analyze_geometry_types,
    buffer_analysis,
    intersection_analysis,
    erase_analysis,
    shortest_path_analysis,
    export_result,
    export_geojson,
    export_shapefile,
    export_csv,
    export_statistics,
)
