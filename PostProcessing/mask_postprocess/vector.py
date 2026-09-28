# -*- coding: utf-8 -*-
"""
矢量转换工具
将栅格掩码转换为矢量要素，支持栅格↔矢量互转和面→线提取。

入参:
- mask: 二值掩码 (H, W) numpy 数组
- mask_path: 栅格文件路径
- polygon_shp_path: 面矢量文件路径

方法:
- mask_to_polygons: 二值掩码 → shapely 多边形列表
- raster_to_vector: 栅格文件 → Shapefile/GeoJSON
- polygon_to_line: 面文件 → 线文件

出参:
- 各函数返回值见签名
"""

import os
import numpy as np
import cv2
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import shape, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

from .coordinate import pixel_to_geo, wkt_to_epsg


def mask_to_polygons(mask, min_area=100, simplify=True, simplify_tolerance=0.5):
    """
    入参:
    - mask (np.ndarray): 二值掩码 (H, W)，支持 uint8/float/bool
    - min_area (float): 最小多边形面积（像素单位），小于此值被过滤
    - simplify (bool): 是否使用 Douglas-Peucker 简化
    - simplify_tolerance (float): 简化容差

    方法:
    - 智能类型转换：bool→0/1, float→阈值二值化, uint8→127 阈值
    - 形态学平滑：开运算去噪 → 闭运算填洞 → 中值滤波 → 高斯模糊
    - rasterio.features.shapes 提取几何 → shapely 转换 → 面积过滤 → 简化

    出参:
    - return (tuple): (polygons_list, areas_list)
    """
    if mask is None or mask.size == 0:
        return [], []

    # 确保 2D
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]

    # 类型归一化到 uint8 二值
    if mask.dtype in (np.bool_, bool):
        binary = mask.astype(np.uint8)
    elif np.issubdtype(mask.dtype, np.floating):
        binary = (mask > 0.5).astype(np.uint8)
    else:
        binary = (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)

    # 形态学平滑
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=2)
    binary = cv2.medianBlur(binary, 5)
    smoothed = cv2.GaussianBlur(binary.astype(np.float32), (9, 9), 2.0)
    binary = (smoothed > 0.5).astype(np.uint8)

    if binary.sum() == 0:
        return [], []

    # 提取多边形
    shapes_gen = features.shapes(binary, mask=binary > 0, connectivity=8)
    polygons, areas = [], []

    for geom, value in shapes_gen:
        if value == 0:
            continue
        poly = shape(geom)
        if not isinstance(poly, (Polygon, MultiPolygon)):
            continue
        if poly.area < min_area:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_valid or poly.is_empty:
            continue
        if simplify and simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance * 2, preserve_topology=True)
        if not poly.is_valid or poly.is_empty or poly.area < min_area:
            continue
        polygons.append(poly)
        areas.append(poly.area)

    return polygons, areas


def raster_to_vector(mask_path, output_shp_path, filter_classes=None,
                     min_area=0.0, source_image_path=None):
    """
    入参:
    - mask_path (str): 分割 mask 栅格文件路径
    - output_shp_path (str): 输出矢量路径
    - filter_classes (list[int]|None): 需要处理的类别 ID，None 处理所有非 0
    - min_area (float): 最小面积（平方米）
    - source_image_path (str|None): 原始影像路径（优先读取其坐标系）

    方法:
    - 优先从 source_image_path 读取 CRS；否则使用 mask 自身 CRS
    - 遍历每个类别，二值化 → rasterio.features.shapes 提取几何
    - 地理坐标系下自动转 UTM 计算面积

    出参:
    - return (gpd.GeoDataFrame|None): 矢量数据
    """
    os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)

    img_crs, transform = None, None
    if source_image_path and os.path.exists(source_image_path):
        with rasterio.open(source_image_path) as src:
            img_crs, transform = src.crs, src.transform

    with rasterio.open(mask_path) as f:
        image = f.read(1)
        if img_crs is None:
            img_crs = f.crs
        if transform is None:
            transform = f.transform
        nodata = f.nodata if f.nodata else 0
        image[image == nodata] = 0

        unique = np.unique(image)
        process = [c for c in unique if c != 0 and (filter_classes is None or c in filter_classes)]

        rows = []
        for cid in process:
            binary = (image == cid).astype(np.uint8)
            for coords, val in features.shapes(binary, transform=transform, mask=binary):
                if val != 1:
                    continue
                geom = shape(coords)
                if min_area > 0 and img_crs and img_crs.is_geographic:
                    centroid = geom.centroid
                    utm_zone = int((centroid.x + 180) / 6) + 1
                    utm_crs = f'EPSG:{32600 + utm_zone}' if centroid.y >= 0 else f'EPSG:{32700 + utm_zone}'
                    area = gpd.GeoDataFrame([geom], crs=img_crs).to_crs(utm_crs).geometry.iloc[0].area
                elif min_area > 0:
                    area = geom.area
                else:
                    area = 0
                if min_area > 0 and area < min_area:
                    continue
                rows.append({'class_id': int(cid), 'geometry': geom})

    if not rows:
        return None

    gdf = gpd.GeoDataFrame(rows)
    gdf.set_geometry('geometry', inplace=True)
    if img_crs:
        gdf.set_crs(img_crs, inplace=True)
    gdf.to_file(output_shp_path, encoding="utf-8")
    return gdf


def polygon_to_line(polygon_shp_path, output_line_shp_path):
    """
    入参:
    - polygon_shp_path (str): 面矢量文件路径
    - output_line_shp_path (str): 输出线矢量路径

    方法:
    - 提取每个 Polygon/MultiPolygon 的 boundary
    - MultiLineString 拆分为独立 LineString

    出参:
    - return (gpd.GeoDataFrame): 线矢量数据
    """
    os.makedirs(os.path.dirname(output_line_shp_path), exist_ok=True)
    gdf_poly = gpd.read_file(polygon_shp_path)
    if gdf_poly.empty:
        return gpd.GeoDataFrame()

    line_geoms, line_attrs = [], []
    for _, row in gdf_poly.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for poly in polys:
            boundary = poly.boundary
            geoms = list(boundary.geoms) if boundary.geom_type == 'MultiLineString' else [boundary]
            for line in geoms:
                line_geoms.append(line)
                attrs = row.drop('geometry').to_dict()
                attrs['class_id'] = row.get('class_id', 0)
                line_attrs.append(attrs)

    if not line_geoms:
        return gpd.GeoDataFrame()

    gdf_line = gpd.GeoDataFrame(line_attrs, geometry=line_geoms, crs=gdf_poly.crs)
    gdf_line.to_file(output_line_shp_path, encoding="utf-8")
    return gdf_line


def export_vector(polygons, areas, output_path, geo_transform=None,
                  projection=None, export_format='shp'):
    """
    入参:
    - polygons (list): shapely 多边形列表
    - areas (list[float]): 对应面积
    - output_path (str): 输出路径
    - geo_transform (tuple|None): 仿射变换参数（用于坐标转换）
    - projection (str|None): 投影 WKT
    - export_format (str): 'shp' 或 'geojson'

    方法:
    - 像素多边形 → 地理多边形（若提供 geo_transform）
    - 添加 ID/Area/Perimeter 属性
    - 设置 CRS 并导出

    出参:
    - return (bool): 是否导出成功
    """
    if not polygons:
        return False

    geometries = []
    for poly in polygons:
        if not isinstance(poly, (Polygon, MultiPolygon)):
            continue
        if geo_transform is not None:
            if isinstance(poly, MultiPolygon):
                converted = []
                for p in poly.geoms:
                    c = _transform_single_polygon(p, geo_transform)
                    if c and c.is_valid:
                        converted.append(c)
                if converted:
                    geometries.append(MultiPolygon(converted))
            else:
                c = _transform_single_polygon(poly, geo_transform)
                if c and c.is_valid:
                    geometries.append(c)
        elif poly.is_valid and not poly.is_empty:
            geometries.append(poly)

    if not geometries:
        return False

    attr = {'ID': list(range(1, len(geometries) + 1))}
    if areas:
        attr['Area'] = [float(a) if i < len(areas) else 0.0 for i, a in enumerate(areas)]
        attr['Perimeter'] = [float(g.length) for g in geometries]

    gdf = gpd.GeoDataFrame(attr, geometry=geometries)
    if projection:
        epsg = wkt_to_epsg(projection)
        gdf.crs = f"EPSG:{epsg}" if epsg else "EPSG:4326"
    else:
        gdf.crs = "EPSG:4326"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    driver = 'GeoJSON' if export_format.lower() == 'geojson' else 'ESRI Shapefile'
    gdf.to_file(output_path, driver=driver)
    return True


def _transform_single_polygon(polygon, geo_transform):
    """单多边形坐标转换（内部辅助函数）"""
    from .coordinate import transform_polygon_to_geo
    return transform_polygon_to_geo(polygon, geo_transform)
