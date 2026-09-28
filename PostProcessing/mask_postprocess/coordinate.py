# -*- coding: utf-8 -*-
"""
坐标转换工具
封装遥感影像中像素坐标与地理坐标之间的双向转换，支持点/线/面要素。

入参:
- x, y: 像素坐标或地理坐标
- geo_transform: GDAL 仿射变换六参数 (x0, dx, rx, y0, ry, -dy)
- polygon/line: shapely 几何对象

方法:
- pixel_to_geo: 单个像素坐标 → 地理坐标
- geo_to_pixel: 单个地理坐标 → 像素坐标
- transform_polygon_to_geo: 像素多边形 → 地理多边形
- transform_polygon_to_pixel: 地理多边形 → 像素多边形
- transform_line_to_geo: 像素线 → 地理线

出参:
- 转换后的坐标或 shapely 几何对象
"""

from osgeo import osr
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString


def pixel_to_geo(x, y, geo_transform):
    """
    入参:
    - x (float): 像素列号
    - y (float): 像素行号
    - geo_transform (tuple): (x0, dx, rx, y0, ry, -dy)

    方法:
    - 使用仿射变换公式将像素坐标映射到地理坐标
    - geo_x = x0 + col*dx + row*rx
    - geo_y = y0 + col*ry - row*|dy|

    出参:
    - return (tuple): (geo_x, geo_y)
    """
    geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
    geo_y = geo_transform[3] + x * geo_transform[4] - y * abs(geo_transform[5])
    return geo_x, geo_y


def geo_to_pixel(geo_x, geo_y, geo_transform):
    """
    入参:
    - geo_x (float): 地理 X 坐标
    - geo_y (float): 地理 Y 坐标
    - geo_transform (tuple): 仿射变换参数

    方法:
    - 仿射变换的逆运算，将地理坐标映射回像素坐标
    - 使用行列式求逆矩阵

    出参:
    - return (tuple): (pixel_col, pixel_row)
    """
    dx = geo_transform[1]
    ry = geo_transform[2]
    ry2 = geo_transform[4]
    neg_dy = -abs(geo_transform[5])

    det = dx * neg_dy - ry * ry2
    if abs(det) < 1e-12:
        return 0.0, 0.0

    px = geo_x - geo_transform[0]
    py = geo_y - geo_transform[3]
    col = (px * neg_dy - ry * py) / det
    row = (dx * py - ry2 * px) / det
    return col, row


def transform_polygon_to_geo(polygon, geo_transform):
    """
    入参:
    - polygon (Polygon): 像素坐标系下的 shapely 多边形
    - geo_transform (tuple): 仿射变换参数

    方法:
    - 转换外环和所有内环坐标
    - 验证多边形有效性，无效时用 buffer(0) 修复

    出参:
    - return (Polygon|None): 地理坐标系下的多边形
    """
    if not isinstance(polygon, Polygon) or not polygon.is_valid:
        return None

    # 外环
    ext = [pixel_to_geo(x, y, geo_transform) for x, y in polygon.exterior.coords]
    if len(ext) < 3:
        return None

    # 内环
    holes = []
    for interior in polygon.interiors:
        hole = [pixel_to_geo(x, y, geo_transform) for x, y in interior.coords]
        if len(hole) >= 3:
            holes.append(hole)

    geo_poly = Polygon(ext, holes) if holes else Polygon(ext)

    if not geo_poly.is_valid:
        geo_poly = geo_poly.buffer(0)
        if not geo_poly.is_valid or geo_poly.is_empty:
            return None
    return geo_poly


def transform_polygon_to_pixel(polygon, geo_transform):
    """
    入参:
    - polygon (Polygon): 地理坐标系下的 shapely 多边形
    - geo_transform (tuple): 仿射变换参数

    方法:
    - 地理坐标 → 像素坐标的逆变换

    出参:
    - return (Polygon|None): 像素坐标系下的多边形
    """
    if not isinstance(polygon, Polygon) or not polygon.is_valid:
        return None

    ext = [geo_to_pixel(x, y, geo_transform) for x, y in polygon.exterior.coords]
    if len(ext) < 3:
        return None

    holes = []
    for interior in polygon.interiors:
        hole = [geo_to_pixel(x, y, geo_transform) for x, y in interior.coords]
        if len(hole) >= 3:
            holes.append(hole)

    px_poly = Polygon(ext, holes) if holes else Polygon(ext)
    if not px_poly.is_valid:
        px_poly = px_poly.buffer(0)
        if not px_poly.is_valid or px_poly.is_empty:
            return None
    return px_poly


def transform_line_to_geo(line, geo_transform):
    """
    入参:
    - line (LineString|MultiLineString): 像素坐标线要素
    - geo_transform (tuple): 仿射变换参数

    方法:
    - 逐点转换线坐标，支持 LineString 和 MultiLineString

    出参:
    - return (LineString|MultiLineString): 地理坐标线要素
    """
    if line.geom_type == 'LineString':
        coords = [pixel_to_geo(x, y, geo_transform) for x, y in line.coords]
        return LineString(coords)
    if line.geom_type == 'MultiLineString':
        return MultiLineString([transform_line_to_geo(l, geo_transform) for l in line.geoms])
    raise ValueError(f"不支持的几何类型: {type(line)}")


def wkt_to_epsg(wkt):
    """
    入参:
    - wkt (str): 投影 WKT 字符串

    方法:
    - 从 WKT 解析出 EPSG 代码

    出参:
    - return (str|None): 如 "4326" 或 None
    """
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return srs.GetAuthorityCode(None)
