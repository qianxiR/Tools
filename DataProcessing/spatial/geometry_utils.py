# -*- coding: utf-8 -*-
"""
几何工具基础层
GeoJSON ↔ Shapely 双向转换、几何验证、类型分布统计、边界计算。

入参:
- geojson: dict，标准 GeoJSON Feature / FeatureCollection
- feature: dict，GeoJSON Feature
- shapely_geom: shapely 几何对象

方法:
- geojson_to_shapely:       GeoJSON FeatureCollection/Feature → shapely 对象列表
- shapely_to_geojson:       shapely 几何 → GeoJSON Feature
- shapely_list_to_feature_collection: shapely 列表 → GeoJSON FeatureCollection
- validate_features:        过滤无效/闭合线要素，返回有效列表
- compute_bounds:           计算要素集合的经纬度边界
- analyze_geometry_types:   统计几何类型分布

出参:
- 各函数返回值见签名
"""

from typing import Any
import json as _json

from shapely.geometry import (
    shape, mapping,
    Point, MultiPoint,
    LineString, MultiLineString,
    Polygon, MultiPolygon,
    GeometryCollection,
)
from shapely.validation import explain_validity


# GeoJSON 几何类型 → Shapely 类型的白名单
SUPPORTED_TYPES = {
    "Point", "MultiPoint",
    "LineString", "MultiLineString",
    "Polygon", "MultiPolygon",
}


def geojson_to_shapely(geojson: dict) -> list:
    """
    入参:
    - geojson (dict): GeoJSON FeatureCollection 或 Feature

    方法:
    - 将 GeoJSON 要素逐个转换为 (shapely_geom, properties) 元组
    - 跳过无效要素并继续处理

    出参:
    - return (list[tuple]): [(shapely_geom, properties_dict), ...]
    """
    if not geojson:
        raise ValueError("GeoJSON 数据不能为空")

    features = []
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
    elif geojson.get("type") == "Feature":
        features = [geojson]
    else:
        raise ValueError(f"不支持的 GeoJSON 类型: {geojson.get('type')}")

    result = []
    for i, feat in enumerate(features):
        geom = feat.get("geometry")
        if not geom or not geom.get("type") or geom.get("coordinates") is None:
            continue
        geom_type = geom["type"]
        if geom_type not in SUPPORTED_TYPES:
            continue
        try:
            s = shape(geom)
            if s.is_valid:
                props = feat.get("properties", {})
                result.append((s, props))
        except Exception:
            continue

    if not result:
        raise ValueError("没有有效的几何要素可以处理")
    return result


def shapely_to_geojson(geom, properties: dict | None = None) -> dict:
    """
    入参:
    - geom: shapely 几何对象
    - properties (dict | None): 要附加到 Feature 的属性

    方法:
    - 调用 shapely.geometry.mapping 将几何序列化为 GeoJSON dict
    - 包装为 Feature 结构

    出参:
    - return (dict): GeoJSON Feature
    """
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": dict(properties) if properties else {},
    }


def shapely_list_to_feature_collection(geoms, props_list: list | None = None) -> dict:
    """
    入参:
    - geoms (list): shapely 几何对象列表，或 (geom, props) 元组列表
    - props_list (list | None): 对应的属性列表，长度与 geoms 一致

    方法:
    - 将 shapely 列表批量转换为 GeoJSON FeatureCollection
    - 支持直接传入元组列表 [(geom, props), ...]

    出参:
    - return (dict): GeoJSON FeatureCollection
    """
    features = []
    for i, item in enumerate(geoms):
        if isinstance(item, tuple) and len(item) == 2:
            g, props = item
        else:
            g = item
            props = {}
            if props_list and i < len(props_list):
                props = dict(props_list[i])
        features.append(shapely_to_geojson(g, props))

    return {"type": "FeatureCollection", "features": features}


def validate_features(geojson: dict, remove_closed_lines: bool = True) -> dict:
    """
    入参:
    - geojson (dict): GeoJSON FeatureCollection
    - remove_closed_lines (bool): 是否移除闭合线要素（首尾坐标重合的线）

    方法:
    - 遍历所有要素，过滤掉：几何类型不在白名单中、shapely 验证不通过、闭合线
    - 返回过滤后的 FeatureCollection

    出参:
    - return (dict): 过滤后的 GeoJSON FeatureCollection，附带过滤统计信息
    """
    if not geojson or geojson.get("type") != "FeatureCollection":
        return geojson

    valid = []
    removed = 0
    invalid = 0

    for feat in geojson.get("features", []):
        geom = feat.get("geometry")
        if not geom or geom.get("type") not in SUPPORTED_TYPES:
            removed += 1
            continue

        geom_type = geom["type"]

        # 闭合线检测：LineString/MultiLineString 首尾坐标相同
        if remove_closed_lines and geom_type in ("LineString", "MultiLineString"):
            if _is_closed_line(geom):
                removed += 1
                continue

        try:
            s = shape(geom)
            if not s.is_valid:
                invalid += 1
                continue
        except Exception:
            invalid += 1
            continue

        valid.append(feat)

    result = {"type": "FeatureCollection", "features": valid}
    result["_filter_stats"] = {
        "original": len(geojson.get("features", [])),
        "valid": len(valid),
        "removed": removed,
        "invalid": invalid,
    }
    return result


def compute_bounds(geojson: dict) -> dict | None:
    """
    入参:
    - geojson (dict): GeoJSON FeatureCollection

    方法:
    - 遍历所有要素坐标，计算 minX/minY/maxX/maxY
    - 计算 center、width、height

    出参:
    - return (dict | None): 边界对象 {minX, minY, maxX, maxY, center, width, height}
    """
    features = geojson.get("features", [])
    if not features:
        return None

    coords_all = []
    for f in features:
        geom = f.get("geometry")
        if not geom:
            continue
        _flatten_coords(geom, coords_all)

    if not coords_all:
        return None

    xs = [c[0] for c in coords_all]
    ys = [c[1] for c in coords_all]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return {
        "minX": min_x, "minY": min_y,
        "maxX": max_x, "maxY": max_y,
        "center": [(min_x + max_x) / 2, (min_y + max_y) / 2],
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def analyze_geometry_types(features: list) -> dict:
    """
    入参:
    - features (list): GeoJSON Feature 数组

    方法:
    - 统计各几何类型出现次数和占比

    出参:
    - return (dict): {类型名: {count, percentage}}
    """
    total = len(features)
    if total == 0:
        return {}

    counts: dict[str, int] = {}
    for f in features:
        t = (f.get("geometry") or {}).get("type", "Unknown")
        counts[t] = counts.get(t, 0) + 1

    return {
        t: {"count": c, "percentage": round(c / total * 100)}
        for t, c in counts.items()
    }


# ==================== 内部辅助 ====================

def _is_closed_line(geom: dict, tol: float = 1e-10) -> bool:
    """判断 LineString/MultiLineString 是否闭合"""
    gt = geom.get("type")
    coords = geom.get("coordinates", [])

    if gt == "LineString":
        if len(coords) < 3:
            return False
        return (abs(coords[0][0] - coords[-1][0]) < tol
                and abs(coords[0][1] - coords[-1][1]) < tol)

    if gt == "MultiLineString":
        return any(_is_closed_line({"type": "LineString", "coordinates": line}) for line in coords)

    return False


def _flatten_coords(geom: dict, out: list):
    """从任意几何体中提取所有二维坐标到 out 列表"""
    gt = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if gt == "Point":
        out.append(coords)
    elif gt == "LineString":
        out.extend(coords)
    elif gt == "Polygon":
        for ring in coords:
            out.extend(ring)
    elif gt == "MultiPoint":
        out.extend(coords)
    elif gt == "MultiLineString":
        for line in coords:
            out.extend(line)
    elif gt == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                out.extend(ring)
