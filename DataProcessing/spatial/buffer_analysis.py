# -*- coding: utf-8 -*-
"""
缓冲区分析
对 GeoJSON 要素（点/线/面）创建指定距离的缓冲区，支持结果合并和面积统计。

入参:
- geojson (dict): GeoJSON FeatureCollection 或 Feature
- radius (float): 缓冲距离（单位由 units 决定）
- units (str): 距离单位，默认 "meters"
- steps (int): 圆弧插值步数，1-64，默认 10
- union_results (bool): 是否将所有缓冲区合并为单个几何，默认 False

方法:
- buffer_analysis: 主入口，执行缓冲区分析并返回 GeoJSON FeatureCollection + 统计信息

出参:
- return (dict): {type: "FeatureCollection", features: [...], statistics: {...}}
"""

import time
from typing import Any

from shapely.geometry import mapping
from shapely.ops import unary_union

from .geometry_utils import geojson_to_shapely, shapely_to_geojson


# 单位 → Shapely buffer 的 resolution 参数映射
# Shapely 的 buffer 使用 CRS 单位（默认度），需要将 meters 转为度的近似系数
_METERS_PER_DEGREE = 111_320.0  # 赤道处 1 度 ≈ 111.32 km


def buffer_analysis(
    geojson: dict,
    radius: float,
    units: str = "meters",
    steps: int = 10,
    union_results: bool = False,
) -> dict:
    """
    入参:
    - geojson (dict): GeoJSON FeatureCollection / Feature
    - radius (float): 缓冲距离，正数
    - units (str): "meters" | "kilometers" | "degrees" | "miles" | "feet"
    - steps (int): 圆弧精度，1-64
    - union_results (bool): 是否合并所有缓冲区

    方法:
    - 验证参数 → GeoJSON 转 shapely → 计算各要素缓冲区 → 可选合并 → 统计面积

    出参:
    - return (dict): {type: "FeatureCollection", features: [...], statistics: {...}}
    """
    if not geojson:
        raise ValueError("GeoJSON 数据不能为空")
    if radius <= 0:
        raise ValueError("缓冲距离必须为正数")
    steps = max(1, min(64, steps))

    start = time.time()

    # 转为 shapely 对象列表（元组列表: [(geom, props), ...]）
    geom_props_list = geojson_to_shapely(geojson)

    # 将距离统一转为度（Shapely 在 EPSG:4326 下的 buffer 单位）
    radius_deg = _to_degrees(radius, units)

    # 逐要素生成缓冲区
    buffered = []
    for geom, orig_props in geom_props_list:
        buf = geom.buffer(radius_deg, resolution=steps)
        if buf.is_valid and not buf.is_empty:
            props = dict(orig_props)
            props["bufferRadius"] = radius
            props["bufferUnit"] = units
            props["analysisType"] = "buffer"
            props["area"] = round(buf.area * _METERS_PER_DEGREE ** 2, 2)
            buffered.append((buf, props))

    if not buffered:
        raise ValueError("缓冲区计算后无有效结果")

    # 可选合并
    if union_results and len(buffered) > 1:
        merged_geom = unary_union([b[0] for b in buffered])
        total_area = round(merged_geom.area * _METERS_PER_DEGREE ** 2, 2)
        merged_props = {
            "analysisType": "buffer",
            "bufferRadius": radius,
            "bufferUnit": units,
            "sourceCount": len(buffered),
            "totalArea": total_area,
            "combinedType": "FeatureCollection" if merged_geom.geom_type == "MultiPolygon" else "single",
        }
        features = [shapely_to_geojson(merged_geom, merged_props)]
    else:
        features = [shapely_to_geojson(g, p) for g, p in buffered]

    elapsed = round(time.time() - start, 3)

    total_area = sum(
        f["properties"].get("area", 0) for f in features
    )

    statistics = {
        "inputFeatureCount": len(geom_props_list),
        "outputFeatureCount": len(features),
        "totalArea": round(total_area, 2),
        "areaUnit": "square_meters",
        "executionTime": f"{elapsed}s",
    }

    result = {"type": "FeatureCollection", "features": features, "statistics": statistics}
    return result


def _to_degrees(distance: float, units: str) -> float:
    """将距离统一转为度（WGS84 近似）
    入参: distance(float), units(str)
    方法: 按单位换算系数转换为度
    出参: float, 度数
    """
    if units == "degrees":
        return distance
    if units == "meters":
        return distance / _METERS_PER_DEGREE
    if units == "kilometers":
        return distance * 1000 / _METERS_PER_DEGREE
    if units == "miles":
        return distance * 1609.344 / _METERS_PER_DEGREE
    if units == "feet":
        return distance * 0.3048 / _METERS_PER_DEGREE
    raise ValueError(f"不支持的距离单位: {units}")
