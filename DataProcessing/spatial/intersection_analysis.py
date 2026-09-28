# -*- coding: utf-8 -*-
"""
相交分析
计算两个 GeoJSON 图层之间的几何相交区域，支持遮罩要素合并优化和批量处理。

入参:
- target_geojson (dict): 目标图层 GeoJSON FeatureCollection
- mask_geojson (dict):   遮罩图层 GeoJSON FeatureCollection
- batch_size (int):      批处理大小，默认 100

方法:
- intersection_analysis: 主入口，执行相交分析并返回 GeoJSON + 统计信息

出参:
- return (dict): {type: "FeatureCollection", features: [...], statistics: {...}, metadata: {...}}
"""

import time
from typing import Any

from shapely.ops import unary_union

from .geometry_utils import (
    geojson_to_shapely,
    shapely_to_geojson,
    validate_features,
)


def intersection_analysis(
    target_geojson: dict,
    mask_geojson: dict,
    batch_size: int = 100,
) -> dict:
    """
    入参:
    - target_geojson (dict): 目标图层 FeatureCollection
    - mask_geojson (dict):   遮罩图层 FeatureCollection
    - batch_size (int):      批处理大小

    方法:
    - 验证输入 → validate_features 过滤无效要素 → 合并遮罩为单个几何（优化策略）
      → 逐目标要素执行 intersection → 收集有效结果 → 统计

    出参:
    - return (dict): {type: "FeatureCollection", features: [...], statistics: {...}, metadata: {...}}
    """
    if not target_geojson or not mask_geojson:
        raise ValueError("目标图层和遮罩图层数据不能为空")

    start = time.time()

    # 过滤无效要素
    valid_target = validate_features(target_geojson)
    valid_mask = validate_features(mask_geojson)

    target_features = valid_target.get("features", [])
    mask_features = valid_mask.get("features", [])

    if not target_features:
        raise ValueError("目标图层过滤后没有有效要素")
    if not mask_features:
        raise ValueError("遮罩图层过滤后没有有效要素")

    # 转为 shapely 对象（元组列表: [(geom, props), ...]）
    target_geom_props = geojson_to_shapely(valid_target)
    mask_geom_props = geojson_to_shapely(valid_mask)

    # 获取第一个目标要素属性作为属性模板
    first_props = {}
    if target_geom_props:
        first_props = dict(target_geom_props[0][1])

    # 合并遮罩：将多个遮罩要素 union 为单个几何，避免 N*M 组合爆炸
    mask_geoms = [g for g, _ in mask_geom_props]
    merged_mask = _merge_geometries(mask_geoms)

    # 逐目标要素与合并遮罩求交
    results = []
    for i, (target_geom, _) in enumerate(target_geom_props):
        if not target_geom.is_valid or target_geom.is_empty:
            continue
        try:
            intersection = target_geom.intersection(merged_mask)
            if intersection.is_empty:
                continue
            # 如果结果是 MultiPolygon，尝试合并
            intersection = _try_merge_multi(intersection)

            props = {
                **first_props,
                "analysisType": "intersection",
                "sourceLayer": "target",
                "maskLayer": "mask",
                "processedAt": _now_iso(),
            }
            results.append(shapely_to_geojson(intersection, props))
        except Exception:
            continue

    elapsed_ms = (time.time() - start) * 1000

    total_pairs = len(target_geom_props) * len(mask_geoms)
    statistics = {
        "totalResults": len(results),
        "targetFeatureCount": len(target_geom_props),
        "maskFeatureCount": len(mask_geoms),
        "totalPairs": total_pairs,
        "processingTime": round(elapsed_ms),
        "successRate": round(len(results) / total_pairs * 100, 2) if total_pairs > 0 else 0,
    }

    metadata = {
        "version": "1.0.0",
        "algorithm": "shapely-intersection",
        "coordinateSystem": "EPSG:4326",
        "batchSize": batch_size,
    }

    return {
        "type": "FeatureCollection",
        "features": results,
        "statistics": statistics,
        "metadata": metadata,
    }


def _merge_geometries(geoms: list):
    """
    入参: geoms (list) - shapely 对象列表
    方法: 将多个 shapely 几何 union 为单个几何
    出参: 合并后的 shapely 几何对象
    """
    if not geoms:
        raise ValueError("几何列表为空")
    if len(geoms) == 1:
        return geoms[0]
    return unary_union(geoms)


def _try_merge_multi(geom):
    """
    入参: geom - shapely 几何对象
    方法: 如果是 GeometryCollection/MultiPolygon，尝试 unary_union 合并
    出参: 合并后的 shapely 几何对象
    """
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        try:
            merged = unary_union(geom)
            return merged
        except Exception:
            return geom
    return geom


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
