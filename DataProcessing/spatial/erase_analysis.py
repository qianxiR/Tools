# -*- coding: utf-8 -*-
"""
擦除分析
计算目标图层与擦除图层之间的几何差集（difference），返回擦除后的区域。

入参:
- target_geojson (dict): 目标图层 GeoJSON FeatureCollection
- erase_geojson (dict):  擦除图层 GeoJSON FeatureCollection
- batch_size (int):      批处理大小，默认 100

方法:
- erase_analysis: 主入口，执行擦除分析并返回 GeoJSON + 统计信息

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


def erase_analysis(
    target_geojson: dict,
    erase_geojson: dict,
    batch_size: int = 100,
) -> dict:
    """
    入参:
    - target_geojson (dict): 目标图层 FeatureCollection
    - erase_geojson (dict):  擦除图层 FeatureCollection
    - batch_size (int):      批处理大小

    方法:
    - 验证输入 → validate_features 过滤无效要素 → 合并擦除要素为单个几何（优化策略）
      → 逐目标要素执行 difference → 收集有效结果 → 统计

    出参:
    - return (dict): {type: "FeatureCollection", features: [...], statistics: {...}, metadata: {...}}
    """
    if not target_geojson or not erase_geojson:
        raise ValueError("目标图层和擦除图层数据不能为空")

    start = time.time()

    # 过滤无效要素
    valid_target = validate_features(target_geojson)
    valid_erase = validate_features(erase_geojson)

    target_features = valid_target.get("features", [])
    erase_features = valid_erase.get("features", [])

    if not target_features:
        raise ValueError("目标图层过滤后没有有效要素")
    if not erase_features:
        raise ValueError("擦除图层过滤后没有有效要素")

    # 转为 shapely 对象（元组列表: [(geom, props), ...]）
    target_geom_props = geojson_to_shapely(valid_target)
    erase_geom_props = geojson_to_shapely(valid_erase)

    # 获取第一个目标要素属性作为属性模板
    first_props = {}
    if target_geom_props:
        first_props = dict(target_geom_props[0][1])

    # 合并擦除要素：将多个擦除要素 union 为单个几何，避免 N*M 组合爆炸
    erase_geoms = [g for g, _ in erase_geom_props]
    merged_erase = _merge_erase_geometries(erase_geoms)

    # 逐目标要素与合并擦除几何求差
    results = []
    for i, (target_geom, _) in enumerate(target_geom_props):
        if not target_geom.is_valid or target_geom.is_empty:
            continue
        try:
            difference = target_geom.difference(merged_erase)
            if difference.is_empty:
                continue
            # 尝试合并 MultiPolygon
            difference = _try_merge_multi(difference)

            props = {
                **first_props,
                "analysisType": "erase",
                "sourceLayer": "target",
                "eraseLayer": "erase",
                "processedAt": _now_iso(),
            }
            results.append(shapely_to_geojson(difference, props))
        except Exception:
            continue

    elapsed_ms = (time.time() - start) * 1000

    total_pairs = len(target_geom_props) * len(erase_geoms)
    statistics = {
        "totalResults": len(results),
        "targetFeatureCount": len(target_geom_props),
        "eraseFeatureCount": len(erase_geoms),
        "totalPairs": total_pairs,
        "processingTime": round(elapsed_ms),
        "successRate": round(len(results) / total_pairs * 100, 2) if total_pairs > 0 else 0,
    }

    metadata = {
        "version": "1.0.0",
        "algorithm": "shapely-difference",
        "coordinateSystem": "EPSG:4326",
        "batchSize": batch_size,
    }

    return {
        "type": "FeatureCollection",
        "features": results,
        "statistics": statistics,
        "metadata": metadata,
    }


def _merge_erase_geometries(erase_geoms: list):
    """
    入参: erase_geoms (list) - shapely 对象列表
    方法: 将多个擦除几何 union 为单个几何以优化性能
    出参: 合并后的 shapely 几何对象
    """
    if not erase_geoms:
        raise ValueError("擦除几何列表为空")
    if len(erase_geoms) == 1:
        return erase_geoms[0]
    return unary_union(erase_geoms)


def _try_merge_multi(geom):
    """
    入参: geom - shapely 几何对象
    方法: 如果是 MultiPolygon/GeometryCollection，尝试 unary_union 合并
    出参: 合并后的 shapely 几何对象
    """
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        try:
            return unary_union(geom)
        except Exception:
            return geom
    return geom


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
