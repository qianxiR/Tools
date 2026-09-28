# -*- coding: utf-8 -*-
"""
GIS 分析结果导出工具
将 GeoJSON FeatureCollection 导出为 JSON/GeoJSON 文件、Shapefile、CSV 等格式。

入参:
- geojson (dict): GeoJSON FeatureCollection（通常是分析函数的返回值）
- output_path (str | Path): 输出文件路径
- format (str): 导出格式 "geojson" | "shapefile" | "csv"

方法:
- export_geojson:    将 GeoJSON 写入 .json/.geojson 文件
- export_shapefile:  将 GeoJSON 转换为 Shapefile (.shp)
- export_csv:        将 GeoJSON 要素属性展开为 CSV 表格
- export_result:     统一入口，根据文件扩展名自动选择格式

出参:
- 各函数返回值见签名（统一返回导出文件路径）
"""

import json
import csv
from pathlib import Path
from typing import Any

from shapely.geometry import shape, mapping


def export_result(geojson: dict, output_path: str | Path) -> Path:
    """
    入参:
    - geojson (dict): GeoJSON FeatureCollection
    - output_path (str | Path): 输出路径，支持 .json/.geojson/.shp/.csv 扩展名

    方法:
    - 根据扩展名自动调用对应的导出函数

    出参:
    - return (Path): 实际写入的文件路径
    """
    p = Path(output_path)
    ext = p.suffix.lower()

    if ext in (".json", ".geojson"):
        return export_geojson(geojson, p)
    if ext == ".shp":
        return export_shapefile(geojson, p)
    if ext == ".csv":
        return export_csv(geojson, p)

    raise ValueError(f"不支持的导出格式: {ext}，支持: .json/.geojson/.shp/.csv")


def export_geojson(geojson: dict, output_path: str | Path, indent: int = 2) -> Path:
    """
    入参:
    - geojson (dict): GeoJSON FeatureCollection
    - output_path (str | Path): 输出文件路径
    - indent (int): JSON 缩进，默认 2

    方法:
    - 剔除 _filter_stats 等非标准字段后序列化为 JSON 文件
    - 自动创建父目录

    出参:
    - return (Path): 写入的文件路径
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # 清理非标准 GeoJSON 字段（如 _filter_stats、statistics、metadata）
    clean = _clean_for_export(geojson)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=indent)

    return p


def export_shapefile(geojson: dict, output_path: str | Path) -> Path:
    """
    入参:
    - geojson (dict): GeoJSON FeatureCollection
    - output_path (str | Path): Shapefile 输出路径（.shp）

    方法:
    - 使用 geopandas 将 GeoJSON 写入 Shapefile
    - 自动处理属性字段类型（统一为 str 避免类型冲突）
    - 多几何类型时会统一为 GeometryCollection 列

    出参:
    - return (Path): .shp 文件路径
    """
    import geopandas as gpd
    from shapely.geometry import shape

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    features = geojson.get("features", [])
    if not features:
        raise ValueError("GeoJSON 中没有要素可导出")

    records = []
    geometries = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        geometries.append(shape(geom))
        props = dict(feat.get("properties", {}))
        # 将所有属性值转为字符串，避免 geopandas 类型推断冲突
        records.append({k: str(v) if v is not None else "" for k, v in props.items()})

    if not geometries:
        raise ValueError("没有有效的几何要素可导出")

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    gdf.to_file(str(p), driver="ESRI Shapefile", encoding="utf-8")

    return p


def export_csv(geojson: dict, output_path: str | Path) -> Path:
    """
    入参:
    - geojson (dict): GeoJSON FeatureCollection
    - output_path (str | Path): CSV 输出路径

    方法:
    - 展开所有要素的 properties 为表格行
    - 附加 geometry_type 和 coordinates（WKT）列
    - 自动收集所有属性键作为表头

    出参:
    - return (Path): CSV 文件路径
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    features = geojson.get("features", [])
    if not features:
        raise ValueError("GeoJSON 中没有要素可导出")

    # 收集所有属性键（保持顺序）
    all_keys = []
    seen = set()
    for feat in features:
        for k in (feat.get("properties") or {}):
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # 额外附加列
    extra_cols = ["geometry_type", "geometry_wkt"]

    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys + extra_cols)
        writer.writeheader()

        for feat in features:
            row = dict(feat.get("properties") or {})
            geom = feat.get("geometry")
            row["geometry_type"] = geom.get("type", "") if geom else ""
            row["geometry_wkt"] = shape(geom).wkt if geom else ""
            writer.writerow(row)

    return p


def export_statistics(analysis_result: dict, output_path: str | Path) -> Path:
    """
    入参:
    - analysis_result (dict): 分析函数返回的完整结果（含 statistics/metadata）
    - output_path (str | Path): 输出 JSON 路径

    方法:
    - 提取 statistics 和 metadata，单独导出为 JSON

    出参:
    - return (Path): 统计信息文件路径
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    stat_data = {}
    if "statistics" in analysis_result:
        stat_data["statistics"] = analysis_result["statistics"]
    if "metadata" in analysis_result:
        stat_data["metadata"] = analysis_result["metadata"]

    with open(p, "w", encoding="utf-8") as f:
        json.dump(stat_data, f, ensure_ascii=False, indent=2)

    return p


# ==================== 内部辅助 ====================

def _clean_for_export(geojson: dict) -> dict:
    """剔除非标准 GeoJSON 字段，仅保留 type + features"""
    clean = {"type": "FeatureCollection", "features": geojson.get("features", [])}
    return clean
