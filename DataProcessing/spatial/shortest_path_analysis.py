# -*- coding: utf-8 -*-
"""
最短路径分析
基于可见性图（Visibility Graph）计算两点间避开障碍物的最短路径，
支持路径距离、预估时间、路径复杂度统计。

入参:
- start_point (dict):  起点 GeoJSON Point {type, coordinates}
- end_point (dict):    终点 GeoJSON Point {type, coordinates}
- obstacles (dict):    障碍物 GeoJSON FeatureCollection（可选）
- options (dict):      分析选项 {units, averageSpeed}

方法:
- shortest_path_analysis: 主入口，执行最短路径分析

出参:
- return (dict): {type: "FeatureCollection", features: [...], statistics: {...}}
"""

import math
import time
from typing import Any

from shapely.geometry import Point, LineString, MultiPoint, mapping, shape
from shapely.ops import nearest_points, unary_union

from .geometry_utils import shapely_to_geojson, SUPPORTED_TYPES

# 赤道处 1 度 ≈ 111.32 km
_METERS_PER_DEGREE = 111_320.0


def shortest_path_analysis(
    start_point: dict,
    end_point: dict,
    obstacles: dict | None = None,
    options: dict | None = None,
) -> dict:
    """
    入参:
    - start_point (dict): GeoJSON Point，如 {"type":"Point","coordinates":[114.3,30.6]}
    - end_point (dict):   GeoJSON Point
    - obstacles (dict | None): 障碍物 FeatureCollection，支持 Point/LineString/Polygon
    - options (dict | None): {units: "kilometers", averageSpeed: 50}

    方法:
    - 验证起终点 → 解析障碍物 → 构建可见性图 → Dijkstra 最短路径
      → 生成路径线 + 起终点 Feature → 统计距离/时间/复杂度

    出参:
    - return (dict): GeoJSON FeatureCollection，含起终点 + 路径线 + statistics
    """
    if not start_point or not end_point:
        raise ValueError("起点和终点不能为空")

    opts = options or {}
    units = opts.get("units", "kilometers")
    avg_speed = opts.get("averageSpeed", 50)

    start = time.time()

    # 解析起终点坐标
    s_coord = _parse_point(start_point)
    e_coord = _parse_point(end_point)

    # 构建障碍物多边形列表
    obstacle_polys = _process_obstacles(obstacles) if obstacles else []

    # 构建可见性图的顶点集合
    vertices = [s_coord, e_coord]
    # 障碍物多边形的顶点也加入候选路径点
    for poly in obstacle_polys:
        coords = list(poly.exterior.coords)[:-1]  # 去掉闭合重复点
        vertices.extend(coords)
        for interior in poly.interiors:
            vertices.extend(list(interior.coords)[:-1])

    # 构建可见性图的边：两个顶点间连线不与任何障碍物相交
    edges = _build_visibility_graph(vertices, obstacle_polys, s_coord, e_coord)

    # Dijkstra 最短路径
    path_coords = _dijkstra(vertices, edges, s_coord, e_coord)

    if not path_coords:
        # 无避障路径时退化为直连
        path_coords = [s_coord, e_coord]

    # 生成路径 LineString
    path_line = LineString(path_coords)
    distance_km = _line_length_km(path_line)
    distance_out = _convert_distance(distance_km, units)
    duration_min = (distance_km / avg_speed) * 60 if avg_speed > 0 else 0

    # 构建 Feature 列表
    features = [
        shapely_to_geojson(Point(s_coord), {
            "name": "起始点", "analysisType": "shortest-path-start",
        }),
        shapely_to_geojson(Point(e_coord), {
            "name": "终点", "analysisType": "shortest-path-end",
        }),
        shapely_to_geojson(path_line, {
            "name": "最短路径", "analysisType": "shortest-path",
            "distance": round(distance_out, 2),
            "duration": round(duration_min, 2),
            "pathType": "optimal",
        }),
    ]

    elapsed = round(time.time() - start, 3)

    statistics = {
        "distance": round(distance_out, 2),
        "distanceUnit": units,
        "duration": round(duration_min, 2),
        "durationUnit": "minutes",
        "complexity": len(path_coords),
        "averageSpeed": avg_speed,
        "speedUnit": "km/h",
        "obstacleCount": len(obstacle_polys),
        "executionTime": f"{elapsed}s",
    }

    return {
        "type": "FeatureCollection",
        "features": features,
        "statistics": statistics,
    }


def _parse_point(point: dict) -> tuple:
    """解析 GeoJSON Point → (lon, lat) 元组
    入参: point(dict) - GeoJSON Point
    方法: 提取 coordinates 前两个值
    出参: (lon, lat)
    """
    if point.get("type") != "Point":
        raise ValueError("点数据必须是 GeoJSON Point 格式")
    coords = point.get("coordinates", [])
    if len(coords) < 2:
        raise ValueError("点坐标至少需要两个值")
    return (coords[0], coords[1])


def _process_obstacles(obstacles: dict) -> list:
    """将障碍物 FeatureCollection 中的各要素转为 Shapely Polygon 列表
    入参: obstacles(dict) - GeoJSON FeatureCollection
    方法: Point/LineString 转为小缓冲区 Polygon；Polygon 直接使用
    出参: list[Polygon]
    """
    polys = []
    for feat in obstacles.get("features", []):
        geom = feat.get("geometry")
        if not geom or geom.get("type") not in SUPPORTED_TYPES:
            continue
        try:
            s = shape(geom)
            if s.is_empty or not s.is_valid:
                continue
            gt = geom["type"]
            if gt in ("Point", "MultiPoint"):
                buf = s.buffer(0.01 / _METERS_PER_DEGREE * 1000)  # ~10m
                polys.append(buf)
            elif gt in ("LineString", "MultiLineString"):
                buf = s.buffer(0.005 / _METERS_PER_DEGREE * 1000)  # ~5m
                polys.append(buf)
            else:
                polys.append(s)
        except Exception:
            continue
    return polys


def _build_visibility_graph(vertices: list, obstacles: list, start: tuple, end: tuple) -> list:
    """构建可见性图边列表
    入参: vertices(list), obstacles(list[Polygon]), start(tuple), end(tuple)
    方法: 遍历所有顶点对，若连线不穿过任何障碍物内部则添加为边
    出参: list[(i, j, weight)]，weight 为欧氏距离
    """
    n = len(vertices)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = vertices[i], vertices[j]
            line = LineString([p1, p2])
            visible = True
            for obs in obstacles:
                # 线段不能穿过障碍物内部（允许与边界相切）
                if line.crosses(obs):
                    visible = False
                    break
            if visible:
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                edges.append((i, j, dist))
    return edges


def _dijkstra(vertices: list, edges: list, start_coord: tuple, end_coord: tuple) -> list:
    """Dijkstra 最短路径
    入参: vertices, edges, start_coord, end_coord
    方法: 标准优先队列 Dijkstra，返回坐标序列
    出参: list[tuple] 路径坐标，失败返回 []
    """
    import heapq

    n = len(vertices)
    start_idx = vertices.index(start_coord)
    end_idx = vertices.index(end_coord)

    # 邻接表
    adj: dict[int, list] = {i: [] for i in range(n)}
    for i, j, w in edges:
        adj[i].append((j, w))
        adj[j].append((i, w))

    # Dijkstra
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[start_idx] = 0
    pq = [(0, start_idx)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end_idx:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # 回溯路径
    if dist[end_idx] == float("inf"):
        return []

    path = []
    cur = end_idx
    while cur != -1:
        path.append(vertices[cur])
        cur = prev[cur]
    path.reverse()
    return path


def _line_length_km(line: LineString) -> float:
    """计算 LineString 在 WGS84 下的近似长度（km）
    入参: line(LineString)
    方法: 逐段使用 Haversine 公式近似
    出参: float，千米
    """
    coords = list(line.coords)
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        total += _haversine_km(lat1, lon1, lat2, lon2)
    return total


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Haversine 公式计算两点间距离（km）"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _convert_distance(km: float, units: str) -> float:
    """将千米转换为指定单位"""
    if units == "kilometers":
        return km
    if units == "meters":
        return km * 1000
    if units == "miles":
        return km * 0.621371
    if units == "degrees":
        return km / (_METERS_PER_DEGREE / 1000)
    return km
