"""Independent test of the SAM3+DFATG geology model on the held-out sample5 tiles."""

from __future__ import annotations

import cv2
import numpy as np

from labelme_config import LABELME_NUM_CLASSES


def majority_neighbor_class(class_map: np.ndarray, region: np.ndarray, excluded_class: int | None = None) -> int | None:
    """
    入参:
    - class_map: HxW 五分类类别图。
    - region: HxW bool 区域掩码，表示待重分配区域。
    - excluded_class: 不参与邻域投票的原类别；为空时所有类别均可投票。

    方法:
    - 提取区域外侧一像素的八邻域边界，统计边界类别并选择像素数最多的类别。
    - 票数相同时选择类别 ID 较小者，保证结果可复现。

    出参:
    - int | None: 邻域多数类别；区域没有有效外邻域时返回 None。
    """
    boundary = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    boundary &= ~region
    neighbors = class_map[boundary]
    if excluded_class is not None:
        neighbors = neighbors[neighbors != excluded_class]
    if neighbors.size == 0:
        return None
    counts = np.bincount(neighbors, minlength=LABELME_NUM_CLASSES)
    return int(counts.argmax())


def smooth_foreground_classes(class_map: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    入参:
    - class_map: HxW 五分类类别图。
    - kernel_size: 椭圆形平滑核边长；必须为非负奇数，0 表示关闭。

    方法:
    - 对类别 1 到 4 分别执行闭运算再开运算。
    - 保留仍位于自身平滑掩码内的原前景；新增区域按候选类别距原类别区域的距离决策。
    - 被平滑移除的区域按外侧八邻域多数类别重分配，避免留下未定义像素。

    出参:
    - np.ndarray: 形态学平滑后的 uint8 五分类类别图。
    """
    source = np.asarray(class_map, dtype=np.uint8)
    if kernel_size == 0:
        return source.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    processed = np.zeros((LABELME_NUM_CLASSES, *source.shape), dtype=bool)
    distances = np.full((LABELME_NUM_CLASSES, *source.shape), np.inf, dtype=np.float32)
    for class_id in range(1, LABELME_NUM_CLASSES):
        binary = (source == class_id).astype(np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        processed[class_id] = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel).astype(bool)
        distances[class_id] = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)

    result = source.copy()
    source_foreground = source > 0
    source_survives = np.zeros_like(source_foreground)
    for class_id in range(1, LABELME_NUM_CLASSES):
        source_survives |= (source == class_id) & processed[class_id]

    candidates = processed[1:]
    candidate_count = candidates.sum(axis=0)
    assignable = (~source_foreground & (candidate_count > 0)) | (source_foreground & ~source_survives & (candidate_count > 0))
    candidate_distances = np.where(candidates, distances[1:], np.inf)
    nearest_candidate = candidate_distances.argmin(axis=0).astype(np.uint8) + 1
    result[assignable] = nearest_candidate[assignable]

    removed_without_candidate = source_foreground & ~source_survives & (candidate_count == 0)
    for class_id in range(1, LABELME_NUM_CLASSES):
        count, components = cv2.connectedComponents((removed_without_candidate & (source == class_id)).astype(np.uint8), connectivity=8)
        for component_id in range(1, count):
            region = components == component_id
            replacement = majority_neighbor_class(result, region, excluded_class=class_id)
            if replacement is not None:
                result[region] = replacement
    return result


def remove_small_components(class_map: np.ndarray, min_area: int) -> np.ndarray:
    """
    入参:
    - class_map: HxW 五分类类别图。
    - min_area: 前景连通域最小保留面积，单位为像素；0 表示关闭。

    方法:
    - 在输入快照上逐类提取八连通区域，将面积严格小于阈值的区域改为外侧邻域多数类别。
    - 所有替换决策均基于同一输入快照，避免类别处理顺序导致级联偏置。

    出参:
    - np.ndarray: 小连通域重分配后的 uint8 五分类类别图。
    """
    source = np.asarray(class_map, dtype=np.uint8)
    if min_area == 0:
        return source.copy()
    result = source.copy()
    replacements: list[tuple[np.ndarray, int]] = []
    for class_id in range(1, LABELME_NUM_CLASSES):
        count, components, stats, _ = cv2.connectedComponentsWithStats((source == class_id).astype(np.uint8), connectivity=8)
        for component_id in range(1, count):
            if int(stats[component_id, cv2.CC_STAT_AREA]) >= min_area:
                continue
            region = components == component_id
            replacement = majority_neighbor_class(source, region, excluded_class=class_id)
            if replacement is not None:
                replacements.append((region, replacement))
    for region, replacement in replacements:
        result[region] = replacement
    return result


def fill_small_background_holes(class_map: np.ndarray, max_hole_area: int) -> np.ndarray:
    """
    入参:
    - class_map: HxW 五分类类别图。
    - max_hole_area: 可填充背景孔洞的最大面积，单位为像素；0 表示关闭。

    方法:
    - 提取背景的八连通区域，仅处理不接触图像边界且面积不超过阈值的区域。
    - 只有孔洞外侧边界属于同一个前景类别时才填充，避免跨类别区域被错误吞并。

    出参:
    - np.ndarray: 小孔洞填充后的 uint8 五分类类别图。
    """
    source = np.asarray(class_map, dtype=np.uint8)
    if max_hole_area == 0:
        return source.copy()
    result = source.copy()
    count, components, stats, _ = cv2.connectedComponentsWithStats((source == 0).astype(np.uint8), connectivity=8)
    height, width = source.shape
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        left = int(stats[component_id, cv2.CC_STAT_LEFT])
        top = int(stats[component_id, cv2.CC_STAT_TOP])
        component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        touches_edge = left == 0 or top == 0 or left + component_width == width or top + component_height == height
        if touches_edge or area > max_hole_area:
            continue
        region = components == component_id
        boundary = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        boundary &= ~region
        surrounding_classes = np.unique(source[boundary])
        surrounding_classes = surrounding_classes[surrounding_classes > 0]
        if surrounding_classes.size == 1:
            result[region] = int(surrounding_classes[0])
    return result


def postprocess_class_map(class_map: np.ndarray, smooth_kernel: int, min_area: int, max_hole_area: int) -> np.ndarray:
    """
    入参:
    - class_map: HxW 五分类类别图。
    - smooth_kernel: 闭运算及开运算的椭圆核边长，0 表示关闭。
    - min_area: 前景小连通域面积阈值，0 表示关闭。
    - max_hole_area: 封闭背景孔洞面积上限，0 表示关闭。

    方法:
    - 依次执行逐类形态学平滑、小连通域邻域重分配和封闭背景小孔洞填充。

    出参:
    - np.ndarray: 后处理完成且值域仍为 0 到 4 的 uint8 类别图。
    """
    if smooth_kernel < 0 or (smooth_kernel != 0 and smooth_kernel % 2 == 0):
        raise ValueError("smooth_kernel must be 0 or a positive odd integer")
    if min_area < 0 or max_hole_area < 0:
        raise ValueError("min_area and max_hole_area must be non-negative")
    smoothed = smooth_foreground_classes(class_map, smooth_kernel)
    filtered = remove_small_components(smoothed, min_area)
    return fill_small_background_holes(filtered, max_hole_area)


