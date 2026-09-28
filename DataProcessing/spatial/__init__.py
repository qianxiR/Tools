# -*- coding: utf-8 -*-
"""
GIS 空间分析工具集
缓冲区分析、相交分析、擦除分析、最短路径分析、几何工具函数。

依赖: shapely (>=2.0)
"""

from Tools.DataProcessing.spatial.geometry_utils import (
    geojson_to_shapely,
    shapely_to_geojson,
    shapely_list_to_feature_collection,
    validate_features,
    compute_bounds,
    analyze_geometry_types,
)
from Tools.DataProcessing.spatial.buffer_analysis import (
    buffer_analysis,
)
from Tools.DataProcessing.spatial.intersection_analysis import (
    intersection_analysis,
)
from Tools.DataProcessing.spatial.erase_analysis import (
    erase_analysis,
)
from Tools.DataProcessing.spatial.shortest_path_analysis import (
    shortest_path_analysis,
)
from Tools.DataProcessing.spatial.export_utils import (
    export_result,
    export_geojson,
    export_shapefile,
    export_csv,
    export_statistics,
)
