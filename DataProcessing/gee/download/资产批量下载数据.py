#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEE资产批量下载脚本（增强版）
从 JSON 文件读取资产列表，逐个资产进行影像下载。
支持任务队列控制、项目级上限检测与进度监控。
"""

import ee
import time
import json
import os
from datetime import datetime

# ==========================================================
# 配置参数
# ==========================================================
CONFIG = {
    "project": "applied-pipe-453411-k9",

    # 【新增】资产列表 JSON 文件路径
    "assetListJson": r"E:\1代码\模型\gee\数据下载\矢量遍历下载数据\gee_assets_list.json",

    # 【资产筛选配置】
    "assetTypeFilter": ["TABLE"],   # 只处理 TABLE 类型的资产，可选: TABLE, IMAGE_COLLECTION
    "skipAssets": [],               # 要跳过的资产名称列表，例如: ["china_city", "china_county"]

    # 【下载模式配置】
    "downloadMode": "ask",          # 下载模式：asset(按资产), feature(按要素), ask(运行时询问)
    "featureIdField": None,         # 要素ID字段名，None表示自动检测常见字段
    "commonIdFields": ["HPBM", "SKBM", "hpbm", "skbm", "ID", "id", "FID", "fid"],  # 常见ID字段列表

    # 【时间和影像参数】
    "startDate": "2025-06-01",
    "endDate": "2025-07-31",
    "cloudFilter": 20,
    "scale": 10,
    "maxPixels": 1e13,
    "bands": ["B2", "B3", "B4", "B8"],
    "crs": "EPSG:4326",
    "compositeMethod": "median",    # median, mean, mosaic

    # 【导出配置】
    "exportFolder": "GEE_Batch_Assets_Exports",
    "fileNamePrefix": "Asset",      # 文件名前缀（按资产模式）
    "featureFilePrefix": "Feature", # 文件名前缀（按要素模式）

    # 【任务控制参数】
    "maxConcurrentTasks": 20,       # 最大并发任务数（推荐20–50）
    "checkInterval": 30,            # 检查任务状态间隔(秒)
    "safeTaskThreshold": 2900,      # 安全任务阈值（项目最大3000）
    "assetDelay": 15,               # 资产间延迟(秒)
    "featureDelay": 10,             # 要素间延迟(秒)

    # 【任务状态记录配置】
    "enableTaskTracking": True,     # 是否启用任务状态跟踪
    "taskStateFile": "task_state.json",  # 任务状态记录文件
    "autoResume": True,             # 是否自动从上次中断处继续

    # 【批次处理配置】
    "enableBatchMode": True,        # 是否启用批次模式
    "batchSize": 5,                 # 每批处理的数量（资产或要素）
    "batchStart": 0,                # 开始批次编号
    "batchEnd": -1,                 # 结束批次编号（-1表示全部）
}

# ==========================================================
# 任务状态管理
# ==========================================================
class TaskStateManager:
    """任务状态管理器 - 记录每个要素对应的GEE任务ID和状态"""

    def __init__(self, state_file):
        self.state_file = state_file
        self.state = {
            'metadata': {
                'created_time': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'download_mode': None,
                'total_items': 0
            },
            'tasks': {}  # key: 唯一标识(资产名_要素ID), value: {task_id, status, submit_time, ...}
        }
        self.load_state()

    def load_state(self):
        """从文件加载状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                    self.state.update(loaded_state)
                print(f"✅ 加载任务状态文件: {self.state_file}")
                print(f"   已记录任务数: {len(self.state['tasks'])}")
                return True
            except Exception as e:
                print(f"⚠️ 加载状态文件失败: {e}，将创建新状态文件")
        return False

    def save_state(self):
        """保存状态到文件"""
        try:
            self.state['metadata']['last_updated'] = datetime.now().isoformat()
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ 保存状态文件失败: {e}")
            return False

    def add_task(self, item_key, task_id, asset_name, feature_id=None, extra_info=None):
        """添加任务记录"""
        task_info = {
            'task_id': task_id,
            'asset_name': asset_name,
            'feature_id': feature_id,
            'status': 'SUBMITTED',
            'submit_time': datetime.now().isoformat(),
            'gee_status': 'READY'
        }
        if extra_info:
            task_info.update(extra_info)

        self.state['tasks'][item_key] = task_info
        self.save_state()

    def is_task_submitted(self, item_key):
        """检查任务是否已提交"""
        return item_key in self.state['tasks']

    def get_submitted_count(self):
        """获取已提交的任务数"""
        return len(self.state['tasks'])

    def get_pending_items(self, all_items):
        """获取未提交的项目列表"""
        pending = []
        for item in all_items:
            # 根据模式生成唯一key
            if 'feature_id' in item:
                # 按要素模式
                item_key = f"{item['asset_name']}_{item['feature_id']}"
            else:
                # 按资产模式
                item_key = item['name']

            if not self.is_task_submitted(item_key):
                pending.append(item)

        return pending

    def update_task_status(self, item_key, status):
        """更新任务状态"""
        if item_key in self.state['tasks']:
            self.state['tasks'][item_key]['gee_status'] = status
            self.state['tasks'][item_key]['last_check_time'] = datetime.now().isoformat()
            self.save_state()

    def get_statistics(self):
        """获取统计信息"""
        total = len(self.state['tasks'])
        status_count = {}
        for task in self.state['tasks'].values():
            status = task.get('gee_status', 'UNKNOWN')
            status_count[status] = status_count.get(status, 0) + 1

        return {
            'total': total,
            'status_breakdown': status_count
        }

# ==========================================================
# 初始化 Earth Engine
# ==========================================================
def init_ee():
    """初始化Earth Engine"""
    try:
        ee.Initialize(project=CONFIG["project"])
        print("✅ Earth Engine 初始化成功")
    except Exception as e:
        print(f"❌ Earth Engine 初始化失败: {e}")
        raise

# ==========================================================
# 用户交互：选择下载模式
# ==========================================================
def ask_download_mode():
    """询问用户选择下载模式"""
    print("\n" + "=" * 60)
    print("📋 请选择下载模式")
    print("=" * 60)
    print("【模式 1】按资产下载")
    print("   - 每个资产导出 1 个影像文件")
    print("   - 包含该资产内所有要素的范围")
    print("   - 适合：需要完整资产影像的场景")
    print("")
    print("【模式 2】按要素下载")
    print("   - 每个湖泊要素导出 1 个影像文件")
    print("   - 只包含单个要素的范围")
    print("   - 适合：需要单独处理每个湖泊的场景")
    print("   - ⚠️  注意：要素数量多时会生成大量任务")
    print("=" * 60)

    while True:
        choice = input("\n请输入选择 (1=按资产, 2=按要素): ").strip()
        if choice == "1":
            print("✅ 已选择：按资产下载")
            return "asset"
        elif choice == "2":
            print("✅ 已选择：按要素下载")
            return "feature"
        else:
            print("❌ 无效输入，请输入 1 或 2")


# ==========================================================
# 检测要素ID字段
# ==========================================================
def detect_feature_id_field(vector_collection):
    """自动检测要素ID字段"""
    print("🔍 正在检测要素ID字段...")

    # 如果用户指定了字段，直接使用
    if CONFIG['featureIdField']:
        print(f"   ✓ 使用配置的字段: {CONFIG['featureIdField']}")
        return CONFIG['featureIdField']

    try:
        # 获取第一个要素的属性
        first_feature = ee.Feature(vector_collection.first())
        properties = first_feature.toDictionary()
        available_fields = properties.keys().getInfo()

        print(f"   可用字段: {', '.join(available_fields)}")

        # 按顺序尝试常见字段
        for field in CONFIG['commonIdFields']:
            if field in available_fields:
                print(f"   ✓ 检测到ID字段: {field}")
                return field

        # 如果没有找到常见字段，使用第一个非几何字段
        for field in available_fields:
            if field not in ['geometry', 'system:index']:
                print(f"   ⚠️  使用默认字段: {field}")
                return field

        print("   ❌ 未找到合适的ID字段")
        return None

    except Exception as e:
        print(f"   ❌ 检测字段失败: {e}")
        return None


# ==========================================================
# 提取要素ID列表
# ==========================================================
def extract_feature_ids(vector_collection, id_field):
    """提取所有要素的ID列表"""
    print(f"🔍 正在提取要素ID列表（字段: {id_field}）...")

    try:
        # 获取要素总数
        total_count = vector_collection.size().getInfo()
        print(f"   📊 要素总数: {total_count}")

        # 批量提取所有ID
        feature_ids = vector_collection.aggregate_array(id_field).getInfo()

        # 过滤空值
        feature_ids = [fid for fid in feature_ids if fid is not None]

        print(f"   ✅ 成功提取 {len(feature_ids)} 个有效要素ID")

        # 显示前几个ID作为预览
        preview_count = min(5, len(feature_ids))
        print(f"   预览前 {preview_count} 个ID: {feature_ids[:preview_count]}")

        return feature_ids

    except Exception as e:
        print(f"   ❌ 提取要素ID失败: {e}")
        return []


# ==========================================================
# 读取资产列表 JSON 文件
# ==========================================================
def load_asset_list(json_file):
    """读取资产列表 JSON 文件"""
    print(f"📂 读取资产列表文件: {json_file}")

    if not os.path.exists(json_file):
        print(f"❌ 文件不存在: {json_file}")
        return None

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'assets' not in data:
            print("❌ JSON 文件格式错误，缺少 'assets' 字段")
            return None

        assets = data['assets']
        print(f"✅ 成功读取 {len(assets)} 个资产")

        # 显示元数据信息
        if 'metadata' in data:
            metadata = data['metadata']
            print(f"📊 资产来源: {metadata.get('folder_path', 'N/A')}")
            print(f"📊 提取时间: {metadata.get('extract_time', 'N/A')}")
            print(f"📊 总资产数: {metadata.get('total_count', 'N/A')}")

        return data

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None


# ==========================================================
# 筛选资产
# ==========================================================
def filter_assets(assets_data):
    """根据配置筛选资产"""
    if not assets_data or 'assets' not in assets_data:
        return []

    all_assets = assets_data['assets']
    type_filter = CONFIG['assetTypeFilter']
    skip_list = CONFIG['skipAssets']

    print(f"\n🔍 筛选资产...")
    print(f"   类型过滤: {type_filter if type_filter else '全部'}")
    print(f"   跳过列表: {skip_list if skip_list else '无'}")

    filtered = []

    for asset in all_assets:
        asset_type = asset.get('type')
        asset_name = asset.get('name')

        # 类型过滤
        if type_filter and asset_type not in type_filter:
            print(f"   ⊗ 跳过 [{asset_type}] {asset_name} (类型不匹配)")
            continue

        # 跳过列表
        if skip_list and asset_name in skip_list:
            print(f"   ⊗ 跳过 [{asset_type}] {asset_name} (在跳过列表中)")
            continue

        filtered.append(asset)
        print(f"   ✓ 保留 [{asset_type}] {asset_name}")

    print(f"\n✅ 筛选完成，保留 {len(filtered)} 个资产")
    return filtered


# ==========================================================
# 获取任务状态统计信息
# ==========================================================
def get_task_stats(prefix=None):
    """获取当前任务统计信息（prefix=None时统计所有任务）"""
    try:
        tasks = ee.batch.Task.list()
        stats = {
            'READY': 0,
            'RUNNING': 0,
            'COMPLETED': 0,
            'FAILED': 0,
            'CANCELLED': 0,
            'total': 0
        }
        for task in tasks:
            desc = task.config.get('description', '')
            # 如果未指定前缀，统计所有任务；否则只统计匹配前缀的任务
            if prefix is None or desc.startswith(prefix):
                state = task.state
                if state in stats:
                    stats[state] += 1
                stats['total'] += 1
        stats['active'] = stats['READY'] + stats['RUNNING']
        return stats
    except Exception as e:
        print(f"⚠️ 获取任务状态失败: {e}")
        return None

# ==========================================================
# 等待任务队列空位
# ==========================================================
def wait_for_task_slot():
    """等待任务队列有空位"""
    max_tasks = CONFIG["maxConcurrentTasks"]
    check_interval = CONFIG["checkInterval"]
    while True:
        stats = get_task_stats()
        if stats is None:
            time.sleep(check_interval)
            continue
        active_count = stats['active']
        if active_count < max_tasks:
            print(f"   ✅ 当前活动任务: {active_count}/{max_tasks} (READY:{stats['READY']}, RUNNING:{stats['RUNNING']})")
            return stats
        print(f"   ⏳ 队列已满 {active_count}/{max_tasks}, 等待 {check_interval} 秒...")
        print(f"      📊 READY:{stats['READY']} | RUNNING:{stats['RUNNING']} | COMPLETED:{stats['COMPLETED']} | FAILED:{stats['FAILED']}")
        time.sleep(check_interval)

# ==========================================================
# 检查项目级任务容量（防止超过2500上限）
# ==========================================================
def check_project_task_capacity(max_tasks_allowed=2500):
    """
    检查整个项目的任务总量是否接近上限（默认2500）
    当任务超过safe_threshold时，自动暂停直到任务数下降
    """
    safe_threshold = CONFIG["safeTaskThreshold"]
    check_interval = CONFIG["checkInterval"]

    while True:
        stats = get_task_stats()
        if stats is None:
            print("⚠️ 无法获取任务状态，等待后重试...")
            time.sleep(check_interval)
            continue

        total_tasks = stats['total']
        active_tasks = stats['active']

        if total_tasks >= safe_threshold:
            print(f"🚫 当前任务总数 {total_tasks}/{max_tasks_allowed} 已接近上限，暂停提交...")
            print(f"   📊 READY:{stats['READY']} | RUNNING:{stats['RUNNING']} | COMPLETED:{stats['COMPLETED']} | FAILED:{stats['FAILED']}")
            time.sleep(check_interval)
            continue

        print(f"   ✅ 当前项目任务总数: {total_tasks}/{max_tasks_allowed} (活跃 {active_tasks})，可以提交新任务")
        return

# ==========================================================
# 获取哨兵-2影像合成
# ==========================================================
def get_sentinel2_composite(region):
    """获取哨兵-2影像合成"""
    s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                   .filterBounds(region)
                   .filterDate(CONFIG["startDate"], CONFIG["endDate"])
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG["cloudFilter"])))
    image_count = s2_collection.size().getInfo()
    print(f"   📊 找到 {image_count} 张可用影像")
    if image_count == 0:
        return None
    if CONFIG["compositeMethod"] == "median":
        composite = s2_collection.median()
    elif CONFIG["compositeMethod"] == "mean":
        composite = s2_collection.mean()
    else:
        composite = s2_collection.mosaic()
    composite = composite.select(CONFIG["bands"]).clip(region)
    return composite

# ==========================================================
# 提交导出任务（含队列与容量控制）
# ==========================================================
def submit_export_task(image, region, name, index, total, mode="asset"):
    """提交导出任务（带队列控制+容量检测），返回任务对象和任务ID"""
    if mode == "asset":
        print(f"\n📤 准备提交资产 {index+1}/{total} ({name})")
        prefix = CONFIG['fileNamePrefix']
    else:
        print(f"\n📤 准备提交要素 {index+1}/{total} (ID: {name})")
        prefix = CONFIG['featureFilePrefix']


    # 不做任何队列或容量检查，直接提交任务

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    description = f"{prefix}_Sentinel2_{name}_{timestamp}"

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=CONFIG['exportFolder'],
        scale=CONFIG['scale'],
        region=region,
        crs=CONFIG['crs'],
        fileFormat='GeoTIFF',
        maxPixels=CONFIG['maxPixels'],
        formatOptions={'cloudOptimized': True}
    )
    task.start()

    # 获取任务ID
    task_id = task.id
    print(f"   ✅ 任务已提交: {description}")
    print(f"   🆔 任务ID: {task_id}")

    return task, task_id

# ==========================================================
# 按资产下载
# ==========================================================
def process_by_asset(filtered_assets, task_manager=None):
    """按资产模式批量下载"""
    print("\n" + "=" * 60)
    print("📋 按资产模式下载")
    print("=" * 60)

    # 如果启用了任务跟踪，过滤出未提交的资产
    if task_manager and CONFIG['enableTaskTracking'] and CONFIG['autoResume']:
        pending_assets = task_manager.get_pending_items(filtered_assets)
        submitted_count = task_manager.get_submitted_count()
        print(f"   已提交任务数: {submitted_count}")
        print(f"   待处理资产数: {len(pending_assets)}")
        if submitted_count > 0:
            print(f"   ✅ 断点续传模式：跳过已提交的 {submitted_count} 个资产")
        filtered_assets = pending_assets

    total_assets = len(filtered_assets)

    # 批次处理
    if CONFIG['enableBatchMode']:
        start_idx = CONFIG['batchStart']
        end_idx = CONFIG['batchEnd'] if CONFIG['batchEnd'] != -1 else total_assets
        if CONFIG['batchSize'] > 0:
            end_idx = min(start_idx + CONFIG['batchSize'], total_assets)
    else:
        start_idx = 0
        end_idx = total_assets

    batch_assets = filtered_assets[start_idx:end_idx]

    print(f"   总资产数: {total_assets}")
    print(f"   当前批次: 第 {start_idx + 1} 到第 {end_idx} 个")
    print(f"   本批数量: {len(batch_assets)}")
    print("=" * 60)

    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for i, asset in enumerate(batch_assets):
        try:
            asset_id = asset['id']
            asset_name = asset['name']
            asset_type = asset['type']

            print(f"\n{'='*60}")
            print(f"🔄 处理资产 {i+1}/{len(batch_assets)} (总进度: {start_idx + i + 1}/{total_assets})")
            print(f"   📦 资产名称: {asset_name}")
            print(f"   🆔 资产ID: {asset_id}")
            print(f"   📂 资产类型: {asset_type}")

            # 加载资产为特征集合
            try:
                vector_collection = ee.FeatureCollection(asset_id)
                asset_region = vector_collection.geometry()
                feature_count = vector_collection.size().getInfo()
                print(f"   📊 要素数量: {feature_count}")
            except Exception as e:
                print(f"   ❌ 加载资产失败: {e}")
                failed += 1
                continue

            # 获取影像合成
            composite_image = get_sentinel2_composite(asset_region)
            if composite_image is None:
                print(f"   ⚠️ 无可用影像，跳过")
                skipped += 1
                continue

            # 提交导出任务
            _, task_id = submit_export_task(composite_image, asset_region, asset_name, i, len(batch_assets), mode="asset")

            # 记录任务状态
            if task_manager and CONFIG['enableTaskTracking']:
                item_key = asset_name
                task_manager.add_task(
                    item_key=item_key,
                    task_id=task_id,
                    asset_name=asset_name,
                    extra_info={
                        'asset_id': asset_id,
                        'asset_type': asset_type,
                        'feature_count': feature_count
                    }
                )
                print(f"   💾 已记录任务状态: {item_key}")

            successful += 1

            # 显示进度和预估时间
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(batch_assets) - i - 1)
            print(f"   ⏱️  已用时: {elapsed/60:.1f}分钟, 预计剩余: {remaining/60:.1f}分钟")

            # 资产间延迟
            if i < len(batch_assets) - 1:
                delay = CONFIG["assetDelay"]
                print(f"   💤 等待 {delay} 秒... (剩余 {len(batch_assets) - i - 1} 个资产)")
                time.sleep(delay)

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            failed += 1
            continue

    return {
        'successful': successful,
        'failed': failed,
        'skipped': skipped,
        'total_time': time.time() - start_time,
        'total_items': total_assets,
        'end_idx': end_idx
    }


# ==========================================================
# 按要素下载
# ==========================================================
def process_by_feature(filtered_assets, task_manager=None):
    """按要素模式批量下载"""
    print("\n" + "=" * 60)
    print("📋 按要素模式下载")
    print("=" * 60)

    # 第一步：收集所有资产的所有要素
    all_features = []

    for asset in filtered_assets:
        asset_id = asset['id']
        asset_name = asset['name']

        print(f"\n🔍 处理资产: {asset_name}")

        try:
            vector_collection = ee.FeatureCollection(asset_id)

            # 检测ID字段
            id_field = detect_feature_id_field(vector_collection)
            if not id_field:
                print(f"   ❌ 未找到ID字段，跳过此资产")
                continue

            # 提取要素ID列表
            feature_ids = extract_feature_ids(vector_collection, id_field)
            if not feature_ids:
                print(f"   ❌ 未提取到要素ID，跳过此资产")
                continue

            # 将要素添加到列表
            for fid in feature_ids:
                all_features.append({
                    'asset_id': asset_id,
                    'asset_name': asset_name,
                    'feature_id': fid,
                    'id_field': id_field,
                    'collection': vector_collection
                })

        except Exception as e:
            print(f"   ❌ 处理资产失败: {e}")
            continue

    if not all_features:
        print("\n❌ 没有找到任何要素")
        return None

    total_features = len(all_features)
    print(f"\n✅ 共收集到 {total_features} 个要素")

    # 如果启用了任务跟踪，过滤出未提交的要素
    if task_manager and CONFIG['enableTaskTracking'] and CONFIG['autoResume']:
        pending_features = task_manager.get_pending_items(all_features)
        submitted_count = task_manager.get_submitted_count()
        print(f"\n📊 任务状态统计:")
        print(f"   总要素数: {total_features}")
        print(f"   已提交: {submitted_count}")
        print(f"   待提交: {len(pending_features)}")
        if submitted_count > 0:
            print(f"   ✅ 断点续传模式：跳过已提交的 {submitted_count} 个要素")
        all_features = pending_features
        total_features = len(all_features)

    # 批次处理
    if CONFIG['enableBatchMode']:
        start_idx = CONFIG['batchStart']
        end_idx = CONFIG['batchEnd'] if CONFIG['batchEnd'] != -1 else total_features
        if CONFIG['batchSize'] > 0:
            end_idx = min(start_idx + CONFIG['batchSize'], total_features)
    else:
        start_idx = 0
        end_idx = total_features

    batch_features = all_features[start_idx:end_idx]

    print("\n" + "=" * 60)
    print("📋 批次处理信息")
    print("=" * 60)
    print(f"   总要素数: {total_features}")
    print(f"   当前批次: 第 {start_idx + 1} 到第 {end_idx} 个")
    print(f"   本批数量: {len(batch_features)}")
    print("=" * 60)

    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for i, feature_info in enumerate(batch_features):
        try:
            asset_name = feature_info['asset_name']
            feature_id = feature_info['feature_id']
            id_field = feature_info['id_field']
            collection = feature_info['collection']

            print(f"\n{'='*60}")
            print(f"🔄 处理要素 {i+1}/{len(batch_features)} (总进度: {start_idx + i + 1}/{total_features})")
            print(f"   📦 资产: {asset_name}")
            print(f"   🆔 要素ID: {feature_id}")

            # 筛选单个要素
            try:
                single_feature = collection.filter(ee.Filter.eq(id_field, feature_id))
                feature_region = single_feature.geometry()
            except Exception as e:
                print(f"   ❌ 加载要素失败: {e}")
                failed += 1
                continue

            # 获取影像合成
            composite_image = get_sentinel2_composite(feature_region)
            if composite_image is None:
                print(f"   ⚠️ 无可用影像，跳过")
                skipped += 1
                continue

            # 生成文件名（资产名_要素ID）
            file_name = f"{asset_name}_{feature_id}"

            # 提交导出任务
            _, task_id = submit_export_task(composite_image, feature_region, file_name, i, len(batch_features), mode="feature")

            # 记录任务状态（关键：每个要素对应一个任务ID）
            if task_manager and CONFIG['enableTaskTracking']:
                item_key = f"{asset_name}_{feature_id}"
                task_manager.add_task(
                    item_key=item_key,
                    task_id=task_id,
                    asset_name=asset_name,
                    feature_id=str(feature_id),
                    extra_info={
                        'asset_id': feature_info['asset_id'],
                        'id_field': id_field,
                        'file_name': file_name
                    }
                )
                print(f"   💾 已记录任务状态: {item_key} -> 任务ID: {task_id}")

            successful += 1

            # 显示进度和预估时间
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(batch_features) - i - 1)
            print(f"   ⏱️  已用时: {elapsed/60:.1f}分钟, 预计剩余: {remaining/60:.1f}分钟")

            # 要素间延迟
            if i < len(batch_features) - 1:
                delay = CONFIG["featureDelay"]
                print(f"   💤 等待 {delay} 秒... (剩余 {len(batch_features) - i - 1} 个要素)")
                time.sleep(delay)

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            failed += 1
            continue

    return {
        'successful': successful,
        'failed': failed,
        'skipped': skipped,
        'total_time': time.time() - start_time,
        'total_items': total_features,
        'end_idx': end_idx
    }


# ==========================================================
# 主函数
# ==========================================================
def main():
    """主函数"""
    print("=" * 60)
    print("🚀 GEE资产批量影像下载工具（增强版）")
    print("=" * 60)
    print(f"⚙️  配置: 最大并发任务={CONFIG['maxConcurrentTasks']}, 检查间隔={CONFIG['checkInterval']}秒")
    print(f"⚙️  任务阈值: {CONFIG['safeTaskThreshold']} (项目最大3000)")
    if CONFIG['enableTaskTracking']:
        print(f"⚙️  任务跟踪: 已启用 (状态文件: {CONFIG['taskStateFile']})")
    print("=" * 60)

    # 1. 初始化
    init_ee()

    # 2. 初始化任务状态管理器
    task_manager = None
    if CONFIG['enableTaskTracking']:
        print("\n📊 步骤1: 初始化任务状态管理器")
        task_manager = TaskStateManager(CONFIG['taskStateFile'])
        stats = task_manager.get_statistics()
        if stats['total'] > 0:
            print(f"   已记录任务数: {stats['total']}")
            print(f"   状态分布: {stats['status_breakdown']}")


    # 3. 读取资产列表 JSON
    print("\n📂 步骤2: 读取资产列表 JSON 文件")
    assets_data = load_asset_list(CONFIG['assetListJson'])
    if not assets_data:
        print("❌ 加载资产列表失败")
        return

    # 3.5 用户确认配置并保存
    print("\n🛠️ 步骤3: 配置确认与保存")
    config_to_save = CONFIG.copy()
    print("当前配置参数如下：")
    for k, v in config_to_save.items():
        print(f"  {k}: {v}")

    confirm = input("是否需要修改配置？(y/N): ").strip().lower()
    if confirm == 'y':
        for k in config_to_save:
            new_val = input(f"  {k} (当前: {config_to_save[k]}，直接回车跳过): ").strip()
            if new_val:
                # 自动类型转换
                if isinstance(config_to_save[k], int):
                    try:
                        config_to_save[k] = int(new_val)
                    except:
                        print(f"  ⚠️ 输入无效，保持原值")
                elif isinstance(config_to_save[k], float):
                    try:
                        config_to_save[k] = float(new_val)
                    except:
                        print(f"  ⚠️ 输入无效，保持原值")
                elif isinstance(config_to_save[k], list):
                    try:
                        config_to_save[k] = json.loads(new_val)
                    except:
                        config_to_save[k] = [v.strip() for v in new_val.split(',') if v.strip()]
                else:
                    config_to_save[k] = new_val
    # 保存配置到 json 文件
    config_save_path = os.path.join(os.path.dirname(CONFIG['assetListJson']), 'gee_download_config.json')
    try:
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        print(f"✅ 已保存配置到: {config_save_path}")
    except Exception as e:
        print(f"❌ 配置保存失败: {e}")

    # 4. 筛选资产
    print("\n🔍 步骤4: 筛选资产")
    filtered_assets = filter_assets(assets_data)
    if not filtered_assets:
        print("⚠️ 没有符合条件的资产")
        return

    # 5. 检查现有任务状态
    print("\n📊 步骤5: 检查现有任务状态...")
    initial_stats = get_task_stats()
    if initial_stats:
        print(f"   已有任务: READY={initial_stats['READY']}, RUNNING={initial_stats['RUNNING']}, " +
              f"COMPLETED={initial_stats['COMPLETED']}, FAILED={initial_stats['FAILED']}")

    # 6. 执行按资产下载
    print("\n📋 步骤6: 开始批量下载（仅资产模式）")
    result = process_by_asset(filtered_assets, task_manager)

    if result is None:
        print("\n❌ 下载过程失败")
        return


    # 7. 最终统计
    print("\n" + "=" * 60)
    print("✅ 任务提交完成！")
    print("=" * 60)

    # 显示任务跟踪统计
    if task_manager and CONFIG['enableTaskTracking']:
        track_stats = task_manager.get_statistics()
        print(f"\n📝 任务跟踪统计:")
        print(f"   总记录数: {track_stats['total']}")
        print(f"   状态文件: {CONFIG['taskStateFile']}")

    final_stats = get_task_stats()
    if final_stats:
        print("\n📊 GEE任务队列状态:")
        print(f"   🟡 READY (等待运行): {final_stats['READY']}")
        print(f"   🔵 RUNNING (正在运行): {final_stats['RUNNING']}")
        print(f"   � 总计: {final_stats['READY'] + final_stats['RUNNING']}")

    print(f"\n📋 本次提交统计:")
    print(f"   总资产数: {result['total_items']}")
    print(f"   成功提交: {result['successful']}")
    print(f"   无影像跳过: {result['skipped']}")
    print(f"   失败: {result['failed']}")

    if CONFIG['enableBatchMode'] and result['end_idx'] < result['total_items']:
        print(f"\n📌 批次处理提示:")
        print(f"   本批已完成，还有 {result['total_items'] - result['end_idx']} 个资产待处理")
        print(f"   要继续下一批，请修改配置:")
        print(f"   CONFIG['batchStart'] = {result['end_idx']}")
        print(f"   然后重新运行脚本")

    print(f"\n⏱️  总耗时: {result['total_time']/60:.1f} 分钟")

    print("\n📌 后续操作:")
    print("   1. 前往 https://code.earthengine.google.com/tasks")
    print("   2. 查看任务列表运行状态或点击 [RUN] 启动任务")
    print("   3. 任务完成后在 Google Drive 查看结果")
    print(f"   4. 结果保存在文件夹: {CONFIG['exportFolder']}")
    if CONFIG['enableTaskTracking']:
        print(f"   5. 任务状态已保存到: {CONFIG['taskStateFile']}")
        print("   6. 如中途中断，重新运行脚本将自动从断点继续")
    print("=" * 60)

# ==========================================================
# 入口
# ==========================================================
if __name__ == "__main__":
    main()
