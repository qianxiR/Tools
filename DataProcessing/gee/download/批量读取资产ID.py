"""
========== 批量读取 GEE 目录下的所有资产 ID ==========
功能说明：
- 读取指定 GEE 项目目录下的所有资产
- 导出资产列表为 JSON 格式
- 支持递归读取子目录
- 支持按资产类型筛选

使用方法：
1. 配置下方的参数
2. 运行脚本: python 批量读取资产ID.py
3. 查看生成的 JSON 文件
========================================================
"""

import ee
import json
import os
from datetime import datetime

# ==================== 配置参数 ====================
CONFIG = {
    # 【1. 项目目录配置】
    # GEE 项目资产根目录
    "asset_folder": "projects/applied-pipe-453411-k9/assets/ys",

    # 【2. 筛选配置】
    # 资产类型过滤（可选）
    # 可选值: "IMAGE", "IMAGE_COLLECTION", "TABLE", "FOLDER", None(不过滤)
    "asset_type_filter": None,

    # 【3. 导出配置】
    "output_file": r"E:\1代码\模型\gee\数据下载\矢量遍历下载数据\gee_assets_list.json",  # 输出文件名

    # 【4. 递归配置】
    "recursive": True,  # 是否递归读取子目录

    # 【5. 详细信息配置】
    "include_details": True,  # 是否包含资产详细信息（大小、创建时间等）
}
# ==================================================


def load_config_from_file(config_file="asset_list_config.json"):
    """
    从配置文件加载配置（可选）

    Args:
        config_file (str): 配置文件路径

    Returns:
        dict: 配置字典
    """
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'config' in data:
                    print(f"✓ 从配置文件加载: {config_file}")
                    return data['config']
        except Exception as e:
            print(f"⚠️  读取配置文件失败: {e}，使用默认配置")

    return CONFIG


def initialize_ee():
    """初始化 Earth Engine"""
    print("==================== 初始化 Earth Engine ====================")
    try:
        ee.Initialize()
        print("✓ Earth Engine 已成功初始化")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"❌ Earth Engine 初始化失败: {e}")
        print("请先运行认证: ee.Authenticate()")
        return False


def list_assets(folder_path, recursive=True, asset_type_filter=None):
    """
    列出指定目录下的所有资产

    Args:
        folder_path (str): GEE 资产文件夹路径
        recursive (bool): 是否递归读取子目录
        asset_type_filter (str): 资产类型过滤，可选值: IMAGE, IMAGE_COLLECTION, TABLE, FOLDER

    Returns:
        list: 资产列表
    """
    print(f"\n正在读取目录: {folder_path}")

    assets = []

    try:
        # 使用 ee.data.listAssets() 列出资产
        asset_list = ee.data.listAssets({'parent': folder_path})

        if 'assets' not in asset_list:
            print(f"  目录为空或不存在")
            return assets

        for asset in asset_list['assets']:
            asset_id = asset['name']
            asset_type = asset['type']

            # 类型过滤
            if asset_type_filter and asset_type != asset_type_filter:
                continue

            # 提取资产信息
            asset_info = {
                'id': asset_id,
                'type': asset_type,
                'name': asset_id.split('/')[-1]  # 提取资产名称
            }

            # 添加详细信息
            if CONFIG['include_details']:
                if 'sizeBytes' in asset:
                    asset_info['size_bytes'] = asset['sizeBytes']
                    asset_info['size_mb'] = round(asset['sizeBytes'] / (1024 * 1024), 2)

                if 'updateTime' in asset:
                    asset_info['update_time'] = asset['updateTime']

            assets.append(asset_info)
            print(f"  ✓ [{asset_type}] {asset_info['name']}")

            # 如果是文件夹且启用递归，继续读取
            if recursive and asset_type == 'FOLDER':
                sub_assets = list_assets(asset_id, recursive, asset_type_filter)
                assets.extend(sub_assets)

        print(f"  共找到 {len(asset_list['assets'])} 个资产")

    except ee.EEException as e:
        print(f"❌ 读取目录失败: {e}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

    return assets


def get_asset_statistics(assets):
    """
    统计资产信息

    Args:
        assets (list): 资产列表

    Returns:
        dict: 统计信息
    """
    stats = {
        'total_count': len(assets),
        'by_type': {}
    }

    total_size_bytes = 0

    for asset in assets:
        asset_type = asset['type']

        # 按类型统计
        if asset_type not in stats['by_type']:
            stats['by_type'][asset_type] = {
                'count': 0,
                'size_bytes': 0
            }

        stats['by_type'][asset_type]['count'] += 1

        # 累计大小
        if 'size_bytes' in asset:
            stats['by_type'][asset_type]['size_bytes'] += asset['size_bytes']
            total_size_bytes += asset['size_bytes']

    # 计算总大小
    stats['total_size_bytes'] = total_size_bytes
    stats['total_size_mb'] = round(total_size_bytes / (1024 * 1024), 2)
    stats['total_size_gb'] = round(total_size_bytes / (1024 * 1024 * 1024), 2)

    # 为每种类型添加 MB 单位
    for asset_type in stats['by_type']:
        size_bytes = stats['by_type'][asset_type]['size_bytes']
        stats['by_type'][asset_type]['size_mb'] = round(size_bytes / (1024 * 1024), 2)
        stats['by_type'][asset_type]['size_gb'] = round(size_bytes / (1024 * 1024 * 1024), 2)

    return stats


def save_to_json(data, filename):
    """
    保存数据为 JSON 文件

    Args:
        data (dict): 要保存的数据
        filename (str): 输出文件名
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 数据已保存到: {filename}")
    except Exception as e:
        print(f"\n❌ 保存文件失败: {e}")


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("资产统计信息".center(60))
    print("=" * 60)
    print(f"总资产数量: {stats['total_count']}")
    print(f"总占用空间: {stats['total_size_gb']:.2f} GB ({stats['total_size_mb']:.2f} MB)")
    print("\n按类型统计:")

    for asset_type, type_stats in stats['by_type'].items():
        print(f"  [{asset_type}]")
        print(f"    - 数量: {type_stats['count']}")
        if type_stats['size_bytes'] > 0:
            print(f"    - 大小: {type_stats['size_gb']:.2f} GB ({type_stats['size_mb']:.2f} MB)")

    print("=" * 60)


def main():
    """主函数"""
    # 加载配置
    global CONFIG
    CONFIG = load_config_from_file()

    print("========== GEE 资产列表提取工具 ==========")
    print(f"目标目录: {CONFIG['asset_folder']}")
    print(f"递归模式: {'启用' if CONFIG['recursive'] else '禁用'}")
    print(f"类型过滤: {CONFIG['asset_type_filter'] or '无'}")
    print(f"输出文件: {CONFIG['output_file']}")
    print("=" * 60)

    # 初始化 EE
    if not initialize_ee():
        return

    # 读取资产列表
    print("\n==================== 开始读取资产 ====================")
    assets = list_assets(
        CONFIG['asset_folder'],
        recursive=CONFIG['recursive'],
        asset_type_filter=CONFIG['asset_type_filter']
    )

    if not assets:
        print("\n⚠️  未找到任何资产")
        return

    # 统计信息
    stats = get_asset_statistics(assets)
    print_statistics(stats)

    # 构建输出数据
    output_data = {
        'metadata': {
            'folder_path': CONFIG['asset_folder'],
            'extract_time': datetime.now().isoformat(),
            'recursive': CONFIG['recursive'],
            'asset_type_filter': CONFIG['asset_type_filter'],
            'total_count': stats['total_count']
        },
        'statistics': stats,
        'assets': assets
    }

    # 保存为 JSON
    save_to_json(output_data, CONFIG['output_file'])

    # 打印预览
    print("\n==================== 资产列表预览 (前10条) ====================")
    for i, asset in enumerate(assets[:10], 1):
        size_info = f" ({asset['size_mb']} MB)" if 'size_mb' in asset else ""
        print(f"{i}. [{asset['type']}] {asset['name']}{size_info}")

    if len(assets) > 10:
        print(f"... (还有 {len(assets) - 10} 个资产)")

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
