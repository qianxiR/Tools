"""
Google Earth Engine 数据下载工具 - Python 版本
基于原 JavaScript 版本转换

作者：[您的姓名]
功能：
- 加载 Sentinel-2 / Landsat 影像
- 按省份/城市筛选研究区域
- 自动去云处理
- 影像覆盖度分析
- 导出数据到 Google Drive
"""

import ee
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class GEEDataDownloader:
    """Google Earth Engine 数据下载工具类"""
    
    def __init__(self, project_id: str = 'applied-pipe-453411-k9'):
        """
        初始化下载器
        
        参数:
            project_id: GEE 项目 ID
        """
        try:
            ee.Initialize(project=project_id)
            print("✓ Earth Engine 初始化成功")
        except Exception as e:
            print(f"✗ Earth Engine 初始化失败: {str(e)}")
            raise
        
        # 加载中国行政区划数据
        self.china_provinces = ee.FeatureCollection(f"projects/{project_id}/assets/china_provinces")
        self.china_city = ee.FeatureCollection(f"projects/{project_id}/assets/china_city")
        self.china_county = ee.FeatureCollection(f"projects/{project_id}/assets/china_county")
        
        # 支持的影像类型
        self.image_types = {
            'S2_SR': 'COPERNICUS/S2_SR_HARMONIZED',
            'S2_TOA': 'COPERNICUS/S2_HARMONIZED',
            'CLOUD_SCORE': 'CLOUD_SCORE_PLUS_HARMONIZED/S2_HARMONIZED_V1',
            'L5_TOA': 'LANDSAT/LT05/C02/T1_TOA',
            'L7_TOA': 'LANDSAT/LE07/C02/T1_TOA',
            'L8_TOA': 'LANDSAT/LC08/C02/T1_TOA',
            'L8_RAW': 'LANDSAT/LC08/C02/T1'
        }
        
        # 波段配置
        self.band_config = {
            'S2_SR': {
                'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
                'scale': 10,
                'rgb': ['B4', 'B3', 'B2'],
                'vis': {'min': 0, 'max': 3000}
            },
            'S2_TOA': {
                'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'],
                'scale': 10,
                'rgb': ['B4', 'B3', 'B2'],
                'vis': {'min': 0, 'max': 3000}
            },
            'L5_TOA': {
                'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                'scale': 30,
                'rgb': ['B3', 'B2', 'B1'],
                'vis': {'min': 0, 'max': 0.4}
            },
            'L7_TOA': {
                'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                'scale': 30,
                'rgb': ['B3', 'B2', 'B1'],
                'vis': {'min': 0, 'max': 0.4}
            },
            'L8_TOA': {
                'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11'],
                'scale': 30,
                'rgb': ['B4', 'B3', 'B2'],
                'vis': {'min': 0, 'max': 0.4}
            }
        }
    
    def get_region_by_name(self, region_name: str, region_type: str = 'province') -> Optional[ee.Geometry]:
        """
        根据名称获取区域几何体
        
        参数:
            region_name: 省份或城市名称
            region_type: 'province' 或 'city'
        
        返回:
            ee.Geometry: 区域几何体
        """
        try:
            if region_type == 'province':
                feature = self.china_provinces.filter(ee.Filter.eq('name', region_name)).first()
            elif region_type == 'city':
                feature = self.china_city.filter(ee.Filter.eq('name', region_name)).first()
            else:
                raise ValueError(f"不支持的区域类型: {region_type}")
            
            if feature is None:
                print(f"✗ 未找到区域: {region_name}")
                return None
            
            return feature.geometry()
        
        except Exception as e:
            print(f"✗ 获取区域失败: {str(e)}")
            return None
    
    def load_image_collection(
        self,
        image_type: str,
        start_date: str,
        end_date: str,
        region: ee.Geometry,
        cloud_threshold: int = 20
    ) -> ee.Image:
        """
        加载影像集合并进行预处理
        
        参数:
            image_type: 影像类型 (S2_SR, L8_TOA 等)
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            region: 研究区域
            cloud_threshold: 云量阈值 (%)
        
        返回:
            ee.Image: 合成后的影像
        """
        try:
            # 获取影像集合ID
            collection_id = self.image_types.get(image_type)
            if not collection_id:
                raise ValueError(f"不支持的影像类型: {image_type}")
            
            # 获取波段配置
            config = self.band_config.get(image_type, {})
            bands = config.get('bands')
            
            # 加载影像集合
            collection = ee.ImageCollection(collection_id).filterDate(start_date, end_date).filterBounds(region)
            
            # 云量过滤
            if 'S2' in image_type:
                collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
            elif 'L' in image_type:
                collection = collection.filter(ee.Filter.lt('CLOUD_COVER', cloud_threshold))
            
            # 波段选择和裁剪
            if bands:
                collection = collection.select(bands)
            
            # 中值合成
            composite = collection.median().clip(region)
            
            print(f"✓ 加载影像成功")
            print(f"  影像类型: {image_type}")
            print(f"  影像数量: {collection.size().getInfo()}")
            print(f"  时间范围: {start_date} 到 {end_date}")
            
            return composite
        
        except Exception as e:
            print(f"✗ 加载影像失败: {str(e)}")
            raise
    
    def calculate_coverage(
        self,
        image_type: str,
        start_date: str,
        end_date: str,
        region: ee.Geometry,
        cloud_threshold: int = 20
    ) -> ee.Image:
        """
        计算影像覆盖度
        
        参数:
            image_type: 影像类型
            start_date: 开始日期
            end_date: 结束日期
            region: 研究区域
            cloud_threshold: 云量阈值
        
        返回:
            ee.Image: 覆盖度影像（像素值=观测次数）
        """
        try:
            collection_id = self.image_types.get(image_type)
            collection = ee.ImageCollection(collection_id).filterDate(start_date, end_date).filterBounds(region)
            
            # 云量过滤
            if 'S2' in image_type:
                collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                coverage_band = 'B4'
            elif 'L' in image_type:
                collection = collection.filter(ee.Filter.lt('CLOUD_COVER', cloud_threshold))
                coverage_band = 'B3'
            else:
                coverage_band = None
            
            # 计算覆盖度
            if coverage_band:
                coverage = collection.select(coverage_band).count().clip(region)
            else:
                coverage = collection.count().clip(region)
            
            print(f"✓ 覆盖度计算完成")
            
            return coverage
        
        except Exception as e:
            print(f"✗ 覆盖度计算失败: {str(e)}")
            raise
    
    def export_image(
        self,
        image: ee.Image,
        region: ee.Geometry,
        description: str,
        folder: str = "GEE_Exports",
        scale: int = 10,
        crs: str = 'EPSG:4326'
    ) -> ee.batch.Task:
        """
        导出影像到 Google Drive
        
        参数:
            image: 要导出的影像
            region: 研究区域
            description: 导出任务描述
            folder: Google Drive 文件夹
            scale: 分辨率(米)
            crs: 坐标系
        
        返回:
            ee.batch.Task: 导出任务对象
        """
        try:
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=description,
                folder=folder,
                scale=scale,
                region=region,
                crs=crs,
                fileFormat='GeoTIFF',
                formatOptions={'cloudOptimized': True},
                maxPixels=1e13
            )
            
            task.start()
            
            print("=" * 60)
            print("✓ 导出任务已提交")
            print(f"  任务名称: {description}")
            print(f"  导出文件夹: {folder}")
            print(f"  坐标系: {crs}")
            print(f"  分辨率: {scale} 米")
            print(f"  任务状态: {task.status()['state']}")
            print("=" * 60)
            print("请在 GEE Code Editor 的 Tasks 面板中点击 'Run' 按钮启动下载")
            
            return task
        
        except Exception as e:
            print(f"✗ 导出任务失败: {str(e)}")
            raise
    
    def download_data(
        self,
        image_type: str,
        start_date: str,
        end_date: str,
        region_name: str,
        region_type: str = 'province',
        cloud_threshold: int = 20,
        export: bool = True
    ) -> Dict[str, Any]:
        """
        完整的数据下载流程
        
        参数:
            image_type: 影像类型
            start_date: 开始日期
            end_date: 结束日期
            region_name: 区域名称
            region_type: 区域类型
            cloud_threshold: 云量阈值
            export: 是否导出到 Google Drive
        
        返回:
            Dict: 包含影像和统计信息的字典
        """
        print("\n" + "=" * 60)
        print(f"  Google Earth Engine 数据下载")
        print("=" * 60)
        
        # 1. 获取区域
        region = self.get_region_by_name(region_name, region_type)
        if region is None:
            return {'success': False, 'message': '区域获取失败'}
        
        # 2. 加载影像
        composite = self.load_image_collection(
            image_type, start_date, end_date, region, cloud_threshold
        )
        
        # 3. 计算覆盖度
        coverage = self.calculate_coverage(
            image_type, start_date, end_date, region, cloud_threshold
        )
        
        # 4. 导出数据
        task = None
        if export:
            config = self.band_config.get(image_type, {})
            scale = config.get('scale', 30)
            
            description = f"{image_type}_{region_name}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
            task = self.export_image(composite, region, description, scale=scale)
        
        # 5. 返回结果
        return {
            'success': True,
            'data': {
                'composite': composite,
                'coverage': coverage,
                'region': region,
                'task': task
            },
            'metadata': {
                'image_type': image_type,
                'region_name': region_name,
                'start_date': start_date,
                'end_date': end_date,
                'cloud_threshold': cloud_threshold
            }
        }


# ========== 使用示例 ==========
def main():
    """主函数 - 演示如何使用下载器"""
    
    # 初始化下载器
    downloader = GEEDataDownloader(project_id='applied-pipe-453411-k9')
    
    # 配置参数
    params = {
        'image_type': 'S2_SR',  # Sentinel-2 地表反射率
        'start_date': '2025-06-01',
        'end_date': '2025-07-31',
        'region_name': '青海',  # 省份名称
        'region_type': 'province',
        'cloud_threshold': 20,
        'export': True  # 是否导出到 Google Drive
    }
    
    # 执行下载
    result = downloader.download_data(**params)
    
    if result['success']:
        print("\n✓ 数据处理完成！")
        print(f"影像类型: {result['metadata']['image_type']}")
        print(f"研究区域: {result['metadata']['region_name']}")
        print(f"时间范围: {result['metadata']['start_date']} 到 {result['metadata']['end_date']}")
        
        # 获取任务状态
        if result['data']['task']:
            task_status = result['data']['task'].status()
            print(f"\n导出任务状态: {task_status['state']}")
            print(f"任务 ID: {task_status['id']}")
    else:
        print(f"\n✗ 数据处理失败: {result['message']}")


if __name__ == "__main__":
    main()
