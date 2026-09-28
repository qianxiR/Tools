// ========== 全局配置（修改此处即可切换研究区域和时间） ==========
/**
 * 全局区域变量 - 扎陵湖
 * 说明：扎陵湖位于青海省玛多县，黄河源头地区
 * 经纬度范围：经度 97.0°E - 97.5°E，纬度 34.8°N - 35.1°N
 * 
 * 使用示例：
 * - 行政区划：var TARGET_REGION = china_city.filter(ee.Filter.eq('name', '玉树藏族自治州'));
 * - 行政区划：var TARGET_REGION = china_county.filter(ee.Filter.eq('name', '某县名'));
 * - 自定义矩形：var TARGET_REGION = ee.Geometry.Rectangle([minLon, minLat, maxLon, maxLat]);
 */
var TARGET_REGION = ee.Geometry.Rectangle([97.0, 34.8, 97.5, 35.1]);
var REGION_DISPLAY_NAME = '扎陵湖';  // 地图显示名称

/**
 * 全局时间配置
 * 说明：修改这些变量即可切换查看的时间范围
 */
var START_DATE = '2025-06-01';  // 开始日期，格式：YYYY-MM-DD
var END_DATE = '2025-07-31';    // 结束日期，格式：YYYY-MM-DD
// ====================================================================

// 1. 定义研究区域
// 对于Geometry对象，直接添加到地图，无需style()方法
Map.addLayer(TARGET_REGION, {color: 'ffff00'}, REGION_DISPLAY_NAME);

Map.centerObject(TARGET_REGION, 10);

var new_clip_region = TARGET_REGION;

// 2. 去云算法函数
/**
 * 获取QA位信息
 * 入参:
 * - image (ee.Image): 包含QA波段的影像
 * - start (number): 起始位位置
 * - end (number): 结束位位置
 * 方法:
 * - 从QA波段中提取指定位范围的信息
 * 出参:
 * - ee.Image: 提取的QA位信息
 */
function getQABits(image, start, end) {
    var pattern = 0;
    for (var i = start; i <= end; i++) {
       pattern += 1 << i;
    }
    // Return a single band image of the extracted QA bits
    return image.bitwiseAnd(pattern).rightShift(start);
}

/**
 * 哨兵-2去云算法
 * 入参:
 * - image (ee.Image): 哨兵-2影像
 * 方法:
 * - 使用QA60波段进行云掩膜
 * - 对原始波段应用云掩膜
 * 出参:
 * - ee.Image: 去云后的原始哨兵-2影像
 */
function rmcloudS2(image){
  // 获取QA60波段（云掩膜）
  var qa = image.select('QA60');
  
  // 提取云掩膜位 (位10: 云, 位11: 云阴影)
  var cloudMask = qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0));
  
  // 对原始影像应用云掩膜
  return image.updateMask(cloudMask);
}

// 3. 获取 Sentinel-2 数据（使用全局配置的时间范围）
/**
 * 获取哨兵-2 SR数据并去云
 * 说明：median()会保留所有波段，然后通过select()只选择B2-B12光谱波段
 * - 光谱波段：B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12（除B10，SR产品中不存在）
 * - 不包含辅助波段：AOT（气溶胶光学厚度）, WVP（水汽）, SCL（场景分类）, TCI等
 */
var S2_2025 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(new_clip_region)
  .filterDate(START_DATE, END_DATE)
  .map(rmcloudS2);

// 4. 计算影像中值合成，只保留B2-B12光谱波段
var composite_2025 = S2_2025.median()
  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
  .clip(TARGET_REGION);

// 5. 可视化合成影像
Map.addLayer(composite_2025, {min: 0, max: 3000, bands: ['B4', 'B3', 'B2']}, 'Sentinel-2 Composite 2025');

// 6. 显示影像统计信息
print('========== 影像统计信息 ==========');
print('哨兵-2影像数量:', S2_2025.size());
print('影像波段信息:', composite_2025.bandNames());
print('影像投影信息:', composite_2025.projection());

// 8. 计算影像覆盖度分析
/**
 * 覆盖度分析说明：
 * - count()：统计每个像素在时间序列中被有效观测的次数
 * - 值为0：该像素在整个时间段内无有效观测（完全被云遮挡或无影像覆盖）
 * - 值≥1：该像素被观测的次数，次数越多说明该位置影像质量越好
 */
var coverage = S2_2025.select('B4').count().clip(TARGET_REGION);

// 可视化覆盖度图层（颜色：蓝色=0次，橙色=低覆盖，黄色=中等，绿色=高覆盖）
Map.addLayer(coverage, {min: 0, max: 10, palette: ['blue', 'orange', 'yellow', 'lightgreen', 'green']}, 
             '影像覆盖度 (观测次数)');

// 打印简要信息
print('========== 覆盖度信息 ==========');
print('说明: 地图上蓝色=0次观测，橙色=低覆盖，黄色=中等，绿色=高覆盖');
print('====================================');

// 计算影像统计信息
var stats = composite_2025.select(['B4', 'B3', 'B2']).reduceRegion({
  reducer: ee.Reducer.minMax().combine({
    reducer2: ee.Reducer.mean(),
    sharedInputs: true
  }),
  geometry: new_clip_region,
  scale: 30,
  maxPixels: 1e9
});

print('========== 合成影像统计信息 ==========');
print('影像统计信息:', stats);