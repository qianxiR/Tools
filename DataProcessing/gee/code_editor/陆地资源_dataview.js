// ========== 全局配置（修改此处即可切换研究区域和时间） ==========
/**
 * 全局区域变量
 * 说明：修改此变量即可切换研究区域，后续代码会自动使用此区域
 * 
 * 使用示例：
 * - 行政区划：var TARGET_REGION = china_city.filter(ee.Filter.eq('name', '玉树藏族自治州'));
 * - 行政区划：var TARGET_REGION = china_county.filter(ee.Filter.eq('name', '某县名'));
 * - 自定义资产：var TARGET_REGION = jipocuo;
 */
var TARGET_REGION = china_city.filter(ee.Filter.eq('name', '玉树藏族自治州'));
var REGION_DISPLAY_NAME = '玉树藏族自治州';  // 地图显示名称

/**
 * 全局时间配置
 * 说明：修改这些变量即可切换查看的时间范围
 */
var START_DATE = '2025-06-01';  // 开始日期，格式：YYYY-MM-DD
var END_DATE = '2025-07-31';    // 结束日期，格式：YYYY-MM-DD
// ====================================================================

// 1. 定义研究区域
Map.addLayer(TARGET_REGION.style({fillColor:'00000000',color:'ffff00'}), {}, REGION_DISPLAY_NAME);

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
 * Landsat去云算法
 * 入参:
 * - image (ee.Image): Landsat影像
 * 方法:
 * - 使用QA_PIXEL波段进行云掩膜
 * - 对原始波段应用云掩膜
 * 出参:
 * - ee.Image: 去云后的原始Landsat影像
 */
function rmL5789Cloud(image) { 
  var cloudsBitMask = (1 << 3); 
  var cloudShadowBitMask = (1 << 4); 
  var qa = image.select('QA_PIXEL'); 
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) 
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0)); 
  return image.updateMask(mask); 
}

// 3. 定义Landsat影像集合
var L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA");
var L8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA");
var L7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA");
var L5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA");

// 4. 获取Landsat数据（使用全局配置的时间范围）
// 获取Landsat-9数据，应用去云算法
var L9_2025 = L9.filterBounds(new_clip_region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUD_COVER', 20))
  .map(rmL5789Cloud);

// 获取Landsat-8数据，应用去云算法
var L8_2025 = L8.filterBounds(new_clip_region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUD_COVER', 20))
  .map(rmL5789Cloud);

// 获取Landsat-7数据，应用去云算法
var L7_2025 = L7.filterBounds(new_clip_region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUD_COVER', 20))
  .map(rmL5789Cloud);

// 获取Landsat-5数据，应用去云算法
var L5_2025 = L5.filterBounds(new_clip_region)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUD_COVER', 20))
  .map(rmL5789Cloud);

// 5. 计算影像中值合成（只保留光谱波段，排除辅助波段）
/**
 * Landsat中值合成说明：
 * - Landsat 8/9 TOA包含光谱波段：B2-B11（包括热红外B10和B11）
 * - Landsat 5/7 TOA包含光谱波段：B2-B7（不含B8的全色波段）
 * - 只选择光谱波段，排除QA_PIXEL、QA_RADSAT等辅助波段
 * - 可用于多种指数计算：NDVI, NDWI, SAVI等
 */
var L9_composite = L9_2025.median()
  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11'])
  .clip(TARGET_REGION);
var L8_composite = L8_2025.median()
  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11'])
  .clip(TARGET_REGION);
var L7_composite = L7_2025.median()
  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
  .clip(TARGET_REGION);
var L5_composite = L5_2025.median()
  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
  .clip(TARGET_REGION);

// 6. 可视化参数（统一使用RGB真彩色波段）
var trueColor432Vis_L8 = {
  bands:['B4','B3','B2'],  // Landsat 8/9: 红、绿、蓝波段
  min: 0.0,
  max: 0.4,
};
var trueColor432Vis_L457 = {
  bands:['B3','B2','B1'],  // Landsat 5/7: 红、绿、蓝波段
  min: 0.0,
  max: 0.4,
};

// 7. 可视化合成影像
Map.addLayer(L9_composite, trueColor432Vis_L8, 'Landsat-9 Composite 2025');
Map.addLayer(L8_composite, trueColor432Vis_L8, 'Landsat-8 Composite 2025');

// 条件性显示Landsat-7和Landsat-5的RGB影像
if (L7_2025.size().getInfo() > 0) {
  Map.addLayer(L7_composite, trueColor432Vis_L457, 'Landsat-7 Composite 2025');
} else {
  print('Landsat-7在指定时间范围内没有可用数据，跳过RGB显示');
}

if (L5_2025.size().getInfo() > 0) {
  Map.addLayer(L5_composite, trueColor432Vis_L457, 'Landsat-5 Composite 2025');
} else {
  print('Landsat-5在指定时间范围内没有可用数据，跳过RGB显示');
}

// 8. 显示影像统计信息
print('Landsat-9影像数量:', L9_2025.size());
print('Landsat-8影像数量:', L8_2025.size());
print('Landsat-7影像数量:', L7_2025.size());
print('Landsat-5影像数量:', L5_2025.size());

print('L9波段信息:', L9_composite.bandNames());
print('L8波段信息:', L8_composite.bandNames());
print('L7波段信息:', L7_composite.bandNames());
print('L5波段信息:', L5_composite.bandNames());

// 计算影像统计信息
var stats_L9 = L9_composite.select(['B4', 'B3', 'B2']).reduceRegion({
  reducer: ee.Reducer.minMax().combine({
    reducer2: ee.Reducer.mean(),
    sharedInputs: true
  }),
  geometry: new_clip_region,
  scale: 30,
  maxPixels: 1e9
});

var stats_L8 = L8_composite.select(['B4', 'B3', 'B2']).reduceRegion({
  reducer: ee.Reducer.minMax().combine({
    reducer2: ee.Reducer.mean(),
    sharedInputs: true
  }),
  geometry: new_clip_region,
  scale: 30,
  maxPixels: 1e9
});

print('L9影像统计信息:', stats_L9);
print('L8影像统计信息:', stats_L8);
