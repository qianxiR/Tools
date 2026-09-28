var roi = table.filter(ee.Filter.eq('name','玉树藏族自治州'));
// 获取四川省的最小外接矩形
var bounds = roi.geometry().bounds();
// 获取最小外接矩形顶点坐标
var lonlat = bounds.coordinates()
print(lonlat)
var styling = {color:'red',fillColor:'00000000'}

// Map.addLayer(bounds)

var generateGrid = function(xmin, ymin, xmax, ymax, dx, dy) {
  var xx = ee.List.sequence(xmin, ee.Number(xmax).subtract(0.0001), dx);
  var yy = ee.List.sequence(ymin, ee.Number(ymax).subtract(0.0001), dy);
  var cells = xx.map(function(x) {
    return yy.map(function(y) {
      var x1 = ee.Number(x);
      var x2 = ee.Number(x).add(ee.Number(dx));
      var y1 = ee.Number(y);
      var y2 = ee.Number(y).add(ee.Number(dy));
      var coords = ee.List([x1, y1, x2, y2]);
      var rect = ee.Algorithms.GeometryConstructors.Rectangle(coords);   //生成矩形
      return ee.Feature(rect);
    });
  }).flatten();  
  return ee.FeatureCollection(cells);
};
var bounds = roi.geometry().bounds();
var coords = ee.List(bounds.coordinates().get(0));
var xmin = ee.List(coords.get(0)).get(0);
var ymin = ee.List(coords.get(0)).get(1);
var xmax = ee.List(coords.get(2)).get(0);
var ymax = ee.List(coords.get(2)).get(1);
var dx = (ee.Number(xmax).subtract(xmin)).divide(4);
var dy = (ee.Number(ymax).subtract(ymin)).divide(4);
var grid = generateGrid(xmin, ymin, xmax, ymax, dx, dy);
var grid = grid.filterBounds(roi); // filter out out-of-boundary tiles
print(grid.size()); //查看生成所有格网数量
Map.addLayer(grid.style({color:'blue',fillColor:'00000000'}), {}, 'grid');
Map.addLayer(roi.style(styling),{},'sichuan_boundary')
Map.centerObject(bounds,8)
