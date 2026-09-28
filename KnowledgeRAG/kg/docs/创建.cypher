// =====================================================
// 震后滑坡生态恢复知识图谱创建脚本
// 功能：创建完整的知识图谱结构，包括节点、关系和约束
// 说明：可安全重复执行，使用 MERGE 避免重复创建
// =====================================================

// =====================================================
// 一、基础约束（可安全重复执行）
// =====================================================
CREATE CONSTRAINT IF NOT EXISTS FOR (o:本体) REQUIRE o.名称 IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (e:实体) REQUIRE e.名称 IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:状态) REQUIRE s.名称 IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (i:指数) REQUIRE i.名称 IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:时间尺度) REQUIRE t.名称 IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (m:指标) REQUIRE m.名称 IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (tp:时相) REQUIRE tp.名称 IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (r:空间区域) REQUIRE r.名称 IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (tb:图斑) REQUIRE tb.ID IS UNIQUE;

// =====================================================
// 二、本体与实体层
// =====================================================
MERGE (onto:本体 {名称:'震后滑坡生态恢复'})
ON CREATE SET onto.描述 = '以地震引发滑坡为核心，构建生态恢复时空知识图谱'
WITH onto

UNWIND ['滑坡','土壤','植被'] AS ent
MERGE (e:实体 {名称: ent})
MERGE (onto)-[:包含实体]->(e);

// =====================================================
// 三、状态层（滑坡前→滑坡后→恢复中→已恢复）
// =====================================================
UNWIND [
  {名称:'滑坡前', 顺序:0, 语义:'灾前基线状态'},
  {名称:'滑坡后', 顺序:1, 语义:'地表扰动状态'},
  {名称:'恢复中', 顺序:2, 语义:'生态恢复过程'},
  {名称:'已恢复', 顺序:3, 语义:'生态稳定阶段'}
] AS st
MERGE (s:状态 {名称: st.名称})
ON CREATE SET s.顺序 = st.顺序, s.语义 = st.语义;

MATCH (a:状态 {名称:'滑坡前'})
MATCH (b:状态 {名称:'滑坡后'})
MATCH (c:状态 {名称:'恢复中'})
MATCH (d:状态 {名称:'已恢复'})
MERGE (a)-[:时间顺序]->(b)
MERGE (b)-[:时间顺序]->(c)
MERGE (c)-[:时间顺序]->(d);

// =====================================================
// 四、遥感指数层
// =====================================================
UNWIND [
  {名称:'NDVI', 含义:'植被覆盖度'},
  {名称:'NDWI', 含义:'水体或湿度特征'},
  {名称:'NBR', 含义:'扰动及火烧强度'},
  {名称:'地表反照率', 含义:'能量反射特性'},
  {名称:'土壤湿度', 含义:'地表含水状态'}
] AS ind
MERGE (i:指数 {名称: ind.名称})
ON CREATE SET i.含义 = ind.含义;

MATCH (veg:实体 {名称:'植被'})
MATCH (ndvi:指数 {名称:'NDVI'})
MATCH (ndwi:指数 {名称:'NDWI'})
MATCH (nbr:指数 {名称:'NBR'})
MERGE (veg)-[:监测指标]->(ndvi)
MERGE (veg)-[:监测指标]->(ndwi)
MERGE (veg)-[:监测指标]->(nbr);

MATCH (soil:实体 {名称:'土壤'})
MATCH (nbr:指数 {名称:'NBR'})
MATCH (albedo:指数 {名称:'地表反照率'})
MATCH (soilmoist:指数 {名称:'土壤湿度'})
MERGE (soil)-[:监测指标]->(nbr)
MERGE (soil)-[:监测指标]->(albedo)
MERGE (soil)-[:监测指标]->(soilmoist);

MATCH (land:实体 {名称:'滑坡'})
MATCH (veg:实体 {名称:'植被'})
MATCH (soil:实体 {名称:'土壤'})
MERGE (land)-[:包含实体]->(veg)
MERGE (land)-[:包含实体]->(soil);

// =====================================================
// 五、时间尺度与时相层
// =====================================================
UNWIND [
  {名称:'宏观分析', 说明:'年度尺度趋势分析', 时间单位:'年'},
  {名称:'中观分析', 说明:'季度或季节尺度分析', 时间单位:'季'},
  {名称:'微观分析', 说明:'月度或事件尺度分析', 时间单位:'月'}
] AS ts
MERGE (t:时间尺度 {名称: ts.名称})
ON CREATE SET t.说明 = ts.说明, t.时间单位 = ts.时间单位;

MERGE (tp1:时相 {名称:'T1_滑坡前'})
MERGE (tp2:时相 {名称:'T2_滑坡后'})
MERGE (tp3:时相 {名称:'T3_恢复中'})
MERGE (tp4:时相 {名称:'T4_已恢复'})
MERGE (tp1)-[:演化到]->(tp2)
MERGE (tp2)-[:演化到]->(tp3)
MERGE (tp3)-[:演化到]->(tp4);

UNWIND [
  {时相:'T1_滑坡前', 状态:'滑坡前'},
  {时相:'T2_滑坡后', 状态:'滑坡后'},
  {时相:'T3_恢复中', 状态:'恢复中'},
  {时相:'T4_已恢复', 状态:'已恢复'}
] AS pair
MATCH (tp:时相 {名称: pair.时相})
MATCH (s:状态 {名称: pair.状态})
MERGE (tp)-[:对应状态]->(s);

// =====================================================
// 六、空间特征层
// =====================================================
MERGE (r:空间区域 {名称:'九寨沟滑坡区'})
ON CREATE SET r.类型 = '灾区', r.行政区 = '阿坝州';

MERGE (m:指标 {名称:'滑坡面积'})
SET m.定义 = '反映滑坡体规模与生态扰动范围的空间量化指标（单位：m²）';

MATCH (land:实体 {名称:'滑坡'})
MERGE (land)-[:空间属性]->(m)
MERGE (land)-[:发生于]->(r);

MATCH (m:指标 {名称:'滑坡面积'}), (s:状态)
MERGE (m)-[:影响阶段]->(s);
MATCH (m:指标 {名称:'滑坡面积'}), (t:时间尺度)
MERGE (m)-[:参与分析]->(t);
MATCH (m:指标 {名称:'滑坡面积'}), (i:指数 {名称:'NDVI'})
MERGE (m)-[:联合评估]->(i);

// =====================================================
// 七、语义层
// =====================================================
MERGE (sem:语义层 {
  说明:'通过滑坡面积、时间尺度及遥感指数变化推理生态恢复阶段'
});
MATCH (onto:本体 {名称:'震后滑坡生态恢复'})
MERGE (onto)-[:语义解释]->(sem);

// =====================================================
// 八、监测场景
// =====================================================
MERGE (scene:监测场景 {
  名称:'震后九寨沟滑坡生态恢复监测',
  描述:'基于多时间尺度遥感监测分析NDVI、NDWI、NBR变化，反映震后生态恢复阶段'
});
MATCH (onto:本体 {名称:'震后滑坡生态恢复'})
MATCH (land:实体 {名称:'滑坡'})
MATCH (r:空间区域 {名称:'九寨沟滑坡区'})
MERGE (scene)-[:监测对象]->(land)
MERGE (scene)-[:监测区域]->(r)
MERGE (onto)-[:包含场景]->(scene);

MATCH (t1:时间尺度 {名称:'宏观分析'})
MATCH (t2:时间尺度 {名称:'中观分析'})
MATCH (t3:时间尺度 {名称:'微观分析'})
MERGE (scene)-[:监测尺度]->(t1)
MERGE (scene)-[:监测尺度]->(t2)
MERGE (scene)-[:监测尺度]->(t3);

MATCH (i1:指数 {名称:'NDVI'})
MATCH (i2:指数 {名称:'NDWI'})
MATCH (i3:指数 {名称:'NBR'})
MERGE (scene)-[:分析指标]->(i1)
MERGE (scene)-[:分析指标]->(i2)
MERGE (scene)-[:分析指标]->(i3);

MATCH (s1:状态 {名称:'恢复中'})
MATCH (s2:状态 {名称:'已恢复'})
MERGE (scene)-[:反映状态]->(s1)
MERGE (scene)-[:反映状态]->(s2);

// =====================================================
// 九、图斑层（ID为主键）
// =====================================================
MATCH (r:空间区域 {名称:'九寨沟滑坡区'})
UNWIND [
  {ID:'T33TWN', 数据源:'Sentinel-2A', 获取日期:'2025-07-12'},
  {ID:'T33TWM', 数据源:'Sentinel-2B', 获取日期:'2025-08-15'},
  {ID:'T33TXN', 数据源:'Sentinel-2A', 获取日期:'2025-09-02'}
] AS tile
MERGE (tb:图斑 {ID: tile.ID})
ON CREATE SET
  tb.数据源 = tile.数据源,
  tb.获取日期 = date(tile.获取日期)
MERGE (r)-[:包含图斑]->(tb);

MATCH (tb:图斑)
MATCH (t_macro:时间尺度 {名称:'宏观分析'})
MATCH (t_meso:时间尺度 {名称:'中观分析'})
MATCH (t_micro:时间尺度 {名称:'微观分析'})
MERGE (tb)-[:观测于]->(t_macro)
MERGE (tb)-[:观测于]->(t_meso)
MERGE (tb)-[:观测于]->(t_micro);

MATCH (s_pre:状态 {名称:'滑坡前'})
MATCH (s_post:状态 {名称:'滑坡后'})
MATCH (s_rec:状态 {名称:'恢复中'})
MATCH (s_fin:状态 {名称:'已恢复'})

// 为每个图斑分配阶段（随机示例）
WITH tb, [s_pre,s_post,s_rec,s_fin][toInteger(rand()*4)] AS stage
MERGE (tb)-[:对应状态]->(stage);

MATCH (land:实体 {名称:'滑坡'})
MERGE (land)-[:包含图斑]->(tb);

// =====================================================
// 十、图斑指数观测层（扩展部分）
// =====================================================

// 图斑 — 指数 直接观测关系
MATCH (land:实体 {名称:'滑坡'})-[:包含实体]->(sub:实体)
WHERE sub.名称 IN ['植被','土壤']
MATCH (sub)-[:监测指标]->(i:指数)
MATCH (r:空间区域 {名称:'九寨沟滑坡区'})-[:包含图斑]->(tb:图斑)
MERGE (tb)-[:观测指数]->(i);

// 图斑 — 时间尺度 — 指数 复合观测关系
MATCH (tb:图斑)-[:观测于]->(t:时间尺度)
MATCH (tb)-[:观测指数]->(i:指数)
MERGE (tb)-[:在时间尺度观测 {粒度:t.时间单位}]->(i);

// =====================================================
// 十一、Graph 可视化查询
// =====================================================
MATCH (r:空间区域 {名称:'九寨沟滑坡区'})-[:包含图斑]->(tb:图斑)
OPTIONAL MATCH (tb)-[:观测于]->(t:时间尺度)
OPTIONAL MATCH (tb)-[:对应状态]->(s:状态)
OPTIONAL MATCH (tb)-[:观测指数]->(i:指数)
RETURN r, tb, t, s, i
ORDER BY tb.ID;

