"""
配置文件 - 统一管理所有脚本的参数
修改此文件可以调整整个数据流程的配置
"""

# ========== 基础路径配置 ==========
BASE_DIR = r'E:\1代码\模型\gee'
CODE_DIR = r'E:\1代码\模型\gee\逐波段提取数据'

# ========== 提取配置 ==========
# 输入数据文件夹（包含CSV/XLS/XLSX文件）- 新增
INPUT_FOLDER = r'E:\1代码\模型\gee\逐波段提取数据\玉树水深\玉树所有\转换后'
# 旧配置（保留兼容性）
INPUT_CSV = r'E:\1代码\模型\gee\逐波段提取数据\玉树水深\玉树所有\转换后\1025点_精简版.csv'
BATCH_SIZE = 5000  # 每批处理的点数（GEE限制5000）
START_DATE = '2025-08-01'
END_DATE = '2025-11-01'
CLOUD_THRESHOLD = 30  # 云覆盖阈值（%）

# ========== GEE项目配置 ==========
PROJECT_ID = 'applied-pipe-453411-k9'

# ========== 输出文件配置 ==========
# 提取输出
OUTPUT_EXTRACT_CSV = '湖库波段值提取结果.csv'
TEMP_BATCH_PATTERN = '湖库波段值提取结果_temp_batch*.csv'

# 合并输出
OUTPUT_MERGED_CSV = '湖库波段值_合并完整.csv'
OUTPUT_PROCESSED_CSV = '湖库波段值_预处理后.csv'

# 报告输出
OUTPUT_REPORT_TXT = '数据分析报告.txt'
OUTPUT_COVERAGE_CSV = '影像覆盖率检查报告.csv'

# ========== 流程控制 ==========
# 是否运行各个步骤（可以在run_pipeline.py中覆盖）
ENABLE_COVERAGE_CHECK = False  # 覆盖率检查（通常第一次运行时需要）
ENABLE_EXTRACTION = True       # 波段值提取
ENABLE_MERGE_PREPROCESS = True # 合并和预处理

# ========== 预处理参数 ==========
# 异常值裁剪分位数
OUTLIER_CLIP_UPPER = 0.995  # 上界（99.5%分位数）
OUTLIER_CLIP_LOWER = 0.005  # 下界（0.5%分位数）

# 是否启用标准化（可选）
ENABLE_NORMALIZATION = False  # 目前使用裁剪而非标准化

# ========== 可视化配置 ==========
FIGURE_DPI = 300  # 图表分辨率
FIGURE_FONTSIZE = 'SimHei'  # 中文字体

# ========== 调试选项 ==========
DEBUG_MODE = False  # 是否启用调试模式
VERBOSE = True      # 是否显示详细信息

