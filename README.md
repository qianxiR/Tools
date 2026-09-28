# Tools — 遥感智能解译工具包

覆盖从视觉工具到视觉语言工具的完整遥感解译链路，按职责分四层组织：

- **DataProcessing** 数据处理层：栅格 I/O 与地理信息、影像增强与变换管线、裁剪/重采样/波段组合、光谱指数、空间分析、可视化、GEE 认证/下载/波段提取、视觉样本制备、数据集工程、VL 数据集生成（分割/变化检测 caption+QA，COCO/MCI）
- **Model** 模型推理层：大图滑窗推理引擎（平滑加权/拼接/显存管理）；视觉模型（语义分割、变化检测、目标检测、场景分类）；视觉语言模型（SamSeg、qwen_sam、qwen_dec）
- **PostProcessing** 后处理层：掩码平滑、连通域、边缘、栅格转矢量、CRF、坐标转换
- **KnowledgeRAG** 知识服务层：Neo4j 知识图谱 Agent、Chroma + Qwen RAG（空间领域知识库）

数据流：DataProcessing（数据/样本制备）→ Model（视觉或视觉语言推理）→ PostProcessing（结果清理/矢量化）→ KnowledgeRAG（领域知识问答）。

## 目录结构

```text
Tools/
├── DataProcessing/              数据处理层
│   ├── raster_utils / raster_io           栅格 I/O 与地理信息
│   ├── image_ops / image_transforms       影像增强与变换管线
│   ├── image_utils / band_utils           裁剪/重采样/波段组合、波段映射
│   ├── spectral / spatial / visualization 光谱指数、空间分析、可视化
│   ├── gee/                               GEE 认证/下载/波段提取/code_editor(JS)
│   ├── sample_prep / dataset_utils        视觉样本制备、数据集工程
│   ├── dataset_generation/                VL 数据集生成（分割/变化检测 caption+QA）
│   └── mask_postprocess/                  兼容转发包（实现已迁至 PostProcessing）
│
├── Model/                        模型推理层（视觉 → 视觉语言）
│   ├── inference/                          大图滑窗推理引擎（两类模型共用）
│   ├── segmentation/                       语义分割：UNet/FPN/PSPNet/LinkNet/NestNet/DeepLab/HRNet
│   ├── changedetect/                       变化检测：DSIFN、Detect_siam_unet
│   ├── detection/                          目标检测：yolov5 完整仓库与封装
│   ├── classification/                     场景分类
│   ├── SamSeg/                             SAM3 开放词汇分割/变化检测/VQA（免训练文本提示）
│   ├── qwen_sam/                           SAM3 双头多类别分割（文本融合+边缘先验）
│   ├── qwen_dec/                           Qwen3-VL 目标检测（LoRA+SigLIP2 特征注入）
│   └── examples/                           各任务流程演示输出与笔记
│
├── PostProcessing/               后处理层
│   └── mask_postprocess/                   掩码平滑/连通域/边缘/栅格转矢量/CRF/坐标转换
│
└── KnowledgeRAG/                 知识服务层
    ├── kg/                                 Neo4j 知识图谱 Agent
    └── rag/                                Chroma + Qwen RAG（空间领域知识库）
```

## 使用方式

Tools 作为 Python 包被上层工程引用（如 Agent-RSCD 按 `Tools.Model.SamSeg.SamSeg.infer` 引用），包入口统一导出各层常用接口：

```python
from Tools import load_img_by_gdal, predict_img_with_smooth_windowing
from Tools import smooth_mask, raster_to_vector
```

使用前需将本仓库的父目录加入 `PYTHONPATH`，保证 `Tools` 可被导入：

```powershell
$env:PYTHONPATH = "E:\1代码\系统"
```

## 环境依赖

- 基础环境：GDAL、rasterio、numpy、opencv-python、shapely、scikit-image、torch
- `Model/qwen_sam`、`Model/qwen_dec` 依赖 llamafactory 环境（transformers 4.57.6 + peft 0.18.1 + torch 2.11.0+cu128），独立运行，不并入 Tools 包导出；qwensam 推理依赖 sam3 模块（本地 SamSeg 内有同源代码）
- `KnowledgeRAG/rag` 依赖 langchain 系列，通义千问 API Key 通过环境变量注入：

```powershell
$env:QWEN_API_KEY = "你的 DashScope API Key"
```

## 命名规范

| 层级 | 规范 | 示例 |
|---|---|---|
| 顶层功能层 | PascalCase | DataProcessing / Model / PostProcessing / KnowledgeRAG |
| 包与子目录 | snake_case 全小写 | qwen_sam / mask_postprocess / sample_prep |
| 例外 | 外部依赖钉死的目录保持原名 | Model/SamSeg（Agent-RSCD 按 `Tools.Model.SamSeg.SamSeg.infer` 引用，Python 导入区分大小写，不可改） |

## 注意事项

1. 模型权重（`*.pt`）与栅格数据（`*.tif`）体积过大，不入库，需另行存放至对应目录（如 `Model/SamSeg/SamSeg/checkpoints/`、`Model/detection/`）。
2. `DataProcessing/mask_postprocess` 为兼容转发包，实现位于 `Tools.PostProcessing.mask_postprocess`，旧导入路径继续有效。
3. `Model/inference/smooth_predict.py` 与 `predict_backbone.py` 为同一平滑滑窗算法的双实现（包导出用前者），后续可合并。
4. `KnowledgeRAG/.history`、`.git-backups` 为本地备份，不入库。

## 相关仓库

- [Agent-RSCD](https://github.com/qianxiR/Agent-RSCD) — 引用本工具包的上层智能体工程
