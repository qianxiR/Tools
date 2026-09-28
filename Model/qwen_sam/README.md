# QwenSam 岩性边界分割

SAM3 双头（类别 + 四通道类别边缘）岩性分割网络：冻结 SAM3 ViT 主干 + SAM3-Adapter prompt 注入 + 纯视觉四尺度解码 + 边缘先验模块 + 语义感知模块。

## 当前选定版本

| 项 | 值 |
|---|---|
| 实验目录 | `F:\岩石\runs\exp_calss_edge\20260822_125357` |
| 训练配置 | train=[2,3]、val=[4]、坡积 8 倍离线过采样 + 在线几何增强、`--structure-ablation both` |
| best checkpoint | step 500：前景 mIoU 0.5748、macro F1 0.7204、像素准确率 0.818 |

各类 IoU：背景 0.672 / 熔岩 0.782（全部实验单类最高）/ 白云岩 0.720 / 第四系 0.594 / 坡积 0.204。

消融对照（同种子）：base 0.5491 / 仅边缘先验 0.5603 / 仅语义感知 0.5534 / 双模块串联 0.5748。

## 目录结构

- `F:\岩石\runs\exp_calss_edge\<YYYYMMDD_HHMMSS>\`：每次训练自动创建，含 `best_model.torch`、`final_model.torch`、`best_val_metrics.json`、`tb\`、`experiment_sources\`（源码与 config 快照）、`VIS\`（可视化输出）。现存四个实验目录：125357（双模块串联）、133400（仅边缘先验）、134100（仅语义感知）、145453（base）。
- `F:\岩石\0820数据集\`：切片数据集（`train/val/infer` + `*.jsonl` + `manifest.json`）。train=[2,3]（125 原始 tile + 77 坡积过采样 tile = 202）、val=[4]（35 tile）、infer=1..26（1978 tile）。
- `F:\xzrsagent\VLLM\QwenSam\archive\`：历史版本代码备份（v1 极简版、v3 双域版）。
- `F:\岩石\0820标注\`：源标注（`label\2/3/4.json`）与拼图影像。

## 数据流水线

1. 修改 `config.yaml`（类别 / 划分）后重建数据集。
2. 重建切片数据集：`prepare_labelme_dataset.py`（train 25% 重叠切片并过滤全背景，val/infer 无重叠网格）。
3. 坡积过采样：`oversample_poji_tiles.py`（含坡积 tile 生成 7 个二面体几何变体并追加 `train.jsonl`，可重复执行）。
4. 训练：`train_labelme_edge.py`（best 出现在 step 500~1000，1000 步内完成）。
5. 可视化推理：`test_labelme_edge.py`（缺省自动选择最新含 best checkpoint 的实验，输出 `VIS\<样本>\` 四图）。
6. 外部目录推理：`infer_external_folder.py`（整图切片 → 拼接 → 内存安全后处理，输出同款四图）。

## 推理流程

- `test_labelme_edge.py`：逐 tile argmax → 投票拼接（val/infer 无重叠，等价直接拼接）→ 后处理（平滑 / 去小区域 / 填洞）→ 由后处理掩码按真值同款 3×3 形态学梯度重建边缘线 → 输出 `VIS\<样本>\` 四图（`mask.png`、`mask_overlay.png`、`edge_lines.png`、`edge_overlay.png`，带类别图例）。
- 模型按 checkpoint 的 `structure_ablation` 字段自动构造对应消融形态（缺失时回退 both），可用 `--structure-ablation` 覆盖；`--samples` 可仅推理指定样本。

## 常用命令

```powershell
D:\anaconda3\envs\llamafactory\python.exe F:\xzrsagent\VLLM\QwenSam\prepare_labelme_dataset.py
D:\anaconda3\envs\llamafactory\python.exe F:\xzrsagent\VLLM\QwenSam\oversample_poji_tiles.py
D:\anaconda3\envs\llamafactory\python.exe F:\xzrsagent\VLLM\QwenSam\train_labelme_edge.py --structure-ablation both
D:\anaconda3\envs\llamafactory\python.exe F:\xzrsagent\VLLM\QwenSam\test_labelme_edge.py --run-dir F:\岩石\runs\exp_calss_edge\20260822_125357
```

## 实验结论

1. 双模块（边缘先验 + 语义感知）串联最优：0.5748，比 base +2.6，串联比单独之和 +0.9（强协同）。
2. 坡积可识别取决于多样本入训：[2,3]/[4] 划分坡积 0.15~0.25，[3,4]/[2] 划分坡积 ≈0.001；坡积离线过采样必要但不充分，对其他类几乎无损；在线几何增强对稀有类 +0.06。
3. 各轮 best 均在 step 500~1000 之间，1000 步后验证不再获益。
