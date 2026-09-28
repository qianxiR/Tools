# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

> 文档优先级：本文件与当前代码共同定义唯一有效的 **v10 最终架构**训练/推理规范（2026-08-26 定稿）；`训推.md` 是运行手册，`docs/` 是架构说明，`实验记录.md` 是当前结果真源。`.zcode/plans/` 中的历史计划及实验报告中的旧架构仅作归档，不作为运行依据。

## 项目本质

Qwen3-VL 目标检测的「文本化」实现：模型自回归生成 `<obj>` 包裹的 JSON 文本 `<obj>[{"bbox_2d":[x1,y1,x2,y2],"label":"<code>"}]</obj>`（bbox 坐标用 0-1000 归一化裸整数，沿用基座原生词表，label 为类别代码），本质是因果语言建模（CAUSAL_LM），不是带检测头的视觉任务。**v10 恢复 v8 双视觉链结构**，两条可训练路径：

1. **语言 decoder LoRA（方案 A：Qwen 仅训练解码器）**：语言 decoder 的 `q/k/v/o/gate/up/down_proj`（正则负向前瞻排除 `external_*`）；Qwen 原生视觉塔 **24 个 block 冻结保留**（仅前向、不挂 LoRA），提供高分辨率原生视觉感知但不参与微调。
2. **外部视觉前缀注入（SigLIP2 主干 LoRA 微调）**：**冻结 SigLIP2 Base**（`google/siglip2-base-patch16-224`，768 hidden，原生 224 输入、运行时插值到 **256×256** → **16×16 patch 网格`）叠加 **LoRA**（r=32, α=64, dropout=0.05，正则限定 `encoder.layers.*` 的 `self_attn.q/k/v/out_proj` 与 `mlp.fc1/fc2`，共 12 层 ×6 目标）取 **四层 hidden_states（第 3/7/9/12 层）拼接**为多深度特征 [B,256,3072]，经 **SimpleFPN**（`SIMPLE_FPN_SCALE_FACTORS=(4,2,1,0.5)`）→ 粗到细 **MultiScaleFusionDecoder** 纯视觉融合池化回 [B,256,16,16] → projector（`Linear(256→512)→SiLU→Linear(512→hidden)` + LayerNorm + 二维正余弦位置编码）生成 **16×16 = 256 个**前缀 token，`masked_scatter` 覆盖输入开头保留槽位后与 Qwen 原生视觉 token 一起进入语言解码器。

**方法选型依据**：v8（冻结 SigLIP2 + 原生塔 LoRA）500 集 epoch1 即 F1≈0.30（可训练，3000 集 Test F1=0.4245），v10 = v8 结构 + SigLIP2 主干 LoRA 单变量升级（`format_version=10`）；**v10-A 当前链：Qwen 侧仅训练语言 decoder LoRA（原生视觉塔冻结保留）**，SigLIP2 主干 LoRA 微调不变。此前一次"文本引导融合 + 原生视觉塔停用"的中间版本已验证不可训练（视觉信息瓶颈），已移除。

通用流程支持任意单类/多类检测数据集，内置 VOC、YOLO 两种格式转换器；切换数据集只需改「数据路径 + prompt JSON + 类别表」。当前默认示例为**管网缺陷 数据_筛选3000（YOLO，12 类）**，数据集根为 `F:/管网/数据_筛选3000`（含 500/aug 快速验证集，均支持 `DATASET_ROOT` 覆盖）。

> 历史注记：更早的 ResNet50/SAM2 外部链及一次"文本引导融合 + 原生视觉塔停用"的中间版本均已移除；外部特征链恒为 SigLIP2 Base（v10 起 LoRA 微调）。

## 常用命令

```powershell
# 环境（注：见下方「环境」必须先核对）
conda activate llamafactory
cd F:\xzrsagent\VLLM\QwenDec

# 0. 数据集划分（7:1:2 种子 42，默认 F:\管网\数据_筛选3000；3000 集实际划分由 scripts\create_split_3000.py 生成——排除 500 集 test/val 防泄露，勿混用）
python scripts\split_dataset.py

# 1. 数据转换 → Qwen3-VL JSONL（写出到 <DATASET_ROOT>）
python scripts\convert_yolo_to_qwenvl.py      # YOLO cxcywh → 0-1000 xyxy
python scripts\convert_voc_to_qwenvl.py       # VOC 绝对像素 xyxy → 0-1000

# 2. 训练（路径/开关全部支持环境变量覆盖；首次运行自动在 <OUTPUT_DIR> 生成 text_prompt_cache.pt）
python train_qwen3vl_lora.py

# 3. 推理（Gradio 单图，默认 http://127.0.0.1:7860）
python inference.py --adapter F:\管网\runs\pipe_defect_lora_v10_siglip2_base\train\best_adapter

# 4. 评估（test 集，IoU 默认 0.5）
python evaluate_detection.py --adapter F:\管网\runs\pipe_defect_lora_v10_siglip2_base\train\best_adapter --test-jsonl F:\管网\数据_筛选3000\pipe_qwenvl_test.jsonl
```

仓库无 lint / 测试配置。训练日志不持久化保存（启动时重定向到 `*.log`，评估完即清理）。

## 环境

| 来源 | conda 环境 | transformers |
|---|---|---|
| [训推.md](训推.md) §0（主文档）、`train_qwen3vl_lora.py` 文件头 | `llamafactory` | **4.57.6** |

统一以 `llamafactory`（transformers 4.57.6 / peft 0.18.1 / qwen-vl-utils / swanlab）为准。`USE_4BIT` 在 transformers 4.57.6 下与自定义 external 模块存在已知不兼容（`AttributeError: 'weight' is not an nn.Module`），故训练默认 **bf16 不量化**，开启 `USE_4BIT=1` 会直接抛错；仅在显存受限时才考虑。

## 核心架构（跨文件理解）

### 特征注入（`model/qwen_feature_injection.py`）
`FeatureInjectedQwen3VLForConditionalGeneration` 继承 `Qwen3VLForConditionalGeneration`：
- **`forward`**：prefill 阶段由 `encode_external_images` 生成 `[B,256,D]` 外部空间 token，并在 `_inject_external_prefix` 中用 `masked_scatter` 覆盖输入开头 256 个保留槽位（`EXTERNAL_PREFIX_GRID_SIZE=16`，16×16=256 token）；随后调 `self.model(...)` 取最后一层 hidden，再由 `lm_head` 出 logits。loss 复用基座 `self.loss_function`（标准 LM CE，坐标用基座原生数字 token，无专项 loss）。
- **`prepare_inputs_for_generation`**：把 `external_images` / `external_hidden` 透传进 generate 的逐步解码；仅 prefill 注入，后续 token 复用 KV cache。
- **保存/加载**：PEFT 只存文本 + 视觉 LoRA；`save_external_adapter` 单独导出 `external_feature_adapter.pt`（`format_version=10`），内容为 SigLIP2 LoRA（`external_siglip_lora.*`）+ SimpleFPN/融合 decoder（`external_feature_adapter.*`）+ 前缀 projector（`external_projector.*`）+ 元数据（`siglip_lora` r/α、`fusion="pure-visual-multiscale"`、`qwen_native_vision="lora-trainable"`）。`load_external_adapter(strict=True)` 缺失、版本（非 10；v8 冻结主干 / 文本引导中间版均不兼容）或主干元数据不匹配时报错，禁止静默回退。

### 外部模块（`model/external_feature_adapter.py`）
- `Siglip2BaseFeatureExtractor`：冻结 SigLIP2 Base 主干叠加 peft LoRA（`SIGLIP2_LORA_R=32`/`SIGLIP2_LORA_ALPHA=64`/`SIGLIP2_LORA_DROPOUT=0.05`，target 正则限定 `encoder.layers.*`，共 144 个 LoRA 参数组），原生 224 输入、运行时插值到 256×256 → 16×16 patch 网格；**前向不再 no_grad**（LoRA 需要梯度回传，主干参数仍冻结）。取四层 hidden_states（`SIGLIP2_MULTISCALE_LAYERS=(3,7,9,12)`）拼接为 [B,256,3072]，经 SimpleFPN 派生四尺度后交 **MultiScaleFusionDecoder**（纯视觉粗到细 U-Net：GroupNorm+SiLU 卷积融合块 + 上采样跳连）池化回 `[B,256,16,16]`。`lora_state_dict()/load_lora_state_dict()` 以键集合严格校验的方式收集/恢复主干 LoRA（嵌入单文件 adapter 协议）。
- `ExternalFeatureProjector`：展平为 256 个 token，使用 `Linear(256→512)→SiLU→Linear(512→hidden)` + LayerNorm 投影到语言侧 hidden_size，并叠加二维正余弦位置编码（hidden_size 四等分给 y-sin/y-cos/x-sin/x-cos）。
- 常量：`SIGLIP2_INPUT_SIZE=256`、`SIGLIP2_PATCH_SIZE=16`、`SIGLIP2_PATCH_GRID_SIZE=16`、`EXTERNAL_FEATURE_CHANNELS=256`、`SIGLIP2_MULTISCALE_LAYERS=(3,7,9,12)`、`SIMPLE_FPN_SCALE_FACTORS=(4,2,1,0.5)`。外部 adapter `format_version=10`（v8 结构 + SigLIP2 LoRA；v8 冻结主干与文本引导中间版结构均不兼容）。

### 坐标表示（`model/location_tokens.py`）
- 当前训练/推理链使用 **裸整数** 0-1000 归一化坐标 `[x1,y1,x2,y2]`（x 以原图宽、y 以原图高为基准），沿用基座原生词表的数字 token，不新增 location token。
- 历史上曾用 Falcon 风格 location token（`<loc_0>`..`<loc_999>`，1000 个新增 added token + 量化 `round(value*999/1000)`），但实测在小数据下训不动，epoch1 即出现 fp≈700 的灾难性失败（坐标完全失效），已弃用。
- `decode_bbox(allow_legacy_numeric=True)`：推理/评估解析端仍兼容裸整数与旧 location token 两种 bbox 形式（解析容错用）；训练链不再使用 location token。

### 训练数据流（[train_qwen3vl_lora.py](train_qwen3vl_lora.py)）
刻意不用 trl `SFTTrainer`（跨版本签名不稳），改用自定义 `Dataset + Collator + 原生 Trainer`：
- **双视觉输入（v10）**：user 段携带图像（`<|image_pad|>` 展开为 Qwen 原生视觉 token，`pixel_values`/`image_grid_thw` 整批级 cat 拼接）+ prompt 文本；collator 左 padding 后在每样本最前插入 `EXTERNAL_PREFIX_LENGTH=256` 个前缀槽位（pad token 占位，embedding 由 SigLIP2 前缀覆盖，labels 恒 IGNORE_INDEX）。
- **assistant 段 loss mask**：用 `<|im_start|>assistant
` 作锚点在 token 序列中定位回答起点，其前全部置 `IGNORE_INDEX=-100`。注意必须用同一 `processor` 编码（裸 tokenizer 不展开 `<|image_pad|>` 会导致前缀长度偏小、user 段/图像 token 误参与 loss）。
- **变长左 padding**：input_ids/labels 左 padding，attention_mask 右侧补 0（保证生成时序列前缀连续）。
- **数据增强**（[model/data_augment.py](model/data_augment.py)，**默认开启**，`AUGMENT_TRAIN=0` 可关闭）：训练集 50% 概率水平翻转，bbox 同步变换 `[x1,y1,x2,y2] → [1000-x2, y1, 1000-x1, y2]`（必须交换 x1/x2）；翻转后 PIL 直接替换 `record["image"]`，保证 Qwen ViT 与 SigLIP2 两条视觉链路看到同一张翻转图。val/test 不增强。
- **损失函数**：仅标准 LM CE（坐标用基座原生数字 token）。历史上曾叠加类别/location token 专项 CE，但裸整数链已移除，对齐 baseline_restore（F1=0.489）的纯 CE 配置。
- **可训参数**：语言 decoder LoRA（r=32, alpha=64, dropout=0.05，`q/k/v/o/gate/up/down_proj`，方案 A 下 Qwen 原生视觉塔冻结不挂 LoRA）；外部链训练 SigLIP2 主干 LoRA（正则限定 encoder layers）+ SimpleFPN/融合 decoder + projector（`enable_external_training` 统一开梯度，主干权重保持冻结）。可选 norm 全量微调（`NORM_MODULES_TO_SAVE=1`，2×2 析因实验证明有害，默认关闭）。
- **训练控制**：默认 20 epoch（`NUM_TRAIN_EPOCHS` 可 env 覆盖；500 集实测 best 多在 epoch 9-15 见顶、后段平台化，2026-08-26 由 30 下调为最终默认），**实际生效统一 lr 1e-4 constant**（`build_optimizer` 分 text_lora / vision_lora / external_decay / external_no_decay / norm_to_save 五组，全部 `UNIFIED_LR=1e-4`；顶部的 `TEXT_LORA_LR`/`VISION_LORA_LR`/`EXTERNAL_ADAPTER_LR` 三个分组常量**已废弃**，仅写入 swanlab config 展示、不进优化器），`TrainingArguments.warmup_steps=0`、`lr_scheduler_type="constant"`、梯度裁剪 `1.0`、`per_device_train_batch_size`（yaml 默认 1）+ `gradient_accumulation_steps=4`（有效 batch=4）、bf16、`optim=paged_adamw_8bit`、gradient checkpointing。**val 评估降频**（`VAL_EVAL_INTERVAL` 默认 3，可选 `VAL_EVAL_SIZE` 子采样）+ **F1 早停默认关闭**（`F1_EARLY_STOPPING_PATIENCE` 默认 0；设 >0 开启）。`BestByF1Callback` 在评估 epoch 末对 val 集逐图 generate → IoU 匹配 → 按 F1 保存 `best_adapter`。
- **可恢复 checkpoint**：每个 `checkpoint-*` 同时写入 `external_feature_adapter.pt`（v10 含 SigLIP2 LoRA）；设置 `RESUME_FROM_CHECKPOINT=latest` 或具体 checkpoint 路径后才能正确恢复 LoRA、外部链、optimizer 和 scheduler。

### 推理/评估复用（[inference.py](inference.py) + [evaluate_detection.py](evaluate_detection.py)）
`evaluate_detection.py` 直接 `from inference import ...` 复用 `load_model / run_inference / parse_bbox_json`，保证评估与部署端 prompt、坐标反归一化、LoRA/外部 adapter 加载方式一致。`run_inference` 图文混合输入 + 256 前缀槽位；`--no-external` 原生基线对照分支恢复（纯 Qwen 原生视觉 token，不注入 SigLIP2 前缀）。`apply_class_aware_nms` 仅为显式传入 `--nms-iou <阈值>` 的对照后处理，默认关闭，避免误删密集或重叠的同类缺陷。训练期 val F1 始终按原始预测计算，保证与历史实验可比。

## 三端一致性铁律（破坏即失效）

1. **外部特征 adapter 必须同时加载**：推理需 LoRA adapter + `external_feature_adapter.pt` v10（`load_external_adapter(strict=True)`，内含 SigLIP2 LoRA/融合链/projector）。漏装 → SigLIP2 注入静默失效 → 训练/推理分布不一致。
2. **坐标协议统一 0-1000 归一化裸整数**（x 以原图宽、y 以原图高为基准），非像素。推理反归一化 `bbox[i]/1000*W|H` 必须与训练前归一化严格互逆。assistant 输出以 `<obj>...</obj>` 包裹 JSON 数组，bbox 四坐标为 0-1000 整数、label 用类别代码；训练/推理均沿用基座原生词表（151669，不 resize、不注册 location token）。解析端（[inference.py](inference.py) `parse_bbox_json`）通过 `decode_bbox(allow_legacy_numeric=True)` 无条件剥离 `<obj>` 后兼容裸整数与旧 location token 两种格式。
3. **prompt 训练/推理分工**：YOLO 转换器与训练从 `data/pipe_defect_3000_prompt_train_en.json` 的 `full_prompt` 读取；推理与评估从 `data/pipe_defect_3000_prompt_infer_en.json` 的 `full_prompt` 读取。两份 prompt 必须共享 12 个 `label_codes`、`<obj>...</obj>` 包装和 0-1000 裸整数坐标协议，但训练版保持简洁稳定，推理版包含完整视觉证据判据。旧 `PROMPT_JSON` 环境变量仅作为兼容回退；修改训练 prompt 后必须重新生成 JSONL 并重新训练。
4. **processor 参数一致**：`AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=128*28*28, max_pixels=512*28*28)` 在训练与推理必须相同，否则视觉 token 数漂移导致检测失败。词表保持基座原生（不注册 location token、不 resize），训练与推理一致。
5. **SigLIP2 LoRA 超参一致**：config.yaml 修改 `siglip_lora.r/alpha` 后推理端 `bind_external_extractor` 须同步传参（`load_external_adapter` 元数据校验显式报错）。

## 关键路径与环境变量

**配置三级优先：环境变量显式设置 > [config.yaml](config.yaml)（仓库根，v10 训练配置的唯一调参入口，含路径/双 LoRA 超参/训练超参/评估控制） > 代码默认值**。环境变量名 = 键路径大写拼接（如 `text_lora.r` → `TEXT_LORA_R`；`epochs` 沿用历史名 `NUM_TRAIN_EPOCHS`）。注意：`config.yaml` 修改 `siglip_lora.r/alpha` 后推理端 `bind_external_extractor` 须同步传参（`load_external_adapter` 的元数据校验会显式报错）；修改 `train_prompt_json` 后须重新生成 JSONL 并重新训练（v10 无文本缓存）。

训练侧主要配置项（[train_qwen3vl_lora.py](train_qwen3vl_lora.py) 顶部 `_pick` 三层加载）：`model_path` / `siglip_model_path` / `dataset_root` / `train_jsonl` / `val_jsonl` / `train_prompt_json`（兼容回退 env `PROMPT_JSON`）/ `output_dir` / `resume_from_checkpoint` / `text_lora.{r,alpha,dropout}` / `siglip_lora.{r,alpha,dropout}` / `epochs` / `per_device_train_batch_size` / `gradient_accumulation_steps` / `learning_rate` / `lr_scheduler_type` / `warmup_steps` / `weight_decay` / `max_grad_norm` / `optim` / `save_total_limit` / `augment_train` / `norm_modules_to_save` / `val_eval_interval`（默认 3） / `val_eval_size`（默认 300=全量） / `f1_early_stopping_patience`（默认 0=关闭早停） / `swanlab_{project,mode,logdir}` / `use_4bit`（开启即抛错）。YOLO 转换器使用 `TRAIN_PROMPT_JSON`；[inference.py](inference.py) / [evaluate_detection.py](evaluate_detection.py) 使用 `INFER_PROMPT_JSON`（兼容回退 `PROMPT_JSON`），默认分别指向训练版和推理版英文 prompt。

固定资源（本机实际位置，config.yaml 与 inference.py 默认值已指向，无需 env 覆盖）：
- 基座模型：`F:\pre\Qwen3-VL-4B-Instruct`
- SigLIP2 Base 权重：`F:\pre\siglip2-base-patch16-224`
- 数据集根：`F:\管网\数据_筛选3000`（`DATASET_ROOT` 覆盖；500/aug 快速验证集同在 `F:\管网` 下）

## 实验记录（已迁出）

所有训练/评估实验数据（val/test F1、per-class 指标、消融对比、output 目录产出明细）已集中到 **[实验记录.md](实验记录.md)**，避免工程文档与易过期的实验结果混杂。

工程侧需要记住的四条**方法选型级结论**（细节见实验记录）：
1. **裸整数坐标链为唯一可用方案**：location token 方案在小数据下训不动（epoch 1 F1≈0.046 vs 裸整数 0.30，起步 loss 相差约 27×），后续 LPM 重试的 best val F1 也仅 0.0811，相关训练产物已移除。
2. **当前实验统一在 `数据_筛选3000`（test=600）上进行**；500 数据集只用于过失败方案的快速验证，其结果不能与 3000 比较。
3. **v8 为当前固定比较基线：Test F1=0.4245**（P=0.4841 / R=0.3780 / AP50=0.1931 / mAP50-95=0.0926）；adapter 为 `F:/管网/runs/pipe_defect_lora_siglip2_base_v8/train/best_adapter`（best epoch=15，val F1=0.4581）。旧 v7 三层拼接链 Test F1=0.4268（`F:/管网/runs/pipe_defect_lora_siglip2_base/train/best_adapter`）为历史对照。详细指标见 [实验记录.md](实验记录.md) §3.3。
4. **最终架构 v10 = v8 全部结构 + SigLIP2 主干 LoRA 微调**（Qwen 原生视觉塔 LoRA 保留，与 v8 完全一致）：500 集实测 val best F1=0.3474（epoch 9）、test F1=0.3526（P=0.381/R=0.328，tp=64/fp=104/fn=131，invalid=0），高于 v8 冻结链的 500 集早期水位（epoch1≈0.30），SigLIP LoRA 微调确认无损且小幅增益；500 集口径不与 3000 集（v8 Test F1=0.4245）直接比较，3000 集验证留待后续。default epochs=20。

当前默认训练链的工程配置（LoRA r/α、lr、batch、epoch、SigLIP2 多尺度层号、前缀 token 数等）见上方「训练数据流」与「关键路径与环境变量」章节，与 [实验记录.md](实验记录.md) §一 互为索引。

## 目录速览

- `config.yaml` — v10 训练配置唯一调参入口（路径/双 LoRA 超参/训练超参/评估控制）；优先级 env > yaml > 代码默认
- `model/` — 特征注入架构（注入主链 [qwen_feature_injection.py](model/qwen_feature_injection.py) + SigLIP2 外部模块 [external_feature_adapter.py](model/external_feature_adapter.py)（LoRA 化主干 + MultiScaleFusionDecoder 纯视觉融合） + 坐标解析 [location_tokens.py](model/location_tokens.py)（裸整数/旧 location token 兼容解析） + prompt 加载 [prompt_loader.py](model/prompt_loader.py) + 数据增强 [data_augment.py](model/data_augment.py)）
- `scripts/` — 数据转换器（YOLO / VOC → JSONL）+ 划分（`split_dataset.py` 通用 7:1:2 种子 42（500 集划分源）/ `create_split_3000.py` 3000 集划分源（排除 500 集 test/val 防泄露，二者勿混用））+ 检测演示可视化 `visualize_detection_demo.py`（原 viis.py 迁移修正）+ v10 模块冒烟 `smoke_v10_modules.py` + prompt 生成器 + captions 生成 + swa_average
- `data/` — prompt JSON（训练与推理分开）；类别映射在两份 prompt JSON 内（`label_codes` 代码 = label 取值 / `code_to_en` 英文释义 / `labels` 代码列表）；`pipe_defect_3000_prompt_train_en.json` 为训练/YOLO 转换默认，`pipe_defect_3000_prompt_infer_en.json` 为推理/评估默认；旧 `pipe_defect_3000_prompt_rawint.json` 保留为历史兼容配置。
- `F:\管网\runs\` — 全部实验产出的统一根（仓库外集中管理，与数据集同盘）；结构为 `runs\<版本>\train\`（best_adapter/final_adapter/checkpoint-*/text_prompt_cache.pt/eval_metrics.jsonl/swanlab log）与 `runs\<版本>\test\`（test_predictions.jsonl/visualizations/confusion_matrix），另有 `runs\log\`（历史 swanlab 日志）与 `runs\infer\`（Gradio 推理临时图）。历史 v7/v8/baseline_native 产物已按此结构迁入；具体版本对应哪个实验与 F1 见 [实验记录.md](实验记录.md) §六
- `.history/` — VSCode 本地历史（gitignore，勿改）
- [实验记录.md](实验记录.md) — 训练/评估实验数据汇总（val/test F1、per-class、消融、产出明细）；**写新实验数据请更新此文件，不要回写 AGENTS.md**
- [训推.md](训推.md) — 完整流程主文档；[docs/qwen3vl_lora_feature_injection.md](docs/qwen3vl_lora_feature_injection.md) — 注入方案设计与推理排查结论
