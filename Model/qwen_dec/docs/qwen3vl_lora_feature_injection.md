# Qwen3-VL 加载、网络结构与外部视觉特征注入说明

> 本文记录特征注入方案的设计推理与排查结论。**当前唯一有效的 v8 注入方案见文末「已落地实现」一节**（SigLIP2 Base 冻结 → SimpleFPN/多尺度 decoder → 16×16 = 256 前缀 token，`masked_scatter` 覆盖注入）。早期方案一/二/三及 v7 均为历史方案，仅作记录，不作为当前训练或推理入口。

## 当前模型加载方式

训练脚本通过 `transformers` 加载 Qwen3-VL 基座模型（注入子类 `FeatureInjectedQwen3VLForConditionalGeneration`），通过 `peft` 挂载 LoRA/QLoRA。

核心位置：

- `train_qwen3vl_lora.py`：`FeatureInjectedQwen3VLForConditionalGeneration.from_pretrained(...)`
- `train_qwen3vl_lora.py`：`get_peft_model(model, lora_cfg)`
- `inference.py`：`PeftModel.from_pretrained(model, adapter_path)`

训练端加载流程（bf16，无量化；`USE_4BIT=1` 在 transformers 4.57.6 下会直接抛错）：

```python
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=128 * 28 * 28,
    max_pixels=512 * 28 * 28,
)
# 坐标用 0-1000 归一化裸整数，沿用基座原生词表，不注册 location token、不 resize
model = FeatureInjectedQwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
model.bind_external_extractor(SIGLIP_MODEL_PATH, torch.bfloat16)   # SigLIP2 Base 冻结 + 空间适配层 + projector
model = get_peft_model(model, lora_cfg)                            # 不含 location token，词表保持原生 151669
```

推理端加载流程：

```python
model = FeatureInjectedQwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
# 不 resize：词表保持基座原生 151669
model.bind_external_extractor(SIGLIP_MODEL_PATH, torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter_path)
model.get_base_model().load_external_adapter(adapter_path, strict=True)  # 恢复 SigLIP2 空间适配层 + projector
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=128 * 28 * 28,
    max_pixels=512 * 28 * 28,
)
# 不注册 location token：坐标用 0-1000 归一化裸整数，沿用原生数字 token
```

## Qwen3-VL 网络结构

当前基座为 `Qwen3-VL-4B-Instruct`，由视觉编码器和语言解码器组成。

视觉侧：

- `depth = 24`
- `hidden_size = 1280`（`out_hidden_size` 投影到语言侧）
- `patch_size = 16`
- `spatial_merge_size = 2`

语言侧：

- decoder-only causal LM
- `num_hidden_layers = 36`
- `hidden_size = 2560`
- 使用 interleaved MRoPE

> 注：以上数值以 `Qwen3-VL-4B-Instruct/config.json` 为准；早期文档误写为 2B 规格（`depth=24 / hidden_size=1024 / num_hidden_layers=28` 等），那是 2B 版本，非当前基座。

整体数据流：

```text
image -> AutoProcessor -> pixel_values / image_grid_thw
prompt -> chat_template / tokenizer -> input_ids
vision encoder -> visual embeddings
visual embeddings 替换 <|image_pad|> 位置
+ 外部 SigLIP2 特征 -> 256 前缀 token 覆盖输入开头保留槽位
language decoder -> 自回归生成 <obj>...</obj> JSON bbox 文本（坐标用 0-1000 归一化裸整数）
```

当前管网缺陷检测不是传统检测头任务，而是因果语言建模任务：

```text
图像 + 检测提示 -> <obj>[{"bbox_2d":[x1,y1,x2,y2],"label":"<code>"}]</obj>
```

## 当前 LoRA 位置

当前 LoRA 同时挂在语言侧线性层和 Qwen 原生视觉塔：

```python
# 语言侧（正则匹配）
"q_proj", "k_proj", "v_proj", "o_proj",
"gate_proj", "up_proj", "down_proj"
# 视觉塔全部 block
"model.visual.blocks.N.attn.qkv",
"model.visual.blocks.N.attn.proj",
"model.visual.blocks.N.mlp.linear_fc1",
"model.visual.blocks.N.mlp.linear_fc2"
```

此外，坐标用 0-1000 归一化裸整数，沿用基座原生词表（不注册 location token、不扩展词表），故 LoRA 不含任何额外的 token embeddings。这意味着 LoRA 同时调整语言解码器与原生视觉塔。外部 SigLIP2 特征不直接进 LoRA，而是经独立的空间适配层 + projector 对齐成 hidden states 注入。

## 外部视觉特征能否注入 LoRA

外部视觉特征不应直接"注入 LoRA 参数"。LoRA 的数学形式是：

```text
W' = W + A @ B
```

它学习的是权重增量，不是动态特征容器。因此，外部视觉特征的合理接入方式是新增 adapter 或 projector，把外部特征变成 Qwen3-VL 可消费的 hidden states，再让 LoRA 学习如何利用这些 hidden states。

## 早期可行方案讨论（设计阶段备选，现已被前缀注入方案取代）

### 方案一：外部视觉特征转 soft visual tokens

流程：

```text
external_features [B,N,C_ext]
-> Linear(C_ext, hidden)
-> soft visual tokens [B,N,hidden]
-> 拼入 Qwen 输入 embedding 序列
-> language decoder 生成结果
```

风险：需要处理 attention mask、position ids、image token 对齐。

### 方案二：decoder hidden-state gated fusion

流程：

```text
external_features
-> pooling / projector
-> external_hidden [B,hidden]
-> hidden = hidden + gate * external_hidden
```

风险：需要改 Qwen3-VL forward 或在 decoder 层注册 hook；需控制 gate 初始值。

> ⚠️ 方案二的残差融合（含早期 `QwenDecoderFusion` / sigmoid 门控）**不在当前主链使用**。早期实验发现 sigmoid 门控训练后 gate 恒为初值、等于未注入，故放弃门控融合。

### 方案三：参考反方向 LanguageAdapter（已删除）

`model/language_adapter.py` 曾提供反方向模块（Qwen hidden → SAM2 维度）。该模块**已从仓库删除**，当前训练/推理主流程不使用。

## 已落地实现：SigLIP2 Base 前缀注入（当前唯一主链）

经方案对比，最终落地的是「前缀 token 覆盖注入」（方案一的安全变体），外部主干从早期 ResNet50/SAM2 收敛到 **SigLIP2 Base**：

```text
SigLIP2 Base (冻结, 768 hidden, 原生 224 -> 运行时插值 256x256 -> 16x16 patch 网格)
-> 提取第 3/7/9/12 层 hidden states 并按通道拼接 [B,256,3072]
-> SimpleFPN 派生 4S/2S/1S/0.5S 四个真多尺度特征
-> MultiScaleFusionDecoder 粗到细融合并池化回 [B,256,16,16]
-> ExternalFeatureProjector: Linear(256->512)->SiLU->Linear(512->hidden)+LayerNorm+二维正余弦位置编码 [可训]
-> 256 个 [B,256,hidden] 前缀 token
-> _inject_external_prefix: masked_scatter 覆盖输入开头 256 个保留槽位
-> 与 Qwen 原生视觉 token 一起进 language decoder
```

关键实现（`model/qwen_feature_injection.py`）：

- `EXTERNAL_PREFIX_GRID_SIZE=16`，`EXTERNAL_PREFIX_LENGTH=256`。
- prefill 阶段注入；后续解码步复用 KV cache，只注入一次（`prepare_inputs_for_generation` 透传 `external_images`/`external_hidden`）。
- 保存/加载：`save_external_adapter` 导出 `external_feature_adapter.pt`（`format_version=8`，3/7/9/12 层多深度特征，SimpleFPN 四尺度 + MultiScaleFusionDecoder，`backbone_architecture="google-siglip2-base-patch16-224-interpolated-256"`，`backbone_frozen=True`），只含可训练空间融合层 + projector；`load_external_adapter(strict=True)` 缺失、版本或主干元数据不匹配时报错，禁止静默回退。v7 三层通道拼接 adapter 与当前 v8 结构不兼容。

坐标表示：0-1000 归一化裸整数 `[x1,y1,x2,y2]`（沿用基座原生词表）。历史上曾用 Falcon 风格 location token `<loc_0>`..`<loc_999>`，但实测小数据下训不动（epoch1 F1≈0.05，fp=708 坐标完全失效），已弃用回归裸整数；训练数据里直接存裸整数，用原生数字 token 训练，标准 LM CE loss，不引入 location token。

## 当前推理问题排查结论（历史）

早期推理报错：

```text
AttributeError: 'int' object has no attribute 'get'
```

直接原因是模型输出被解析成了整数列表（如 `[317,343,681,613]`），旧版 `parse_bbox_json()` 直接返回该 list，随后 `draw_boxes()` 遍历到整数并调用 `.get()` 而崩溃。

修复原则（已实现）：

- JSON 解析后规范化为 `List[dict]`，只允许 `{"bbox_2d":[...],"label":"..."}` 进入画框逻辑。
- `decode_bbox(allow_legacy_numeric=True)` 直接按裸整数解析 bbox（亦兼容历史 location token 输出）；其他 malformed 输出仅显示 raw output，不中断 Gradio。

另一个关键问题是推理端 processor 必须与训练端一致：

```python
AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=128 * 28 * 28,
    max_pixels=512 * 28 * 28,
)
```

否则视觉 token 数量与训练时不同，容易出现检测失败或格式漂移。词表保持基座原生（不注册 location token、不 `resize_token_embeddings`），坐标用 0-1000 归一化裸整数。

## 运行命令

```powershell
python inference.py --adapter output\pipe_defect_lora_siglip2_base_v8\best_adapter
```

如果坐标协议、prompt 或数据集已经更新，旧 adapter 不会自动适配新数据，需要重新训练 LoRA。当前默认数据集为管网缺陷 `数据_筛选3000`，坐标用 0-1000 归一化裸整数（坐标格式已回归裸整数），与历史 ResNet 链的 adapter 不兼容。
