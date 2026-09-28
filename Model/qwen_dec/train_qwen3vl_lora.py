"""
Qwen3-VL-4B-Instruct + LoRA 管网缺陷目标检测微调（v10：v8 结构 + SigLIP2 主干 LoRA）。

做什么:
    读取数据集目录下的 pipe_qwenvl_train/val/test.jsonl（YOLO → Qwen3-VL JSONL 转换产物），
    构建对话样本，仅对回答段计算 loss。双视觉信号链：
      - Qwen 原生视觉塔（24 block）挂 LoRA，高分辨率原生视觉 token；
      - SigLIP2 Base（冻结主干 + LoRA 微调 q/k/v/out_proj+fc1/fc2）→ 四层多深度特征 →
        SimpleFPN → MultiScaleFusionDecoder 纯视觉融合 → 256 个前缀 token。
    bbox 坐标用 0-1000 归一化裸整数（基座原生数字 token，不引入 location token）。

为什么:
    - v8（原生塔 LoRA + 冻结 SigLIP2 前缀）被证明可训：500 集 epoch1 val F1≈0.30、
      3000 集 Test F1=0.4245（对照基线）。
    - v10 = v8 结构 + SigLIP2 主干 LoRA 单变量升级（其他结构全部与 v8 一致）。
    - 不用 trl SFTTrainer（跨版本签名不稳），改用自定义 Dataset+Collator+原生 Trainer。

环境:
    conda activate llamafactory（transformers 4.57.6 / peft 0.18.1 / qwen-vl-utils / swanlab）
    运行: python train_qwen3vl_lora.py（配置优先级：环境变量 > config.yaml > 代码默认值）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
import swanlab
from qwen_vl_utils import process_vision_info

from model.qwen_feature_injection import EXTERNAL_PREFIX_LENGTH, FeatureInjectedQwen3VLForConditionalGeneration
from model.prompt_loader import load_full_prompt

# ----------------------------------------------------------------------------
# 常量（三层配置：环境变量显式设置 > config.yaml > 代码默认值）
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent


def _load_yaml_config() -> dict:
    """入参: 无。方法: 读仓库根 config.yaml（缺失或空文件返回空 dict，不报错）。
    出参: yaml 配置 dict。"""
    path = HERE / "config.yaml"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


_YAML_CONFIG = _load_yaml_config()


def _pick(env_key: str, yaml_keys: list[str], default):
    """入参: env_key 环境变量名、yaml_keys 逐级键路径（如 ["text_lora", "r"]）、default 兜底值。
    方法: 环境变量存在则优先（保留训推手册的 env 覆盖约定）；否则按路径下钻 yaml；
          yaml 值为 null 视为未配置，回落默认值。
    出参: 解析后的配置值（类型与 yaml/默认值一致）。"""
    if env_key in os.environ:
        return os.environ[env_key]
    node: Any = _YAML_CONFIG
    for key in yaml_keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


def _to_bool(value) -> bool:
    """入参: 布尔/整数/字符串。方法: 兼容 env "1"/"0"/"true"/"false" 与 yaml bool 的统一转换。
    出参: bool。"""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


MODEL_PATH = _pick("MODEL_PATH", ["model_path"], r"F:/pre/Qwen3-VL-4B-Instruct")
DATASET_ROOT = Path(_pick("DATASET_ROOT", ["dataset_root"], r"F:/管网/数据_筛选3000"))
TRAIN_JSONL = Path(_pick("TRAIN_JSONL", ["train_jsonl"], str(DATASET_ROOT / "pipe_qwenvl_train.jsonl")))
VAL_JSONL = Path(_pick("VAL_JSONL", ["val_jsonl"], str(DATASET_ROOT / "pipe_qwenvl_val.jsonl")))
PROMPT_JSON = os.environ.get("PROMPT_JSON", "")  # 兼容回退：仅环境变量
TRAIN_PROMPT_JSON = Path(_pick(
    "TRAIN_PROMPT_JSON", ["train_prompt_json"],
    PROMPT_JSON or str(HERE / "data" / "pipe_defect_3000_prompt_train_en.json"),
))
OUTPUT_DIR = _pick("OUTPUT_DIR", ["output_dir"], r"F:/管网/runs/pipe_defect_lora_v10_siglip2_base")
# 版本内 train/ 保存权重、checkpoint 与训练日志；test/（见 evaluate_detection.py）保存评估指标与可视化。
TRAIN_DIR = Path(OUTPUT_DIR) / "train"
RESUME_FROM_CHECKPOINT = str(_pick("RESUME_FROM_CHECKPOINT", ["resume_from_checkpoint"], "")).strip()
IGNORE_INDEX = -100        # label mask 标记（仅 assistant 段参与 loss）
# 已废弃的分组 lr 常量：仅写入 swanlab config 展示，实际优化器统一 UNIFIED_LR（见 build_optimizer）。
TEXT_LORA_LR = 2e-5
VISION_LORA_LR = 5e-6
EXTERNAL_ADAPTER_LR = 5e-5
WEIGHT_DECAY = float(_pick("WEIGHT_DECAY", ["weight_decay"], 0.01))
# LoRA 超参（Qwen decoder+原生视觉塔 / SigLIP2 主干）
TEXT_LORA_R = int(_pick("TEXT_LORA_R", ["text_lora", "r"], 32))
TEXT_LORA_ALPHA = int(_pick("TEXT_LORA_ALPHA", ["text_lora", "alpha"], 64))
TEXT_LORA_DROPOUT = float(_pick("TEXT_LORA_DROPOUT", ["text_lora", "dropout"], 0.05))
SIGLIP2_LORA_R_CFG = int(_pick("SIGLIP2_LORA_R", ["siglip_lora", "r"], 32))
SIGLIP2_LORA_ALPHA_CFG = int(_pick("SIGLIP2_LORA_ALPHA", ["siglip_lora", "alpha"], 64))
SIGLIP2_LORA_DROPOUT_CFG = float(_pick("SIGLIP2_LORA_DROPOUT", ["siglip_lora", "dropout"], 0.05))
# 训练超参（TrainingArguments）
NUM_TRAIN_EPOCHS = int(_pick("NUM_TRAIN_EPOCHS", ["epochs"], 30))
PER_DEVICE_TRAIN_BATCH_SIZE = int(_pick("PER_DEVICE_TRAIN_BATCH_SIZE", ["per_device_train_batch_size"], 2))
GRADIENT_ACCUMULATION_STEPS = int(_pick("GRADIENT_ACCUMULATION_STEPS", ["gradient_accumulation_steps"], 4))
LEARNING_RATE = float(_pick("LEARNING_RATE", ["learning_rate"], 1e-4))
LR_SCHEDULER_TYPE = str(_pick("LR_SCHEDULER_TYPE", ["lr_scheduler_type"], "constant"))
WARMUP_STEPS_CFG = int(_pick("WARMUP_STEPS", ["warmup_steps"], 0))
MAX_GRAD_NORM = float(_pick("MAX_GRAD_NORM", ["max_grad_norm"], 1.0))
OPTIM = str(_pick("OPTIM", ["optim"], "paged_adamw_8bit"))
# 分块 CE：训练 forward 以分块 lm_head+CE 替代全词表 logits（Windows 无 triton，用纯 torch 等价实现）
CHUNKED_CE = _to_bool(_pick("CHUNKED_CE", ["chunked_ce"], False))
SAVE_TOTAL_LIMIT = int(_pick("SAVE_TOTAL_LIMIT", ["save_total_limit"], 2))
# Norm 全量微调开关（消融实验已证明有害，默认关闭）
NORM_MODULES_TO_SAVE = _to_bool(_pick("NORM_MODULES_TO_SAVE", ["norm_modules_to_save"], False))
# 早停：连续 patience 次 val 评估 F1 无提升则终止训练。计数单位是「评估次数」(非 epoch 数，
# 因 VAL_EVAL_INTERVAL>1 时评估本就降频)。默认 0=关闭早停(跑满 num_train_epochs)；设 >0 开启。
F1_EARLY_STOPPING_PATIENCE = int(_pick("F1_EARLY_STOPPING_PATIENCE", ["f1_early_stopping_patience"], 0))
SWANLAB_PROJECT = _pick("SWANLAB_PROJECT", ["swanlab_project"], "qwen3vl-pipe-defect-detect")
SWANLAB_MODE = str(_pick("SWANLAB_MODE", ["swanlab_mode"], "local"))
SWANLAB_LOGDIR = _pick("SWANLAB_LOGDIR", ["swanlab_logdir"], str(TRAIN_DIR / "log"))
# 外部特征提取器固定为 google/siglip2-base-patch16-224；可用本地目录或 yaml 覆盖。
SIGLIP_MODEL_PATH = Path(_pick(
    "SIGLIP_MODEL_PATH", ["siglip_model_path"],
    r"F:/pre/siglip2-base-patch16-224",
))
# val 评估控制（每 epoch 末逐图 generate 很慢，故默认降频 + 可子采样以加速）：
# VAL_EVAL_INTERVAL=3 表示每 3 个 epoch 评估一次；VAL_EVAL_SIZE=300 表示用前 300 张 val 图（默认全量）。
VAL_EVAL_INTERVAL = int(_pick("VAL_EVAL_INTERVAL", ["val_eval_interval"], 3))
VAL_EVAL_SIZE = int(_pick("VAL_EVAL_SIZE", ["val_eval_size"], 300))
# 数据增强开关：训练集 50% 水平翻转（bbox 同步变换）
AUGMENT_TRAIN = _to_bool(_pick("AUGMENT_TRAIN", ["augment_train"], True))

# 保留环境变量用于显式检测旧配置；当前外部特征训练链固定使用 bf16。
USE_4BIT = _to_bool(_pick("USE_4BIT", ["use_4bit"], False))
MIN_PIXELS = 128 * 28 * 28          # Qwen-VL 视觉最小 patch 数下限（避免过低分辨率）
MAX_PIXELS = 512 * 28 * 28          # 视觉像素上限：限制图像 token 数 → 省激活显存

USER_PROMPT = load_full_prompt(TRAIN_PROMPT_JSON)


# ----------------------------------------------------------------------------
# JSONL → Qwen 消息（OpenAI content list 格式，供 chat_template 与 process_vision_info 用）
# ----------------------------------------------------------------------------
def record_to_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """入参: JSONL 单条 {"id","image","conversations":[{from,value},{from,value}]}。
    方法: 图像路径来自记录，user 文本统一取 prompt 真源，assistant 文本保留标注结果。
    出参: [{"role":"user","content":[{type:image},{type:text}]},{"role":"assistant","content":str}]。"""
    image_path = record["image"]
    assistant_text = record["conversations"][1]["value"]  # 裸整数坐标，直接用原始标注文本
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]


# ----------------------------------------------------------------------------
# 单样本 → input_ids + labels + pixel_values
# ----------------------------------------------------------------------------
def build_inputs(processor: AutoProcessor, record: Dict[str, Any]) -> Dict[str, Any]:
    """入参: processor, 单条 JSONL 记录。
    方法:
      1) 构造前缀(system+user+<|im_start|>assistant\n) 与完整(前缀+回答+<|im_end|>) 文本。
      2) 用 processor 编码完整序列（带图像，会展开 vision 占位符为多个 <|image_pad|> token）。
      3) 定位 assistant 段起点: 在完整 token 序列里查找前缀的「真实 token 化」长度——
         前缀必须经「同一 processor」编码(而非裸 tokenizer，否则 <|image_pad|> 不展开导致 n_prefix 偏小，
         会让 user 段 + 图像 token 错误地参与 loss)。因此用「单条前缀文本(无图像会被 processor 当作
         无图，但仍展开占位符文本)」不可靠，改为直接用字符串锚点:
         完整文本里 assistant 段以 '<|im_start|>assistant\n' 起始，找到其后的回答起始字节偏移，
         再用 processor.tokenizer 的字符→token 映射定位 token 位置。
      4) labels = input_ids.clone()，assistant 段之前置 IGNORE_INDEX，保留完整序列。
         仅用标准 LM CE（裸整数坐标沿用基座原生词表，无专项 loss）。
    出参: 含 LM labels 与两条视觉链输入的单样本字典。"""
    messages = record_to_messages(record)
    assistant_text = messages[1]["content"]
    tok = processor.tokenizer
    im_end = tok.eos_token  # <|im_end|> 在该词表即 eos_token

    # 抽取图像（实测返回 2 元组；取前两项）
    vision_out = process_vision_info([messages])
    images, videos = vision_out[0], vision_out[1]

    # assistant 段锚点：模板里固定为 <|im_start|>assistant\n，回答紧随其后
    assistant_anchor = "<|im_start|>assistant\n"
    prefix_text = processor.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )  # 末尾即 assistant_anchor
    full_text = prefix_text + assistant_text + im_end + "\n"

    # 编码完整序列（带图像）
    inputs = processor(
        text=[full_text], images=images, videos=videos,
        padding="longest", return_tensors="pt",
    )
    input_ids: torch.Tensor = inputs["input_ids"][0]

    # 定位 assistant 回答起始 token：回答前缀 assistant_anchor 在 token 序列中的位置
    anchor_ids = tok(assistant_anchor, add_special_tokens=False)["input_ids"]
    answer_ids = tok(assistant_text, add_special_tokens=False)["input_ids"]
    ids_list = input_ids.tolist()

    def find_subseq(hay, needle):
        """入参: hay 为完整 token 列表，needle 为回答锚点 token; 方法: 顺序查找首个连续子序列; 出参: 起始下标或 -1。"""
        n, m = len(hay), len(needle)
        for i in range(n - m + 1):
            if hay[i:i + m] == needle:
                return i
        return -1

    a_idx = find_subseq(ids_list, anchor_ids)
    if a_idx < 0:
        # 兜底：锚点未命中，退化为仅 mask 固定前几个 token（不应发生）
        n_prefix = max(0, len(ids_list) - len(answer_ids) - 2)
    else:
        n_prefix = a_idx + len(anchor_ids)  # 回答第一个 token 的下标

    labels = input_ids.clone()
    labels[:n_prefix] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "attention_mask": inputs["attention_mask"][0, :input_ids.shape[0]],
        "labels": labels,
        # Qwen-VL 的 pixel_values 形如 [total_patches, C*P*P]（所有图 patch 拼成 1D 序列，无 batch 维），
        # image_grid_thw 形如 [N_images, 3]。二者为「整批图像」级张量，不能按样本切片，
        # 也不能 stack。此处保留原始（单样本即整批）张量，由 collator 用 cat 在 batch 间拼接。
        "pixel_values": inputs["pixel_values"] if "pixel_values" in inputs else None,
        "image_grid_thw": inputs["image_grid_thw"] if "image_grid_thw" in inputs else None,
        "external_image": record["image"],
    }


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class QwenDetectionDataset(Dataset):
    """入参: jsonl_path, processor, augment(默认 False)。
    方法: 读全部 JSONL 行缓存为 list；__getitem__ 调 build_inputs 得单样本张量。
          augment=True 时在 build_inputs 前对训练样本做水平翻转（同步变换 bbox）。
    出参: 单样本 dict（由 collator 拼 batch）。"""

    def __init__(self, jsonl_path: str, processor: AutoProcessor, augment: bool = False):
        """入参: JSONL 数据路径、Qwen processor、是否训练期增强; 方法: 逐行解析有效样本并缓存; 出参: 数据集实例。"""
        self.records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.processor = processor
        self.augment = augment

    def __len__(self) -> int:
        """入参: 无; 方法: 读取已缓存样本数量; 出参: 数据集长度。"""
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """入参: idx 为有效样本下标; 方法: 训练增强(可选)+build_inputs 编码; 出参: 单样本训练张量字典。"""
        record = self.records[idx]
        if self.augment:
            from model.data_augment import maybe_augment_record
            record = maybe_augment_record(record, p=0.5)
        return build_inputs(self.processor, record)


# ----------------------------------------------------------------------------
# Collator：变长序列左 padding 到 batch 内最长
# ----------------------------------------------------------------------------
@dataclass
class Qwen3VLCollator:
    """入参: pad_token_id (右 padding 数值), processor。
    方法:
      - input_ids/labels/attention_mask 按 batch 最长左 padding（attention_mask 右侧补 0）。
        左 padding 保证生成时序列前缀连续，labels padding 处补 IGNORE_INDEX；
        左 padding 之后统一在每样本最前插入 EXTERNAL_PREFIX_LENGTH 个前缀槽位
        （槽位 embedding 会被 SigLIP2 前缀 masked_scatter 覆盖，labels 恒为 IGNORE_INDEX）。
      - pixel_values / image_grid_thw 各样本在第 0 维 cat（图像信息天然等维）。
    出参: batch dict，所有张量带 batch 维度 [B,...]。"""

    pad_token_id: int
    processor: Any

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """入参: batch 为变长单样本字典序列; 方法: 左侧补齐文本并拼接视觉 patch 与外部图像路径; 出参: Trainer 批张量字典。"""
        L = max(item["input_ids"].shape[0] for item in batch)
        input_ids, labels, attn = [], [], []
        for item in batch:
            n = item["input_ids"].shape[0]
            pad = L - n
            # 左 padding + 前缀槽位（pad token 占位，仅 embedding 会被覆盖）
            input_ids.append(
                torch.cat([torch.full((pad,), self.pad_token_id, dtype=item["input_ids"].dtype),
                           torch.full((EXTERNAL_PREFIX_LENGTH,), self.pad_token_id, dtype=item["input_ids"].dtype),
                           item["input_ids"]])
            )
            labels.append(
                torch.cat([torch.full((pad,), IGNORE_INDEX, dtype=item["labels"].dtype),
                           torch.full((EXTERNAL_PREFIX_LENGTH,), IGNORE_INDEX, dtype=item["labels"].dtype),
                           item["labels"]])
            )
            attn.append(
                torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(EXTERNAL_PREFIX_LENGTH, dtype=torch.long),
                           item["attention_mask"].long()])
            )

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
        }
        # pixel_values / image_grid_thw 为「整批图像」级拼接张量（无 batch 维），
        # 用 cat 沿第 0 维（patch 维 / 图维）拼接各样本，而非 stack。
        if batch[0]["pixel_values"] is not None:
            result["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
        if batch[0]["image_grid_thw"] is not None:
            result["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        # external_images 保留原始类型：路径保持 str，翻转后的 PIL 保持 PIL.Image 对象。
        # 不能用 str() 强转（会把 PIL 变成 repr 字符串，下游 Image.open 崩溃）。
        result["external_images"] = [item["external_image"] for item in batch]
        return result


# ----------------------------------------------------------------------------
# Best adapter 回调：每 epoch 对 val 跑检测评估，按最优 F1 完整保存 best adapter
# ----------------------------------------------------------------------------
class BestByF1Callback(TrainerCallback):
    """入参: peft_model, feature_model(基座含 external 模块), processor, val_records(val JSONL 记录),
          eval_prompt(与训练同源的 user prompt), save_dir, metrics_path, iou_threshold。
    方法: 每 epoch 结束对 val 集逐图 generate → 解析 <obj> bbox → 与同类别 GT 按 IoU 匹配 → 算 P/R/F1；
          F1 创新高时完整保存 adapter（LoRA + external + processor）到 save_dir（三者同源一致，
          规避 PEFT checkpoint 只存 LoRA 的错配）；每轮指标追加到 metrics_path（JSONL）。
          best 按 F1 而非 eval_loss：检测任务的定位+分类质量只有 F1 能反映，loss 不能。
          不因 F1 停滞而提前终止训练，Trainer 按 num_train_epochs 完整运行。
    出参: 无返回；副作用为写 best_adapter 与 eval_metrics.jsonl。"""

    def __init__(self, peft_model, feature_model, processor, val_records, eval_prompt,
                 save_dir, metrics_path, iou_threshold=0.5,
                 eval_interval=1, eval_size=None, early_stopping_patience=0):
        """入参: 训练模型、验证数据、提示词、保存路径、IoU 阈值、评估间隔(每 N epoch 评估一次)、
                 评估子采样大小(None 或 <=0 表示全量)、早停耐心(连续多少次评估 F1 无提升即停，<=0 关闭);
           方法: 保存评估依赖并初始化最优 F1; 出参: 回调实例。"""
        self.peft_model = peft_model
        self.feature_model = feature_model
        self.processor = processor
        self.val_records = val_records
        self.eval_prompt = eval_prompt
        self.save_dir = save_dir
        self.metrics_path = metrics_path
        self.iou_threshold = iou_threshold
        self.eval_interval = max(1, int(eval_interval))
        self.eval_size = int(eval_size) if eval_size and int(eval_size) > 0 else None
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self.best_f1 = -1.0
        self.epochs_without_improvement = 0
        self._restore_history()

    def _restore_history(self) -> None:
        """入参: 无; 方法: 从已有指标恢复最高 F1 和末尾未提升计数，支持中断续训; 出参: None。"""
        path = Path(self.metrics_path)
        if not path.is_file():
            return
        metrics = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not metrics:
            return
        f1_values = [float(item["f1"]) for item in metrics if "f1" in item]
        if not f1_values:
            return
        self.best_f1 = max(f1_values)
        last_best = max(index for index, value in enumerate(f1_values) if value == self.best_f1)
        self.epochs_without_improvement = len(f1_values) - last_best - 1
        print(f"[resume] restored best F1={self.best_f1:.4f}, stale_epochs={self.epochs_without_improvement}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """入参: Trainer 标准回调参数。方法: 按 eval_interval 决定是否评估，评估时跑 val 检测 F1 选 best；
        非评估 epoch 跳过（仅打印跳过提示），不触发早停。出参: 无。"""
        epoch = float(state.epoch)
        # 按 interval 降频：仅当当前 epoch 是 interval 的整数倍、或到达最后一个 epoch 时才评估。
        is_last = epoch >= args.num_train_epochs - 1e-6
        if (round(epoch) % self.eval_interval != 0) and not is_last:
            print(f"[eval-skip] epoch={epoch:.1f} (interval={self.eval_interval}, skip until epoch "
                  f"{(round(epoch) // self.eval_interval + 1) * self.eval_interval}); continuing training.")
            return
        metrics = self._evaluate_f1()
        metrics["epoch"] = epoch
        self._append_metrics(metrics)
        print(f"[eval] epoch={metrics['epoch']:.1f} P={metrics['precision']:.4f} "
              f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
              f"(tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']})")
        if metrics["f1"] > self.best_f1:
            self.best_f1 = metrics["f1"]
            self.epochs_without_improvement = 0
            os.makedirs(self.save_dir, exist_ok=True)
            self.peft_model.save_pretrained(self.save_dir)
            self.feature_model.save_external_adapter(self.save_dir)
            self.processor.save_pretrained(self.save_dir)
            print(f"[best] F1={metrics['f1']:.4f} (new best) -> full adapter saved to {self.save_dir}")
        else:
            self.epochs_without_improvement += 1
            if self.early_stopping_patience > 0 and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"[early-stop] F1 unchanged for {self.epochs_without_improvement} eval(s) "
                      f"(>= patience {self.early_stopping_patience}); stopping training.")
                control.should_training_stop = True
            else:
                es = f", early_stop at {self.early_stopping_patience}" if self.early_stopping_patience > 0 else ", early_stop off"
                print(f"[no-improve] F1 unchanged for {self.epochs_without_improvement} eval(s){es}; continuing training.")

    def _evaluate_f1(self):
        """入参: 无（用 self.val_records/eval_prompt/peft_model）。
        方法: 临时切 eval + 开 use_cache（generate 必需，训练期默认 False），逐图 run_inference
              → parse_bbox_json → 与同类别 GT 按 IoU 匹配，累计 TP/FP/FN；finally 恢复 use_cache 与训练态。
              按 self.eval_size 对 val 集做确定性前缀子采样（默认 None=全量）以加速评估。
        出参: {precision, recall, f1, tp, fp, fn}。"""
        from PIL import Image
        from inference import run_inference, parse_bbox_json
        from detection_metrics import match_image, precision_recall_f1
        from evaluate_detection import ground_truth_boxes, normalized_targets_to_pixels
        was_training = self.peft_model.training
        self.peft_model.eval()
        prev_use_cache = getattr(self.feature_model.config, "use_cache", None)
        self.feature_model.config.use_cache = True
        eval_records = self.val_records if self.eval_size is None else self.val_records[:self.eval_size]
        tp = fp = fn = 0
        try:
            with torch.no_grad():
                for rec in eval_records:
                    image_path = rec["image"]
                    with Image.open(image_path) as im:
                        w, h = im.size
                    raw, _, _ = run_inference(self.peft_model, self.processor, image_path, self.eval_prompt)
                    pred_boxes, pred_labels = normalized_targets_to_pixels(parse_bbox_json(raw), w, h)
                    gt_boxes, gt_labels = ground_truth_boxes(rec, w, h)
                    m = match_image(pred_boxes, gt_boxes, pred_labels, gt_labels, self.iou_threshold)
                    tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
        finally:
            self.feature_model.config.use_cache = prev_use_cache
            if was_training:
                self.peft_model.train()
        prf = precision_recall_f1(tp, fp, fn)
        return {"precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
                "tp": tp, "fp": fp, "fn": fn}

    def _append_metrics(self, metrics):
        """入参: metrics 指标 dict。方法: 追加一行 JSON 到 self.metrics_path。出参: 无。"""
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


class ExternalAdapterCheckpointCallback(TrainerCallback):
    """入参: feature_model; 方法: 每个 Trainer checkpoint 同步保存 SigLIP2 LoRA/空间适配层和 projector; 出参: 无。"""

    def __init__(self, feature_model):
        """入参: 含外部特征 adapter 的基础模型；方法: 保存 checkpoint 回调依赖；出参: 回调实例。"""
        self.feature_model = feature_model

    def on_save(self, args, state, control, **kwargs):
        """入参: Trainer 保存事件参数；方法: 将外部特征权重写入当前 checkpoint；出参: 无。"""
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.feature_model.save_external_adapter(checkpoint_dir)
        print(f"[checkpoint] external adapter saved to {checkpoint_dir}")


def _is_no_decay_parameter(name: str) -> bool:
    """入参: 参数名; 方法: 识别 bias 与归一化参数; 出参: 是否禁用 weight decay。"""
    lowered = name.lower()
    return name.endswith(".bias") or "norm" in lowered or ".bn" in lowered


def build_optimizer(model):
    """入参: 已注入 PEFT 和外部模块的模型; 方法: 按文本 LoRA、Qwen 视觉塔 LoRA 与外部链
           （SigLIP2 LoRA + FPN/融合 decoder + projector）分组构建 8bit AdamW; 出参: optimizer。"""
    try:
        from bitsandbytes.optim import PagedAdamW8bit
    except ImportError as exc:
        raise RuntimeError("paged_adamw_8bit requires bitsandbytes in the active environment.") from exc

    # 所有参数组统一 lr=1e-4 constant（历史 baseline 配置），不分组差异化
    UNIFIED_LR = 1e-4
    groups = {
        "text_lora": {"lr": UNIFIED_LR, "weight_decay": 0.0, "params": []},
        "vision_lora": {"lr": UNIFIED_LR, "weight_decay": 0.0, "params": []},
        "external_decay": {"lr": UNIFIED_LR, "weight_decay": WEIGHT_DECAY, "params": []},
        "external_no_decay": {"lr": UNIFIED_LR, "weight_decay": 0.0, "params": []},
        "norm_to_save": {"lr": UNIFIED_LR, "weight_decay": 0.0, "params": []},  # norm 全量微调（modules_to_save）
    }
    counts = {name: 0 for name in groups}
    unknown = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "modules_to_save" in name:
            # PEFT 的 modules_to_save 包装的 norm 参数（weight/bias 全量可训）
            group_name = "norm_to_save"
        elif "external_" in name:
            # SigLIP2 主干 LoRA + FPN/融合 decoder + projector（含 lora_ 参数名）
            group_name = "external_no_decay" if _is_no_decay_parameter(name) else "external_decay"
        elif "lora_" in name and ".visual.blocks." in name:
            group_name = "vision_lora"
        elif "lora_" in name:
            group_name = "text_lora"
        else:
            unknown.append(name)
            continue
        groups[group_name]["params"].append(param)
        counts[group_name] += param.numel()
    if unknown:
        raise RuntimeError(f"Unexpected trainable parameters: {unknown[:10]}")
    active_groups = [group for group in groups.values() if group["params"]]
    if not active_groups:
        raise RuntimeError("No trainable parameters were assigned to optimizer groups.")
    for name, group in groups.items():
        print(f"[optimizer] {name}: params={counts[name]:,} lr={group['lr']:.1e} wd={group['weight_decay']}")
    return PagedAdamW8bit(active_groups)


def find_resume_checkpoint(output_dir: str, requested: str) -> str | None:
    """入参: 输出目录与 RESUME_FROM_CHECKPOINT 值; 方法: 解析显式路径或 latest; 出参: checkpoint 路径或 None。"""
    if not requested:
        return None
    if requested.lower() != "latest":
        path = Path(requested)
    else:
        candidates = list(Path(output_dir).glob("checkpoint-*"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directory found in {output_dir}.")
        path = max(candidates, key=lambda item: int(item.name.removeprefix("checkpoint-")))
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return str(path)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def load_training_processor() -> tuple[AutoProcessor, int]:
    """入参: 无，模型路径和视觉像素上下限来自模块配置。
    方法: 加载训练/推理一致的 processor（基座原生词表，不注册 location token），并解析左 padding token。
    出参: (processor, pad_token_id)；词表没有 pad token 时使用 eos token。"""
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    return processor, pad_token_id


def load_training_datasets(processor: AutoProcessor) -> tuple[QwenDetectionDataset, QwenDetectionDataset]:
    """入参: 已加载的 Qwen processor。
    方法: 使用同一数据编码器加载 train/val JSONL，只有训练集按开关启用增强。
    出参: (train_dataset, val_dataset)；验证集固定关闭增强。"""
    train_ds = QwenDetectionDataset(str(TRAIN_JSONL), processor, augment=AUGMENT_TRAIN)
    val_ds = QwenDetectionDataset(str(VAL_JSONL), processor, augment=False)
    print(
        f"train={len(train_ds)} val={len(val_ds)} augment_train={AUGMENT_TRAIN} "
        f"coord_format=raw_integer (no location tokens)"
    )
    return train_ds, val_ds


def load_training_model() -> FeatureInjectedQwen3VLForConditionalGeneration:
    """入参: 无。
    方法: 以 bf16、SDPA 和自动设备映射加载 Qwen3-VL（基座原生词表，不 resize、不注册 location token）；
         外部链不支持 4bit。
    出参: 已加载且开启 gradient checkpointing 的基础模型。"""
    if USE_4BIT:
        raise RuntimeError("USE_4BIT is unsupported for the current external feature training path; use bf16.")
    model = FeatureInjectedQwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print("[model] loaded in bf16")
    return model


def configure_training_model(
    model: FeatureInjectedQwen3VLForConditionalGeneration,
    train_ds: QwenDetectionDataset,
    processor: AutoProcessor,
) -> tuple[Any, Any]:
    """入参: 基础模型（基座原生词表）、训练数据集与 processor。
    方法: 绑定 SigLIP2 LoRA 外部特征链，初始化 projector，挂载语言/原生视觉塔 LoRA。
         仅用标准 LM CE（裸整数坐标沿用基座原生词表，无 location token、无专项 loss）。
    出参: (peft_model, feature_model)；前者交给 Trainer，后者负责外部 adapter 保存与训练开关。"""
    model.bind_external_extractor(
        SIGLIP_MODEL_PATH,
        torch.bfloat16,
        siglip_lora_r=SIGLIP2_LORA_R_CFG,
        siglip_lora_alpha=SIGLIP2_LORA_ALPHA_CFG,
        siglip_lora_dropout=SIGLIP2_LORA_DROPOUT_CFG,
    )
    model.initialize_external_projector(train_ds.records[0]["image"])
    # v10 完整版：Qwen 原生视觉塔 24 block + 语言 decoder 双路 LoRA；
    # 负向前瞻排除 external_* 子模块（SigLIP2 已有独立内层 LoRA，避免外层重复包装）。
    lora_targets = (
        r"^(?!.*external_)"
        r"(?:"
        r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"
        r"|.*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        r")$"
    )
    modules_to_save = [
        "input_layernorm",
        "post_attention_layernorm",
        "q_norm",
        "k_norm",
        "norm",
        "norm1",
        "norm2",
    ] if NORM_MODULES_TO_SAVE else []
    lora_cfg = LoraConfig(
        r=TEXT_LORA_R,
        lora_alpha=TEXT_LORA_ALPHA,
        lora_dropout=TEXT_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
        modules_to_save=modules_to_save or None,
    )
    model = get_peft_model(model, lora_cfg)
    feature_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    feature_model.enable_external_training()
    feature_model.use_chunked_ce = CHUNKED_CE
    model.print_trainable_parameters()
    return model, feature_model


def main() -> None:
    """入参: 无，配置来自 config.yaml/环境变量; 方法: 构建数据、Qwen3-VL/SigLIP2 LoRA 与 Trainer 后训练保存; 出参: None。"""
    os.makedirs(TRAIN_DIR, exist_ok=True)

    processor, pad_token_id = load_training_processor()
    train_ds, val_ds = load_training_datasets(processor)
    model, feature_model = configure_training_model(
        load_training_model(),
        train_ds,
        processor,
    )

    collator = Qwen3VLCollator(pad_token_id=pad_token_id, processor=processor)

    args = TrainingArguments(
        output_dir=str(TRAIN_DIR),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS_CFG,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        bf16=True,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="no",  # 检测 F1 由 BestByF1Callback 在 epoch 末计算
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=OPTIM,
        remove_unused_columns=False,
        report_to="swanlab",
        dataloader_num_workers=0,           # Windows 下避免多进程读图卡死
    )

    swanlab.init(project=SWANLAB_PROJECT, logdir=SWANLAB_LOGDIR, mode=SWANLAB_MODE, config={
        "model": "Qwen3-VL-4B-Instruct",
        "quant": "bf16",
        "lora_r": TEXT_LORA_R, "lora_alpha": TEXT_LORA_ALPHA, "lora_dropout": TEXT_LORA_DROPOUT,
        "siglip_lora_r": SIGLIP2_LORA_R_CFG, "siglip_lora_alpha": SIGLIP2_LORA_ALPHA_CFG,
        "norm_modules_to_save": NORM_MODULES_TO_SAVE,
        "text_lora": "all decoder q/k/v/o/gate/up/down projections",
        "vision_lora": "all 24 visual blocks: qkv/proj/fc1/fc2 (v8-identical)",
        "qwen_native_vision": "lora-trainable",
        "external_feature": "siglip2-base LoRA(q/k/v/out+fc1/fc2) + SimpleFPN + MultiScaleFusionDecoder + 16x16 MLP prefix",
        "siglip_model_path": str(SIGLIP_MODEL_PATH),
        "siglip_pretrained_input_size": 224,
        "siglip_runtime_input_size": 256,
        "siglip_patch_grid": "16x16",
        "warmup_steps": WARMUP_STEPS_CFG,
        "text_lora_lr": TEXT_LORA_LR, "vision_lora_lr": VISION_LORA_LR,
        "external_adapter_lr": EXTERNAL_ADAPTER_LR,
        "epochs": NUM_TRAIN_EPOCHS, "batch": PER_DEVICE_TRAIN_BATCH_SIZE, "grad_accum": GRADIENT_ACCUMULATION_STEPS,
        "lr_scheduler": LR_SCHEDULER_TYPE,
        "early_stopping_patience": F1_EARLY_STOPPING_PATIENCE,
        "unified_lr": LEARNING_RATE,
        "coordinate_format": "raw_integer 0-1000 (base vocab, no location tokens)",
        "loss": "standard LM CE only",
        "val_eval_interval": VAL_EVAL_INTERVAL,
        "val_eval_size": VAL_EVAL_SIZE,
        "augment_train": AUGMENT_TRAIN, "chunked_ce": CHUNKED_CE,
        "goal": "v10 = v8 structure completely + SigLIP2 backbone LoRA (single variable vs v8)",
    })

    best_adapter_dir = os.path.join(TRAIN_DIR, "best_adapter")
    eval_metrics_path = os.path.join(TRAIN_DIR, "eval_metrics.jsonl")
    # val 生成直接复用 prompt 真源，避免使用 JSONL 中可能过期的 user 文本。
    eval_prompt = USER_PROMPT
    # best 按 F1（非 loss）：每 epoch 对 val 集 generate→IoU 匹配→F1，最优时完整保存 adapter。
    optimizer = build_optimizer(model)
    best_callback = BestByF1Callback(
        model, feature_model, processor, val_ds.records,
        eval_prompt, best_adapter_dir, eval_metrics_path,
        eval_interval=VAL_EVAL_INTERVAL, eval_size=VAL_EVAL_SIZE,
        early_stopping_patience=F1_EARLY_STOPPING_PATIENCE,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        optimizers=(optimizer, None),
        callbacks=[best_callback, ExternalAdapterCheckpointCallback(feature_model)],
    )
    resume_checkpoint = find_resume_checkpoint(str(TRAIN_DIR), RESUME_FROM_CHECKPOINT)
    if resume_checkpoint:
        external_checkpoint = Path(resume_checkpoint) / "external_feature_adapter.pt"
        if not external_checkpoint.is_file():
            raise FileNotFoundError(
                f"Cannot resume from legacy checkpoint without external_feature_adapter.pt: {resume_checkpoint}"
            )
        feature_model.load_external_adapter(resume_checkpoint, strict=True)
        print(f"[resume] restored external state from {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    # 保存结束时状态；部署和正式评估仍优先使用按 F1 保存的 best_adapter。
    final_adapter_dir = os.path.join(TRAIN_DIR, "final_adapter")
    model.save_pretrained(final_adapter_dir)
    external_path = feature_model.save_external_adapter(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)
    swanlab.finish()
    print(f"training done, last-state adapter saved to {final_adapter_dir}")
    print(f"external feature adapter saved to {external_path}")
    print(f"best (highest val F1) adapter saved to {best_adapter_dir}  <- use this for inference/eval")


if __name__ == "__main__":
    main()