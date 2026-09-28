"""FeatureInjectedQwen3VL（v10：v8 前缀注入结构 + SigLIP2 主干 LoRA 微调）。

做什么: Qwen 原生视觉塔保留（LoRA 微调），联合 SigLIP2 链（LoRA 微调主干 → SimpleFPN →
       MultiScaleFusionDecoder 纯视觉融合 → 16×16 前缀）提供双重视觉信号；prefill 阶段把
       256 个外部前缀 token 以 masked_scatter 覆盖输入开头保留槽位后与原生视觉 token、
       文本一起进入语言解码器。
为什么: v8（原生塔 LoRA + 冻结 SigLIP2 前缀）被证明可训（500 集 epoch1 F1≈0.30）；v10 仅把
       SigLIP2 主干升级为 LoRA 微调，其余结构与 v8 严格一致，隔离「主干微调」这一单变量。
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

from model.external_feature_adapter import (
    EXTERNAL_FEATURE_CHANNELS,
    ExternalFeatureProjector,
    SIGLIP2_PATCH_GRID_SIZE,
    Siglip2BaseFeatureExtractor,
)


EXTERNAL_ADAPTER_FILE = "external_feature_adapter.pt"
EXTERNAL_PREFIX_GRID_SIZE = SIGLIP2_PATCH_GRID_SIZE
EXTERNAL_PREFIX_LENGTH = EXTERNAL_PREFIX_GRID_SIZE ** 2  # 16×16 = 256 个外部视觉前缀 token


class FeatureInjectedQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """入参: Qwen3-VL config 与外部图像; 方法: 将 SigLIP2 的 16×16 patch token 作为前缀送入语言解码器; 出参: causal LM 输出。"""

    def __init__(self, config):
        """入参: Qwen3-VL config; 方法: 初始化 16×16 外部特征投影，SigLIP2 Base 延迟绑定; 出参: 前缀注入版 Qwen3-VL。"""
        super().__init__(config)
        self.external_hidden_size = int(config.text_config.hidden_size)
        self.external_projector = ExternalFeatureProjector(
            self.external_hidden_size,
            input_dim=256,
            output_grid_size=EXTERNAL_PREFIX_GRID_SIZE,
        )
        self.external_extractor = None
        self.use_chunked_ce = False  # 训练端开关：True 时 forward 走分块 CE（不物化全词表 logits）
        self.loaded_external_adapter_metadata = None

    def bind_external_extractor(
        self,
        model_path: str | Path,
        dtype: torch.dtype,
        siglip_lora_r: int | None = None,
        siglip_lora_alpha: int | None = None,
        siglip_lora_dropout: float | None = None,
    ) -> None:
        """入参: SigLIP2 Base 本地模型目录或模型 ID、计算精度，以及可选主干 LoRA 超参
              （None 时用模块默认常量；训练端从 config.yaml 透传，推理端保持默认并靠
              load_external_adapter 的元数据校验发现错配）。
        方法: 加载 LoRA 化视觉编码器，并初始化 256-token projector。
        出参: 无；模型目录不存在时抛出 FileNotFoundError。"""
        device = self._external_device()
        self.external_projector = ExternalFeatureProjector(
            self.external_hidden_size,
            input_dim=256,
            output_grid_size=EXTERNAL_PREFIX_GRID_SIZE,
        )
        extractor_kwargs = {}
        if siglip_lora_r is not None:
            extractor_kwargs["lora_r"] = siglip_lora_r
        if siglip_lora_alpha is not None:
            extractor_kwargs["lora_alpha"] = siglip_lora_alpha
        if siglip_lora_dropout is not None:
            extractor_kwargs["lora_dropout"] = siglip_lora_dropout
        self.external_extractor = Siglip2BaseFeatureExtractor(
            model_path=model_path,
            device=device,
            dtype=dtype,
            **extractor_kwargs,
        )
        self.external_projector.to(device=device, dtype=dtype)

    def initialize_external_projector(self, image_path: str) -> None:
        """入参: 任意训练图像路径; 方法: 跑通 SigLIP2、空间适配层与 16×16 projector 并校验 token 形状; 出参: None。"""
        if self.external_extractor is None:
            raise RuntimeError("SigLIP2 extractor must be bound before projector initialization.")
        with torch.no_grad():
            features = self.external_extractor([image_path])
            _ = self.external_projector(features)

    def enable_external_training(self) -> None:
        """入参: 无; 方法: 打开 FPN/融合 decoder、projector 与 SigLIP2 主干 LoRA 的梯度
              （冻结主干保持 requires_grad=False）; 出参: None。"""
        if self.external_extractor is None:
            raise RuntimeError("SigLIP2 extractor must be bound before enabling training.")
        for module in (self.external_extractor.feature_adapter, self.external_projector):
            module.train()
            for param in module.parameters():
                param.requires_grad = True
        for name, param in self.external_extractor.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def _external_adapter_metadata(self) -> dict:
        """入参: 无; 方法: 生成外部 adapter 的结构与训练状态元数据; 出参: 可序列化元数据字典。"""
        if self.external_extractor is None:
            raise RuntimeError("SigLIP2 extractor must be bound before serializing metadata.")
        extractor = self.external_extractor
        return {
            "format_version": 10,
            "backbone_architecture": "google-siglip2-base-patch16-224-interpolated-256",
            "backbone_frozen": True,
            "backbone_peft": "lora",
            "siglip_lora": {
                "r": extractor.lora_r,
                "alpha": extractor.lora_alpha,
                "dropout": extractor.lora_dropout,
                "target_modules": extractor.lora_target_modules,
            },
            "prefix_grid_size": EXTERNAL_PREFIX_GRID_SIZE,
            "backbone_input_size": extractor.input_size,
            "backbone_patch_grid_size": extractor.patch_grid_size,
            "backbone_pretrained_image_size": int(extractor.model.config.image_size),
            "backbone_hidden_size": int(extractor.model.config.hidden_size),
            "multiscale_layers": list(extractor.multiscale_layers),
            "adapter_in_channels": int(extractor.model.config.hidden_size) * len(extractor.multiscale_layers),
            "fusion": "pure-visual-multiscale",  # v8 结构（文本引导中间版已弃）
            "qwen_native_vision": "lora-trainable",
        }

    def save_external_adapter(self, output_dir: str | Path) -> Path:
        """入参: adapter 输出目录; 方法: 保存 SigLIP2 LoRA、空间适配层、projector 与结构元数据，不复制冻结主干; 出参: 权重文件路径。"""
        if self.external_extractor is None:
            raise RuntimeError("SigLIP2 extractor must be bound before saving its adapter.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {"external_adapter_metadata": self._external_adapter_metadata()}
        for name, tensor in self.external_extractor.feature_adapter.state_dict().items():
            state[f"external_feature_adapter.{name}"] = tensor.detach().cpu()
        for name, tensor in self.external_projector.state_dict().items():
            state[f"external_projector.{name}"] = tensor.detach().cpu()
        for name, tensor in self.external_extractor.lora_state_dict().items():
            state[f"external_siglip_lora.{name}"] = tensor
        path = output_dir / EXTERNAL_ADAPTER_FILE
        torch.save(state, path)
        return path

    def load_external_adapter(self, adapter_dir: str | Path, strict: bool = True) -> Path | None:
        """入参: adapter 目录与 strict 开关。
        方法: 校验 v10 协议（LoRA 化主干、纯视觉融合、16×16 前缀），恢复 SigLIP2 LoRA 与
              空间适配层及 projector；拒绝 v8 结构（冻结主干，无 LoRA）与文本引导中间版结构。
        出参: 权重文件路径；非严格模式缺文件时返回 None。"""
        if self.external_extractor is None:
            raise RuntimeError("SigLIP2 extractor must be bound before loading its adapter.")
        path = Path(adapter_dir) / EXTERNAL_ADAPTER_FILE
        if not path.is_file():
            if strict:
                raise FileNotFoundError(f"External feature adapter not found: {path}")
            return None
        state = torch.load(path, map_location=self._external_device())
        metadata = state.get("external_adapter_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid external adapter metadata: {path}")
        architecture = metadata.get("backbone_architecture")
        if architecture != "google-siglip2-base-patch16-224-interpolated-256":
            raise ValueError(
                f"External adapter backbone mismatch in {path}: expected SigLIP2 Base patch16 with 256 input, "
                f"got {architecture!r}."
            )
        # format_version 校验：v10 = v8 纯视觉多尺度融合结构 + SigLIP2 主干 LoRA；
        # v8 中间版 = 冻结主干（无 LoRA）、文本引导中间版 = 融合结构不同，均不兼容。
        fmt_version = metadata.get("format_version")
        if fmt_version != 10:
            raise ValueError(
                f"External adapter format_version mismatch in {path}: expected 10 "
                f"(v8 structure + SigLIP2 LoRA), got {fmt_version!r}. "
                "文本引导中间版与 v8 冻结主干链均不兼容，请用 v10 配置重新训练。"
            )
        if metadata and int(metadata.get("prefix_grid_size", EXTERNAL_PREFIX_GRID_SIZE)) != EXTERNAL_PREFIX_GRID_SIZE:
            raise ValueError(f"External prefix grid mismatch in {path}: expected {EXTERNAL_PREFIX_GRID_SIZE}.")
        input_size = int(metadata.get("backbone_input_size", -1))
        if input_size != self.external_extractor.input_size:
            raise ValueError(f"External adapter input size mismatch in {path}: expected {self.external_extractor.input_size}, got {input_size}.")
        patch_grid_size = int(metadata.get("backbone_patch_grid_size", -1))
        if patch_grid_size != self.external_extractor.patch_grid_size:
            raise ValueError(
                f"External adapter patch grid mismatch in {path}: "
                f"expected {self.external_extractor.patch_grid_size}, got {patch_grid_size}."
            )
        # SigLIP LoRA 超参校验：训练端可经 config.yaml 覆盖 r/alpha，推理端默认常量与 adapter
        # 不一致时在此显式报错。
        siglip_lora_meta = metadata.get("siglip_lora") or {}
        if int(siglip_lora_meta.get("r", -1)) != self.external_extractor.lora_r or \
                int(siglip_lora_meta.get("alpha", -1)) != self.external_extractor.lora_alpha:
            raise ValueError(
                f"SigLIP2 LoRA config mismatch in {path}: adapter r/alpha="
                f"{siglip_lora_meta.get('r')}/{siglip_lora_meta.get('alpha')}, "
                f"current={self.external_extractor.lora_r}/{self.external_extractor.lora_alpha}. "
                "请将 bind_external_extractor 的 LoRA 超参与训练端 config.yaml 对齐。"
            )
        feature_adapter_state = {
            key.removeprefix("external_feature_adapter."): value
            for key, value in state.items()
            if key.startswith("external_feature_adapter.")
        }
        projector_state = {
            key.removeprefix("external_projector."): value
            for key, value in state.items()
            if key.startswith("external_projector.")
        }
        lora_state = {
            key.removeprefix("external_siglip_lora."): value
            for key, value in state.items()
            if key.startswith("external_siglip_lora.")
        }
        if not feature_adapter_state or not projector_state or not lora_state:
            raise ValueError(f"Invalid v10 external adapter state: {path}")
        self.external_extractor.feature_adapter.load_state_dict(feature_adapter_state, strict=True)
        self.external_projector.load_state_dict(projector_state, strict=True)
        self.external_extractor.load_lora_state_dict(lora_state)
        self.loaded_external_adapter_metadata = metadata
        return path

    def _external_device(self) -> torch.device:
        """入参: 无; 方法: 优先读取 lm_head 参数设备，使外部 projector 与 logits 计算所在设备一致; 出参: torch.device。"""
        return next(self.lm_head.parameters()).device

    def encode_external_images(self, external_images) -> torch.Tensor | None:
        """入参: batch 图像路径/PIL/tensor 或 None; 方法: SigLIP2 编码并投影为 256 个空间 token; 出参: None 或 [B,256,D]。"""
        if external_images is None:
            return None
        if self.external_extractor is None:
            raise RuntimeError("external_images requires a bound SigLIP2 extractor.")
        features = self.external_extractor(external_images)
        return self.external_projector(features).to(self._external_device())

    def _chunked_lm_loss(self, hidden_states: torch.Tensor, labels: torch.Tensor,
                         chunk_size: int = 1024) -> torch.Tensor:
        """入参: 最后一层 hidden [B,T,H]、labels [B,T]（含 -100 mask）、chunk_size 展平 token 分块大小。
        方法: 与基座 ForCausalLMLoss 等价的因果 shift CE（hidden[:,:-1] vs labels[:,1:]），但把
              lm_head+CE 沿展平 token 维分块计算——任一时刻只物化 chunk 级 logits（含反向图），
              避免 [B,T,151936] 全词表 logits 与其梯度同时驻留，削减训练激活峰值。
        出参: 标量 loss（mean，忽略 -100）。"""
        shift_hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        valid_tokens = (shift_labels != -100).sum()
        total_loss = hidden_states.new_zeros((), dtype=torch.float32)
        for start in range(0, shift_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, shift_hidden.shape[0])
            chunk_logits = self.lm_head(shift_hidden[start:end]).float()
            total_loss = total_loss + F.cross_entropy(
                chunk_logits, shift_labels[start:end], ignore_index=-100, reduction="sum",
            )
        return total_loss / valid_tokens.clamp(min=1).to(total_loss.device)

    def _resolve_external_hidden(self, external_images, external_hidden) -> torch.Tensor | None:
        """入参: 可空图像输入与可空外部 token; 方法: 优先复用 external_hidden，否则在线编码图像; 出参: None 或 [B,N,D]。"""
        if external_hidden is not None:
            return external_hidden.to(self._external_device())
        return self.encode_external_images(external_images)

    def _inject_external_prefix(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.FloatTensor | None,
        external_hidden: torch.Tensor,
        image_grid_thw: torch.LongTensor | None,
        video_grid_thw: torch.LongTensor | None,
        position_ids: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """入参: 含 N×N 个前缀槽位的 Qwen 输入及外部 token; 方法: 覆盖前缀 embedding 并计算完整 mRoPE 位置; 出参: embedding、position_ids、attention_mask。"""
        if input_ids is None:
            raise ValueError("External visual prefix requires input_ids for Qwen3-VL position indexing.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeddings = self.model.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        active_rank = attention_mask.long().cumsum(dim=-1)
        prefix_mask = attention_mask.bool() & (active_rank <= EXTERNAL_PREFIX_LENGTH)
        prefix_counts = prefix_mask.sum(dim=-1)
        if not torch.all(prefix_counts == EXTERNAL_PREFIX_LENGTH):
            raise ValueError(f"Each sample must reserve {EXTERNAL_PREFIX_LENGTH} active external prefix slots.")
        expected_shape = (input_ids.shape[0], EXTERNAL_PREFIX_LENGTH, self.external_hidden_size)
        if tuple(external_hidden.shape) != expected_shape:
            raise ValueError(f"External prefix shape must be {expected_shape}, got {tuple(external_hidden.shape)}.")
        external_hidden = external_hidden.to(device=embeddings.device, dtype=embeddings.dtype)
        expanded_mask = prefix_mask.unsqueeze(-1).expand_as(embeddings)
        embeddings = embeddings.masked_scatter(expanded_mask, external_hidden.reshape(-1))
        if position_ids is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        return embeddings, position_ids, attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        external_images=None,
        external_hidden: torch.Tensor | None = None,
        **kwargs,
    ) -> Qwen3VLCausalLMOutputWithPast:
        """入参: 含前缀槽位的 Qwen3-VL batch 与外部图像/token; 方法: prefill 覆盖 N×N 视觉前缀后调用原语言解码器; 出参: causal LM 输出。"""
        if self.training or labels is not None:
            kwargs["use_cache"] = False
        cache_position = kwargs.get("cache_position")
        is_prefill = cache_position is None or cache_position[0] == 0
        external_tokens = self._resolve_external_hidden(external_images, external_hidden) if is_prefill else None
        model_input_ids = input_ids
        if external_tokens is not None:
            inputs_embeds, position_ids, attention_mask = self._inject_external_prefix(
                input_ids,
                attention_mask,
                inputs_embeds,
                external_tokens,
                image_grid_thw,
                video_grid_thw,
                position_ids,
            )
            model_input_ids = None
        outputs = self.model(
            input_ids=model_input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        hidden_states = outputs[0]
        # 训练（labels 存在且开关开启）走分块 CE：不物化全词表 logits（见 _chunked_lm_loss），
        # logits 置 None——Trainer 训练步只消费 loss；generate 路径不受影响。
        use_chunked = labels is not None and getattr(self, "use_chunked_ce", False)
        logits = None
        loss = None
        if use_chunked:
            loss = self._chunked_lm_loss(hidden_states, labels)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(self, *args, external_images=None, external_hidden=None, **kwargs):
        """入参: generate 解码阶段参数与外部特征; 方法: 复用父类缓存裁剪逻辑并保留外部注入参数; 出参: 单步生成 forward 输入。"""
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["external_images"] = external_images
        model_inputs["external_hidden"] = external_hidden
        return model_inputs