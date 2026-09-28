"""SigLIP2 外部视觉链（v10：v8 纯视觉融合结构 + SigLIP2 主干 LoRA 微调）。

做什么: SigLIP2 Base 冻结主干叠加 LoRA（q/k/v/out_proj + fc1/fc2）提取四层多深度特征，
       经 SimpleFPN 派生四尺度后由 MultiScaleFusionDecoder 纯视觉粗到细融合，池化回 16×16
       供 projector 生成 256 个视觉前缀 token。
为什么: v8（冻结主干 + 原生视觉塔）被证明可训（500 集 epoch1 F1≈0.30）。v10 恢复 v8 结构，
       仅将 SigLIP2 主干从「完全冻结」升级为「LoRA 微调」（低成本适配检测域，保留冻结主干稳定性）。
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, SiglipVisionModel


SIGLIP2_INPUT_SIZE = 256
SIGLIP2_PATCH_SIZE = 16
SIGLIP2_PATCH_GRID_SIZE = SIGLIP2_INPUT_SIZE // SIGLIP2_PATCH_SIZE
EXTERNAL_FEATURE_CHANNELS = 256
# 多尺度：取 SigLIP2 Base（共 12 层 transformer）的浅/中/深四层 hidden_states 拼接。
# 索引对应 SiglipVisionModel(..., output_hidden_states=True).hidden_states 的位置（0=embedding 输出）。
# 浅层(3)含边缘/纹理（腐蚀、裂纹），中层(7,9)含局部结构，深层(12)含语义（障碍物、树根）。
SIGLIP2_MULTISCALE_LAYERS = (3, 7, 9, 12)
# SimpleFPN 多尺度融合的尺度因子（对齐 DFATG-Net）：4S/2S/1S/0.5S。
SIMPLE_FPN_SCALE_FACTORS = (4.0, 2.0, 1.0, 0.5)
# SigLIP2 主干 LoRA：冻结主干上叠加低秩适配（v10 起主干参与微调）。
# 正则限定 encoder 各层（后缀匹配会额外命中 pooling head 的 out_proj/fc1/fc2，
# 而 pooled_output 不在本链消费范围内，属无效参数）。
SIGLIP2_LORA_R = 32
SIGLIP2_LORA_ALPHA = 64
SIGLIP2_LORA_DROPOUT = 0.05
SIGLIP2_LORA_TARGETS = r".*encoder\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.(?:fc1|fc2))"


class SimpleFPN(nn.Module):
    """入参: embed_dim 输入通道数, out_dim 每层输出通道数, scale_factors 4 个尺度因子。
    方法: 从单一特征图 [B,embed_dim,S,S] 用并行分支派生 4 个真多尺度特征——
          4S: 两层 ConvTranspose2d(×2) 上采样; 2S: 一层 ConvTranspose2d(×2);
          1S: 恒等; 0.5S: MaxPool2d(×2) 下采样。每分支末尾 Conv1×1+Conv3×3 统一到 out_dim。
    出参: forward 返回 4 个 [B,out_dim,H_i,W_i] 特征列表，分辨率分别为 S×{4,2,1,0.5}。"""

    def __init__(self, embed_dim: int, out_dim: int,
                 scale_factors: tuple = SIMPLE_FPN_SCALE_FACTORS):
        super().__init__()
        self.scale_factors = scale_factors
        self.convs = nn.ModuleList()
        for scale in scale_factors:
            layers = nn.Sequential()
            if scale == 4.0:
                layers.add_module("dconv_0", nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2))
                layers.add_module("act_0", nn.SiLU())
                layers.add_module("dconv_1", nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2))
                branch_dim = embed_dim // 4
            elif scale == 2.0:
                layers.add_module("dconv", nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2))
                branch_dim = embed_dim // 2
            elif scale == 1.0:
                branch_dim = embed_dim
            elif scale == 0.5:
                layers.add_module("maxpool", nn.MaxPool2d(kernel_size=2, stride=2))
                branch_dim = embed_dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported")
            layers.add_module("conv1x1", nn.Conv2d(branch_dim, out_dim, kernel_size=1, bias=True))
            layers.add_module("conv3x3", nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=True))
            self.convs.append(layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """入参: x [B,embed_dim,S,S] 单一特征图。
        方法: 4 个分支并行处理同一输入，产出 4 个不同分辨率的特征。
        出参: 4 个 [B,out_dim,H_i,W_i] 特征列表（索引 0 最细=4S，3 最粗=0.5S）。"""
        return [conv(x) for conv in self.convs]


class _UpsampleBlock(nn.Module):
    """入参: in_channels/out_channels 通道数。
    方法: 2× 双线性上采样 + 两层 3×3 卷积（GroupNorm+SiLU，bf16 友好）。
    出参: 分辨率翻倍后的特征图。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class MultiScaleFusionDecoder(nn.Module):
    """入参: d_model 各尺度通道数, out_dim 输出通道数, output_grid 最终网格边长(默认 16)。
    方法: 粗到细 U-Net 式融合——从最粗(0.5S)开始，逐级 UpsampleBlock(×2) + 对齐 padding +
          逐元素加法 skip(更细一级) + 卷积融合块。末尾自适应平均池化到 output_grid×output_grid。
    出参: forward 返回 [B,out_dim,output_grid,output_grid]。"""

    def __init__(self, d_model: int, out_dim: int,
                 output_grid: int = SIGLIP2_PATCH_GRID_SIZE):
        super().__init__()
        self.output_grid = output_grid
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(32, d_model),
                nn.SiLU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(32, d_model),
                nn.SiLU(),
            )
            for _ in range(4)
        ])
        self.upsample_blocks = nn.ModuleList([
            _UpsampleBlock(d_model, d_model) for _ in range(3)
        ])
        self.output_proj = nn.Sequential(
            nn.Conv2d(d_model, out_dim, kernel_size=1, bias=True),
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((output_grid, output_grid))

    @staticmethod
    def _align_features(upsampled: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """入参: 上采样特征与 skip 目标特征。方法: 对称 pad 修正尺寸差。出参: 尺寸对齐后的上采样特征。"""
        diff_h = upsampled.size(2) - target.size(2)
        diff_w = upsampled.size(3) - target.size(3)
        if diff_h > 0 or diff_w > 0:
            upsampled = F.pad(upsampled, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
            ])
        return upsampled

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """入参: SimpleFPN 输出的 4 个尺度特征（索引 0=4S 最细, 3=0.5S 最粗）。
        方法: 从最粗(index 3)起，逐级上采样+加法 skip 更细尺度，卷积融合。
        出参: [B,out_dim,output_grid,output_grid]。"""
        x = self.fusion_blocks[3](features[3])
        for up_idx, fuse_idx, skip_idx in ((0, 2, 2), (1, 1, 1), (2, 0, 0)):
            x = self.upsample_blocks[up_idx](x)
            x = self._align_features(x, features[skip_idx])
            x = x + features[skip_idx]
            x = self.fusion_blocks[fuse_idx](x)
        x = self.output_proj(x)
        return self.pool(x)


class Siglip2BaseFeatureExtractor(nn.Module):
    """入参: SigLIP2 Base 本地模型目录、设备和计算精度，以及可选 LoRA 超参。
    方法: 冻结 vision tower 叠加 LoRA，以 256×256 输入生成 16×16 patch token，再用可训练适配层压到 256 通道。
    出参: [B,256,16,16] 外部空间特征。"""

    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_r: int = SIGLIP2_LORA_R,
        lora_alpha: int = SIGLIP2_LORA_ALPHA,
        lora_dropout: float = SIGLIP2_LORA_DROPOUT,
    ):
        """入参: model_path 必须包含完整配置和 model.safetensors，device/dtype 指定 vision tower 运行位置，
              lora_r/lora_alpha/lora_dropout 为主干 LoRA 超参（训练端 config.yaml 透传，加载端元数据校验）。
        方法: 加载 SigLIP2 Base 冻结主干后叠加 LoRA（q/k/v/out_proj+fc1/fc2），
              创建 768→256 空间适配链（SimpleFPN + 多尺度融合 decoder）。
        出参: LoRA 化 SigLIP2 与可训练空间适配层组成的特征提取器。"""
        super().__init__()
        self.model_path, local_only = self._resolve_model_source(model_path)
        self.device_ref = torch.device(device)
        self.dtype_ref = dtype
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_path,
            use_fast=False,
            local_files_only=local_only,
        )
        self.model = SiglipVisionModel.from_pretrained(
            self.model_path,
            dtype=self.dtype_ref,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=local_only,
        ).to(device=self.device_ref, dtype=self.dtype_ref)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = SIGLIP2_LORA_TARGETS
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                target_modules=SIGLIP2_LORA_TARGETS,
            ),
        ).to(device=self.device_ref, dtype=self.dtype_ref)

        hidden_size = int(self.model.config.hidden_size)
        patch_size = int(self.model.config.patch_size)
        if patch_size != SIGLIP2_PATCH_SIZE:
            raise ValueError(f"SigLIP2 patch size mismatch: expected {SIGLIP2_PATCH_SIZE}, got {patch_size}.")
        self.input_size = SIGLIP2_INPUT_SIZE
        self.patch_grid_size = SIGLIP2_PATCH_GRID_SIZE
        self.multiscale_layers = SIGLIP2_MULTISCALE_LAYERS
        adapter_in_channels = hidden_size * len(self.multiscale_layers)  # 768×4 = 3072
        # SimpleFPN + 粗到细 decoder：从多深度拼接特征派生真多尺度（4S/2S/1S/0.5S），
        # 融合后池化回 [B,256,16,16]（v8 纯视觉结构，v10 保留；主干升级为 LoRA 微调）。
        self.feature_fpn = SimpleFPN(
            embed_dim=adapter_in_channels, out_dim=EXTERNAL_FEATURE_CHANNELS,
        ).to(device=self.device_ref, dtype=self.dtype_ref)
        self.feature_decoder = MultiScaleFusionDecoder(
            d_model=EXTERNAL_FEATURE_CHANNELS, out_dim=EXTERNAL_FEATURE_CHANNELS,
            output_grid=SIGLIP2_PATCH_GRID_SIZE,
        ).to(device=self.device_ref, dtype=self.dtype_ref)
        # feature_adapter 保留为兼容属性（enable_external_training / save_external_adapter 引用），
        # 指向 FPN+decoder 组合，使旧代码路径无需改动。
        self.feature_adapter = nn.ModuleList([self.feature_fpn, self.feature_decoder])

    @staticmethod
    def _resolve_model_source(model_path: str | Path) -> tuple[str, bool]:
        """入参: 本地 google/siglip2-base-patch16-224 模型目录。
        方法: 校验文件完整性，并按仓库真实配置确认 siglip/siglip_vision_model 架构。
        出参: (本地目录字符串, True)；目录、架构或文件不完整时抛出异常。"""
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"SigLIP2 model directory not found: {path}.")
        required_files = (path / "config.json", path / "preprocessor_config.json", path / "model.safetensors")
        missing = [file.name for file in required_files if not file.is_file() or file.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"Incomplete SigLIP2 model directory {path}: missing {missing}.")
        with (path / "config.json").open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        model_type = config.get("model_type", "")
        vision_model_type = config.get("vision_config", {}).get("model_type", "")
        if model_type != "siglip" or vision_model_type != "siglip_vision_model":
            raise ValueError(
                f"Model directory {path} has incompatible architecture: "
                f"model_type={model_type!r}, vision_model_type={vision_model_type!r}."
            )
        return str(path), True

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """入参: 无。
        方法: 收集主干 peft 包装内全部 lora_ 前缀参数（lora_A/lora_B 权重），脱离计算图转 CPU。
        出参: {参数名: CPU 张量}；用于嵌入 external_feature_adapter.pt 单文件协议。"""
        return {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if "lora_" in name
        }

    def load_lora_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """入参: lora_state_dict 产出的参数字典。
        方法: 校验键集合与当前 LoRA 结构完全一致（多键/缺键即结构漂移，拒绝加载），
              就地 copy_ 恢复权重并对齐设备与 dtype。
        出参: 无；键集合不匹配时抛出 ValueError。"""
        current = {
            name: param for name, param in self.model.named_parameters() if "lora_" in name
        }
        missing = sorted(set(current) - set(state))
        extra = sorted(set(state) - set(current))
        if missing or extra:
            raise ValueError(
                f"SigLIP2 LoRA state mismatch: missing={missing[:5]} extra={extra[:5]}; "
                "adapter 与当前 LoRA 配置（r/alpha/target_modules）不一致。"
            )
        with torch.no_grad():
            for name, param in current.items():
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        """入参: CHW 图像张量，数值范围允许 0-1 或 0-255。
        方法: 移到 CPU、转 HWC uint8，并将单通道扩展为 RGB。
        出参: RGB PIL.Image；维度或通道非法时抛出 ValueError。"""
        if image.ndim != 3:
            raise ValueError(f"SigLIP2 tensor image must be CHW, got shape {tuple(image.shape)}.")
        tensor = image.detach().float().cpu()
        if tensor.shape[0] not in (1, 3):
            raise ValueError(f"SigLIP2 tensor image must have 1 or 3 channels, got {tensor.shape[0]}.")
        if tensor.max().item() <= 1.0:
            tensor = tensor * 255.0
        array = tensor.clamp(0, 255).byte().permute(1, 2, 0).numpy()
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        return Image.fromarray(array, mode="RGB")

    @classmethod
    def _normalize_images(cls, images) -> list[Image.Image]:
        """入参: 图像路径、PIL、CHW/BCHW 张量或上述对象列表。
        方法: 将 batch 逐项转为 RGB PIL，确保训练增强图和原始路径走同一官方预处理。
        出参: 非空 RGB PIL.Image 列表。"""
        if torch.is_tensor(images):
            tensor_batch = images.unsqueeze(0) if images.ndim == 3 else images
            if tensor_batch.ndim != 4:
                raise ValueError(f"SigLIP2 tensor batch must be BCHW, got shape {tuple(images.shape)}.")
            return [cls._tensor_to_pil(image) for image in tensor_batch]
        image_items = images if isinstance(images, (list, tuple)) else [images]
        normalized: list[Image.Image] = []
        for image in image_items:
            if isinstance(image, (str, Path)):
                normalized.append(Image.open(image).convert("RGB"))
            elif isinstance(image, Image.Image):
                normalized.append(image.convert("RGB"))
            elif torch.is_tensor(image):
                normalized.append(cls._tensor_to_pil(image))
            else:
                raise TypeError(f"Unsupported SigLIP2 image type: {type(image).__name__}.")
        if not normalized:
            raise ValueError("SigLIP2 image batch must not be empty.")
        return normalized

    def forward(self, images) -> torch.Tensor:
        """入参: 图像路径、PIL、tensor 或其 batch。
        方法: 官方预处理到 256×256并插值位置编码，运行 LoRA 化 vision tower（带梯度），
              取四层 hidden_states 拼接为多深度特征 [B,3072,16,16]，经 SimpleFPN 派生四个
              真多尺度特征（4S/2S/1S/0.5S），再由粗到细 decoder 融合并池化回 [B,256,16,16]。
        出参: [B,256,16,16] 特征；token 数或 hidden size不符合 checkpoint 配置时抛出 ValueError。"""
        pil_images = self._normalize_images(images)
        pixel_values = self.image_processor(
            images=pil_images,
            size={"height": self.input_size, "width": self.input_size},
            return_tensors="pt",
        )["pixel_values"].to(device=self.device_ref, dtype=self.dtype_ref)
        outputs = self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
        )
        all_layers = outputs.hidden_states  # tuple, 长度 = num_layers + 1（含 embedding 层）
        # 校验层数够取
        max_layer = max(self.multiscale_layers)
        if len(all_layers) <= max_layer:
            raise ValueError(
                f"SigLIP2 hidden_states has {len(all_layers)} layers, "
                f"cannot pick layer {max_layer}."
            )
        picked = [all_layers[idx] for idx in self.multiscale_layers]
        patch_tokens = torch.cat(picked, dim=-1)  # [B, 256, hidden*4]
        batch_size, token_count, concat_hidden = patch_tokens.shape
        expected_tokens = self.patch_grid_size ** 2
        if token_count != expected_tokens:
            raise ValueError(f"SigLIP2 patch token mismatch: expected {expected_tokens}, got {token_count}.")
        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            concat_hidden,
            self.patch_grid_size,
            self.patch_grid_size,
        )
        # SimpleFPN 派生 4 个真多尺度特征 → 粗到细 decoder 融合 → 池化回 [B,256,16,16]
        multiscale = self.feature_fpn(feature_map)
        return self.feature_decoder(multiscale)


class ExternalFeatureProjector(nn.Module):
    """入参: 16×16 外部特征和 Qwen hidden size。
    方法: 保留每个 patch 位置，将 256 通道映射到语言维度并叠加二维位置编码。
    出参: [B,256,D] 外部视觉前缀。"""

    def __init__(
        self,
        hidden_size: int,
        input_dim: int = EXTERNAL_FEATURE_CHANNELS,
        output_grid_size: int = SIGLIP2_PATCH_GRID_SIZE,
    ):
        """入参: hidden_size 为 Qwen 维度，input_dim 为适配特征通道，output_grid_size 为前缀网格边长。
        方法: 构建 256→512→hidden_size MLP 和输出 LayerNorm。
        出参: projector 实例；网格或 hidden size 不符合协议时抛出 ValueError。"""
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.output_grid_size = int(output_grid_size)
        if self.output_grid_size != SIGLIP2_PATCH_GRID_SIZE:
            raise ValueError(
                f"External prefix grid must match SigLIP2 patch grid {SIGLIP2_PATCH_GRID_SIZE}, "
                f"got {self.output_grid_size}."
            )
        if self.hidden_size % 4 != 0:
            raise ValueError("Qwen hidden_size must be divisible by 4 for 2D position encoding.")
        intermediate_dim = input_dim * 2
        self.project = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def _position_encoding(feature: torch.Tensor, hidden_size: int) -> torch.Tensor:
        """入参: [B,C,H,W] 特征与 Qwen hidden size。
        方法: 用归一化 xy 网格生成二维正余弦位置编码。
        出参: [1,H*W,D] 位置张量。"""
        height, width = feature.shape[-2:]
        quarter_size = hidden_size // 4
        frequency = torch.exp(
            torch.arange(quarter_size, device=feature.device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=feature.device)) / quarter_size)
        )
        y_position = torch.linspace(0.0, 1.0, height, device=feature.device, dtype=torch.float32)
        x_position = torch.linspace(0.0, 1.0, width, device=feature.device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_position, x_position, indexing="ij")
        y_angle = y_grid.reshape(-1, 1) * (2.0 * torch.pi) * frequency.reshape(1, -1)
        x_angle = x_grid.reshape(-1, 1) * (2.0 * torch.pi) * frequency.reshape(1, -1)
        position = torch.cat((y_angle.sin(), y_angle.cos(), x_angle.sin(), x_angle.cos()), dim=-1)
        return position.unsqueeze(0).to(dtype=feature.dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """入参: SigLIP2 空间特征 [B,256,16,16]。
        方法: 校验原生网格后逐 patch 展平，MLP 对齐到 Qwen hidden size，并叠加二维位置编码。
        出参: [B,256,D] 外部前缀 token。"""
        expected_shape = (SIGLIP2_PATCH_GRID_SIZE, SIGLIP2_PATCH_GRID_SIZE)
        if not torch.is_tensor(features) or features.ndim != 4:
            raise ValueError("External features must be a [B,C,H,W] tensor.")
        if tuple(features.shape[-2:]) != expected_shape:
            raise ValueError(f"External feature grid must be {expected_shape}, got {tuple(features.shape[-2:])}.")
        reference = next(self.parameters())
        feature = features.to(device=reference.device, dtype=reference.dtype)
        tokens = feature.flatten(2).transpose(1, 2)
        position = self._position_encoding(feature, self.hidden_size)
        return self.output_norm(self.project(tokens) + position)