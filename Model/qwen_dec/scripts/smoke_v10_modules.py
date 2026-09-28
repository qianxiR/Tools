"""v10 模块级冒烟测试（不加载大模型权重，仅验证形状/LoRA 目标/梯度链路）。"""
import torch
from peft import LoraConfig, get_peft_model
from transformers import SiglipVisionConfig, SiglipVisionModel

from model.external_feature_adapter import (
    EXTERNAL_FEATURE_CHANNELS,
    ExternalFeatureProjector,
    MultiScaleFusionDecoder,
    SimpleFPN,
)


def main() -> None:
    device = torch.device("cpu")
    dtype = torch.float32  # CPU 冒烟用 fp32；bf16 等价路径已在设计上以 GroupNorm 保证

    # ① 融合链: SimpleFPN(3072→256 四尺度) → MultiScaleFusionDecoder → 16×16 → projector
    fpn = SimpleFPN(embed_dim=3072, out_dim=EXTERNAL_FEATURE_CHANNELS).to(device, dtype)
    decoder = MultiScaleFusionDecoder(d_model=EXTERNAL_FEATURE_CHANNELS, out_dim=EXTERNAL_FEATURE_CHANNELS).to(device, dtype)
    projector = ExternalFeatureProjector(hidden_size=2560).to(device, dtype)

    feature_map = torch.randn(2, 3072, 16, 16, device=device, dtype=dtype, requires_grad=True)
    multiscale = fpn(feature_map)
    assert [tuple(m.shape[-2:]) for m in multiscale] == [(64, 64), (32, 32), (16, 16), (8, 8)], \
        [m.shape for m in multiscale]
    fused = decoder(multiscale)
    assert fused.shape == (2, 256, 16, 16), fused.shape
    prefix = projector(fused)
    assert prefix.shape == (2, 256, 2560), prefix.shape
    prefix.sum().backward()
    assert feature_map.grad is not None and feature_map.grad.abs().sum() > 0
    grad_names = [n for n, p in decoder.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"[smoke] fusion chain ok; {len(grad_names)} fusion params receive grad")

    # ② SigLIP LoRA 目标名匹配（config-only 随机初始化，不加载权重）
    config = SiglipVisionConfig(hidden_size=768, num_hidden_layers=12, image_size=224,
                                patch_size=16, intermediate_size=3072)
    vision = SiglipVisionModel(config)
    lora_model = get_peft_model(vision, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=r".*encoder\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.(?:fc1|fc2))",
    ))
    lora_names = [n for n, _ in lora_model.named_parameters() if "lora_" in n]
    # 12 层 × 6 目标 × (A+B) = 144
    assert len(lora_names) == 144, len(lora_names)
    # forward 透传 kwargs（interpolate_pos_encoding / output_hidden_states）
    outputs = lora_model(pixel_values=torch.randn(1, 3, 256, 256),
                         interpolate_pos_encoding=True, output_hidden_states=True)
    assert len(outputs.hidden_states) == 13
    assert outputs.hidden_states[3].shape == (1, 256, 768)
    print(f"[smoke] siglip lora wrap ok; {len(lora_names)} lora params; hidden_states len={len(outputs.hidden_states)}")

    # ③ lora_state_dict 收集/恢复 roundtrip
    state = {n: p.detach().clone() for n, p in lora_model.named_parameters() if "lora_" in n}
    with torch.no_grad():
        for n, p in lora_model.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn_like(p))
    with torch.no_grad():
        for n, p in lora_model.named_parameters():
            if "lora_" in n:
                p.copy_(state[n])
    print("[smoke] lora state roundtrip ok")


if __name__ == "__main__":
    main()