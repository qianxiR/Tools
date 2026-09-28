"""上采样卷积细化模块。"""

import torch.nn as nn


class UpsampleBlock(nn.Module):
    """
    入参:
    - in_channels/out_channels: 输入与输出通道数。

    方法:
    - 双线性上采样两倍后使用两层 3x3 卷积细化特征。

    出参:
    - forward 返回空间尺寸扩大两倍的特征图。
    """

    def __init__(self, in_channels, out_channels):
        """
        入参:
        - in_channels/out_channels: 卷积通道配置，须为正整数。

        方法:
        - 构造上采样、卷积、批归一化与 GELU 序列。

        出参:
        - 无；完成上采样块初始化。
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        """
        入参:
        - x: 输入特征 [B, C, H, W]。

        方法:
        - 先扩大空间尺寸，再执行卷积细化。

        出参:
        - 上采样特征 [B, out_channels, 2H, 2W]。
        """
        return self.conv(self.up(x))
