import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv3D_TimeScale3(nn.Module):
    """
    3D深度可分离卷积，专门处理时间刻度为3的情况
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        spatial_kernel_size: 空间维度卷积核大小 (H, W)
        stride: 卷积步长，默认为1
        padding_mode: 填充模式，默认'zeros'
        bias: 是否使用偏置
        use_batch_norm: 是否使用批归一化
        activation: 激活函数类型
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 spatial_kernel_size=(3, 3),
                 stride=1,
                 padding_mode='zeros',
                 bias=False,
                 use_batch_norm=True,
                 activation='relu'):
        super(SeparableConv3D_TimeScale3, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 第一阶段：时空深度卷积（不融合通道）
        # 时间维度固定为3，空间维度可配置
        temporal_kernel = 3
        spatial_h, spatial_w = spatial_kernel_size
        
        # 计算padding以保持特征图尺寸
        temporal_padding = temporal_kernel // 2  # 1
        spatial_padding_h = spatial_h // 2
        spatial_padding_w = spatial_w // 2
        
        self.depthwise_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,  # 输出通道数等于输入通道数
            kernel_size=(temporal_kernel, spatial_h, spatial_w),  # (3, 3, 3)
            stride=stride,
            padding=(temporal_padding, spatial_padding_h, spatial_padding_w),
            groups=in_channels,  # 关键：每个通道独立卷积
            bias=bias,
            padding_mode=padding_mode
        )
        
        # 第二阶段：通道融合卷积（3×1×1卷积核）
        self.pointwise_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 1, 1),  # 沿时间维度融合通道
            stride=1,
            padding=(1, 0, 0),  # 只在时间维度padding
            groups=1,  # 全连接，融合所有通道
            bias=bias,
            padding_mode=padding_mode
        )
        
        # 批归一化层
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm3d(in_channels)
            self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, T, H, W]
            
        Returns:
            输出张量，形状为 [B, out_channels, T, H, W]
        """
        # 第一阶段：深度卷积（时空特征提取，不融合通道）
        x = self.depthwise_conv(x)
        
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        
        # 第二阶段：逐点卷积（通道融合）
        x = self.pointwise_conv(x)
        
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        
        return x
    
    def get_params_count(self):
        """计算参数数量"""
        depthwise_params = (
            self.in_channels * 3 * self.depthwise_conv.kernel_size[1] * self.depthwise_conv.kernel_size[2]
        )
        pointwise_params = self.in_channels * self.out_channels * 3
        
        total_params = depthwise_params + pointwise_params
        
        if self.use_batch_norm:
            total_params += 2 * (self.in_channels + self.out_channels)  # BN参数
            
        return {
            'depthwise_params': depthwise_params,
            'pointwise_params': pointwise_params,
            'total_params': total_params
        }


class InvertedResidual3D_TimeScale3(nn.Module):
    """
    3D倒残差模块，集成时间刻度为3的深度可分离卷积
    
    结构：扩张 -> 深度卷积 -> 压缩 + 残差连接
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion_ratio=6,
                 spatial_kernel_size=(3, 3),
                 stride=1,
                 use_residual=True,
                 use_batch_norm=True,
                 activation='relu'):
        super(InvertedResidual3D_TimeScale3, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = expansion_ratio
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        # 计算扩张后的通道数
        expanded_channels = in_channels * expansion_ratio
        
        # 第一阶段：扩张卷积（1×1×1）
        self.expand_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=1,
            bias=False
        )
        
        # 第二阶段：深度可分离卷积（使用已有的实现）
        self.depthwise_separable = SeparableConv3D_TimeScale3(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            spatial_kernel_size=spatial_kernel_size,
            stride=stride,
            use_batch_norm=use_batch_norm,
            activation=activation
        )
        
        # 第三阶段：压缩卷积（1×1×1）
        self.project_conv = nn.Conv3d(
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        
        # 批归一化
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn_expand = nn.BatchNorm3d(expanded_channels)
            self.bn_project = nn.BatchNorm3d(out_channels)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, T, H, W]
            
        Returns:
            输出张量，形状为 [B, out_channels, T, H, W]
        """
        residual = x
        
        # 第一阶段：扩张
        x = self.expand_conv(x)
        if self.use_batch_norm:
            x = self.bn_expand(x)
        x = self.activation(x)
        
        # 第二阶段：深度可分离卷积
        x = self.depthwise_separable(x)
        
        # 第三阶段：压缩（不使用激活函数）
        x = self.project_conv(x)
        if self.use_batch_norm:
            x = self.bn_project(x)
        
        # 残差连接
        if self.use_residual:
            x = x + residual
        
        return x
    
    def get_params_count(self):
        """计算参数数量"""
        expanded_channels = self.in_channels * self.expansion_ratio
        
        # 扩张卷积参数
        expand_params = self.in_channels * expanded_channels
        
        # 深度可分离卷积参数
        separable_params = self.depthwise_separable.get_params_count()['total_params']
        
        # 压缩卷积参数
        project_params = expanded_channels * self.out_channels
        
        total_params = expand_params + separable_params + project_params
        
        if self.use_batch_norm:
            total_params += 2 * (expanded_channels + self.out_channels)
        
        return {
            'expand_params': expand_params,
            'separable_params': separable_params,
            'project_params': project_params,
            'total_params': total_params
        }


class StackedInvertedResidual3D(nn.Module):
    """
    堆叠的3D倒残差模块
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion_ratio=6,
                 spatial_kernel_size=(3, 3),
                 intermediate_channels=None):
        super(StackedInvertedResidual3D, self).__init__()
        
        self.blocks = nn.ModuleList()
        
        # 如果没有指定中间通道数，使用输出通道数
        if intermediate_channels is None:
            intermediate_channels = out_channels
        
        # 第一个块：可能改变通道数
        self.blocks.append(
            InvertedResidual3D_TimeScale3(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                expansion_ratio=expansion_ratio,
                spatial_kernel_size=spatial_kernel_size,
                use_residual=(in_channels == intermediate_channels)
            )
        )
        
        # 中间块：保持通道数不变
        for i in range(num_blocks - 2):
            self.blocks.append(
                InvertedResidual3D_TimeScale3(
                    in_channels=intermediate_channels,
                    out_channels=intermediate_channels,
                    expansion_ratio=expansion_ratio,
                    spatial_kernel_size=spatial_kernel_size,
                    use_residual=True
                )
            )
        
        # 最后一个块：输出指定通道数
        if num_blocks > 1:
            self.blocks.append(
                InvertedResidual3D_TimeScale3(
                    in_channels=intermediate_channels,
                    out_channels=out_channels,
                    expansion_ratio=expansion_ratio,
                    spatial_kernel_size=spatial_kernel_size,
                    use_residual=(intermediate_channels == out_channels)
                )
            )
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class StackedSeparableConv3D(nn.Module):
    """
    堆叠的3D深度可分离卷积块
    """
    
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=2,
                 spatial_kernel_size=(3, 3),
                 use_residual=True):
        super(StackedSeparableConv3D, self).__init__()
        
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(
            SeparableConv3D_TimeScale3(
                in_channels=in_channels,
                out_channels=hidden_channels,
                spatial_kernel_size=spatial_kernel_size
            )
        )
        
        # 中间层
        for i in range(num_layers - 2):
            self.layers.append(
                SeparableConv3D_TimeScale3(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    spatial_kernel_size=spatial_kernel_size
                )
            )
        
        # 最后一层
        if num_layers > 1:
            self.layers.append(
                SeparableConv3D_TimeScale3(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    spatial_kernel_size=spatial_kernel_size
                )
            )
        
        # 残差连接的通道调整
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x):
        residual = x
        
        for layer in self.layers:
            x = layer(x)
        
        if self.use_residual:
            residual = self.residual_proj(residual)
            x = x + residual
            
        return x


def demo_usage():
    """演示使用方法"""
    
    # 创建输入数据 [B, C, T, H, W]
    batch_size = 2
    in_channels = 64
    temporal_length = 8
    height = 32
    width = 32
    
    input_tensor = torch.randn(batch_size, in_channels, temporal_length, height, width)
    print(f"输入张量形状: {input_tensor.shape}")
    
    print("\n=== 1. 基础3D深度可分离卷积 ===")
    # 创建模型
    model = SeparableConv3D_TimeScale3(
        in_channels=64,
        out_channels=128,
        spatial_kernel_size=(3, 3),
        use_batch_norm=True,
        activation='relu'
    )
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"输出张量形状: {output.shape}")
    
    # 打印参数统计
    params_info = model.get_params_count()
    print(f"参数统计: {params_info}")
    
    print("\n=== 2. 倒残差模块 ===")
    # 倒残差模块示例
    inverted_model = InvertedResidual3D_TimeScale3(
        in_channels=64,
        out_channels=64,  # 相同通道数以启用残差连接
        expansion_ratio=6,
        spatial_kernel_size=(3, 3),
        use_residual=True,
        activation='relu'
    )
    
    with torch.no_grad():
        inverted_output = inverted_model(input_tensor)
    
    print(f"倒残差输出形状: {inverted_output.shape}")
    
    # 打印倒残差参数统计
    inverted_params = inverted_model.get_params_count()
    print(f"倒残差参数统计: {inverted_params}")
    
    print("\n=== 3. 堆叠倒残差模块 ===")
    # 堆叠倒残差模块
    stacked_inverted = StackedInvertedResidual3D(
        in_channels=64,
        out_channels=128,
        num_blocks=3,
        expansion_ratio=4,
        spatial_kernel_size=(3, 3)
    )
    
    with torch.no_grad():
        stacked_inverted_output = stacked_inverted(input_tensor)
    
    print(f"堆叠倒残差输出形状: {stacked_inverted_output.shape}")
    
    print("\n=== 4. 堆叠深度可分离卷积 ===")
    # 堆叠版本示例
    stacked_model = StackedSeparableConv3D(
        in_channels=64,
        hidden_channels=96,
        out_channels=128,
        num_layers=3,
        use_residual=True
    )
    
    with torch.no_grad():
        stacked_output = stacked_model(input_tensor)
    
    print(f"堆叠深度可分离输出形状: {stacked_output.shape}")
    
    print("\n=== 5. 不同扩张比例对比 ===")
    # 不同扩张比例的倒残差模块对比
    expansion_ratios = [2, 4, 6, 8]
    for ratio in expansion_ratios:
        test_model = InvertedResidual3D_TimeScale3(
            in_channels=32,
            out_channels=32,
            expansion_ratio=ratio,
            use_residual=True
        )
        test_params = test_model.get_params_count()
        print(f"扩张比例 {ratio}: 总参数量 {test_params['total_params']}")


if __name__ == "__main__":
    demo_usage() 