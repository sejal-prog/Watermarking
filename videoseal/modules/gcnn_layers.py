
"""
Group Equivariant CNN layers for rotation-robust watermarking.
Uses p4 group (4 rotations: 0°, 90°, 180°, 270°).
"""

import torch
from torch import nn
from escnn import gspaces, nn as gnn

# Initialize the rotation group (p4)
# This is the group of 90° rotations + translations
R2_ACT = gspaces.rot2dOnR2(N=4)


def get_trivial_field_type(n_channels: int):
    """
    Create field type for Z2 (regular 2D space, no rotations).
    Used for input/output layers.
    
    Args:
        n_channels: Number of channels
    Returns:
        FieldType for regular images
    """
    return gnn.FieldType(R2_ACT, n_channels * [R2_ACT.trivial_repr])


def get_regular_field_type(n_channels: int):
    """
    Create field type for p4 (4 rotation channels).
    Used for hidden layers.
    
    Args:
        n_channels: Number of channels
    Returns:
        FieldType with 4 rotation channels per feature
    """
    return gnn.FieldType(R2_ACT, n_channels * [R2_ACT.regular_repr])


class GResnetBlock(gnn.EquivariantModule):
    """
    Group Equivariant ResNet Block.
    """
    
    def __init__(
        self, 
        in_type: gnn.FieldType,
        out_type: gnn.FieldType,
    ):
        super().__init__()
        
        # Use out_type for middle layer to keep things simple
        mid_type = out_type
        
        # Double convolution path
        self.conv1 = gnn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False)
        self.norm1 = gnn.InnerBatchNorm(mid_type)
        self.act1 = gnn.ReLU(mid_type, inplace=True)
        
        self.conv2 = gnn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False)
        self.norm2 = gnn.InnerBatchNorm(out_type)
        self.act2 = gnn.ReLU(out_type, inplace=True)
        
        # Residual connection (1x1 conv to match dimensions)
        self.res_conv = gnn.R2Conv(in_type, out_type, kernel_size=1, bias=False) if in_type != out_type else None
        
        # Store types for forward
        self.in_type = in_type
        self.out_type = out_type
    
    def forward(self, x: gnn.GeometricTensor) -> gnn.GeometricTensor:
        # Verify input type
        assert x.type == self.in_type, f"Input type mismatch: {x.type} vs {self.in_type}"
        
        # Main path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        # Residual connection
        residual = self.res_conv(x) if self.res_conv is not None else x
        
        # Add (must use same field type)
        out = gnn.GeometricTensor(out.tensor + residual.tensor, self.out_type)
        return out
    
    def evaluate_output_shape(self, input_shape):
        return input_shape

class GDBlock(gnn.EquivariantModule):
    """
    Group Equivariant Downsampling Block.
    Downsamples by 2 + applies G-ResnetBlock.
    
    Args:
        in_type: Input field type (p4)
        out_type: Output field type (p4)
    """
    
    def __init__(
        self,
        in_type: gnn.FieldType,
        out_type: gnn.FieldType,
    ):
        super().__init__()
        
        # Downsample with stride 2
        self.down = gnn.R2Conv(in_type, out_type, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Refine features
        self.conv = GResnetBlock(out_type, out_type)
    
    def forward(self, x: gnn.GeometricTensor) -> gnn.GeometricTensor:
        x = self.down(x)
        x = self.conv(x)
        return x
    
    def evaluate_output_shape(self, input_shape):
        # Height and width are halved
        return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2)


class GUBlock(gnn.EquivariantModule):
    """
    Group Equivariant Upsampling Block.
    Upsamples by 2 using interpolation + applies G-ResnetBlock.
    
    Args:
        in_type: Input field type (p4)
        out_type: Output field type (p4)
    """
    
    def __init__(
        self,
        in_type: gnn.FieldType,
        out_type: gnn.FieldType,
    ):
        super().__init__()
        
        self.in_type = in_type
        self.out_type = out_type
        
        # Projection to target channels
        self.proj = gnn.R2Conv(in_type, out_type, kernel_size=1, bias=False)
        
        # Refine features
        self.conv = GResnetBlock(out_type, out_type)
    
    def forward(self, x: gnn.GeometricTensor) -> gnn.GeometricTensor:
        # Upsample using bilinear interpolation
        # Extract tensor, upsample, wrap back
        tensor = x.tensor  # [B, C*4, H, W] for p4
        B, C_total, H, W = tensor.shape
        
        # Upsample
        tensor_up = torch.nn.functional.interpolate(
            tensor, 
            scale_factor=2, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Wrap back into GeometricTensor
        x_up = gnn.GeometricTensor(tensor_up, self.in_type)
        
        # Project to output channels
        x_up = self.proj(x_up)
        
        # Refine
        x_up = self.conv(x_up)
        
        return x_up
    
    def evaluate_output_shape(self, input_shape):
        # Height and width are doubled
        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2)


class GBottleNeck(gnn.EquivariantModule):
    """
    Group Equivariant Bottleneck.
    Multiple G-ResnetBlocks at same resolution.
    
    Args:
        num_blocks: Number of residual blocks
        field_type: Field type (p4)
    """
    
    def __init__(
        self,
        num_blocks: int,
        field_type: gnn.FieldType,
    ):
        super().__init__()
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(GResnetBlock(field_type, field_type))
        
        self.blocks = gnn.SequentialModule(*blocks)
    
    def forward(self, x: gnn.GeometricTensor) -> gnn.GeometricTensor:
        return self.blocks(x)
    
    def evaluate_output_shape(self, input_shape):
        return input_shape


def wrap_tensor(tensor: torch.Tensor, field_type: gnn.FieldType) -> gnn.GeometricTensor:
    """
    Wrap a PyTorch tensor into a GeometricTensor.
    
    Args:
        tensor: Regular PyTorch tensor [B, C, H, W]
        field_type: Target field type
    Returns:
        GeometricTensor
    """
    return gnn.GeometricTensor(tensor, field_type)


def unwrap_tensor(geom_tensor: gnn.GeometricTensor) -> torch.Tensor:
    """
    Extract PyTorch tensor from GeometricTensor.
    
    Args:
        geom_tensor: GeometricTensor
    Returns:
        Regular PyTorch tensor
    """
    return geom_tensor.tensor