"""
Group Equivariant CNN layers for rotation-robust watermarking.
Uses p4 group (4 rotations: 0°, 90°, 180°, 270°).
"""

import torch
from torch import nn
from escnn import gspaces, nn as gnn

# Initialize the rotation group (p4)
R2_ACT = gspaces.rot2dOnR2(N=4)


def get_trivial_field_type(n_channels: int):
    """
    Create field type for Z2 (regular 2D space, no rotations).
    Used for input/output layers.
    """
    return gnn.FieldType(R2_ACT, n_channels * [R2_ACT.trivial_repr])


def get_regular_field_type(n_channels: int):
    """
    Create field type for p4 (4 rotation channels).
    Used for hidden layers.
    """
    return gnn.FieldType(R2_ACT, n_channels * [R2_ACT.regular_repr])


class GResnetBlock(gnn.EquivariantModule):
    """Group Equivariant ResNet Block."""
    
    def __init__(self, in_type: gnn.FieldType, out_type: gnn.FieldType):
        super().__init__()
        
        self.conv1 = gnn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        self.norm1 = gnn.InnerBatchNorm(out_type)
        self.act1 = gnn.ReLU(out_type, inplace=True)
        
        self.conv2 = gnn.R2Conv(out_type, out_type, kernel_size=3, padding=1, bias=False)
        self.norm2 = gnn.InnerBatchNorm(out_type)
        self.act2 = gnn.ReLU(out_type, inplace=True)
        
        self.res_conv = gnn.R2Conv(in_type, out_type, kernel_size=1, bias=False) if in_type != out_type else None
        
        self.in_type = in_type
        self.out_type = out_type
    
    def forward(self, x: gnn.GeometricTensor) -> gnn.GeometricTensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        residual = self.res_conv(x) if self.res_conv is not None else x
        out = gnn.GeometricTensor(out.tensor + residual.tensor, self.out_type)
        
        return out
    
    def evaluate_output_shape(self, input_shape):
        return input_shape
    
class GUBlock(gnn.EquivariantModule):
    """
    Group Equivariant Upsampling Block.
    """
    
    def __init__(self, in_type: gnn.FieldType, out_type: gnn.FieldType):
        super().__init__()
        
        self.in_type = in_type
        self.out_type = out_type
        
        self.proj = gnn.R2Conv(in_type, out_type, kernel_size=1, bias=False)
        self.conv = GResnetBlock(out_type, out_type)
    
    def forward(self, x: gnn.GeometricTensor) -> gnn.GeometricTensor:
        # Upsample
        tensor = x.tensor
        tensor_up = torch.nn.functional.interpolate(
            tensor, scale_factor=2, mode='bilinear', align_corners=False
        )
        x_up = gnn.GeometricTensor(tensor_up, self.in_type)
        
        # Project and refine
        x_up = self.proj(x_up)
        x_up = self.conv(x_up)
        
        return x_up
    
    def evaluate_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2)


class GDBlock(gnn.EquivariantModule):
    """Group Equivariant Downsampling Block."""
    
    def __init__(self, in_type: gnn.FieldType, out_type: gnn.FieldType):
        super().__init__()
        
        self.down = gnn.R2Conv(in_type, out_type, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv = GResnetBlock(out_type, out_type)
    
    def forward(self, x: gnn.GeometricTensor) -> gnn.GeometricTensor:
        x = self.down(x)
        x = self.conv(x)
        return x
    
    def evaluate_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2)


def wrap_tensor(tensor: torch.Tensor, field_type: gnn.FieldType) -> gnn.GeometricTensor:
    """Wrap PyTorch tensor into GeometricTensor."""
    return gnn.GeometricTensor(tensor, field_type)


def unwrap_tensor(geom_tensor: gnn.GeometricTensor) -> torch.Tensor:
    """Extract PyTorch tensor from GeometricTensor."""
    return geom_tensor.tensor