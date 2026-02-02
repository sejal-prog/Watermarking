# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Group Equivariant CNN common building blocks using escnn
# For C4 rotation group (0°, 90°, 180°, 270°)

import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn


def get_gspace(group_type: str = "C4"):
    """Get the group space for equivariant convolutions."""
    if group_type == "C4":
        return gspaces.rot2dOnR2(N=4)
    elif group_type == "C8":
        return gspaces.rot2dOnR2(N=8)
    elif group_type == "D4":
        return gspaces.flipRot2dOnR2(N=4)
    else:
        raise ValueError(f"Unknown group type: {group_type}")


class GConvBlock(nn.Module):
    """Group equivariant convolution block: Conv -> Norm -> Act"""
    
    def __init__(
        self,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = enn.R2Conv(
            in_type, out_type,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias
        )
        self.norm = enn.InnerBatchNorm(out_type) # Should it be groupnorm ? 
        self.act = enn.ReLU(out_type, inplace=True)
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class GResnetBlock(nn.Module):
    """Group equivariant ResNet block: Conv-Norm-Act-Conv-Norm-Act + Skip"""
    
    def __init__(
        self,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
        mid_type: enn.FieldType = None,
    ):
        super().__init__()
        if mid_type is None:
            mid_type = out_type
        
        self.double_conv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True),
        )
        
        # Skip connection
        if in_type != out_type:
            self.res_conv = enn.R2Conv(in_type, out_type, kernel_size=1, bias=False)
        else:
            self.res_conv = None
        
        self.out_type = out_type
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        residual = x
        out = self.double_conv(x)
        
        if self.res_conv is not None:
            residual = self.res_conv(residual)
        
        # Add tensors (both are GeometricTensors with same type)
        return enn.GeometricTensor(out.tensor + residual.tensor, self.out_type)


class GDownBlock(nn.Module):
    """Group equivariant downsampling block: Downsample -> ResnetBlock"""
    
    def __init__(
        self,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
    ):
        super().__init__()
        # Downsample with strided conv
        self.down = enn.R2Conv(
            in_type, out_type,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv = GResnetBlock(out_type, out_type)
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        x = self.down(x)
        return self.conv(x)





class GBottleneck(nn.Module):
    """Group equivariant bottleneck: multiple ResNet blocks"""
    
    def __init__(
        self,
        num_blocks: int,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
    ):
        super().__init__()
        blocks = []
        current_type = in_type
        for i in range(num_blocks):
            next_type = out_type if i == num_blocks - 1 else in_type
            blocks.append(GResnetBlock(current_type, next_type))
            current_type = next_type
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        for block in self.blocks:
            x = block(x)
        return x


def create_field_type(gspace, channels: int, type_name: str = "regular"):
    """
    Create a field type for the given group space.
    """
    if type_name == "regular":
        return enn.FieldType(gspace, channels * [gspace.regular_repr])
    elif type_name == "trivial":
        return enn.FieldType(gspace, channels * [gspace.trivial_repr])
    else:
        raise ValueError(f"Unknown field type: {type_name}")