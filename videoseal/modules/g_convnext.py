# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Group Equivariant ConvNeXt-style backbone for watermark extraction
# Uses C4 rotation group by default
# FIXED: Pixelwise prediction like VideoSeal instead of global pooling

import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn

from .g_common import get_gspace


class GConvNeXtBlock(nn.Module):
    """
    Group Equivariant ConvNeXt-style block.
    Simplified version: Conv -> Norm -> Act -> Conv
    """
    
    def __init__(
        self,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
        expansion: int = 4,
    ):
        super().__init__()
        
        gspace = in_type.gspace
        in_channels = len(in_type)
        out_channels = len(out_type)
        mid_channels = out_channels * expansion
        
        mid_type = enn.FieldType(gspace, mid_channels * [gspace.regular_repr])
        
        self.block = enn.SequentialModule(
            enn.R2Conv(in_type, in_type, kernel_size=7, padding=3, bias=False),
            enn.InnerBatchNorm(in_type),
            enn.R2Conv(in_type, mid_type, kernel_size=1, bias=False),
            enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=1, bias=False),
            enn.InnerBatchNorm(out_type),
        )
        
        if in_type != out_type:
            self.skip = enn.R2Conv(in_type, out_type, kernel_size=1, bias=False)
        else:
            self.skip = None
        
        self.out_type = out_type
        self.act = enn.ReLU(out_type, inplace=True)
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        residual = x
        out = self.block(x)
        
        if self.skip is not None:
            residual = self.skip(residual)
        
        out = enn.GeometricTensor(out.tensor + residual.tensor, self.out_type)
        return self.act(out)


class GConvNeXtStage(nn.Module):
    """
    Group Equivariant ConvNeXt stage with optional downsampling.
    """
    
    def __init__(
        self,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
        num_blocks: int,
        downsample: bool = True,
    ):
        super().__init__()
        
        if downsample:
            self.downsample = enn.SequentialModule(
                enn.PointwiseAvgPool(in_type, kernel_size=2, stride=2),
                enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
                enn.InnerBatchNorm(out_type),
            )
        else:
            if in_type != out_type:
                self.downsample = enn.R2Conv(in_type, out_type, kernel_size=1, bias=False)
            else:
                self.downsample = None
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(GConvNeXtBlock(out_type, out_type))
        self.blocks = nn.ModuleList(blocks)
        
        self.out_type = out_type
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        if self.downsample is not None:
            x = self.downsample(x)
        
        for block in self.blocks:
            x = block(x)
        
        return x


class GConvNeXtExtractor(nn.Module):
    """
    Group Equivariant ConvNeXt-style backbone for watermark extraction.
    
    FIXED: Uses pixelwise prediction like VideoSeal instead of global pooling.
    
    Architecture:
        1. Lift RGB to group space
        2. Multiple stages with G-Conv blocks and downsampling
        3. Group pooling (average over rotations) → Invariant features
        4. 1x1 Conv → Pixelwise predictions [B, nbits, H, W]
        5. Spatial average → Final predictions [B, nbits]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        nbits: int = 32,
        depths: list = [2, 2, 6, 2],
        dims: list = [64, 128, 256, 512],
        group_type: str = "C4",
    ):
        super().__init__()
        
        self.nbits = nbits
        self.gspace = get_gspace(group_type)
        self.group_order = self.gspace.fibergroup.order()
        
        # === LIFT: RGB to Group Space ===
        self.in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])
        self.first_type = enn.FieldType(self.gspace, dims[0] * [self.gspace.regular_repr])
        
        # Stem: lift and initial downsample
        self.stem = enn.SequentialModule(
            enn.R2Conv(self.in_type, self.first_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(self.first_type),
            enn.ReLU(self.first_type, inplace=True),
            enn.PointwiseAvgPool(self.first_type, kernel_size=4, stride=4),
        )
        
        # === STAGES ===
        self.stages = nn.ModuleList()
        
        for i in range(len(depths)):
            in_dim = dims[i-1] if i > 0 else dims[0]
            out_dim = dims[i]
            
            in_type = enn.FieldType(self.gspace, in_dim * [self.gspace.regular_repr])
            out_type = enn.FieldType(self.gspace, out_dim * [self.gspace.regular_repr])
            
            downsample = (i > 0)
            
            stage = GConvNeXtStage(
                in_type=in_type,
                out_type=out_type,
                num_blocks=depths[i],
                downsample=downsample,
            )
            self.stages.append(stage)
        
        # === GROUP POOLING ===
        self.final_type = enn.FieldType(self.gspace, dims[-1] * [self.gspace.regular_repr])
        self.group_pool = enn.GroupPooling(self.final_type)
        
        # === PIXELWISE HEAD (like VideoSeal) ===
        # After group pooling: [B, dims[-1], H, W]
        # Predict at each pixel, then average
        self.pixel_head = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[-1], nbits, kernel_size=1, bias=True),
        )
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract watermark message from image.
        
        Args:
            imgs: Input images [B, 3, H, W] in range [-1, 1]
        
        Returns:
            Message logits [B, nbits]
        """
        # === LIFT ===
        x = enn.GeometricTensor(imgs, self.in_type)
        x = self.stem(x)
        
        # === STAGES ===
        for stage in self.stages:
            x = stage(x)
        
        # === GROUP POOLING (rotation invariance) ===
        x = self.group_pool(x)  # [B, C*G, H, W] -> [B, C, H, W]
        x = x.tensor  # Extract tensor
        
        # === PIXELWISE PREDICTION ===
        x = self.pixel_head(x)  # [B, nbits, H, W]
        
        # === SPATIAL AVERAGE ===
        logits = x.mean(dim=[-2, -1])  # [B, nbits]
        
        return logits


def build_g_convnext_extractor(
    nbits: int = 32,
    depths: list = [2, 2, 6, 2],
    dims: list = [64, 128, 256, 512],
    group_type: str = "C4",
):
    """
    Build a Group Equivariant ConvNeXt Extractor.
    """
    return GConvNeXtExtractor(
        nbits=nbits,
        depths=depths,
        dims=dims,
        group_type=group_type,
    )