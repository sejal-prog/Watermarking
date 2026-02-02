# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Group Equivariant UNet for Watermark Embedding
# Uses C4 rotation group by default

import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn

from .g_common import (
    get_gspace, 
    create_field_type,
    GResnetBlock, 
    GDownBlock, 
    GBottleneck
)
from .g_msg_processor import GMsgProcessor


class GUpBlock(nn.Module):
    """Group equivariant upsampling block: Upsample -> Conv -> ResnetBlock"""
    
    def __init__(
        self,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
    ):
        super().__init__()
        self.in_type = in_type  
        self.out_type = out_type
        
        # Conv to change channels after upsample
        self.up_conv = enn.R2Conv(
            in_type, out_type,
            kernel_size=3, padding=1, bias=False
        )
        self.conv = GResnetBlock(out_type, out_type)
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        # Bilinear upsampling (equivariant operation)
        upsampled = torch.nn.functional.interpolate(
            x.tensor, scale_factor=2, mode='bilinear', align_corners=False
        )
        x = enn.GeometricTensor(upsampled, self.in_type)
        
        x = self.up_conv(x)
        return self.conv(x)


class GUNetMsg(nn.Module):
    """
    Group Equivariant UNet for watermark embedding.
    """
    
    def __init__(
        self,
        msg_processor: GMsgProcessor,
        in_channels: int = 3,
        out_channels: int = 3,
        z_channels: int = 64,
        num_blocks: int = 2,
        z_channels_mults: tuple = (1, 2, 4),
        group_type: str = "C4",
        last_tanh: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.msg_processor = msg_processor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.num_blocks = num_blocks
        self.z_channels_mults = z_channels_mults
        self.last_tanh = last_tanh
        self.connect_scale = 2 ** -0.5
        
        
        self.gspace = get_gspace(group_type)
        self.group_order = self.gspace.fibergroup.order()
        
        
        z_channels_list = [z_channels * m for m in z_channels_mults]
        self.z_channels_list = z_channels_list
        
        
        self.in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])
        self.first_type = enn.FieldType(self.gspace, z_channels_list[0] * [self.gspace.regular_repr])
        
        self.lift = enn.R2Conv(
            self.in_type, self.first_type,
            kernel_size=3, padding=1, bias=False
        )
        
        # ENCODER 
        self.inc = GResnetBlock(self.first_type, self.first_type)
        
        self.downs = nn.ModuleList()
       
        self.encoder_types = [self.first_type]
        
        for i in range(len(z_channels_list) - 1):
            in_type = enn.FieldType(self.gspace, z_channels_list[i] * [self.gspace.regular_repr])
            out_type = enn.FieldType(self.gspace, z_channels_list[i + 1] * [self.gspace.regular_repr])
            self.downs.append(GDownBlock(in_type, out_type))
            self.encoder_types.append(out_type)
        
        # MESSAGE INJECTION 
        bottleneck_in_channels = z_channels_list[-1] + msg_processor.hidden_size
        bottleneck_in_type = enn.FieldType(
            self.gspace, bottleneck_in_channels * [self.gspace.regular_repr]
        )
        
        self.bottleneck = GBottleneck(num_blocks, bottleneck_in_type, bottleneck_in_type)
        self.bottleneck_out_type = bottleneck_in_type
        
        # DECODER 
        self.ups = nn.ModuleList()
        self.up_in_types = []  
        
        x_channels = bottleneck_in_channels
        
        for i in reversed(range(len(z_channels_list) - 1)):
            skip_channels = z_channels_list[i + 1]
            concat_channels = x_channels + skip_channels
            out_channels_up = z_channels_list[i]
            
            in_type = enn.FieldType(self.gspace, concat_channels * [self.gspace.regular_repr])
            out_type = enn.FieldType(self.gspace, out_channels_up * [self.gspace.regular_repr])
            
            self.ups.append(GUpBlock(in_type, out_type))
            self.up_in_types.append(in_type)
            
            x_channels = out_channels_up
        
        self.final_type = enn.FieldType(self.gspace, z_channels_list[0] * [self.gspace.regular_repr])
        self.out_type = enn.FieldType(self.gspace, out_channels * [self.gspace.trivial_repr])
        
        self.outc = enn.R2Conv(self.final_type, self.out_type, kernel_size=3, padding=1, bias=True)
        
        #putting 0.2 here because it is standard - vidoeseal
        self.register_buffer('watermark_strength', torch.tensor(0.2))
    
    def forward(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for watermark embedding.
        
        Args:
            imgs: Input images [B, 3, H, W] in range [-1, 1]
            msgs: Message bits [B, nbits]
        
        Returns:
            Watermarked images [B, 3, H, W] in range [-1, 1]
        """
        x = enn.GeometricTensor(imgs, self.in_type)
        x = self.lift(x)
        
        #  ENCODER 
        x = self.inc(x)
        hiddens = [x]
        
        for down in self.downs:
            x = down(x)
            hiddens.append(x)
        
        #  MESSAGE INJECTION 
        x = self.msg_processor(x, msgs)
        
        #BOTTLENECK 
        x = self.bottleneck(x)
        
        # DECODER 
        for idx, up in enumerate(self.ups):
            skip = hiddens.pop()
            
            x_tensor = x.tensor
            skip_tensor = skip.tensor * self.connect_scale
            
            concat_tensor = torch.cat([x_tensor, skip_tensor], dim=1)
            
            x = enn.GeometricTensor(concat_tensor, self.up_in_types[idx])
            
            x = up(x)
        
        x = self.outc(x)
        watermark = x.tensor
        
        if self.last_tanh:
            watermark = torch.tanh(watermark)
        
        watermark = self.watermark_strength * watermark
        
        out = imgs + watermark
        
        out = torch.clamp(out, -1, 1)
        
        return out