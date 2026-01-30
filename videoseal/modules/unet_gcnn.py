# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Full p4 U-Net with FiLM message conditioning.
Maintains equivariance until final output.
"""

import torch
from torch import nn
from escnn import nn as gnn

from .gcnn_layers import (
    R2_ACT,
    get_trivial_field_type,
    get_regular_field_type,
    GResnetBlock,
    GDBlock,
    GUBlock,
    wrap_tensor,
    unwrap_tensor,
)
from .film_layer import FiLMGenerator, FiLMedGCNNLayer


class UNetMsgGCNN(nn.Module):
    """
    Full p4 U-Net with FiLM conditioning.
    
    Architecture:
        Z2 → p4 encoder → p4 bottleneck (FiLM) → p4 decoder → Average → Z2
    """
    
    def __init__(
        self,
        msg_processor: nn.Module,
        in_channels: int = 3,
        out_channels: int = 3,
        z_channels: int = 64,
        num_blocks: int = 2,
        z_channels_mults: tuple = (1, 2, 4, 8),
        last_tanh: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        
        self.msg_processor = msg_processor
        self.connect_scale = 2 ** -0.5
        self.last_tanh = last_tanh
        
        z_channels_list = [z_channels * m for m in z_channels_mults]
        
        self.in_type = get_trivial_field_type(in_channels)
        self.out_type = get_trivial_field_type(out_channels)
        self.hidden_types = [get_regular_field_type(c) for c in z_channels_list]
        
        # ============ p4 ENCODER ============
        self.inc = GResnetBlock(self.in_type, self.hidden_types[0])
        
        self.downs = nn.ModuleList()
        for i in range(len(z_channels_list) - 1):
            self.downs.append(GDBlock(self.hidden_types[i], self.hidden_types[i + 1]))
        
        # ============ p4 BOTTLENECK with FiLM ============
        # Create FiLM generators for bottleneck blocks
        self.film_generators = nn.ModuleList()
        self.bottleneck_blocks = nn.ModuleList()
        
        for _ in range(num_blocks):
            # FiLM generator
            self.film_generators.append(
                FiLMGenerator(
                    msg_dim=msg_processor.nbits,
                    num_features=z_channels_list[-1],
                    hidden_dim=msg_processor.hidden_size,
                )
            )
            
            # Regular p4 ResNet block
            self.bottleneck_blocks.append(
                GResnetBlock(self.hidden_types[-1], self.hidden_types[-1])
            )
        
        # ============ p4 DECODER ============
        self.ups = nn.ModuleList()
        for i in reversed(range(len(z_channels_list) - 1)):
            up_in_channels = 2 * z_channels_list[i + 1]
            up_in_type = get_regular_field_type(up_in_channels)
            up_out_type = self.hidden_types[i]
            
            self.ups.append(GUBlock(up_in_type, up_out_type))
        
        # ============ FINAL p4 → Z2 PROJECTION ============
        # This averages over rotations
        self.outc = gnn.R2Conv(
            self.hidden_types[0],
            self.out_type,
            kernel_size=1,
        )
        
        if zero_init:
            # Zero-init final layer
            for param in self.outc.parameters():
                nn.init.zeros_(param)
    
    def forward(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: [B, 3, H, W]
            msgs: [B, nbits]
        Returns:
            [B, 3, H, W]
        """
        
        # ============ p4 ENCODER ============
        x = wrap_tensor(imgs, self.in_type)
        x = self.inc(x)
        
        skips = [x]
        for down in self.downs:
            x = down(skips[-1])
            skips.append(x)
        
        # ============ p4 BOTTLENECK with FiLM ============
        x_tensor = unwrap_tensor(x)  # [B, C*4, H, W]
        
        for film_gen, block in zip(self.film_generators, self.bottleneck_blocks):
            # Apply FiLM conditioning
            film_layer = FiLMedGCNNLayer(film_gen)
            x_tensor = film_layer(x_tensor, msgs)
            
            # Apply p4 convolutions
            x = wrap_tensor(x_tensor, self.hidden_types[-1])
            x = block(x)
            x_tensor = unwrap_tensor(x)
        
        x = wrap_tensor(x_tensor, self.hidden_types[-1])
        
        # ============ p4 DECODER ============
        for up in self.ups:
            # Concatenate with skip (in p4 space)
            skip = skips.pop()
            
            skip_tensor = unwrap_tensor(skip) * self.connect_scale
            x_tensor = unwrap_tensor(x) * self.connect_scale
            
            concat_tensor = torch.cat([x_tensor, skip_tensor], dim=1)
            x = wrap_tensor(concat_tensor, up.in_type)
            
            # Upsample
            x = up(x)
        
        # ============ FINAL OUTPUT (p4 → Z2 with averaging) ============
        x = self.outc(x)  # This layer averages over rotations internally
        x_tensor = unwrap_tensor(x)
        
        if self.last_tanh:
            x_tensor = torch.tanh(x_tensor)
        
        return x_tensor


def build_unet_gcnn(msg_processor, **kwargs):
    gcnn_args = {
        'in_channels': kwargs.get('in_channels', 3),
        'out_channels': kwargs.get('out_channels', 3),
        'z_channels': kwargs.get('z_channels', 64),
        'num_blocks': kwargs.get('num_blocks', 2),
        'z_channels_mults': kwargs.get('z_channels_mults', (1, 2, 4, 8)),
        'last_tanh': kwargs.get('last_tanh', True),
        'zero_init': kwargs.get('zero_init', False),
    }
    return UNetMsgGCNN(msg_processor=msg_processor, **gcnn_args)