# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Group Equivariant Extractor for rotation-robust watermark detection.
"""

import torch
from torch import nn

from .gcnn_layers import (
    R2_ACT,
    get_trivial_field_type,
    get_regular_field_type,
    GResnetBlock,
    GDBlock,
    wrap_tensor,
    unwrap_tensor,
)
from .pixel_decoder import PixelDecoder


class GCNNEncoder(nn.Module):
    """
    GCNN Encoder: Z2 → p4 → pool → Z2
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        z_channels: int = 64,
        z_channels_mults: tuple = (1, 2, 4, 8),
    ):
        super().__init__()
        
        z_channels_list = [z_channels * m for m in z_channels_mults]
        
        self.in_type = get_trivial_field_type(in_channels)
        self.hidden_types = [get_regular_field_type(c) for c in z_channels_list]
        
        # Lift to p4
        self.inc = GResnetBlock(self.in_type, self.hidden_types[0])
        
        # Downsample in p4
        self.downs = nn.ModuleList()
        for i in range(len(z_channels_list) - 1):
            self.downs.append(GDBlock(self.hidden_types[i], self.hidden_types[i + 1]))
        
        self.out_channels = z_channels_list[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, C, H/32, W/32]
        """
        # Lift to p4
        x = wrap_tensor(x, self.in_type)
        x = self.inc(x)
        
        # Downsample
        for down in self.downs:
            x = down(x)
        
        # Pool rotations
        x_tensor = unwrap_tensor(x)
        B, C_times_4, H, W = x_tensor.shape
        C = C_times_4 // 4
        x = x_tensor.view(B, C, 4, H, W).mean(dim=2)
        
        return x


class GCNNExtractor(nn.Module):
    """
    GCNN Extractor: p4 encoder → pool → decoder
    """
    
    def __init__(
        self,
        gcnn_encoder: GCNNEncoder,
        pixel_decoder: PixelDecoder,
    ):
        super().__init__()
        self.preprocess = lambda x: x * 2 - 1
        self.gcnn_encoder = gcnn_encoder
        self.pixel_decoder = pixel_decoder
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: [B, 3, H, W]
        Returns:
            [B, 1+nbits, H, W]
        """
        imgs = self.preprocess(imgs)
        latents = self.gcnn_encoder(imgs)
        preds = self.pixel_decoder(latents)
        return preds