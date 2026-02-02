# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Group Equivariant Extractor Wrapper
# Drop-in replacement for ConvnextExtractor

import torch
from torch import nn

from ..modules.g_convnext import GConvNeXtExtractor, build_g_convnext_extractor


class GConvnextExtractor(nn.Module):
    """
    Group Equivariant Watermark Extractor.
    
    Drop-in replacement for VideoSeal's ConvnextExtractor.
    Uses C4 rotation group for rotation invariance.
    
    Key difference from embedder:
        - Embedder: equivariant output (watermark rotates with image)
        - Extractor: INVARIANT output (same message bits regardless of rotation)
    
    Args:
        g_convnext: Group equivariant ConvNeXt backbone
    """
    
    def __init__(
        self,
        g_convnext: GConvNeXtExtractor,
    ):
        super().__init__()
        self.g_convnext = g_convnext
        self.preprocess = lambda x: x * 2 - 1  # [0,1] -> [-1,1]
        self.nbits = g_convnext.nbits
    
    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract watermark message from images.
        
        Args:
            imgs: Input images [B, 3, H, W] in range [0, 1]
        
        Returns:
            Message logits [B, nbits]
            (Apply sigmoid + threshold 0.5 to get binary bits)
        """
        # Preprocess to [-1, 1]
        imgs = self.preprocess(imgs)
        
        # Extract message
        logits = self.g_convnext(imgs)
        
        return logits
    
    def decode_message(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to binary message.
        
        Args:
            logits: Message logits [B, nbits]
        
        Returns:
            Binary message [B, nbits] (0 or 1)
        """
        return (torch.sigmoid(logits) > 0.5).float()


def build_g_extractor(
    nbits: int = 32,
    depths: list = [2, 2, 6, 2],
    dims: list = [64, 128, 256, 512],
    group_type: str = "C4",
):
    """
    Build a complete Group Equivariant Extractor.
    
    Args:
        nbits: Number of message bits to extract
        depths: Number of blocks at each stage
        dims: Channel dimensions at each stage
        group_type: Rotation group ("C4", "C8", "D4")
    
    Returns:
        GConvnextExtractor model
    """
    g_convnext = build_g_convnext_extractor(
        nbits=nbits,
        depths=depths,
        dims=dims,
        group_type=group_type,
    )
    
    return GConvnextExtractor(g_convnext)