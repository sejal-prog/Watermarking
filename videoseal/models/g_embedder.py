# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Group Equivariant Embedder Wrapper
# Drop-in replacement for UnetEmbedder

import torch
from torch import nn

from ..modules.g_unet import GUNetMsg
from ..modules.g_msg_processor import GMsgProcessor
from ..modules.g_common import get_gspace


class GUnetEmbedder(nn.Module):
    """
    Group Equivariant Watermark Embedder.
    
    Drop-in replacement for VideoSeal's UnetEmbedder.
    Uses C4 rotation group for rotation equivariance.
    
    Args:
        gunet: Group equivariant UNet
        msg_processor: Group-aware message processor
    """
    
    def __init__(
        self,
        gunet: GUNetMsg,
        msg_processor: GMsgProcessor
    ):
        super().__init__()
        self.gunet = gunet
        self.msg_processor = msg_processor
        self.preprocess = lambda x: x * 2 - 1  # [0,1] -> [-1,1]
        self.yuv = False
    
    def get_random_msg(self, bsz: int = 1, nb_repetitions: int = 1) -> torch.Tensor:
        """Generate random message bits."""
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)
    
    def get_last_layer(self) -> torch.Tensor:
        """Get last layer weights (for loss computation)."""
        return self.gunet.outc.weights
    
    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Embed watermark into images.
        
        Args:
            imgs: Input images [B, 3, H, W] in range [0, 1]
            msgs: Message bits [B, nbits]
        
        Returns:
            Watermarked images [B, 3, H, W] in range [-1, 1]
        """
        # Preprocess to [-1, 1]
        imgs = self.preprocess(imgs)
        
        # Embed watermark (output is already in [-1, 1] due to tanh)
        imgs_w = self.gunet(imgs, msgs)
        
        return imgs_w


def build_g_embedder(
    nbits: int = 32,
    hidden_size_multiplier: int = 2,
    z_channels: int = 64,
    z_channels_mults: tuple = (1, 2, 4),
    num_blocks: int = 2,
    group_type: str = "C4",
):
    """
    Build a complete Group Equivariant Embedder.
    
    Args:
        nbits: Number of message bits
        hidden_size_multiplier: Multiplier for hidden size
        z_channels: Base feature channels
        z_channels_mults: Channel multipliers for each level
        num_blocks: Number of bottleneck blocks
        group_type: Rotation group ("C4", "C8", "D4")
    
    Returns:
        GUnetEmbedder model
    """
    hidden_size = int(nbits * hidden_size_multiplier)
    
    # Get group order for message processor
    gspace = get_gspace(group_type)
    group_order = gspace.fibergroup.order()
    
    msg_processor = GMsgProcessor(
        nbits=nbits,
        hidden_size=hidden_size,
        group_order=group_order,
        msg_processor_type="binary+concat",
    )
    
    gunet = GUNetMsg(
        msg_processor=msg_processor,
        z_channels=z_channels,
        z_channels_mults=z_channels_mults,
        num_blocks=num_blocks,
        group_type=group_type,
    )
    
    return GUnetEmbedder(gunet, msg_processor)