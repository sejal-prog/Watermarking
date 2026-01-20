# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Group Equivariant U-Net for rotation-robust watermarking.
"""

import torch
from torch import nn
from escnn import gspaces, nn as gnn

from .gcnn_layers import (
    R2_ACT,
    get_trivial_field_type,
    get_regular_field_type,
    GResnetBlock,
    GDBlock,
    GUBlock,
    GBottleNeck,
    wrap_tensor,
    unwrap_tensor,
)


class UNetMsgGCNN(nn.Module):
    """
    Group Equivariant U-Net with message injection at bottleneck.
    
    Architecture:
        Input (Z2) → Encoder (p4) → Bottleneck (p4) → Decoder (p4) → Output (Z2)
    
    Args:
        msg_processor: Message embedding module
        in_channels: Input image channels (3 for RGB)
        out_channels: Output channels (3 for RGB watermark)
        z_channels: Base hidden channels
        num_blocks: Number of bottleneck blocks
        z_channels_mults: Channel multipliers for each stage
        last_tanh: Apply tanh to output
        zero_init: Zero-initialize output layer
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.num_blocks = num_blocks
        self.z_channels_mults = z_channels_mults
        self.last_tanh = last_tanh
        self.connect_scale = 2 ** -0.5
        
        # Calculate channel dimensions
        z_channels_list = [z_channels * m for m in z_channels_mults]
        
        # Create field types
        self.in_type = get_trivial_field_type(in_channels)  # Z2 (regular image)
        self.out_type = get_trivial_field_type(out_channels)  # Z2 (regular image)
        
        # Hidden layer types (p4 - with rotations)
        self.hidden_types = [get_regular_field_type(c) for c in z_channels_list]
        
        # ============ ENCODER ============
        
        # Initial lifting layer (Z2 → p4)
        self.inc = GResnetBlock(
            self.in_type,
            self.hidden_types[0],
        )
        
        # Downsampling layers (p4 → p4)
        self.downs = nn.ModuleList()
        for i in range(len(z_channels_list) - 1):
            self.downs.append(
                GDBlock(
                    self.hidden_types[i],
                    self.hidden_types[i + 1],
                )
            )
        
        # ============ BOTTLENECK ============
        
        # After message injection, channels increase
        bottleneck_channels = z_channels_list[-1] + self.msg_processor.hidden_size
        self.bottleneck_type = get_regular_field_type(bottleneck_channels)
        
        self.bottleneck = GBottleNeck(
            num_blocks=num_blocks,
            field_type=self.bottleneck_type,
        )
        
        # ============ DECODER ============
        
        # Upsampling layers (p4 → p4)
        self.ups = nn.ModuleList()
        for i in reversed(range(len(z_channels_list) - 1)):
            # Input has skip connection: 2 × channels
            up_in_channels = 2 * z_channels_list[i + 1]
            if i == len(z_channels_list) - 2:
                # First upsampling: comes from bottleneck
                up_in_channels = 2 * bottleneck_channels
            
            up_in_type = get_regular_field_type(up_in_channels)
            up_out_type = self.hidden_types[i]
            
            self.ups.append(
                GUBlock(up_in_type, up_out_type)
            )
        
        # ============ OUTPUT ============
        
        # Final projection layer (p4 → Z2)
        self.outc = gnn.R2Conv(
            self.hidden_types[0],
            self.out_type,
            kernel_size=1,
        )
        
        if zero_init:
            self._zero_init_output()
    
    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            imgs: Input images [B, 3, H, W]
            msgs: Binary messages [B, nbits]
        Returns:
            Watermarked images [B, 3, H, W]
        """
        
        # Wrap input as GeometricTensor (Z2)
        x = wrap_tensor(imgs, self.in_type)
        
        # ============ ENCODER ============
        
        # Initial convolution (Z2 → p4)
        x = self.inc(x)  # [B, 64*4, H, W] in tensor form
        hiddens = [x]
        
        # Downsampling (p4 → p4)
        for down in self.downs:
            x = down(hiddens[-1])
            hiddens.append(x)
        
        # ============ MESSAGE INJECTION ============
        
        # Unwrap to regular tensor
        x_tensor = unwrap_tensor(hiddens.pop())  # [B, 512*4, H, W]
        
        # For p4, we need to handle the group dimension
        # escnn stores as [B, C*|G|, H, W] where |G|=4
        # We need to reshape to [B, C, |G|, H, W] for message processing
        B, C_times_G, H, W = x_tensor.shape
        C = C_times_G // 4  # Recover channel dimension
        
        # Reshape to [B, C, 4, H, W]
        x_tensor = x_tensor.view(B, C, 4, H, W)
        
        # Average over rotation dimension for message processing
        x_avg = x_tensor.mean(dim=2)  # [B, C, H, W]
        
        # Apply message processor
        x_with_msg = self.msg_processor(x_avg, msgs)  # [B, C+hidden_size, H, W]
        
        # Expand back to all rotations
        C_new = x_with_msg.shape[1]
        x_with_msg = x_with_msg.unsqueeze(2).repeat(1, 1, 4, 1, 1)  # [B, C_new, 4, H, W]
        
        # Reshape back to [B, C_new*4, H, W]
        x_with_msg = x_with_msg.view(B, C_new * 4, H, W)
        
        # Wrap back as GeometricTensor
        x = wrap_tensor(x_with_msg, self.bottleneck_type)
        hiddens.append(x)
        
        # ============ BOTTLENECK ============
        
        x = self.bottleneck(hiddens[-1])
        
        # ============ DECODER ============
        
        # Upsampling with skip connections
        for up in self.ups:
            # Concatenate with skip connection
            skip = hiddens.pop()
            
            # Scale skip connection
            skip_tensor = unwrap_tensor(skip) * self.connect_scale
            skip_scaled = wrap_tensor(skip_tensor, skip.type)
            
            # Concatenate
            x_tensor = unwrap_tensor(x)
            concat_tensor = torch.cat([x_tensor, skip_tensor], dim=1)
            
            # Wrap with appropriate type
            concat_type = up.in_type
            x = wrap_tensor(concat_tensor, concat_type)
            
            # Upsample
            x = up(x)
        
        # ============ OUTPUT ============
        
        # Project to Z2 (average over rotations)
        logits = self.outc(x)
        logits_tensor = unwrap_tensor(logits)
        
        if self.last_tanh:
            logits_tensor = torch.tanh(logits_tensor)
        
        return logits_tensor
    
    def _zero_init_output(self):
        """Zero-initialize the output layer."""
        for param in self.outc.parameters():
            nn.init.zeros_(param)


# ============ BUILDER FUNCTION ============

def build_unet_gcnn(msg_processor, **kwargs):
    """
    Build a GCNN U-Net.
    
    Args:
        msg_processor: Message embedding module
        **kwargs: Arguments for UNetMsgGCNN (filters out regular UNet args)
    Returns:
        UNetMsgGCNN model
    """
    # Filter only GCNN-relevant arguments
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