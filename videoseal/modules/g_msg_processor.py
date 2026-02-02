# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Group Equivariant Message Processor
# Replicates message across group dimension for proper fusion

import torch
import torch.nn as nn
from escnn import nn as enn


class GMsgProcessor(nn.Module):
    """
    Group-aware message processor.
    
    Takes message bits, creates embeddings, and replicates across group dimension
    for proper fusion with equivariant image features.
    
    Args:
        nbits: Number of bits in the message
        hidden_size: Dimension of message embedding
        group_order: Order of the rotation group (e.g., 4 for C4)
        msg_processor_type: Type of message processing ("binary+concat" or "binary+add")
        msg_mult: Multiplier for message embedding
    """
    
    def __init__(
        self,
        nbits: int,
        hidden_size: int,
        group_order: int = 4,
        msg_processor_type: str = "binary+concat",
        msg_mult: float = 1.0,
    ):
        super().__init__()
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.group_order = group_order
        self.msg_mult = msg_mult
        
        # Parse processor type
        self.msg_processor_type = msg_processor_type if nbits > 0 else "none+_"
        self.msg_type = self.msg_processor_type.split("+")[0]
        self.msg_agg = self.msg_processor_type.split("+")[1]
        
        # Create message embeddings
        if self.msg_type.startswith("no"):
            self.msg_embeddings = None
        elif self.msg_type.startswith("bin"):
            self.msg_embeddings = nn.Embedding(2 * nbits, hidden_size)
        elif self.msg_type.startswith("gau"):
            self.msg_embeddings = nn.Embedding(nbits, hidden_size)
        else:
            raise ValueError(f"Invalid msg_processor_type: {self.msg_processor_type}")
    
    def get_random_msg(self, bsz: int = 1, nb_repetitions: int = 1) -> torch.Tensor:
        """Generate a random message."""
        if self.msg_type.startswith("bin"):
            if nb_repetitions != 1:
                assert self.nbits % nb_repetitions == 0
                aux = torch.randint(0, 2, (bsz, self.nbits // nb_repetitions))
                return aux.unsqueeze(1).repeat(1, nb_repetitions, 1).view(bsz, self.nbits)
            else:
                return torch.randint(0, 2, (bsz, self.nbits))
        elif self.msg_type.startswith("gau"):
            gauss_vecs = torch.randn(bsz, self.nbits)
            gauss_vecs = gauss_vecs / torch.norm(gauss_vecs, dim=-1, keepdim=True)
            return gauss_vecs
        return torch.tensor([])
    
    def forward(
        self,
        latents: enn.GeometricTensor,
        msg: torch.Tensor,
        verbose: bool = False
    ) -> enn.GeometricTensor:
        """
        Apply the message to the group-equivariant latents.
        
        Args:
            latents: GeometricTensor with shape [B, C * |G|, H, W] internally
                     where C is number of channels and |G| is group order
            msg: Message tensor [B, nbits]
        
        Returns:
            GeometricTensor with message fused (concatenated or added)
        """
        if self.nbits == 0:
            return latents
        
        # Get tensor and shape info
        latent_tensor = latents.tensor  # [B, C * |G|, H, W]
        B, total_channels, H, W = latent_tensor.shape
        G = self.group_order
        C = total_channels // G  # Channels per group element
        
        # Create message embeddings (same as original VideoSeal)
        if self.msg_type.startswith("bin"):
            indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)
            indices = indices.repeat(msg.shape[0], 1)
            indices = (indices + msg).long()
            msg_aux = self.msg_embeddings(indices)  # [B, nbits, hidden_size]
            msg_aux = msg_aux.sum(dim=-2)  # [B, hidden_size]
        elif self.msg_type.startswith("gau"):
            indices = torch.arange(msg.shape[-1]).to(msg.device).long()
            msg_aux = self.msg_embeddings(indices)
            msg_aux = torch.einsum("kd, bk -> bd", msg_aux, msg)
        else:
            raise ValueError(f"Invalid msg_type: {self.msg_type}")
        
        # Expand spatially: [B, hidden_size] -> [B, hidden_size, H, W]
        msg_aux = msg_aux.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        
        # CRITICAL: Replicate across group dimension
        # escnn expects layout: [ch0_rot0, ch0_rot1, ch0_rot2, ch0_rot3, ch1_rot0, ...]
        # So each channel value must be repeated G times CONSECUTIVELY
        
        # Method: [B, hidden_size, H, W] -> [B, hidden_size, G, H, W] -> [B, hidden_size * G, H, W]
        msg_aux = msg_aux.unsqueeze(2)  # [B, hidden_size, 1, H, W]
        msg_aux = msg_aux.expand(-1, -1, G, -1, -1)  # [B, hidden_size, G, H, W]
        msg_aux = msg_aux.reshape(B, self.hidden_size * G, H, W)  # [B, hidden_size * G, H, W]
        
        # Now: ch0,ch0,ch0,ch0, ch1,ch1,ch1,ch1, ... (each channel repeated G times)
        
        if verbose:
            print(f"Latent shape: {latent_tensor.shape}")
            print(f"Message shape after replication: {msg_aux.shape}")
        
        # Apply message to latents
        if self.msg_agg == "concat":
            # Concatenate along channel dimension
            fused = torch.cat([latent_tensor, self.msg_mult * msg_aux], dim=1)
        elif self.msg_agg == "add":
            # For add, hidden_size must match C
            assert self.hidden_size == C, \
                f"For 'add' mode, hidden_size ({self.hidden_size}) must equal channels ({C})"
            fused = latent_tensor + self.msg_mult * msg_aux
        else:
            raise ValueError(f"Invalid msg_agg: {self.msg_agg}")
        
        # Create new FieldType for the output
        gspace = latents.type.gspace
        if self.msg_agg == "concat":
            # New type has additional channels for message
            new_channels = C + self.hidden_size
            out_type = enn.FieldType(gspace, new_channels * [gspace.regular_repr])
        else:
            out_type = latents.type
        
        return enn.GeometricTensor(fused, out_type)