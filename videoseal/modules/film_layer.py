"""
Feature-wise Linear Modulation (FiLM) for message conditioning in p4 space.
Maintains equivariance by modulating per-channel (not per-rotation).
"""

import torch
from torch import nn


class FiLMGenerator(nn.Module):
    """
    Generates scale (gamma) and shift (beta) from message.
    Converts binary messages to modulation parameters via MLP.
    """
    
    def __init__(self, msg_dim: int, num_features: int, hidden_dim: int = 256):
        super().__init__()
        
        self.num_features = num_features
        
        # MLP to generate gamma and beta from binary messages
        self.mlp = nn.Sequential(
            nn.Linear(msg_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_features),
        )
        
        # Initialize to identity transformation
        # gamma starts at 1.0 (no scaling), beta at 0.0 (no shift)
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()
        self.mlp[-1].bias.data[:num_features] = 1.0  # gamma = 1
    
    def forward(self, msg: torch.Tensor) -> tuple:
        """
        Args:
            msg: [B, msg_dim] - binary messages {0,1}
        Returns:
            gamma: [B, num_features, 1, 1] - scale parameters
            beta: [B, num_features, 1, 1] - shift parameters
        """
        # Convert binary to float if needed
        if msg.dtype == torch.long or msg.dtype == torch.int:
            msg = msg.float()
        
        # Normalize to [-1, 1] for better MLP training
        msg = 2.0 * msg - 1.0  # {0,1} → {-1,1}
        
        # Generate FiLM parameters
        film_params = self.mlp(msg)  # [B, 2*num_features]
        
        # Split into gamma and beta
        gamma = film_params[:, :self.num_features]  # [B, num_features]
        beta = film_params[:, self.num_features:]   # [B, num_features]
        
        # Reshape for broadcasting
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        
        return gamma, beta


class FiLMedGCNNLayer(nn.Module):
    """
    Applies FiLM conditioning to p4 features.
    
    For p4: x has shape [B, C*4, H, W]
    FiLM parameters gamma, beta are [B, C, 1, 1]
    We apply per-channel (across all 4 rotations): x = x * gamma + beta
    
    This preserves p4 equivariance because same transformation
    is applied to all rotation channels of each feature.
    """
    
    def __init__(self, film_generator: FiLMGenerator):
        super().__init__()
        self.film_generator = film_generator
    
    def forward(self, x_p4: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_p4: [B, C*4, H, W] - p4 features
            msg: [B, msg_dim] - binary messages
        Returns:
            [B, C*4, H, W] - FiLM-modulated p4 features
        """
        B, C_times_4, H, W = x_p4.shape
        C = C_times_4 // 4
        
        # Generate FiLM parameters from message
        gamma, beta = self.film_generator(msg)  # [B, C, 1, 1] each
        
        # Reshape to separate rotation dimension
        # [B, C*4, H, W] → [B, C, 4, H, W]
        x_reshaped = x_p4.view(B, C, 4, H, W)
        
        # Apply FiLM (broadcast over rotation dimension)
        # gamma, beta: [B, C, 1, 1] → [B, C, 1, 1, 1] for broadcasting
        gamma = gamma.unsqueeze(2)  # [B, C, 1, 1, 1]
        beta = beta.unsqueeze(2)    # [B, C, 1, 1, 1]
        
        # Apply: same gamma/beta to all 4 rotations of each feature
        x_filmed = x_reshaped * gamma + beta  # [B, C, 4, H, W]
        
        # Reshape back to p4 format
        # [B, C, 4, H, W] → [B, C*4, H, W]
        x_filmed = x_filmed.view(B, C_times_4, H, W)
        
        return x_filmed