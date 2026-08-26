"""
Unified Denoising Convolutional Neural Network (DnCNN) architecture.
A single, highly configurable class supporting residual noise estimation, direct mapping,
and per-layer skip connections.
"""

from typing import Optional
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    Unified DnCNN architecture.
    
    Supports:
      - Residual learning (residual=True): network outputs noise residual,
        clean image is recovered via: x_clean = x - model(x)
      - Direct mapping (residual=False): network directly outputs estimated clean image
      - Per-layer Skip Connections (use_skip=True): adds a residual skip connection
        across each intermediate convolutional layer: out = out + layer(out)
    """

    def __init__(
        self,
        depth: int = 17,
        num_channels: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
        use_bn: bool = True,
        act_type: str = "relu",
        leaky_slope: float = 0.2,
        residual: bool = True,
        use_skip: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.act_type = act_type
        self.residual = residual
        self.use_skip = use_skip

        def get_activation():
            if act_type.lower() == "leaky_relu":
                return nn.LeakyReLU(leaky_slope, inplace=True)
            return nn.ReLU(inplace=True)

        # 1. Initial feature extraction layer
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1),
            get_activation(),
        )

        # 2. Intermediate convolutional layers
        self.layers = nn.ModuleList()
        for _ in range(depth - 2):
            block = []
            if use_bn:
                block.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False))
                block.append(nn.BatchNorm2d(num_channels))
                block.append(get_activation())
            else:
                block.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
                block.append(get_activation())
            self.layers.append(nn.Sequential(*block))

        # 3. Final projection layer
        self.final_layer = nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional per-layer skip connections.
        
        Args:
            x: Input image tensor (B, C, H, W).
            
        Returns:
            Network output tensor (B, C, H, W).
        """
        out = self.first_layer(x)
        for layer in self.layers:
            if self.use_skip:
                out = out + layer(out)  # Skip connection in each intermediate conv layer
            else:
                out = layer(out)
        return self.final_layer(out)
