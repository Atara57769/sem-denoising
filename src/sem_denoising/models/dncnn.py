"""
Unified Denoising Convolutional Neural Network (DnCNN) architecture.
A single, highly configurable class supporting both residual noise estimation and direct mapping.
"""

from typing import Optional
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    Unified DnCNN architecture.
    
    Supports both:
      - Residual learning (residual=True): network outputs noise residual,
        clean image is recovered via: x_clean = x - model(x)
      - Direct mapping (residual=False): network directly outputs estimated clean image
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
    ):
        super().__init__()
        self.depth = depth
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.act_type = act_type
        self.residual = residual

        def get_activation():
            if act_type.lower() == "leaky_relu":
                return nn.LeakyReLU(leaky_slope, inplace=True)
            return nn.ReLU(inplace=True)

        layers = [
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1),
            get_activation(),
        ]

        for _ in range(depth - 2):
            if use_bn:
                layers.extend([
                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_channels),
                    get_activation(),
                ])
            else:
                layers.extend([
                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                    get_activation(),
                ])

        layers.append(nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw network output directly.
        
        Args:
            x: Input noisy image tensor (B, C, H, W).
            
        Returns:
            Network output tensor (B, C, H, W).
        """
        return self.net(x)
