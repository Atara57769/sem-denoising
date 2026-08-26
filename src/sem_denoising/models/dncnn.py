"""
Unified Denoising Convolutional Neural Network (DnCNN) architecture.
A single, highly configurable class supporting small (5L, 32ch) to deep research (17L, 64ch) models.
"""

from typing import Optional
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    Unified DnCNN architecture (Zhang et al.).
    
    Residual formulation estimating the noise component with batch normalization
    and configurable activation function and depth.
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
    ):
        super(DnCNN, self).__init__()
        self.depth = depth
        self.num_channels = num_channels
        self.use_bn = use_bn
        self.act_type = act_type

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
        residual = self.net(x)
        return torch.clamp(x - residual, 0.0, 1.0)

    @classmethod
    def create_small(cls, depth: int = 5, num_channels: int = 32) -> "DnCNN":
        """Preset for 5-layer lightweight Small DnCNN with LeakyReLU."""
        return cls(
            depth=depth,
            num_channels=num_channels,
            use_bn=True,
            act_type="leaky_relu",
            leaky_slope=0.2,
        )

    @classmethod
    def create_strong(cls, depth: int = 17, num_channels: int = 64) -> "DnCNN":
        """Preset for 17-layer standard research Strong DnCNN with ReLU."""
        return cls(
            depth=depth,
            num_channels=num_channels,
            use_bn=True,
            act_type="relu",
        )
