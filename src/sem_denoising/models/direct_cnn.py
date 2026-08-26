"""
Direct Prediction Convolutional Neural Network for SEM image denoising.
Directly maps noisy input to clean image: x_hat = CNN(y).
"""

import torch
import torch.nn as nn


class DirectPredictionCNN(nn.Module):
    """
    Direct-Prediction Convolutional Neural Network.
    Estimates the clean image directly without explicit residual formulation.
    """

    def __init__(self, depth: int = 5, num_channels: int = 32, in_channels: int = 1, out_channels: int = 1):
        super(DirectPredictionCNN, self).__init__()
        self.depth = depth
        self.num_channels = num_channels

        layers = [
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend([
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        layers.append(nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(x), 0.0, 1.0)

