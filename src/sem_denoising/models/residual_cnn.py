"""
Residual Prediction Convolutional Neural Network for SEM image denoising.
Estimates the noise residual and subtracts it from the noisy input: x_hat = y - CNN(y).
Has the exact same layer architecture and parameter count as DirectPredictionCNN.
"""

import torch
import torch.nn as nn


class ResidualPredictionCNN(nn.Module):
    """
    Residual-Prediction Convolutional Neural Network.
    Estimates the noise residual R(y) such that x_hat = y - R(y).
    """

    def __init__(
        self,
        depth: int = 5,
        num_channels: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super(ResidualPredictionCNN, self).__init__()
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
        residual = self.net(x)
        return torch.clamp(x - residual, 0.0, 1.0)
