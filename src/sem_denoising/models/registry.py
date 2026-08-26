"""
Model registry and factory functions for SEM denoising architectures.
"""

from typing import Dict, Any, Type, Callable
import torch
import torch.nn as nn

from sem_denoising.models.direct_cnn import DirectPredictionCNN
from sem_denoising.models.residual_cnn import ResidualPredictionCNN
from sem_denoising.models.dncnn import DnCNN


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "direct_cnn": lambda **kwargs: DirectPredictionCNN(**kwargs),
    "residual_cnn": lambda **kwargs: ResidualPredictionCNN(**kwargs),
    "small_dncnn": lambda **kwargs: DnCNN.create_small(**kwargs),
    "strong_dncnn": lambda **kwargs: DnCNN.create_strong(**kwargs),
    "dncnn": lambda **kwargs: DnCNN(**kwargs),
}


def build_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate a model from registry by name.

    Args:
        model_name: Registry key (e.g. 'direct_cnn', 'residual_cnn', 'small_dncnn', 'strong_dncnn', 'dncnn').
        kwargs: Additional model parameters.

    Returns:
        Instantiated nn.Module.
    """
    key = model_name.lower().replace("-", "_").replace(" ", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key](**kwargs)
