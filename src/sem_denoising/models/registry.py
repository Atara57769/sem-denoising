"""
Model registry and factory functions for SEM denoising architectures.
All models are instantiated using the single unified DnCNN class with customized parameters.
"""

from typing import Dict, Any, Callable
import torch
import torch.nn as nn

from sem_denoising.models.dncnn import DnCNN


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_REGISTRY: Dict[str, Callable[..., DnCNN]] = {
    # Direct 5-layer CNN (direct mapping, residual=False, no batch norm)
    "direct_cnn": lambda **kwargs: DnCNN(depth=5, num_channels=32, use_bn=False, act_type="leaky_relu", residual=False, **kwargs),
    # Residual 5-layer CNN (residual formulation, residual=True, no batch norm)
    "residual_cnn": lambda **kwargs: DnCNN(depth=5, num_channels=32, use_bn=False, act_type="leaky_relu", residual=True, **kwargs),
    # 5-layer Small DnCNN (residual=True, with batch norm)
    "small_dncnn": lambda **kwargs: DnCNN(depth=5, num_channels=32, use_bn=True, act_type="leaky_relu", residual=True, **kwargs),
    # 17-layer Strong DnCNN (residual=True, with batch norm)
    "strong_dncnn": lambda **kwargs: DnCNN(depth=17, num_channels=64, use_bn=True, act_type="relu", residual=True, **kwargs),
    # Default DnCNN instance
    "dncnn": lambda **kwargs: DnCNN(**kwargs),
}


def build_model(model_name: str = "dncnn", **kwargs) -> DnCNN:
    """
    Factory function to instantiate a model from the registry.
    Every model is created from the single DnCNN class with specific architecture parameters.

    Args:
        model_name: Registry key ('direct_cnn', 'residual_cnn', 'small_dncnn', 'strong_dncnn', etc.).
        kwargs: Additional model parameter overrides to pass to DnCNN.

    Returns:
        Instantiated DnCNN instance.
    """
    key = model_name.lower().replace("-", "_").replace(" ", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key](**kwargs)
