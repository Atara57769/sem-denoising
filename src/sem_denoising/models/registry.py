"""
Model registry and factory functions for SEM denoising architectures.
All models are registered via ModelType enum and instantiated using the single unified DnCNN class.
"""

from enum import Enum
from typing import Dict, Any, Callable, Union
import torch
import torch.nn as nn

from sem_denoising.models.dncnn import DnCNN


class ModelType(str, Enum):
    """Enumeration of supported neural model architecture configurations."""
    DIRECT_CNN = "direct_cnn"
    RESIDUAL_CNN = "residual_cnn"
    SMALL_DNCNN = "small_dncnn"
    STRONG_DNCNN = "strong_dncnn"
    DNCNN = "dncnn"


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_REGISTRY: Dict[ModelType, Callable[..., DnCNN]] = {
    # Direct 5-layer CNN (direct mapping, residual=False, no batch norm)
    ModelType.DIRECT_CNN: lambda **kwargs: DnCNN(depth=5, num_channels=32, use_bn=False, act_type="leaky_relu", residual=False, **kwargs),
    # Residual 5-layer CNN (residual formulation, residual=True, no batch norm)
    ModelType.RESIDUAL_CNN: lambda **kwargs: DnCNN(depth=5, num_channels=32, use_bn=False, act_type="leaky_relu", residual=True, **kwargs),
    # 5-layer Small DnCNN (residual=True, with batch norm)
    ModelType.SMALL_DNCNN: lambda **kwargs: DnCNN(depth=5, num_channels=32, use_bn=True, act_type="leaky_relu", residual=True, **kwargs),
    # 17-layer Strong DnCNN (residual=True, with batch norm)
    ModelType.STRONG_DNCNN: lambda **kwargs: DnCNN(depth=17, num_channels=64, use_bn=True, act_type="relu", residual=True, **kwargs),
    # Default DnCNN instance
    ModelType.DNCNN: lambda **kwargs: DnCNN(**kwargs),
}


def build_model(model_name: Union[ModelType, str] = ModelType.DNCNN, **kwargs) -> DnCNN:
    """
    Factory function to instantiate a model from the registry.
    Every model is created from the single DnCNN class with specific architecture parameters.

    Args:
        model_name: ModelType enum or string representation ('direct_cnn', 'small_dncnn', etc.).
        kwargs: Additional model parameter overrides to pass to DnCNN.

    Returns:
        Instantiated DnCNN instance.
    """
    if isinstance(model_name, str):
        cleaned_key = model_name.lower().replace("-", "_").replace(" ", "_")
        try:
            model_type = ModelType(cleaned_key)
        except ValueError:
            raise ValueError(f"Unknown model name '{model_name}'. Available: {[m.value for m in ModelType]}")
    elif isinstance(model_name, ModelType):
        model_type = model_name
    else:
        raise ValueError(f"Invalid model_name type: {type(model_name)}")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_type} not found in MODEL_REGISTRY.")

    return MODEL_REGISTRY[model_type](**kwargs)
