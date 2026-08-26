"""
Checkpoint management and reproducibility validation for SEM denoising models.
"""

import os
from typing import Type, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, filepath: str) -> float:
    """
    Save model state dict to disk and return file size in KB.

    Args:
        model: PyTorch model.
        filepath: Target checkpoint path.

    Returns:
        Checkpoint file size in KB.
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    torch.save(model.state_dict(), filepath)
    file_size_kb = os.path.getsize(filepath) / 1024.0
    return round(file_size_kb, 2)


def load_checkpoint(
    model_or_class,
    filepath: str,
    device: str = "cpu",
    **kwargs,
) -> nn.Module:
    """
    Load weights from checkpoint filepath into either an existing model instance
    or instantiate a new model of model_or_class.

    Args:
        model_or_class: nn.Module instance or class.
        filepath: Checkpoint file path.
        device: Device to load state dict to.
        kwargs: Keyword arguments for instantiation if class passed.

    Returns:
        Loaded model in eval mode.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if isinstance(model_or_class, nn.Module):
        instance = model_or_class
    elif isinstance(model_or_class, type) and issubclass(model_or_class, nn.Module):
        instance = model_or_class(**kwargs)
    elif callable(model_or_class):
        instance = model_or_class(**kwargs)
    else:
        raise ValueError(f"Invalid model_or_class: {model_or_class}")

    state_dict = torch.load(filepath, map_location=device)
    instance.load_state_dict(state_dict)
    instance.to(device)
    instance.eval()
    return instance


def verify_reproducibility(
    orig_model: nn.Module,
    reloaded_model: nn.Module,
    sample_img: np.ndarray,
    device: str = "cpu",
    tolerance: float = 1e-6,
) -> float:
    """
    Assert that reloaded model produces outputs numerically identical to original model.

    Args:
        orig_model: Original PyTorch model.
        reloaded_model: Reloaded PyTorch model from disk checkpoint.
        sample_img: 2D numpy array test image.
        device: 'cpu' or 'cuda'.
        tolerance: Maximum acceptable absolute difference.

    Returns:
        Max absolute difference float.
    """
    orig_model = orig_model.to(device)
    reloaded_model = reloaded_model.to(device)
    orig_model.eval()
    reloaded_model.eval()

    inp = torch.from_numpy(sample_img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out_orig = orig_model(inp)
        out_reload = reloaded_model(inp)

    max_diff = float(torch.max(torch.abs(out_orig - out_reload)).item())
    assert (
        max_diff <= tolerance
    ), f"Reproducibility verification failed! Max difference: {max_diff:.10f} > tolerance: {tolerance}"

    return max_diff

