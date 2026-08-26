"""
Patch extraction utilities for SEM images.
"""

from typing import List, Tuple
import numpy as np


def extract_patches(img: np.ndarray, patch_size: int = 64, stride: int = 32) -> np.ndarray:
    """
    Extract overlapping 2D spatial patches from a 2D image array.

    Args:
        img: 2D numpy array of shape (H, W).
        patch_size: Square patch dimension.
        stride: Step size between consecutive patch origins.

    Returns:
        3D numpy array of shape (N, patch_size, patch_size).
    """
    h, w = img.shape
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image shape ({h}, {w}) is smaller than patch size ({patch_size}, {patch_size})")

    patches: List[np.ndarray] = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y : y + patch_size, x : x + patch_size]
            patches.append(patch)

    if not patches:
        raise ValueError("No patches could be extracted with the given parameters.")

    return np.array(patches, dtype=np.float32)

