"""
Image loading and normalization routines for NIST SEM dataset.
"""

import os
from typing import Optional
import cv2
import numpy as np


def get_clean_reference_path(set_id: int, data_root: str) -> str:
    """Return the absolute path to the clean reference image for a given set ID."""
    primary_path = os.path.join(data_root, "mask_sets", "masks", f"set{set_id}_cex_noise_000_contrast_100.tiff")
    if os.path.exists(primary_path):
        return primary_path

    alt_path = os.path.join(data_root, "mask_sets", f"set{set_id}_cex_noise_000_contrast_100.tiff")
    if os.path.exists(alt_path):
        return alt_path

    raise FileNotFoundError(f"Clean reference image for set {set_id} not found in {data_root}")


def read_image_grayscale(filepath: str) -> np.ndarray:
    """Read an image from disk and return as a 2D float32 grayscale numpy array."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image file: {filepath}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img.astype(np.float32)


def normalize_minmax(img: np.ndarray) -> np.ndarray:
    """Normalize 2D image intensities to the range [0.0, 1.0]."""
    min_val = float(img.min())
    max_val = float(img.max())
    if max_val > min_val:
        return (img - min_val) / (max_val - min_val)
    return np.zeros_like(img, dtype=np.float32)


def load_image(filepath: str) -> np.ndarray:
    """Load and normalize a grayscale image from disk."""
    raw = read_image_grayscale(filepath)
    return normalize_minmax(raw)
