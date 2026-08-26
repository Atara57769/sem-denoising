import os
import numpy as np
import pytest
import cv2
from sem_denoising.data.loader import normalize_minmax, read_image_grayscale
from sem_denoising.data.patch import extract_patches


def test_normalize_minmax():
    arr = np.array([[10.0, 20.0], [30.0, 110.0]], dtype=np.float32)
    norm = normalize_minmax(arr)
    assert norm.min() == 0.0
    assert norm.max() == 1.0
    assert norm[0, 0] == 0.0
    assert norm[1, 1] == 1.0


def test_normalize_minmax_flat():
    flat = np.ones((10, 10), dtype=np.float32) * 5.0
    norm = normalize_minmax(flat)
    assert np.all(norm == 0.0)


def test_extract_patches():
    img = np.zeros((128, 128), dtype=np.float32)
    patches = extract_patches(img, patch_size=64, stride=32)
    # (128 - 64) // 32 + 1 = 3 rows, 3 cols -> 9 patches
    assert patches.shape == (9, 64, 64)


def test_extract_patches_invalid_size():
    img = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        extract_patches(img, patch_size=64, stride=32)

