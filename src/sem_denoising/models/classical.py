"""
Classical (non-learned) image processing denoising baselines.
Includes Identity, Gaussian Blur, Non-Local Means (NLM), and BayesShrink Wavelet denoising.
"""

from typing import Tuple
import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, denoise_wavelet, estimate_sigma

from sem_denoising.metrics import Timer


def denoise_identity(noisy_img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Baseline 1: No-op Identity return."""
    with Timer() as timer:
        out = noisy_img.copy()
    return out.astype(np.float32), timer.elapsed_ms


def denoise_gaussian_filter(
    noisy_img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Baseline 2: Spatial Gaussian Blur Filter."""
    with Timer() as timer:
        out = cv2.GaussianBlur(noisy_img, (kernel_size, kernel_size), sigmaX=sigma)
    return np.clip(out, 0.0, 1.0).astype(np.float32), timer.elapsed_ms


def denoise_nlm(
    noisy_img: np.ndarray,
    h_factor: float = 0.8,
    patch_size: int = 5,
    patch_distance: int = 9,
) -> Tuple[np.ndarray, float]:
    """Baseline 3: Non-Local Means (NLM) CPU Denoising."""
    with Timer() as timer:
        sigma_est = float(np.mean(estimate_sigma(noisy_img)))
        h = h_factor * sigma_est
        out = denoise_nl_means(
            noisy_img,
            h=h if h > 0 else 0.1,
            sigma=sigma_est,
            fast_mode=True,
            patch_size=patch_size,
            patch_distance=patch_distance,
            channel_axis=None,
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32), timer.elapsed_ms


def denoise_wavelet_baseline(
    noisy_img: np.ndarray,
    method: str = "BayesShrink",
    mode: str = "soft",
) -> Tuple[np.ndarray, float]:
    """Baseline 4: Wavelet Denoising (BayesShrink soft thresholding)."""
    with Timer() as timer:
        out = denoise_wavelet(
            noisy_img,
            method=method,
            mode=mode,
            rescale_sigma=True,
            channel_axis=None,
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32), timer.elapsed_ms


CLASSICAL_METHODS = {
    "Identity": denoise_identity,
    "Gaussian Filter": denoise_gaussian_filter,
    "Non-Local Means": denoise_nlm,
    "Wavelet (BayesShrink)": denoise_wavelet_baseline,
}

