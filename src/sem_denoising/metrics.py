"""
Evaluation metrics suite for image denoising (MSE, PSNR, SSIM, and runtime profiling).
"""

import time
from typing import Dict, Any, Optional
import numpy as np
from skimage.metrics import mean_squared_error as mse_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func


def compute_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE) between ground truth and prediction."""
    return float(mse_func(gt, pred))


def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
    return float(psnr_func(gt, pred, data_range=data_range))


def compute_ssim(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Structural Similarity Index Measure (SSIM)."""
    return float(ssim_func(gt, pred, data_range=data_range))


def evaluate_predictions(
    gt: np.ndarray,
    pred: np.ndarray,
    runtime_ms: float = 0.0,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """
    Unified evaluator calculating MSE, PSNR, SSIM, and runtime in milliseconds.

    Args:
        gt: 2D ground truth clean image (values in [0.0, 1.0]).
        pred: 2D predicted / denoised image (values in [0.0, 1.0]).
        runtime_ms: Execution duration in milliseconds.
        data_range: Dynamic range of input images (default 1.0).

    Returns:
        Dictionary with keys 'mse', 'psnr', 'ssim', 'runtime_ms'.
    """
    mse_val = compute_mse(gt, pred)
    psnr_val = compute_psnr(gt, pred, data_range=data_range)
    ssim_val = compute_ssim(gt, pred, data_range=data_range)

    return {
        "mse": round(mse_val, 6),
        "psnr": round(psnr_val, 4),
        "ssim": round(ssim_val, 4),
        "runtime_ms": round(runtime_ms, 2),
    }


class Timer:
    """Context manager and utility to accurately measure execution time in milliseconds."""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0

