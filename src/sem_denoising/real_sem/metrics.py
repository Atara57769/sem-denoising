"""
Real-SEM Metric Permission Logic and Non-Reference Quality Indicators.

ENFORCES SCIENTIFIC GUARDRAIL:
If evidence_mode is UNPAIRED_ONLY, PSNR/SSIM evaluation claims are strictly FORBIDDEN.
Non-reference metrics (Equivalent Number of Looks, Contrast-to-Noise Ratio, Edge Sharpness)
are calculated instead.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from scipy.ndimage import sobel

from sem_denoising.real_sem.manifest import EvidenceMode
from sem_denoising.metrics import compute_psnr, compute_ssim, compute_mse, compute_sobel_mae


def compute_enl(img: np.ndarray) -> float:
    """Compute Equivalent Number of Looks (ENL = mean^2 / var)."""
    mean_val = float(np.mean(img))
    var_val = float(np.var(img))
    if var_val > 1e-12:
        return (mean_val ** 2) / var_val
    return 0.0


def compute_cnr(img: np.ndarray) -> float:
    """Compute Contrast-to-Noise Ratio (CNR) between upper and lower intensity percentiles."""
    p90 = np.percentile(img, 90)
    p10 = np.percentile(img, 10)
    noise_std = float(np.std(img))
    if noise_std > 1e-12:
        return float((p90 - p10) / noise_std)
    return 0.0


def compute_edge_sharpness(img: np.ndarray) -> float:
    """Compute mean Sobel gradient magnitude as an indicator of edge sharpness."""
    gx = sobel(img, axis=0)
    gy = sobel(img, axis=1)
    mag = np.hypot(gx, gy)
    return float(np.mean(mag))


def evaluate_real_sem_field(
    denoised_img: np.ndarray,
    noisy_img: np.ndarray,
    evidence_mode: EvidenceMode,
    reference_img: Optional[np.ndarray] = None,
    runtime_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate real-SEM denoised field respecting strict metric permission rules.

    Args:
        denoised_img: Model output image [H, W] float32.
        noisy_img: Original input SEM image [H, W] float32.
        evidence_mode: PAIRED_REFERENCE, REPEATED_FRAME, or UNPAIRED_ONLY.
        reference_img: Ground truth clean or consensus reference image (if available).
        runtime_ms: Model latency in ms.

    Returns:
        Dictionary containing permitted metrics and explicit permission flags.
    """
    results: Dict[str, Any] = {
        "evidence_mode": evidence_mode.value,
        "runtime_ms": round(runtime_ms, 2),
        "psnr_permitted": False,
        "ssim_permitted": False,
        "psnr": None,
        "ssim": None,
        "mse": None,
        "sobel_mae": None,
        "enl": round(compute_enl(denoised_img), 4),
        "cnr": round(compute_cnr(denoised_img), 4),
        "edge_sharpness": round(compute_edge_sharpness(denoised_img), 4),
        "noise_variance_reduction_pct": round(float((1.0 - (np.var(denoised_img) / (np.var(noisy_img) + 1e-12))) * 100.0), 2),
    }

    # Strict Metric Permission Logic
    if evidence_mode == EvidenceMode.UNPAIRED_ONLY:
        results["permission_notice"] = (
            "FORBIDDEN: PSNR/SSIM evaluation is prohibited on unpaired real-SEM images without ground truth reference. "
            "Showing non-reference indicators (ENL, CNR, Edge Sharpness) instead."
        )
        return results

    if evidence_mode in (EvidenceMode.PAIRED_REFERENCE, EvidenceMode.REPEATED_FRAME):
        if reference_img is None:
            raise ValueError(f"reference_img is required for evidence mode '{evidence_mode.value}'")

        results["psnr_permitted"] = True
        results["ssim_permitted"] = True
        results["psnr"] = round(compute_psnr(reference_img, denoised_img), 4)
        results["ssim"] = round(compute_ssim(reference_img, denoised_img), 4)
        results["mse"] = round(compute_mse(reference_img, denoised_img), 6)
        results["sobel_mae"] = round(compute_sobel_mae(reference_img, denoised_img), 6)
        results["permission_notice"] = f"PERMITTED: Reference-based metrics computed using {evidence_mode.value} reference."

    return results
