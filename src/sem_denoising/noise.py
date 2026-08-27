"""
Noise degradation models for SEM image synthesis.
Includes Gaussian thermal noise, Poisson shot noise, and mixed Poisson-Gaussian noise.
"""

from enum import Enum
from typing import Callable, Dict, Any, Union
import numpy as np


class NoiseRegime(str, Enum):
    """Supported synthetic noise degradation regimes and correlated stress conditions."""

    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    MIXED = "mixed"
    STRIPING = "striping"
    BLUR = "blur"
    SCAN_DRIFT = "scan_drift"
    MIXED_CORRELATED = "mixed_correlated"


def add_gaussian_noise(img: np.ndarray, sigma: float = 0.10) -> np.ndarray:
    """Add zero-mean Gaussian readout noise with standard deviation sigma."""
    noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def add_poisson_noise(img: np.ndarray, peak: float = 50.0) -> np.ndarray:
    """Add Poisson quantum shot noise based on electron emission scale."""
    scaled = np.maximum(img, 0.0) * peak
    noisy_scaled = np.random.poisson(scaled).astype(np.float32)
    return np.clip(noisy_scaled / peak, 0.0, 1.0)


def add_mixed_noise(img: np.ndarray, sigma: float = 0.06, peak: float = 50.0) -> np.ndarray:
    """Add combined Poisson electron shot noise and Gaussian readout noise."""
    p_noisy = add_poisson_noise(img, peak=peak)
    return add_gaussian_noise(p_noisy, sigma=sigma)


def add_striping_noise(img: np.ndarray, intensity: float = 0.08, line_frequency: float = 0.15) -> np.ndarray:
    """Add horizontal/vertical detector line striping artifacts."""
    h, w = img.shape
    num_striped_rows = int(h * line_frequency)
    striped_indices = np.random.choice(h, size=num_striped_rows, replace=False)
    offsets = np.random.normal(0.0, intensity, size=(num_striped_rows, 1)).astype(np.float32)

    corrupted = img.copy()
    corrupted[striped_indices, :] += offsets
    return np.clip(corrupted, 0.0, 1.0)


def add_blur_degradation(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Add spatial Gaussian beam blur (defocus)."""
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img, sigma=sigma)
    return np.clip(blurred, 0.0, 1.0)


def add_scan_drift_degradation(img: np.ndarray, amplitude: float = 4.0, frequency: float = 0.05) -> np.ndarray:
    """Add scan line shear/drift jitter simulating stage vibration."""
    h, w = img.shape
    corrupted = np.zeros_like(img)
    rows = np.arange(h)
    shifts = (amplitude * np.sin(2 * np.pi * frequency * rows)).astype(np.int32)

    for i in range(h):
        corrupted[i] = np.roll(img[i], shifts[i])
    return np.clip(corrupted, 0.0, 1.0)


def add_mixed_correlated_degradation(
    img: np.ndarray,
    noise_sigma: float = 0.05,
    striping_intensity: float = 0.05,
    drift_amplitude: float = 2.0,
) -> np.ndarray:
    """Add combined Poisson-Gaussian noise, line striping, and scan drift."""
    c1 = add_mixed_noise(img, sigma=noise_sigma, peak=50.0)
    c2 = add_striping_noise(c1, intensity=striping_intensity)
    c3 = add_scan_drift_degradation(c2, amplitude=drift_amplitude)
    return c3


NOISE_REGIMES: Dict[NoiseRegime, Callable[..., np.ndarray]] = {
    NoiseRegime.GAUSSIAN: add_gaussian_noise,
    NoiseRegime.POISSON: add_poisson_noise,
    NoiseRegime.MIXED: add_mixed_noise,
    NoiseRegime.STRIPING: add_striping_noise,
    NoiseRegime.BLUR: add_blur_degradation,
    NoiseRegime.SCAN_DRIFT: add_scan_drift_degradation,
    NoiseRegime.MIXED_CORRELATED: add_mixed_correlated_degradation,
}


def get_noise_fn(regime: Union[str, NoiseRegime], **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory function to get a noise degradation callable with preset parameters.

    Args:
        regime: NoiseRegime enum member or string name (e.g. 'gaussian', 'striping', 'scan_drift').
        **kwargs: Parameters forwarded to the noise function (e.g. sigma, peak, amplitude).

    Returns:
        Callable taking a 2D numpy image array and returning the corrupted image.
    """
    if isinstance(regime, str):
        try:
            regime = NoiseRegime(regime.lower())
        except ValueError:
            valid = [r.value for r in NoiseRegime]
            raise ValueError(f"Unknown noise regime '{regime}'. Available: {valid}")
    elif not isinstance(regime, NoiseRegime):
        valid = [r.value for r in NoiseRegime]
        raise ValueError(f"Invalid regime type: {type(regime)}. Available: {valid}")

    base_fn = NOISE_REGIMES[regime]
    return lambda img: base_fn(img, **kwargs)
