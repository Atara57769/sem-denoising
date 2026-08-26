"""
Noise degradation models for SEM image synthesis.
Includes Gaussian thermal noise, Poisson shot noise, and mixed Poisson-Gaussian noise.
"""

from enum import Enum
from typing import Callable, Dict, Any, Union
import numpy as np


class NoiseRegime(str, Enum):
    """Supported synthetic noise degradation regimes."""

    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    MIXED = "mixed"


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


NOISE_REGIMES: Dict[NoiseRegime, Callable[..., np.ndarray]] = {
    NoiseRegime.GAUSSIAN: add_gaussian_noise,
    NoiseRegime.POISSON: add_poisson_noise,
    NoiseRegime.MIXED: add_mixed_noise,
}


def get_noise_fn(regime: Union[str, NoiseRegime], **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory function to get a noise degradation callable with preset parameters.

    Args:
        regime: NoiseRegime enum member or string name (e.g. 'gaussian', 'poisson', 'mixed').
        **kwargs: Parameters forwarded to the noise function (e.g. sigma, peak).

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
