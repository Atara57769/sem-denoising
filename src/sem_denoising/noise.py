"""
Noise degradation models for SEM image synthesis.
Includes Gaussian thermal noise, Poisson shot noise, and mixed Poisson-Gaussian noise.
"""

from typing import Callable, Dict, Any
import numpy as np


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


NOISE_REGIMES: Dict[str, Callable[..., np.ndarray]] = {
    "gaussian": add_gaussian_noise,
    "poisson": add_poisson_noise,
    "mixed": add_mixed_noise,
}


def get_noise_fn(regime: str, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """Factory function to get a noise degradation callable with preset parameters."""
    regime_key = regime.lower()
    if regime_key not in NOISE_REGIMES:
        raise ValueError(f"Unknown noise regime '{regime}'. Available: {list(NOISE_REGIMES.keys())}")
    
    base_fn = NOISE_REGIMES[regime_key]
    return lambda img: base_fn(img, **kwargs)
