import numpy as np
import pytest
from sem_denoising.noise import (
    add_gaussian_noise,
    add_poisson_noise,
    add_mixed_noise,
    get_noise_fn,
    NOISE_REGIMES,
)


@pytest.fixture
def clean_image():
    np.random.seed(42)
    return np.random.uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)


def test_gaussian_noise(clean_image):
    noisy = add_gaussian_noise(clean_image, sigma=0.10)
    assert noisy.shape == clean_image.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert not np.array_equal(clean_image, noisy)


def test_poisson_noise(clean_image):
    noisy = add_poisson_noise(clean_image, peak=50.0)
    assert noisy.shape == clean_image.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert not np.array_equal(clean_image, noisy)


def test_mixed_noise(clean_image):
    noisy = add_mixed_noise(clean_image, sigma=0.06, peak=50.0)
    assert noisy.shape == clean_image.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert not np.array_equal(clean_image, noisy)


def test_get_noise_fn(clean_image):
    fn = get_noise_fn("gaussian", sigma=0.05)
    noisy = fn(clean_image)
    assert noisy.shape == clean_image.shape

    with pytest.raises(ValueError):
        get_noise_fn("unknown_regime")

