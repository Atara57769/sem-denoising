import numpy as np
import pytest
from sem_denoising.models import build_model
from sem_denoising.noise import add_gaussian_noise
from sem_denoising.training.sanity_check import run_overfit_sanity_check


def test_overfit_sanity_check():
    model = build_model("small_dncnn")
    patch = np.random.uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)

    init_loss, final_loss, history = run_overfit_sanity_check(
        model=model,
        clean_patch=patch,
        noise_fn=lambda x: add_gaussian_noise(x, sigma=0.10),
        epochs=30,
        lr=1e-3,
    )

    assert len(history) == 30
    assert final_loss < init_loss
    assert final_loss < 0.05

