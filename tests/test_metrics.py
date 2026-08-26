import numpy as np
import pytest
from sem_denoising.metrics import compute_mse, compute_psnr, compute_ssim, evaluate_predictions, Timer


def test_metrics_identical_images():
    img = np.random.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)
    assert compute_mse(img, img) == 0.0
    assert compute_psnr(img, img) > 90.0 or np.isinf(compute_psnr(img, img))
    assert pytest.approx(compute_ssim(img, img), 1e-4) == 1.0


def test_evaluate_predictions():
    gt = np.ones((32, 32), dtype=np.float32) * 0.5
    pred = np.ones((32, 32), dtype=np.float32) * 0.6
    res = evaluate_predictions(gt, pred, runtime_ms=12.34)
    assert "mse" in res
    assert "psnr" in res
    assert "ssim" in res
    assert res["runtime_ms"] == 12.34
    assert res["mse"] > 0.0


def test_timer():
    with Timer() as t:
        _ = sum(range(10000))
    assert t.elapsed_ms >= 0.0

