import torch
import numpy as np
import pytest
from sem_denoising.models import (
    DirectPredictionCNN,
    ResidualPredictionCNN,
    DnCNN,
    build_model,
    count_parameters,
    CLASSICAL_METHODS,
)


def test_classical_methods():
    noisy = np.random.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)
    for name, fn in CLASSICAL_METHODS.items():
        out, runtime = fn(noisy)
        assert out.shape == noisy.shape
        assert runtime >= 0.0
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_direct_cnn_forward():
    model = DirectPredictionCNN(depth=5, num_channels=32)
    inp = torch.rand(2, 1, 64, 64)
    out = model(inp)
    assert out.shape == (2, 1, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert count_parameters(model) == 28353


def test_residual_cnn_forward():
    model = ResidualPredictionCNN(depth=5, num_channels=32)
    inp = torch.rand(2, 1, 64, 64)
    out = model(inp)
    assert out.shape == (2, 1, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert count_parameters(model) == 28353


def test_dncnn_small_and_strong():
    small = DnCNN.create_small(depth=5, num_channels=32)
    strong = DnCNN.create_strong(depth=17, num_channels=64)

    inp = torch.rand(1, 1, 32, 32)
    out_s = small(inp)
    out_st = strong(inp)

    assert out_s.shape == (1, 1, 32, 32)
    assert out_st.shape == (1, 1, 32, 32)
    assert count_parameters(small) == 28449
    assert count_parameters(strong) == 556097


def test_model_registry_builder():
    m = build_model("small_dncnn")
    assert isinstance(m, DnCNN)
    assert m.depth == 5

    with pytest.raises(ValueError):
        build_model("unregistered_model")

