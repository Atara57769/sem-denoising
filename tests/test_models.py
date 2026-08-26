import torch
import numpy as np
import pytest
from sem_denoising.models import (
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


def test_dncnn_forward_returns_residual_directly():
    model = DnCNN(depth=5, num_channels=32, use_bn=False, act_type="leaky_relu")
    inp = torch.rand(2, 1, 64, 64)
    residual = model(inp)
    assert residual.shape == (2, 1, 64, 64)

    # Clean estimation via img - model_output
    clean_est = torch.clamp(inp - residual, 0.0, 1.0)
    assert clean_est.shape == (2, 1, 64, 64)
    assert clean_est.min() >= 0.0 and clean_est.max() <= 1.0
    assert count_parameters(model) == 28353


def test_dncnn_small_and_strong_instantiation():
    small = DnCNN(depth=5, num_channels=32, use_bn=True, act_type="leaky_relu")
    strong = DnCNN(depth=17, num_channels=64, use_bn=True, act_type="relu")

    inp = torch.rand(1, 1, 32, 32)
    res_s = small(inp)
    res_st = strong(inp)

    assert res_s.shape == (1, 1, 32, 32)
    assert res_st.shape == (1, 1, 32, 32)
    assert count_parameters(small) == 28449
    assert count_parameters(strong) == 556097


def test_model_registry_builder():
    m_direct = build_model("direct_cnn")
    assert isinstance(m_direct, DnCNN)
    assert m_direct.depth == 5
    assert m_direct.num_channels == 32
    assert m_direct.use_bn is False
    assert m_direct.residual is False

    m_small = build_model("small_dncnn")
    assert isinstance(m_small, DnCNN)
    assert m_small.depth == 5
    assert m_small.num_channels == 32
    assert m_small.use_bn is True
    assert m_small.residual is True

    m_strong = build_model("strong_dncnn")
    assert isinstance(m_strong, DnCNN)
    assert m_strong.depth == 17
    assert m_strong.num_channels == 64
    assert m_strong.use_bn is True
    assert m_strong.residual is True

    with pytest.raises(ValueError):
        build_model("unregistered_model")
