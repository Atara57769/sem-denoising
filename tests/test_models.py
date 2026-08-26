import torch
import numpy as np
import pytest
from sem_denoising.models import (
    DnCNN,
    ModelType,
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
    model = DnCNN(depth=5, num_channels=32, use_bn=False, act_type="leaky_relu", residual=True)
    inp = torch.rand(2, 1, 64, 64)
    residual = model(inp)
    assert residual.shape == (2, 1, 64, 64)

    # Clean estimation via img - model_output
    clean_est = torch.clamp(inp - residual, 0.0, 1.0)
    assert clean_est.shape == (2, 1, 64, 64)
    assert clean_est.min() >= 0.0 and clean_est.max() <= 1.0
    assert count_parameters(model) == 28353


def test_dncnn_small_and_strong_instantiation():
    small = DnCNN(depth=5, num_channels=32, use_bn=True, act_type="leaky_relu", residual=True)
    strong = DnCNN(depth=17, num_channels=64, use_bn=True, act_type="relu", residual=True)

    inp = torch.rand(1, 1, 32, 32)
    res_s = small(inp)
    res_st = strong(inp)

    assert res_s.shape == (1, 1, 32, 32)
    assert res_st.shape == (1, 1, 32, 32)
    assert count_parameters(small) == 28449
    assert count_parameters(strong) == 556097


def test_dncnn_with_skip_connection():
    model = DnCNN(depth=17, num_channels=64, use_bn=True, act_type="relu", residual=True, use_skip=True)
    assert model.use_skip is True
    inp = torch.rand(2, 1, 32, 32)
    out = model(inp)
    assert out.shape == (2, 1, 32, 32)
    assert count_parameters(model) == 556097

    m_skip = build_model(ModelType.SKIP_DNCNN)
    assert isinstance(m_skip, DnCNN)
    assert m_skip.use_skip is True


def test_model_registry_builder():
    # Build using ModelType Enum
    m_direct = build_model(ModelType.DIRECT_CNN)
    assert isinstance(m_direct, DnCNN)
    assert m_direct.depth == 5
    assert m_direct.num_channels == 32
    assert m_direct.use_bn is False
    assert m_direct.residual is False

    # Build using string key
    m_small = build_model("small_dncnn")
    assert isinstance(m_small, DnCNN)
    assert m_small.depth == 5
    assert m_small.num_channels == 32
    assert m_small.use_bn is True
    assert m_small.residual is True

    # Build using ModelType Enum
    m_strong = build_model(ModelType.STRONG_DNCNN)
    assert isinstance(m_strong, DnCNN)
    assert m_strong.depth == 17
    assert m_strong.num_channels == 64
    assert m_strong.use_bn is True
    assert m_strong.residual is True

    with pytest.raises(ValueError):
        build_model("unregistered_model")
