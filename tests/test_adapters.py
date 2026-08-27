"""
Unit tests for model adapters (TinyDnCNN D8/W48 and AMAT-1 ConvRefinedAddUNet-L).
"""

import os
import numpy as np
import torch
import pytest

from sem_denoising.adapters import DnCNNAdapter, AMAT1Adapter


def test_dncnn_adapter_fp32():
    adapter = DnCNNAdapter(model_name="tiny_dncnn_d8w48", checkpoint_path=None, precision="fp32")
    meta = adapter.get_metadata()

    assert meta.parameters == 125905
    assert meta.precision == "fp32"
    assert "TinyDnCNN" in meta.architecture

    dummy_img = np.random.rand(64, 64).astype(np.float32)
    denoised, runtime_ms = adapter.infer(dummy_img, precision="fp32")

    assert denoised.shape == (64, 64)
    assert denoised.dtype == np.float32
    assert 0.0 <= denoised.min() and denoised.max() <= 1.0
    assert runtime_ms > 0.0


def test_dncnn_adapter_fp16():
    adapter = DnCNNAdapter(model_name="tiny_dncnn_d8w48", checkpoint_path=None, precision="fp16")
    meta = adapter.get_metadata()

    assert meta.precision == "fp16"

    dummy_img = np.random.rand(128, 128).astype(np.float32)
    denoised, runtime_ms = adapter.infer(dummy_img, precision="fp16")

    assert denoised.shape == (128, 128)
    assert denoised.dtype == np.float32
    assert 0.0 <= denoised.min() and denoised.max() <= 1.0
    assert runtime_ms > 0.0


def test_model_name_enum_and_factory():
    from sem_denoising.adapters import ModelName, get_adapter, ClassicalAdapter

    assert ModelName.NLM == "nlm"
    assert ModelName.TINY_DNCNN_FP16 == "tiny_dncnn_d8w48_fp16"

    adapter = get_adapter(ModelName.NLM)
    assert isinstance(adapter, ClassicalAdapter)
    assert adapter.get_metadata().model_id == "nlm_classical"

    dummy_img = np.random.rand(32, 32).astype(np.float32)
    denoised, runtime_ms = adapter.infer(dummy_img)
    assert denoised.shape == (32, 32)
    assert runtime_ms >= 0.0
