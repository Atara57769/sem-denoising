import os
import numpy as np
import torch
import pytest
from sem_denoising.models import build_model
from sem_denoising.training.checkpoint import save_checkpoint, load_checkpoint, verify_reproducibility


def test_checkpoint_save_and_reload(tmp_path):
    model = build_model("small_dncnn")
    ckpt_path = str(tmp_path / "test_model.pth")

    size_kb = save_checkpoint(model, ckpt_path)
    assert os.path.exists(ckpt_path)
    assert size_kb > 0.0

    reloaded = load_checkpoint(lambda: build_model("small_dncnn"), ckpt_path)

    sample_img = np.random.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)
    max_diff = verify_reproducibility(model, reloaded, sample_img)
    assert max_diff < 1e-6
