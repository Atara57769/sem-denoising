"""
Unit tests for Real-SEM validation infrastructure and metric permission logic.
"""

import os
import tempfile
import numpy as np
import pytest

from sem_denoising.real_sem.manifest import EvidenceMode
from sem_denoising.real_sem.registration import register_repeated_frames, build_consensus_reference
from sem_denoising.real_sem.metrics import evaluate_real_sem_field


def test_repeated_frame_registration():
    np.random.seed(42)
    base = np.random.rand(64, 64).astype(np.float32)
    f1 = base.copy()
    f2 = np.roll(base, shift=2, axis=0)  # Shifted frame

    aligned = register_repeated_frames([f1, f2])
    assert len(aligned) == 2
    assert aligned[0].shape == (64, 64)

    consensus_mean, var_map = build_consensus_reference(aligned)
    assert consensus_mean.shape == (64, 64)
    assert var_map.shape == (64, 64)


def test_unpaired_metric_permission_guardrail():
    noisy = np.random.rand(64, 64).astype(np.float32)
    denoised = np.clip(noisy * 0.9, 0.0, 1.0)

    # UNPAIRED_ONLY must strictly block PSNR/SSIM
    results = evaluate_real_sem_field(
        denoised_img=denoised,
        noisy_img=noisy,
        evidence_mode=EvidenceMode.UNPAIRED_ONLY,
    )

    assert results["psnr_permitted"] is False
    assert results["ssim_permitted"] is False
    assert results["psnr"] is None
    assert results["ssim"] is None
    assert "FORBIDDEN" in results["permission_notice"]
    assert results["enl"] > 0.0
    assert results["cnr"] > 0.0


def test_paired_metric_permission():
    gt = np.random.rand(64, 64).astype(np.float32)
    denoised = np.clip(gt + np.random.normal(0, 0.05, (64, 64)), 0.0, 1.0).astype(np.float32)

    results = evaluate_real_sem_field(
        denoised_img=denoised,
        noisy_img=gt,
        evidence_mode=EvidenceMode.PAIRED_REFERENCE,
        reference_img=gt,
    )

    assert results["psnr_permitted"] is True
    assert results["ssim_permitted"] is True
    assert results["psnr"] > 20.0
    assert results["ssim"] > 0.5
    assert "PERMITTED" in results["permission_notice"]
