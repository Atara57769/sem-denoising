"""
Dedicated Command-Line Interface (CLI) for Training SEM Denoising Models.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from sem_denoising.config import PipelineConfig
from sem_denoising.noise import add_gaussian_noise, add_mixed_noise
from sem_denoising.data import (
    SEMPatchDataset,
    get_clean_reference_path,
    load_image,
)
from sem_denoising.models import build_model
from sem_denoising.training import (
    train_model,
    save_checkpoint,
    load_checkpoint,
    verify_reproducibility,
)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for training."""
    parser = argparse.ArgumentParser(
        description="SEM Denoising Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file",
    )
    return parser


def run_training(config: PipelineConfig):
    """Train all learned models, save checkpoints, and assert reproducibility using configuration settings."""
    print("=== EXECUTING FULL DATASET TRAINING ===")
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    device = config.training.device
    epochs = config.training.epochs
    lr = config.training.lr
    batch_size = config.training.batch_size

    # 1. Prepare Datasets & Loaders
    g_sigma = config.noise.gaussian.get("sigma", 0.10)
    m_sigma = config.noise.mixed.get("sigma", 0.06)
    m_peak = config.noise.mixed.get("peak", 50.0)

    ds_train_gauss = SEMPatchDataset(
        config.data.train_sets,
        patch_size=config.data.patch_size,
        stride=config.data.stride,
        corruption_fn=lambda x: add_gaussian_noise(x, sigma=g_sigma),
        data_root=config.data.data_root,
    )
    ds_val_gauss = SEMPatchDataset(
        config.data.val_sets,
        patch_size=config.data.patch_size,
        stride=config.data.stride,
        corruption_fn=lambda x: add_gaussian_noise(x, sigma=g_sigma),
        data_root=config.data.data_root,
    )
    loader_train_gauss = DataLoader(ds_train_gauss, batch_size=batch_size, shuffle=True)
    loader_val_gauss = DataLoader(ds_val_gauss, batch_size=batch_size, shuffle=False)

    ds_train_mixed = SEMPatchDataset(
        config.data.train_sets,
        patch_size=config.data.patch_size,
        stride=config.data.stride,
        corruption_fn=lambda x: add_mixed_noise(x, sigma=m_sigma, peak=m_peak),
        data_root=config.data.data_root,
    )
    ds_val_mixed = SEMPatchDataset(
        config.data.val_sets,
        patch_size=config.data.patch_size,
        stride=config.data.stride,
        corruption_fn=lambda x: add_mixed_noise(x, sigma=m_sigma, peak=m_peak),
        data_root=config.data.data_root,
    )
    loader_train_mixed = DataLoader(ds_train_mixed, batch_size=batch_size, shuffle=True)
    loader_val_mixed = DataLoader(ds_val_mixed, batch_size=batch_size, shuffle=False)

    ckpt_dir = config.training.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    train_plan = [
        ("Direct CNN (Gaussian)", "direct_cnn", loader_train_gauss, loader_val_gauss, "checkpoint_direct_cnn.pth"),
        ("Residual CNN (Gaussian)", "residual_cnn", loader_train_gauss, loader_val_gauss, "checkpoint_residual_cnn.pth"),
        ("Small DnCNN (Gaussian)", "small_dncnn", loader_train_gauss, loader_val_gauss, "checkpoint_small_dncnn.pth"),
        ("Strong DnCNN (Gaussian)", "strong_dncnn", loader_train_gauss, loader_val_gauss, "checkpoint_strong_dncnn_gaussian.pth"),
        ("Strong DnCNN (Mixed)", "strong_dncnn", loader_train_mixed, loader_val_mixed, "checkpoint_strong_dncnn_mixed.pth"),
    ]

    # Sample reference image for reproducibility check
    ref_path = get_clean_reference_path(config.data.train_sets[0], data_root=config.data.data_root)
    ref_img = load_image(ref_path)
    noisy_sample = add_gaussian_noise(ref_img, sigma=g_sigma)

    for name, model_key, t_loader, v_loader, ckpt_name in train_plan:
        model = build_model(model_key)
        train_model(
            model=model,
            train_loader=t_loader,
            val_loader=v_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            model_name=name,
        )

        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        size_kb = save_checkpoint(model, ckpt_path)
        print(f"Saved: {ckpt_path} ({size_kb} KB)")

        # Verify Delta = 0.0 reproducibility
        reloaded = load_checkpoint(lambda: build_model(model_key), ckpt_path, device=device)
        max_diff = verify_reproducibility(model, reloaded, noisy_sample, device=device)
        print(f"  Reproducibility delta verified: {max_diff:.10f}")

    print("\nAll models trained and checkpoints verified successfully.")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = PipelineConfig.from_yaml(args.config)
    else:
        print(f"Notice: Config file '{args.config}' not found, using default configuration.")
        config = PipelineConfig()

    run_training(config=config)


if __name__ == "__main__":
    main()
