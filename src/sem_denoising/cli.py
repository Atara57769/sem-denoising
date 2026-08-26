"""
Unified Command-Line Interface (CLI) for the SEM Denoising Pipeline.
"""

import os
import sys
import argparse
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from sem_denoising.config import PipelineConfig
from sem_denoising.noise import add_gaussian_noise, add_mixed_noise, get_noise_fn
from sem_denoising.data import (
    SEMPatchDataset,
    get_clean_reference_path,
    load_image,
    extract_patches,
)

from sem_denoising.models import build_model, count_parameters
from sem_denoising.training import (
    train_model,
    save_checkpoint,
    load_checkpoint,
    verify_reproducibility,
)
from sem_denoising.experiments.benchmark import BenchmarkRunner, run_neural_inference
from sem_denoising.experiments.visualizer import plot_benchmark_results


def cmd_train(args: argparse.Namespace, config: PipelineConfig):
    """Train all learned models, save checkpoints, and assert reproducibility."""
    print("=== EXECUTING FULL DATASET TRAINING ===")
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

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
    loader_train_gauss = DataLoader(ds_train_gauss, batch_size=config.training.batch_size, shuffle=True)
    loader_val_gauss = DataLoader(ds_val_gauss, batch_size=config.training.batch_size, shuffle=False)

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
    loader_train_mixed = DataLoader(ds_train_mixed, batch_size=config.training.batch_size, shuffle=True)
    loader_val_mixed = DataLoader(ds_val_mixed, batch_size=config.training.batch_size, shuffle=False)

    ckpt_dir = config.training.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    epochs = args.epochs or config.training.epochs

    train_plan = [
        ("Direct CNN (Gaussian)", "direct_cnn", loader_train_gauss, loader_val_gauss, "checkpoint_direct_cnn.pth"),
        ("Residual CNN (Gaussian)", "residual_cnn", loader_train_gauss, loader_val_gauss, "checkpoint_residual_cnn.pth"),
        ("Small DnCNN (Gaussian)", "small_dncnn", loader_train_gauss, loader_val_gauss, "checkpoint_small_dncnn.pth"),
        ("Strong DnCNN (Gaussian)", "strong_dncnn", loader_train_gauss, loader_val_gauss, "checkpoint_strong_dncnn_gaussian.pth"),
        ("Strong DnCNN (Mixed)", "strong_dncnn", loader_train_mixed, loader_val_mixed, "checkpoint_strong_dncnn_mixed.pth"),
    ]

    # Sample reference image for reproducibility check
    ref_img = load_image(get_clean_reference_path(config.data.train_sets[0], data_root=config.data.data_root))
    noisy_sample = add_gaussian_noise(ref_img, sigma=g_sigma)

    for name, model_key, t_loader, v_loader, ckpt_name in train_plan:
        model = build_model(model_key)
        train_model(
            model=model,
            train_loader=t_loader,
            val_loader=v_loader,
            epochs=epochs,
            lr=config.training.lr,
            device=config.training.device,
            model_name=name,
        )

        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        size_kb = save_checkpoint(model, ckpt_path)
        print(f"Saved: {ckpt_path} ({size_kb} KB)")

        # Verify Delta = 0.0 reproducibility
        reloaded = load_checkpoint(lambda: build_model(model_key), ckpt_path, device=config.training.device)
        max_diff = verify_reproducibility(model, reloaded, noisy_sample, device=config.training.device)
        print(f"  Reproducibility delta verified: {max_diff:.10f}")

    print("\nAll models trained and checkpoints verified.")


def cmd_benchmark(args: argparse.Namespace, config: PipelineConfig):
    """Run comparative benchmark against test set and generate charts."""
    print("=== EXECUTING COMPARATIVE BENCHMARK ===")
    ckpt_dir = config.training.checkpoint_dir

    models_to_bench = {
        "Direct CNN": ("direct_cnn", "checkpoint_direct_cnn.pth"),
        "Residual CNN": ("residual_cnn", "checkpoint_residual_cnn.pth"),
        "Small DnCNN": ("small_dncnn", "checkpoint_small_dncnn.pth"),
        "Strong DnCNN (Gaussian)": ("strong_dncnn", "checkpoint_strong_dncnn_gaussian.pth"),
        "Strong DnCNN (Mixed)": ("strong_dncnn", "checkpoint_strong_dncnn_mixed.pth"),
    }

    loaded_neural_models: Dict[str, Tuple[nn.Module, float]] = {}

    for display_name, (model_key, ckpt_filename) in models_to_bench.items():
        # Check inside ckpt_dir or root directory
        candidate_paths = [
            os.path.join(ckpt_dir, ckpt_filename),
            os.path.join("..", ckpt_filename),
            ckpt_filename,
        ]
        found_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                found_path = p
                break

        if found_path is not None:
            m = load_checkpoint(lambda: build_model(model_key), found_path, device=config.training.device)
            size_kb = round(os.path.getsize(found_path) / 1024.0, 2)
            loaded_neural_models[display_name] = (m, size_kb)
            print(f"Loaded checkpoint for {display_name} from: {found_path}")
        else:
            print(f"Notice: Checkpoint {ckpt_filename} not found. Instantiating fresh {model_key} model.")
            m = build_model(model_key)
            loaded_neural_models[display_name] = (m, 0.0)

    runner = BenchmarkRunner(config)
    full_df, summary_df = runner.run(
        neural_models=loaded_neural_models,
        include_classical=not args.no_classical,
        max_images=args.max_images,
    )

    plot_path = os.path.join(config.evaluation.output_dir, config.evaluation.plot_filename)
    plot_benchmark_results(summary_df, output_filepath=plot_path)
    print("Benchmark completed successfully.")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config YAML file",
    )

    parser = argparse.ArgumentParser(
        description="AMAT-2 SEM Denoising Pipeline CLI",
        parents=[config_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train command
    p_train = subparsers.add_parser(
        "train",
        parents=[config_parser],
        help="Train all neural models and save checkpoints",
    )
    p_train.add_argument("--epochs", type=int, default=None, help="Override training epochs")

    # Benchmark command
    p_bench = subparsers.add_parser(
        "benchmark",
        parents=[config_parser],
        help="Benchmark models on test set and plot results",
    )
    p_bench.add_argument("--no-classical", action="store_true", help="Exclude classical filters from benchmark")
    p_bench.add_argument("--max-images", type=int, default=None, help="Maximum test images to evaluate (for fast runs)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if os.path.exists(args.config):
        config = PipelineConfig.from_yaml(args.config)
    else:
        print(f"Notice: Config file '{args.config}' not found, using default configuration.")
        config = PipelineConfig()

    if args.command == "train":
        cmd_train(args, config)
    elif args.command == "benchmark":
        cmd_benchmark(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
