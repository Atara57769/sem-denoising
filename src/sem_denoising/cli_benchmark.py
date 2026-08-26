"""
Dedicated Command-Line Interface (CLI) for Benchmarking SEM Denoising Models.
"""

import os
import sys
import argparse
from typing import Dict, Tuple
import torch.nn as nn

from sem_denoising.config import PipelineConfig
from sem_denoising.models import build_model
from sem_denoising.training import load_checkpoint
from sem_denoising.experiments.benchmark import BenchmarkRunner
from sem_denoising.experiments.visualizer import plot_benchmark_results


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for benchmarking."""
    parser = argparse.ArgumentParser(
        description="SEM Denoising Comparative Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-classical",
        action="store_true",
        help="Exclude classical filters (Gaussian, NLM, Wavelet) from benchmark",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum test images to evaluate (for fast test runs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override evaluation device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for plots and CSVs",
    )
    return parser


def run_benchmark(config: PipelineConfig, no_classical: bool = False, max_images=None, device_override=None, output_dir_override=None):
    """Run comparative benchmark against test set and generate summary plots/CSVs."""
    print("=== EXECUTING COMPARATIVE BENCHMARK ===")
    ckpt_dir = config.training.checkpoint_dir
    device = device_override or config.training.device
    output_dir = output_dir_override or config.evaluation.output_dir
    os.makedirs(output_dir, exist_ok=True)

    models_to_bench = {
        "Direct CNN": ("direct_cnn", "checkpoint_direct_cnn.pth"),
        "Residual CNN": ("residual_cnn", "checkpoint_residual_cnn.pth"),
        "Small DnCNN": ("small_dncnn", "checkpoint_small_dncnn.pth"),
        "Strong DnCNN (Gaussian)": ("strong_dncnn", "checkpoint_strong_dncnn_gaussian.pth"),
        "Strong DnCNN (Mixed)": ("strong_dncnn", "checkpoint_strong_dncnn_mixed.pth"),
    }

    loaded_neural_models: Dict[str, Tuple[nn.Module, float]] = {}

    for display_name, (model_key, ckpt_filename) in models_to_bench.items():
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
            m = load_checkpoint(lambda: build_model(model_key), found_path, device=device)
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
        include_classical=not no_classical,
        max_images=max_images,
    )

    plot_path = os.path.join(output_dir, config.evaluation.plot_filename)
    plot_benchmark_results(summary_df, output_filepath=plot_path)
    print("Benchmark completed successfully.")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = PipelineConfig.from_yaml(args.config)
    else:
        print(f"Notice: Config file '{args.config}' not found, using default configuration.")
        config = PipelineConfig()

    run_benchmark(
        config=config,
        no_classical=args.no_classical,
        max_images=args.max_images,
        device_override=args.device,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
