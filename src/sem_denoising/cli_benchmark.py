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
from sem_denoising.models import build_model, ModelType
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
    return parser


def run_benchmark(config: PipelineConfig):
    """Run comparative benchmark against test set and generate summary plots/CSVs using configuration settings."""
    print("=== EXECUTING COMPARATIVE BENCHMARK ===")
    ckpt_dir = config.training.checkpoint_dir
    device = config.training.device
    output_dir = config.evaluation.output_dir
    include_classical = config.evaluation.include_classical
    max_images = config.evaluation.max_images
    os.makedirs(output_dir, exist_ok=True)

    models_to_bench = {
        "Direct CNN": (ModelType.DIRECT_CNN, "checkpoint_direct_cnn.pth"),
        "Small DnCNN": (ModelType.SMALL_DNCNN, "checkpoint_small_dncnn.pth"),
        "Strong DnCNN (Gaussian)": (ModelType.STRONG_DNCNN, "checkpoint_strong_dncnn_gaussian.pth"),
        "Skip DnCNN (Gaussian)": (ModelType.SKIP_DNCNN, "checkpoint_skip_dncnn.pth"),
        "Strong DnCNN (Mixed)": (ModelType.STRONG_DNCNN, "checkpoint_strong_dncnn_mixed.pth"),
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
        include_classical=include_classical,
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

    run_benchmark(config=config)


if __name__ == "__main__":
    main()
