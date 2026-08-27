"""
Dedicated Command-Line Interface (CLI) for Full AMAT-2 & AMAT-1 Benchmarking Suite.
"""

import os
import sys
import argparse

from typing import List, Optional
from sem_denoising.config import PipelineConfig
from sem_denoising.experiments.benchmark import BenchmarkRunner
from sem_denoising.adapters import (
    ModelAdapter,
    get_adapter_registry,
    list_available_adapters,
)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for full benchmark pipeline."""
    parser = argparse.ArgumentParser(
        description="AMAT-2 Full Reproducible Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/amat2_frozen_d8w48.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap on test images for rapid validation",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of selected model names/IDs to run from the registry (e.g. --models tinydncnn_fp16 amat1 nlm)",
    )
    return parser


def run_cli_benchmark(
    config_path: str,
    max_images: Optional[int] = None,
    models: Optional[List[str]] = None,
):
    if os.path.exists(config_path):
        config = PipelineConfig.from_yaml(config_path)
    else:
        print(f"Notice: Config file '{config_path}' not found, using default configuration.")
        config = PipelineConfig()

    print(f"=== EXECUTING AMAT-2 REPRODUCIBLE BENCHMARK PIPELINE ===")
    print(f"Config: {config_path}")

    registry = get_adapter_registry()

    if not models:
        available = list_available_adapters()
        print(
            f"Error: No models specified via --models. Please pass model names to benchmark.\n"
            f"Available models in registry: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    selected_adapters: dict[str, ModelAdapter] = {}

    for m in models:
        m_key = m.lower().strip()
        if m_key in registry:
            adapter = registry[m_key]
            selected_adapters[adapter.get_metadata().architecture] = adapter
        else:
            available = list_available_adapters()
            raise ValueError(
                f"Unknown model name '{m}'. Available models in registry: {available}"
            )

    print(f"Selected Adapters ({len(selected_adapters)}): {list(selected_adapters.keys())}")

    runner = BenchmarkRunner(config)
    full_df, summary_df = runner.run(
        adapters=selected_adapters,
        max_images=max_images,
        verbose=True,
    )

    print("\nBenchmark pipeline execution complete.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_cli_benchmark(
        config_path=args.config,
        max_images=args.max_images,
        models=args.models,
    )


if __name__ == "__main__":
    main()
