"""
Visualization suite for SEM denoising benchmark results.
"""

import os
from typing import Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_benchmark_results(
    summary_df: pd.DataFrame,
    output_filepath: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate clean 4-panel comparative bar chart (PSNR, SSIM, Latency, Parameter Count).

    Args:
        summary_df: Aggregated summary DataFrame with columns:
                    'method', 'psnr', 'ssim', 'runtime_ms', 'params'.
        output_filepath: Target image file path (PNG).
        dpi: Image resolution.

    Returns:
        matplotlib Figure object.
    """
    # Sort by PSNR for clean bar chart presentation
    df_sorted = summary_df.sort_values(by="psnr", ascending=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. PSNR (Higher is better)
    axes[0, 0].barh(df_sorted["method"], df_sorted["psnr"], color="#2ecc71")
    axes[0, 0].set_title("Average PSNR (dB) — Higher is Better", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("PSNR (dB)")
    axes[0, 0].grid(axis="x", linestyle="--", alpha=0.7)

    # 2. SSIM (Higher is better)
    axes[0, 1].barh(df_sorted["method"], df_sorted["ssim"], color="#3498db")
    axes[0, 1].set_title("Average SSIM — Higher is Better", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("SSIM")
    axes[0, 1].grid(axis="x", linestyle="--", alpha=0.7)

    # 3. CPU Runtime ms (Lower is better)
    axes[1, 0].barh(df_sorted["method"], df_sorted["runtime_ms"], color="#e74c3c")
    axes[1, 0].set_title("Average CPU Runtime (ms) — Lower is Better", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Runtime (ms)")
    axes[1, 0].grid(axis="x", linestyle="--", alpha=0.7)

    # 4. Parameter Count
    axes[1, 1].barh(df_sorted["method"], df_sorted["params"], color="#9b59b6")
    axes[1, 1].set_title("Trainable Parameters Count", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Parameter Count")
    axes[1, 1].grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if output_filepath:
        dirpath = os.path.dirname(output_filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(output_filepath, dpi=dpi)
        print(f"Saved comparative benchmarking chart figure: {output_filepath}")

    return fig
