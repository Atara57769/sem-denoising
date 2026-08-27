"""
Comprehensive Architecture-Agnostic Benchmarking Engine for AMAT-2 & AMAT-1 Models.

Evaluates models across all 5 registered degradation conditions:
1. Primary i.i.d. Mixed Poisson-Gaussian Noise
2. Detector Line Striping
3. Beam Defocus Blur
4. Scan Line Shear Drift
5. Mixed Correlated Degradation
"""

import os
import glob
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from sem_denoising.data.loader import get_clean_reference_path, load_image
from sem_denoising.models.classical import CLASSICAL_METHODS, denoise_nlm
from sem_denoising.metrics import evaluate_predictions, compute_sobel_mae, Timer
from sem_denoising.config import PipelineConfig
from sem_denoising.noise import (
    add_mixed_noise,
    add_striping_noise,
    add_blur_degradation,
    add_scan_drift_degradation,
    add_mixed_correlated_degradation,
)
from sem_denoising.adapters import get_adapter, list_available_adapters

STRESS_CONDITIONS = {
    "i.i.d. Mixed Noise": add_mixed_noise,
    "Detector Line Striping": add_striping_noise,
    "Beam Defocus Blur": add_blur_degradation,
    "Scan Line Shear Drift": add_scan_drift_degradation,
    "Mixed Correlated": add_mixed_correlated_degradation,
}


def run_full_benchmark(
    config: PipelineConfig,
    adapters: Optional[Union[Dict[str, ModelAdapter], List[str]]] = None,
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute generic architecture-agnostic benchmark on test set using model IDs or adapters.

    Returns:
        (df_long, summary_df)
    """
    data_root = config.data.data_root
    test_set_id = config.data.test_set
    test_dir = os.path.join(data_root, "intensity_sets", f"set{test_set_id}")

    clean_paths = sorted(glob.glob(os.path.join(test_dir, "*.tiff")))
    if not clean_paths:
        clean_paths = sorted(glob.glob(os.path.join(data_root, f"*set{test_set_id}*", "*.tiff")))

    if not clean_paths:
        raise FileNotFoundError(f"No clean reference images found for Set {test_set_id} in {test_dir}")

    if max_images is not None and max_images > 0:
        clean_paths = clean_paths[:max_images]

    adapter_dict: Dict[str, ModelAdapter] = {}
    if adapters is None:
        model_ids = ["tiny_dncnn_d8w48_fp16", "tiny_dncnn_d8w48_fp32", "amat1_conv_refined_add_unet_l"]
        for m_id in model_ids:
            adapter = get_adapter(m_id)
            meta = adapter.get_metadata()
            adapter_dict[meta.architecture] = adapter
    elif isinstance(adapters, list):
        for m_id in adapters:
            adapter = get_adapter(m_id)
            meta = adapter.get_metadata()
            adapter_dict[meta.architecture] = adapter
    else:
        adapter_dict = adapters

    records: List[Dict[str, Any]] = []
    pbar = tqdm(clean_paths, desc="Benchmarking Images", disable=not verbose)

    for img_path in pbar:
        img_name = os.path.basename(img_path)
        gt_clean = load_image(img_path)

        for condition_name, corrupt_fn in STRESS_CONDITIONS.items():
            np.random.seed(42)  # Deterministic corruption per image x condition
            noisy_img = corrupt_fn(gt_clean)

            # 1. Classical Baseline: Non-Local Means (NLM)
            denoised_nlm, runtime_nlm = denoise_nlm(noisy_img)
            metrics_nlm = evaluate_predictions(gt_clean, denoised_nlm, runtime_ms=runtime_nlm)
            sobel_nlm = compute_sobel_mae(gt_clean, denoised_nlm)
            records.append({
                "image": img_name,
                "condition": condition_name,
                "method": "Non-Local Means (NLM)",
                "category": "Classical Baseline",
                "psnr": metrics_nlm["psnr"],
                "ssim": metrics_nlm["ssim"],
                "mse": metrics_nlm["mse"],
                "sobel_mae": sobel_nlm,
                "runtime_ms": metrics_nlm["runtime_ms"],
                "params": 0,
                "precision": "CPU float32",
                "status": "ELIGIBLE",
            })

            # 2. Neural Model Adapters
            for adapter_name, adapter in adapter_dict.items():
                meta = adapter.get_metadata()
                denoised, runtime_ms = adapter.infer(noisy_img)
                metrics = evaluate_predictions(gt_clean, denoised, runtime_ms=runtime_ms)
                sobel_mae = compute_sobel_mae(gt_clean, denoised)

                status_val = meta.extra_fields.get("status", "ELIGIBLE" if "TinyDnCNN" in adapter_name else "RESEARCH_ONLY")

                records.append({
                    "image": img_name,
                    "condition": condition_name,
                    "method": adapter_name,
                    "category": meta.extra_fields.get("category", "Neural Model"),
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "mse": metrics["mse"],
                    "sobel_mae": sobel_mae,
                    "runtime_ms": metrics["runtime_ms"],
                    "params": meta.parameters,
                    "precision": meta.precision.upper(),
                    "status": status_val,
                })

    df_long = pd.DataFrame(records)

    # Save outputs
    output_dir = config.evaluation.output_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "amat2_full_benchmark_results.csv")
    parquet_path = os.path.join(output_dir, "amat2_full_benchmark_results.parquet")

    df_long.to_csv(csv_path, index=False)
    try:
        df_long.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    summary_df = (
        df_long.groupby(["condition", "method", "precision", "status"])
        .agg({
            "psnr": "mean",
            "ssim": "mean",
            "sobel_mae": "mean",
            "runtime_ms": "mean",
            "params": "first",
        })
        .reset_index()
        .sort_values(by=["condition", "psnr"], ascending=[True, False])
    )

    summary_csv = os.path.join(output_dir, "amat2_benchmark_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    if verbose:
        print(f"\n=== AMAT-2 BENCHMARK COMPLETED ===")
        print(f"Exported long-form results to: {csv_path}")
        print(summary_df.to_string(index=False))

    return df_long, summary_df


class BenchmarkRunner:
    """
    High-level class coordinator for full multi-condition benchmark execution.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        adapters: Optional[Dict[str, ModelAdapter]] = None,
        max_images: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute full benchmark using configured model adapters and return (df_long, summary_df).
        """
        return run_full_benchmark(
            config=self.config,
            adapters=adapters,
            max_images=max_images,
            verbose=verbose,
        )
