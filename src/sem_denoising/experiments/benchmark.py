"""
Comprehensive benchmarking engine for classical and neural SEM denoising baselines.
"""

import os
import glob
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from sem_denoising.data.loader import get_clean_reference_path, load_image
from sem_denoising.models.classical import CLASSICAL_METHODS
from sem_denoising.models.registry import count_parameters
from sem_denoising.metrics import evaluate_predictions, Timer
from sem_denoising.config import PipelineConfig


def run_neural_inference(model: nn.Module,noisy_img: np.ndarray,device: str = "cpu",) -> Tuple[np.ndarray, float]:
    """
    Run neural model inference on a single image and record CPU/device latency in ms.

    Args:
        model: PyTorch neural model.
        noisy_img: 2D numpy array [0.0, 1.0].
        device: 'cpu' or 'cuda'.

    Returns:
        (denoised_img as float32 numpy array, runtime_ms)
    """
    model.eval()
    model.to(device)
    inp = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).float().to(device)

    with Timer() as timer:
        with torch.no_grad():
            out = model(inp)

    denoised_img = out.squeeze().cpu().numpy()
    return np.clip(denoised_img, 0.0, 1.0).astype(np.float32), timer.elapsed_ms


def evaluate_dataset(
    image_paths: List[str],
    gt_clean: np.ndarray,
    neural_models: Dict[str, Tuple[nn.Module, float]],
    include_classical: bool = True,
    max_images: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all classical and neural models across a list of test images.

    Args:
        image_paths: List of file paths to noisy test images.
        gt_clean: Ground truth clean reference image.
        neural_models: Dict mapping model_name -> (model_instance, checkpoint_size_kb).
        include_classical: Whether to run classical baselines.
        max_images: Optional maximum number of images to evaluate.
        device: 'cpu' or 'cuda'.
        verbose: Whether to show progress bar.

    Returns:
        DataFrame containing per-image evaluation results.
    """
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    records: List[Dict[str, Any]] = []
    pbar = tqdm(image_paths, desc="Benchmarking Test Images", disable=not verbose)

    for p in pbar:
        img_name = os.path.basename(p)
        noisy_img = load_image(p)

        # 1. Classical Baselines
        if include_classical:
            for method_name, method_fn in CLASSICAL_METHODS.items():
                denoised, runtime_ms = method_fn(noisy_img)
                metrics = evaluate_predictions(gt_clean, denoised, runtime_ms=runtime_ms)
                records.append({
                    "image": img_name,
                    "method": method_name,
                    "type": "classical",
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "mse": metrics["mse"],
                    "runtime_ms": metrics["runtime_ms"],
                    "params": 0,
                    "size_kb": 0.0,
                })

        # 2. Neural Models
        for model_name, (model, size_kb) in neural_models.items():
            denoised, runtime_ms = run_neural_inference(model, noisy_img, device=device)
            metrics = evaluate_predictions(gt_clean, denoised, runtime_ms=runtime_ms)
            records.append({
                "image": img_name,
                "method": model_name,
                "type": "neural",
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "mse": metrics["mse"],
                "runtime_ms": metrics["runtime_ms"],
                "params": count_parameters(model),
                "size_kb": size_kb,
            })

    return pd.DataFrame(records)


class BenchmarkRunner:
    """High-level benchmarking coordinator."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        neural_models: Dict[str, Tuple[nn.Module, float]],
        include_classical: bool = True,
        max_images: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute full benchmark on configured test set, save CSV, and return (full_df, summary_df).
        """
        test_set_id = self.config.data.test_set
        data_root = self.config.data.data_root
        test_dir = os.path.join(data_root, "intensity_sets", f"set{test_set_id}")

        test_paths = sorted(glob.glob(os.path.join(test_dir, "*.tiff")))
        if not test_paths:
            # Fallback to general search if path structure differs
            test_paths = sorted(glob.glob(os.path.join(data_root, f"*set{test_set_id}*", "*.tiff")))

        if not test_paths:
            raise FileNotFoundError(f"No test images found for Set {test_set_id} in {test_dir}")

        gt_clean = load_image(get_clean_reference_path(test_set_id, data_root=data_root))

        df_all = evaluate_dataset(
            image_paths=test_paths,
            gt_clean=gt_clean,
            neural_models=neural_models,
            include_classical=include_classical,
            max_images=max_images,
            device=self.config.training.device,
        )

        output_dir = self.config.evaluation.output_dir
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, self.config.evaluation.results_csv)
        df_all.to_csv(csv_path, index=False)
        print(f"Exported detailed per-image benchmarking results to: {csv_path}")

        summary_df = (
            df_all.groupby(["method", "type"])
            .agg({
                "psnr": "mean",
                "ssim": "mean",
                "mse": "mean",
                "runtime_ms": "mean",
                "params": "first",
                "size_kb": "first",
            })
            .reset_index()
            .sort_values(by="psnr", ascending=False)
        )

        print("\n=== FINAL COMPARATIVE BENCHMARK TABLE ===")
        print(summary_df.to_string(index=False))

        return df_all, summary_df

