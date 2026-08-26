"""
Overfitting sanity check routines to verify network capacity on single patches.
"""

from typing import Callable, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn


def run_overfit_sanity_check(
    model: nn.Module,
    clean_patch: np.ndarray,
    noise_fn: Callable[[np.ndarray], np.ndarray],
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[float, float, List[float]]:
    """
    Overfit model on a single patch tensor to test backprop and optimization capacity.

    Args:
        model: PyTorch model.
        clean_patch: 2D numpy array representing a single clean patch.
        noise_fn: Callable that applies noise to clean patch.
        epochs: Number of iterations on the single patch.
        lr: Learning rate.
        device: 'cpu' or 'cuda'.

    Returns:
        (initial_loss, final_loss, loss_history)
    """
    model = model.to(device)
    model.train()

    noisy_patch = noise_fn(clean_patch)
    clean_t = torch.from_numpy(clean_patch).unsqueeze(0).unsqueeze(0).float().to(device)
    noisy_t = torch.from_numpy(noisy_patch).unsqueeze(0).unsqueeze(0).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history: List[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(noisy_t)
        loss = criterion(pred, clean_t)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return loss_history[0], loss_history[-1], loss_history


def run_all_models_sanity_check(
    models_dict: Dict[str, nn.Module],
    sample_patch: np.ndarray,
    noise_fn: Callable[[np.ndarray], np.ndarray],
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Run sanity check across a dictionary of models and return summary results.
    """
    results = {}
    print(f"Running Overfit Sanity Check ({epochs} steps on 1 patch):")
    for name, model in models_dict.items():
        init_loss, final_loss, _ = run_overfit_sanity_check(
            model, sample_patch, noise_fn, epochs=epochs, lr=lr, device=device
        )
        reduction = ((init_loss - final_loss) / max(init_loss, 1e-8)) * 100.0
        results[name] = {
            "initial_loss": init_loss,
            "final_loss": final_loss,
            "reduction_pct": reduction,
        }
        print(f"  Model: {name:20s} | Initial: {init_loss:.6f} -> Final: {final_loss:.6f} ({reduction:.1f}% reduction)")

    return results

