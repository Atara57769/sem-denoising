"""
Unified training and validation engine for PyTorch SEM denoising models.
"""

from typing import Tuple, List, Dict, Optional, Callable
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """Train model for a single epoch and return average training loss."""
    model.train()
    total_loss = 0.0

    for noisy_b, clean_b in dataloader:
        noisy_b = noisy_b.to(device)
        clean_b = clean_b.to(device)

        optimizer.zero_grad()
        out = model(noisy_b)
        pred_clean = (noisy_b - out) if model.residual else out
        loss = criterion(pred_clean, clean_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * noisy_b.size(0)

    return total_loss / len(dataloader.dataset)


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """Validate model for a single epoch and return average validation loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for noisy_b, clean_b in dataloader:
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)

            out = model(noisy_b)
            pred_clean = (noisy_b - out) if model.residual else out
            loss = criterion(pred_clean, clean_b)
            total_loss += loss.item() * noisy_b.size(0)

    return total_loss / len(dataloader.dataset)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    device: str = "cpu",
    model_name: str = "Model",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Unified training loop across all epochs returning loss histories.

    Args:
        model: PyTorch model instance.
        train_loader: DataLoader for training patches.
        val_loader: DataLoader for validation patches.
        epochs: Number of complete training epochs.
        lr: Learning rate for Adam optimizer.
        device: 'cpu' or 'cuda'.
        model_name: Name for progress logging.
        verbose: Whether to print per-epoch progress.

    Returns:
        Dictionary with 'train_loss' and 'val_loss' lists.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_history: List[float] = []
    val_history: List[float] = []

    if verbose:
        print(f"--- Training {model_name} for {epochs} epoch(s) on {device.upper()} ---")

    for epoch in range(1, epochs + 1):
        t_start = time.perf_counter()
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        v_loss = validate_one_epoch(model, val_loader, criterion, device=device)
        t_elapsed = time.perf_counter() - t_start

        train_history.append(t_loss)
        val_history.append(v_loss)

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({t_elapsed:.1f}s) | "
                f"Train Loss: {t_loss:.6f} | Val Loss: {v_loss:.6f}"
            )

    return {"train_loss": train_history, "val_loss": val_history}

