"""
DnCNN Model Adapter Implementation for AMAT-2 Candidates (TinyDnCNN D8/W48, etc.).
"""

import os
import hashlib
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from sem_denoising.adapters.base import ModelAdapter, ModelMetadata
from sem_denoising.models.registry import count_parameters, build_model, ModelType
from sem_denoising.metrics import Timer


def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "N/A"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class DnCNNAdapter(ModelAdapter):
    """
    Adapter implementation for DnCNN architecture variants (D8/W48 FP32 and FP16).
    """

    def __init__(
        self,
        model_name: str = "tiny_dncnn_d8w48",
        checkpoint_path: Optional[str] = "checkpoints/checkpoint_tinydncnn_d8w48.pth",
        precision: str = "fp16",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.precision = precision.lower()
        self.device = device

        self.model = build_model(model_name)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model.to(device)

        self.ckpt_hash = compute_file_sha256(checkpoint_path) if checkpoint_path else "N/A"
        self.param_count = count_parameters(self.model)

    def infer(self, noisy_img: np.ndarray, precision: Optional[str] = None) -> Tuple[np.ndarray, float]:
        prec = (precision or self.precision).lower()
        inp_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with Timer() as timer:
            with torch.no_grad():
                if prec == "fp16":
                    # Whole-image FP16 inference
                    inp_half = inp_tensor.half()
                    model_half = self.model.half()
                    out_half = model_half(inp_half)
                    out = out_half.float()
                    if getattr(model_half, "residual", True):
                        denoised = inp_tensor - out
                    else:
                        denoised = out
                else:
                    model_fp32 = self.model.float()
                    out = model_fp32(inp_tensor)
                    if getattr(model_fp32, "residual", True):
                        denoised = inp_tensor - out
                    else:
                        denoised = out

        denoised_img = torch.clamp(denoised, 0.0, 1.0).squeeze().cpu().numpy().astype(np.float32)
        return denoised_img, timer.elapsed_ms

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=f"{self.model_name}_{self.precision}",
            architecture=f"TinyDnCNN D8/W48 ({self.precision.upper()})",
            parameters=self.param_count,
            precision=self.precision,
            checkpoint_hash=self.ckpt_hash[:16],
            preprocessing_declaration="grayscale [0,1] float32 array, whole-image",
            output_range=(0.0, 1.0),
            inference_command=f"python -m sem_denoising.cli_benchmark --model {self.model_name} --precision {self.precision}",
            runtime_backend=f"PyTorch {torch.__version__} ({self.device})",
            macs_per_512x512=3300122624,
            extra_fields={
                "residual_learning": True,
                "bn_layers": True,
                "depth": 8,
                "width": 48,
            },
        )
