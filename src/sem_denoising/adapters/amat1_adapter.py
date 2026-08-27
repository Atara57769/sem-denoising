"""
AMAT-1 Model Adapter Implementation for ConvRefinedAddUNet-L (using AMAT1_TO_AMAT2_PACKAGE_V1).

Ingests the provisional AMAT-1 reference candidate (ConvRefinedAddUNet-L, 2,040,419 params)
via the standardized ModelAdapter interface for head-to-head benchmarking against AMAT-2 models.
"""

import os
import sys
import hashlib
import importlib.util
from typing import Tuple, Optional
import numpy as np
import torch

from sem_denoising.adapters.base import ModelAdapter, ModelMetadata
from sem_denoising.metrics import Timer


def _load_amat1_package_module(package_dir: str):
    impl_path = os.path.join(package_dir, "model", "implementation.py")
    if not os.path.exists(impl_path):
        raise FileNotFoundError(f"AMAT-1 implementation file not found at: {impl_path}")

    spec = importlib.util.spec_from_file_location("amat1_implementation", impl_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AMAT1Adapter(ModelAdapter):
    """
    Adapter implementation for AMAT-1 ConvRefinedAddUNet-L model.
    """

    def __init__(
        self,
        package_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.package_dir = package_dir
        self.device = device
        self.module = _load_amat1_package_module(package_dir)
        self.model = self.module.build_model()
        
        self.checkpoint_path = checkpoint_path
        self.ckpt_hash = "N/A"

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            self.model.load_state_dict(state_dict, strict=False)
            hasher = hashlib.sha256()
            with open(checkpoint_path, "rb") as f:
                hasher.update(f.read())
            self.ckpt_hash = hasher.hexdigest()

        self.model.eval()
        self.model.to(device)
        self.param_count = self.module.count_parameters(self.model)

    def infer(self, noisy_img: np.ndarray, precision: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        Execute inference for ConvRefinedAddUNet-L.
        Input image must be float32 with dimensions divisible by 4.
        """
        h, w = noisy_img.shape
        # Pad image to nearest multiple of 4 if necessary
        pad_h = (4 - (h % 4)) % 4
        pad_w = (4 - (w % 4)) % 4

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode="edge")
        else:
            padded = noisy_img

        inp = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with Timer() as timer:
            with torch.no_grad():
                out = self.model(inp)

        denoised_padded = out.squeeze().cpu().numpy()
        denoised = denoised_padded[:h, :w]
        denoised_clipped = np.clip(denoised, 0.0, 1.0).astype(np.float32)

        return denoised_clipped, timer.elapsed_ms

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="conv_refined_add_unet_l_fp32",
            architecture="ConvRefinedAddUNet-L (AMAT-1 Reference)",
            parameters=self.param_count,
            precision="fp32",
            checkpoint_hash=self.ckpt_hash[:16],
            preprocessing_declaration="uint8 / 255 -> float32 HxW divisible by 4, direct prediction, clip_0_1",
            output_range=(0.0, 1.0),
            inference_command="python verify_package.py",
            runtime_backend=f"PyTorch {torch.__version__} ({self.device})",
            macs_per_512x512=110791491584,
            extra_fields={
                "base": 67,
                "levels": 3,
                "legacy_family": "additive_unet",
                "package_version": "V1",
                "status": "PROVISIONAL_QUALITY_REFERENCE",
            },
        )
