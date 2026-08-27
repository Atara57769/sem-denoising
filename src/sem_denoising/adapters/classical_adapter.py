"""
Classical (non-learned) Baseline Model Adapter.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

from sem_denoising.adapters.base import ModelAdapter, ModelMetadata
from sem_denoising.models.classical import (
    denoise_nlm,
    denoise_gaussian_filter,
    denoise_wavelet_baseline,
    denoise_identity,
)


class ClassicalAdapter(ModelAdapter):
    """
    Adapter wrapper for non-learned classical baselines (NLM, Gaussian, Wavelet, Identity).
    """

    def __init__(self, method_name: str = "nlm"):
        self.method_key = method_name.lower().strip()
        if self.method_key in ("nlm", "non_local_means", "non-local means"):
            self.func = denoise_nlm
            self.name = "Non-Local Means (NLM)"
            self.model_id = "nlm_classical"
        elif self.method_key in ("gaussian", "gaussian_filter", "gaussian filter"):
            self.func = denoise_gaussian_filter
            self.name = "Gaussian Filter"
            self.model_id = "gaussian_classical"
        elif self.method_key in ("wavelet", "bayesshrink", "wavelet (bayesshrink)"):
            self.func = denoise_wavelet_baseline
            self.name = "Wavelet (BayesShrink)"
            self.model_id = "wavelet_classical"
        elif self.method_key in ("identity",):
            self.func = denoise_identity
            self.name = "Identity Baseline"
            self.model_id = "identity_classical"
        else:
            raise ValueError(f"Unknown classical method '{method_name}'. Supported: nlm, gaussian, wavelet, identity")

    def infer(self, noisy_img: np.ndarray, precision: Optional[str] = None) -> Tuple[np.ndarray, float]:
        return self.func(noisy_img)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            architecture=self.name,
            parameters=0,
            precision="cpu_float32",
            checkpoint_hash="none_classical",
            preprocessing_declaration="none",
            output_range=(0.0, 1.0),
            inference_command=f"Classical {self.name}",
            runtime_backend="CPU",
        )
