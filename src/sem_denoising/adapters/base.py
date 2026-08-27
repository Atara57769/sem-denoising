"""
Architecture-Agnostic Model Adapter Contract (Base Specification).

Defines the universal adapter interface for evaluating any SEM denoising model
(AMAT-2 TinyDnCNN, AMAT-1 ConvRefinedAddUNet-L, or future external candidates)
without modifying the benchmark pipeline or evaluator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch


class ModelName(str, Enum):
    """Enumeration of supported SEM Denoising Model & Baseline Adapter names."""
    TINY_DNCNN_FP16 = "tiny_dncnn_d8w48_fp16"
    TINY_DNCNN_FP32 = "tiny_dncnn_d8w48_fp32"
    DIRECT_CNN = "direct_cnn"
    RESIDUAL_CNN = "residual_cnn"
    SMALL_DNCNN = "small_dncnn"
    STRONG_DNCNN_GAUSSIAN = "strong_dncnn_gaussian"
    STRONG_DNCNN_MIXED = "strong_dncnn_mixed"
    AMAT1 = "amat1"
    NLM = "nlm"
    GAUSSIAN = "gaussian"
    WAVELET = "wavelet"
    IDENTITY = "identity"


@dataclass
class ModelMetadata:
    """Standardized metadata structure for cross-architecture comparison."""
    model_id: str
    architecture: str
    parameters: int
    precision: str  # 'fp32', 'fp16', 'int8'
    checkpoint_hash: str
    preprocessing_declaration: str
    output_range: Tuple[float, float]
    inference_command: str
    runtime_backend: str
    macs_per_512x512: Optional[int] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelAdapter(ABC):
    """
    Abstract Model Adapter Contract.
    
    Input Contract:
        - 2D grayscale NumPy array (H x W), float32, range [0.0, 1.0] (or unclipped noise).
    
    Output Contract:
        - 2D grayscale NumPy array (H x W), float32, clipped to [0.0, 1.0].
        - Inference latency in milliseconds.
    """

    @abstractmethod
    def infer(self, noisy_img: np.ndarray, precision: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        Execute full-field model inference.

        Args:
            noisy_img: 2D numpy array [H, W] float32.
            precision: Optional precision override ('fp32', 'fp16').

        Returns:
            (denoised_img as 2D numpy array [H, W] float32 in [0,1], runtime_ms as float)
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return standardized model provenance and architecture metadata."""
        pass
