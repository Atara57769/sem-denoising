"""
Architecture-Agnostic Model Adapters Package.
"""

from typing import Dict, List, Any, Union
from sem_denoising.adapters.base import ModelAdapter, ModelMetadata, ModelName
from sem_denoising.adapters.dncnn_adapter import DnCNNAdapter
from sem_denoising.adapters.amat1_adapter import AMAT1Adapter
from sem_denoising.adapters.classical_adapter import ClassicalAdapter


def get_adapter_registry() -> Dict[str, ModelAdapter]:
    """Get dictionary mapping model names/IDs to instantiated ModelAdapter instances."""
    return {
        ModelName.TINY_DNCNN_FP16.value: DnCNNAdapter(
            model_name="tiny_dncnn_d8w48",
            checkpoint_path="checkpoints/checkpoint_tinydncnn_d8w48.pth",
            precision="fp16",
        ),
        ModelName.TINY_DNCNN_FP32.value: DnCNNAdapter(
            model_name="tiny_dncnn_d8w48",
            checkpoint_path="checkpoints/checkpoint_tinydncnn_d8w48.pth",
            precision="fp32",
        ),
        ModelName.DIRECT_CNN.value: DnCNNAdapter(
            model_name="direct_cnn",
            checkpoint_path="checkpoints/checkpoint_direct_cnn.pth",
            precision="fp32",
        ),
        ModelName.RESIDUAL_CNN.value: DnCNNAdapter(
            model_name="residual_cnn",
            checkpoint_path="checkpoints/checkpoint_residual_cnn.pth",
            precision="fp32",
        ),
        ModelName.SMALL_DNCNN.value: DnCNNAdapter(
            model_name="small_dncnn",
            checkpoint_path="checkpoints/checkpoint_small_dncnn.pth",
            precision="fp32",
        ),
        ModelName.STRONG_DNCNN_GAUSSIAN.value: DnCNNAdapter(
            model_name="strong_dncnn",
            checkpoint_path="checkpoints/checkpoint_strong_dncnn_gaussian.pth",
            precision="fp32",
        ),
        ModelName.STRONG_DNCNN_MIXED.value: DnCNNAdapter(
            model_name="strong_dncnn",
            checkpoint_path="checkpoints/checkpoint_strong_dncnn_mixed.pth",
            precision="fp32",
        ),
        ModelName.AMAT1.value: AMAT1Adapter(),
        ModelName.NLM.value: ClassicalAdapter("nlm"),
        ModelName.GAUSSIAN.value: ClassicalAdapter("gaussian"),
        ModelName.WAVELET.value: ClassicalAdapter("wavelet"),
        ModelName.IDENTITY.value: ClassicalAdapter("identity"),
    }


def get_adapter(name: Union[str, ModelName]) -> ModelAdapter:
    """Retrieve an adapter instance by string name or ModelName enum from the registry."""
    registry = get_adapter_registry()
    if isinstance(name, ModelName):
        key = name.value.lower()
    else:
        key = str(name).lower().strip()

    if key in registry:
        return registry[key]
    raise ValueError(f"Unknown adapter '{name}'. Available: {list(registry.keys())}")


def list_available_adapters() -> List[str]:
    """List available adapter keys in the registry."""
    return list(get_adapter_registry().keys())


__all__ = [
    "ModelName",
    "ModelAdapter",
    "ModelMetadata",
    "DnCNNAdapter",
    "AMAT1Adapter",
    "ClassicalAdapter",
    "get_adapter_registry",
    "get_adapter",
    "list_available_adapters",
]
