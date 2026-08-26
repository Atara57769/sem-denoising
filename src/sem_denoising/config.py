"""
Configuration management for SEM Denoising Pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import yaml
import json


@dataclass
class DataConfig:
    data_root: str = r"C:\Users\USER\Desktop\workspace\AMAT_data"
    train_sets: List[int] = field(default_factory=lambda: [1, 2])
    val_sets: List[int] = field(default_factory=lambda: [3])
    test_set: int = 5
    patch_size: int = 64
    stride: int = 32


@dataclass
class NoiseConfig:
    gaussian: Dict[str, float] = field(default_factory=lambda: {"sigma": 0.10})
    poisson: Dict[str, float] = field(default_factory=lambda: {"peak": 50.0})
    mixed: Dict[str, float] = field(default_factory=lambda: {"sigma": 0.06, "peak": 50.0})


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 3
    lr: float = 1e-3
    seed: int = 42
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"


@dataclass
class EvaluationConfig:
    output_dir: str = "outputs"
    results_csv: str = "stronger_test_results.csv"
    plot_filename: str = "benchmark_comparison_fig.png"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "PipelineConfig":
        data_cfg = DataConfig(**data_dict.get("data", {}))
        noise_cfg = NoiseConfig(**data_dict.get("noise", {}))
        train_cfg = TrainingConfig(**data_dict.get("training", {}))
        eval_cfg = EvaluationConfig(**data_dict.get("evaluation", {}))
        return cls(data=data_cfg, noise=noise_cfg, training=train_cfg, evaluation=eval_cfg)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        return cls.from_dict(content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": {
                "data_root": self.data.data_root,
                "train_sets": self.data.train_sets,
                "val_sets": self.data.val_sets,
                "test_set": self.data.test_set,
                "patch_size": self.data.patch_size,
                "stride": self.data.stride,
            },
            "noise": {
                "gaussian": self.noise.gaussian,
                "poisson": self.noise.poisson,
                "mixed": self.noise.mixed,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "lr": self.training.lr,
                "seed": self.training.seed,
                "device": self.training.device,
                "checkpoint_dir": self.training.checkpoint_dir,
            },
            "evaluation": {
                "output_dir": self.evaluation.output_dir,
                "results_csv": self.evaluation.results_csv,
                "plot_filename": self.evaluation.plot_filename,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

