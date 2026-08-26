import os
import pytest
from sem_denoising.config import PipelineConfig, DataConfig, NoiseConfig, TrainingConfig, EvaluationConfig


def test_default_config():
    config = PipelineConfig()
    assert config.data.train_sets == [1, 2]
    assert config.data.val_sets == [3]
    assert config.data.test_set == 5
    assert config.training.batch_size == 32
    assert config.training.epochs == 3
    assert config.training.device == "cpu"


def test_config_from_dict():
    cfg_dict = {
        "data": {"patch_size": 128, "stride": 64},
        "training": {"epochs": 10, "lr": 5e-4},
    }
    config = PipelineConfig.from_dict(cfg_dict)
    assert config.data.patch_size == 128
    assert config.data.stride == 64
    assert config.training.epochs == 10
    assert config.training.lr == 5e-4


def test_config_from_yaml(tmp_path):
    yaml_content = """
data:
  data_root: "/test/path"
  train_sets: [1]
  val_sets: [2]
  test_set: 3
  patch_size: 32
  stride: 16
noise:
  gaussian:
    sigma: 0.15
training:
  epochs: 5
"""
    yaml_file = tmp_path / "test_cfg.yaml"
    yaml_file.write_text(yaml_content)

    config = PipelineConfig.from_yaml(str(yaml_file))
    assert config.data.data_root == "/test/path"
    assert config.data.patch_size == 32
    assert config.noise.gaussian["sigma"] == 0.15
    assert config.training.epochs == 5

