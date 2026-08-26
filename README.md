# SEM Denoising Pipeline

A clean, modular, research-ready Python library and benchmarking suite for Scanning Electron Microscope (SEM) image denoising across classical filters and learned deep convolutional neural networks.

## Features

- **Specimen-Safe Data Pipeline**: Clean NIST SEM dataset loader with independent train (Sets 1 & 2), validation (Set 3), and test (Set 5) splits.
- **Physical Noise Modeling**: Configurable Gaussian (readout), Poisson (electron shot), and mixed Poisson-Gaussian degradation regimes.
- **Classical Baselines**:
  - Identity (No-op)
  - Spatial Gaussian Smoothing Filter
  - Non-Local Means (NLM)
  - Wavelet BayesShrink Soft-Thresholding
- **Learned Neural Architecture**:
  - `DnCNN`: Single unified convolutional neural network architecture supporting both direct mapping and residual learning via a `residual: bool` flag:
    - `direct_cnn`: 5 layers, 32 channels, direct mapping (`residual=False`, `use_bn=False`, LeakyReLU)
    - `small_dncnn`: 5 layers, 32 channels, residual learning (`residual=True`, `use_bn=True`, LeakyReLU)
    - `strong_dncnn`: 17 layers, 64 channels, residual learning (`residual=True`, `use_bn=True`, ReLU)
- **Extensible Architecture**: Registry/factory patterns for dynamic instantiation of models and noise models.
- **Unified Evaluation Suite**: Evaluates MSE, PSNR, SSIM, and exact CPU execution latency.
- **Reproducibility & Verification**: Full checkpoint save/reload validation asserting $\Delta = 0.0$.
- **CLI & Visualization**: Command-line interface for training, evaluation, benchmarking, and multi-metric plotting.

## Project Structure

```
project/
├── configs/                  # YAML experiment configurations
│   ├── default_config.yaml
│   └── fast_test_config.yaml
├── src/
│   └── sem_denoising/
│       ├── config.py         # Dataclass configs & YAML loaders
│       ├── noise.py          # Consolidated noise models
│       ├── metrics.py        # Evaluator (MSE, PSNR, SSIM, Timer)
│       ├── data/             # Dataset, patch extraction, image loader
│       ├── models/           # Classical baselines, unified DnCNN & registry
│       ├── training/         # Unified trainer & checkpoint manager
│       ├── experiments/      # Benchmarking runner & visualizer
│       ├── cli_train.py      # Training CLI module
│       └── cli_benchmark.py  # Benchmarking CLI module
├── tests/                    # Comprehensive unit tests
├── main_train.py             # Training runner script
└── main_benchmark.py         # Benchmarking runner script
```

## Quickstart

### Installation
```bash
pip install -e .
```

### CLI Usage

1. **Train All Models**:
   ```bash
   python main_train.py --config configs/default_config.yaml
   ```

2. **Benchmark Test Set (Set 5)**:
   ```bash
   python main_benchmark.py --config configs/default_config.yaml
   ```

3. **Run Unit Tests**:
   ```bash
   pytest tests/ -v
   ```

