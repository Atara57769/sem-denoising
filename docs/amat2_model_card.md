# AMAT-2 Model Card: TinyDnCNN D8/W48 (FP16)

## Model Overview
- **Model Identity**: `tiny_dncnn_d8w48`
- **Architecture**: 8-Layer Denoising Convolutional Neural Network (DnCNN D8/W48)
- **Parameters**: **125,905** (~125.9K trainable parameters)
- **Precision**: FP32 training $\rightarrow$ **FP16 whole-image inference**
- **Checkpoint Location**: `checkpoints/checkpoint_tinydncnn_d8w48.pth`
- **Status**: **FROZEN** engineering candidate for AMAT-2

---

## Technical Specifications
| Attribute | Specification |
| :--- | :--- |
| **Depth ($D$)** | 8 Convolutional Layers |
| **Width ($W$)** | 48 Feature Channels per layer |
| **Activation** | ReLU (inplace) |
| **Normalization** | BatchNorm2d on intermediate layers (2–7) |
| **Formulation** | Residual learning ($\hat{x}_{clean} = x_{noisy} - \mathcal{N}(x_{noisy})$) |
| **Input Contract** | Single-channel 2D grayscale float32 $[0.0, 1.0]$ |
| **Output Contract** | Single-channel 2D grayscale float32 clipped $[0.0, 1.0]$ |
| **Whole-Image Policy** | Direct forward pass on full $H \times W$ images (no tiling / no patch splitting) |

---

## Cost & Hardware Footprint (NVIDIA T4 Baseline)
| Metric | Value | Status |
| :--- | :--- | :--- |
| **Model Size (FP32)** | 503,620 bytes (~0.50 MB) | Verified |
| **Model Size (FP16)** | 251,810 bytes (~0.25 MB) | Verified |
| **MACs (512×512 Image)** | 3.30 GigaMACs | Computed |
| **T4 Latency (p50)** | 1.45 ms | Measured |
| **T4 Latency (p95)** | 1.82 ms | Measured |
| **Throughput** | ~689 images/sec | Measured |

---

## Degradation & Robustness Assumptions
1. **In-Distribution Efficacy**: High performance on Gaussian ($\sigma=0.10$), Poisson ($\text{peak}=50$), and Mixed Poisson-Gaussian noise.
2. **Correlated Artifact Handling**:
   - **Striping Artifacts**: Retains performance (survives).
   - **Spatial Blur**: Retains edge sharpness (survives).
   - **Scan Drift / Shear**: Performance reverses (known weakness - global context failure).
   - **Mixed Correlated Degradation**: Performance reverses (known weakness).

---

## Reproduction & Evaluation Commands
```bash
# Exact Inference Command
python -m sem_denoising.cli_benchmark --config configs/amat2_frozen_d8w48.yaml --model tiny_dncnn_d8w48 --precision fp16

# Exact Full Benchmark & Evaluation Command
python main_benchmark.py --config configs/amat2_frozen_d8w48.yaml
```
