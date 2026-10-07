# AdversarialVision: Vision Model Robustness Evaluation & Defense Framework

An empirical research framework evaluating computer vision models against adversarial perturbations and natural image corruptions, analyzing frequency domain signatures (FFT/DCT), and testing lightweight defenses and detection mechanisms.

---

## Hardware & Environment Baseline

As defined in Phase 1 of the implementation roadmap:

| Component | Specification |
|---|---|
| **Operating System** | Windows 11 (pwsh) |
| **Python Version** | `3.11.9` |
| **PyTorch Version** | `2.11.0+cu128` |
| **Torchvision Version** | `0.26.0+cu128` |
| **CUDA Driver / Runtime** | `12.8` |
| **GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| **Total VRAM** | `6.0 GB` |
| **Core Dependencies** | OpenCV (`opencv-python-headless`), NumPy, Pandas, Matplotlib, scikit-learn, tqdm, PyYAML |

---

## Architecture & Roadmap Overview

- **Dataset:** CIFAR-100 (Stage 1 prototyping) $\to$ ImageNet-1k (Stage 2 cloud evaluation)
- **Model:** CIFAR-adapted ResNet-18 ($3 \times 3$ stride 1 conv1, no initial maxpool, internal differentiable normalization)
- **Threat Models:**
  - Natural Corruptions: 10 types across 5 severities (Noise, Blur, Compression, Lighting, Geometry)
  - White-Box Adversarial: FGSM, PGD ($L_\infty$)
- **Defenses & Detection:**
  - OpenCV Preprocessing (Gaussian, Median, Bilateral, JPEG recompression)
  - PGD Adversarial Training
  - Spectral & activation feature-based Adversarial Detector
- **Analysis:** 2D FFT & DCT frequency domain energy and spectral entropy profiling
