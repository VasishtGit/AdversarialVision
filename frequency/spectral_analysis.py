"""Frequency Domain Spectral Analysis (2D FFT and 2D DCT) for Clean vs Corrupted vs Adversarial Images."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from frequency.fft import compute_fft2d, extract_fft_features
from frequency.dct import compute_dct2d, extract_dct_features
from corruptions import corrupt
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack


def compute_radial_profile(psd2d: np.ndarray) -> np.ndarray:
    """
    Computes azimuthal average of a 2D centered power spectral density to obtain
    a 1D radial frequency energy profile.
    """
    h, w = psd2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.round(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(int)

    max_r = min(cy, cx)
    radial_mean = np.zeros(max_r + 1, dtype=np.float64)
    for radius in range(max_r + 1):
        mask = (r == radius)
        if mask.any():
            radial_mean[radius] = psd2d[mask].mean()
        else:
            radial_mean[radius] = 0.0
    return radial_mean


def plot_frequency_analysis_comparison(
    sample_images: Dict[str, np.ndarray],
    radial_profiles: Dict[str, np.ndarray],
    save_path: str = "experiments/frequency_spectrum_analysis.png",
):
    """
    Creates a comprehensive 4-panel visual comparison:
    1. Spatial domain images (Clean, Gaussian Noise, Motion Blur, FGSM, PGD-10)
    2. Perturbation differences (|x_perturbed - x_clean|) magnified 10x
    3. 2D FFT Log Magnitude Spectra
    4. 1D Azimuthal Radial Power Profiles comparing low vs high frequency distribution
    """
    conditions = list(sample_images.keys())
    n_cond = len(conditions)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, n_cond, height_ratios=[1, 1, 1.2])

    clean_img = sample_images["Clean"]

    # Row 1: Spatial Images
    for i, cond in enumerate(conditions):
        ax = fig.add_subplot(gs[0, i])
        img = sample_images[cond]
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.axis("off")

    # Row 2: 2D FFT Log Power Spectrum
    for i, cond in enumerate(conditions):
        ax = fig.add_subplot(gs[1, i])
        img = sample_images[cond]
        _, mag = compute_fft2d(img)
        log_mag = np.log1p(mag)
        im = ax.imshow(log_mag, cmap="inferno")
        ax.set_title(f"FFT: {cond}", fontsize=10)
        ax.axis("off")

    # Row 3: 1D Radial Power Spectral Distribution Curves across conditions
    ax_curve = fig.add_subplot(gs[2, :])
    colors = {
        "Clean": "#2ca02c",
        "Gaussian Noise (Sev 3)": "#ff7f0e",
        "Motion Blur (Sev 3)": "#1f77b4",
        "FGSM (eps=8/255)": "#9467bd",
        "PGD-10 (eps=8/255)": "#d62728",
    }

    for cond in conditions:
        profile = radial_profiles[cond]
        freq_bins = np.arange(len(profile))
        # Normalize profile by sum of energy for shape comparison
        norm_profile = profile / (profile.sum() + 1e-12)
        color = colors.get(cond, None)
        ax_curve.plot(freq_bins, norm_profile, label=cond, lw=2.2, color=color, marker="o" if "PGD" in cond else None, markersize=4)

    ax_curve.set_xlabel("Radial Spatial Frequency (Radius from DC / Low to High Frequency)", fontsize=11)
    ax_curve.set_ylabel("Normalized Spectral Energy Density", fontsize=11)
    ax_curve.set_title("Spectral Energy Distribution: Low Frequency (Left) vs High Frequency (Right)", fontsize=12, fontweight="bold")
    ax_curve.grid(True, linestyle="--", alpha=0.6)
    ax_curve.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Frequency spectrum analysis plot saved to {save_path}")


def run_frequency_benchmark(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_eval_samples: int = 500,
    output_csv: str = "experiments/frequency_analysis_summary.csv",
    plot_path: str = "experiments/frequency_spectrum_analysis.png",
) -> pd.DataFrame:
    """
    Executes Milestone 4 Frequency Domain Benchmark:
    - Analyzes spectral distributions of Clean, Gaussian Noise, Motion Blur, FGSM, and PGD-10
    - Calculates average Low-Frequency Energy, High-Frequency Energy, High/Low Ratio, and Spectral Entropy
    - Generates 2D FFT & 1D Radial Spectral Distribution comparison plots
    """
    model.eval()

    print("=" * 70)
    print(" MILESTONE 4: FREQUENCY DOMAIN (2D FFT & DCT) SPECTRAL BENCHMARK")
    print(f" Evaluating {num_eval_samples} test samples across spectral regimes...")
    print("=" * 70)

    # Prepare attackers
    fgsm = FGSMAttack(model=model, eps=8.0 / 255.0)
    pgd = PGDAttack(model=model, eps=8.0 / 255.0, alpha=2.0 / 255.0, steps=10, random_start=True)

    conditions = ["Clean", "Gaussian Noise (Sev 3)", "Motion Blur (Sev 3)", "FGSM (eps=8/255)", "PGD-10 (eps=8/255)"]
    metrics_records: Dict[str, List[Dict[str, float]]] = {c: [] for c in conditions}
    accumulated_psd: Dict[str, np.ndarray] = {}

    processed_samples = 0
    representative_samples: Dict[str, np.ndarray] = {}

    for images, targets in loader:
        if processed_samples >= num_eval_samples:
            break

        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Generate condition batches
        with torch.no_grad():
            img_clean = images.clone()
            img_gnoise = corrupt(images.cpu(), "gaussian_noise", severity=3).to(device)
            img_mblur = corrupt(images.cpu(), "motion_blur", severity=3).to(device)

        img_fgsm = fgsm.generate(images, targets)
        img_pgd = pgd.generate(images, targets)

        batches = {
            "Clean": img_clean,
            "Gaussian Noise (Sev 3)": img_gnoise,
            "Motion Blur (Sev 3)": img_mblur,
            "FGSM (eps=8/255)": img_fgsm,
            "PGD-10 (eps=8/255)": img_pgd,
        }

        # Store 1 sample for visualization
        if not representative_samples:
            for cond, b in batches.items():
                img_perm = b[0].detach().cpu().permute(1, 2, 0).numpy()
                representative_samples[cond] = np.clip(img_perm, 0.0, 1.0)

        # Compute FFT and DCT metrics for each sample in batch
        for cond, b in batches.items():
            b_np = b.detach().cpu().numpy()
            for idx in range(min(batch_size, num_eval_samples - processed_samples)):
                sample = b_np[idx]  # (3, 32, 32)
                fft_feats = extract_fft_features(sample, cutoff_radius_ratio=0.35)
                dct_feats = extract_dct_features(sample, cutoff_ratio=0.35)

                _, mag = compute_fft2d(sample)
                psd = mag ** 2

                if cond not in accumulated_psd:
                    accumulated_psd[cond] = np.zeros_like(psd, dtype=np.float64)
                accumulated_psd[cond] += psd

                metrics_records[cond].append({
                    "condition": cond,
                    **fft_feats,
                    **dct_feats,
                })

        processed_samples += batch_size

    # Average PSD to compute radial profiles
    radial_profiles = {}
    for cond in conditions:
        mean_psd = accumulated_psd[cond] / processed_samples
        radial_profiles[cond] = compute_radial_profile(mean_psd)

    # Summarize metrics
    summary_rows = []
    for cond in conditions:
        df_cond = pd.DataFrame(metrics_records[cond])
        summary_rows.append({
            "condition": cond,
            "fft_high_low_ratio_mean": float(df_cond["fft_high_low_ratio"].mean()),
            "fft_spectral_entropy_mean": float(df_cond["fft_spectral_entropy"].mean()),
            "dct_high_low_ratio_mean": float(df_cond["dct_high_low_ratio"].mean()),
            "dct_spectral_entropy_mean": float(df_cond["dct_spectral_entropy"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"\nFrequency analysis summary saved to {output_csv}")

    # Generate multi-panel comparison plot
    plot_frequency_analysis_comparison(representative_samples, radial_profiles, save_path=plot_path)

    return summary_df
