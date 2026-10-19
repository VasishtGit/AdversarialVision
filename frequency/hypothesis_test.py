"""Hypothesis-Driven Frequency Analysis:
1. Isolate spectral power of adversarial perturbations delta = x_adv - x_clean vs natural noise delta = x_corrupt - x_clean.
2. Fourier Band-Filtering Probe: Evaluate model accuracy when stripping high frequencies vs low frequencies at varying radii.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from frequency.fft import compute_fft2d
from frequency.spectral_analysis import compute_radial_profile
from corruptions import corrupt
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack


def fourier_filter_batch(images: torch.Tensor, cutoff_radius: float, filter_type: str = "lowpass") -> torch.Tensor:
    """
    Applies ideal circular 2D Fourier lowpass or highpass filter to a batch of images in [0, 1].
    Preserves channel count and spatial dimensions.
    """
    b, c, h, w = images.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if filter_type == "lowpass":
        mask = (dist <= cutoff_radius).astype(np.float32)
    elif filter_type == "highpass":
        mask = (dist > cutoff_radius).astype(np.float32)
    else:
        raise ValueError(f"Unknown filter type {filter_type}")

    filtered_batch = []
    images_np = images.detach().cpu().numpy()

    for i in range(b):
        channels = []
        for ch in range(c):
            img_ch = images_np[i, ch]
            f = np.fft.fft2(img_ch)
            fshift = np.fft.fftshift(f)
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            channels.append(np.real(img_back))
        filtered_batch.append(np.stack(channels))

    filtered_np = np.stack(filtered_batch)
    filtered_np = np.clip(filtered_np, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(filtered_np).to(images.device)


def run_frequency_hypothesis_benchmark(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    clean_accuracy: float = 77.84,
    num_samples: int = 1000,
    output_dir: str = "experiments",
    amp_enabled: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Experiment A: Isolated Perturbation Spectral Energy Density (delta = x_perturbed - x_clean)
    Experiment B: Fourier Filtering Accuracy Probe (Model reliance on high vs low frequencies)
    """
    model.eval()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" EXPERIMENT A: ISOLATED PERTURBATION SPECTRAL DECOMPOSITION (delta)")
    print("=" * 70)

    fgsm = FGSMAttack(model=model, eps=8.0 / 255.0)
    pgd = PGDAttack(model=model, eps=8.0 / 255.0, alpha=2.0 / 255.0, steps=10, random_start=True)

    perturbation_names = ["Gaussian Noise", "Salt & Pepper", "FGSM (eps=8/255)", "PGD-10 (eps=8/255)"]
    accumulated_delta_psd: Dict[str, np.ndarray] = {}
    total_processed = 0

    cached_clean_batches = []

    for images, targets in loader:
        if total_processed >= num_samples:
            break

        images = images.to(device)
        targets = targets.to(device)
        b_size = images.size(0)

        with torch.no_grad():
            img_clean = images.clone()
            img_gn = corrupt(images.cpu(), "gaussian_noise", severity=3).to(device)
            img_sp = corrupt(images.cpu(), "salt_and_pepper_noise", severity=3).to(device)

        img_fgsm = fgsm.generate(images, targets)
        img_pgd = pgd.generate(images, targets)

        cached_clean_batches.append((images.clone(), targets.clone()))

        # Compute isolated deltas: delta = perturbed - clean
        deltas = {
            "Gaussian Noise": (img_gn - img_clean).detach().cpu().numpy(),
            "Salt & Pepper": (img_sp - img_clean).detach().cpu().numpy(),
            "FGSM (eps=8/255)": (img_fgsm - img_clean).detach().cpu().numpy(),
            "PGD-10 (eps=8/255)": (img_pgd - img_clean).detach().cpu().numpy(),
        }

        for name, delta_batch in deltas.items():
            for i in range(min(b_size, num_samples - total_processed)):
                delta_img = delta_batch[i].mean(axis=0)  # Average across color channels
                fshift, mag = compute_fft2d(delta_img)
                psd = mag ** 2

                if name not in accumulated_delta_psd:
                    accumulated_delta_psd[name] = np.zeros_like(psd, dtype=np.float64)
                accumulated_delta_psd[name] += psd

        total_processed += b_size

    # Compute radial energy profiles for deltas
    radial_profiles_delta = {}
    delta_energy_distribution = []

    for name in perturbation_names:
        mean_psd = accumulated_delta_psd[name] / total_processed
        rad_profile = compute_radial_profile(mean_psd)
        radial_profiles_delta[name] = rad_profile

        # Compute low (r <= 4) vs high (r > 4) frequency energy fraction
        total_energy = rad_profile.sum() + 1e-12
        low_energy = rad_profile[:5].sum()
        high_energy = rad_profile[5:].sum()

        delta_energy_distribution.append({
            "perturbation": name,
            "low_freq_fraction (%)": float((low_energy / total_energy) * 100.0),
            "high_freq_fraction (%)": float((high_energy / total_energy) * 100.0),
            "high_to_low_ratio": float(high_energy / (low_energy + 1e-12)),
        })

    df_delta = pd.DataFrame(delta_energy_distribution)
    df_delta.to_csv(f"{output_dir}/isolated_perturbation_spectral_analysis.csv", index=False)
    print("\nIsolated Perturbation Spectral Distribution:")
    print(df_delta.to_string(index=False))

    # Plot Isolated Delta Spectra
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Gaussian Noise": "#ff7f0e", "Salt & Pepper": "#8c564b", "FGSM (eps=8/255)": "#1f77b4", "PGD-10 (eps=8/255)": "#d62728"}
    for name in perturbation_names:
        prof = radial_profiles_delta[name]
        norm_prof = prof / (prof.sum() + 1e-12)
        ax.plot(np.arange(len(prof)), norm_prof, label=name, color=colors[name], lw=2.2, marker="o" if "PGD" in name else None, markersize=4)

    ax.axvline(4.5, color="black", linestyle="--", alpha=0.7, label="Low / High Cutoff (r=4.5)")
    ax.set_xlabel("Radial Frequency Radius (0=DC / Low Frequency to 16=Nyquist / High Frequency)", fontsize=11)
    ax.set_ylabel("Normalized Energy Fraction", fontsize=11)
    ax.set_title("Isolated Adversarial & Corruption Perturbation Energy (delta = x_adv - x_clean)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/isolated_perturbation_spectrum.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir}/isolated_perturbation_spectrum.png")

    # =========================================================================
    # EXPERIMENT B: FOURIER FILTERING ACCURACY PROBE
    # =========================================================================
    print("\n" + "=" * 70)
    print(" EXPERIMENT B: FOURIER FILTERING PROBE (MODEL FREQUENCY RELIANCE)")
    print("=" * 70)

    radii = [2, 4, 6, 8, 10, 12, 14, 16]
    probe_results = []

    for r in radii:
        # 1. Lowpass test (only low frequencies below radius r preserved)
        correct_lp = 0
        correct_hp = 0
        total_eval = 0

        for imgs, tgts in cached_clean_batches:
            imgs_lp = fourier_filter_batch(imgs, cutoff_radius=r, filter_type="lowpass")
            imgs_hp = fourier_filter_batch(imgs, cutoff_radius=r, filter_type="highpass")

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
                    out_lp = model(imgs_lp)
                    out_hp = model(imgs_hp)

                correct_lp += out_lp.argmax(dim=1).eq(tgts).sum().item()
                correct_hp += out_hp.argmax(dim=1).eq(tgts).sum().item()
                total_eval += imgs.size(0)

        acc_lp = (correct_lp / total_eval) * 100.0
        acc_hp = (correct_hp / total_eval) * 100.0

        probe_results.append({
            "cutoff_radius": r,
            "lowpass_accuracy (%)": acc_lp,
            "highpass_accuracy (%)": acc_hp,
        })
        print(f"Radius {r:2d} | Lowpass Acc (Retaining Low Freq <= {r:2d}): {acc_lp:5.2f}% | Highpass Acc (Retaining High Freq > {r:2d}): {acc_hp:5.2f}%")

    df_probe = pd.DataFrame(probe_results)
    df_probe.to_csv(f"{output_dir}/fourier_filtering_probe_results.csv", index=False)

    # Plot Frequency Probe
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_probe["cutoff_radius"], df_probe["lowpass_accuracy (%)"], marker="o", lw=2.2, color="#1f77b4", label="Lowpass Filter (Keeping frequencies <= r)")
    ax.plot(df_probe["cutoff_radius"], df_probe["highpass_accuracy (%)"], marker="s", lw=2.2, color="#d62728", label="Highpass Filter (Keeping frequencies > r)")
    ax.axhline(clean_accuracy, color="gray", linestyle="--", label=f"Unfiltered Clean Accuracy ({clean_accuracy:.2f}%)")

    ax.set_xlabel("Filter Cutoff Radius (r)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("Fourier Probe: Accuracy under Low-Pass vs High-Pass Filtering", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(-2, 85)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fourier_probe_accuracy.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir}/fourier_probe_accuracy.png")

    return df_delta, df_probe
