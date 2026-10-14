"""Preprocessing Defenses Benchmark against Clean, Natural Corruptions, and Adversarial Attacks."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from defenses.preprocessing import PreprocessDefense, DEFENSES_REGISTRY
from corruptions import corrupt
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack


DEFENSE_CONFIGS = {
    "None (No Defense)": None,
    "Gaussian Blur (k=3, s=0.5)": {"defense_type": "gaussian", "ksize": 3, "sigma": 0.5},
    "Median Filter (k=3)": {"defense_type": "median", "ksize": 3},
    "Bilateral Filter (d=3)": {"defense_type": "bilateral", "d": 3, "sigma_color": 0.1, "sigma_space": 3.0},
    "JPEG Compression (Q=75)": {"defense_type": "jpeg", "quality": 75},
}


def evaluate_batch_with_defense(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    defense: Optional[PreprocessDefense] = None,
    amp_enabled: bool = True,
) -> int:
    """Passes images through optional defense filter, then through model."""
    if defense is not None:
        images = defense(images)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and images.device.type == "cuda"):
            logits = model(images)
        preds = logits.argmax(dim=1)
        correct = preds.eq(targets).sum().item()
    return correct


def plot_defense_benchmark_comparison(
    df: pd.DataFrame,
    save_path: str = "experiments/defense_mitigation_benchmark.png",
):
    """
    Plots grouped bar chart of defense performance across regimes:
    - Clean Test Accuracy
    - Salt & Pepper Noise (Sev 3)
    - Gaussian Noise (Sev 3)
    - FGSM (eps=8/255)
    - PGD-10 (eps=8/255)
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    conditions = df["condition"].unique()
    defenses = df["defense"].unique()

    n_cond = len(conditions)
    n_def = len(defenses)
    width = 0.8 / n_def

    x = np.arange(n_cond)

    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    for i, def_name in enumerate(defenses):
        sub = df[df["defense"] == def_name]
        accs = [sub[sub["condition"] == c]["accuracy"].values[0] for c in conditions]
        offset = (i - n_def / 2 + 0.5) * width
        rects = ax.bar(x + offset, accs, width, label=def_name, color=colors[i % len(colors)], alpha=0.9)

        # Value annotations
        for rect in rects:
            height = rect.get_height()
            if height > 2.0:
                ax.annotate(f"{height:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("Efficacy of Preprocessing Defenses across Clean, Natural Corruptions, and Adversarial Attacks", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(0, 90)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Defense benchmark plot saved to {save_path}")


def run_defense_benchmark(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_eval_samples: int = 1000,
    output_csv: str = "experiments/defense_mitigation_results.csv",
    plot_path: str = "experiments/defense_mitigation_benchmark.png",
    amp_enabled: bool = True,
) -> pd.DataFrame:
    """
    Executes Milestone 5 Preprocessing Defense Benchmark:
    Evaluates 5 defenses (None, Gaussian, Median, Bilateral, JPEG) across 5 test conditions:
    1. Clean Test
    2. Salt & Pepper Noise (Sev 3)
    3. Gaussian Noise (Sev 3)
    4. FGSM (eps=8/255)
    5. PGD-10 (eps=8/255)
    """
    model.eval()

    print("=" * 70)
    print(" MILESTONE 5: INFERENCE-TIME PREPROCESSING DEFENSES BENCHMARK")
    print(f" Evaluating {num_eval_samples} samples across {len(DEFENSE_CONFIGS)} defenses...")
    print("=" * 70)

    # Attackers
    fgsm = FGSMAttack(model=model, eps=8.0 / 255.0)
    pgd = PGDAttack(model=model, eps=8.0 / 255.0, alpha=2.0 / 255.0, steps=10, random_start=True)

    conditions = [
        "Clean",
        "Salt & Pepper (Sev 3)",
        "Gaussian Noise (Sev 3)",
        "FGSM (eps=8/255)",
        "PGD-10 (eps=8/255)",
    ]

    # Pre-generate perturbed batches
    cached_batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]] = []
    total_samples = 0

    print("Generating test regimes...")
    for images, targets in loader:
        if total_samples >= num_eval_samples:
            break

        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        with torch.no_grad():
            img_clean = images.clone()
            img_sp = corrupt(images.cpu(), "salt_and_pepper_noise", severity=3).to(device)
            img_gn = corrupt(images.cpu(), "gaussian_noise", severity=3).to(device)

        img_fgsm = fgsm.generate(images, targets)
        img_pgd = pgd.generate(images, targets)

        batch_dict = {
            "Clean": img_clean,
            "Salt & Pepper (Sev 3)": img_sp,
            "Gaussian Noise (Sev 3)": img_gn,
            "FGSM (eps=8/255)": img_fgsm,
            "PGD-10 (eps=8/255)": img_pgd,
        }

        cached_batches.append((batch_dict, targets))
        total_samples += batch_size

    print(f"Loaded and perturbed {total_samples} test samples.")

    # Evaluate each defense
    results = []

    for def_name, def_kwargs in DEFENSE_CONFIGS.items():
        print(f"\nEvaluating Defense: '{def_name}'...")
        defense_obj = PreprocessDefense(**def_kwargs) if def_kwargs is not None else None

        for cond in conditions:
            correct = 0
            for batch_dict, targets in cached_batches:
                imgs = batch_dict[cond]
                correct += evaluate_batch_with_defense(
                    model=model,
                    images=imgs,
                    targets=targets,
                    defense=defense_obj,
                    amp_enabled=amp_enabled,
                )

            acc = (correct / total_samples) * 100.0
            print(f"  - Condition: {cond:<22} | Top-1 Acc: {acc:.2f}%")

            results.append({
                "defense": def_name,
                "condition": cond,
                "accuracy": acc,
                "num_samples": total_samples,
            })

    df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved defense results to {output_csv}")

    plot_defense_benchmark_comparison(df, save_path=plot_path)

    return df
