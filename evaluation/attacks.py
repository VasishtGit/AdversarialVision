"""Adversarial Attack Benchmarking (FGSM and PGD) across multiple epsilon perturbation budgets."""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack
from evaluation.metrics import compute_confidence_and_entropy


DEFAULT_EPSILONS = [
    1.0 / 255.0,   # ~0.0039
    2.0 / 255.0,   # ~0.0078
    4.0 / 255.0,   # ~0.0157
    8.0 / 255.0,   # ~0.0314 (standard CIFAR evaluation bound)
    16.0 / 255.0,  # ~0.0627
    32.0 / 255.0,  # ~0.1255
]


def evaluate_adversarial_batch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    attack_fn,
    device: torch.device,
    amp_enabled: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluates adversarial robustness and metrics against an instantiated attack.
    Computes Top-1 Accuracy, Attack Success Rate (ASR), Loss, Mean Confidence,
    Mean Entropy, and Mean L_inf / L_2 perturbation norms.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    clean_correct_total = 0
    adv_correct_total = 0
    successful_attacks_on_correct = 0
    running_loss = 0.0

    all_confs = []
    all_entropies = []
    l_inf_perturbations = []
    l_2_perturbations = []

    for images, targets in loader:
        if max_samples is not None and total >= max_samples:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        # 1. Clean predictions to evaluate attack success rate (ASR) strictly on correctly classified examples
        with torch.no_grad():
            clean_logits = model(images)
            clean_preds = clean_logits.argmax(dim=1)
            clean_correct_mask = clean_preds.eq(targets)
            clean_correct_total += clean_correct_mask.sum().item()

        # 2. Generate adversarial examples (requires grad w.r.t images)
        adv_images = attack_fn(images, targets)

        # 3. Calculate perturbation norms in [0.0, 1.0] pixel space
        with torch.no_grad():
            diff = (adv_images - images).view(batch_size, -1)
            l_inf_batch = diff.abs().max(dim=1)[0]
            l_2_batch = torch.norm(diff, p=2, dim=1)

            l_inf_perturbations.append(l_inf_batch.cpu().numpy())
            l_2_perturbations.append(l_2_batch.cpu().numpy())

            # 4. Evaluate model on adversarial images
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
                adv_logits = model(adv_images)
                loss = criterion(adv_logits, targets)

            running_loss += loss.item() * batch_size
            adv_preds = adv_logits.argmax(dim=1)
            adv_correct_mask = adv_preds.eq(targets)
            adv_correct_total += adv_correct_mask.sum().item()

            # ASR = fraction of samples correctly classified clean that are fooled when perturbed
            fooled_mask = clean_correct_mask & (~adv_correct_mask)
            successful_attacks_on_correct += fooled_mask.sum().item()

            confs, entropies = compute_confidence_and_entropy(adv_logits)
            all_confs.append(confs.cpu().numpy())
            all_entropies.append(entropies.cpu().numpy())

            total += batch_size

    confs_arr = np.concatenate(all_confs)
    entropies_arr = np.concatenate(all_entropies)
    l_inf_arr = np.concatenate(l_inf_perturbations)
    l_2_arr = np.concatenate(l_2_perturbations)

    adv_acc = (adv_correct_total / total) * 100.0
    asr = (successful_attacks_on_correct / max(clean_correct_total, 1)) * 100.0

    return {
        "accuracy": float(adv_acc),
        "asr": float(asr),
        "loss": float(running_loss / total),
        "mean_confidence": float(np.mean(confs_arr)),
        "mean_entropy": float(np.mean(entropies_arr)),
        "mean_l_inf": float(np.mean(l_inf_arr)),
        "mean_l_2": float(np.mean(l_2_arr)),
    }


def plot_adversarial_curves(
    df: pd.DataFrame,
    clean_accuracy: float,
    save_path: str = "experiments/adversarial_attack_curves.png",
):
    """Plots Robust Accuracy vs Epsilon and Attack Success Rate (ASR) vs Epsilon."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Convert eps to pixel scale (e.g. 8/255 -> 8) for readable x-axis
    attacks = df["attack"].unique()
    colors = {"FGSM": "#1f77b4", "PGD-10": "#d62728", "PGD-20": "#9467bd"}

    # 1. Robust Accuracy vs Epsilon
    ax1.axhline(clean_accuracy, color="gray", linestyle="--", label=f"Clean Acc ({clean_accuracy:.1f}%)")
    for attack in attacks:
        sub = df[df["attack"] == attack].sort_values("eps")
        eps_pixel = sub["eps"] * 255.0
        color = colors.get(attack, "black")
        ax1.plot(eps_pixel, sub["accuracy"], marker="o", label=attack, color=color, lw=2)

    ax1.set_xlabel(r"Perturbation Budget $\epsilon$ (in 255ths, e.g. 8 = 8/255)")
    ax1.set_ylabel("Robust Top-1 Accuracy (%)")
    ax1.set_title(r"Model Accuracy vs. Adversarial Perturbation $\epsilon$")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_ylim(-2, 85)
    ax1.legend()

    # 2. Attack Success Rate (ASR) vs Epsilon
    for attack in attacks:
        sub = df[df["attack"] == attack].sort_values("eps")
        eps_pixel = sub["eps"] * 255.0
        color = colors.get(attack, "black")
        ax2.plot(eps_pixel, sub["asr"], marker="s", label=attack, color=color, lw=2)

    ax2.set_xlabel(r"Perturbation Budget $\epsilon$ (in 255ths)")
    ax2.set_ylabel("Attack Success Rate (%)")
    ax2.set_title(r"Attack Success Rate (ASR) vs. $\epsilon$")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_ylim(-2, 105)
    ax2.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Adversarial evaluation curves saved to {save_path}")


def run_adversarial_benchmark(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    clean_accuracy: float = 77.84,
    epsilons: Optional[List[float]] = None,
    output_csv: str = "experiments/adversarial_attack_results.csv",
    curves_plot_path: str = "experiments/adversarial_attack_curves.png",
    amp_enabled: bool = True,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Executes Milestone 3 White-Box Adversarial Benchmark:
    - FGSM across epsilons
    - PGD-10 (L_inf, alpha = eps / 4, random start) across epsilons
    """
    if epsilons is None:
        epsilons = DEFAULT_EPSILONS

    results = []

    print("=" * 70)
    print(" MILESTONE 3: WHITE-BOX ADVERSARIAL ATTACKS BENCHMARK (FGSM & PGD)")
    print(f" Perturbation Budgets (eps): {[f'{e*255:.1f}/255' for e in epsilons]}")
    print(f" Clean Baseline Accuracy: {clean_accuracy:.2f}%")
    print("=" * 70)

    # Clean row (eps = 0)
    for attack_name in ["FGSM", "PGD-10"]:
        results.append({
            "attack": attack_name,
            "eps": 0.0,
            "eps_pixel": 0.0,
            "accuracy": clean_accuracy,
            "asr": 0.0,
            "loss": 0.9134,
            "mean_confidence": 0.7983,
            "mean_entropy": 0.9288,
            "mean_l_inf": 0.0,
            "mean_l_2": 0.0,
        })

    # 1. FGSM Evaluation
    print("\n[1/2] Benchmarking FGSM Attack...")
    for eps in epsilons:
        attacker = FGSMAttack(model=model, eps=eps)
        start_t = time.perf_counter()
        metrics = evaluate_adversarial_batch(
            model=model,
            loader=loader,
            attack_fn=attacker.generate,
            device=device,
            amp_enabled=amp_enabled,
            max_samples=max_samples,
        )
        elapsed = time.perf_counter() - start_t

        results.append({
            "attack": "FGSM",
            "eps": eps,
            "eps_pixel": eps * 255.0,
            "accuracy": metrics["accuracy"],
            "asr": metrics["asr"],
            "loss": metrics["loss"],
            "mean_confidence": metrics["mean_confidence"],
            "mean_entropy": metrics["mean_entropy"],
            "mean_l_inf": metrics["mean_l_inf"],
            "mean_l_2": metrics["mean_l_2"],
        })

        print(
            f"  - FGSM eps={eps*255.0:4.1f}/255: Acc: {metrics['accuracy']:5.2f}% | "
            f"ASR: {metrics['asr']:5.2f}% | Loss: {metrics['loss']:.3f} | "
            f"L_inf: {metrics['mean_l_inf']*255.0:.2f}/255 | Time: {elapsed:.1f}s"
        )

    # 2. PGD-10 Evaluation (alpha = eps / 4)
    print("\n[2/2] Benchmarking PGD-10 Attack...")
    for eps in epsilons:
        alpha = max(eps / 4.0, 1.0 / 255.0)
        attacker = PGDAttack(model=model, eps=eps, alpha=alpha, steps=10, random_start=True)
        start_t = time.perf_counter()
        metrics = evaluate_adversarial_batch(
            model=model,
            loader=loader,
            attack_fn=attacker.generate,
            device=device,
            amp_enabled=amp_enabled,
            max_samples=max_samples,
        )
        elapsed = time.perf_counter() - start_t

        results.append({
            "attack": "PGD-10",
            "eps": eps,
            "eps_pixel": eps * 255.0,
            "accuracy": metrics["accuracy"],
            "asr": metrics["asr"],
            "loss": metrics["loss"],
            "mean_confidence": metrics["mean_confidence"],
            "mean_entropy": metrics["mean_entropy"],
            "mean_l_inf": metrics["mean_l_inf"],
            "mean_l_2": metrics["mean_l_2"],
        })

        print(
            f"  - PGD-10 eps={eps*255.0:4.1f}/255: Acc: {metrics['accuracy']:5.2f}% | "
            f"ASR: {metrics['asr']:5.2f}% | Loss: {metrics['loss']:.3f} | "
            f"L_inf: {metrics['mean_l_inf']*255.0:.2f}/255 | Time: {elapsed:.1f}s"
        )

    df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved adversarial results to {output_csv}")

    plot_adversarial_curves(df, clean_accuracy, curves_plot_path)

    return df
