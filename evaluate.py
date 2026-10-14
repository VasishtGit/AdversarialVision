"""Evaluation entry point for baseline, corruptions, and adversarial benchmarking."""

import argparse
import yaml
from pathlib import Path
import numpy as np
import torch

from data.cifar100 import get_cifar100_loaders
from models.resnet_cifar import resnet18_cifar
from evaluation.clean import evaluate_clean, save_baseline_results
from evaluation.corruptions import run_natural_corruption_benchmark
from evaluation.attacks import run_adversarial_benchmark
from evaluation.defenses import run_defense_benchmark
from frequency.spectral_analysis import run_frequency_benchmark
from visualization.plots import plot_confusion_matrix, plot_class_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate vision model robustness")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/resnet18_cifar100_best.pth", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="clean",
        choices=["clean", "corruptions", "attacks", "frequency", "defenses", "all"],
        help="Benchmark mode: 'clean', 'corruptions', 'attacks', 'frequency', 'defenses', or 'all'",
    )
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" VISION MODEL ROBUSTNESS FRAMEWORK: MODEL EVALUATION")
    print("=" * 70)
    print(f"Device:      {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Checkpoint:  {args.checkpoint}")
    print("-" * 70)

    # 1. Load Data
    _, _, test_loader = get_cifar100_loaders(
        data_dir=config["dataset"]["data_dir"],
        eval_batch_size=args.batch_size,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        normalize_in_transform=False,
    )

    # 2. Load Model
    model = resnet18_cifar(num_classes=config["model"]["num_classes"], normalize=True).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}. Please train the model first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "unknown")
    val_acc = ckpt.get("val_acc", "unknown")
    print(f"Loaded checkpoint from epoch {epoch} (Recorded val acc: {val_acc}%)")

    # 3. Clean Evaluation
    clean_acc = 77.84
    if args.benchmark in ["clean", "all"]:
        print("\nRunning evaluation on test set (10,000 samples)...")
        results = evaluate_clean(model, test_loader, device=device, amp_enabled=True)
        clean_acc = results["accuracy"]

        print("\n" + "=" * 50)
        print(f" Test Accuracy (Top-1):  {results['accuracy']:.2f}%")
        print(f" Test Loss:              {results['loss']:.4f}")
        print(f" Mean Confidence:        {results['mean_confidence'] * 100:.2f}%")
        print(f" Mean Entropy:           {results['mean_entropy']:.4f}")
        print(f" Latency per sample:     {results['inference_latency_ms_per_sample']:.3f} ms")
        print(f" Total Inference Time:   {results['total_inference_time_sec']:.2f} s")
        print("=" * 50)

        save_baseline_results(results, "experiments/baseline_results.json")

        conf_mat = np.array(results["confusion_matrix"])
        plot_confusion_matrix(conf_mat, "experiments/confusion_matrix.png")
        plot_class_accuracy(results["class_wise_accuracy"], "experiments/class_accuracy.png")
        print("\nClean evaluation complete! Baseline metrics and plots saved to experiments/")

    # 4. Phase 9: Natural Corruptions Benchmark
    if args.benchmark in ["corruptions", "all"]:
        print("\nStarting Phase 9 Natural Image Corruptions Benchmark...")
        run_natural_corruption_benchmark(
            model=model,
            loader=test_loader,
            device=device,
            clean_accuracy=clean_acc,
            output_csv="experiments/natural_corruption_results.csv",
            curves_plot_path="experiments/natural_corruption_curves.png",
            ranking_plot_path="experiments/natural_corruption_ranking.png",
            amp_enabled=True,
        )
        print("\nNatural Corruptions benchmark complete! Results saved to experiments/")

    # 5. Milestone 3: White-Box Adversarial Attacks Benchmark (FGSM & PGD)
    if args.benchmark in ["attacks", "all"]:
        print("\nStarting Milestone 3 White-Box Adversarial Attacks Benchmark...")
        run_adversarial_benchmark(
            model=model,
            loader=test_loader,
            device=device,
            clean_accuracy=clean_acc,
            output_csv="experiments/adversarial_attack_results.csv",
            curves_plot_path="experiments/adversarial_attack_curves.png",
            amp_enabled=True,
        )
        print("\nAdversarial Attacks benchmark complete! Results saved to experiments/")

    # 6. Milestone 4: Frequency Domain Spectral Benchmark (2D FFT & DCT)
    if args.benchmark in ["frequency", "all"]:
        print("\nStarting Milestone 4 Frequency Domain Benchmark...")
        run_frequency_benchmark(
            model=model,
            loader=test_loader,
            device=device,
            num_eval_samples=500,
            output_csv="experiments/frequency_analysis_summary.csv",
            plot_path="experiments/frequency_spectrum_analysis.png",
        )
        print("\nFrequency domain analysis complete! Results saved to experiments/")

    # 7. Milestone 5: Preprocessing Defenses Benchmark
    if args.benchmark in ["defenses", "all"]:
        print("\nStarting Milestone 5 Preprocessing Defenses Benchmark...")
        run_defense_benchmark(
            model=model,
            loader=test_loader,
            device=device,
            num_eval_samples=1000,
            output_csv="experiments/defense_mitigation_results.csv",
            plot_path="experiments/defense_mitigation_benchmark.png",
            amp_enabled=True,
        )
        print("\nPreprocessing Defenses benchmark complete! Results saved to experiments/")


if __name__ == "__main__":
    main()
