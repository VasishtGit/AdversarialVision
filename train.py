"""Unified training entry point for AdversarialVision.

Supports multiple training paradigms:
- 'standard': Clean Empirical Risk Minimization (ERM)
- 'pgd' / 'pgd_at': Multi-step PGD Adversarial Training (Madry et al.)
- 'fast_fgsm': Fast Adversarial Training with Random Start (Wong et al.)
- 'trades': TRADES Adversarial Defense (Zhang et al.)
"""

import sys
import os
import time
import random
from pathlib import Path
from typing import Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.cifar100 import get_cifar100_loaders
from models.resnet_cifar import resnet18_cifar
from trainers import TRAINING_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Train vision models (standard or adversarial)")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument(
        "--method",
        type=str,
        default="standard",
        choices=["standard", "pgd", "pgd_at", "fast_fgsm", "trades"],
        help="Training methodology",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate")
    parser.add_argument("--eps", type=float, default=None, help="Adversarial perturbation budget epsilon")
    parser.add_argument("--alpha", type=float, default=None, help="Adversarial attack step size alpha")
    parser.add_argument("--steps", type=int, default=None, help="Adversarial attack inner steps")
    parser.add_argument("--trades-beta", type=float, default=None, help="TRADES beta regularization weight")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save-name", type=str, default=None, help="Override checkpoint save filename")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float, float]:
    """Evaluates top-1 and top-5 accuracy and loss on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * targets.size(0)
        total += targets.size(0)

        _, preds_top5 = outputs.topk(5, 1, largest=True, sorted=True)
        targets_expanded = targets.view(-1, 1).expand_as(preds_top5)
        correct_in_top5 = preds_top5.eq(targets_expanded)
        correct_top1 += correct_in_top5[:, 0].sum().item()
        correct_top5 += correct_in_top5.any(dim=1).sum().item()

    val_loss = running_loss / total
    val_acc1 = 100.0 * correct_top1 / total
    val_acc5 = 100.0 * correct_top5 / total
    return val_loss, val_acc1, val_acc5


def main():
    args = parse_args()
    config = load_config(args.config)

    method = args.method.lower()
    is_adversarial = method in ["pgd", "pgd_at", "fast_fgsm", "trades"]

    adv_cfg = config.get("adversarial_training", {})

    # Determine hyperparameters based on method and CLI overrides
    if is_adversarial:
        epochs = args.epochs or adv_cfg.get("epochs", 50)
        batch_size = args.batch_size or adv_cfg.get("batch_size", 128)
        lr = args.lr or adv_cfg.get("lr", 0.05)
        eps = args.eps or adv_cfg.get("eps", 8.0 / 255.0)
        alpha = args.alpha or adv_cfg.get("alpha", 2.0 / 255.0)
        steps = args.steps or adv_cfg.get("steps", 7)
        trades_beta = args.trades_beta or adv_cfg.get("trades_beta", 6.0)
        default_save_name = f"resnet18_cifar100_{method}.pth"
    else:
        epochs = args.epochs or config["training"]["epochs"]
        batch_size = args.batch_size or config["training"]["batch_size"]
        lr = args.lr or config["training"]["optimizer"]["lr"]
        eps = 8.0 / 255.0
        alpha = 2.0 / 255.0
        steps = 7
        trades_beta = 6.0
        default_save_name = config["training"]["checkpoint"]["save_name"]

    save_name = args.save_name or default_save_name
    device_name = args.device or config.get("experiment", {}).get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    set_seed(config.get("experiment", {}).get("seed", 42))

    print("=" * 70)
    print(f" VISION MODEL ROBUSTNESS FRAMEWORK: TRAINING ({method.upper()})")
    print("=" * 70)
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Method:           {method}")
    print(f"Batch Size:       {batch_size}")
    print(f"Epochs:           {epochs}")
    print(f"Initial LR:       {lr}")
    if is_adversarial:
        print(f"Adversarial eps:  {eps*255:.1f}/255")
        print(f"Adversarial alpha:{alpha*255:.2f}/255 | steps: {steps}")
        if method == "trades":
            print(f"TRADES Beta:      {trades_beta}")
    print(f"Mixed Precision:  {config['training']['amp']}")
    print("-" * 70)

    # 1. Data Loaders
    data_dir = config["dataset"]["data_dir"]
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        eval_batch_size=config["evaluation"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"] and (device.type == "cuda"),
        train_size=config["dataset"]["train_split"],
        val_size=config["dataset"]["val_split"],
        normalize_in_transform=False,
    )

    # 2. Model
    model = resnet18_cifar(
        num_classes=config["model"]["num_classes"],
        normalize=True,
    ).to(device)

    # 3. Loss, Optimizer, Scaler, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=config["training"]["optimizer"]["momentum"],
        weight_decay=config["training"]["optimizer"]["weight_decay"],
        nesterov=config["training"]["optimizer"].get("nesterov", True),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config["training"]["scheduler"]["eta_min"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] and device.type == "cuda")

    # 4. Instantiate Trainer from Registry
    trainer_cls = TRAINING_REGISTRY.get(method)
    if trainer_cls is None:
        raise ValueError(f"Unknown training method '{method}'. Available: {list(TRAINING_REGISTRY.keys())}")

    trainer = trainer_cls(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        amp_enabled=config["training"]["amp"],
        eps=eps,
        alpha=alpha,
        steps=steps,
        beta=trades_beta,
    )

    start_epoch = 1
    best_val_acc = 0.0

    # Resume support
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed at epoch {start_epoch} (Best Val Acc: {best_val_acc:.2f}%)")

    # Output paths
    save_dir = Path(config["training"]["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_name
    log_dir = Path("experiments")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{method}_training_log.csv"

    history = []
    print(f"\nStarting training for {epochs} epochs...")
    training_start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = trainer.train_epoch(train_loader)

        val_loss, val_acc1, val_acc5 = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=config["training"]["amp"],
        )

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

        is_best = val_acc1 > best_val_acc
        if is_best:
            best_val_acc = val_acc1
            torch.save(
                {
                    "epoch": epoch,
                    "method": method,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_acc": val_acc1,
                    "val_loss": val_loss,
                    "config": config,
                },
                save_path,
            )

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc1:.2f}% (Top-5: {val_acc5:.2f}%) | "
            f"LR: {current_lr:.5f} | "
            f"Time: {epoch_duration:.1f}s"
            f"{' [BEST]' if is_best else ''}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "lr": current_lr,
            "epoch_duration_sec": epoch_duration,
        })

        if epoch % 5 == 0 or epoch == epochs:
            pd.DataFrame(history).to_csv(log_file, index=False)

    total_time = time.time() - training_start_time
    print("=" * 70)
    print(f"Training Complete in {total_time / 60:.1f} minutes!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Checkpoint Saved To: {save_path}")

    # Final held-out test set evaluation
    print("\nRunning final held-out test set evaluation with best checkpoint...")
    best_ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc1, test_acc5 = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp_enabled=config["training"]["amp"],
    )
    print("=" * 70)
    print(f" FINAL HELD-OUT TEST RESULTS ({method.upper()}):")
    print(f" Test Top-1 Accuracy: {test_acc1:.2f}%")
    print(f" Test Top-5 Accuracy: {test_acc5:.2f}%")
    print(f" Test Loss:           {test_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
