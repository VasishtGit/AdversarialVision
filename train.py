"""Baseline training script for CIFAR-100 ResNet-18 with AMP and metrics logging."""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from tqdm import tqdm

from data.cifar100 import get_cifar100_loaders
from models.resnet_cifar import resnet18_cifar


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline ResNet-18 on CIFAR-100")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * targets.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * targets.size(0)
        total += targets.size(0)

        # Top-1 and Top-5 computation
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

    # Apply command-line overrides
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.lr or config["training"]["optimizer"]["lr"]
    device_name = args.device or config.get("experiment", {}).get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    set_seed(config.get("experiment", {}).get("seed", 42))

    print("=" * 70)
    print(" VISION MODEL ROBUSTNESS FRAMEWORK: BASELINE RESNET-18 TRAINING")
    print("=" * 70)
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Batch Size:       {batch_size}")
    print(f"Epochs:           {epochs}")
    print(f"Initial LR:       {lr}")
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
        normalize_in_transform=False,  # Kept in [0, 1] for differentiable model NormalizeLayer
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
        T_max=config["training"]["scheduler"]["t_max"],
        eta_min=config["training"]["scheduler"]["eta_min"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"])

    start_epoch = 1
    best_val_acc = 0.0

    # Resume support
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed at epoch {start_epoch} (Best Val Acc: {best_val_acc:.2f}%)")

    # Output paths
    save_dir = Path(config["training"]["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / config["training"]["checkpoint"]["save_name"]
    log_dir = Path("experiments")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "baseline_training_log.csv"

    history = []

    print(f"\nStarting training for {epochs} epochs...")
    training_start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=config["training"]["amp"],
        )

        val_loss, val_acc1, val_acc5 = evaluate(
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
            f"VRAM: {peak_vram:.0f}MB | "
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
            "peak_vram_mb": peak_vram,
            "epoch_duration_sec": epoch_duration,
        })

        # Periodically persist training log
        if epoch % 5 == 0 or epoch == epochs:
            pd.DataFrame(history).to_csv(log_file, index=False)

    total_time = time.time() - training_start_time
    print("=" * 70)
    print(f"Training Complete in {total_time / 60:.1f} minutes!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Checkpoint Saved To: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
