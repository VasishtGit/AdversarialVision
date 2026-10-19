"""Adversarial Training for CIFAR-100 ResNet-18 (Fast Adversarial Training with FGSM + Random Start and PGD-AT)."""

import os
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import pandas as pd

from tqdm import tqdm
from data.cifar100 import get_cifar100_loaders
from models.resnet_cifar import resnet18_cifar
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack
from evaluation.clean import evaluate_clean


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarially Train ResNet-18 on CIFAR-100")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to base config")
    parser.add_argument("--epochs", type=int, default=50, help="Number of adversarial training epochs")
    parser.add_argument("--eps", type=float, default=8.0 / 255.0, help="Adversarial budget epsilon")
    parser.add_argument("--alpha", type=float, default=2.0 / 255.0, help="PGD step size alpha")
    parser.add_argument("--steps", type=int, default=7, help="PGD attack steps during training (7 is standard PGD-AT)")
    parser.add_argument("--method", type=str, default="pgd", choices=["pgd", "fast_fgsm"], help="Adversarial training method")
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate")
    parser.add_argument("--save-path", type=str, default="models/checkpoints/resnet18_cifar100_adv_trained.pth")
    return parser.parse_args()


def generate_adv_train_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, alpha: float, steps: int, method: str):
    """Generates on-the-fly adversarial perturbations during the training loop in [0, 1]."""
    if method == "fast_fgsm":
        # Fast Adversarial Training with Random Start (Wong et al., ICLR 2020)
        x_adv = x.clone().detach()
        noise = torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0).requires_grad_(True)

        with autocast("cuda", enabled=x.is_cuda):
            outputs = model(x_adv)
            loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()

        grad = x_adv.grad.data
        x_adv = x_adv.detach() + 1.25 * eps * torch.sign(grad)
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        return torch.clamp(x + eta, min=0.0, max=1.0).detach()

    else:
        # Standard PGD-AT (Madry et al., ICLR 2018)
        x_adv = x.clone().detach()
        noise = torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

        for _ in range(steps):
            x_adv.requires_grad_()
            with autocast("cuda", enabled=x.is_cuda):
                outputs = model(x_adv)
                loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()

            grad = x_adv.grad.data
            x_adv = x_adv.detach() + alpha * torch.sign(grad)
            eta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + eta, min=0.0, max=1.0)

        return x_adv.detach()


def train_adversarial():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" VISION MODEL ROBUSTNESS: ADVERSARIAL TRAINING (PGD-AT / FAST-FGSM)")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Method:           {args.method.upper()}-AT (eps={args.eps*255:.1f}/255, steps={args.steps})")
    print(f"Epochs:           {args.epochs} | Init LR: {args.lr}")
    print(f"Output Checkpoint:{args.save_path}")
    print("-" * 70)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        data_dir="./data",
        batch_size=128,
        eval_batch_size=256,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        normalize_in_transform=False,
    )

    # 2. Instantiate Model
    model = resnet18_cifar(num_classes=100, normalize=True).to(device)

    # Transfer baseline pretrained weights if available for warm start, or train
    base_ckpt = Path("models/checkpoints/resnet18_cifar100_best.pth")
    if base_ckpt.exists():
        print("Warm-starting from clean baseline weights...")
        ckpt = torch.load(base_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Training from scratch...")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=True)

    best_val_rob_acc = 0.0
    history = []

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_t = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Generate adversarial training samples
            model.eval()  # Set to eval mode to prevent BN stats contamination by adversarial perturbation step
            x_adv = generate_adv_train_batch(
                model=model,
                x=images,
                y=targets,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                method=args.method,
            )
            model.train()

            optimizer.zero_grad()
            with autocast("cuda", enabled=True):
                logits = model(x_adv)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * targets.size(0)
            train_correct += logits.argmax(dim=1).eq(targets).sum().item()
            train_total += targets.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{(train_correct/train_total)*100:.1f}%"})

        scheduler.step()
        train_loss = running_train_loss / train_total
        train_acc = (train_correct / train_total) * 100.0

        # Validate on Clean and PGD-3 validation set (fast robust monitor)
        model.eval()
        val_clean_correct = 0
        val_rob_correct = 0
        val_total = 0

        val_attacker = PGDAttack(model=model, eps=args.eps, alpha=args.alpha, steps=3, random_start=True)

        with torch.no_grad():
            for v_imgs, v_tgts in val_loader:
                v_imgs = v_imgs.to(device, non_blocking=True)
                v_tgts = v_tgts.to(device, non_blocking=True)

                # Clean
                out_clean = model(v_imgs)
                val_clean_correct += out_clean.argmax(dim=1).eq(v_tgts).sum().item()
                val_total += v_tgts.size(0)

        # Robust sample (evaluated with grad on small subset for speed)
        val_rob_total = 0
        for idx, (v_imgs, v_tgts) in enumerate(val_loader):
            if idx >= 4:  # up to 4 batches
                break
            v_imgs = v_imgs.to(device)
            v_tgts = v_tgts.to(device)
            v_adv = val_attacker.generate(v_imgs, v_tgts)
            with torch.no_grad():
                out_rob = model(v_adv)
                val_rob_correct += out_rob.argmax(dim=1).eq(v_tgts).sum().item()
                val_rob_total += v_tgts.size(0)

        val_clean_acc = (val_clean_correct / val_total) * 100.0
        val_rob_acc = (val_rob_correct / max(val_rob_total, 1)) * 100.0
        elapsed = time.perf_counter() - start_t

        is_best = val_rob_acc > best_val_rob_acc
        if is_best:
            best_val_rob_acc = val_rob_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_clean_acc": val_clean_acc,
                "val_rob_acc": val_rob_acc,
                "optimizer_state_dict": optimizer.state_dict(),
            }, args.save_path)

        flag = " [BEST ROBUST]" if is_best else ""
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Tr Loss: {train_loss:.3f} | Tr Acc: {train_acc:.1f}% | Val Clean: {val_clean_acc:.2f}% | Val Rob: {val_rob_acc:.2f}% | Time: {elapsed:.1f}s{flag}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_clean_acc": val_clean_acc,
            "val_rob_acc": val_rob_acc,
        })

    pd.DataFrame(history).to_csv("experiments/adversarial_training_log.csv", index=False)
    print(f"\nAdversarial training finished! Best checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    train_adversarial()
