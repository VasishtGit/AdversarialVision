"""AutoAttack (Croce & Hein, ICML 2020) reliable evaluation runner.
Standard reliable benchmark against gradient masking and obfuscation.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from tqdm import tqdm


def run_autoattack_benchmark(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float = 8.0 / 255.0,
    norm: str = "Linf",
    max_samples: Optional[int] = 1000,
) -> Dict[str, Any]:
    """
    Evaluates model against AutoAttack standard ensemble (APGD-CE, APGD-DLR, FAB, Square Attack).
    If autoattack library is not installed, falls back cleanly with helpful instruction.
    """
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("\n[AutoAttack] 'autoattack' package is not installed. Install with: pip install autoattack")
        return {"status": "skipped", "reason": "autoattack package not installed"}

    model.eval()
    adversary = AutoAttack(model, norm=norm, eps=eps, version="standard", device=device)

    all_x = []
    all_y = []
    total = 0

    for imgs, tgts in loader:
        if max_samples is not None and total >= max_samples:
            break
        all_x.append(imgs)
        all_y.append(tgts)
        total += tgts.size(0)

    x_test = torch.cat(all_x, dim=0)[:max_samples].to(device)
    y_test = torch.cat(all_y, dim=0)[:max_samples].to(device)

    print(f"\n[AutoAttack] Running standard evaluation on {len(y_test)} samples (norm={norm}, eps={eps*255:.1f}/255)...")
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)

    with torch.no_grad():
        clean_acc = 100.0 * model(x_test).argmax(dim=1).eq(y_test).float().mean().item()
        robust_acc = 100.0 * model(x_adv).argmax(dim=1).eq(y_test).float().mean().item()

    return {
        "norm": norm,
        "eps": eps,
        "samples": len(y_test),
        "clean_accuracy": clean_acc,
        "robust_accuracy": robust_acc,
        "asr": 100.0 - robust_acc,
    }
