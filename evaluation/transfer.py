"""Transfer attack evaluation module.
Tests if adversarial examples generated on a source model fool an unseen target model.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from tqdm import tqdm

from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack


def evaluate_transferability(
    source_model: nn.Module,
    target_model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float = 8.0 / 255.0,
    max_samples: Optional[int] = 1000,
) -> Dict[str, float]:
    """
    Crafts PGD-10 adversarial perturbations against `source_model`
    and measures classification error on `target_model`.
    """
    source_model.eval()
    target_model.eval()

    attacker = PGDAttack(model=source_model, eps=eps, alpha=eps / 4.0, steps=10, random_start=True)

    total = 0
    source_correct = 0
    target_correct_clean = 0
    target_correct_adv = 0

    for imgs, tgts in loader:
        if max_samples is not None and total >= max_samples:
            break
        imgs = imgs.to(device)
        tgts = tgts.to(device)

        with torch.no_grad():
            out_target_clean = target_model(imgs)
            target_correct_clean += out_target_clean.argmax(dim=1).eq(tgts).sum().item()

        # Craft on source
        adv_imgs = attacker.generate(imgs, tgts)

        with torch.no_grad():
            out_src = source_model(adv_imgs)
            out_tgt = target_model(adv_imgs)

            source_correct += out_src.argmax(dim=1).eq(tgts).sum().item()
            target_correct_adv += out_tgt.argmax(dim=1).eq(tgts).sum().item()

        total += tgts.size(0)

    src_acc = 100.0 * source_correct / total
    tgt_clean_acc = 100.0 * target_correct_clean / total
    tgt_adv_acc = 100.0 * target_correct_adv / total
    transfer_asr = 100.0 - tgt_adv_acc

    return {
        "samples": total,
        "eps": eps,
        "source_adv_acc": src_acc,
        "target_clean_acc": tgt_clean_acc,
        "target_adv_acc": tgt_adv_acc,
        "transfer_asr": transfer_asr,
    }
