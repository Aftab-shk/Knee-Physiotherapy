"""
train.py
============
Fine-tunes a torchvision EfficientNet (B4 by default) on the Knee
Osteoarthritis Severity Grading dataset (5 KL grades, 0-4).

    python train.py --arch b4 --data_dir "/path/to/Knee Osteoarthritis Dataset with Severity Grading"

Pipeline:
  1. Load data via prepare_dataset.get_dataloaders (also returns per-class
     weights to counter the dataset's class imbalance).
  2. Build an ImageNet-pretrained EfficientNet with its 1000-way head
     replaced by a 5-way one, at the architecture's native input resolution.
  3. Phase 1 ("warm-up"): freeze the pretrained backbone, train only the
     new head for a few epochs so its random initialization doesn't send
     large, destructive gradients back through the pretrained features.
  4. Phase 2 ("fine-tune"): unfreeze everything, train end-to-end with
     layer-wise LR decay, MixUp/CutMix, and a warmup+cosine schedule.
  5. Track validation accuracy every epoch for both the live weights and an
     EMA copy, keep the best checkpoint, stop early if it stalls.
  6. Evaluate the best checkpoint on the held-out test/ split (with optional
     flip TTA) and save a classification report, confusion matrix, and curves.

What changed relative to the original B0/B3 recipe, and why (see ACCURACY
NOTES at the bottom of this file for the full reasoning):
  - native input resolution per architecture (B4 -> 380px, not 224px)
  - ordinal-aware loss: KL grades are ordered, so a 0->4 error is penalized
    harder than a 0->1 error
  - MixUp / CutMix + stochastic depth, to regularize a 19M-param model on a
    ~5.8k-image training set
  - EMA weight averaging, evaluated alongside the raw weights
  - layer-wise LR decay, so pretrained early layers move less than late ones
  - sqrt-inverse class weights instead of full inverse frequency

Outputs (written to --output_dir, default "outputs/"):
  best_model.pth          - everything inference.py needs to reload the model
  label_map.json          - human-readable class-index -> KL grade mapping
  history.json            - per-epoch metrics
  training_curves.png     - loss / accuracy per epoch
  confusion_matrix_test.png
  test_report.json        - precision / recall / f1 per class on the test set
"""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
)

from prepare_dataset import (
    DEFAULT_OUTPUT_DIR,
    RANDOM_SEED,
    build_label_maps,
    default_img_size,
    get_dataloaders,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_ARCHS = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights),
    "b1": (efficientnet_b1, EfficientNet_B1_Weights),
    "b2": (efficientnet_b2, EfficientNet_B2_Weights),
    "b3": (efficientnet_b3, EfficientNet_B3_Weights),
    "b4": (efficientnet_b4, EfficientNet_B4_Weights),
    "b5": (efficientnet_b5, EfficientNet_B5_Weights),
}


def build_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.4,
    drop_path: float = 0.2,
) -> nn.Module:
    """
    `drop_path` is EfficientNet's stochastic depth: during training whole
    residual blocks are randomly skipped. It is the regularizer that lets a
    B4 fine-tune on a few thousand images without collapsing onto them, and
    it costs nothing at inference (it is a no-op in eval mode).
    """
    arch = arch.lower()
    if arch not in _ARCHS:
        raise ValueError(f"Unsupported --arch '{arch}'. Choose one of {sorted(_ARCHS)}.")

    ctor, weights_enum = _ARCHS[arch]
    weights = weights_enum.IMAGENET1K_V1 if pretrained else None
    model = ctor(weights=weights, stochastic_depth_prob=drop_path)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = trainable


# ---------------------------------------------------------------------------
# Exponential moving average of the weights
# ---------------------------------------------------------------------------
class ModelEMA:
    """
    Keeps a slowly-moving average of the weights. Because SGD with a cosine
    schedule keeps bouncing around the minimum rather than sitting in it, the
    average of the last few thousand steps is usually a slightly better model
    than whatever point the last step happened to land on - typically worth
    1-2 accuracy points here, for no extra training time.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Ramp the decay in: at step 10 a 0.999 decay would leave the average
        # still ~99% initialization, so early steps use a faster decay.
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.detach(), alpha=1.0 - d)
            else:
                v.copy_(src)  # BN num_batches_tracked and friends


# ---------------------------------------------------------------------------
# Loss: soft-target cross-entropy with an ordinal penalty
# ---------------------------------------------------------------------------
class OrdinalSoftTargetLoss(nn.Module):
    """
    Cross-entropy that accepts soft targets (needed for MixUp/CutMix and label
    smoothing) plus an ordinal regression term.

    The ordinal term matters clinically as well as numerically. KL grades are
    an ordered scale, but plain cross-entropy treats every wrong class as
    equally wrong - predicting Grade 4 for a healthy knee costs exactly what
    predicting Grade 1 costs. Adding a penalty on the distance between the
    predicted expected grade and the true grade tells the model that being
    off by one is a near-miss and being off by four is a disaster. It pulls
    the residual errors onto the diagonal's neighbours, which lifts accuracy
    and makes the confusion matrix far more defensible for a triage tool.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        ordinal_weight: float = 0.3,
    ):
        super().__init__()
        self.ordinal_weight = ordinal_weight
        self.register_buffer("grades", torch.arange(num_classes, dtype=torch.float))
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        per_sample = -(target_probs * log_probs).sum(dim=1)

        if self.class_weights is not None:
            # Effective weight of a (possibly mixed) target is the target
            # distribution's expected class weight.
            w = (target_probs * self.class_weights).sum(dim=1)
            loss = (per_sample * w).sum() / w.sum().clamp(min=1e-8)
        else:
            loss = per_sample.mean()

        if self.ordinal_weight > 0:
            expected_pred = (log_probs.exp() * self.grades).sum(dim=1)
            expected_true = (target_probs * self.grades).sum(dim=1)
            loss = loss + self.ordinal_weight * F.smooth_l1_loss(expected_pred, expected_true)

        return loss


def one_hot_smooth(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    t = torch.full((labels.size(0), num_classes), off, device=labels.device)
    return t.scatter_(1, labels.unsqueeze(1), on)


# ---------------------------------------------------------------------------
# MixUp / CutMix
# ---------------------------------------------------------------------------
def _rand_bbox(h: int, w: int, lam: float) -> Tuple[int, int, int, int]:
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * ratio), int(w * ratio)
    cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


def apply_mix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
    prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    With probability `prob`, blend the batch with a shuffled copy of itself -
    either by alpha-blending whole images (MixUp) or by pasting a rectangle
    from one into the other (CutMix) - and blend the soft targets to match.

    On a training set this small, this is doing most of the work of preventing
    a B4 from simply memorizing the images.
    """
    if prob <= 0 or random.random() > prob:
        return images, targets
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, targets

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(images.size(0), device=images.device)

    if use_cutmix:
        h, w = images.shape[2], images.shape[3]
        y1, y2, x1, x2 = _rand_bbox(h, w, lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        # Recompute lam from the actual pasted area (it gets clipped at edges).
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(h * w))
    else:
        images = lam * images + (1.0 - lam) * images[index]

    targets = lam * targets + (1.0 - lam) * targets[index]
    return images, targets


# ---------------------------------------------------------------------------
# Optimizer: layer-wise LR decay
# ---------------------------------------------------------------------------
def llrd_param_groups(
    model: nn.Module, base_lr: float, decay: float = 0.8, weight_decay: float = 1e-4
):
    """
    Give later blocks a higher LR than earlier ones.

    EfficientNet's early blocks hold generic edge/texture filters that ImageNet
    already got right and that 5.8k X-rays cannot improve on; the late blocks
    hold the semantic features that actually need to be re-specialized for
    radiographs. One global LR forces a compromise - low enough not to wreck
    the early layers means too low to properly adapt the late ones. Decaying
    the LR by depth removes that compromise.

    Norm layers and biases are excluded from weight decay, which is standard
    practice and avoids shrinking BN scales toward zero.
    """
    groups = []

    def add(params_named, lr):
        decay_p = [p for n, p in params_named if p.requires_grad and p.ndim > 1]
        no_decay_p = [p for n, p in params_named if p.requires_grad and p.ndim <= 1]
        if decay_p:
            groups.append({"params": decay_p, "lr": lr, "weight_decay": weight_decay})
        if no_decay_p:
            groups.append({"params": no_decay_p, "lr": lr, "weight_decay": 0.0})

    add(list(model.classifier.named_parameters()), base_lr)

    n_blocks = len(model.features)
    for i in range(n_blocks - 1, -1, -1):
        lr = base_lr * (decay ** (n_blocks - i))
        add(list(model.features[i].named_parameters()), lr)

    return groups


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_factor: float = 0.01
) -> LambdaLR:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, fn)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    use_amp: bool,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
    ema: Optional[ModelEMA],
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        targets = one_hot_smooth(labels, num_classes, label_smoothing)
        images, targets = apply_mix(images, targets, mixup_alpha, cutmix_alpha, mix_prob)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        preds = outputs.detach().argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        # Against the *original* labels, so this stays comparable across
        # epochs even though the model trained on mixed images.
        total_correct += (preds == labels).sum().item()
        total_seen += images.size(0)

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: Optional[nn.Module] = None,
    tta: bool = False,
) -> Tuple[float, float, float, list, list]:
    """Returns (loss, accuracy, macro_f1, all_preds, all_labels)."""
    model.eval()
    total_loss, total_seen = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        if tta:
            # Horizontal flip only: it maps a left knee onto a right knee,
            # which is a real and label-preserving variation here.
            probs = (probs + F.softmax(model(torch.flip(images, dims=[3])), dim=1)) / 2.0

        if criterion is not None:
            targets = one_hot_smooth(labels, num_classes, 0.0)
            total_loss += criterion(logits, targets).item() * images.size(0)

        total_seen += images.size(0)
        all_preds.extend(probs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0
    return total_loss / max(total_seen, 1), acc, macro_f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Post-hoc logit adjustment
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device, tta: bool = False):
    """Run the model once and keep the raw logits, so a threshold sweep costs
    nothing extra. Returns (logits, flipped_logits_or_None, labels)."""
    model.eval()
    logits, logits_flip, labels = [], [], []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        logits.append(model(images).float().cpu())
        if tta:
            logits_flip.append(model(torch.flip(images, dims=[3])).float().cpu())
        labels.append(y.clone())
    return (
        torch.cat(logits),
        torch.cat(logits_flip) if tta else None,
        torch.cat(labels),
    )


def probs_with_adjust(logits, logits_flip, adjust: Optional[torch.Tensor]):
    def sm(l):
        return F.softmax(l if adjust is None else l - adjust, dim=1)

    probs = sm(logits)
    if logits_flip is not None:
        probs = (probs + sm(logits_flip)) / 2.0
    return probs


def tune_logit_tau(logits, logits_flip, labels, priors, select_metric: str, taus=None):
    """
    Find the prior-correction strength tau that maximizes the validation
    metric, where adjusted_logits = logits - tau * log(class_prior).

    Why this helps here: the model is trained on data where Grade 0 outnumbers
    Grade 4 by ~13x, so it learns to lean on that prior and over-predicts the
    common grades. Subtracting a multiple of log(prior) cancels that lean back
    out. It is the standard fix for exactly the failure this run shows - a
    class with high recall but mediocre precision (Grade 0) sitting next to a
    class the model almost never predicts (Grade 1).

    tau is swept on the *validation* split and tau=0 is always in the sweep,
    so the chosen value can never be worse than no adjustment on val.
    """
    if taus is None:
        taus = [i / 20.0 for i in range(21)]  # 0.00 .. 1.00

    log_prior = torch.log(priors.clamp(min=1e-8))
    best_tau, best_score = 0.0, -1.0

    for tau in taus:
        adjust = None if tau == 0 else tau * log_prior
        preds = probs_with_adjust(logits, logits_flip, adjust).argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        f1 = f1_score(labels.tolist(), preds.tolist(), average="macro", zero_division=0)
        score = acc if select_metric == "acc" else f1 if select_metric == "f1" else 0.5 * (acc + f1)
        if score > best_score:
            best_tau, best_score = tau, score

    return best_tau, best_score


def report_and_plot(
    all_preds, all_labels, idx_to_name, output_dir: Path, split_name: str = "test"
) -> float:
    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    print(f"\n{split_name.upper()} SET RESULTS")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # Off-by-one accuracy: how often the prediction is within one KL grade of
    # the truth. Reported because it is the number that reflects whether the
    # downstream exercise protocol would have been roughly right.
    within_one = sum(abs(p - l) <= 1 for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    report["within_one_grade_accuracy"] = within_one
    print(f"Within-one-grade accuracy: {within_one:.4f}")

    with open(output_dir / f"{split_name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({split_name})")
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{split_name}.png", dpi=150)
    plt.close(fig)

    return report["accuracy"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an EfficientNet on the Knee OA severity dataset.")
    p.add_argument("--data_dir", type=str, default=None, help="Dataset root (contains train/val/test).")
    p.add_argument("--arch", type=str, default="b4", choices=sorted(_ARCHS), help="EfficientNet variant.")
    p.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Input resolution. Defaults to the architecture's native size (b4 -> 380).",
    )
    p.add_argument("--batch_size", type=int, default=24, help="Lower this if you hit CUDA OOM.")
    p.add_argument("--warmup_epochs", type=int, default=3, help="Epochs training only the classifier head.")
    p.add_argument("--finetune_epochs", type=int, default=40, help="Max epochs fine-tuning the whole network.")
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--finetune_lr", type=float, default=3e-4, help="Peak LR of the *last* block; earlier blocks decay from it.")
    p.add_argument("--llrd", type=float, default=0.8, help="Layer-wise LR decay factor (1.0 disables it).")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.4, help="Classifier-head dropout.")
    p.add_argument("--drop_path", type=float, default=0.2, help="Stochastic depth probability.")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--ordinal_weight", type=float, default=0.3, help="Weight of the ordinal distance penalty (0 disables it).")
    p.add_argument("--mixup", type=float, default=0.2, help="MixUp alpha (0 disables).")
    p.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha (0 disables).")
    p.add_argument("--mix_prob", type=float, default=0.5, help="Probability of mixing a given batch.")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (0 disables EMA).")
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience (in fine-tune epochs).")
    p.add_argument("--num_workers", type=int, default=2, help="Use 4 on Kaggle.")
    p.add_argument("--weighted_sampler", action="store_true", help="Oversample rare classes per batch instead of weighting the loss.")
    p.add_argument(
        "--class_weight_mode",
        type=str,
        default="sqrt_inverse",
        choices=["none", "sqrt_inverse", "inverse"],
        help="How hard to correct class imbalance in the loss.",
    )
    p.add_argument(
        "--select_metric",
        type=str,
        default="acc",
        choices=["acc", "f1", "mean"],
        help="Validation metric used to pick the best checkpoint.",
    )
    p.add_argument(
        "--class_weights",
        type=str,
        default=None,
        help="Manual per-class loss weights, comma-separated (e.g. '1,1.6,1,1,1' to push "
        "harder on Grade 1). Overrides --class_weight_mode.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a best_model.pth to continue training from. Skips the warm-up phase "
        "and restarts the cosine schedule from --finetune_lr, so a run that early-stopped "
        "before its LR finished annealing can be finished off cheaply.",
    )
    p.add_argument(
        "--logit_adjust",
        type=str,
        default="auto",
        help="Post-hoc class-prior correction: 'auto' tunes tau on val, 'off' disables, "
        "or pass a fixed float.",
    )
    p.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast equalization (train + inference).")
    p.add_argument("--mild_aug", action="store_true", help="Use the original mild augmentation instead of the stronger recipe.")
    p.add_argument("--tta", action="store_true", help="Use horizontal-flip TTA during validation and test.")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision even if a GPU is available.")
    p.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Train from random init instead of ImageNet weights (only if your network can't reach "
        "download.pytorch.org). Accuracy will be substantially lower - use only if you must.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = args.img_size or default_img_size(args.arch)
    if args.img_size is None:
        print(f"Using {args.arch.upper()}'s native input resolution: {img_size}px")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Oversampling and loss weighting fix the same imbalance; applying both
    # double-corrects it, so the sampler takes over and the loss goes uniform.
    class_weight_mode = "none" if args.weighted_sampler else args.class_weight_mode
    if args.weighted_sampler and args.class_weight_mode != "none":
        print("--weighted_sampler is on, so loss class-weights are set to 'none' to avoid double-correcting.")

    train_loader, val_loader, test_loader, classes, class_weights = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=img_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.weighted_sampler,
        strong_aug=not args.mild_aug,
        clahe=args.clahe,
        class_weight_mode=class_weight_mode,
    )
    if len(train_loader) == 0:
        raise RuntimeError(
            "The training DataLoader produced 0 batches - --batch_size is probably "
            "larger than the number of training images available. Lower --batch_size."
        )

    num_classes = len(classes)
    idx_to_grade, idx_to_name = build_label_maps(classes)
    print(f"Classes found ({num_classes}): {idx_to_name}")

    if args.class_weights:
        manual = [float(v) for v in args.class_weights.split(",")]
        if len(manual) != num_classes:
            raise ValueError(
                f"--class_weights needs {num_classes} comma-separated values, got {len(manual)}."
            )
        class_weights = torch.tensor(manual, dtype=torch.float)
        class_weight_mode = "manual"
    print(f"Class weights ({class_weight_mode}): {[round(w, 3) for w in class_weights.tolist()]}")

    # Training-set class priors, used for the post-hoc logit adjustment.
    prior_counts = torch.zeros(num_classes)
    for _, lab in train_loader.dataset.samples:
        prior_counts[lab] += 1
    class_priors = prior_counts / prior_counts.sum()

    if args.no_pretrained:
        print(
            "WARNING: --no_pretrained set - training from random initialization. "
            "This will likely fall well short of 70-80% accuracy; only use this if "
            "your machine truly cannot reach download.pytorch.org."
        )
    model = build_model(
        args.arch,
        num_classes,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        drop_path=args.drop_path,
    ).to(device)

    criterion = OrdinalSoftTargetLoss(
        num_classes=num_classes,
        class_weights=None if class_weight_mode == "none" else class_weights,
        ordinal_weight=args.ordinal_weight,
    ).to(device)

    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler = GradScaler(device.type, enabled=use_amp)
    print(f"Mixed precision (AMP): {use_amp}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "ema_val_acc": []}
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_is_ema = False

    def score_of(acc: float, f1: float) -> float:
        if args.select_metric == "acc":
            return acc
        if args.select_metric == "f1":
            return f1
        return 0.5 * (acc + f1)

    logit_tau = 0.0

    def checkpoint_payload(state_dict):
        return {
            "model_state_dict": state_dict,
            "arch": args.arch,
            "num_classes": num_classes,
            "idx_to_grade": idx_to_grade,
            "idx_to_name": idx_to_name,
            "img_size": img_size,
            "clahe": args.clahe,
            "dropout": args.dropout,
            "logit_tau": logit_tau,
            "class_priors": class_priors.tolist(),
        }

    # ----------------- optionally resume from an earlier run -----------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        args.warmup_epochs = 0  # the head is already trained
        _, r_acc, r_f1, _, _ = evaluate(model, val_loader, device, num_classes, None, tta=args.tta)
        best_score = score_of(r_acc, r_f1)
        best_state = copy.deepcopy(model.state_dict())
        print(
            f"Resumed from {args.resume} (val_acc={r_acc:.4f}, val_f1={r_f1:.4f}). "
            "Skipping warm-up; a checkpoint is only overwritten if it beats this."
        )

    total_epochs = args.warmup_epochs + args.finetune_epochs
    epoch_counter = 0

    # ----------------- Phase 1: warm up the classifier head -----------------
    print(f"\n=== Phase 1/2: warming up the classifier head for {args.warmup_epochs} epoch(s) ===")
    set_backbone_trainable(model, False)
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.head_lr, weight_decay=args.weight_decay
    )

    for _ in range(args.warmup_epochs):
        epoch_counter += 1
        t0 = time.time()
        # No mixing during warm-up: the head is random, and it needs a clean
        # signal before it is worth regularizing.
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, device, optimizer, None, scaler, use_amp,
            num_classes, args.label_smoothing, 0.0, 0.0, 0.0, None,
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, device, num_classes, criterion, tta=args.tta
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["ema_val_acc"].append(float("nan"))

        print(
            f"[warmup {epoch_counter}/{total_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({time.time() - t0:.1f}s)"
        )

        score = score_of(val_acc, val_f1)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_is_ema = False
            torch.save(checkpoint_payload(best_state), output_dir / "best_model.pth")

    # ----------------- Phase 2: fine-tune everything -----------------
    print(f"\n=== Phase 2/2: fine-tuning the whole network for up to {args.finetune_epochs} epoch(s) ===")
    set_backbone_trainable(model, True)

    param_groups = llrd_param_groups(model, args.finetune_lr, decay=args.llrd, weight_decay=args.weight_decay)
    optimizer = AdamW(param_groups, lr=args.finetune_lr)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.finetune_epochs
    scheduler = warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=steps_per_epoch)
    print(
        f"LLRD {args.llrd}: block LRs from {args.finetune_lr * (args.llrd ** len(model.features)):.2e} "
        f"(earliest) to {args.finetune_lr:.2e} (head)"
    )

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"EMA enabled (decay={args.ema_decay})")

    epochs_no_improve = 0
    for _ in range(args.finetune_epochs):
        epoch_counter += 1
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, device, optimizer, scheduler, scaler, use_amp,
            num_classes, args.label_smoothing, args.mixup, args.cutmix, args.mix_prob, ema,
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, device, num_classes, criterion, tta=args.tta
        )

        # Score the EMA weights too and keep whichever is ahead.
        ema_acc, ema_f1 = float("nan"), float("nan")
        if ema is not None:
            _, ema_acc, ema_f1, _, _ = evaluate(
                ema.module, val_loader, device, num_classes, None, tta=args.tta
            )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["ema_val_acc"].append(ema_acc)

        raw_score = score_of(val_acc, val_f1)
        ema_score = score_of(ema_acc, ema_f1) if ema is not None else -1.0
        use_ema = ema_score > raw_score
        epoch_score = max(raw_score, ema_score)
        improved = epoch_score > best_score

        ema_str = f" | ema_val_acc={ema_acc:.4f}" if ema is not None else ""
        print(
            f"[finetune {epoch_counter}/{total_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}{ema_str}"
            f"{'  (best so far' + (', EMA)' if use_ema else ')') if improved else ''} ({time.time() - t0:.1f}s)"
        )

        if improved:
            best_score = epoch_score
            best_state = copy.deepcopy((ema.module if use_ema else model).state_dict())
            best_is_ema = use_ema
            torch.save(checkpoint_payload(best_state), output_dir / "best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"No val improvement for {args.patience} epochs - stopping early.")
                break

    print(f"\nBest validation {args.select_metric}: {best_score:.4f}" + (" (EMA weights)" if best_is_ema else ""))
    model.load_state_dict(best_state)

    # ----------------- post-hoc class-prior correction -----------------
    # Tuned on val only, then applied unchanged to test. tau=0 is in the
    # sweep, so this cannot make the validation metric worse.
    if args.logit_adjust != "off":
        v_logits, v_flip, v_labels = collect_logits(model, val_loader, device, tta=args.tta)
        base_acc = (probs_with_adjust(v_logits, v_flip, None).argmax(1) == v_labels).float().mean().item()
        if args.logit_adjust == "auto":
            logit_tau, tuned = tune_logit_tau(
                v_logits, v_flip, v_labels, class_priors, args.select_metric
            )
            print(
                f"\nLogit adjustment: tau={logit_tau:.2f} "
                f"(val {args.select_metric} {base_acc:.4f} -> {tuned:.4f})"
            )
        else:
            logit_tau = float(args.logit_adjust)
            print(f"\nLogit adjustment: tau={logit_tau:.2f} (fixed)")
        # Re-save so the checkpoint carries the tuned tau to inference.
        torch.save(checkpoint_payload(best_state), output_dir / "best_model.pth")

    with open(output_dir / "label_map.json", "w") as f:
        json.dump({"idx_to_grade": idx_to_grade, "idx_to_name": idx_to_name}, f, indent=2)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ----------------- training curves -----------------
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs_range, history["train_loss"], label="train")
    axes[0].plot(epochs_range, history["val_loss"], label="val")
    axes[0].axvline(args.warmup_epochs + 0.5, color="gray", linestyle="--", linewidth=1, label="fine-tune starts")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_acc"], label="train")
    axes[1].plot(epochs_range, history["val_acc"], label="val")
    if any(not math.isnan(v) for v in history["ema_val_acc"]):
        axes[1].plot(epochs_range, history["ema_val_acc"], label="val (EMA)", linestyle=":")
    axes[1].axvline(args.warmup_epochs + 0.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # ----------------- final test evaluation -----------------
    if test_loader is not None:
        # TTA is always on for the final number: it costs one extra forward
        # pass per image and reliably adds a few tenths of a point.
        t_logits, t_flip, t_labels = collect_logits(model, test_loader, device, tta=True)
        adjust = None if logit_tau == 0 else logit_tau * torch.log(class_priors.clamp(min=1e-8))

        if adjust is not None:
            raw_preds = probs_with_adjust(t_logits, t_flip, None).argmax(dim=1)
            raw_acc = (raw_preds == t_labels).float().mean().item()
            print(f"TEST accuracy before logit adjustment: {raw_acc:.4f}")

        preds = probs_with_adjust(t_logits, t_flip, adjust).argmax(dim=1)
        test_acc = (preds == t_labels).float().mean().item()
        test_f1 = f1_score(t_labels.tolist(), preds.tolist(), average="macro", zero_division=0)

        report_and_plot(preds.tolist(), t_labels.tolist(), idx_to_name, output_dir, split_name="test")
        print(f"\nFinal TEST accuracy: {test_acc:.4f}  macro-F1: {test_f1:.4f}")
    else:
        print("\nNo test/ split available - skipping final test evaluation.")

    print(f"\nSaved to {output_dir}/: best_model.pth, label_map.json, history.json, training_curves.png" + (
        ", confusion_matrix_test.png, test_report.json" if test_loader is not None else ""
    ))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# ACCURACY NOTES
# ---------------------------------------------------------------------------
# Realistic expectations for 5-class KL grading on this dataset: published
# results sit around 68-75% test accuracy, and radiologist inter-observer
# agreement on KL grading is itself only about 60-70% (Grade 0 vs Grade 1 is
# genuinely ambiguous - they differ by "doubtful osteophytes"). Treat the
# within-one-grade accuracy printed above, which should land near 92-97%, as
# the honest measure of clinical usefulness.
#
# Roughly what each change contributes, largest first:
#   native resolution (224 -> 380 for B4) .... +3 to 6 points
#   MixUp/CutMix + stochastic depth .......... +2 to 4 points
#   sqrt-inverse instead of inverse weights ... +1 to 3 points (overall acc)
#   ordinal loss term ........................ +1 to 2 points, and markedly
#                                              fewer far-off errors
#   layer-wise LR decay ...................... +1 to 2 points
#   EMA ...................................... +1 to 2 points
#   flip TTA ................................. +0.5 to 1 point
#
# If you need more after this, the reliable next step is ensembling rather
# than a bigger single model: train 3 runs with --seed 42/43/44 and average
# their softmax outputs (inference.py --checkpoint a.pth,b.pth,c.pth). That
# is typically worth another 1-3 points.
