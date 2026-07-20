"""
training.py
============
Fine-tunes a torchvision EfficientNet (B0 by default) on the Knee
Osteoarthritis Severity Grading dataset (5 KL grades, 0-4).

    python training.py --data_dir "/path/to/Knee Osteoarthritis Dataset with Severity Grading"

Pipeline:
  1. Load data via prepare_dataset.get_dataloaders (also returns per-class
     weights to counter the dataset's heavy class imbalance).
  2. Build an ImageNet-pretrained EfficientNet with its 1000-way head
     replaced by a 5-way one.
  3. Phase 1 ("warm-up"): freeze the pretrained backbone, train only the
     new head for a few epochs so its random initialization doesn't send
     large, destructive gradients back through the pretrained features.
  4. Phase 2 ("fine-tune"): unfreeze everything, train end-to-end at a much
     lower learning rate with a cosine schedule.
  5. Track validation accuracy every epoch, keep the best checkpoint, stop
     early if it stalls.
  6. Evaluate the best checkpoint on the held-out test/ split and save a
     classification report, confusion matrix, and training curves.

Outputs (written to --output_dir, default "outputs/"):
  best_model.pth          - everything inference.py needs to reload the model
  label_map.json          - human-readable class-index -> KL grade mapping
  training_curves.png     - loss / accuracy per epoch
  confusion_matrix_test.png
  test_report.json        - precision / recall / f1 per class on the test set
"""

import argparse
import copy
import json
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b3,
)

from prepare_dataset import DEFAULT_OUTPUT_DIR, RANDOM_SEED, build_label_maps, get_dataloaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    arch = arch.lower()
    if arch == "b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
    elif arch == "b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b3(weights=weights)
    else:
        raise ValueError(f"Unsupported --arch '{arch}'. Choose 'b0' or 'b3'.")

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = trainable


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
) -> Tuple[float, float, float]:
    """One pass over `loader`. Trains if `optimizer` is given, else just evaluates."""
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, total_correct, total_seen = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            preds = outputs.detach().argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(total_seen, 1)
    accuracy = total_correct / max(total_seen, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if total_seen else 0.0
    return avg_loss, accuracy, macro_f1


def evaluate_and_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_to_name,
    output_dir: Path,
    split_name: str = "test",
) -> float:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    print(f"\n{split_name.upper()} SET RESULTS")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

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
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
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
    p.add_argument("--arch", type=str, default="b0", choices=["b0", "b3"], help="EfficientNet variant.")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--warmup_epochs", type=int, default=5, help="Epochs training only the classifier head.")
    p.add_argument("--finetune_epochs", type=int, default=25, help="Max epochs fine-tuning the whole network.")
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6, help="Early-stopping patience (in fine-tune epochs).")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--weighted_sampler", action="store_true", help="Also oversample rare classes per batch.")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes, class_weights = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.weighted_sampler,
    )
    if len(train_loader) == 0:
        raise RuntimeError(
            "The training DataLoader produced 0 batches - --batch_size is probably "
            "larger than the number of training images available. Lower --batch_size."
        )

    num_classes = len(classes)
    idx_to_grade, idx_to_name = build_label_maps(classes)
    print(f"Classes found ({num_classes}): {idx_to_name}")
    print(f"Class weights (imbalance correction): {class_weights.tolist()}")

    if args.no_pretrained:
        print(
            "WARNING: --no_pretrained set - training from random initialization. "
            "This will likely fall well short of 70-80% accuracy; only use this if "
            "your machine truly cannot reach download.pytorch.org."
        )
    model = build_model(args.arch, num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)

    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler = GradScaler(device.type, enabled=use_amp)
    print(f"Mixed precision (AMP): {use_amp}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())  # always a valid fallback, even at 0 epochs

    def checkpoint_payload():
        return {
            "model_state_dict": model.state_dict(),
            "arch": args.arch,
            "num_classes": num_classes,
            "idx_to_grade": idx_to_grade,
            "idx_to_name": idx_to_name,
            "img_size": args.img_size,
        }

    total_epochs = args.warmup_epochs + args.finetune_epochs
    epoch_counter = 0

    # ----------------- Phase 1: warm up the classifier head -----------------
    print(f"\n=== Phase 1/2: warming up the classifier head for {args.warmup_epochs} epoch(s) ===")
    set_backbone_trainable(model, False)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.head_lr, weight_decay=1e-4)

    for _ in range(args.warmup_epochs):
        epoch_counter += 1
        t0 = time.time()
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"[warmup {epoch_counter}/{total_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({time.time() - t0:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(checkpoint_payload(), output_dir / "best_model.pth")

    # ----------------- Phase 2: fine-tune everything -----------------
    print(f"\n=== Phase 2/2: fine-tuning the whole network for up to {args.finetune_epochs} epoch(s) ===")
    set_backbone_trainable(model, True)
    optimizer = AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.finetune_epochs, 1))

    epochs_no_improve = 0
    for _ in range(args.finetune_epochs):
        epoch_counter += 1
        t0 = time.time()
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        improved = val_acc > best_val_acc
        print(
            f"[finetune {epoch_counter}/{total_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            f"{'  (best so far)' if improved else ''} ({time.time() - t0:.1f}s)"
        )

        if improved:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(checkpoint_payload(), output_dir / "best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"No val_acc improvement for {args.patience} epochs - stopping early.")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_state)

    with open(output_dir / "label_map.json", "w") as f:
        json.dump({"idx_to_grade": idx_to_grade, "idx_to_name": idx_to_name}, f, indent=2)

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
        test_acc = evaluate_and_report(model, test_loader, device, idx_to_name, output_dir, split_name="test")
        print(f"\nFinal TEST accuracy: {test_acc:.4f}")
    else:
        print("\nNo test/ split available - skipping final test evaluation.")

    print(f"\nSaved to {output_dir}/: best_model.pth, label_map.json, training_curves.png" + (
        ", confusion_matrix_test.png, test_report.json" if test_loader is not None else ""
    ))


if __name__ == "__main__":
    main()