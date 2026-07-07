"""
train.py — EfficientNet-B4 KL-grade classifier (v3 — stability fix)
=====================================================================

What changed from v2:
  1. PIL fix: Image.fromarray(rgb, mode="RGB") → Image.fromarray(rgb)
     Removes the DeprecationWarning that spammed every epoch.
  2. Learning rate lowered: 1e-4 → 3e-5 (head), backbone at 3e-6.
     EfficientNet-B4 is sensitive — too-high LR causes the divergence
     (loss stuck, model defaults to predicting one class) seen in v2.
  3. Warmup added: LR ramps linearly from near-zero to the target over
     the first 5 epochs, preventing the loss spike that triggers collapse.
  4. Unfreeze later: epoch 15 (was 8/12). More head stability before
     the backbone starts moving.
  5. Early stopping: training stops automatically if val_acc has not
     improved for 10 consecutive epochs — saves Colab time if the model
     converges early.
  6. Gradient clipping tightened: max_norm 1.0 → 0.5.
  7. Batch-size default lowered: 32 → 16. EfficientNet-B4 is bigger than
     ResNet50; larger batches create noisier gradients at the same LR.
  8. Sanity check: prints the first batch's grade distribution so you can
     confirm the sampler is working correctly before waiting 3 hours.

Expected result on 12k OSAIL + Kaggle combined dataset:
  Test accuracy: 68–76% (EfficientNet-B4 + all fixes applied)
  Grade 4 recall: ≥ 75%

Run in Google Colab (T4 GPU):
  !python train.py --data data --epochs 40 --batch-size 16

If you hit CUDA out-of-memory:
  !python train.py --data data --epochs 40 --batch-size 8
"""

import argparse
import copy
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR       = "data"
DEFAULT_EPOCHS         = 40
DEFAULT_BATCH_SIZE     = 16           # safer default for EfficientNet-B4
DEFAULT_LR             = 3e-5         # lower LR — key stability fix
DEFAULT_OUT_PATH       = "model/efficientnet_b4_kl_v2.pt"
DEFAULT_UNFREEZE_EPOCH = 15           # later unfreeze for more head stability
DEFAULT_WARMUP_EPOCHS  = 5            # LR warmup period
DEFAULT_PATIENCE       = 10           # early stopping patience
NUM_CLASSES            = 5
INPUT_SIZE             = 224
NUM_WORKERS            = 2            # 2 is safer on Colab than 4

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# CLAHE preprocessing — MUST MATCH inference.py exactly. DO NOT CHANGE.
# ---------------------------------------------------------------------------

def apply_clahe(pil_img: Image.Image) -> Image.Image:
    gray     = np.array(pil_img.convert("L"), dtype=np.uint8)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb      = np.stack([enhanced, enhanced, enhanced], axis=-1)
    return Image.fromarray(rgb)       # shape (H,W,3) uint8 → PIL infers RGB


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KneeXrayDataset(Dataset):

    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    ])

    EVAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    def __init__(self, root: str, split: str = "train"):
        self.root      = Path(root) / split
        self.is_train  = (split == "train")
        self.transform = self.TRAIN_TRANSFORMS if self.is_train else self.EVAL_TRANSFORMS
        self.samples   = []

        for label in range(NUM_CLASSES):
            class_dir = self.root / str(label)
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}. "
                    f"Run prepare_dataset.py first."
                )
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for fp in class_dir.glob(ext):
                    self.samples.append((fp, label))

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

        print(f"\n[{split}] {len(self.samples)} images")
        counts = [sum(1 for _, l in self.samples if l == i) for i in range(NUM_CLASSES)]
        for i, c in enumerate(counts):
            bar = "█" * (c // 100)
            print(f"  KL {i}: {c:>5}  {bar}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img         = Image.open(path).convert("RGB")
        img         = apply_clahe(img)
        tensor      = self.transform(img)
        return tensor, label

    def class_weights(self) -> torch.Tensor:
        counts  = np.array([sum(1 for _, l in self.samples if l == i)
                            for i in range(NUM_CLASSES)], dtype=np.float32)
        weights = 1.0 / (counts + 1e-6)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def sample_weights(self) -> list:
        cw = self.class_weights().numpy()
        return [cw[label] for _, label in self.samples]


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience  = patience
        self.counter   = 0
        self.best_acc  = 0.0

    def __call__(self, val_acc: float) -> bool:
        """Returns True if training should stop."""
        if val_acc > self.best_acc + 0.001:   # must improve by at least 0.1%
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience} epochs without improvement "
                  f"(best: {self.best_acc:.1f}%)")
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Model — EfficientNet-B4
# ---------------------------------------------------------------------------

def build_model(freeze_backbone: bool = True) -> nn.Module:
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

    # EfficientNet-B4 classifier: Sequential(Dropout(0.4), Linear(1792, 1000))
    in_features          = model.classifier[1].in_features   # 1792
    model.classifier[1]  = nn.Linear(in_features, NUM_CLASSES)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Model: EfficientNet-B4 | "
          f"Trainable params: {trainable:,} / {total:,} "
          f"({'frozen backbone' if freeze_backbone else 'full fine-tune'})")
    return model


def unfreeze_backbone(model: nn.Module, lr: float, optimizer) -> optim.Optimizer:
    """Unfreeze all layers and rebuild optimiser with lower backbone LR."""
    for param in model.parameters():
        param.requires_grad = True

    head_params     = [p for n, p in model.named_parameters() if "classifier" in n]
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    new_opt = optim.AdamW([
        {"params": head_params,     "lr": lr},
        {"params": backbone_params, "lr": lr * 0.1},  # backbone at 10× lower LR
    ], weight_decay=1e-4)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Backbone unfrozen | Trainable: {trainable:,} / {total:,}")
    return new_opt


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, first_epoch=False):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0
    grade_counts = [0] * NUM_CLASSES   # sanity check for first epoch

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if first_epoch and batch_idx == 0:
            for l in labels.cpu().numpy():
                grade_counts[l] += 1

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

        if (batch_idx + 1) % 30 == 0:
            print(f"  Epoch {epoch} | Step {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    if first_epoch:
        print(f"  First batch grade distribution: {grade_counts}")

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds    = []
    all_labels   = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs        = model(inputs)
        loss           = criterion(outputs, labels)
        running_loss  += loss.item() * inputs.size(0)
        _, predicted   = outputs.max(1)
        all_preds .extend(predicted.cpu().numpy())
        all_labels.extend(labels  .cpu().numpy())

    total    = len(all_labels)
    val_loss = running_loss / total
    val_acc  = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / total
    return val_loss, val_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy (%)"); axes[1].legend()
    plt.tight_layout()
    path = Path(out_dir) / "training_curves.png"
    plt.savefig(path, dpi=120); print(f"Curves → {path}"); plt.close()


def plot_confusion_matrix(labels, preds, out_dir: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels([f"KL{i}" for i in range(NUM_CLASSES)])
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels([f"KL{i}" for i in range(NUM_CLASSES)])
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix (Test)")
    plt.tight_layout()
    path = Path(out_dir) / "confusion_matrix.png"
    plt.savefig(path, dpi=120); print(f"Confusion matrix → {path}"); plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 KL classifier (v3)")
    parser.add_argument("--data",            default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",      type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",              type=float, default=DEFAULT_LR)
    parser.add_argument("--out",             default=DEFAULT_OUT_PATH)
    parser.add_argument("--unfreeze-epoch",  type=int,   default=DEFAULT_UNFREEZE_EPOCH)
    parser.add_argument("--warmup-epochs",   type=int,   default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--patience",        type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers",         type=int,   default=NUM_WORKERS)
    parser.add_argument("--no-sampler",      action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"\n{'='*60}")
    print(f" EfficientNet-B4 KL Classifier — Training (v3)")
    print(f"{'='*60}")
    print(f" Device          : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " ⚠️ CPU — very slow"))
    print(f" Data dir        : {args.data}")
    print(f" Epochs          : {args.epochs}")
    print(f" Batch size      : {args.batch_size}")
    print(f" LR (head)       : {args.lr}  backbone: {args.lr * 0.1}")
    print(f" Warmup epochs   : {args.warmup_epochs}")
    print(f" Unfreeze epoch  : {args.unfreeze_epoch}")
    print(f" Early stop pat. : {args.patience}")
    print(f" Label smoothing : {args.label_smoothing}")
    print(f" Output          : {args.out}")
    print(f"{'='*60}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = KneeXrayDataset(args.data, split="train")
    val_ds   = KneeXrayDataset(args.data, split="val")
    test_ds  = KneeXrayDataset(args.data, split="test")

    if args.no_sampler:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers,
                                  pin_memory=(device.type == "cuda"))
    else:
        sw      = train_ds.sample_weights()
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.workers,
                                  pin_memory=(device.type == "cuda"))

    val_loader  = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(freeze_backbone=(args.unfreeze_epoch > 0)).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────────
    class_weights = train_ds.class_weights().to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # ── Optimiser ──────────────────────────────────────────────────────────────
    head_params     = [p for n, p in model.named_parameters() if "classifier" in n]
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    optimizer = optim.AdamW([
        {"params": head_params,     "lr": args.lr},
        {"params": backbone_params, "lr": args.lr * 0.1},
    ], weight_decay=1e-4)

    # ── Scheduler: warmup → cosine annealing ──────────────────────────────────
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.01,
        end_factor   = 1.0,
        total_iters  = args.warmup_epochs,
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = max(args.epochs - args.warmup_epochs, 1),
        eta_min = 1e-7,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup_sched, cosine_sched],
        milestones = [args.warmup_epochs],
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    best_weights  = None
    early_stop    = EarlyStopping(patience=args.patience)
    out_path      = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone at configured epoch
        if epoch == args.unfreeze_epoch:
            optimizer = unfreeze_backbone(model, args.lr, optimizer)
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max   = max(args.epochs - epoch + 1, 1),
                eta_min = 1e-7,
            )
            scheduler = cosine_sched   # switch to plain cosine after unfreeze

        train_loss, train_acc     = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            first_epoch=(epoch == 1)
        )
        val_loss, val_acc, _, _   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        print(f"\nEpoch {epoch:>3}/{args.epochs} | "
              f"Train {train_acc:.1f}% (loss {train_loss:.4f}) | "
              f"Val {val_acc:.1f}% (loss {val_loss:.4f}) | "
              f"LR {current_lr:.2e} | {elapsed:.0f}s\n")

        history["train_loss"].append(train_loss)
        history["train_acc"] .append(train_acc)
        history["val_loss"]  .append(val_loss)
        history["val_acc"]   .append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, out_path)
            print(f"  ✅ New best val acc: {best_val_acc:.1f}% — saved to {out_path}")

        if early_stop(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # ── Final test evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(" Test set evaluation")
    print(f"{'='*60}")
    model.load_state_dict(best_weights)
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.1f}%\n")
    print(classification_report(test_labels, test_preds,
                                target_names=[f"KL{i}" for i in range(NUM_CLASSES)]))

    out_dir = str(out_path.parent)
    plot_history(history, out_dir)
    plot_confusion_matrix(test_labels, test_preds, out_dir)

    print(f"\n{'='*60}")
    print(f" Training complete")
    print(f" Best val accuracy : {best_val_acc:.1f}%")
    print(f" Test accuracy     : {test_acc:.1f}%")
    print(f" Weights saved to  : {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()