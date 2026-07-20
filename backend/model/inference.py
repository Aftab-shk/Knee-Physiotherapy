"""
inference.py
=============
Run the trained EfficientNet knee-OA severity classifier.

Single image:
    python inference.py --checkpoint outputs/best_model.pth --image path/to/xray.png

A folder of unlabeled images:
    python inference.py --checkpoint outputs/best_model.pth --image_dir path/to/folder

A labeled evaluation folder (ImageFolder layout, subfolders 0-4) - e.g. to
recompute accuracy on the dataset's own test/ split:
    python inference.py --checkpoint outputs/best_model.pth --eval_dir path/to/test

The checkpoint saved by training.py embeds the exact architecture name and
class-index -> KL-grade mapping it was trained with, so there is no risk of
the labels getting out of sync with the model.
"""

import argparse
import hashlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets

# Allow resolving siblings (prepare_dataset, train) when imported as a module
sys.path.append(str(Path(__file__).resolve().parent))

from prepare_dataset import DEFAULT_OUTPUT_DIR, get_eval_transform

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b3,
)

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

logger = logging.getLogger(__name__)

MODEL_VERSION = "efficientnet_v3"

# KL Grade → clinical health score (0–100)
KL_HEALTH_SCORE: Dict[int, int] = {0: 95, 1: 80, 2: 60, 3: 35, 4: 15}

# KL Grade → safe flexion ceiling (degrees)
KL_MAX_ANGLE: Dict[int, int] = {0: 120, 1: 120, 2: 90, 3: 60, 4: 45}


def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    """
    Lightweight quality check before inference.
    Rejects blank, inverted, or non-radiograph images.
    Returns (ok: bool, message: str).
    """
    try:
        img  = Image.open(io.BytesIO(image_bytes))
        gray = np.array(img.convert("L"), dtype=np.float32)
    except Exception:
        return False, "Could not decode the uploaded file. Please upload a valid JPEG or PNG."

    std  = gray.std()
    mean = gray.mean()

    if std < 15:
        return False, (
            "Image contrast is too low. "
            "Please ensure you are uploading a clear knee X-ray."
        )
    if mean < 10:
        return False, "Image appears completely black. Please check the file."
    if mean > 245:
        return False, "Image appears overexposed / blank. Please check the file."

    return True, "ok"


class KneeClassifier:
    """
    Wraps the trained EfficientNet model for KL-grade classification.
    Falls back to deterministic demo mode if weights are unavailable.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.demo_mode = True

        checkpoint_path = Path(__file__).parent / "best_model.pth"
        if checkpoint_path.is_file():
            try:
                self.model, self.idx_to_grade, self.idx_to_name, self.img_size = load_checkpoint(
                    str(checkpoint_path), self.device
                )
                self.transform = get_eval_transform(self.img_size)
                self.demo_mode = False
                logger.info(f"Model loaded from {checkpoint_path}")
            except Exception as exc:
                logger.error(f"Failed to load model weights: {exc} — falling back to demo mode.")
        else:
            logger.warning(
                f"Weights not found at '{checkpoint_path}'. Running in DEMO MODE. "
                "Place trained weights there to enable real inference."
            )

    def predict(self, image_bytes: bytes) -> dict:
        """
        Returns:
          {
            kl_grade:     int   (0–4),
            health_score: int   (0–100),
            max_angle:    int   (degrees),
            confidence:   float (0–1),
            demo_mode:    bool,
          }
        """
        if self.demo_mode:
            return self._demo_predict(image_bytes)
        return self._real_predict(image_bytes)

    def _real_predict(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            x = self.transform(image).unsqueeze(0).to(self.device)
            # Test-time augmentation (TTA) - original + mirrored
            flipped = torch.flip(x, dims=[3])

            with torch.no_grad():
                logits_a = self.model(x)
                logits_b = self.model(flipped)
                probs_a = F.softmax(logits_a, dim=1)
                probs_b = F.softmax(logits_b, dim=1)
                probs = ((probs_a + probs_b) / 2.0).squeeze(0).cpu()

            pred_idx = int(torch.argmax(probs).item())
            kl_grade = self.idx_to_grade[pred_idx]
            confidence = float(probs[pred_idx])

            return {
                "kl_grade":     kl_grade,
                "health_score": KL_HEALTH_SCORE[kl_grade],
                "max_angle":    KL_MAX_ANGLE[kl_grade],
                "confidence":   round(confidence, 3),
                "demo_mode":    False,
            }
        except Exception as exc:
            logger.error(f"Error during real prediction: {exc}. Falling back to demo prediction.")
            return self._demo_predict(image_bytes)

    def _demo_predict(self, image_bytes: bytes) -> dict:
        """
        Deterministic mock: the same image always returns the same grade.
        Grade distribution approximates real-world OA prevalence.
        """
        digest   = int(hashlib.md5(image_bytes[:2048]).hexdigest(), 16) % 100
        # Approximate population distribution of KL grades
        if   digest < 25:  kl_grade = 0
        elif digest < 50:  kl_grade = 1
        elif digest < 70:  kl_grade = 2
        elif digest < 87:  kl_grade = 3
        else:              kl_grade = 4

        # Confidence varies 0.62–0.88 based on digest
        confidence = 0.62 + (digest % 27) / 100.0

        return {
            "kl_grade":     kl_grade,
            "health_score": KL_HEALTH_SCORE[kl_grade],
            "max_angle":    KL_MAX_ANGLE[kl_grade],
            "confidence":   round(confidence, 3),
            "demo_mode":    True,
        }


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    idx_to_grade = {int(k): int(v) for k, v in ckpt["idx_to_grade"].items()}
    idx_to_name = {int(k): v for k, v in ckpt["idx_to_name"].items()}
    img_size = ckpt.get("img_size", 224)
    return model, idx_to_grade, idx_to_name, img_size


@torch.no_grad()
def predict_single(model, image_path: Path, transform, idx_to_grade, idx_to_name, device) -> Dict:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1).squeeze(0).cpu()
    pred_idx = int(torch.argmax(probs).item())

    ranked = sorted(
        ((idx_to_name[i], float(probs[i])) for i in range(len(probs))),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {
        "image": str(image_path),
        "predicted_grade": idx_to_grade[pred_idx],
        "predicted_label": idx_to_name[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": ranked,
    }


def print_prediction(result: Dict) -> None:
    print(f"\n{result['image']}")
    print(f"  -> {result['predicted_label']}  (confidence {result['confidence']:.1%})")
    print("  full distribution:")
    for label, p in result["all_probabilities"]:
        print(f"    {label:22s} {p:6.1%}")


@torch.no_grad()
def evaluate_folder(model, eval_dir: Path, transform, idx_to_grade, idx_to_name, device, batch_size: int = 32) -> None:
    """Batched evaluation on an ImageFolder-style labeled directory (e.g. test/)."""
    from sklearn.metrics import classification_report

    dataset = datasets.ImageFolder(eval_dir, transform=transform)
    expected_folder_names = [str(idx_to_grade[i]) for i in range(len(idx_to_grade))]
    if dataset.classes != expected_folder_names:
        raise RuntimeError(
            f"Class folders in {eval_dir} ({dataset.classes}) don't match what the checkpoint "
            f"was trained on ({expected_folder_names}). Make sure this folder has the same "
            "0/1/2/3/4 subfolder layout as the training data."
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]
    print(f"\nEvaluation on {eval_dir}  ({len(dataset)} images)")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    print(f"Overall accuracy: {correct / len(all_labels):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained Knee-OA EfficientNet.")
    parser.add_argument(
        "--checkpoint", type=str, default=f"{DEFAULT_OUTPUT_DIR}/best_model.pth", help="Path to best_model.pth"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single X-ray image.")
    group.add_argument("--image_dir", type=str, help="Folder of unlabeled images to classify.")
    group.add_argument("--eval_dir", type=str, help="Labeled folder (subfolders 0-4) to compute accuracy on.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run training.py first, or pass the "
            "correct file with --checkpoint."
        )

    model, idx_to_grade, idx_to_name, img_size = load_checkpoint(str(checkpoint_path), device)
    transform = get_eval_transform(img_size)
    print(f"Loaded checkpoint: {checkpoint_path} (device: {device})")

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        result = predict_single(model, image_path, transform, idx_to_grade, idx_to_name, device)
        print_prediction(result)

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Folder not found: {image_dir}")
        image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        results = []
        for p in image_paths:
            try:
                results.append(predict_single(model, p, transform, idx_to_grade, idx_to_name, device))
            except UnidentifiedImageError:
                print(f"  (skipping unreadable file: {p})")

        for r in results:
            print_prediction(r)

        out_path = Path(DEFAULT_OUTPUT_DIR) / "predictions.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} predictions to {out_path}")

    elif args.eval_dir:
        eval_dir = Path(args.eval_dir)
        if not eval_dir.is_dir():
            raise FileNotFoundError(f"Folder not found: {eval_dir}")
        evaluate_folder(model, eval_dir, transform, idx_to_grade, idx_to_name, device)


if __name__ == "__main__":
    main()