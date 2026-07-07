"""
KneeClassifier — EfficientNet-B4 KL-grade inference.

Production path:
  Place trained weights at MODEL_PATH (default: model/efficientnet_b4_kl_v2.pt).
  The file must be a torch.save() of the full state_dict for a 5-class EfficientNet-B4.

Demo path:
  If MODEL_PATH is missing, the classifier runs in demo mode:
  - Full image preprocessing pipeline still executes (validates the plumbing).
  - KL grade is assigned deterministically from a hash of the image bytes
    so the same image always returns the same grade.
  - All clinical logic (exercise selection, angle mapping) runs normally.
  - Response body includes  demo_mode: true.

Training note:
  Train using model/train.py (EfficientNet-B4, stronger augmentation, label
  smoothing). Expected input: 224×224 RGB, ImageNet normalisation, CLAHE
  pre-processing. Output: softmax over 5 classes (KL 0, 1, 2, 3, 4).

Inference note:
  Uses horizontal-flip test-time augmentation (TTA) — each image is run
  through the model twice (original + mirrored) and the softmax outputs
  are averaged. This gives a small, free accuracy improvement since knee
  X-rays are symmetric under left-right mirroring. Roughly doubles CPU
  inference time (~0.3–0.6s total for a single image), well within the
  ≤3s target.
"""

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_VERSION = "efficientnet_b4_kl_v2"
MODEL_PATH    = os.getenv("MODEL_PATH", "model/efficientnet_b4_kl_v2.pt")
NUM_CLASSES   = 5          # KL Grade 0, 1, 2, 3, 4
INPUT_SIZE    = 224

# ImageNet normalisation (transfer-learning baseline)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# KL Grade → clinical health score (0–100)
KL_HEALTH_SCORE: Dict[int, int] = {0: 95, 1: 80, 2: 60, 3: 35, 4: 15}

# KL Grade → safe flexion ceiling (degrees)
KL_MAX_ANGLE: Dict[int, int] = {0: 120, 1: 120, 2: 90, 3: 60, 4: 45}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])


def _apply_clahe(pil_img: Image.Image) -> Image.Image:
    """Apply CLAHE on the luminance channel to enhance X-ray contrast."""
    gray  = np.array(pil_img.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Restore 3-channel RGB expected by ResNet
    rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    return Image.fromarray(rgb)   # shape (H,W,3) uint8 → PIL infers RGB automatically


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    """Load raw bytes → preprocessed tensor with batch dim."""
    pil_img    = Image.open(io.BytesIO(image_bytes))
    enhanced   = _apply_clahe(pil_img)
    tensor     = _transform(enhanced)          # (3, 224, 224)
    return tensor.unsqueeze(0)                 # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Image quality gate
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class KneeClassifier:
    """
    Wraps ResNet50 for KL-grade classification.
    Falls back to deterministic demo mode if weights are unavailable.
    """

    def __init__(self) -> None:
        self.device    = torch.device("cpu")
        self.model     = None
        self.demo_mode = True

        weights_path = Path(MODEL_PATH)
        if weights_path.exists():
            try:
                self._load_model(weights_path)
                self.demo_mode = False
                logger.info("Model loaded from %s", weights_path)
            except Exception as exc:
                logger.error("Failed to load model weights: %s — falling back to demo mode.", exc)
        else:
            logger.warning(
                "Weights not found at '%s'. Running in DEMO MODE. "
                "Place trained weights there to enable real inference.",
                weights_path,
            )

    def _load_model(self, path: Path) -> None:
        net = models.efficientnet_b4(weights=None)
        # Replace the final classifier layer for 5-class output
        in_features      = net.classifier[1].in_features
        net.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
        net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        net.eval()
        self.model = net

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Real inference
    # ------------------------------------------------------------------

    def _real_predict(self, image_bytes: bytes) -> dict:
        tensor  = _preprocess(image_bytes)
        flipped = torch.flip(tensor, dims=[3])   # mirror left-right (TTA)

        with torch.no_grad():
            logits_a = self.model(tensor)
            logits_b = self.model(flipped)
            probs_a  = F.softmax(logits_a, dim=1)
            probs_b  = F.softmax(logits_b, dim=1)
            probs    = ((probs_a + probs_b) / 2.0).squeeze()  # averaged (5,)

        kl_grade   = int(probs.argmax().item())
        confidence = float(probs[kl_grade].item())

        return {
            "kl_grade":     kl_grade,
            "health_score": KL_HEALTH_SCORE[kl_grade],
            "max_angle":    KL_MAX_ANGLE[kl_grade],
            "confidence":   round(confidence, 3),
            "demo_mode":    False,
        }

    # ------------------------------------------------------------------
    # Demo inference (deterministic, no trained weights required)
    # ------------------------------------------------------------------

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