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
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets

# Allow resolving siblings (prepare_dataset, train) when imported as a module
sys.path.append(str(Path(__file__).resolve().parent))

import torch.nn as nn
from prepare_dataset import DEFAULT_OUTPUT_DIR, get_eval_transform
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

# Must stay in sync with train.py's _ARCHS - a checkpoint records the arch it
# was trained with, and loading fails loudly here if that arch is unknown.
_ARCHS = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights),
    "b1": (efficientnet_b1, EfficientNet_B1_Weights),
    "b2": (efficientnet_b2, EfficientNet_B2_Weights),
    "b3": (efficientnet_b3, EfficientNet_B3_Weights),
    "b4": (efficientnet_b4, EfficientNet_B4_Weights),
    "b5": (efficientnet_b5, EfficientNet_B5_Weights),
}


def build_model(
    arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.4
) -> nn.Module:
    arch = arch.lower()
    if arch not in _ARCHS:
        raise ValueError(f"Unsupported arch '{arch}'. Choose one of {sorted(_ARCHS)}.")

    ctor, weights_enum = _ARCHS[arch]
    model = ctor(weights=weights_enum.IMAGENET1K_V1 if pretrained else None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model

logger = logging.getLogger(__name__)

# Fallback only. The real version is derived from the loaded checkpoint by
# _derive_model_version() so a prescription can be traced to the exact weights
# that produced it.
MODEL_VERSION = "efficientnet_b4_v2"

# Shared with clinical_logic via backend/kl_constants.py — see that module for
# why these are not defined twice any more.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.append(str(_BACKEND_DIR))
from kl_constants import KL_HEALTH_SCORE, KL_MAX_ANGLE

# Calibrated-confidence bands. Reporting "82%" to a patient beside a movement
# restriction implies a precision the model does not have, so the API returns a
# band and the UI leads with it.
CONFIDENCE_BANDS = ((0.75, "high"), (0.50, "moderate"), (0.0, "low"))


def confidence_band(confidence: float) -> str:
    for floor, label in CONFIDENCE_BANDS:
        if confidence >= floor:
            return label
    return "low"



# validate_image lives in image_checks so that it can be imported without torch
# (see that module). Re-exported here because the CLI below and existing callers
# import it from this module — F401 is the re-export, not a stray import.
from image_checks import validate_image  # noqa: F401


class KneeClassifier:
    """
    Wraps the trained EfficientNet model for KL-grade classification.
    Falls back to deterministic demo mode if weights are unavailable.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.demo_mode = True
        self.logit_adjust = None
        self.calibration = {"temperature": 1.0, "calibrated": False,
                            "warn_threshold": None, "reject_threshold": None, "ece": None}
        self.model_version = MODEL_VERSION

        # MODEL_PATH lets the container mount weights anywhere; relative paths
        # resolve against this directory so the default stays a bare filename.
        env_path = os.getenv("MODEL_PATH", "").strip()
        checkpoint_path = Path(env_path) if env_path else Path("best_model.pth")
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).parent / checkpoint_path

        if checkpoint_path.is_file():
            try:
                (
                    self.model,
                    self.idx_to_grade,
                    self.idx_to_name,
                    self.img_size,
                    self.clahe,
                ) = load_checkpoint(str(checkpoint_path), self.device)
                self.transform = get_eval_transform(self.img_size, clahe=self.clahe)
                adjust = load_logit_adjust(str(checkpoint_path))
                self.logit_adjust = adjust.to(self.device) if adjust is not None else None
                self.calibration = load_calibration(str(checkpoint_path))
                self.model_version = _derive_model_version(
                    torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
                )
                self.demo_mode = False
                logger.info(
                    f"Model loaded from {checkpoint_path} "
                    f"(version={self.model_version}, img_size={self.img_size}, clahe={self.clahe})"
                )
                if not self.calibration["calibrated"]:
                    logger.warning(
                        "Checkpoint carries no temperature — confidence is raw softmax and "
                        "is NOT calibrated. Retrain with calibration enabled before "
                        "presenting these numbers to patients."
                    )
                if self.calibration["reject_threshold"] is None:
                    logger.warning(
                        "Checkpoint carries no OOD energy reference — non-radiograph images "
                        "cannot be screened out and will receive a KL grade."
                    )
                if self.logit_adjust is None:
                    logger.warning(
                        "Checkpoint carries no class-prior correction (logit_tau/class_priors). "
                        "Serving does not match the tuned test numbers for this recipe."
                    )
            except Exception:
                logger.exception(
                    "Failed to load model weights from %s — falling back to DEMO MODE. "
                    "Predictions will be deterministic mocks, not readings.",
                    checkpoint_path,
                )
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
        """
        Raises on failure rather than falling back to the demo predictor.

        The demo grade is an MD5 of the image bytes. Serving that as though it
        were a reading — where it goes on to set a movement ceiling — is worse
        than returning an error, so an inference failure is allowed to surface
        as a 500 and main.py turns it into "try a different image".
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        # Test-time augmentation (TTA) - original + mirrored
        flipped = torch.flip(x, dims=[3])
        T = self.calibration["temperature"]

        with torch.no_grad():
            raw_a = self.model(x)
            raw_b = self.model(flipped)

            # Energy score is read off the RAW logits, before any correction —
            # the reference percentiles in the checkpoint were fitted the same
            # way, and prior correction/temperature would shift the scale.
            energy = float(
                (-torch.logsumexp(raw_a.float(), dim=1)
                 - torch.logsumexp(raw_b.float(), dim=1)).mean() / 2.0
            )

            # Same class-prior correction the checkpoint was tuned with,
            # applied to logits before softmax so serving matches the
            # reported test accuracy.
            logits_a, logits_b = raw_a, raw_b
            if self.logit_adjust is not None:
                logits_a = logits_a - self.logit_adjust
                logits_b = logits_b - self.logit_adjust

            # Temperature scaling, fitted on val. Monotonic per-branch, but the
            # two TTA branches are averaged after softmax, so T can flip a
            # borderline case (measured: 1 in 1656).
            probs_a = F.softmax(logits_a / T, dim=1)
            probs_b = F.softmax(logits_b / T, dim=1)
            probs = ((probs_a + probs_b) / 2.0).squeeze(0).cpu()

        pred_idx = int(torch.argmax(probs).item())
        kl_grade = self.idx_to_grade[pred_idx]
        confidence = float(probs[pred_idx])

        # The whole distribution, keyed by KL grade rather than model index —
        # the two are not the same, and a checkpoint is free to order its classes
        # however it was trained.
        #
        # Worth surfacing because the model was trained with an ordinal loss: the
        # mass next to the winner is not noise, it is the model saying the answer
        # is nearby. "Probably 2, possibly 3" is both truer and more useful than
        # "Grade 2, moderate confidence", especially where the difference between
        # 2 and 3 is a 30-degree difference in what someone is allowed to bend to.
        by_grade = {self.idx_to_grade[i]: float(probs[i]) for i in range(len(probs))}
        grade_probabilities = [round(by_grade.get(g, 0.0), 4) for g in range(5)]

        # How much probability sits within one grade of the winner. The
        # checkpoint reports 95.3% within-one-grade accuracy against 70.3%
        # exact, so this is the number that actually describes how confident the
        # reading is at the scale the ceiling changes.
        neighbourhood = sum(
            grade_probabilities[g] for g in (kl_grade - 1, kl_grade, kl_grade + 1)
            if 0 <= g <= 4
        )

        warn_t = self.calibration["warn_threshold"]
        reject_t = self.calibration["reject_threshold"]

        return {
            "kl_grade":          kl_grade,
            "health_score":      KL_HEALTH_SCORE[kl_grade],
            "max_angle":         KL_MAX_ANGLE[kl_grade],
            "confidence":        round(confidence, 3),
            "confidence_band":   confidence_band(confidence),
            "grade_probabilities": grade_probabilities,
            "within_one_grade":  round(neighbourhood, 3),
            "calibrated":        self.calibration["calibrated"],
            "energy":            round(energy, 3),
            "ood_suspected":     warn_t is not None and energy > warn_t,
            "ood_reject":        outside_energy_range(energy, self.calibration),
            "ood_screened":      reject_t is not None,
            "demo_mode":         False,
        }

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

        # A distribution shaped like a real one — mass on the winner, the rest
        # spilling onto its neighbours as an ordinal model's would — so the
        # frontend has something correctly shaped to draw. It is arithmetic on a
        # hash, not a probability, which is why `calibrated` stays false and the
        # UI leads with the demo banner.
        remainder = 1.0 - confidence
        weights = [1.0 / (1 + 2 * abs(g - kl_grade)) if g != kl_grade else 0.0 for g in range(5)]
        total = sum(weights) or 1.0
        grade_probabilities = [
            round(confidence if g == kl_grade else remainder * weights[g] / total, 4)
            for g in range(5)
        ]
        neighbourhood = sum(
            grade_probabilities[g] for g in (kl_grade - 1, kl_grade, kl_grade + 1)
            if 0 <= g <= 4
        )

        return {
            "kl_grade":        kl_grade,
            "health_score":    KL_HEALTH_SCORE[kl_grade],
            "max_angle":       KL_MAX_ANGLE[kl_grade],
            "confidence":      round(confidence, 3),
            "confidence_band": confidence_band(confidence),
            "grade_probabilities": grade_probabilities,
            "within_one_grade":  round(neighbourhood, 3),
            # Demo confidence is a hash, not a probability. Never claim it is
            # calibrated, and never claim an OOD screen ran.
            "calibrated":      False,
            "energy":          None,
            "ood_suspected":   False,
            "ood_reject":      False,
            "ood_screened":    False,
            "demo_mode":       True,
        }


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Returns (model, idx_to_grade, idx_to_name, img_size, clahe).

    Input resolution and CLAHE are read back out of the checkpoint rather than
    hardcoded, so inference preprocessing always matches what the model was
    trained on. A B4 trained at 380px fed 224px images would silently lose a
    large chunk of its accuracy.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(
        ckpt["arch"], ckpt["num_classes"], pretrained=False, dropout=ckpt.get("dropout", 0.4)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    idx_to_grade = {int(k): int(v) for k, v in ckpt["idx_to_grade"].items()}
    idx_to_name = {int(k): v for k, v in ckpt["idx_to_name"].items()}
    img_size = ckpt.get("img_size", 224)
    clahe = bool(ckpt.get("clahe", False))
    return model, idx_to_grade, idx_to_name, img_size, clahe


def _derive_model_version(ckpt: dict) -> str:
    """
    Build the version string from what the checkpoint actually contains rather
    than a hardcoded constant, so /health and every prescription name the
    weights that produced them.
    """
    arch = ckpt.get("arch", "unknown")
    parts = [f"efficientnet_{arch}"]
    trained = ckpt.get("trained_at")
    if trained:
        parts.append(str(trained)[:10].replace("-", ""))
    if ckpt.get("temperature", 1.0) != 1.0:
        parts.append("cal")
    return "_".join(parts)


def outside_energy_range(energy: float, calibration: dict) -> bool:
    """
    Is this energy score outside the band real knee films occupy?

    Both ends matter. Too high is a degenerate or near-blank image; too low is
    an image the network answers with runaway activations, which is what a
    photograph or a screenshot does. Inert while the checkpoint carries no
    energy reference, which is also why the upper bound decides that.
    """
    high = calibration.get("reject_threshold")
    low = calibration.get("floor_threshold")
    if high is None:
        return False
    return energy > high or (low is not None and energy < low)


def load_calibration(checkpoint_path: str) -> dict:
    """
    Read the temperature and OOD energy reference fitted on the validation split
    at the end of training.

    Both are optional. A checkpoint trained before this existed carries neither,
    in which case serving falls back to raw softmax and reports
    calibrated=False rather than implying a confidence it never validated.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    temperature = float(ckpt.get("temperature", 1.0) or 1.0)
    energy_ref = ckpt.get("energy_ref") or {}

    reject_threshold = None
    warn_threshold = None
    floor_threshold = None
    if energy_ref.get("p99") is not None and energy_ref.get("p50") is not None:
        p50, p95, p99 = energy_ref["p50"], energy_ref.get("p95", energy_ref["p99"]), energy_ref["p99"]
        # p95 by design, so about 1 genuine film in 20 carries the "unusual image"
        # note. Measured on the 1656 films of the Kaggle test split: 5.25%. A QA
        # report that it "flags nearly everything" came from synthetic test images,
        # which really are unusual; on real films it does what it says.
        warn_threshold = p95
        # How far past p99 an image has to score before it is refused outright,
        # measured in p50-to-p99 spreads.
        #
        # This was a hard-coded 1.0, which put the bar at -1.31 for the shipped
        # checkpoint. Nothing reaches that. A chest film scored -1.71, random
        # noise -1.71, a blank white image -1.52; all were graded, and the chest
        # film came back as a healthy knee with a 120-degree ceiling. A gate that
        # cannot fire is not a gate.
        #
        # 0.5 puts the bar at -1.571 for the shipped checkpoint. Checked on the
        # 1656 films of the Kaggle test split, which the model never trained on:
        # none of them was refused. The highest-scoring genuine film sat at -1.653,
        # so any margin above about 0.34 refuses nothing on that set.
        #
        # Be clear about how weak this gate is, because the old value hid it. A
        # chest radiograph scores -1.71 on this checkpoint and random noise
        # -1.71, both sitting comfortably inside the range real knee films
        # occupy. Catching those would mean a threshold near -1.72 — a margin of
        # about 0.21. This used to say that throws away more than 1% of genuine
        # studies; measured on the real test split it refuses 1 film in 1656
        # (0.06%). So tightening is far cheaper than was assumed. What stops it
        # is the other side of the ledger: one chest film and one noise image,
        # each 0.01 past that line, are not evidence that the next chest film
        # would be caught too. A patient whose own X-ray is refused cannot use
        # the app at all, while a misread one is held to a cautious ceiling by
        # build_prescription(), so this still errs towards letting images through
        # until there is a set of negatives to measure against.
        #
        # So: this rejects blank, uniform and near-degenerate uploads, most of
        # which the contrast check in image_checks.py already catches. It is not
        # protection against the wrong body part, and nothing downstream should
        # be written as though it were. Tuning it into something that is needs a
        # labelled out-of-distribution set, which is why it is an environment
        # variable and why every rejection logs its energy.
        margin = float(os.getenv("OOD_REJECT_MARGIN", "0.5"))
        reject_threshold = p99 + max(margin * (p99 - p50), 1e-3)

        # The other side of the same gate, and the side that was open.
        #
        # Energy is minus a log-sum-exp of the logits, so an image the network
        # answers with enormous activations scores very NEGATIVE, not positive.
        # Only the upper bound existed, so those sailed through. Measured on the
        # shipped checkpoint: a colour portrait scored -48285 and came back KL 4
        # at 100% confidence, a screenshot of this app's own landing page -28651
        # and came back KL 2 at 100%, an app icon -4595, a screenshot of a
        # tutorial dialog -215.
        #
        # All 1656 films of the Kaggle test split sit between -3.737 and -1.712,
        # p50 -2.493. Eight p50-to-p99 spreads below p50 puts the floor near
        # -7.1: about three spreads clear of the lowest genuine film, and orders
        # of magnitude above the junk. Nothing on that split is refused by it,
        # and every image listed above is.
        floor_spreads = float(os.getenv("OOD_FLOOR_SPREADS", "8"))
        floor_threshold = p50 - max(floor_spreads * (p99 - p50), 1e-3)

    return {
        "temperature":      temperature,
        "calibrated":       temperature != 1.0,
        "energy_ref":       energy_ref,
        "warn_threshold":   warn_threshold,
        "reject_threshold": reject_threshold,
        "floor_threshold":  floor_threshold,
        "ece":              (ckpt.get("calibration") or {}).get("ece_after"),
    }


def load_logit_adjust(checkpoint_path: str) -> Optional[torch.Tensor]:
    """
    Rebuild the class-prior correction the checkpoint was tuned with.

    The model is trained on data where Grade 0 outnumbers Grade 4 ~13x, so it
    learns to lean on that prior. train.py sweeps a correction strength (tau)
    on the validation split; applying the same correction here keeps serving
    predictions identical to the reported test numbers. Returns None when the
    checkpoint predates this or was trained with --logit_adjust off.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tau = float(ckpt.get("logit_tau", 0.0) or 0.0)
    priors = ckpt.get("class_priors")
    if tau == 0.0 or not priors:
        return None
    return tau * torch.log(torch.tensor(priors, dtype=torch.float).clamp(min=1e-8))


@torch.no_grad()
def predict_single(
    model, image_path: Path, transform, idx_to_grade, idx_to_name, device, logit_adjust=None
) -> Dict:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    logits = model(x)
    if logit_adjust is not None:
        logits = logits - logit_adjust
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()
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
def evaluate_folder(
    model, eval_dir: Path, transform, idx_to_grade, idx_to_name, device,
    batch_size: int = 32, logit_adjust=None,
) -> None:
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
        # Flip TTA plus the checkpoint's class-prior correction, matching what
        # train.py reports for the test split, so the numbers are comparable.
        la = logit_adjust
        lg_a, lg_b = model(images), model(torch.flip(images, dims=[3]))
        if la is not None:
            lg_a, lg_b = lg_a - la, lg_b - la
        probs = F.softmax(lg_a, dim=1) + F.softmax(lg_b, dim=1)
        all_preds.extend(probs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())

    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]
    print(f"\nEvaluation on {eval_dir}  ({len(dataset)} images)")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    within_one = sum(abs(p - l) <= 1 for p, l in zip(all_preds, all_labels))
    print(f"Overall accuracy: {correct / len(all_labels):.4f}")
    print(f"Within-one-grade accuracy: {within_one / len(all_labels):.4f}")


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

    model, idx_to_grade, idx_to_name, img_size, clahe = load_checkpoint(str(checkpoint_path), device)
    transform = get_eval_transform(img_size, clahe=clahe)
    logit_adjust = load_logit_adjust(str(checkpoint_path))
    if logit_adjust is not None:
        logit_adjust = logit_adjust.to(device)
    print(
        f"Loaded checkpoint: {checkpoint_path} (device: {device}, "
        f"img_size: {img_size}, clahe: {clahe}, "
        f"logit_adjust: {'on' if logit_adjust is not None else 'off'})"
    )

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        result = predict_single(
            model, image_path, transform, idx_to_grade, idx_to_name, device, logit_adjust
        )
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
                results.append(
                    predict_single(
                        model, p, transform, idx_to_grade, idx_to_name, device, logit_adjust
                    )
                )
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
        evaluate_folder(
            model, eval_dir, transform, idx_to_grade, idx_to_name, device,
            logit_adjust=logit_adjust,
        )


if __name__ == "__main__":
    main()