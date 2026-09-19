"""
image_checks.py — upload quality checks, before any model sees the file.

Split out of inference.py so that importing it does not drag in torch. The check
itself is PIL and numpy arithmetic; it was only ever coupled to the ML stack by
where it happened to live. That coupling meant `import main` required a 2 GB
framework, which made the API tests — and the CI job that runs them without
torch — impossible to satisfy.

inference.py re-exports validate_image, so existing callers and the CLI are
unaffected.
"""

import io
import os

import numpy as np
from PIL import Image

# Thresholds are on the 0–255 grey scale of the decoded image.
MIN_CONTRAST_STD = 15   # below this the image carries no structure to grade
MIN_MEAN         = 10   # essentially black
MAX_MEAN         = 245  # essentially blank / overexposed

# A 10 MB upload cap says nothing about how much memory the file becomes. PNG
# compresses flat colour to almost nothing, so a 137 KB file can decode to
# 12000x12000 — 576 MB once it reaches float32 below, from a request small
# enough to send twenty of per minute. Pillow warns about this and carries on
# regardless, so the ceiling is enforced here instead, against the header rather
# than the decoded pixels.
MAX_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(40_000_000)))

# Enough of a picture to be a radiograph. A 4000x8 strip passed every check
# below and came back with a KL grade and a movement ceiling.
MIN_SIDE          = 64
MAX_ASPECT_RATIO  = 4.0

# The endpoint advertises JPEG and PNG, and the browser-supplied content type is
# a claim, not a fact: a GIF, BMP, TIFF or WEBP sent as "image/png" was decoded
# happily. Checking what the bytes actually are keeps the decoder surface down
# to the two formats that were meant to be supported.
ALLOWED_FORMATS = {"JPEG", "PNG", "MPO"}   # MPO: multi-frame JPEG from phone cameras

# An inverted (negative) radiograph clears every check above and still comes back
# with a grade — a different one. On the synthetic films in this repo, negating
# the image moved the reading from KL 0 to KL 3, which is a 30-degree change in
# what the patient is then told they may bend to. The energy-based OOD screen
# does not see it: -1.718 against -1.640, the same number twice.
#
# What separates a film from its own negative is where the bright pixels sit.
# The anatomy is centred and denser than what surrounds it, so the middle of the
# frame reads brighter than its border. Medians rather than means, so a white
# collimation band down one edge does not swing it.
#
# The statistic is exactly antisymmetric — negating the image negates it, since
# 255-x flips both medians and leaves the spread alone — so a threshold on one
# side is a guarantee on the other: any film scoring above +0.5 has its own
# negative rejected.
#
# First fitted against synthetic films; since checked on 1656 real ones (the
# Kaggle KL test split, which the model never trained on). The real distribution
# sits well clear of the threshold: median +1.27, and 5% of films below +0.78.
# Those films are 224px crops of the joint, the case the synthetic tests feared
# would read as zero — it does not, because bone is still the bright centre of a
# real crop.
#
# The gate refused 26 of the 1656, and every one checked by eye is a genuine
# negative: the joint space shows as a white band where it should be dark. The
# dataset ships them — 13 patients, both knees each. The most normal-looking of
# the refused scored -0.93; the least typical film let through, a badly
# washed-out but correctly oriented one, scored +0.32. -0.5 sits in that gap.
#
# One thing the real films changed: the model graded 22 of those 26 negatives
# correctly, because it trained on the same source and so on the same negatives.
# Refusing them costs a re-upload, not accuracy. The gate is kept anyway — the
# synthetic result above shows a negative from anywhere else is not safe.
MIN_POLARITY = float(os.getenv("MIN_POLARITY", "-0.5"))


def _polarity(gray: np.ndarray) -> float:
    """
    How much brighter the middle of the frame is than its border, in units of the
    image's own spread. Positive for a radiograph, negative for its negative.
    """
    height, width = gray.shape
    band_y, band_x = max(1, int(height * 0.12)), max(1, int(width * 0.12))
    border = np.concatenate([
        gray[:band_y].ravel(),
        gray[-band_y:].ravel(),
        gray[band_y:-band_y, :band_x].ravel(),
        gray[band_y:-band_y, -band_x:].ravel(),
    ])
    centre = gray[int(height * 0.25):int(height * 0.75),
                  int(width * 0.25):int(width * 0.75)]
    return float((np.median(centre) - np.median(border)) / (gray.std() + 1e-6))


def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    """
    Lightweight quality check before inference.
    Rejects blank, inverted, or non-radiograph images.
    Returns (ok: bool, message: str).
    """
    try:
        # Lazy: this parses the header only, so size and format are known before
        # anything is decoded into memory.
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        fmt = (img.format or "").upper()
    except Exception:
        return False, "Could not decode the uploaded file. Please upload a valid JPEG or PNG."

    if fmt not in ALLOWED_FORMATS:
        return False, f"{fmt or 'That file'} is not a supported format. Please upload a JPEG or PNG."

    if width * height > MAX_PIXELS:
        return False, (
            f"That image is {width}x{height}, which is larger than this service will "
            f"open. Please upload a radiograph under {MAX_PIXELS // 1_000_000} megapixels."
        )

    if width < MIN_SIDE or height < MIN_SIDE:
        return False, (
            f"That image is only {width}x{height}. A knee X-ray needs to be at least "
            f"{MIN_SIDE}x{MIN_SIDE} to be readable."
        )

    longest, shortest = max(width, height), min(width, height)
    if longest / shortest > MAX_ASPECT_RATIO:
        return False, (
            "That image is a long thin strip rather than a radiograph. Please upload the "
            "whole X-ray."
        )

    try:
        gray = np.array(img.convert("L"), dtype=np.float32)
    except Exception:
        return False, "Could not decode the uploaded file. Please upload a valid JPEG or PNG."

    std  = gray.std()
    mean = gray.mean()

    if std < MIN_CONTRAST_STD:
        return False, (
            "Image contrast is too low. "
            "Please ensure you are uploading a clear knee X-ray."
        )
    if mean < MIN_MEAN:
        return False, "Image appears completely black. Please check the file."
    if mean > MAX_MEAN:
        return False, "Image appears overexposed / blank. Please check the file."

    if _polarity(gray) < MIN_POLARITY:
        return False, (
            "This looks like a negative: the bone appears dark against a light background. "
            "Please upload the X-ray the way it is normally viewed."
        )

    return True, "ok"
