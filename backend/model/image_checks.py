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

    return True, "ok"
