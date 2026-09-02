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

import numpy as np
from PIL import Image

# Thresholds are on the 0–255 grey scale of the decoded image.
MIN_CONTRAST_STD = 15   # below this the image carries no structure to grade
MIN_MEAN         = 10   # essentially black
MAX_MEAN         = 245  # essentially blank / overexposed


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
