"""
prosthesis.py — is there metalwork in this knee?

The gap this addresses is the one the README has always named: the KL classifier
is trained on native knees, so grading a *replaced* joint is out of distribution.
The reading it returns is not wrong so much as meaningless — and it is meaningless
for precisely the patients the TKR protocols are written for.

What is actually done about it
------------------------------
Two signals, used very differently, because their reliability is very different.

**The declared surgery type is the reliable one.** A patient who says they had a
total knee replacement has a replaced joint. Nothing needs detecting: the KL
grade simply must not set their ceiling. That is handled in clinical_logic.

**This module is the backstop**, for a prosthesis nobody declared. It is a
heuristic over pixel intensities, not a trained classifier, and it is treated
accordingly: it can raise a warning and it can tighten, but it is never allowed
to loosen a restriction on its own. A false positive that removed a genuinely
needed limit would be worse than the problem it is guarding against.

Why a heuristic and not a model
-------------------------------
A learned native-versus-replaced classifier is the right answer and is not
buildable here: it needs a labelled set of post-arthroplasty radiographs, which
this project does not have. The physics, though, is unusually favourable to a
simple rule — cobalt-chrome and titanium are far more radio-opaque than bone, so
an implant saturates the detector over a large, solid, geometrically regular
area in a way trabecular bone does not.

So this measures that, honestly, and reports a score rather than a verdict.
Torch-free on purpose, like image_checks: it runs on every upload.
"""

import io

import numpy as np
from PIL import Image

# Pixel value above which a region is "as bright as the detector goes". Metal
# implants routinely saturate; cortical bone rarely reaches here across any
# meaningful area.
SATURATION_LEVEL = 235

# Fraction of the image at that level before hardware is suspected. Set high on
# purpose. A knee radiograph has bright cortical margins and often a collimation
# border, so a low threshold would fire on healthy knees constantly — and a
# warning that fires constantly is one nobody reads.
SUSPECT_FRACTION = 0.045

# Metalwork is manufactured: its bright region is *solid*. Bone's bright pixels
# are scattered through trabecular texture, and an over-exposed film is bright
# more or less everywhere. Distinguishing those needs a measure of shape, not
# just of area.
#
# Erosion gives one cheaply. A pixel survives if every pixel around it is also
# bright, so a solid component keeps its whole interior while speckle vanishes:
# for a 5x5 window, scattered pixels at even 50% density survive with
# probability 0.5^25, which is to say never.
#
# (A bounding box will not do this. Saturation spread evenly across a frame
# fills its box just as thoroughly as a single block does, and the first version
# of this file flagged an over-exposed image as an implant for exactly that
# reason.)
SOLIDITY_MIN = 0.35
EROSION_WINDOW = 5

# Below this the bright pixels are too few to say anything about their shape.
MIN_SATURATED_PIXELS = 400

# The solidity rule above rests on an assumption that turned out to be wrong: that
# an over-exposed film is bright in scattered speckle. Measured on 1656 real native
# knees (the Kaggle KL test split), it is not. A washed-out film is bright in one
# large solid region, which erosion keeps, and the detector flagged 296 of them —
# 17.9% of knees with no metal in them at all. Solidity blocked almost none.
#
# What does separate the two is everything *else* in the frame. Metal is bright
# against bone and soft tissue that are exposed normally, so the pixels below
# saturation stay mid-grey. On an over-exposed film they are already bright. On
# that same test split, requiring the unsaturated remainder to average below this
# cut the false alarms from 296 to 43 (2.6%):
#
#     140 -> 0.7%    150 -> 1.2%    160 -> 2.6%    170 -> 5.0%
#
# 160 rather than lower because normally exposed native knees reach a remainder
# mean of about 170 (p90), and an implant film exposed the same way would sit in
# that range too; a stricter cut buys fewer false alarms by missing real metal.
#
# ponytail: fitted on native knees only. Whether it still catches real implants is
# unmeasured — there were no post-arthroplasty films to test against. It is also a
# gate on exposure rather than on implant size, which matters: the Kaggle films are
# 224px crops of the joint, but patients upload whole radiographs, where an implant
# fills far less of the frame. A size threshold tuned on the crops would miss most
# real ones; this does not depend on size. The Emory MRKR set (controlled access,
# data.hitilab.com) labels arthroplasty per image and is the data to settle both.
MAX_REST_MEAN = 160


def _erode(mask: np.ndarray, window: int) -> np.ndarray:
    """Binary erosion by a square: keep only pixels whose whole neighbourhood is set."""
    radius = window // 2
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = mask.copy()
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            out &= padded[radius + dy:radius + dy + height,
                          radius + dx:radius + dx + width]
    return out


def detect_hardware(image_bytes: bytes) -> dict:
    """
    Look for signs of a joint replacement.

    Returns a dict — never raises, and never a bare boolean, because the caller
    needs the numbers to decide how much to trust it:

        {
          "suspected":         bool,
          "saturated_fraction": float,   # 0-1, how much of the image is at ceiling
          "solidity":          float,    # 0-1, how solid that bright region is
          "rest_mean":         float,    # mean of the unsaturated pixels, 0-255
          "reason":            str,      # plain-English, for the rationale
        }

    An undecodable image reports `suspected: False` with a reason saying so.
    validate_image has already rejected those by the time this runs; this is
    only here so a surprise cannot take down an analysis.
    """
    unknown = {
        "suspected": False, "saturated_fraction": 0.0, "solidity": 0.0, "rest_mean": 0.0,
        "reason": "Could not assess this image for metalwork.",
    }

    try:
        gray = np.array(Image.open(io.BytesIO(image_bytes)).convert("L"), dtype=np.uint8)
    except Exception:
        return unknown

    if gray.size == 0:
        return unknown

    bright = gray >= SATURATION_LEVEL
    saturated = int(bright.sum())
    fraction = saturated / gray.size

    # Solidity: what fraction of the bright pixels are interior rather than
    # edge. A femoral component is nearly all interior; trabecular highlights and
    # over-exposure are nearly all edge.
    solidity = 0.0
    if saturated >= MIN_SATURATED_PIXELS:
        solidity = int(_erode(bright, EROSION_WINDOW).sum()) / saturated

    # Mean of everything below saturation. A fully saturated frame has no
    # remainder; treat it as over-exposed, since a white page is not an implant.
    rest = gray[~bright]
    rest_mean = float(rest.mean()) if rest.size else 255.0
    overexposed = rest_mean >= MAX_REST_MEAN

    suspected = (
        fraction >= SUSPECT_FRACTION
        and solidity >= SOLIDITY_MIN
        and not overexposed
    )

    if suspected:
        reason = (
            f"About {fraction * 100:.0f}% of this image is as bright as the detector goes, "
            "in one solid region. That pattern is typical of a metal implant rather than bone."
        )
    elif fraction >= SUSPECT_FRACTION and overexposed:
        reason = (
            f"{fraction * 100:.0f}% of this image is very bright, but so is the rest of it — "
            "more like an over-exposed film than metalwork."
        )
    elif fraction >= SUSPECT_FRACTION:
        reason = (
            f"{fraction * 100:.0f}% of this image is very bright, but scattered rather than "
            "solid — more like bone texture or over-exposure than metalwork."
        )
    else:
        reason = "No sign of a joint replacement in this image."

    return {
        "suspected": bool(suspected),
        "saturated_fraction": round(float(fraction), 4),
        "solidity": round(float(solidity), 3),
        "rest_mean": round(rest_mean, 1),
        "reason": reason,
    }
