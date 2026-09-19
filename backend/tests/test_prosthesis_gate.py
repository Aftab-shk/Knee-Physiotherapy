"""
The prosthesis gate.

The gap the README has always named: the KL classifier is trained on native
knees, so grading a *replaced* joint is out of distribution. The reading is not
so much wrong as meaningless — and it is meaningless for precisely the patients
the TKR protocols are written for.

Two signals, treated very differently because their reliability differs:

  * **A declared total knee replacement is certain.** The grade must not set the
    ceiling; the surgical protocol and the week do.
  * **Suspected metalwork is a heuristic** over pixel brightness, and is allowed
    to warn but never to loosen. Relaxing a real restriction on a guess is the
    one mistake that cannot be walked back.

Run:  python -m pytest backend/tests -q
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_logic import build_prescription, kl_applies, protocol_ceiling
from exercise_protocols import get_phase


def prescribe(**kw):
    base = {
        "kl_grade": 4, "health_score": 15, "max_angle": 45, "confidence": 0.9,
        "demo_mode": False, "knee_side": "left", "surgery_type": "tkr",
        "weeks_post_op": 8, "model_version": "test",
        "confidence_band": "high", "calibrated": True,
    }
    base.update(kw)
    return build_prescription(**base)


# ---------------------------------------------------------------------------
# A replaced joint does not get graded
# ---------------------------------------------------------------------------

def test_a_replacement_is_the_one_surgery_the_grade_does_not_apply_to():
    assert kl_applies("tkr") is False
    for surgery in ("acl", "meniscus", "arthroscopy", "none"):
        assert kl_applies(surgery) is True, surgery


def test_the_xray_does_not_set_the_ceiling_after_a_replacement():
    """
    A KL 4 reading would cap this knee at 45°. That number describes wear on a
    joint surface that has been removed, so it must not be what a patient eight
    weeks out of a replacement is held to.
    """
    p = prescribe(kl_grade=4, max_angle=45)

    assert p["kl_applicable"] is False
    assert p["max_angle"] > 45, "the protocol should govern, not the grade"
    assert p["max_angle"] == protocol_ceiling(get_phase("tkr", 8, 4))


def test_the_grade_is_still_reported_it_just_does_nothing():
    """
    Hiding it would be its own kind of dishonesty — the model did return it, and
    a clinician may want to know what it said.
    """
    p = prescribe(kl_grade=4, health_score=15)
    assert p["kl_grade"] == 4
    assert p["health_score"] == 15
    assert p["kl_applicable"] is False


def test_the_patient_is_told_why_the_xray_was_set_aside():
    r = prescribe()["rationale"]
    assert "replaced" in r.lower()
    assert "surgeon" in r.lower() or "surgical" in r.lower()
    # The ceiling wording from the graded path would describe a limit that was
    # never applied, so it must not survive alongside this.
    assert "safe ceiling of 45" not in r


@pytest.mark.parametrize("grade", [0, 1, 2, 3, 4])
def test_a_replaced_knee_gets_the_same_limits_whatever_the_grade_says(grade):
    """
    The clearest statement of the rule: if the grade changed the answer, it would
    still be setting the ceiling.
    """
    from kl_constants import KL_MAX_ANGLE

    p = prescribe(kl_grade=grade, max_angle=KL_MAX_ANGLE[grade])
    baseline = prescribe(kl_grade=0, max_angle=KL_MAX_ANGLE[0])

    assert p["max_angle"] == baseline["max_angle"]
    assert [e["angle_limit"] for e in p["exercise_list"]] == \
           [e["angle_limit"] for e in baseline["exercise_list"]]


def test_a_native_knee_is_still_governed_by_its_grade():
    """The gate must not leak into everyone else."""
    severe = prescribe(surgery_type="acl", kl_grade=4, max_angle=45)
    mild = prescribe(surgery_type="acl", kl_grade=0, max_angle=120)

    assert severe["kl_applicable"] is True
    assert severe["max_angle"] == 45
    assert mild["max_angle"] == 120
    assert severe["max_angle"] < mild["max_angle"]


def test_conservative_management_still_depends_entirely_on_the_grade():
    """
    surgery_type 'none' is osteoarthritis management. The grade is the whole
    basis there, so switching it off would remove the only restriction.
    """
    p = prescribe(surgery_type="none", weeks_post_op=None, kl_grade=4, max_angle=45)
    assert p["kl_applicable"] is True
    assert p["max_angle"] == 45


# ---------------------------------------------------------------------------
# Suspected metalwork warns; it never loosens
# ---------------------------------------------------------------------------

def test_suspected_hardware_leaves_the_ceiling_exactly_where_it_was():
    """
    The detector is pixel brightness, not a classifier. A false positive that
    lifted a genuinely needed restriction would be worse than the problem it
    guards against.
    """
    without = prescribe(surgery_type="none", weeks_post_op=None, kl_grade=4, max_angle=45)
    with_flag = prescribe(surgery_type="none", weeks_post_op=None, kl_grade=4, max_angle=45,
                          hardware_suspected=True, hardware_reason="Bright solid region.")

    assert with_flag["max_angle"] == without["max_angle"] == 45
    assert with_flag["kl_applicable"] is True
    assert [e["angle_limit"] for e in with_flag["exercise_list"]] == \
           [e["angle_limit"] for e in without["exercise_list"]]


def test_suspected_hardware_says_so_in_the_rationale():
    p = prescribe(surgery_type="acl", hardware_suspected=True,
                  hardware_reason="Bright solid region.")
    assert "replace" in p["rationale"].lower()
    assert p["hardware_suspected"] is True


def test_a_declared_replacement_needs_no_warning():
    """It is not a suspicion when the patient told us."""
    p = prescribe(surgery_type="tkr")
    assert p["hardware_suspected"] is False
    assert "may contain" not in p["rationale"]


# ---------------------------------------------------------------------------
# The detector itself
# ---------------------------------------------------------------------------

pytest.importorskip("numpy", reason="the detector is numpy arithmetic")
pytest.importorskip("PIL", reason="the detector decodes an image")

from model.prosthesis import SATURATION_LEVEL, detect_hardware


def make_png(pixels):
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(pixels, dtype="uint8")).save(buf, format="PNG")
    return buf.getvalue()


def textured_knee(size=200, seed=0):
    """Trabecular bone: bright pixels, but scattered through texture."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(30, 200, (size, size), dtype="uint8")


def with_implant(size=200, block=70):
    """The same, with a solid saturated rectangle where a component would sit."""
    import numpy as np

    arr = textured_knee(size)
    top = (size - block) // 2
    arr[top:top + block, top:top + block] = 255
    return np.asarray(arr, dtype=np.uint8)


def test_a_solid_bright_component_is_detected():
    result = detect_hardware(make_png(with_implant()))
    assert result["suspected"] is True
    assert result["saturated_fraction"] > 0.1
    assert "implant" in result["reason"]


def test_ordinary_bone_texture_is_not_flagged():
    result = detect_hardware(make_png(textured_knee()))
    assert result["suspected"] is False


def test_scattered_bright_pixels_are_not_metalwork():
    """
    A bright but speckled image — an over-exposed radiograph, say. Plenty of
    saturation, no solid region. Flagging it would be the false positive that
    makes clinicians stop reading the warning.
    """
    import numpy as np

    rng = np.random.default_rng(1)
    arr = rng.integers(0, 2, (200, 200), dtype="uint8") * 255
    result = detect_hardware(make_png(arr))

    assert result["saturated_fraction"] > 0.4, "the test image really is very bright"
    assert result["suspected"] is False
    assert "scattered" in result["reason"]


def test_an_over_exposed_film_is_not_metalwork():
    """
    The false alarm that actually happens. On 1656 real native knees the
    detector flagged 17.9%, and they were not speckled — a washed-out film is
    bright in one large solid block, which the solidity rule keeps. What gives it
    away is that everything else in the frame is bright too; beside real metal,
    bone and soft tissue stay mid-grey.
    """
    import numpy as np

    rng = np.random.default_rng(2)
    arr = rng.integers(175, 225, (200, 200), dtype="uint8")   # washed-out bone
    arr[40:160, 40:160] = 255                                 # one solid blown-out region
    result = detect_hardware(make_png(arr))

    assert result["saturated_fraction"] > 0.3, "the region really is large and solid"
    assert result["solidity"] > 0.35
    assert result["suspected"] is False
    assert "over-exposed" in result["reason"]


def test_metal_beside_normally_exposed_bone_is_still_caught():
    """The same block, against bone at ordinary exposure — the gate must not cost this."""
    result = detect_hardware(make_png(with_implant()))
    assert result["suspected"] is True
    assert result["rest_mean"] < 160


def test_a_blank_dark_image_is_not_metalwork():
    import numpy as np

    result = detect_hardware(make_png(np.zeros((100, 100), dtype="uint8")))
    assert result["suspected"] is False
    assert result["saturated_fraction"] == 0.0


def test_an_undecodable_file_does_not_take_the_analysis_down():
    result = detect_hardware(b"not an image at all")
    assert result["suspected"] is False
    assert "could not" in result["reason"].lower()


def test_the_threshold_is_where_the_module_says_it_is():
    """Guards against the constant drifting away from its own documentation."""
    import numpy as np

    just_under = np.full((200, 200), SATURATION_LEVEL - 1, dtype="uint8")
    assert detect_hardware(make_png(just_under))["saturated_fraction"] == 0.0

    at_level = np.full((200, 200), SATURATION_LEVEL, dtype="uint8")
    assert detect_hardware(make_png(at_level))["saturated_fraction"] == 1.0
