"""
clinical_logic.py

Converts model output + patient inputs → full prescription dict.

Key rule (do not change):
  X-ray KL grade  → safety ceiling (max_angle)
  Surgery + weeks → exercise selector (rehab phase)
  These are independent. The ceiling caps each exercise's angle limit.
"""

from datetime import date
from typing import Optional

from exercise_protocols import get_phase
from kl_constants import KL_DESCRIPTIONS, KL_HEALTH_SCORE, KL_MAX_ANGLE  # noqa: F401

DISCLAIMER = (
    "⚠️ This output is for informational purposes only and is not a substitute "
    "for professional medical advice. Always consult your physiotherapist or "
    "surgeon before starting any exercise programme."
)

# Wording for each confidence band. Deliberately plain: these appear beside a
# number that restricts how far someone moves a recovering joint.
CONFIDENCE_PHRASES = {
    "high":     "high confidence",
    "moderate": "moderate confidence",
    "low":      "low confidence",
}

SURGERY_LABELS = {
    "acl":         "ACL reconstruction",
    "tkr":         "Total Knee Replacement (TKR)",
    "meniscus":    "meniscus repair",
    "arthroscopy": "knee arthroscopy",
    "none":        "conservative OA management (no surgery)",
}


# The longest recovery the protocols describe. Matches the API's own bound, so
# a date far enough back to exceed it is reported rather than silently clamped.
MAX_WEEKS_POST_OP = 520


def weeks_since_surgery(surgery_date: date, today: Optional[date] = None) -> int:
    """
    Whole weeks between the operation and today.

    Floored, not rounded. The protocols are written as spans — "weeks 0–2",
    "weeks 3–6" — and a patient is in week N until day 7(N+1). Rounding would
    advance someone into the next phase three days early, which for a knee two
    weeks out of a replacement means loading it before it is ready.

    Day of surgery is week 0, matching the form's own instruction to enter 0 for
    the first week post-op.
    """
    today = today or date.today()
    days = (today - surgery_date).days
    if days < 0:
        raise ValueError("surgery_date is in the future")
    return min(days // 7, MAX_WEEKS_POST_OP)


def _cap_exercises(raw_exercises: list, max_angle: int) -> tuple[list, list]:
    """
    Cap each exercise's protocol_angle_limit by the X-ray-derived max_angle.
    Sets angle_capped=True and adds a caution note when the cap fires.

    Exercises with a start_angle above max_angle are screened out entirely and
    returned separately. Capping their limit would not help: a seated exercise
    puts the knee at ~90° before the patient moves, so prescribing it to someone
    with a 45° ceiling means either breaching the ceiling or triggering the
    safety alarm just by sitting down. They are reported rather than silently
    dropped so a physiotherapist can see what was withheld and why.

    Returns (prescribed, excluded).
    """
    result: list = []
    excluded: list = []

    for ex in raw_exercises:
        proto_limit = ex.get("protocol_angle_limit", 120)
        effective   = min(proto_limit, max_angle)
        capped      = effective < proto_limit

        start_angle = ex.get("start_angle")
        if start_angle is not None and start_angle > max_angle:
            excluded.append({
                "name":        ex["name"],
                "start_angle": start_angle,
                "max_angle":   max_angle,
                "reason": (
                    f"This exercise is performed with the knee already at about "
                    f"{start_angle}°, which is beyond your {max_angle}° safe ceiling. "
                    f"It has been withheld rather than adapted — the starting "
                    f"position itself is the problem, so reducing the target angle "
                    f"would not make it safe. Ask your physiotherapist about a "
                    f"lying or standing alternative."
                ),
            })
            continue

        # A hold target can never sit above the effective ceiling.
        hold_angle = ex.get("hold_angle")
        if hold_angle is not None:
            hold_angle = min(hold_angle, effective)

        caution = ex.get("cautions") or ""
        if capped:
            cap_note = (
                f"⚠️ Standard angle for this exercise is {proto_limit}°, "
                f"reduced to {effective}° based on your X-ray (KL grade severity). "
            )
            caution = cap_note + caution if caution else cap_note.rstrip()

        result.append({
            "name":          ex["name"],
            "description":   ex["description"],
            "target_reps":   ex["target_reps"],
            "target_sets":   ex["target_sets"],
            "angle_limit":   effective,
            "hold_seconds":  ex.get("hold_seconds"),
            "tracked_joint": ex.get("tracked_joint", "knee"),
            "hold_target":   ex.get("hold_target"),
            "start_angle":   start_angle,
            "hold_angle":    hold_angle,
            "instructions":  ex["instructions"],
            "cautions":      caution if caution else None,
            "angle_capped":  capped,
        })
    return result, excluded


def _build_rationale(
    kl_grade:     int,
    health_score: int,
    max_angle:    int,
    surgery_type: str,
    weeks_post_op: Optional[int],
    phase_label:  str,
    phase_goal:   str,
    has_caps:     bool,
    confidence:   float,
    excluded:        Optional[list] = None,
    confidence_band: str  = "low",
    calibrated:      bool = False,
    ood_suspected:   bool = False,
) -> str:
    surgery_name = SURGERY_LABELS.get(surgery_type, surgery_type)
    kl_desc      = KL_DESCRIPTIONS.get(kl_grade, "")

    # Quote a percentage only when it has been calibrated against held-out data.
    # An uncalibrated softmax maximum reads as a probability of being right and
    # is not one — stating "82% confident" next to a movement restriction claims
    # a precision this model has not demonstrated.
    if calibrated:
        confidence_str = f"{CONFIDENCE_PHRASES[confidence_band]} ({int(confidence * 100)}% calibrated confidence)"
    else:
        confidence_str = CONFIDENCE_PHRASES[confidence_band]

    if surgery_type == "none" or weeks_post_op is None:
        base = (
            f"Your X-ray has been classified as KL Grade {kl_grade} "
            f"({kl_desc}) with {confidence_str}. "
            f"Joint Health Score: {health_score}/100. "
            f"Your safe flexion ceiling is {max_angle}°. "
            f"You have been placed in: {phase_label}. "
            f"Clinical goal: {phase_goal}"
        )
    else:
        week_str = f"{weeks_post_op} week{'s' if weeks_post_op != 1 else ''} post-op"
        base = (
            f"Your X-ray has been classified as KL Grade {kl_grade} "
            f"({kl_desc}) with {confidence_str}. "
            f"Joint Health Score: {health_score}/100. "
            f"Combined with {surgery_name} at {week_str}, "
            f"you are in: {phase_label}. "
            f"Your imaging-derived safe flexion ceiling is {max_angle}°. "
            f"Clinical goal: {phase_goal}"
        )

    if confidence_band == "low":
        base += (
            " The model is not confident in this grade. Treat the ceiling above as "
            "provisional and have a physiotherapist confirm it before relying on it."
        )

    if ood_suspected:
        base += (
            " This image also sits outside the range the model usually sees, which "
            "often means it is not a standard front-on knee radiograph. Double-check "
            "you uploaded the right file."
        )

    if has_caps:
        base += (
            f" One or more exercise angle limits have been reduced to stay within "
            f"your {max_angle}° ceiling — see individual exercise cautions."
        )

    if excluded:
        names = ", ".join(e["name"] for e in excluded)
        base += (
            f" {len(excluded)} exercise{'s' if len(excluded) != 1 else ''} from the "
            f"standard protocol ({names}) {'have' if len(excluded) != 1 else 'has'} been "
            f"withheld: {'they are' if len(excluded) != 1 else 'it is'} performed from a "
            f"seated or equipment-supported position that already exceeds your "
            f"{max_angle}° ceiling before any movement begins. Ask your physiotherapist "
            f"about lying or standing alternatives."
        )

    return base


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def build_prescription(
    kl_grade:     int,
    health_score: int,
    max_angle:    int,
    confidence:   float,
    demo_mode:    bool,
    knee_side:    str,
    surgery_type: str,
    weeks_post_op: Optional[int],
    model_version: str,
    confidence_band: str  = "low",
    calibrated:      bool = False,
    ood_suspected:   bool = False,
) -> dict:
    """Full prescription from X-ray analysis. Matches AnalyseXrayResponse schema."""
    phase                 = get_phase(surgery_type, weeks_post_op, kl_grade)
    exercises, excluded   = _cap_exercises(phase["exercises"], max_angle)
    any_capped            = any(e["angle_capped"] for e in exercises)

    rationale = _build_rationale(
        kl_grade      = kl_grade,
        health_score  = health_score,
        max_angle     = max_angle,
        surgery_type  = surgery_type,
        weeks_post_op = weeks_post_op,
        phase_label   = phase["label"],
        phase_goal    = phase["goal"],
        has_caps        = any_capped,
        confidence      = confidence,
        excluded        = excluded,
        confidence_band = confidence_band,
        calibrated      = calibrated,
        ood_suspected   = ood_suspected,
    )

    return {
        "kl_grade":          kl_grade,
        "health_score":      health_score,
        "max_angle":         max_angle,
        "confidence":        confidence,
        "confidence_band":   confidence_band,
        "calibrated":        calibrated,
        "ood_suspected":     ood_suspected,
        "knee_side":         knee_side,
        "surgery_type":      surgery_type,
        "weeks_post_op":     weeks_post_op,
        "rehab_phase":       phase["key"],
        "rehab_phase_label": phase["label"],
        "rehab_phase_goal":  phase["goal"],
        "exercise_list":       exercises,
        "excluded_exercises":  excluded,
        "rationale":           rationale,
        "disclaimer":          DISCLAIMER,
        "model_version":       model_version,
        "demo_mode":           demo_mode,
    }


def get_exercises_only(
    surgery_type:  str,
    weeks_post_op: Optional[int],
    kl_grade:      int = 0,
) -> dict:
    """
    Exercise list without X-ray analysis.
    Used by GET /exercises endpoint.
    kl_grade defaults to 0 (no angle restriction) if not provided.
    """
    max_angle           = KL_MAX_ANGLE.get(kl_grade, 120)
    phase               = get_phase(surgery_type, weeks_post_op, kl_grade)
    exercises, excluded = _cap_exercises(phase["exercises"], max_angle)

    return {
        "surgery_type":       surgery_type,
        "weeks_post_op":      weeks_post_op,
        "kl_grade":           kl_grade,
        "rehab_phase":        phase["key"],
        "rehab_phase_label":  phase["label"],
        "rehab_phase_goal":   phase["goal"],
        "max_angle":          max_angle,
        "exercise_list":      exercises,
        "excluded_exercises": excluded,
        "disclaimer":         DISCLAIMER,
    }