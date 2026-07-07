"""
clinical_logic.py

Converts model output + patient inputs → full prescription dict.

Key rule (do not change):
  X-ray KL grade  → safety ceiling (max_angle)
  Surgery + weeks → exercise selector (rehab phase)
  These are independent. The ceiling caps each exercise's angle limit.
"""

from typing import Optional

from exercise_protocols import get_phase

# KL Grade → clinical values (duplicated from model/inference.py intentionally
# to avoid importing torch just for two dicts)
KL_MAX_ANGLE   = {0: 120, 1: 120, 2: 90, 3: 60, 4: 45}
KL_HEALTH_SCORE = {0: 95,  1: 80,  2: 60, 3: 35, 4: 15}

DISCLAIMER = (
    "⚠️ This output is for informational purposes only and is not a substitute "
    "for professional medical advice. Always consult your physiotherapist or "
    "surgeon before starting any exercise programme."
)

KL_DESCRIPTIONS = {
    0: "Normal — no radiographic features of osteoarthritis",
    1: "Doubtful — possible osteophytes, no joint-space narrowing",
    2: "Minimal — definite osteophytes, possible narrowing",
    3: "Moderate — multiple osteophytes, definite narrowing, some sclerosis",
    4: "Severe — large osteophytes, significant narrowing, bone-on-bone changes",
}

SURGERY_LABELS = {
    "acl":         "ACL reconstruction",
    "tkr":         "Total Knee Replacement (TKR)",
    "meniscus":    "meniscus repair",
    "arthroscopy": "knee arthroscopy",
    "none":        "conservative OA management (no surgery)",
}


def _cap_exercises(raw_exercises: list, max_angle: int) -> list:
    """
    Cap each exercise's protocol_angle_limit by the X-ray-derived max_angle.
    Sets angle_capped=True and adds a caution note when the cap fires.
    """
    result = []
    for ex in raw_exercises:
        proto_limit = ex.get("protocol_angle_limit", 120)
        effective   = min(proto_limit, max_angle)
        capped      = effective < proto_limit

        caution = ex.get("cautions") or ""
        if capped:
            cap_note = (
                f"⚠️ Standard angle for this exercise is {proto_limit}°, "
                f"reduced to {effective}° based on your X-ray (KL grade severity). "
            )
            caution = cap_note + caution if caution else cap_note.rstrip()

        result.append({
            "name":         ex["name"],
            "description":  ex["description"],
            "target_reps":  ex["target_reps"],
            "target_sets":  ex["target_sets"],
            "angle_limit":  effective,
            "hold_seconds": ex.get("hold_seconds"),
            "instructions": ex["instructions"],
            "cautions":     caution if caution else None,
            "angle_capped": capped,
        })
    return result


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
) -> str:
    surgery_name = SURGERY_LABELS.get(surgery_type, surgery_type)
    kl_desc      = KL_DESCRIPTIONS.get(kl_grade, "")
    confidence_pct = int(confidence * 100)

    if surgery_type == "none" or weeks_post_op is None:
        base = (
            f"Your X-ray has been classified as KL Grade {kl_grade} "
            f"({kl_desc}) with {confidence_pct}% model confidence. "
            f"Joint Health Score: {health_score}/100. "
            f"Your safe flexion ceiling is {max_angle}°. "
            f"You have been placed in: {phase_label}. "
            f"Clinical goal: {phase_goal}"
        )
    else:
        week_str = f"{weeks_post_op} week{'s' if weeks_post_op != 1 else ''} post-op"
        base = (
            f"Your X-ray has been classified as KL Grade {kl_grade} "
            f"({kl_desc}) with {confidence_pct}% model confidence. "
            f"Joint Health Score: {health_score}/100. "
            f"Combined with {surgery_name} at {week_str}, "
            f"you are in: {phase_label}. "
            f"Your imaging-derived safe flexion ceiling is {max_angle}°. "
            f"Clinical goal: {phase_goal}"
        )

    if has_caps:
        base += (
            f" One or more exercise angle limits have been reduced to stay within "
            f"your {max_angle}° ceiling — see individual exercise cautions."
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
) -> dict:
    """Full prescription from X-ray analysis. Matches AnalyseXrayResponse schema."""
    phase      = get_phase(surgery_type, weeks_post_op, kl_grade)
    exercises  = _cap_exercises(phase["exercises"], max_angle)
    any_capped = any(e["angle_capped"] for e in exercises)

    rationale = _build_rationale(
        kl_grade      = kl_grade,
        health_score  = health_score,
        max_angle     = max_angle,
        surgery_type  = surgery_type,
        weeks_post_op = weeks_post_op,
        phase_label   = phase["label"],
        phase_goal    = phase["goal"],
        has_caps      = any_capped,
        confidence    = confidence,
    )

    return {
        "kl_grade":          kl_grade,
        "health_score":      health_score,
        "max_angle":         max_angle,
        "confidence":        confidence,
        "knee_side":         knee_side,
        "surgery_type":      surgery_type,
        "weeks_post_op":     weeks_post_op,
        "rehab_phase":       phase["key"],
        "rehab_phase_label": phase["label"],
        "rehab_phase_goal":  phase["goal"],
        "exercise_list":     exercises,
        "rationale":         rationale,
        "disclaimer":        DISCLAIMER,
        "model_version":     model_version,
        "demo_mode":         demo_mode,
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
    max_angle  = KL_MAX_ANGLE.get(kl_grade, 120)
    phase      = get_phase(surgery_type, weeks_post_op, kl_grade)
    exercises  = _cap_exercises(phase["exercises"], max_angle)

    return {
        "surgery_type":      surgery_type,
        "weeks_post_op":     weeks_post_op,
        "kl_grade":          kl_grade,
        "rehab_phase":       phase["key"],
        "rehab_phase_label": phase["label"],
        "rehab_phase_goal":  phase["goal"],
        "max_angle":         max_angle,
        "exercise_list":     exercises,
        "disclaimer":        DISCLAIMER,
    }