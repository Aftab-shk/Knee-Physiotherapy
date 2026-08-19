"""
Contract tests between the exercise protocol database and the webcam tracker.

The tracker reads `tracked_joint` and `hold_target` off each exercise to decide
how to count reps/holds. These tests pin the invariants it relies on. The
regression they exist to prevent:

  Before `hold_target` existed, tracker.html treated "in position" as
  `angle > 15` for every hold exercise. Quad Sets, Straight Leg Raise and
  Patellar Mobilisation are held with the knee LOCKED STRAIGHT and carry a 5°
  limit — so the hold timer only advanced past 15°, while anything over 5°
  tripped the safety alarm and auto-paused the session. Those three are Phase I
  (weeks 0-2) exercises for ACL, TKR and meniscus: the feature was broken for
  the freshest post-op patients.

Run:  python -m pytest backend/tests -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_logic import _cap_exercises, get_exercises_only
from exercise_protocols import OA_LEVELS, PROTOCOLS, get_phase

VALID_JOINTS = {"knee", "ankle"}
VALID_HOLD_TARGETS = {"straight", "flexed", None}

# Mirrors tracker.html's updateHoldTimer()
FLEXED_HOLD_FRACTION = 0.7


def all_exercises():
    """Every exercise dict in the database, de-duplicated by name."""
    seen = {}
    for protocol in PROTOCOLS.values():
        for phase in protocol.get("phases", []):
            for ex in phase["exercises"]:
                seen.setdefault(ex["name"], ex)
    for level in OA_LEVELS:
        for ex in level["exercises"]:
            seen.setdefault(ex["name"], ex)
    return list(seen.values())


ALL = all_exercises()
HOLDS = [e for e in ALL if e.get("hold_seconds")]


def test_database_is_not_empty():
    assert len(ALL) >= 20, f"expected the full protocol database, got {len(ALL)}"
    assert len(HOLDS) >= 5


@pytest.mark.parametrize("ex", ALL, ids=lambda e: e["name"])
def test_every_exercise_declares_tracker_fields(ex):
    assert ex.get("tracked_joint") in VALID_JOINTS, (
        f"{ex['name']}: tracked_joint must be one of {VALID_JOINTS}"
    )
    assert ex.get("hold_target", None) in VALID_HOLD_TARGETS, (
        f"{ex['name']}: hold_target must be one of {VALID_HOLD_TARGETS}"
    )


@pytest.mark.parametrize("ex", ALL, ids=lambda e: e["name"])
def test_hold_target_set_exactly_when_holding(ex):
    """hold_target is meaningful only for timed holds — and required for them."""
    if ex.get("hold_seconds"):
        assert ex.get("hold_target") is not None, (
            f"{ex['name']} has hold_seconds={ex['hold_seconds']} but no hold_target; "
            "the tracker would fall back to the broken 'bent past 15 degrees' rule."
        )
    else:
        assert ex.get("hold_target") is None, (
            f"{ex['name']} has no hold_seconds, so hold_target should be None"
        )


@pytest.mark.parametrize("ex", HOLDS, ids=lambda e: e["name"])
def test_hold_position_never_breaches_the_safety_limit(ex):
    """
    THE REGRESSION TEST.

    For every timed hold, the position the tracker asks the patient to reach
    must sit at or below that exercise's own angle limit. If it doesn't, holding
    correctly triggers the alarm and the exercise is impossible to complete.
    """
    limit = ex["protocol_angle_limit"]

    if ex["hold_target"] == "straight":
        required = 0.0  # held at or below the limit; patient targets full extension
    else:
        required = limit * FLEXED_HOLD_FRACTION

    assert required <= limit, (
        f"{ex['name']}: hold position {required:.0f} deg exceeds its {limit} deg "
        "limit — the safety alarm would fire while performing it correctly."
    )


@pytest.mark.parametrize("ex", HOLDS, ids=lambda e: e["name"])
def test_low_limit_holds_are_straight_not_flexed(ex):
    """
    A hold with a limit under 15 deg can only be a straight-leg hold. This is the
    exact class of exercise the old absolute `angle > 15` threshold broke.
    """
    if ex["protocol_angle_limit"] < 15:
        assert ex["hold_target"] == "straight", (
            f"{ex['name']} has a {ex['protocol_angle_limit']} deg limit but is marked "
            f"'{ex['hold_target']}' — a flexed hold is unreachable under that limit."
        )


def test_ankle_pumps_is_the_only_ankle_tracked_exercise():
    ankle = [e["name"] for e in ALL if e["tracked_joint"] == "ankle"]
    assert ankle == ["Ankle Pumps"], (
        f"unexpected ankle-tracked exercises: {ankle}. The tracker's ankle rep "
        "counter is tuned specifically for pumps."
    )


# ---------------------------------------------------------------------------
# Fields survive the capping + serialisation path to the frontend
# ---------------------------------------------------------------------------

def test_cap_exercises_preserves_tracker_fields():
    raw = [e for e in ALL if e["name"] == "Quad Sets (Isometric)"]
    prescribed, _ = _cap_exercises(raw, max_angle=45)
    capped = prescribed[0]
    assert capped["tracked_joint"] == "knee"
    assert capped["hold_target"] == "straight"


def test_tracker_fields_reach_the_api_response():
    """The tracker reads these off the JSON — they must not be dropped."""
    resp = get_exercises_only(surgery_type="tkr", weeks_post_op=1, kl_grade=4)
    for ex in resp["exercise_list"]:
        assert "tracked_joint" in ex, f"{ex['name']} lost tracked_joint"
        assert "hold_target" in ex, f"{ex['name']} lost hold_target"


def test_kl_ceiling_tightens_the_flexed_hold_target():
    """
    A KL4 ceiling caps Standing Knee Bend from 90 to 45 deg. The flexed hold
    target is a fraction of the *effective* limit, so it must move down with it —
    otherwise the cap is cosmetic and the patient is still asked for 63 deg.

    (Standing Knee Bend rather than the seated version: seated exercises have a
    start_angle and get screened out entirely at a 45 deg ceiling.)
    """
    raw = [e for e in ALL if e["name"] == "Standing Knee Bend"]
    prescribed, excluded = _cap_exercises(raw, max_angle=45)

    assert excluded == [], "Standing Knee Bend has no start_angle, so it stays prescribable"
    bend = prescribed[0]
    assert bend["angle_limit"] == 45
    assert bend["angle_capped"] is True
    assert bend["angle_limit"] * FLEXED_HOLD_FRACTION <= 45


# ---------------------------------------------------------------------------
# Phase selection boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surgery", ["acl", "tkr", "meniscus", "arthroscopy"])
@pytest.mark.parametrize("weeks", [0, 1, 2, 3, 6, 7, 12, 13, 52, 520])
def test_get_phase_always_returns_a_phase(surgery, weeks):
    phase = get_phase(surgery, weeks, kl_grade=2)
    assert phase["exercises"], f"{surgery} @ week {weeks} returned an empty phase"
    assert phase["label"]


@pytest.mark.parametrize("kl", [0, 1, 2, 3, 4])
def test_non_surgical_falls_back_to_an_oa_level(kl):
    phase = get_phase("none", None, kl_grade=kl)
    assert phase["exercises"]


# ---------------------------------------------------------------------------
# Starting position vs safe ceiling
#
# Some exercises are performed from a position the patient cannot avoid: sitting
# in a chair puts the knee at ~90 deg before they move. Capping the target angle
# does nothing about that, so those exercises must be screened out for patients
# whose ceiling is below the starting position — otherwise the prescription
# alarms at them for sitting down.
# ---------------------------------------------------------------------------

KL_MAX_ANGLE = {0: 120, 1: 120, 2: 90, 3: 60, 4: 45}


@pytest.mark.parametrize("ex", ALL, ids=lambda e: e["name"])
def test_start_angle_is_plausible(ex):
    start = ex.get("start_angle")
    if start is not None:
        assert 0 < start <= 135, f"{ex['name']}: implausible start_angle {start}"


@pytest.mark.parametrize("ex", ALL, ids=lambda e: e["name"])
def test_angle_limit_is_reachable_from_the_start_position(ex):
    """
    An exercise's own ceiling must accommodate its own starting position.
    Seated Knee Extension used to declare a 45 deg ceiling while being performed
    seated at ~90 deg — the alarm fired before the patient moved.
    """
    start = ex.get("start_angle")
    if start is not None:
        assert start <= ex["protocol_angle_limit"], (
            f"{ex['name']} starts at {start} deg but declares a "
            f"{ex['protocol_angle_limit']} deg ceiling — unperformable as written."
        )


@pytest.mark.parametrize("kl", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("surgery", ["acl", "tkr", "meniscus", "arthroscopy", "none"])
def test_no_prescribed_exercise_starts_above_the_ceiling(surgery, kl):
    """The end-to-end guarantee: nothing reaches the patient that they cannot
    safely get into position for."""
    weeks = None if surgery == "none" else 1
    resp = get_exercises_only(surgery_type=surgery, weeks_post_op=weeks, kl_grade=kl)
    ceiling = KL_MAX_ANGLE[kl]

    for ex in resp["exercise_list"]:
        start = ex.get("start_angle")
        if start is not None:
            assert start <= ceiling, (
                f"{surgery} KL{kl}: '{ex['name']}' starts at {start} deg, above the "
                f"{ceiling} deg ceiling — it should have been excluded."
            )


@pytest.mark.parametrize("kl", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("surgery", ["acl", "tkr", "meniscus", "arthroscopy", "none"])
def test_prescription_is_never_left_empty_by_exclusions(surgery, kl):
    """Screening must not strip a phase down to nothing."""
    weeks = None if surgery == "none" else 1
    resp = get_exercises_only(surgery_type=surgery, weeks_post_op=weeks, kl_grade=kl)
    assert resp["exercise_list"], (
        f"{surgery} KL{kl}: every exercise was excluded — patient gets an empty plan"
    )


def test_severe_oa_excludes_the_seated_extension():
    """The case that prompted this: oa_severe is what a KL4 patient receives, and
    it contains a seated exercise starting at ~90 deg against a 45 deg ceiling."""
    resp = get_exercises_only(surgery_type="none", weeks_post_op=None, kl_grade=4)
    excluded = {e["name"] for e in resp["excluded_exercises"]}
    assert "Seated Knee Extension (Gravity Only)" in excluded
    prescribed = {e["name"] for e in resp["exercise_list"]}
    assert "Seated Knee Extension (Gravity Only)" not in prescribed


def test_exclusions_carry_a_reason():
    resp = get_exercises_only(surgery_type="none", weeks_post_op=None, kl_grade=4)
    for ex in resp["excluded_exercises"]:
        assert ex["reason"], f"{ex['name']} excluded without explanation"
        assert ex["start_angle"] > ex["max_angle"]


def test_mild_patients_keep_their_seated_exercises():
    """Screening must not over-fire — a KL0 knee has a 120 deg ceiling."""
    resp = get_exercises_only(surgery_type="none", weeks_post_op=None, kl_grade=0)
    assert resp["excluded_exercises"] == []


@pytest.mark.parametrize("ex", HOLDS, ids=lambda e: e["name"])
def test_hold_angle_never_exceeds_the_limit(ex):
    hold = ex.get("hold_angle")
    if hold is not None:
        assert hold <= ex["protocol_angle_limit"], (
            f"{ex['name']}: hold target {hold} deg is above its own ceiling"
        )


def test_seated_extension_needs_a_ceiling_of_at_least_its_start_angle():
    """
    Seated Knee Extension is performed sitting at ~90 deg, so any ceiling below
    that excludes it outright — capping the 45 deg extension target would not
    make sitting down safe.
    """
    raw = [e for e in ALL if e["name"] == "Seated Knee Extension (Gravity Only)"]

    prescribed, excluded = _cap_exercises(raw, max_angle=60)
    assert prescribed == [] and len(excluded) == 1, "60 deg ceiling cannot seat the patient"

    prescribed, excluded = _cap_exercises(raw, max_angle=120)
    assert excluded == [] and len(prescribed) == 1
    assert prescribed[0]["hold_angle"] == 45, "extension target survives an ample ceiling"


def test_hold_angle_is_capped_by_the_effective_ceiling():
    """
    Guard for future protocol entries: a hold target above the effective ceiling
    must be pulled down to it. No shipped exercise exercises this path today —
    Seated Knee Extension is the only one with a hold_angle, and its start_angle
    forces a ceiling well above its target — so it is tested synthetically.
    """
    synthetic = [{
        "name": "Synthetic Deep Hold",
        "description": "test fixture",
        "target_reps": 1,
        "target_sets": 1,
        "protocol_angle_limit": 110,
        "hold_seconds": 5,
        "tracked_joint": "knee",
        "hold_target": "flexed",
        "start_angle": None,
        "hold_angle": 100,
        "instructions": ["test fixture"],
        "cautions": None,
    }]

    prescribed, _ = _cap_exercises(synthetic, max_angle=60)
    assert prescribed[0]["angle_limit"] == 60
    assert prescribed[0]["hold_angle"] == 60, "hold target must not exceed the ceiling"
