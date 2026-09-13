"""
Two knees, two ceilings.

`knee_side` has been collected since the first version of this app, echoed back
in every response, and used for nothing. "Both" meant one X-ray graded once and
the result applied to a joint it had never seen — which is unsafe in one
direction and pointlessly restrictive in the other, depending on which knee was
photographed.

Two knees routinely differ by two KL grades, and grade 2 to grade 4 is 90° down
to 45°. So each film is read, screened and prescribed for on its own.

The flat fields carry the **more restrictive** side, so anything that ignores the
split — an older client, the denormalised columns on the stored record — still
gets a safe answer rather than a number too high for one of the two knees.

Run:  python -m pytest backend/tests -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_logic import build_prescription, merge_bilateral


def one_knee(side, grade, ceiling, **kw):
    base = {
        "kl_grade": grade, "health_score": 60, "max_angle": ceiling,
        "confidence": 0.8, "demo_mode": False, "knee_side": side,
        "surgery_type": "acl", "weeks_post_op": 8, "model_version": "test",
        "confidence_band": "high", "calibrated": True,
        "grade_probabilities": [0.05, 0.1, 0.7, 0.1, 0.05], "within_one_grade": 0.9,
    }
    base.update(kw)
    return build_prescription(**base)


# ---------------------------------------------------------------------------
# Each knee keeps its own answer
# ---------------------------------------------------------------------------

def test_each_knee_keeps_its_own_grade_and_ceiling():
    merged = merge_bilateral(one_knee("left", 2, 90), one_knee("right", 4, 45))

    assert merged["bilateral"] is True
    assert merged["knee_side"] == "both"
    assert len(merged["sides"]) == 2

    left, right = merged["sides"]
    assert (left["knee_side"], left["kl_grade"], left["max_angle"]) == ("left", 2, 90)
    assert (right["knee_side"], right["kl_grade"], right["max_angle"]) == ("right", 4, 45)


def test_each_side_carries_its_own_exercise_limits():
    """
    The point of the whole item. A 90° knee and a 45° knee get the same
    exercises at different limits — and the tracker reads `angle_limit` off
    whichever side it was launched for.
    """
    merged = merge_bilateral(one_knee("left", 2, 90), one_knee("right", 4, 45))
    left, right = merged["sides"]

    assert max(e["angle_limit"] for e in left["exercise_list"]) > \
           max(e["angle_limit"] for e in right["exercise_list"])
    for exercise in right["exercise_list"]:
        assert exercise["angle_limit"] <= 45


def test_the_worse_knee_may_have_exercises_withheld_the_better_one_keeps():
    merged = merge_bilateral(one_knee("left", 0, 120), one_knee("right", 4, 45))
    left, right = merged["sides"]
    assert len(right["excluded_exercises"]) >= len(left["excluded_exercises"])


def test_each_side_carries_its_own_confidence_and_distribution():
    merged = merge_bilateral(
        one_knee("left", 2, 90, confidence=0.91, grade_probabilities=[0, 0, 0.91, 0.09, 0]),
        one_knee("right", 3, 60, confidence=0.42, confidence_band="low",
                 grade_probabilities=[0, 0.1, 0.3, 0.42, 0.18]),
    )
    left, right = merged["sides"]
    assert left["confidence"] == 0.91
    assert right["confidence"] == 0.42
    assert right["confidence_band"] == "low"
    assert left["grade_probabilities"] != right["grade_probabilities"]


# ---------------------------------------------------------------------------
# The flat fields stay safe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("left_ceiling,right_ceiling", [(90, 45), (45, 90), (60, 60)])
def test_the_summary_shows_the_more_restricted_knee(left_ceiling, right_ceiling):
    """
    Anything reading only the flat fields — an older client, the stored
    columns — must get the cautious number, never one too high for one knee.
    """
    merged = merge_bilateral(
        one_knee("left", 2, left_ceiling), one_knee("right", 3, right_ceiling)
    )
    assert merged["max_angle"] == min(left_ceiling, right_ceiling)


def test_the_flat_exercise_list_is_the_restricted_ones():
    merged = merge_bilateral(one_knee("left", 0, 120), one_knee("right", 4, 45))
    for exercise in merged["exercise_list"]:
        assert exercise["angle_limit"] <= 45


def test_a_single_knee_analysis_says_it_is_not_bilateral():
    """The keys are always present, so a client never branches on their absence."""
    p = one_knee("left", 2, 90)
    assert p["bilateral"] is False
    assert p["sides"] == []


# ---------------------------------------------------------------------------
# What the patient is told
# ---------------------------------------------------------------------------

def test_the_rationale_names_both_knees_when_they_differ():
    merged = merge_bilateral(one_knee("left", 2, 90), one_knee("right", 4, 45))
    r = merged["rationale"]

    assert "Left" in r and "Right" in r
    assert "90" in r and "45" in r
    assert "own limit" in r


def test_the_rationale_says_so_when_they_match():
    merged = merge_bilateral(one_knee("left", 2, 90), one_knee("right", 2, 90))
    assert "same" in merge_bilateral(one_knee("left", 2, 90), one_knee("right", 2, 90))["rationale"]
    assert "Left:" not in merged["rationale"], "no need to spell out two identical readings"


def test_the_single_knee_rationale_is_kept_underneath():
    """The per-knee explanation still matters; the bilateral note goes on top."""
    strict = one_knee("right", 4, 45)
    merged = merge_bilateral(one_knee("left", 2, 90), strict)
    assert merged["rationale"].endswith(strict["rationale"])


# ---------------------------------------------------------------------------
# It composes with the other model-trust rules
# ---------------------------------------------------------------------------

def test_a_replaced_joint_on_one_side_only_is_handled_per_side():
    """
    Staged bilateral replacement is ordinary: one knee done, the other waiting.
    The gate is per knee because the prescription is.
    """
    replaced = one_knee("left", 4, 45, surgery_type="tkr")
    native = one_knee("right", 4, 45, surgery_type="tkr")

    merged = merge_bilateral(replaced, native)
    for side in merged["sides"]:
        assert side["kl_applicable"] is False


def test_suspected_hardware_travels_with_the_knee_it_was_seen_in():
    merged = merge_bilateral(
        one_knee("left", 2, 90),
        one_knee("right", 2, 90, hardware_suspected=True, hardware_reason="Bright solid region."),
    )
    left, right = merged["sides"]
    assert left["hardware_suspected"] is False
    assert right["hardware_suspected"] is True


def test_each_side_can_carry_its_own_explanation():
    left = one_knee("left", 2, 90)
    right = one_knee("right", 3, 60)
    left["explanation"] = "data:image/png;base64,AAA"
    right["explanation"] = "data:image/png;base64,BBB"

    merged = merge_bilateral(left, right)
    assert merged["sides"][0]["explanation"] == "data:image/png;base64,AAA"
    assert merged["sides"][1]["explanation"] == "data:image/png;base64,BBB"


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="the app needs sqlalchemy")

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier, png_bytes

import main


@pytest.fixture
def client(monkeypatch):
    reset_database()
    stub_inference(monkeypatch)
    main._rate_buckets.clear()
    with TestClient(main.app) as c:
        monkeypatch.setattr(main, "classifier", StubClassifier())
        yield c


def analyse(client, files=None, **form):
    data = {"knee_side": "left", "surgery_type": "acl", "weeks_post_op": "8"}
    data.update(form)
    return client.post(
        "/analyse-xray",
        files=files or {"image": ("knee.png", png_bytes(), "image/png")},
        data=data,
    )


def test_asking_for_both_knees_with_one_xray_is_refused(client):
    """
    One film cannot answer for two joints. Silently grading it twice is exactly
    what this item exists to stop.
    """
    r = analyse(client, knee_side="both")
    assert r.status_code == 422
    assert "two X-rays" in r.json()["detail"]


def test_two_xrays_produce_two_sides(client):
    r = analyse(
        client,
        files={
            "image": ("left.png", png_bytes(), "image/png"),
            "image_right": ("right.png", png_bytes(), "image/png"),
        },
        knee_side="both",
    )
    assert r.status_code == 200

    body = r.json()
    assert body["bilateral"] is True
    assert [s["knee_side"] for s in body["sides"]] == ["left", "right"]
    assert all(s["exercise_list"] for s in body["sides"])


def test_a_single_knee_request_is_unchanged(client):
    """The second file is optional; nothing about the old shape moved."""
    body = analyse(client, knee_side="left").json()
    assert body["bilateral"] is False
    assert body["sides"] == []
    assert body["knee_side"] == "left"
    assert body["exercise_list"]


def test_a_second_image_is_ignored_for_a_single_knee(client):
    body = analyse(
        client,
        files={
            "image": ("left.png", png_bytes(), "image/png"),
            "image_right": ("right.png", png_bytes(), "image/png"),
        },
        knee_side="right",
    ).json()
    assert body["bilateral"] is False
    assert body["knee_side"] == "right"


def test_a_bad_second_image_says_which_one_it_means(client):
    """
    "Image contrast is too low" is not much help when two films were sent.
    """
    import io

    import numpy as np
    from PIL import Image

    flat = io.BytesIO()
    Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).convert("RGB").save(flat, "PNG")

    r = analyse(
        client,
        files={
            "image": ("left.png", png_bytes(), "image/png"),
            "image_right": ("right.png", flat.getvalue(), "image/png"),
        },
        knee_side="both",
    )
    assert r.status_code == 422
    assert r.json()["detail"].startswith("Right X-ray:")
