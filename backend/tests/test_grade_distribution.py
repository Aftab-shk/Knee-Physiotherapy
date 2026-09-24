"""
The full grade distribution.

The model was trained with an ordinal loss and its confidence is temperature
scaled, so it already knows more than "grade 2". The mass sitting beside the
winner is the model saying the answer is *nearby* — and between grade 2 and
grade 3 lies thirty degrees of permitted flexion, so "probably 2, possibly 3" is
not a nicety. It is the difference between a reading someone should act on and
one they should check.

The checkpoint scores 70.3% exact and 95.3% within one grade. Reporting only the
first number, as a single confidence figure did, describes the model as far less
useful than it is — and hides the cases where it is genuinely torn.

Run:  python -m pytest backend/tests -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_logic import build_prescription


def prescribe(**kw):
    base = {
        "kl_grade": 2, "health_score": 60, "max_angle": 90, "confidence": 0.62,
        "demo_mode": False, "knee_side": "left", "surgery_type": "acl",
        "weeks_post_op": 6, "model_version": "test",
    }
    base.update(kw)
    return build_prescription(**base)


# ---------------------------------------------------------------------------
# It reaches the response
# ---------------------------------------------------------------------------

def test_the_distribution_travels_with_the_prescription():
    probs = [0.05, 0.15, 0.62, 0.15, 0.03]
    p = prescribe(grade_probabilities=probs, within_one_grade=0.92)

    assert p["grade_probabilities"] == probs
    assert p["within_one_grade"] == 0.92


def test_a_caller_that_supplies_nothing_still_gets_a_valid_shape():
    """
    Five slots, always. A frontend drawing five bars should never have to guard
    against a missing one.
    """
    p = prescribe()
    assert len(p["grade_probabilities"]) == 5
    assert p["within_one_grade"] == 0.0


def test_the_distribution_is_copied_not_aliased():
    """A caller mutating its own list afterwards must not rewrite the record."""
    probs = [0.1, 0.1, 0.6, 0.1, 0.1]
    p = prescribe(grade_probabilities=probs)
    probs[0] = 0.99
    assert p["grade_probabilities"][0] == 0.1


# ---------------------------------------------------------------------------
# What the model produces
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="the real predictor needs torch")
pytest.importorskip("PIL")

import io

import numpy as np
from model.inference import KneeClassifier
from PIL import Image


@pytest.fixture(scope="module")
def classifier():
    clf = KneeClassifier()
    if clf.demo_mode:
        pytest.skip("no checkpoint available; the demo path is covered separately")
    return clf


def an_image(seed=0, size=256):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(20, 235, (size, size), dtype=np.uint8)).convert("RGB").save(
        buf, format="PNG")
    return buf.getvalue()


def test_the_distribution_is_a_distribution(classifier):
    result = classifier.predict(an_image())
    probs = result["grade_probabilities"]

    assert len(probs) == 5
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=0.01)


def test_the_reported_grade_is_the_one_with_the_most_mass(classifier):
    """
    Not a tautology: the probability vector is indexed by *model class*, and the
    checkpoint carries its own class-to-grade mapping. Getting that backwards
    would produce a plausible-looking distribution pointing at the wrong grade.
    """
    result = classifier.predict(an_image(seed=7))
    probs = result["grade_probabilities"]
    assert probs.index(max(probs)) == result["kl_grade"]


def test_the_top_probability_matches_the_confidence(classifier):
    result = classifier.predict(an_image(seed=2))
    assert max(result["grade_probabilities"]) == pytest.approx(result["confidence"], abs=0.01)


def test_within_one_grade_is_the_winner_and_its_neighbours(classifier):
    result = classifier.predict(an_image(seed=4))
    grade, probs = result["kl_grade"], result["grade_probabilities"]

    expected = sum(probs[g] for g in (grade - 1, grade, grade + 1) if 0 <= g <= 4)
    assert result["within_one_grade"] == pytest.approx(expected, abs=0.01)


def test_within_one_grade_is_never_below_the_top_confidence(classifier):
    """It contains the winner, so it cannot be smaller than it."""
    result = classifier.predict(an_image(seed=5))
    assert result["within_one_grade"] >= result["confidence"] - 0.01


def test_the_same_image_gives_the_same_distribution(classifier):
    image = an_image(seed=9)
    assert classifier.predict(image)["grade_probabilities"] == \
           classifier.predict(image)["grade_probabilities"]


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def test_demo_mode_returns_a_correctly_shaped_distribution():
    """
    Shaped like a real one so the frontend has five bars to draw, and never
    claimed to be calibrated — it is arithmetic on an MD5, not a probability.
    """
    import os

    from model.inference import KneeClassifier as KC

    os.environ["MODEL_PATH"] = "definitely-not-a-checkpoint.pth"
    try:
        clf = KC()
        assert clf.demo_mode is True
        result = clf.predict(an_image(seed=11))
    finally:
        os.environ.pop("MODEL_PATH", None)

    probs = result["grade_probabilities"]
    assert len(probs) == 5
    assert sum(probs) == pytest.approx(1.0, abs=0.01)
    assert probs.index(max(probs)) == result["kl_grade"]
    assert result["calibrated"] is False, "a hash is not a calibrated probability"


def test_demo_mass_falls_away_from_the_chosen_grade():
    """An ordinal model's neighbours carry more than its distant grades."""
    import os

    from model.inference import KneeClassifier as KC

    os.environ["MODEL_PATH"] = "definitely-not-a-checkpoint.pth"
    try:
        clf = KC()
        # Search for an image the demo hash puts in the middle, so both
        # neighbours exist.
        for seed in range(40):
            result = clf.predict(an_image(seed=seed))
            if result["kl_grade"] == 2:
                break
        else:
            pytest.skip("no mid-grade demo image found")
    finally:
        os.environ.pop("MODEL_PATH", None)

    probs = result["grade_probabilities"]
    assert probs[1] > probs[0]
    assert probs[3] > probs[4]
