"""
Confidence calibration and out-of-distribution screening.

Two claims this project makes to patients that need to hold up:

  1. The confidence number means something. A softmax maximum is not a
     probability of being right; at ~71% accuracy a model routinely reports
     90%+. Temperature scaling fixes the number without touching the
     prediction, and until it has been fitted the API must say so rather than
     quote a percentage.

  2. The thing graded was a knee X-ray. The classifier has five outputs and no
     "not a knee" class, so without an energy screen a chest film or a photo of
     a wall returns a confident grade that then sets a movement ceiling.

The tests that need torch are skipped when it is absent, so the suite still
runs on a machine with only the API dependencies installed.

Run:  python -m pytest backend/tests -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_logic import build_prescription

torch = pytest.importorskip("torch", reason="calibration maths needs torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))


# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("conf,expected", [
    (0.00, "low"), (0.30, "low"), (0.49, "low"),
    (0.50, "moderate"), (0.62, "moderate"), (0.74, "moderate"),
    (0.75, "high"), (0.91, "high"), (1.00, "high"),
])
def test_confidence_bands(conf, expected):
    from model.inference import confidence_band
    assert confidence_band(conf) == expected


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def _overconfident_logits(n=600, classes=5, seed=0):
    """Logits that are right ~70% of the time but scream ~99% certainty —
    the failure temperature scaling exists to correct."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (n,), generator=g)
    logits = torch.randn(n, classes, generator=g) * 0.5
    for i, y in enumerate(labels):
        # Right most of the time, but always with a huge margin.
        target = int(y) if torch.rand(1, generator=g).item() < 0.70 else int((y + 1) % classes)
        logits[i, target] += 12.0
    return logits, labels


def test_temperature_reduces_calibration_error():
    from train import expected_calibration_error, fit_temperature, serve_probs

    logits, labels = _overconfident_logits()

    before = expected_calibration_error(serve_probs(logits, None, None, 1.0), labels)
    T, _ = fit_temperature(logits, None, labels, None)
    after = expected_calibration_error(serve_probs(logits, None, None, T), labels)

    assert T > 1.0, f"an overconfident model needs T>1 to soften, got {T:.3f}"
    assert after < before, f"ECE should improve: {before:.4f} -> {after:.4f}"


def test_temperature_never_changes_the_prediction_without_tta():
    """
    On a single forward pass, dividing logits is monotonic, so temperature
    cannot move argmax. This must hold exactly.
    """
    from train import fit_temperature, serve_probs

    logits, labels = _overconfident_logits()
    T, _ = fit_temperature(logits, None, labels, None)

    preds_before = serve_probs(logits, None, None, 1.0).argmax(dim=1)
    preds_after = serve_probs(logits, None, None, T).argmax(dim=1)

    assert torch.equal(preds_before, preds_after)


def test_temperature_drift_under_tta_stays_negligible():
    """
    With flip-TTA the branches are averaged AFTER softmax, and that average is
    NOT monotonic in T — a sharper T lets the more confident branch dominate, so
    a borderline case can flip. Measured on the real test split: 1 image in 1656
    (0.06%).

    This is the honest version of the invariant above. Asserting exact equality
    here would be asserting something false; the real requirement is that drift
    stays in the noise rather than silently altering diagnoses at scale.
    """
    from train import fit_temperature, serve_probs

    g = torch.Generator().manual_seed(7)
    logits, labels = _overconfident_logits(seed=7)
    # A second view that disagrees slightly, as a horizontal flip would.
    flip = logits + torch.randn(logits.shape, generator=g) * 0.4

    T, _ = fit_temperature(logits, flip, labels, None)
    before = serve_probs(logits, flip, None, 1.0).argmax(dim=1)
    after = serve_probs(logits, flip, None, T).argmax(dim=1)

    drift = (before != after).float().mean().item()
    assert drift < 0.02, f"temperature moved {drift:.1%} of predictions under TTA"


def test_temperature_can_sharpen_an_underconfident_model():
    """
    T<1 is a real outcome, not a bug. This project's B4 fitted to T=0.73 — mean
    confidence 0.636 against 0.699 accuracy, i.e. under-claiming. The fit must
    bracket both sides of 1.0.
    """
    from train import fit_temperature, serve_probs

    g = torch.Generator().manual_seed(3)
    n, classes = 600, 5
    labels = torch.randint(0, classes, (n,), generator=g)
    # Correct most of the time but with a thin margin -> under-confident.
    logits = torch.randn(n, classes, generator=g) * 0.3
    for i, y in enumerate(labels):
        target = int(y) if torch.rand(1, generator=g).item() < 0.85 else int((y + 1) % classes)
        logits[i, target] += 0.55

    T, _ = fit_temperature(logits, None, labels, None)
    mean_before = serve_probs(logits, None, None, 1.0).max(dim=1).values.mean().item()
    mean_after = serve_probs(logits, None, None, T).max(dim=1).values.mean().item()

    assert T < 1.0, f"an under-confident model needs T<1 to sharpen, got {T:.3f}"
    assert mean_after > mean_before


def test_temperature_lowers_reported_confidence_when_overconfident():
    from train import fit_temperature, serve_probs

    logits, labels = _overconfident_logits()
    T, _ = fit_temperature(logits, None, labels, None)

    mean_before = serve_probs(logits, None, None, 1.0).max(dim=1).values.mean().item()
    mean_after = serve_probs(logits, None, None, T).max(dim=1).values.mean().item()
    accuracy = (serve_probs(logits, None, None, T).argmax(1) == labels).float().mean().item()

    assert mean_before > 0.95, "fixture should start wildly overconfident"
    assert mean_after < mean_before
    # The corrected number should land near actual accuracy, not far above it.
    assert abs(mean_after - accuracy) < abs(mean_before - accuracy)


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    from train import expected_calibration_error

    n = 1000
    probs = torch.full((n, 2), 0.5)
    labels = torch.tensor([0, 1] * (n // 2))
    assert expected_calibration_error(probs, labels) < 0.02


# ---------------------------------------------------------------------------
# OOD energy reference
# ---------------------------------------------------------------------------

def test_energy_is_lower_for_confident_in_distribution_logits():
    """Energy = -logsumexp(logits): low for inputs the model recognises."""
    from train import fit_energy_reference

    g = torch.Generator().manual_seed(1)
    in_dist = torch.randn(300, 5, generator=g) * 0.5
    in_dist[:, 0] += 10.0                       # a decisive, familiar response
    ood = torch.randn(300, 5, generator=g) * 0.5  # flat, unsure response

    e_in = fit_energy_reference(in_dist)
    e_ood = fit_energy_reference(ood)

    assert e_in["p50"] < e_ood["p50"], "in-distribution energy must sit lower"


def test_energy_reference_percentiles_are_ordered():
    from train import fit_energy_reference

    g = torch.Generator().manual_seed(2)
    ref = fit_energy_reference(torch.randn(500, 5, generator=g))
    assert ref["p50"] <= ref["p95"] <= ref["p99"]


def test_reject_threshold_sits_beyond_p99():
    """A genuine radiograph in the tail should warn, not be rejected."""
    from model.inference import load_calibration

    ckpt = {"temperature": 1.3, "energy_ref": {"p50": -8.0, "p95": -5.0, "p99": -4.0}}
    saved = torch.load
    try:
        torch.load = lambda *a, **k: ckpt
        cal = load_calibration("ignored")
    finally:
        torch.load = saved

    assert cal["calibrated"] is True
    assert cal["warn_threshold"] == -5.0
    assert cal["reject_threshold"] > cal["warn_threshold"]
    assert cal["reject_threshold"] > -4.0, "reject must sit beyond p99"


def test_missing_calibration_degrades_instead_of_pretending():
    """A checkpoint predating calibration must report calibrated=False and
    disable OOD screening, not silently behave as though both are present."""
    from model.inference import load_calibration

    saved = torch.load
    try:
        torch.load = lambda *a, **k: {"arch": "b4"}
        cal = load_calibration("ignored")
    finally:
        torch.load = saved

    assert cal["temperature"] == 1.0
    assert cal["calibrated"] is False
    assert cal["reject_threshold"] is None
    assert cal["warn_threshold"] is None


def test_model_version_is_derived_from_the_checkpoint():
    """Version must name the actual weights, not a hardcoded constant."""
    from model.inference import _derive_model_version

    assert _derive_model_version({"arch": "b4"}) == "efficientnet_b4"
    assert _derive_model_version(
        {"arch": "b3", "trained_at": "2026-08-18T10:00:00+00:00", "temperature": 1.4}
    ) == "efficientnet_b3_20260818_cal"


# ---------------------------------------------------------------------------
# What reaches the patient
# ---------------------------------------------------------------------------

def _prescription(**kw):
    base = {
        "kl_grade": 2, "health_score": 60, "max_angle": 90, "confidence": 0.82,
        # ACL, not TKR: these tests are about how confidence in the KL grade is
        # worded, and a replaced joint does not use the grade at all — its
        # rationale says so instead. See test_prosthesis_gate.py.
        "demo_mode": False, "knee_side": "left", "surgery_type": "acl",
        "weeks_post_op": 3, "model_version": "test",
    }
    base.update(kw)
    return build_prescription(**base)


def test_uncalibrated_prescription_does_not_quote_a_percentage():
    p = _prescription(confidence_band="high", calibrated=False)
    assert "82%" not in p["rationale"]
    assert "high confidence" in p["rationale"]


def test_calibrated_prescription_may_quote_the_percentage():
    p = _prescription(confidence_band="high", calibrated=True)
    assert "82% calibrated confidence" in p["rationale"]


def test_low_confidence_tells_the_patient_to_check_with_a_physio():
    p = _prescription(confidence=0.31, confidence_band="low", calibrated=True)
    assert "not confident" in p["rationale"]
    assert "physiotherapist" in p["rationale"]


def test_ood_suspicion_is_surfaced_in_the_rationale():
    p = _prescription(ood_suspected=True)
    assert "outside the range" in p["rationale"]


def test_flags_reach_the_response_body():
    p = _prescription(confidence_band="moderate", calibrated=True, ood_suspected=True)
    assert p["confidence_band"] == "moderate"
    assert p["calibrated"] is True
    assert p["ood_suspected"] is True
