"""
Upload quality checks — the polarity gate in particular.

An inverted radiograph is the failure mode worth a test of its own, because it
is the one that does not look like a failure. Every other rejected upload is
obviously wrong: blank, a thin strip, a GIF. A negative is a real X-ray carrying
real anatomy, so it passes contrast, size, format and exposure, reaches the
model, and comes back with a confident grade that happens to be the wrong one —
KL 0 became KL 3 on the films in this repo, and the energy-based OOD screen read
both at the same number.

The guarantee asserted below is the useful one: the statistic is antisymmetric,
so accepting a film is the same statement as rejecting its negative.

Run:  python -m pytest backend/tests/test_image_checks.py -q
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from model.image_checks import MIN_POLARITY, _polarity, validate_image
from PIL import Image


def png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def film(surround: float = 8.0, seed: int = 7, size: int = 512):
    """
    A knee-shaped radiograph: dense bone down the middle, soft tissue around it,
    and whatever the caller wants beyond that — direct exposure at 8, or tissue
    running to the edge at 95 for a tightly collimated view.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), float(surround), np.float32)
    img[np.abs(xx - size / 2) < size * 0.30] = 95                       # soft tissue
    shaft = (np.abs(xx - size / 2) < size * 0.16) & (
        (yy < size * 0.42) | (yy > size * 0.58)
    )
    img[shaft] = 205                                                     # femur / tibia
    img[(np.abs(xx - size / 2) < size * 0.20)
        & (np.abs(yy - size / 2) < size * 0.05)] = 150                   # joint space
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("surround", [8.0, 95.0], ids=["dark-surround", "collimated"])
def test_a_radiograph_passes_and_its_negative_does_not(surround):
    ok, message = validate_image(png(film(surround)))
    assert ok, message

    ok, message = validate_image(png(255 - film(surround)))
    assert not ok
    assert "negative" in message.lower()


def test_the_statistic_is_antisymmetric():
    """
    Why one threshold covers both directions: negating an image negates the
    score exactly, because 255-x flips both medians and leaves the spread alone.
    So a film scoring above +|MIN_POLARITY| has its own negative rejected, and no
    separate 'is it too bright' rule is needed.
    """
    arr = film()
    assert _polarity(arr) == pytest.approx(-_polarity(255 - arr), abs=1e-6)
    assert _polarity(arr) > abs(MIN_POLARITY)


def test_a_frame_with_no_background_is_let_through_rather_than_refused():
    """
    A frame with no centre-to-border contrast — bands running straight across —
    scores near zero whichever way round it is. The gate cannot see this case,
    and the threshold is set slack so that it passes: refusing a genuine film is
    the worse of the two errors.

    Real 224px crops of the joint do not look like this, as was first feared:
    bone is still the bright centre, and 1656 real films scored a median +1.27.
    """
    rng = np.random.default_rng(3)
    size = 512
    yy, _ = np.mgrid[0:size, 0:size]
    roi = np.full((size, size), 95.0, np.float32)
    roi[(yy < size * 0.40) | (yy > size * 0.60)] = 205
    roi = np.clip(roi + rng.normal(0, 6, roi.shape), 0, 255).astype(np.uint8)

    assert abs(_polarity(roi)) < abs(MIN_POLARITY)
    assert validate_image(png(roi))[0]
    assert validate_image(png(255 - roi))[0]


def test_a_white_collimation_band_does_not_trip_the_gate():
    """Medians, not means — an unexposed strip along one edge must not flip it."""
    arr = film()
    arr[: int(arr.shape[0] * 0.06)] = 250
    arr[-int(arr.shape[0] * 0.06):] = 250
    assert validate_image(png(arr))[0]


def test_noise_still_passes_every_other_check():
    """The fixture the API tests upload: no structure, but no polarity claim either."""
    rng = np.random.default_rng(0)
    noise = rng.integers(20, 235, (64, 64), dtype=np.uint8)
    assert validate_image(png(noise))[0]
