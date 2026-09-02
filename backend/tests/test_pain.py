"""
Pain and perceived exertion.

Two taps that carry more clinical signal than anything else the app collects.
Range of motion says how far the knee moved; it says nothing about whether
moving it that far was a good idea. Rehab that hurts a little during and settles
afterwards is working. The same exercise leaving someone worse every single time
is not — and only the before/after pair shows that.

What these pin down:

  * A rating is asked for, never required. Nothing is blocked without one.
  * Impossible scores are refused: NPRS and Borg CR10 are both 0–10, and a
    number outside that is a bug, not a very sore knee.
  * A report cannot invent a session. A rating with no completed set behind it
    would otherwise inflate an adherence record with work that never happened.
  * Pain survives an unverified camera angle. The gate says nothing about what
    the patient felt.

Run:  python -m pytest backend/tests -q
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="pain tests need sqlalchemy")

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier

import main
from db import SessionLocal
from models import ExerciseSession, ExerciseSet, Patient

PASSWORD = "correct-horse-battery"


@pytest.fixture
def client(monkeypatch):
    reset_database()
    stub_inference(monkeypatch)
    main._rate_buckets.clear()
    with TestClient(main.app) as c:
        monkeypatch.setattr(main, "classifier", StubClassifier())
        yield c


@pytest.fixture
def token(client):
    return client.post(
        "/auth/register", json={"email": "ada@example.com", "password": PASSWORD}
    ).json()["access_token"]


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


SITTING = "sitting-aaaaaaaa"


def set_payload(**over):
    body = {
        "client_session_id": SITTING,
        "exercise_name":     "Mini Squat",
        "tracked_joint":     "knee",
        "knee_side":         "left",
        "angle_limit":       60,
        "target_reps":       10,
        "target_sets":       2,
        "set_index":         1,
        "reps_completed":    10,
        "duration_seconds":  44.0,
        "peak_flexion_deg":  54.0,
        "breach_count":      0,
        "breach_seconds":    0.0,
        "mean_visibility":   0.93,
        "suspended_seconds": 0.0,
        "view_verified":     True,
    }
    body.update(over)
    return body


def post_set(client, tok, **over):
    return client.post("/sessions/sets", json=set_payload(**over), headers=bearer(tok))


def report(client, tok, **body):
    body.setdefault("client_session_id", SITTING)
    return client.post("/sessions/report", json=body, headers=bearer(tok))


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def test_pain_before_arrives_with_the_first_set(client, token):
    r = post_set(client, token, pain_before=4)
    assert r.status_code == 201
    assert r.json()["pain_before"] == 4


def test_pain_before_is_kept_from_the_first_set_only(client, token):
    """
    It rides on every set so a retry of the first one still carries it, but it
    describes the moment before the session started — a later set must not be
    able to rewrite it.
    """
    post_set(client, token, set_index=1, pain_before=4)
    r = post_set(client, token, set_index=2, pain_before=9)
    assert r.json()["pain_before"] == 4


def test_a_finished_session_records_how_it_felt(client, token):
    post_set(client, token, pain_before=4)

    r = report(client, token, pain_after=2, rpe=6)
    assert r.status_code == 200
    body = r.json()
    assert body["pain_before"] == 4
    assert body["pain_after"] == 2
    assert body["rpe"] == 6


def test_a_second_report_adds_without_erasing(client, token):
    post_set(client, token, pain_before=5)
    report(client, token, pain_after=3)

    r = report(client, token, rpe=7)
    assert r.json()["pain_after"] == 3, "adding exertion must not wipe the pain score"
    assert r.json()["rpe"] == 7


def test_a_report_cannot_invent_a_session(client, token):
    r = report(client, token, client_session_id="never-happened-1", pain_after=3)
    assert r.status_code == 404

    with SessionLocal() as db:
        assert db.query(ExerciseSession).count() == 0


def test_reporting_requires_an_account(client):
    assert client.post("/sessions/report", json={"client_session_id": SITTING, "pain_after": 3}).status_code == 401


def test_one_patient_cannot_report_on_anothers_session(client, token):
    other = client.post(
        "/auth/register", json={"email": "bob@example.com", "password": PASSWORD}
    ).json()["access_token"]
    post_set(client, token, pain_before=4)

    # Same client_session_id, different account: it simply does not exist there.
    assert report(client, other, pain_after=9).status_code == 404

    with SessionLocal() as db:
        assert db.query(ExerciseSession).one().pain_after is None


# ---------------------------------------------------------------------------
# It is optional, and it is bounded
# ---------------------------------------------------------------------------

def test_a_session_without_any_rating_is_still_valid(client, token):
    r = post_set(client, token)
    assert r.status_code == 201
    assert r.json()["pain_before"] is None
    assert r.json()["pain_after"] is None
    assert r.json()["rpe"] is None


@pytest.mark.parametrize("value", [0, 10])
def test_the_ends_of_the_scale_are_valid(client, token, value):
    assert post_set(client, token, pain_before=value).status_code == 201
    assert report(client, token, pain_after=value, rpe=value).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("pain_before", 11), ("pain_before", -1),
])
def test_impossible_pain_scores_are_refused_on_a_set(client, token, field, value):
    assert post_set(client, token, **{field: value}).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("pain_after", 11), ("pain_after", -1), ("rpe", 11), ("rpe", -1),
])
def test_impossible_scores_are_refused_on_a_report(client, token, field, value):
    post_set(client, token)
    assert report(client, token, **{field: value}).status_code == 422


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def seed(email="ada@example.com", *, days_ago=0, before=None, after=None,
         rpe=None, verified=True, peak=50.0, exercise="Mini Squat"):
    started = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    started -= timedelta(days=days_ago)
    with SessionLocal() as db:
        patient = db.query(Patient).filter_by(email=email).one()
        session = ExerciseSession(
            patient_id="", client_session_id=f"seed-{days_ago}-{exercise}",
            exercise_name=exercise, tracked_joint="knee", knee_side="left",
            angle_limit=60, target_reps=10, target_sets=1,
            started_at=started, last_set_at=started, sets_completed=1,
            completed=True, view_verified=verified,
            pain_before=before, pain_after=after, rpe=rpe,
        )
        session.patient_id = patient.id
        db.add(session)
        db.flush()
        db.add(ExerciseSet(session_id=session.id, set_index=1, reps_completed=10,
                           duration_seconds=40.0, peak_flexion_deg=peak, mean_visibility=0.9))
        db.commit()


def progress(client, tok, **params):
    return client.get("/me/progress", params=params, headers=bearer(tok)).json()


def test_pain_is_charted_day_by_day(client, token):
    seed(days_ago=2, before=6, after=4)
    seed(days_ago=1, before=5, after=3)
    seed(days_ago=0, before=4, after=2)

    trend = progress(client, token)["pain_trend"]
    assert [p["pain_before"] for p in trend] == [6.0, 5.0, 4.0]
    assert [p["pain_after"] for p in trend] == [4.0, 3.0, 2.0]


def test_two_sessions_in_a_day_are_averaged(client, token):
    seed(days_ago=0, before=6, after=4, exercise="Mini Squat")
    seed(days_ago=0, before=4, after=2, exercise="Quad Sets")

    point = progress(client, token)["pain_trend"][0]
    assert point["pain_before"] == 5.0
    assert point["pain_after"] == 3.0
    assert point["sessions"] == 2


def test_the_direction_of_pain_change_is_summarised(client, token):
    # Settling afterwards: the change is negative, which is the good direction.
    seed(days_ago=1, before=6, after=4)
    seed(days_ago=0, before=5, after=2)

    summary = progress(client, token)["summary"]
    assert summary["mean_pain_change"] == -2.5
    assert summary["latest_pain_after"] == 2
    assert summary["sessions_with_pain"] == 2


def test_sessions_that_leave_a_patient_worse_read_as_positive_change(client, token):
    seed(days_ago=1, before=2, after=6)
    seed(days_ago=0, before=3, after=7)

    assert progress(client, token)["summary"]["mean_pain_change"] == 4.0


def test_a_half_reported_session_does_not_distort_the_change(client, token):
    """Only sessions with both halves can contribute a change."""
    seed(days_ago=1, before=6, after=3)     # change -3
    seed(days_ago=0, before=8)              # no "after": contributes nothing

    summary = progress(client, token)["summary"]
    assert summary["mean_pain_change"] == -3.0
    assert summary["sessions_with_pain"] == 2


def test_pain_survives_an_unverified_camera_angle(client, token):
    """
    The view gate is about measured angles. It says nothing about what the
    patient felt, so discarding their pain score with the geometry would throw
    away the more clinically useful of the two.
    """
    seed(days_ago=0, before=7, after=5, verified=False, peak=120.0)

    body = progress(client, token)
    assert body["pain_trend"][0]["pain_before"] == 7.0
    assert body["pain_trend"][0]["pain_after"] == 5.0
    # …while the nonsense angle from the same session is still excluded.
    assert body["rom_by_exercise"] == []
    assert body["summary"]["best_flexion_deg"] is None


def test_no_ratings_means_an_empty_pain_series(client, token):
    seed(days_ago=0)
    body = progress(client, token)
    assert body["pain_trend"] == []
    assert body["summary"]["mean_pain_change"] is None
    assert body["summary"]["sessions_with_pain"] == 0
