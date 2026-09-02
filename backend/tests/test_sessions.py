"""
Recording what the webcam tracker measures.

Before this existed, tracker.html computed peak flexion, breach time and
landmark visibility at thirty frames a second and threw all of it away when the
set ended. These tests pin the properties that make the stored version worth
trusting:

  * Sets are posted one at a time, so stopping after two of three keeps two.
  * A retried post must not turn one set into two — the browser retries on a
    dropped connection, and a duplicated set inflates an adherence record.
  * A session performed with the camera gate overridden must stay marked as
    such. Angles measured from an unverified viewpoint read low, and quietly
    charting them as range-of-motion progress would show a patient improving
    while they were doing no such thing.
  * Nobody can file a session against someone else's prescription.

Run:  python -m pytest backend/tests -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="session tests need sqlalchemy")

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier, png_bytes

import main
from db import SessionLocal
from models import ExerciseSession, ExerciseSet

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
    r = client.post("/auth/register", json={"email": "ada@example.com", "password": PASSWORD})
    assert r.status_code == 201
    return r.json()["access_token"]


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


def set_payload(**over):
    """A plausible completed set: 10 reps of a 60°-limited mini squat."""
    body = {
        "client_session_id": "session-aaaaaaaaaaaa",
        "exercise_name":     "Mini Squat",
        "tracked_joint":     "knee",
        "knee_side":         "left",
        "angle_limit":       60,
        "target_reps":       10,
        "target_sets":       3,
        "set_index":         1,
        "reps_completed":    10,
        "duration_seconds":  48.5,
        "peak_flexion_deg":  57.4,
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


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def test_a_set_is_stored_with_everything_the_tracker_measured(client, token):
    r = post_set(client, token)
    assert r.status_code == 201

    body = r.json()
    assert body["exercise_name"] == "Mini Squat"
    assert body["sets_completed"] == 1
    assert body["completed"] is False          # 1 of 3
    assert len(body["sets"]) == 1

    row = body["sets"][0]
    assert row["set_index"] == 1
    assert row["reps_completed"] == 10
    assert row["peak_flexion_deg"] == pytest.approx(57.4)
    assert row["mean_visibility"] == pytest.approx(0.93)
    assert row["duration_seconds"] == pytest.approx(48.5)


def test_sets_accumulate_into_one_session(client, token):
    for i in (1, 2, 3):
        r = post_set(client, token, set_index=i)
        assert r.status_code == 201

    body = r.json()
    assert body["sets_completed"] == 3
    assert body["completed"] is True
    assert [s["set_index"] for s in body["sets"]] == [1, 2, 3]

    with SessionLocal() as db:
        assert db.query(ExerciseSession).count() == 1, "three sets, one sitting"
        assert db.query(ExerciseSet).count() == 3


def test_stopping_early_keeps_the_sets_already_done(client, token):
    post_set(client, token, set_index=1)
    post_set(client, token, set_index=2)

    sessions = client.get("/sessions", headers=bearer(token)).json()["sessions"]
    assert sessions[0]["sets_completed"] == 2
    assert sessions[0]["completed"] is False


def test_reposting_the_same_set_does_not_double_count_it(client, token):
    post_set(client, token, set_index=1)
    r = post_set(client, token, set_index=1)   # the browser retried

    assert r.status_code == 201
    assert r.json()["sets_completed"] == 1

    with SessionLocal() as db:
        assert db.query(ExerciseSet).count() == 1


def test_a_new_sitting_starts_a_new_session(client, token):
    post_set(client, token, client_session_id="session-aaaaaaaaaaaa")
    post_set(client, token, client_session_id="session-bbbbbbbbbbbb")

    assert client.get("/sessions", headers=bearer(token)).json()["count"] == 2


def test_breaches_are_recorded_as_count_and_duration(client, token):
    # One long breach and six brief ones mean different things clinically, so
    # both numbers are kept.
    r = post_set(client, token, breach_count=6, breach_seconds=4.2, peak_flexion_deg=71.0)
    row = r.json()["sets"][0]
    assert row["breach_count"] == 6
    assert row["breach_seconds"] == pytest.approx(4.2)
    assert row["peak_flexion_deg"] == pytest.approx(71.0)


def test_a_hold_records_what_was_actually_held(client, token):
    r = post_set(
        client, token,
        exercise_name="Quad Sets", target_reps=None, reps_completed=0,
        hold_seconds=10, hold_seconds_achieved=10.0, peak_flexion_deg=3.2,
    )
    assert r.json()["sets"][0]["hold_seconds_achieved"] == pytest.approx(10.0)
    assert r.json()["hold_seconds"] == 10


# ---------------------------------------------------------------------------
# Measurement quality
# ---------------------------------------------------------------------------

def test_an_unverified_camera_view_marks_the_whole_session(client, token):
    """
    A knee angle measured square-on to the camera reads far too low. Charting
    that alongside properly measured sessions would show a recovery that never
    happened, so the flag has to survive onto the session.
    """
    post_set(client, token, set_index=1, view_verified=True)
    r = post_set(client, token, set_index=2, view_verified=False)
    assert r.json()["view_verified"] is False

    # And it must not be cleared again by a later good set.
    r = post_set(client, token, set_index=3, view_verified=True)
    assert r.json()["view_verified"] is False


def test_time_without_a_usable_measurement_is_kept(client, token):
    r = post_set(client, token, suspended_seconds=12.5, mean_visibility=0.61)
    row = r.json()["sets"][0]
    assert row["suspended_seconds"] == pytest.approx(12.5)
    assert row["mean_visibility"] == pytest.approx(0.61)


@pytest.mark.parametrize("field,value", [
    ("peak_flexion_deg", 400.0),     # no knee bends this far
    ("peak_flexion_deg", -5.0),
    ("mean_visibility", 1.5),        # it is a probability
    ("reps_completed", -1),
    ("set_index", 0),                # 1-based
    ("angle_limit", 900),
    ("breach_seconds", -2.0),
])
def test_impossible_measurements_are_refused(client, token, field, value):
    assert post_set(client, token, **{field: value}).status_code == 422


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------

def test_recording_requires_an_account(client):
    assert client.post("/sessions/sets", json=set_payload()).status_code == 401
    assert client.get("/sessions").status_code == 401


def test_one_patient_cannot_see_another_patients_sessions(client, token):
    other = client.post(
        "/auth/register", json={"email": "bob@example.com", "password": PASSWORD}
    ).json()["access_token"]

    post_set(client, token)

    assert client.get("/sessions", headers=bearer(token)).json()["count"] == 1
    assert client.get("/sessions", headers=bearer(other)).json()["count"] == 0


def test_a_session_links_to_the_analysis_it_was_performed_under(client, token):
    analysis = client.post(
        "/analyse-xray",
        files={"image": ("knee.png", png_bytes(), "image/png")},
        data={"knee_side": "left", "surgery_type": "tkr", "weeks_post_op": "3"},
        headers=bearer(token),
    ).json()

    r = post_set(client, token, prescription_id=analysis["prescription_id"])
    assert r.json()["prescription_id"] == analysis["prescription_id"]


def test_a_session_cannot_be_filed_against_someone_elses_analysis(client, token):
    other = client.post(
        "/auth/register", json={"email": "bob@example.com", "password": PASSWORD}
    ).json()["access_token"]
    theirs = client.post(
        "/analyse-xray",
        files={"image": ("knee.png", png_bytes(), "image/png")},
        data={"knee_side": "left", "surgery_type": "tkr", "weeks_post_op": "3"},
        headers=bearer(other),
    ).json()["prescription_id"]

    # Recorded, but unlinked: losing the session would be worse than losing the
    # link, and silently attaching it to another patient's record is worst.
    r = post_set(client, token, prescription_id=theirs)
    assert r.status_code == 201
    assert r.json()["prescription_id"] is None


def test_deleting_a_patient_takes_their_sessions_with_them(client, token):
    post_set(client, token)
    from models import Patient

    with SessionLocal() as db:
        assert db.query(ExerciseSet).count() == 1
        db.delete(db.query(Patient).filter_by(email="ada@example.com").one())
        db.commit()
        assert db.query(ExerciseSession).count() == 0
        assert db.query(ExerciseSet).count() == 0


def test_sessions_are_newest_first(client, token):
    for i, sid in enumerate(["session-aaaaaaaaaaaa", "session-bbbbbbbbbbbb", "session-cccccccccccc"]):
        post_set(client, token, client_session_id=sid, exercise_name=f"Exercise {i}")

    rows = client.get("/sessions", headers=bearer(token)).json()["sessions"]
    started = [r["started_at"] for r in rows]
    assert started == sorted(started, reverse=True)
