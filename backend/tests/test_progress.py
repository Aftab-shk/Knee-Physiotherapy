"""
Turning stored sessions into a progress view.

The rule these exist to defend: a session performed with the camera-view gate
overridden counts as work done, but none of its angles are trustworthy. A knee
measured square-on to the camera reads close to straight however far it is
actually bent, so mixing those readings into a range-of-motion trend would show
a patient a recovery that never happened — from a screen they will read as
evidence that what they are doing is working.

So: unverified sessions count for adherence, and are excluded from every angle.

Run:  python -m pytest backend/tests -q
"""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="progress tests need sqlalchemy")

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


def seed_session(
    email="ada@example.com",
    *,
    days_ago=0,
    exercise="Mini Squat",
    peaks=(55.0,),
    angle_limit=60,
    verified=True,
    completed=True,
    reps=10,
    breach_count=0,
    breach_seconds=0.0,
    visibility=0.9,
    at_hour=12,
):
    """Write a session straight to the database, dated relative to now."""
    started = datetime.now(timezone.utc).replace(hour=at_hour, minute=0, second=0, microsecond=0)
    started -= timedelta(days=days_ago)

    with SessionLocal() as db:
        patient = db.query(Patient).filter_by(email=email).one()
        session = ExerciseSession(
            patient_id        = patient.id,
            client_session_id = f"seed-{email}-{days_ago}-{exercise}-{at_hour}-{len(peaks)}",
            exercise_name     = exercise,
            tracked_joint     = "knee",
            knee_side         = "left",
            angle_limit       = angle_limit,
            target_reps       = 10,
            target_sets       = len(peaks),
            started_at        = started,
            last_set_at       = started,
            sets_completed    = len(peaks),
            completed         = completed,
            view_verified     = verified,
        )
        db.add(session)
        db.flush()
        for i, peak in enumerate(peaks, start=1):
            db.add(ExerciseSet(
                session_id       = session.id,
                set_index        = i,
                reps_completed   = reps,
                duration_seconds = 40.0,
                peak_flexion_deg = peak,
                breach_count     = breach_count,
                breach_seconds   = breach_seconds,
                mean_visibility  = visibility,
            ))
        db.commit()


def progress(client, tok, **params):
    return client.get("/me/progress", params=params, headers=bearer(tok)).json()


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def test_a_patient_with_no_sessions_gets_an_empty_but_valid_view(client, token):
    body = progress(client, token)
    assert body["summary"]["sessions"] == 0
    assert body["summary"]["current_streak_days"] == 0
    # Null, not zero: "no measurement yet" and "measured 0°" are different, and
    # a chart starting at zero would imply the second.
    assert body["summary"]["best_flexion_deg"] is None
    assert body["rom_by_exercise"] == []
    assert body["by_exercise"] == []
    assert body["summary"]["primary_exercise"] is None


def test_progress_requires_an_account(client):
    assert client.get("/me/progress").status_code == 401


# ---------------------------------------------------------------------------
# Range of motion
# ---------------------------------------------------------------------------

def test_rom_trend_is_the_best_flexion_of_each_day(client, token):
    seed_session(days_ago=2, peaks=(40.0, 44.0, 41.0))
    seed_session(days_ago=1, peaks=(48.0,))
    seed_session(days_ago=0, peaks=(52.0, 55.5))

    body = progress(client, token)
    series = body["rom_by_exercise"][0]
    assert [p["peak_flexion_deg"] for p in series["points"]] == [44.0, 48.0, 55.5]
    assert body["summary"]["best_flexion_deg"] == 55.5
    assert body["summary"]["latest_flexion_deg"] == 55.5
    assert body["summary"]["primary_exercise"] == "Mini Squat"


def test_two_sessions_in_one_day_collapse_to_one_point(client, token):
    seed_session(days_ago=0, at_hour=8, peaks=(41.0,))
    seed_session(days_ago=0, at_hour=19, peaks=(49.0,))

    body = progress(client, token)
    points = body["rom_by_exercise"][0]["points"]
    assert len(points) == 1
    assert points[0]["peak_flexion_deg"] == 49.0
    assert points[0]["sessions"] == 2


def test_the_ceiling_in_force_travels_with_each_point(client, token):
    """
    A clinician raising the limit later must not rewrite what the patient was
    working under at the time, so the ceiling is charted per point.
    """
    seed_session(days_ago=1, peaks=(44.0,), angle_limit=45)
    seed_session(days_ago=0, peaks=(58.0,), angle_limit=60)

    limits = [p["angle_limit"] for p in progress(client, token)["rom_by_exercise"][0]["points"]]
    assert limits == [45, 60]


def test_an_unverified_session_is_left_out_of_every_angle(client, token):
    seed_session(days_ago=1, peaks=(50.0,), verified=True)
    seed_session(days_ago=0, peaks=(120.0,), verified=False)   # square-on: nonsense

    body = progress(client, token)
    assert [p["peak_flexion_deg"] for p in body["rom_by_exercise"][0]["points"]] == [50.0]
    assert body["summary"]["best_flexion_deg"] == 50.0
    assert body["by_exercise"][0]["peak_flexion_deg"] == 50.0


def test_an_unverified_session_still_counts_as_work_done(client, token):
    seed_session(days_ago=0, peaks=(120.0,), verified=False)

    body = progress(client, token)
    assert body["summary"]["sessions"] == 1
    assert body["summary"]["sets"] == 1
    assert body["summary"]["active_days"] == 1
    assert body["summary"]["unverified_sessions"] == 1
    # …but there is nothing measurable to report.
    assert body["summary"]["best_flexion_deg"] is None
    assert body["rom_by_exercise"] == []


# ---------------------------------------------------------------------------
# Adherence
# ---------------------------------------------------------------------------

def test_adherence_counts_sessions_sets_and_completions_per_day(client, token):
    seed_session(days_ago=1, peaks=(40.0, 42.0), completed=True)
    seed_session(days_ago=0, peaks=(44.0,), completed=False)

    days = progress(client, token)["adherence"]
    assert [d["sessions"] for d in days] == [1, 1]
    assert [d["sets"] for d in days] == [2, 1]
    assert [d["completed_sessions"] for d in days] == [1, 0]


def test_a_run_of_days_is_a_streak(client, token):
    for d in (3, 2, 1, 0):
        seed_session(days_ago=d, peaks=(45.0,))

    summary = progress(client, token)["summary"]
    assert summary["current_streak_days"] == 4
    assert summary["longest_streak_days"] == 4
    assert summary["active_days"] == 4


def test_a_gap_breaks_the_streak_but_not_the_record(client, token):
    for d in (9, 8, 7, 6, 5):     # a five-day run, then a gap
        seed_session(days_ago=d, peaks=(45.0,))
    for d in (1, 0):
        seed_session(days_ago=d, peaks=(50.0,))

    summary = progress(client, token)["summary"]
    assert summary["current_streak_days"] == 2
    assert summary["longest_streak_days"] == 5


def test_not_having_exercised_yet_today_does_not_break_the_streak(client, token):
    """
    Zeroing someone's streak at midnight, before they have had a chance to do
    anything, is both wrong and discouraging.
    """
    for d in (3, 2, 1):
        seed_session(days_ago=d, peaks=(45.0,))

    assert progress(client, token)["summary"]["current_streak_days"] == 3


def test_an_old_streak_is_not_current(client, token):
    for d in (9, 8, 7):
        seed_session(days_ago=d, peaks=(45.0,))

    summary = progress(client, token)["summary"]
    assert summary["current_streak_days"] == 0
    assert summary["longest_streak_days"] == 3


# ---------------------------------------------------------------------------
# Per exercise
# ---------------------------------------------------------------------------

def test_exercises_are_broken_down_and_ordered_by_practice(client, token):
    seed_session(days_ago=2, exercise="Mini Squat", peaks=(50.0,), reps=10)
    seed_session(days_ago=1, exercise="Mini Squat", peaks=(54.0,), reps=10)
    seed_session(days_ago=0, exercise="Quad Sets", peaks=(4.0,), reps=0)

    rows = progress(client, token)["by_exercise"]
    assert [r["exercise_name"] for r in rows] == ["Mini Squat", "Quad Sets"]
    assert rows[0]["sessions"] == 2
    assert rows[0]["reps"] == 20
    assert rows[0]["peak_flexion_deg"] == 54.0


def test_breaches_are_totalled_per_exercise(client, token):
    seed_session(days_ago=1, exercise="Mini Squat", peaks=(64.0,), breach_count=2, breach_seconds=1.5)
    seed_session(days_ago=0, exercise="Mini Squat", peaks=(66.0,), breach_count=3, breach_seconds=2.5)

    row = progress(client, token)["by_exercise"][0]
    assert row["breach_count"] == 5
    assert row["breach_seconds"] == pytest.approx(4.0)


def test_tracking_quality_is_reported_alongside_the_numbers(client, token):
    seed_session(days_ago=0, peaks=(50.0,), visibility=0.6)
    row = progress(client, token)["by_exercise"][0]
    assert row["mean_visibility"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Range and time zone
# ---------------------------------------------------------------------------

def test_sessions_outside_the_window_are_not_counted(client, token):
    seed_session(days_ago=40, peaks=(30.0,))
    seed_session(days_ago=2, peaks=(50.0,))

    assert progress(client, token, days=7)["summary"]["sessions"] == 1
    assert progress(client, token, days=90)["summary"]["sessions"] == 2


def test_days_are_bucketed_in_the_patients_own_offset(client, token):
    """
    An evening session in a western time zone lands on the next UTC day. Bucketed
    in UTC it would appear on a day the patient was asleep, splitting a streak
    they never broke.
    """
    # 02:00 UTC — the previous evening at UTC-5.
    seed_session(days_ago=0, at_hour=2, peaks=(50.0,))

    utc_day = progress(client, token, tz_offset_minutes=0)["rom_by_exercise"][0]["points"][0]["date"]
    local_day = progress(client, token, tz_offset_minutes=-300)["rom_by_exercise"][0]["points"][0]["date"]
    assert date.fromisoformat(local_day) == date.fromisoformat(utc_day) - timedelta(days=1)


@pytest.mark.parametrize("params", [
    {"days": 0}, {"days": 400}, {"tz_offset_minutes": 900}, {"tz_offset_minutes": -900},
])
def test_nonsense_parameters_are_refused(client, token, params):
    assert client.get("/me/progress", params=params, headers=bearer(token)).status_code == 422


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_one_patient_never_sees_another_patients_progress(client, token):
    other = client.post(
        "/auth/register", json={"email": "bob@example.com", "password": PASSWORD}
    ).json()["access_token"]

    seed_session(email="ada@example.com", days_ago=0, peaks=(55.0,))

    assert progress(client, token)["summary"]["sessions"] == 1
    assert progress(client, other)["summary"]["sessions"] == 0


def test_a_straight_hold_day_does_not_look_like_lost_range_of_motion(client, token):
    """
    Quad Sets are held with the knee locked straight — about 4° by design, not a
    failed attempt at bending. Pooled with Mini Squats into one "furthest bend
    today" line, a day of quad sets plots as a collapse from 55° to 4°, and a
    patient reads their own chart as a relapse. Each exercise gets its own
    series, so its numbers are only ever compared against itself.
    """
    seed_session(days_ago=2, exercise="Mini Squat", peaks=(52.0,))
    seed_session(days_ago=1, exercise="Mini Squat", peaks=(55.0,))
    seed_session(days_ago=0, exercise="Quad Sets", peaks=(4.2,))

    body = progress(client, token)
    series = {r["exercise_name"]: r for r in body["rom_by_exercise"]}

    assert set(series) == {"Mini Squat", "Quad Sets"}
    # The squat series never dips — the quad-set day is simply not in it.
    squat = [p["peak_flexion_deg"] for p in series["Mini Squat"]["points"]]
    assert squat == [52.0, 55.0]
    assert series["Quad Sets"]["points"][0]["peak_flexion_deg"] == 4.2

    # The headline figures describe the most-measured exercise, and say which.
    assert body["summary"]["primary_exercise"] == "Mini Squat"
    assert body["summary"]["latest_flexion_deg"] == 55.0
    assert body["summary"]["best_flexion_deg"] == 55.0


def test_the_primary_series_is_the_one_with_the_most_data(client, token):
    for d in (4, 3, 2):
        seed_session(days_ago=d, exercise="Quad Sets", peaks=(4.0,))
    seed_session(days_ago=1, exercise="Mini Squat", peaks=(55.0,))

    body = progress(client, token)
    assert [r["exercise_name"] for r in body["rom_by_exercise"]] == ["Quad Sets", "Mini Squat"]
    assert body["summary"]["primary_exercise"] == "Quad Sets"
    assert body["rom_by_exercise"][0]["days_measured"] == 3
