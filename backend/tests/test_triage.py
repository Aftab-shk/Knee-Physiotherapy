"""
Red-flag triage.

These are the rules that decide when a human should look. Two failure modes
matter, and they pull against each other:

  * Missing a patient whose knee is going backwards.
  * Firing so often that the flags get ignored — which is the same as missing
    them, only with more noise.

So most of what follows is about the rules staying *quiet*: one sore session is
not a pattern, a patient who never started is not deteriorating, and an angle
measured from an unverified camera view is not evidence of anything.

The rules are exercised directly, on plain objects, because they are ordinary
functions over data and deserve to be tested as such. The API tests below only
check the wiring.

Run:  python -m pytest backend/tests -q
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triage

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


# ── Plain stand-ins for the ORM rows ────────────────────────────────────────

@dataclass
class FakeSet:
    peak_flexion_deg: float = 50.0
    breach_count: int = 0
    breach_seconds: float = 0.0


@dataclass
class FakeSession:
    days_ago: int = 0
    view_verified: bool = True
    pain_before: int | None = None
    pain_after: int | None = None
    sets: list = field(default_factory=lambda: [FakeSet()])

    @property
    def started_at(self):
        return NOW - timedelta(days=self.days_ago)


@dataclass
class FakePrescription:
    days_ago: int = 0
    status: str = "draft"

    @property
    def created_at(self):
        return NOW - timedelta(days=self.days_ago)


def codes(flags):
    return {f.code for f in flags}


def find(flags, code):
    return next(f for f in flags if f.code == code)


def evaluate(sessions=(), prescriptions=()):
    return triage.evaluate(list(sessions), list(prescriptions), now=NOW)


def steady(n=6, start_day=25, peak=50.0, **kw):
    """A run of ordinary sessions, oldest first."""
    return [FakeSession(days_ago=start_day - i * 3, sets=[FakeSet(peak_flexion_deg=peak)], **kw)
            for i in range(n)]


# ---------------------------------------------------------------------------
# Quiet by default
# ---------------------------------------------------------------------------

def test_nothing_fires_for_a_patient_who_is_doing_well():
    sessions = [
        FakeSession(days_ago=d, pain_before=4, pain_after=3,
                    sets=[FakeSet(peak_flexion_deg=45 + i * 2)])
        for i, d in enumerate([20, 16, 12, 8, 4, 1])
    ]
    assert evaluate(sessions) == []


def test_no_history_raises_nothing():
    assert evaluate([]) == []


def test_a_patient_who_never_started_is_not_deteriorating():
    """
    Flagging every dormant account would bury the patients who were going and
    stopped, which is the one this rule exists to catch.
    """
    assert codes(evaluate([FakeSession(days_ago=20)])) == set()


# ---------------------------------------------------------------------------
# Pain
# ---------------------------------------------------------------------------

def test_a_single_severe_pain_score_is_urgent():
    flags = evaluate([FakeSession(days_ago=2, pain_after=triage.PAIN_SEVERE)])
    flag = find(flags, "pain_severe")
    assert flag.severity == triage.URGENT
    assert flag.patient_message
    assert flag.evidence["max_pain_after"] == triage.PAIN_SEVERE


def test_severe_pain_long_ago_is_not_current():
    assert "pain_severe" not in codes(evaluate([FakeSession(days_ago=20, pain_after=9)]))


def test_sessions_that_consistently_end_worse_are_flagged():
    sessions = [FakeSession(days_ago=d, pain_before=3, pain_after=6) for d in (9, 6, 3)]
    flag = find(evaluate(sessions), "pain_rising")
    assert flag.severity == triage.WARNING
    assert flag.evidence["mean_change"] == 3.0


def test_one_bad_session_is_not_a_pattern():
    sessions = [
        FakeSession(days_ago=9, pain_before=4, pain_after=3),
        FakeSession(days_ago=6, pain_before=4, pain_after=3),
        FakeSession(days_ago=3, pain_before=3, pain_after=7),   # one hard day
    ]
    assert "pain_rising" not in codes(evaluate(sessions))


def test_pain_that_settles_during_rehab_is_normal():
    """Hurting a little during and settling afterwards is what working looks like."""
    sessions = [FakeSession(days_ago=d, pain_before=6, pain_after=4) for d in (9, 6, 3)]
    assert "pain_rising" not in codes(evaluate(sessions))


def test_a_half_reported_session_cannot_trigger_the_pattern():
    sessions = [
        FakeSession(days_ago=9, pain_before=3, pain_after=7),
        FakeSession(days_ago=6, pain_before=3),                 # no "after"
        FakeSession(days_ago=3, pain_after=7),                  # no "before"
    ]
    assert "pain_rising" not in codes(evaluate(sessions))


# ---------------------------------------------------------------------------
# Range of motion
# ---------------------------------------------------------------------------

def test_flexion_going_backwards_is_urgent():
    sessions = steady(n=4, start_day=25, peak=70.0)
    sessions.append(FakeSession(days_ago=2, sets=[FakeSet(peak_flexion_deg=55.0)]))

    flag = find(evaluate(sessions), "rom_regression")
    assert flag.severity == triage.URGENT
    assert flag.evidence["drop_deg"] == 15.0


def test_a_small_dip_is_measurement_noise_not_regression():
    sessions = steady(n=4, start_day=25, peak=70.0)
    sessions.append(FakeSession(days_ago=2, sets=[FakeSet(peak_flexion_deg=64.0)]))
    assert "rom_regression" not in codes(evaluate(sessions))


def test_regression_needs_a_baseline_worth_comparing_against():
    """Two earlier sessions is not a baseline; it is two numbers."""
    sessions = [
        FakeSession(days_ago=20, sets=[FakeSet(peak_flexion_deg=70.0)]),
        FakeSession(days_ago=2, sets=[FakeSet(peak_flexion_deg=40.0)]),
    ]
    assert "rom_regression" not in codes(evaluate(sessions))


def test_an_unverified_camera_view_cannot_raise_a_regression():
    """
    Square-on to the camera a bent knee reads near zero. A flag saying "range of
    motion is collapsing" on the strength of a badly-placed webcam would train
    clinicians to ignore the ones that matter.
    """
    sessions = steady(n=4, start_day=25, peak=70.0)
    sessions.append(FakeSession(days_ago=2, view_verified=False,
                                sets=[FakeSet(peak_flexion_deg=5.0)]))
    assert "rom_regression" not in codes(evaluate(sessions))


def test_an_unverified_baseline_is_not_a_baseline_either():
    sessions = steady(n=4, start_day=25, peak=70.0, view_verified=False)
    sessions.append(FakeSession(days_ago=2, sets=[FakeSet(peak_flexion_deg=40.0)]))
    assert "rom_regression" not in codes(evaluate(sessions))


# ---------------------------------------------------------------------------
# Breaches
# ---------------------------------------------------------------------------

def test_repeatedly_going_past_the_limit_is_flagged():
    sessions = [FakeSession(days_ago=3, sets=[FakeSet(breach_count=3), FakeSet(breach_count=3)])]
    flag = find(evaluate(sessions), "repeated_breaches")
    assert flag.severity == triage.WARNING
    assert flag.evidence["breach_count"] == 6


def test_a_long_single_breach_counts_too():
    sessions = [FakeSession(days_ago=3, sets=[FakeSet(breach_count=1, breach_seconds=45.0)])]
    assert "repeated_breaches" in codes(evaluate(sessions))


def test_the_odd_overshoot_is_not_flagged():
    sessions = [FakeSession(days_ago=3, sets=[FakeSet(breach_count=2, breach_seconds=1.5)])]
    assert "repeated_breaches" not in codes(evaluate(sessions))


def test_old_breaches_do_not_keep_firing():
    sessions = [FakeSession(days_ago=20, sets=[FakeSet(breach_count=20, breach_seconds=90.0)])]
    assert "repeated_breaches" not in codes(evaluate(sessions))


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------

def test_someone_who_was_going_and_stopped_is_flagged():
    flags = evaluate(steady(n=5, start_day=25))
    flag = find(flags, "stopped_exercising")
    assert flag.evidence["prior_sessions"] == 5
    assert flag.evidence["days_since_last"] >= triage.LAPSE_DAYS


def test_a_current_patient_is_not_flagged_as_stopped():
    sessions = [*steady(n=5, start_day=25), FakeSession(days_ago=1)]
    assert "stopped_exercising" not in codes(evaluate(sessions))


def test_one_or_two_early_sessions_do_not_make_a_lapse():
    assert "stopped_exercising" not in codes(evaluate([FakeSession(days_ago=20)] * 2))


# ---------------------------------------------------------------------------
# Unreviewed drafts
# ---------------------------------------------------------------------------

def test_a_draft_nobody_has_read_is_raised_for_the_clinician():
    flags = evaluate([], [FakePrescription(days_ago=5, status="draft")])
    flag = find(flags, "unreviewed_prescription")
    assert flag.severity == triage.INFO
    # No patient message: telling someone their physiotherapist has not read
    # their notes is alarming and not theirs to act on.
    assert flag.patient_message is None


def test_a_fresh_draft_is_not_yet_overdue():
    assert "unreviewed_prescription" not in codes(evaluate([], [FakePrescription(days_ago=1)]))


def test_a_reviewed_prescription_is_not_outstanding():
    assert "unreviewed_prescription" not in codes(
        evaluate([], [FakePrescription(days_ago=30, status="clinician_approved")]))


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_the_worst_thing_comes_first():
    sessions = steady(n=4, start_day=25, peak=70.0)
    sessions.append(FakeSession(days_ago=2, pain_after=9,
                                sets=[FakeSet(peak_flexion_deg=50.0, breach_count=9)]))

    flags = evaluate(sessions, [FakePrescription(days_ago=10)])
    assert len(flags) >= 3
    assert flags[0].severity == triage.URGENT
    assert flags[-1].severity == triage.INFO
    assert triage.worst_severity(flags) == triage.URGENT


def test_worst_severity_of_nothing_is_nothing():
    assert triage.worst_severity([]) is None


def test_every_flag_carries_the_numbers_it_was_raised_on():
    """A clinician has to be able to disagree with a flag at a glance."""
    sessions = steady(n=4, start_day=25, peak=70.0)
    sessions.append(FakeSession(days_ago=2, pain_before=2, pain_after=9,
                                sets=[FakeSet(peak_flexion_deg=45.0, breach_count=9)]))
    for flag in evaluate(sessions, [FakePrescription(days_ago=10)]):
        assert flag.summary, flag.code
        assert flag.evidence, flag.code


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------
# The rules are covered above. These only check that both audiences reach them,
# and that each gets the version written for them.

import pytest

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="wiring tests need sqlalchemy")

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


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture
def linked(client):
    pat = client.post("/auth/register", json={
        "email": "ada@example.com", "password": PASSWORD, "display_name": "Ada L",
    }).json()["access_token"]
    clin = client.post("/clinician/register", json={
        "email": "okonkwo@clinic.example", "password": PASSWORD, "display_name": "N. Okonkwo",
    }).json()["access_token"]
    code = client.post("/clinician/invites", json={}, headers=bearer(clin)).json()["code"]
    client.post("/me/clinicians/redeem", json={"code": code}, headers=bearer(pat))
    return pat, clin


def seed(days_ago, *, peak=50.0, verified=True, pain_after=None, breaches=0):
    started = datetime.now(timezone.utc) - timedelta(days=days_ago)
    with SessionLocal() as db:
        p = db.query(Patient).filter_by(email="ada@example.com").one()
        s = ExerciseSession(
            patient_id=p.id, client_session_id=f"seed-{days_ago}-{peak}-{breaches}",
            exercise_name="Mini Squat", tracked_joint="knee", knee_side="left",
            angle_limit=60, target_reps=10, target_sets=1,
            started_at=started, last_set_at=started, sets_completed=1,
            completed=True, view_verified=verified, pain_after=pain_after,
        )
        db.add(s)
        db.flush()
        db.add(ExerciseSet(session_id=s.id, set_index=1, reps_completed=10,
                           duration_seconds=40.0, peak_flexion_deg=peak,
                           breach_count=breaches, mean_visibility=0.9))
        db.commit()


def test_a_clean_history_shows_no_flags_to_either_side(client, linked):
    pat, clin = linked
    for d in (12, 8, 4, 1):
        seed(d, peak=50.0, pain_after=3)

    assert client.get("/me/flags", headers=bearer(pat)).json()["flags"] == []
    pid = client.get("/clinician/patients", headers=bearer(clin)).json()["patients"][0]["patient_id"]
    assert client.get(f"/clinician/patients/{pid}/flags", headers=bearer(clin)).json()["flags"] == []


def test_both_sides_see_the_same_finding_worded_differently(client, linked):
    pat, clin = linked
    seed(2, pain_after=9)

    mine = client.get("/me/flags", headers=bearer(pat)).json()
    assert mine["worst"] == "urgent"
    assert mine["flags"][0]["code"] == "pain_severe"
    # Second person, and it tells them what to do about it.
    assert "you" in mine["flags"][0]["message"].lower()

    pid = client.get("/clinician/patients", headers=bearer(clin)).json()["patients"][0]["patient_id"]
    theirs = client.get(f"/clinician/patients/{pid}/flags", headers=bearer(clin)).json()
    assert theirs["flags"][0]["code"] == "pain_severe"
    assert theirs["flags"][0]["evidence"]["max_pain_after"] == 9


def test_the_caseload_carries_flags_so_it_can_be_sorted_by_them(client, linked):
    _, clin = linked
    seed(2, pain_after=9)

    entry = client.get("/clinician/patients", headers=bearer(clin)).json()["patients"][0]
    assert entry["worst_flag"] == "urgent"
    assert any(f["code"] == "pain_severe" for f in entry["flags"])


def test_a_workflow_flag_is_not_shown_to_the_patient(client, linked):
    """
    An unread prescription is the clinician's business. Telling a patient their
    physiotherapist has not looked at their notes is alarming and not theirs to
    act on.
    """
    pat, clin = linked
    from test_api_security import png_bytes
    client.post("/analyse-xray",
                files={"image": ("knee.png", png_bytes(), "image/png")},
                data={"knee_side": "left", "surgery_type": "tkr", "weeks_post_op": "3"},
                headers=bearer(pat))

    with SessionLocal() as db:
        from models import Prescription
        row = db.query(Prescription).one()
        row.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        db.commit()

    pid = client.get("/clinician/patients", headers=bearer(clin)).json()["patients"][0]["patient_id"]
    theirs = client.get(f"/clinician/patients/{pid}/flags", headers=bearer(clin)).json()
    assert any(f["code"] == "unreviewed_prescription" for f in theirs["flags"])

    mine = client.get("/me/flags", headers=bearer(pat)).json()
    assert all(f["code"] != "unreviewed_prescription" for f in mine["flags"])


def test_flags_need_the_right_account(client, linked):
    pat, clin = linked
    pid = client.get("/clinician/patients", headers=bearer(clin)).json()["patients"][0]["patient_id"]

    assert client.get("/me/flags").status_code == 401
    assert client.get("/me/flags", headers=bearer(clin)).status_code == 401
    assert client.get(f"/clinician/patients/{pid}/flags", headers=bearer(pat)).status_code == 401


def test_a_clinician_cannot_read_flags_for_someone_elses_patient(client, linked):
    _, clin = linked
    stranger = client.post("/clinician/register", json={
        "email": "other@clinic.example", "password": PASSWORD}).json()["access_token"]
    pid = client.get("/clinician/patients", headers=bearer(clin)).json()["patients"][0]["patient_id"]

    assert client.get(f"/clinician/patients/{pid}/flags", headers=bearer(stranger)).status_code == 404
