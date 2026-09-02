"""
Clinician accounts, invite codes and the caseload.

The direction of the invite is the design: a clinician issues a code, the patient
redeems it. Redeeming is the patient's own act, and that act is the consent —
nobody can attach themselves to someone's medical record without the patient
doing something first.

What these pin down:

  * A clinician's token cannot open a patient's endpoints, or the reverse.
  * A code works once, expires, and gives nothing away when it fails.
  * Either side can end the link, and the record says which of them did.
  * A clinician sees exactly the patients on their caseload and no others.

Run:  python -m pytest backend/tests -q
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="clinician tests need sqlalchemy")

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier

import main
from db import SessionLocal
from models import CareLink, ExerciseSession, ExerciseSet, Patient

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
def patient(client):
    return client.post("/auth/register", json={
        "email": "ada@example.com", "password": PASSWORD, "display_name": "Ada L",
    }).json()["access_token"]


@pytest.fixture
def clinician(client):
    return client.post("/clinician/register", json={
        "email": "okonkwo@clinic.example", "password": PASSWORD,
        "display_name": "N. Okonkwo", "registration": "HCPC PH123456",
    }).json()["access_token"]


def invite(client, clin, **body):
    return client.post("/clinician/invites", json=body or {}, headers=bearer(clin))


def link_up(client, clin, pat, **body):
    """The whole handshake: clinician issues, patient redeems."""
    code = invite(client, clin, **body).json()["code"]
    return client.post("/me/clinicians/redeem", json={"code": code}, headers=bearer(pat))


def seed_session(email="ada@example.com", *, days_ago=0, peak=52.0,
                 verified=True, pain_after=3, breaches=0):
    started = datetime.now(timezone.utc) - timedelta(days=days_ago)
    with SessionLocal() as db:
        p = db.query(Patient).filter_by(email=email).one()
        session = ExerciseSession(
            patient_id=p.id, client_session_id=f"seed-{email}-{days_ago}-{peak}",
            exercise_name="Mini Squat", tracked_joint="knee", knee_side="left",
            angle_limit=60, target_reps=10, target_sets=1,
            started_at=started, last_set_at=started, sets_completed=1,
            completed=True, view_verified=verified, pain_after=pain_after,
        )
        db.add(session)
        db.flush()
        db.add(ExerciseSet(session_id=session.id, set_index=1, reps_completed=10,
                           duration_seconds=40.0, peak_flexion_deg=peak,
                           breach_count=breaches, mean_visibility=0.9))
        db.commit()


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

def test_a_clinician_can_register_and_sign_in(client):
    r = client.post("/clinician/register", json={
        "email": "okonkwo@clinic.example", "password": PASSWORD,
        "display_name": "N. Okonkwo", "registration": "HCPC PH123456",
    })
    assert r.status_code == 201
    me = client.get("/clinician/me", headers=bearer(r.json()["access_token"]))
    assert me.json()["registration"] == "HCPC PH123456"

    again = client.post("/clinician/login",
                        json={"email": "OKONKWO@clinic.example", "password": PASSWORD})
    assert again.status_code == 200


def test_a_clinician_and_a_patient_may_share_an_email(client):
    """
    A physiotherapist recovering from their own knee surgery is a real person,
    and separate tables mean nothing has to be decided about them.
    """
    assert client.post("/auth/register", json={
        "email": "same@example.com", "password": PASSWORD}).status_code == 201
    assert client.post("/clinician/register", json={
        "email": "same@example.com", "password": PASSWORD}).status_code == 201


def test_wrong_clinician_credentials_give_nothing_away(client, clinician):
    wrong = client.post("/clinician/login",
                        json={"email": "okonkwo@clinic.example", "password": "nope"})
    unknown = client.post("/clinician/login",
                          json={"email": "ghost@clinic.example", "password": "nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


# ---------------------------------------------------------------------------
# The two roles do not overlap
# ---------------------------------------------------------------------------

def test_a_clinician_token_cannot_open_a_patients_endpoints(client, clinician):
    for path in ("/auth/me", "/me/prescriptions", "/sessions", "/me/progress"):
        assert client.get(path, headers=bearer(clinician)).status_code == 401, path


def test_a_patient_token_cannot_open_a_clinicians_endpoints(client, patient):
    for path in ("/clinician/me", "/clinician/patients", "/clinician/invites"):
        assert client.get(path, headers=bearer(patient)).status_code == 401, path


def test_a_patient_cannot_issue_invites(client, patient):
    assert client.post("/clinician/invites", json={}, headers=bearer(patient)).status_code == 401


# ---------------------------------------------------------------------------
# Invites
# ---------------------------------------------------------------------------

def test_the_code_is_returned_once_and_only_its_hash_kept(client, clinician):
    created = invite(client, clinician, patient_label="Ada, left TKR").json()
    assert created["code"]
    assert created["invite_hint"] == created["code"][:3]

    listed = client.get("/clinician/invites", headers=bearer(clinician)).json()
    assert "code" not in listed[0]
    assert listed[0]["patient_label"] == "Ada, left TKR"

    with SessionLocal() as db:
        stored = db.query(CareLink).one().invite_code_hash
    assert created["code"] not in stored
    assert stored == main._hash_invite_code(created["code"])


def test_the_code_survives_being_retyped_carelessly(client, clinician, patient):
    """People retype codes with the wrong case, extra spaces, or no dashes."""
    code = invite(client, clinician).json()["code"]
    mangled = f"  {code.replace('-', '').lower()}  "
    assert client.post("/me/clinicians/redeem", json={"code": mangled},
                       headers=bearer(patient)).status_code == 201


def test_redeeming_puts_the_patient_on_the_caseload(client, clinician, patient):
    r = link_up(client, clinician, patient)
    assert r.status_code == 201
    assert r.json()["is_active"] is True
    assert r.json()["clinician_name"] == "N. Okonkwo"

    caseload = client.get("/clinician/patients", headers=bearer(clinician)).json()
    assert caseload["count"] == 1
    assert caseload["patients"][0]["patient_name"] == "Ada L"


def test_a_code_only_works_once(client, clinician, patient):
    code = invite(client, clinician).json()["code"]
    other = client.post("/auth/register", json={
        "email": "bob@example.com", "password": PASSWORD}).json()["access_token"]

    assert client.post("/me/clinicians/redeem", json={"code": code},
                       headers=bearer(patient)).status_code == 201
    assert client.post("/me/clinicians/redeem", json={"code": code},
                       headers=bearer(other)).status_code == 404


def test_an_expired_code_does_not_work(client, clinician, patient):
    code = invite(client, clinician, days=1).json()["code"]
    with SessionLocal() as db:
        link = db.query(CareLink).one()
        link.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()

    assert client.post("/me/clinicians/redeem", json={"code": code},
                       headers=bearer(patient)).status_code == 404


def test_every_bad_code_gives_the_same_answer(client, clinician, patient):
    """Anything more specific turns this into an oracle for guessing codes."""
    used = invite(client, clinician).json()["code"]
    client.post("/me/clinicians/redeem", json={"code": used}, headers=bearer(patient))

    expired = invite(client, clinician, days=1).json()["code"]
    with SessionLocal() as db:
        link = db.query(CareLink).filter_by(invite_code_hash=main._hash_invite_code(expired)).one()
        link.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()

    other = client.post("/auth/register", json={
        "email": "bob@example.com", "password": PASSWORD}).json()["access_token"]

    answers = [
        client.post("/me/clinicians/redeem", json={"code": used}, headers=bearer(other)),
        client.post("/me/clinicians/redeem", json={"code": expired}, headers=bearer(other)),
        client.post("/me/clinicians/redeem", json={"code": "ZZZZZ-ZZZZZ"}, headers=bearer(other)),
    ]
    assert {a.status_code for a in answers} == {404}
    assert len({a.json()["detail"] for a in answers}) == 1


def test_the_same_clinician_cannot_be_linked_twice(client, clinician, patient):
    link_up(client, clinician, patient)
    second = link_up(client, clinician, patient)
    assert second.status_code == 409


def test_redeeming_requires_a_patient_account(client, clinician):
    code = invite(client, clinician).json()["code"]
    assert client.post("/me/clinicians/redeem", json={"code": code}).status_code == 401


# ---------------------------------------------------------------------------
# Ending it
# ---------------------------------------------------------------------------

def test_a_patient_can_withdraw_access(client, clinician, patient):
    seed_session()
    link = link_up(client, clinician, patient).json()
    pid = client.get("/clinician/patients", headers=bearer(clinician)).json()["patients"][0]["patient_id"]

    r = client.delete(f"/me/clinicians/{link['id']}", headers=bearer(patient))
    assert r.status_code == 200
    assert r.json()["is_active"] is False
    assert r.json()["revoked_by"] == "patient"

    assert client.get("/clinician/patients", headers=bearer(clinician)).json()["count"] == 0
    assert client.get(f"/clinician/patients/{pid}/progress",
                      headers=bearer(clinician)).status_code == 404


def test_a_clinician_can_discharge_a_patient(client, clinician, patient):
    link_up(client, clinician, patient)
    link_id = client.get("/clinician/patients", headers=bearer(clinician)).json()["patients"][0]["link_id"]

    r = client.delete(f"/clinician/patients/{link_id}", headers=bearer(clinician))
    assert r.json()["revoked_by"] == "clinician"
    assert client.get("/me/clinicians", headers=bearer(patient)).json()[0]["is_active"] is False


def test_who_ended_it_is_recorded(client, clinician, patient):
    """
    "My physiotherapist discharged me" and "I withdrew access" are different
    events, and a record that cannot tell them apart says something false.
    """
    link = link_up(client, clinician, patient).json()
    client.delete(f"/me/clinicians/{link['id']}", headers=bearer(patient))
    assert client.get("/me/clinicians", headers=bearer(patient)).json()[0]["revoked_by"] == "patient"


def test_nobody_can_end_a_link_that_is_not_theirs(client, clinician, patient):
    link = link_up(client, clinician, patient).json()
    stranger = client.post("/clinician/register", json={
        "email": "other@clinic.example", "password": PASSWORD}).json()["access_token"]

    assert client.delete(f"/clinician/patients/{link['id']}",
                         headers=bearer(stranger)).status_code == 404
    assert client.get("/me/clinicians", headers=bearer(patient)).json()[0]["is_active"] is True


def test_a_withdrawn_link_can_be_re_established(client, clinician, patient):
    """Discharge and re-refer is ordinary, so it must not be a dead end."""
    link = link_up(client, clinician, patient).json()
    client.delete(f"/me/clinicians/{link['id']}", headers=bearer(patient))
    assert link_up(client, clinician, patient).status_code == 201
    assert client.get("/clinician/patients", headers=bearer(clinician)).json()["count"] == 1


# ---------------------------------------------------------------------------
# The caseload
# ---------------------------------------------------------------------------

def test_the_caseload_carries_what_a_clinician_triages_by(client, clinician, patient):
    client.patch("/me/surgery",
                 json={"surgery_date": (datetime.now(timezone.utc) - timedelta(days=35)).date().isoformat(),
                       "surgery_type": "tkr"},
                 headers=bearer(patient))
    seed_session(days_ago=2, peak=48.0, pain_after=5, breaches=2)
    seed_session(days_ago=0, peak=54.0, pain_after=3, breaches=1)
    link_up(client, clinician, patient)

    entry = client.get("/clinician/patients", headers=bearer(clinician)).json()["patients"][0]
    assert entry["weeks_post_op"] == 5
    assert entry["surgery_type"] == "tkr"
    assert entry["sessions_last_7_days"] == 2
    assert entry["latest_flexion_deg"] == 54.0
    assert entry["latest_pain_after"] == 3
    assert entry["breaches_last_7_days"] == 3


def test_an_unverified_camera_angle_is_left_off_the_caseload(client, clinician, patient):
    """
    A clinician scanning the flexion column would otherwise read a collapse in
    range of motion that never happened.
    """
    seed_session(days_ago=1, peak=54.0, verified=True)
    seed_session(days_ago=0, peak=8.0, verified=False)
    link_up(client, clinician, patient)

    entry = client.get("/clinician/patients", headers=bearer(clinician)).json()["patients"][0]
    assert entry["latest_flexion_deg"] == 54.0
    # The session still counts as work done.
    assert entry["sessions_last_7_days"] == 2


def test_a_patient_with_no_sessions_still_appears(client, clinician, patient):
    link_up(client, clinician, patient)
    entry = client.get("/clinician/patients", headers=bearer(clinician)).json()["patients"][0]
    assert entry["last_session_at"] is None
    assert entry["sessions_last_7_days"] == 0
    assert entry["latest_flexion_deg"] is None


def test_a_clinician_sees_only_their_own_patients(client, clinician, patient):
    other_clinician = client.post("/clinician/register", json={
        "email": "other@clinic.example", "password": PASSWORD}).json()["access_token"]
    seed_session()
    link_up(client, clinician, patient)

    assert client.get("/clinician/patients", headers=bearer(other_clinician)).json()["count"] == 0
    pid = client.get("/clinician/patients", headers=bearer(clinician)).json()["patients"][0]["patient_id"]
    assert client.get(f"/clinician/patients/{pid}/progress",
                      headers=bearer(other_clinician)).status_code == 404


def test_a_linked_clinician_sees_the_full_progress_view(client, clinician, patient):
    seed_session(days_ago=1, peak=50.0)
    seed_session(days_ago=0, peak=55.0)
    link_up(client, clinician, patient)

    pid = client.get("/clinician/patients", headers=bearer(clinician)).json()["patients"][0]["patient_id"]
    progress = client.get(f"/clinician/patients/{pid}/progress", headers=bearer(clinician)).json()

    assert progress["summary"]["sessions"] == 2
    assert progress["summary"]["best_flexion_deg"] == 55.0
    assert progress["rom_by_exercise"][0]["exercise_name"] == "Mini Squat"


def test_deleting_a_patient_removes_them_from_every_caseload(client, clinician, patient):
    link_up(client, clinician, patient)

    with SessionLocal() as db:
        db.delete(db.query(Patient).filter_by(email="ada@example.com").one())
        db.commit()

    assert client.get("/clinician/patients", headers=bearer(clinician)).json()["count"] == 0
