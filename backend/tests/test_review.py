"""
Clinical review: the model drafts, a clinician approves.

Until this existed, the model set a movement ceiling and the patient followed it,
with nobody accountable for the number. The two rules these tests defend:

  1. The clinician's judgement wins. They operated on the knee; the radiograph
     did not. A system that refused to let them raise a limit would be ignored,
     and then the ceiling would be enforced by nothing at all.

  2. It does not get to be invisible. Every value the patient is asked to keep
     their knee under carries a record of where it came from, what it replaced,
     and — when a human loosened it — why. Loosening needs a reason; tightening
     does not, because tightening is not the direction that hurts.

Run:  python -m pytest backend/tests -q
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="review tests need sqlalchemy")

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier, png_bytes

import main
from db import SessionLocal
from models import Prescription, PrescriptionAudit

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
        "email": "okonkwo@clinic.example", "password": PASSWORD, "display_name": "N. Okonkwo",
    }).json()["access_token"]


@pytest.fixture
def linked(client, patient, clinician):
    code = client.post("/clinician/invites", json={}, headers=bearer(clinician)).json()["code"]
    client.post("/me/clinicians/redeem", json={"code": code}, headers=bearer(patient))
    return patient, clinician


@pytest.fixture
def prescription(client, linked):
    """An analysis by the linked patient — a draft awaiting review."""
    pat, _ = linked
    r = client.post(
        "/analyse-xray",
        files={"image": ("knee.png", png_bytes(), "image/png")},
        data={"knee_side": "left", "surgery_type": "tkr", "weeks_post_op": "3"},
        headers=bearer(pat),
    )
    assert r.status_code == 200
    return r.json()


def review(client, clin, pid, **body):
    return client.post(f"/clinician/prescriptions/{pid}/review", json=body, headers=bearer(clin))


def first_exercise(detail_or_payload):
    exercises = detail_or_payload.get("effective", detail_or_payload)["exercise_list"]
    return exercises[0]


# ---------------------------------------------------------------------------
# A new analysis is a draft
# ---------------------------------------------------------------------------

def test_a_fresh_analysis_is_a_draft_nobody_has_seen(client, prescription):
    assert prescription["status"] == "draft"

    with SessionLocal() as db:
        row = db.query(Prescription).one()
        assert row.status == "draft"
        assert row.reviewed_at is None
        assert row.approved_payload is None


def test_the_models_own_ceiling_decision_is_recorded(client, prescription):
    """
    "The machine decided" is a decision. A clinician reading the trail later
    should not have to infer where the original number came from.
    """
    with SessionLocal() as db:
        rows = db.query(PrescriptionAudit).all()
        assert len(rows) == 1
        entry = rows[0]
        assert entry.actor == "model"
        assert entry.field == "ceiling"
        assert entry.previous_value is None
        assert entry.new_value == str(prescription["max_angle"])
        assert "KL grade" in entry.reason


def test_a_clinician_sees_the_draft_awaiting_review(client, linked, prescription):
    _, clin = linked
    pid = prescription["prescription_id"]

    detail = client.get(f"/clinician/prescriptions/{pid}", headers=bearer(clin)).json()
    assert detail["status"] == "draft"
    assert detail["patient_name"] == "Ada L"
    # Both versions travel together; until review they are the same document.
    assert detail["draft"] == detail["effective"]
    assert detail["max_angle"] == detail["model_max_angle"]


# ---------------------------------------------------------------------------
# Approving without changes
# ---------------------------------------------------------------------------

def test_approving_unchanged_puts_a_name_to_the_model_output(client, linked, prescription):
    _, clin = linked
    r = review(client, clin, prescription["prescription_id"], note="Agrees with my assessment.")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "clinician_approved"
    assert body["reviewed_by"] == "N. Okonkwo"
    assert body["review_note"] == "Agrees with my assessment."
    # Nothing changed, so there is nothing to store separately.
    assert body["effective"] == body["draft"]

    with SessionLocal() as db:
        assert db.query(Prescription).one().approved_payload is None


def test_exercises_not_mentioned_are_kept_not_dropped(client, linked, prescription):
    """
    A clinician who edits one exercise out of nine has approved the other eight.
    Treating silence as deletion would quietly empty a prescription.
    """
    _, clin = linked
    draft_names = [e["name"] for e in prescription["exercise_list"]]
    assert len(draft_names) >= 2

    r = review(client, clin, prescription["prescription_id"],
               decisions=[{"name": draft_names[0], "action": "adjust", "target_sets": 4}])

    kept = [e["name"] for e in r.json()["effective"]["exercise_list"]]
    assert set(kept) == set(draft_names)


# ---------------------------------------------------------------------------
# Loosening needs a reason; tightening does not
# ---------------------------------------------------------------------------

def test_raising_the_ceiling_without_a_reason_is_refused(client, linked, prescription):
    _, clin = linked
    ceiling = prescription["max_angle"]

    r = review(client, clin, prescription["prescription_id"], ceiling=ceiling + 30)
    assert r.status_code == 422
    assert "reason" in r.json()["detail"].lower()

    # And nothing was written.
    with SessionLocal() as db:
        assert db.query(Prescription).one().status == "draft"


def test_raising_the_ceiling_with_a_reason_is_allowed_and_recorded(client, linked, prescription):
    """
    The surgeon who replaced the joint knows things the radiograph does not. The
    system must not stand in their way — only insist they say why.
    """
    _, clin = linked
    was = prescription["max_angle"]

    r = review(client, clin, prescription["prescription_id"],
               ceiling=was + 30,
               ceiling_reason="Cemented TKR, intra-operative ROM 0-110°. Radiograph grading does not apply.")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "clinician_modified"
    assert body["max_angle"] == was + 30
    # The model's own number is still there, unrewritten.
    assert body["model_max_angle"] == was

    entry = next(a for a in body["audit"] if a["field"] == "ceiling" and a["actor"] != "model")
    assert entry["previous_value"] == str(was)
    assert entry["new_value"] == str(was + 30)
    assert "intra-operative" in entry["reason"]


def test_lowering_the_ceiling_needs_no_justification(client, linked, prescription):
    _, clin = linked
    r = review(client, clin, prescription["prescription_id"], ceiling=prescription["max_angle"] - 20)
    assert r.status_code == 200
    assert r.json()["max_angle"] == prescription["max_angle"] - 20


def test_raising_one_exercise_without_a_reason_is_refused(client, linked, prescription):
    _, clin = linked
    ex = prescription["exercise_list"][0]

    r = review(client, clin, prescription["prescription_id"],
               decisions=[{"name": ex["name"], "action": "adjust", "angle_limit": ex["angle_limit"] + 25}])
    assert r.status_code == 422
    assert ex["name"] in r.json()["detail"]


def test_raising_one_exercise_with_a_reason_is_allowed(client, linked, prescription):
    _, clin = linked
    ex = prescription["exercise_list"][0]
    higher = ex["angle_limit"] + 25

    r = review(client, clin, prescription["prescription_id"],
               decisions=[{"name": ex["name"], "action": "adjust", "angle_limit": higher,
                           "reason": "Tolerating this well in clinic."}])
    assert r.status_code == 200

    changed = next(e for e in r.json()["effective"]["exercise_list"] if e["name"] == ex["name"])
    assert changed["angle_limit"] == higher
    # Carried on the exercise, so the patient's own screen can say who changed it.
    assert changed["override"]["by"] == "N. Okonkwo"
    assert changed["override"]["original_angle_limit"] == ex["angle_limit"]
    assert changed["override"]["reason"] == "Tolerating this well in clinic."


def test_tightening_one_exercise_needs_no_reason(client, linked, prescription):
    _, clin = linked
    ex = prescription["exercise_list"][0]

    r = review(client, clin, prescription["prescription_id"],
               decisions=[{"name": ex["name"], "action": "adjust", "angle_limit": max(0, ex["angle_limit"] - 15)}])
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Removing and adding
# ---------------------------------------------------------------------------

def test_an_exercise_can_be_removed(client, linked, prescription):
    _, clin = linked
    ex = prescription["exercise_list"][0]

    r = review(client, clin, prescription["prescription_id"],
               decisions=[{"name": ex["name"], "action": "remove", "reason": "Aggravates the patellar tendon."}])

    names = [e["name"] for e in r.json()["effective"]["exercise_list"]]
    assert ex["name"] not in names
    entry = next(a for a in r.json()["audit"] if a["field"] == "removed")
    assert entry["exercise_name"] == ex["name"]


def test_an_exercise_can_be_added_from_the_catalogue(client, linked, prescription):
    _, clin = linked
    catalogue = client.get("/exercises/catalogue").json()
    drafted = {e["name"] for e in prescription["exercise_list"]}
    candidate = next(c for c in catalogue
                     if c["name"] not in drafted and c["angle_limit"] <= prescription["max_angle"])

    r = review(client, clin, prescription["prescription_id"],
               decisions=[{"name": candidate["name"], "action": "add"}])
    assert r.status_code == 200

    added = next(e for e in r.json()["effective"]["exercise_list"] if e["name"] == candidate["name"])
    # The tracker needs these to count it at all.
    assert added["tracked_joint"] in ("knee", "ankle")
    assert added["instructions"]
    assert added["override"]["by"] == "N. Okonkwo"


def test_adding_above_the_ceiling_needs_a_reason(client, linked, prescription):
    _, clin = linked
    catalogue = client.get("/exercises/catalogue").json()
    drafted = {e["name"] for e in prescription["exercise_list"]}
    candidate = next(c for c in catalogue if c["name"] not in drafted)

    over = prescription["max_angle"] + 20
    assert review(client, clin, prescription["prescription_id"],
                  decisions=[{"name": candidate["name"], "action": "add",
                              "angle_limit": over}]).status_code == 422

    assert review(client, clin, prescription["prescription_id"],
                  decisions=[{"name": candidate["name"], "action": "add", "angle_limit": over,
                              "reason": "Cleared in clinic."}]).status_code == 200


def test_an_exercise_that_exists_nowhere_is_refused(client, linked, prescription):
    _, clin = linked
    r = review(client, clin, prescription["prescription_id"],
               decisions=[{"name": "Trampolining", "action": "add"}])
    assert r.status_code == 422
    assert "Trampolining" in r.json()["detail"]


# ---------------------------------------------------------------------------
# What the patient gets
# ---------------------------------------------------------------------------

def test_the_patient_follows_the_clinicians_version(client, linked, prescription):
    pat, clin = linked
    pid = prescription["prescription_id"]
    ex = prescription["exercise_list"][0]
    higher = ex["angle_limit"] + 20

    review(client, clin, pid,
           decisions=[{"name": ex["name"], "action": "adjust", "angle_limit": higher,
                       "reason": "Good quality movement in clinic."}])

    mine = client.get(f"/me/prescriptions/{pid}", headers=bearer(pat)).json()
    assert mine["status"] == "clinician_modified"
    assert mine["reviewed_by"] == "N. Okonkwo"

    followed = next(e for e in mine["payload"]["exercise_list"] if e["name"] == ex["name"])
    assert followed["angle_limit"] == higher, "the tracker reads this number"
    assert followed["override"]["original_angle_limit"] == ex["angle_limit"]


def test_an_unreviewed_prescription_still_reaches_the_patient(client, linked, prescription):
    pat, _ = linked
    mine = client.get(f"/me/prescriptions/{prescription['prescription_id']}",
                      headers=bearer(pat)).json()
    assert mine["status"] == "draft"
    assert mine["payload"]["exercise_list"]


def test_the_patients_history_says_whether_it_was_reviewed(client, linked, prescription):
    pat, clin = linked
    review(client, clin, prescription["prescription_id"])

    row = client.get("/me/prescriptions", headers=bearer(pat)).json()["prescriptions"][0]
    assert row["status"] == "clinician_approved"
    assert row["reviewed_at"] is not None


def test_the_patients_history_carries_the_ceiling_they_must_follow(client, linked, prescription):
    """
    The stored column is what the model read off the X-ray. Once a clinician has
    lowered it, that is the number the patient is held to — and this list is what
    their own screens read, so it must not still be quoting the model.
    """
    pat, clin = linked
    lowered = prescription["max_angle"] - 25
    review(client, clin, prescription["prescription_id"], ceiling=lowered)

    row = client.get("/me/prescriptions", headers=bearer(pat)).json()["prescriptions"][0]
    assert row["max_angle"] == lowered, "the patient's history must show the approved ceiling"


def test_an_unreviewed_history_row_still_shows_the_models_ceiling(client, linked, prescription):
    """Nobody has changed it, so the model's number is the one in force."""
    pat, _ = linked
    row = client.get("/me/prescriptions", headers=bearer(pat)).json()["prescriptions"][0]
    assert row["max_angle"] == prescription["max_angle"]


def test_the_original_draft_is_never_rewritten(client, linked, prescription):
    """The point of an audit trail is that the original is still there."""
    _, clin = linked
    before = json.dumps(prescription["exercise_list"], sort_keys=True, default=str)

    review(client, clin, prescription["prescription_id"],
           ceiling=prescription["max_angle"] - 25,
           decisions=[{"name": prescription["exercise_list"][0]["name"], "action": "remove"}])

    with SessionLocal() as db:
        stored = json.loads(db.query(Prescription).one().payload)
    assert json.dumps(stored["exercise_list"], sort_keys=True, default=str) == before


# ---------------------------------------------------------------------------
# Who may review
# ---------------------------------------------------------------------------

def test_a_clinician_cannot_review_someone_not_on_their_caseload(client, prescription, clinician):
    """The prescription fixture links a different clinician."""
    stranger = client.post("/clinician/register", json={
        "email": "stranger@clinic.example", "password": PASSWORD}).json()["access_token"]
    pid = prescription["prescription_id"]

    assert client.get(f"/clinician/prescriptions/{pid}", headers=bearer(stranger)).status_code == 404
    assert review(client, stranger, pid, note="mine now").status_code == 404


def test_a_patient_cannot_approve_their_own_prescription(client, linked, prescription):
    pat, _ = linked
    pid = prescription["prescription_id"]
    assert client.post(f"/clinician/prescriptions/{pid}/review", json={},
                       headers=bearer(pat)).status_code == 401


def test_withdrawing_access_closes_the_review_route(client, linked, prescription):
    pat, clin = linked
    link_id = client.get("/me/clinicians", headers=bearer(pat)).json()[0]["id"]
    client.delete(f"/me/clinicians/{link_id}", headers=bearer(pat))

    pid = prescription["prescription_id"]
    assert client.get(f"/clinician/prescriptions/{pid}", headers=bearer(clin)).status_code == 404


def test_a_review_can_be_revised(client, linked, prescription):
    """Clinical opinion changes. The trail keeps both decisions."""
    _, clin = linked
    pid = prescription["prescription_id"]
    was = prescription["max_angle"]

    review(client, clin, pid, ceiling=was - 20)
    second = review(client, clin, pid, ceiling=was - 10)

    assert second.json()["max_angle"] == was - 10
    ceilings = [a for a in second.json()["audit"] if a["field"] == "ceiling"]
    assert len(ceilings) == 3, "the model's, then both clinician decisions"


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

def test_the_catalogue_carries_what_the_tracker_needs(client):
    catalogue = client.get("/exercises/catalogue").json()
    assert len(catalogue) >= 20

    for ex in catalogue:
        assert ex["tracked_joint"] in ("knee", "ankle"), ex["name"]
        assert ex["instructions"], ex["name"]
        assert ex["target_reps"] >= 1
        # A hold has to say where it is held, or the tracker cannot time it.
        if ex["hold_seconds"]:
            assert ex["hold_target"] in ("straight", "flexed"), ex["name"]


# ---------------------------------------------------------------------------
# Free text a clinician types, that another account is shown
# ---------------------------------------------------------------------------

def test_markup_in_a_review_is_escaped_before_it_is_stored(client, linked, prescription):
    """
    The note, the reason on an exercise and the reason on a ceiling change are
    all typed by one account and displayed to another. Nothing renders them with
    innerHTML today, so this is not a live hole — it is the guarantee that the
    screen which eventually does render them cannot become one.
    """
    _, clin = linked
    pid = prescription["prescription_id"]
    drafted = first_exercise(prescription)

    r = review(
        client, clin, pid,
        note="<script>alert(document.cookie)</script>",
        ceiling=prescription["max_angle"] - 10,
        ceiling_reason="tightened <b>after</b> review",
        decisions=[{
            "name":        drafted["name"],
            "action":      "adjust",
            "angle_limit": drafted["angle_limit"] - 5,
            "reason":      '<img src=x onerror="alert(1)">',
        }],
    )
    assert r.status_code == 200
    body = r.json()

    assert body["review_note"] == "&lt;script&gt;alert(document.cookie)&lt;/script&gt;"
    assert "<img" not in first_exercise(body)["override"]["reason"]
    assert all("<b>" not in (a["reason"] or "") for a in body["audit"])

    # Nothing raw reached the row the patient's own screen reads back.
    with SessionLocal() as db:
        row = db.query(Prescription).one()
        assert "<script>" not in (row.review_note or "")
        assert "<img" not in (row.approved_payload or "")
        assert all("<" not in (a.reason or "") for a in db.query(PrescriptionAudit).all())
