"""
Share links.

A clinician will click a link. A clinician will not create an account in an app
their patient uses — insisting they do is where this feature dies. So the link
itself is the credential, and everything below follows from that:

  * It must be revocable. Sharing health data you cannot un-share is not worth
    having, which is why these are rows rather than self-contained signed tokens.
  * The stored form must be useless if the database leaks — hashed, like a
    password.
  * Every failure looks the same. Telling an old link apart from a revoked one
    confirms to whoever holds it that it was real, and whose it was.
  * It shows progress and nothing else. No email, no X-ray, no prescriptions.

Run:  python -m pytest backend/tests -q
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="share tests need sqlalchemy")

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier

import main
from db import SessionLocal
from models import ExerciseSession, ExerciseSet, Patient, ShareLink

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
        "/auth/register",
        json={"email": "ada@example.com", "password": PASSWORD, "display_name": "Ada L"},
    ).json()["access_token"]


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


def make_link(client, tok, **body):
    return client.post("/me/share-links", json=body or {}, headers=bearer(tok))


def seed_session(email="ada@example.com", *, days_ago=0, peak=52.0, before=6, after=3):
    started = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    started -= timedelta(days=days_ago)
    with SessionLocal() as db:
        patient = db.query(Patient).filter_by(email=email).one()
        session = ExerciseSession(
            patient_id=patient.id, client_session_id=f"seed-{email}-{days_ago}",
            exercise_name="Mini Squat", tracked_joint="knee", knee_side="left",
            angle_limit=60, target_reps=10, target_sets=1,
            started_at=started, last_set_at=started, sets_completed=1,
            completed=True, view_verified=True,
            pain_before=before, pain_after=after,
        )
        db.add(session)
        db.flush()
        db.add(ExerciseSet(session_id=session.id, set_index=1, reps_completed=10,
                           duration_seconds=40.0, peak_flexion_deg=peak, mean_visibility=0.9))
        db.commit()


# ---------------------------------------------------------------------------
# Creating and opening
# ---------------------------------------------------------------------------

def test_a_link_opens_without_any_account(client, token):
    seed_session()
    created = make_link(client, token, label="Dr Okonkwo, 14 Aug").json()

    # No Authorization header at all — the whole point of the feature.
    r = client.get(f"/share/{created['token']}")
    assert r.status_code == 200

    body = r.json()
    assert body["patient_name"] == "Ada L"
    assert body["label"] == "Dr Okonkwo, 14 Aug"
    assert body["progress"]["summary"]["sessions"] == 1
    assert body["progress"]["summary"]["best_flexion_deg"] == 52.0


def test_the_shared_view_carries_the_clinical_picture(client, token):
    seed_session(days_ago=2, peak=48.0, before=7, after=5)
    seed_session(days_ago=0, peak=55.0, before=5, after=2)

    body = client.get(f"/share/{make_link(client, token).json()['token']}").json()
    progress = body["progress"]

    assert len(progress["rom_by_exercise"][0]["points"]) == 2
    assert len(progress["pain_trend"]) == 2
    assert progress["by_exercise"][0]["exercise_name"] == "Mini Squat"
    assert progress["summary"]["current_streak_days"] >= 1


def test_it_shares_progress_and_nothing_else(client, token):
    """
    Minimal disclosure. A name so the clinician knows whose knee this is, and
    the charts. Not the email, not an X-ray, not the stored prescriptions.
    """
    seed_session()
    r = client.get(f"/share/{make_link(client, token).json()['token']}")

    assert "ada@example.com" not in r.text
    assert "password" not in r.text.lower()
    body = r.json()
    assert set(body) == {"patient_name", "shared_at", "expires_at", "label", "progress"}


def test_the_token_is_returned_once_and_never_again(client, token):
    created = make_link(client, token).json()
    assert created["token"]

    listed = client.get("/me/share-links", headers=bearer(token)).json()
    assert len(listed) == 1
    assert "token" not in listed[0]


def test_only_a_hash_of_the_token_is_stored(client, token):
    """A leaked database has to yield no working links, as it yields no passwords."""
    created = make_link(client, token).json()

    with SessionLocal() as db:
        link = db.query(ShareLink).one()
        assert created["token"] not in link.token_hash
        assert len(link.token_hash) == 64            # SHA-256, hex
        assert link.token_hash == main._hash_share_token(created["token"])


def test_creating_a_link_requires_an_account(client):
    assert client.post("/me/share-links", json={}).status_code == 401
    assert client.get("/me/share-links").status_code == 401


# ---------------------------------------------------------------------------
# Taking it back
# ---------------------------------------------------------------------------

def test_a_revoked_link_stops_working_immediately(client, token):
    seed_session()
    created = make_link(client, token).json()
    assert client.get(f"/share/{created['token']}").status_code == 200

    r = client.delete(f"/me/share-links/{created['id']}", headers=bearer(token))
    assert r.status_code == 200
    assert r.json()["is_active"] is False

    assert client.get(f"/share/{created['token']}").status_code == 404


def test_revoking_twice_is_not_an_error(client, token):
    created = make_link(client, token).json()
    first = client.delete(f"/me/share-links/{created['id']}", headers=bearer(token))
    second = client.delete(f"/me/share-links/{created['id']}", headers=bearer(token))

    assert second.status_code == 200
    # The revocation time is the first one — it did not move.
    assert second.json()["revoked_at"] == first.json()["revoked_at"]


def test_an_expired_link_stops_working(client, token):
    seed_session()
    created = make_link(client, token, days=1).json()

    with SessionLocal() as db:
        link = db.query(ShareLink).one()
        link.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()

    assert client.get(f"/share/{created['token']}").status_code == 404


def test_nobody_can_revoke_someone_elses_link(client, token):
    other = client.post(
        "/auth/register", json={"email": "bob@example.com", "password": PASSWORD}
    ).json()["access_token"]
    created = make_link(client, token).json()

    assert client.delete(f"/me/share-links/{created['id']}", headers=bearer(other)).status_code == 404
    # And it still works, because it was never revoked.
    seed_session()
    assert client.get(f"/share/{created['token']}").status_code == 200


def test_deleting_a_patient_takes_their_links_with_them(client, token):
    created = make_link(client, token).json()

    with SessionLocal() as db:
        db.delete(db.query(Patient).filter_by(email="ada@example.com").one())
        db.commit()
        assert db.query(ShareLink).count() == 0

    assert client.get(f"/share/{created['token']}").status_code == 404


# ---------------------------------------------------------------------------
# Failures give nothing away
# ---------------------------------------------------------------------------

def test_every_failure_looks_identical(client, token):
    """
    Distinguishing "expired", "revoked" and "never existed" would confirm to
    whoever holds an old link that it was once real — and, since they have it,
    whose it was.
    """
    seed_session()
    revoked = make_link(client, token).json()
    client.delete(f"/me/share-links/{revoked['id']}", headers=bearer(token))

    expired = make_link(client, token).json()
    with SessionLocal() as db:
        link = db.query(ShareLink).filter_by(id=expired["id"]).one()
        link.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()

    responses = [
        client.get(f"/share/{revoked['token']}"),
        client.get(f"/share/{expired['token']}"),
        client.get("/share/a-token-that-never-existed-at-all"),
    ]
    assert {r.status_code for r in responses} == {404}
    assert len({r.json()["detail"] for r in responses}) == 1


def test_a_guessed_token_does_not_work(client, token):
    seed_session()
    real = make_link(client, token).json()["token"]
    assert client.get(f"/share/{real[:-1]}x").status_code == 404


# ---------------------------------------------------------------------------
# What the patient sees about their own link
# ---------------------------------------------------------------------------

def test_the_patient_can_see_whether_it_was_opened(client, token):
    seed_session()
    created = make_link(client, token).json()
    assert created["access_count"] == 0
    assert created["last_accessed_at"] is None

    client.get(f"/share/{created['token']}")
    client.get(f"/share/{created['token']}")

    listed = client.get("/me/share-links", headers=bearer(token)).json()[0]
    assert listed["access_count"] == 2
    assert listed["last_accessed_at"] is not None


def test_links_are_listed_newest_first(client, token):
    for label in ("first", "second", "third"):
        make_link(client, token, label=label)

    labels = [r["label"] for r in client.get("/me/share-links", headers=bearer(token)).json()]
    assert labels == ["third", "second", "first"]


def test_one_patient_cannot_list_anothers_links(client, token):
    other = client.post(
        "/auth/register", json={"email": "bob@example.com", "password": PASSWORD}
    ).json()["access_token"]
    make_link(client, token)

    assert client.get("/me/share-links", headers=bearer(other)).json() == []


@pytest.mark.parametrize("days", [0, 91, -5])
def test_an_unreasonable_lifetime_is_refused(client, token, days):
    assert make_link(client, token, days=days).status_code == 422


def test_the_default_lifetime_is_a_fortnight(client, token):
    created = make_link(client, token).json()
    expires = datetime.fromisoformat(created["expires_at"])
    created_at = datetime.fromisoformat(created["created_at"])
    assert 13 <= (expires - created_at).days <= 14
