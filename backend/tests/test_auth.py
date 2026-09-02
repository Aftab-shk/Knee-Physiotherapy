"""
Accounts, tokens, and the persistence they unlock.

What these pin down, in rough order of how much it would cost to get wrong:

  * A failed login must not reveal whether the account exists. In a medical app
    that is not an abstract enumeration concern — it tells an attacker that a
    given person is a patient.
  * A token must be rejected unless this server signed it, and unless it is
    still in date.
  * One patient must never see another patient's history.
  * Guest use has to keep working: login.html offers a guest route, and
    /analyse-xray was open before accounts existed.

Run:  python -m pytest backend/tests -q
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="account tests need sqlalchemy")
pytest.importorskip("jwt", reason="account tests need PyJWT")

import jwt
from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier, png_bytes

import auth
import main
from db import SessionLocal
from models import Patient, Prescription

GOOD_PASSWORD = "correct-horse-battery"


@pytest.fixture
def client(monkeypatch):
    """
    A clean database per test, and no real inference.

    The stub is installed after entering TestClient because the lifespan handler
    builds a real KneeClassifier and assigns it to the module global — patching
    beforehand is silently undone.
    """
    reset_database()
    stub_inference(monkeypatch)
    main._rate_buckets.clear()
    with TestClient(main.app) as c:
        monkeypatch.setattr(main, "classifier", StubClassifier())
        yield c


def register(client, email="ada@example.com", password=GOOD_PASSWORD, **extra):
    body = {"email": email, "password": password}
    body.update(extra)
    return client.post("/auth/register", json=body)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def analyse(client, headers=None):
    return client.post(
        "/analyse-xray",
        files={"image": ("knee.png", png_bytes(), "image/png")},
        data={"knee_side": "left", "surgery_type": "tkr", "weeks_post_op": "3"},
        headers=headers or {},
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_returns_a_working_token(client):
    r = register(client, display_name="Ada")
    assert r.status_code == 201

    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    assert body["patient"]["display_name"] == "Ada"

    me = client.get("/auth/me", headers=bearer(body["access_token"]))
    assert me.status_code == 200
    assert me.json()["id"] == body["patient"]["id"]


def test_email_is_normalised_so_one_address_is_one_account(client):
    assert register(client, email="Ada@Example.COM").status_code == 201
    assert register(client, email="  ada@example.com  ").status_code == 409
    assert register(client, email="ADA@EXAMPLE.COM").status_code == 409


def test_duplicate_registration_is_rejected(client):
    assert register(client).status_code == 201
    r = register(client, password="a-completely-different-one")
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"].lower()


@pytest.mark.parametrize("password", ["", "short", "seven77"])
def test_short_passwords_are_refused(client, password):
    assert register(client, password=password).status_code == 422


@pytest.mark.parametrize("email", ["not-an-email", "@example.com", "ada@", ""])
def test_malformed_emails_are_refused(client, email):
    assert register(client, email=email).status_code == 422


def test_the_password_is_never_stored_or_returned(client):
    r = register(client)
    assert GOOD_PASSWORD not in r.text

    with SessionLocal() as db:
        patient = db.query(Patient).one()
        assert GOOD_PASSWORD not in patient.password_hash
        assert patient.password_hash.startswith("$argon2")
        # The hash must actually verify, or "not stored in plaintext" would be
        # satisfied by storing nonsense.
        assert auth.verify_password(patient.password_hash, GOOD_PASSWORD)


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def test_login_succeeds_regardless_of_email_case(client):
    register(client, email="ada@example.com")
    r = client.post("/auth/login", json={"email": "ADA@Example.com", "password": GOOD_PASSWORD})
    assert r.status_code == 200
    assert client.get("/auth/me", headers=bearer(r.json()["access_token"])).status_code == 200


def test_wrong_password_and_unknown_account_are_indistinguishable(client):
    register(client, email="ada@example.com")

    wrong = client.post("/auth/login", json={"email": "ada@example.com", "password": "not-it-at-all"})
    unknown = client.post("/auth/login", json={"email": "ghost@example.com", "password": "not-it-at-all"})

    assert wrong.status_code == unknown.status_code == 401
    # Identical bodies: any difference at all — wording, field order, a stray
    # code — is enough to tell whether the address is registered.
    assert wrong.json() == unknown.json()


def test_login_records_the_time(client):
    register(client)
    with SessionLocal() as db:
        assert db.query(Patient).one().last_login_at is None

    assert client.post(
        "/auth/login", json={"email": "ada@example.com", "password": GOOD_PASSWORD}
    ).status_code == 200

    with SessionLocal() as db:
        assert db.query(Patient).one().last_login_at is not None


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def test_protected_routes_require_a_token(client):
    assert client.get("/auth/me").status_code == 401
    assert client.get("/me/prescriptions").status_code == 401


@pytest.mark.parametrize("header", [
    "Bearer garbage",
    "Bearer ",
    "Basic abc123",
    "",
])
def test_malformed_authorization_headers_are_rejected(client, header):
    assert client.get("/auth/me", headers={"Authorization": header}).status_code == 401


def test_a_token_this_server_did_not_sign_is_rejected(client):
    patient_id = register(client).json()["patient"]["id"]
    forged = jwt.encode({"sub": patient_id, "exp": 2 ** 31}, "some-other-secret", algorithm="HS256")
    assert client.get("/auth/me", headers=bearer(forged)).status_code == 401


def test_an_unsigned_token_is_rejected(client):
    """
    The classic JWT failure: a token declaring alg "none" and no signature.
    decode_token pins algorithms=["HS256"], which is what refuses it.
    """
    patient_id = register(client).json()["patient"]["id"]
    unsigned = jwt.encode({"sub": patient_id, "exp": 2 ** 31}, key="", algorithm="none")
    assert client.get("/auth/me", headers=bearer(unsigned)).status_code == 401


def test_an_expired_token_is_rejected(client):
    patient_id = register(client).json()["patient"]["id"]
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    expired = jwt.encode(
        {"sub": patient_id, "exp": int(past.timestamp())},
        auth.JWT_SECRET,
        algorithm=auth.ALGORITHM,
    )
    assert client.get("/auth/me", headers=bearer(expired)).status_code == 401


def test_a_token_naming_a_deleted_account_is_rejected(client):
    token = register(client).json()["access_token"]
    assert client.get("/auth/me", headers=bearer(token)).status_code == 200

    with SessionLocal() as db:
        db.delete(db.query(Patient).one())
        db.commit()

    # The token is still validly signed and in date; it just names nobody.
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_a_guest_analysis_still_works_and_is_not_stored(client):
    r = analyse(client)
    assert r.status_code == 200
    assert r.json()["prescription_id"] is None

    with SessionLocal() as db:
        assert db.query(Prescription).count() == 0


def test_a_stale_token_does_not_lock_a_guest_out(client):
    """
    /analyse-xray treats an unusable token as absent rather than rejecting it.
    The alternative is that an expired token left in an old tab stops someone
    using a feature that never needed an account.
    """
    r = analyse(client, headers=bearer("not-a-real-token"))
    assert r.status_code == 200
    assert r.json()["prescription_id"] is None


def test_a_signed_in_analysis_is_saved_and_listed(client):
    token = register(client).json()["access_token"]

    r = analyse(client, headers=bearer(token))
    assert r.status_code == 200
    prescription_id = r.json()["prescription_id"]
    assert prescription_id

    history = client.get("/me/prescriptions", headers=bearer(token))
    assert history.status_code == 200
    body = history.json()
    assert body["count"] == 1

    row = body["prescriptions"][0]
    assert row["id"] == prescription_id
    # The indexed columns must agree with the analysis they were copied from.
    assert row["kl_grade"] == r.json()["kl_grade"]
    assert row["max_angle"] == r.json()["max_angle"]
    assert row["surgery_type"] == "tkr"
    assert row["weeks_post_op"] == 3


def test_the_stored_payload_is_the_whole_response(client):
    """
    The verbatim copy is what lets an old prescription still render after the
    exercise database or the wording of the rationale changes.
    """
    import json

    token = register(client).json()["access_token"]
    response = analyse(client, headers=bearer(token)).json()

    with SessionLocal() as db:
        stored = json.loads(db.query(Prescription).one().payload)

    assert stored["exercise_list"] == response["exercise_list"]
    assert stored["rationale"] == response["rationale"]
    assert stored["disclaimer"] == response["disclaimer"]


def test_history_is_newest_first(client):
    token = register(client).json()["access_token"]
    for _ in range(3):
        assert analyse(client, headers=bearer(token)).status_code == 200

    rows = client.get("/me/prescriptions", headers=bearer(token)).json()["prescriptions"]
    assert len(rows) == 3
    timestamps = [r["created_at"] for r in rows]
    assert timestamps == sorted(timestamps, reverse=True)


def test_one_patient_cannot_see_another_patients_history(client):
    ada = register(client, email="ada@example.com").json()["access_token"]
    bob = register(client, email="bob@example.com").json()["access_token"]

    analyse(client, headers=bearer(ada))

    assert client.get("/me/prescriptions", headers=bearer(ada)).json()["count"] == 1
    assert client.get("/me/prescriptions", headers=bearer(bob)).json()["count"] == 0


def test_deleting_a_patient_takes_their_records_with_them(client):
    token = register(client).json()["access_token"]
    analyse(client, headers=bearer(token))

    with SessionLocal() as db:
        assert db.query(Prescription).count() == 1
        db.delete(db.query(Patient).one())
        db.commit()
        # Orphaned clinical records belonging to a deleted account would be both
        # wrong and a data-protection problem.
        assert db.query(Prescription).count() == 0


def test_a_failed_save_does_not_cost_the_patient_their_result(client, monkeypatch):
    token = register(client).json()["access_token"]

    def explode(*_args, **_kwargs):
        raise RuntimeError("database is on fire")

    monkeypatch.setattr(main.Session, "commit", explode, raising=False)

    r = analyse(client, headers=bearer(token))
    assert r.status_code == 200, "a storage failure must not fail the analysis"
    # …but it must not claim to have saved it either.
    assert r.json()["prescription_id"] is None
