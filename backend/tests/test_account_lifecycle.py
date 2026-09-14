"""
Signing out, changing a password, resetting a forgotten one, and leaving.

The thing being defended here is that a token can be taken back. A JWT is valid
until it expires, so before token_version existed "sign out" meant the browser
dropped its copy while the token kept working for the rest of a fortnight —
including the copy on the laptop the patient no longer has.

Run:  python -m pytest backend/tests/test_account_lifecycle.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="needs fastapi")
from fastapi.testclient import TestClient

import auth
import main

PASSWORD = "Str0ngPassphrase!"
NEW_PASSWORD = "An0therGoodPassphrase!"


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """
    These tests sign in and out far more often per second than a person does.
    RATE_LIMIT_REQUESTS = 0 disables the limiter; test_api_security.py is where
    the limiter itself is tested.
    """
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 0)
    main._rate_buckets.clear()


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def register(client, email, password=PASSWORD):
    r = client.post("/auth/register", json={
        "email": email, "password": password, "display_name": "Test Person",
    })
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


def signed_in(client, token):
    return client.get("/auth/me", headers=bearer(token)).status_code == 200


# ── Signing out ─────────────────────────────────────────────────────────────

def test_signing_out_kills_the_token_that_did_it(client):
    token = register(client, "logout1@test.com")
    assert signed_in(client, token)

    assert client.post("/auth/logout", headers=bearer(token)).status_code == 200
    assert not signed_in(client, token), "the token still worked after signing out"


def test_signing_out_kills_every_other_session_too(client):
    """The point of it: a token on a device you no longer have."""
    token = register(client, "logout2@test.com")
    other = client.post("/auth/login", json={
        "email": "logout2@test.com", "password": PASSWORD,
    }).json()["access_token"]
    assert signed_in(client, other)

    client.post("/auth/logout", headers=bearer(token))
    assert not signed_in(client, other), "a second device stayed signed in"


def test_signing_in_again_works_after_signing_out(client):
    token = register(client, "logout3@test.com")
    client.post("/auth/logout", headers=bearer(token))
    fresh = client.post("/auth/login", json={
        "email": "logout3@test.com", "password": PASSWORD,
    })
    assert fresh.status_code == 200
    assert signed_in(client, fresh.json()["access_token"])


# ── Changing a password ─────────────────────────────────────────────────────

def test_changing_a_password_needs_the_old_one(client):
    token = register(client, "change1@test.com")
    r = client.post("/auth/change-password", headers=bearer(token), json={
        "current_password": "not it", "new_password": NEW_PASSWORD,
    })
    assert r.status_code == 401
    assert signed_in(client, token)


def test_changing_a_password_ends_other_sessions_but_not_this_one(client):
    token = register(client, "change2@test.com")
    elsewhere = client.post("/auth/login", json={
        "email": "change2@test.com", "password": PASSWORD,
    }).json()["access_token"]

    r = client.post("/auth/change-password", headers=bearer(token), json={
        "current_password": PASSWORD, "new_password": NEW_PASSWORD,
    })
    assert r.status_code == 200
    replacement = r.json()["access_token"]

    assert not signed_in(client, elsewhere), "the other session survived a password change"
    assert signed_in(client, replacement), "the caller was logged out of their own change"


def test_the_new_password_is_the_one_that_works(client):
    token = register(client, "change3@test.com")
    client.post("/auth/change-password", headers=bearer(token), json={
        "current_password": PASSWORD, "new_password": NEW_PASSWORD,
    })
    assert client.post("/auth/login", json={
        "email": "change3@test.com", "password": PASSWORD}).status_code == 401
    assert client.post("/auth/login", json={
        "email": "change3@test.com", "password": NEW_PASSWORD}).status_code == 200


def test_a_weak_new_password_is_refused(client):
    token = register(client, "change4@test.com")
    r = client.post("/auth/change-password", headers=bearer(token), json={
        "current_password": PASSWORD, "new_password": "password",
    })
    assert r.status_code == 422


# ── Forgetting one ──────────────────────────────────────────────────────────

def test_forgot_password_says_the_same_thing_either_way(client):
    """
    Otherwise the form answers "is this person a patient here?" for anyone who
    asks.
    """
    register(client, "forgot1@test.com")
    known = client.post("/auth/forgot-password", json={"email": "forgot1@test.com"})
    unknown = client.post("/auth/forgot-password", json={"email": "nobody@test.com"})
    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json()


def test_a_reset_link_sets_the_password_and_can_only_be_used_once(client, monkeypatch):
    sent = {}
    monkeypatch.setattr(main.mailer, "send_password_reset",
                        lambda email, token, name=None: sent.update(token=token))
    register(client, "reset1@test.com")
    client.post("/auth/forgot-password", json={"email": "reset1@test.com"})
    token = sent["token"]

    r = client.post("/auth/reset-password", json={
        "token": token, "new_password": NEW_PASSWORD,
    })
    assert r.status_code == 200
    assert signed_in(client, r.json()["access_token"])
    assert client.post("/auth/login", json={
        "email": "reset1@test.com", "password": NEW_PASSWORD}).status_code == 200

    again = client.post("/auth/reset-password", json={
        "token": token, "new_password": "YetAnotherPassphrase9!",
    })
    assert again.status_code == 400, "the same link worked twice"


def test_a_reset_ends_sessions_someone_else_may_be_holding(client, monkeypatch):
    """The reason people reset a password they still know."""
    sent = {}
    monkeypatch.setattr(main.mailer, "send_password_reset",
                        lambda email, token, name=None: sent.update(token=token))
    intruder = register(client, "reset2@test.com")
    assert signed_in(client, intruder)

    client.post("/auth/forgot-password", json={"email": "reset2@test.com"})
    client.post("/auth/reset-password", json={
        "token": sent["token"], "new_password": NEW_PASSWORD,
    })
    assert not signed_in(client, intruder), "the stolen session survived the reset"


def test_a_made_up_reset_token_is_refused(client):
    register(client, "reset3@test.com")
    r = client.post("/auth/reset-password", json={
        "token": "a" * 43, "new_password": NEW_PASSWORD,
    })
    assert r.status_code == 400


def test_an_expired_reset_token_is_refused(client, monkeypatch):
    sent = {}
    monkeypatch.setattr(main.mailer, "send_password_reset",
                        lambda email, token, name=None: sent.update(token=token))
    register(client, "reset4@test.com")
    client.post("/auth/forgot-password", json={"email": "reset4@test.com"})
    monkeypatch.setattr(auth, "reset_token_is_live", lambda account: False)
    r = client.post("/auth/reset-password", json={
        "token": sent["token"], "new_password": NEW_PASSWORD,
    })
    assert r.status_code == 400


# ── Leaving ─────────────────────────────────────────────────────────────────

def test_export_returns_the_patients_own_data_without_their_credentials(client):
    token = register(client, "export1@test.com")
    r = client.get("/me/export", headers=bearer(token))
    assert r.status_code == 200
    body = r.json()
    assert body["account"]["email"] == "export1@test.com"
    assert "password_hash" not in body["account"]
    assert "reset_token_hash" not in body["account"]
    for key in ("prescriptions", "exercise_sessions", "outcome_scores", "share_links"):
        assert key in body
    assert "attachment" in r.headers["content-disposition"]


def test_deleting_needs_the_password_and_the_word_delete(client):
    token = register(client, "delete1@test.com")
    wrong_password = client.post("/me/delete", headers=bearer(token), json={
        "password": "not it", "confirm": "DELETE"})
    assert wrong_password.status_code == 401

    no_confirm = client.post("/me/delete", headers=bearer(token), json={
        "password": PASSWORD, "confirm": "yes"})
    assert no_confirm.status_code == 422

    assert signed_in(client, token), "a refused delete should change nothing"


def test_deleting_removes_the_account_its_data_and_its_sessions(client):
    token = register(client, "delete2@test.com")
    client.post("/sessions/sets", headers=bearer(token), json={
        "client_session_id": "delete-me-session", "exercise_name": "Mini Squat",
        "angle_limit": 60, "target_sets": 3, "set_index": 1, "reps_completed": 10,
        "peak_flexion_deg": 50.0, "duration_seconds": 40.0,
        "breach_count": 0, "breach_seconds": 0.0, "suspended_seconds": 0.0,
        "view_verified": True,
    })
    client.post("/me/share-links", headers=bearer(token), json={"expires_in_days": 7})

    r = client.post("/me/delete", headers=bearer(token), json={
        "password": PASSWORD, "confirm": "DELETE"})
    assert r.status_code == 200, r.text
    receipt = r.json()
    assert receipt["sessions"] == 1
    assert receipt["share_links"] == 1

    # The account, the token, and the address are all gone.
    assert not signed_in(client, token)
    assert client.post("/auth/login", json={
        "email": "delete2@test.com", "password": PASSWORD}).status_code == 401
    assert client.post("/auth/register", json={
        "email": "delete2@test.com", "password": PASSWORD}).status_code == 201


def test_a_deleted_patients_share_link_stops_working(client):
    token = register(client, "delete3@test.com")
    link = client.post("/me/share-links", headers=bearer(token),
                       json={"expires_in_days": 7}).json()
    assert client.get(f"/share/{link['token']}").status_code == 200

    client.post("/me/delete", headers=bearer(token),
                json={"password": PASSWORD, "confirm": "DELETE"})
    assert client.get(f"/share/{link['token']}").status_code == 404, \
        "a link to a deleted patient still served their progress"


# ── The token check itself ──────────────────────────────────────────────────

def test_the_version_check_refuses_anything_but_an_exact_match(client):
    """
    Guards the comparison directly. A counter rather than a timestamp is the
    whole reason signing out works within the same second it was issued.
    """
    class Account:
        token_version = 3

    assert auth._still_valid(Account, 3) is True
    assert auth._still_valid(Account, 2) is False, "a pre-revocation token was accepted"
    assert auth._still_valid(Account, 4) is False, "a token from the future was accepted"

    # An account that predates the column, and a token that predates the claim,
    # both read as 0 — so shipping this signs nobody out.
    class Old:
        token_version = 0
    assert auth._still_valid(Old, 0) is True


def test_signing_out_twice_keeps_moving_the_counter(client):
    token = register(client, "twice@test.com")
    client.post("/auth/logout", headers=bearer(token))
    again = client.post("/auth/login", json={
        "email": "twice@test.com", "password": PASSWORD}).json()["access_token"]
    client.post("/auth/logout", headers=bearer(again))
    assert not signed_in(client, again)
    assert not signed_in(client, token)
