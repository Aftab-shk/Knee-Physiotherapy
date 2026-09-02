"""
Deriving weeks-post-op from a stored surgery date.

`weeks_post_op` selects the entire rehab protocol — which exercises, and which
phase's goals. Until now a patient retyped it into a form on every visit, from
memory, and a plausible wrong number is indistinguishable from a right one, so
nothing catches the mistake. The consequence is being prescribed a phase you are
not in: for a knee two weeks out of a replacement, that means loading it before
it is ready.

What these pin down:

  * The arithmetic is floored, not rounded. Week N lasts until day 7(N+1), and
    rounding would advance someone into the next phase three days early.
  * A date on file is used only when the caller does not supply a week. An
    explicit value always wins, because it may be a correction.
  * Guests are unaffected. Nothing about this requires an account.

Run:  python -m pytest backend/tests -q
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="surgery tests need sqlalchemy")

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier, png_bytes

import main
from clinical_logic import MAX_WEEKS_POST_OP, weeks_since_surgery

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


def days_ago(n):
    return (date.today() - timedelta(days=n)).isoformat()


def analyse(client, tok=None, **form):
    data = {"knee_side": "left", "surgery_type": "tkr"}
    data.update(form)
    return client.post(
        "/analyse-xray",
        files={"image": ("knee.png", png_bytes(), "image/png")},
        data=data,
        headers=bearer(tok) if tok else {},
    )


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("days,expected", [
    (0, 0),      # the day itself
    (6, 0),      # still the first week
    (7, 1),      # day 7 begins week 1
    (13, 1),
    (14, 2),
    (27, 3),     # not 4 — floored, so phase 3 holds until day 28
    (28, 4),
    (365, 52),
])
def test_weeks_are_floored_not_rounded(days, expected):
    surgery = date(2026, 1, 1)
    assert weeks_since_surgery(surgery, surgery + timedelta(days=days)) == expected


def test_a_future_date_is_an_error_not_a_negative_week():
    surgery = date(2026, 6, 1)
    with pytest.raises(ValueError):
        weeks_since_surgery(surgery, surgery - timedelta(days=1))


def test_a_very_old_date_is_capped_at_the_protocol_limit():
    surgery = date(2000, 1, 1)
    assert weeks_since_surgery(surgery, date(2026, 1, 1)) == MAX_WEEKS_POST_OP


# ---------------------------------------------------------------------------
# Recording it
# ---------------------------------------------------------------------------

def test_recording_a_surgery_returns_the_derived_week(client, token):
    r = client.patch(
        "/me/surgery",
        json={"surgery_date": days_ago(30), "surgery_type": "tkr"},
        headers=bearer(token),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["surgery_date"] == days_ago(30)
    assert body["surgery_type"] == "tkr"
    assert body["weeks_post_op"] == 4


def test_the_date_survives_and_the_week_moves_with_it(client, token):
    client.patch("/me/surgery", json={"surgery_date": days_ago(21)}, headers=bearer(token))

    me = client.get("/auth/me", headers=bearer(token)).json()
    assert me["surgery_date"] == days_ago(21)
    assert me["weeks_post_op"] == 3, "derived on read, not stored"


def test_a_wrong_date_can_be_taken_back(client, token):
    client.patch("/me/surgery", json={"surgery_date": days_ago(30), "surgery_type": "acl"},
                 headers=bearer(token))
    r = client.patch("/me/surgery", json={"surgery_date": None, "surgery_type": None},
                     headers=bearer(token))
    assert r.json()["surgery_date"] is None
    assert r.json()["weeks_post_op"] is None


def test_a_future_surgery_date_is_refused(client, token):
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    r = client.patch("/me/surgery", json={"surgery_date": tomorrow}, headers=bearer(token))
    assert r.status_code == 422
    assert "future" in r.json()["detail"].lower()


def test_recording_a_surgery_requires_an_account(client):
    assert client.patch("/me/surgery", json={"surgery_date": days_ago(10)}).status_code == 401


def test_one_patient_cannot_see_anothers_surgery_date(client, token):
    other = client.post(
        "/auth/register", json={"email": "bob@example.com", "password": PASSWORD}
    ).json()["access_token"]
    client.patch("/me/surgery", json={"surgery_date": days_ago(14)}, headers=bearer(token))

    assert client.get("/auth/me", headers=bearer(other)).json()["surgery_date"] is None


# ---------------------------------------------------------------------------
# Using it
# ---------------------------------------------------------------------------

def test_an_analysis_no_longer_needs_the_week_retyped(client, token):
    client.patch("/me/surgery", json={"surgery_date": days_ago(35), "surgery_type": "tkr"},
                 headers=bearer(token))

    r = analyse(client, token)          # note: no weeks_post_op sent
    assert r.status_code == 200
    assert r.json()["weeks_post_op"] == 5


def test_an_explicit_week_still_wins(client, token):
    """The patient may be correcting the record, or asking a what-if."""
    client.patch("/me/surgery", json={"surgery_date": days_ago(35)}, headers=bearer(token))

    r = analyse(client, token, weeks_post_op="12")
    assert r.json()["weeks_post_op"] == 12


def test_the_derived_week_reaches_the_prescription(client, token):
    """
    Not just echoed back — it has to be the number that selects the phase, or
    the whole feature is decorative.
    """
    early = analyse(client, token, weeks_post_op="1").json()

    client.patch("/me/surgery", json={"surgery_date": days_ago(150)}, headers=bearer(token))
    late = analyse(client, token).json()

    assert late["weeks_post_op"] == 21
    assert late["rehab_phase"] != early["rehab_phase"]


def test_without_a_recorded_date_the_week_is_still_required(client, token):
    r = analyse(client, token)
    assert r.status_code == 422
    assert "weeks_post_op" in r.json()["detail"]


def test_guests_are_unaffected(client):
    assert analyse(client, weeks_post_op="4").status_code == 200
    assert analyse(client).status_code == 422


def test_a_non_surgical_analysis_ignores_the_stored_date(client, token):
    """
    surgery_type 'none' means conservative OA management: there is no operation
    to count from, so a date left on the record must not leak into it.
    """
    client.patch("/me/surgery", json={"surgery_date": days_ago(35)}, headers=bearer(token))

    r = analyse(client, token, surgery_type="none")
    assert r.status_code == 200
    assert r.json()["weeks_post_op"] is None
