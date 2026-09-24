"""
KOOS-JR — the one number in this app the patient supplies.

Everything else recorded here is something the app measured: how far the knee
bent, how many sets were finished, how long the safe angle was exceeded. None of
it answers whether the knee is getting better to live with, and a joint that
flexes to 120° and hurts on every stair is not a success.

What these tests pin down:

  * The published lookup table, exactly. This is the whole reason to use a
    standard instrument: 58 here has to mean what 58 means in a registry. A
    transcription error in that table is invisible — every value it produces
    still looks like a plausible score — so the endpoints, the direction and the
    spacing are all asserted rather than trusted.
  * A partly completed form is refused, not imputed. KOOS-JR has no published
    rule for a missing item, and averaging the six that were answered produces
    something indistinguishable from a real score that is not one.
  * Two knees are two questionnaires. KOOS-JR asks about "your knee", singular.
  * A change smaller than the instrument can detect is reported as no change,
    not as a small improvement.
  * It is never a gate, and it never moves a movement ceiling.

Run:  python -m pytest backend/tests -q
"""

import sys
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import outcome_measures as om

PERFECT = [0, 0, 0, 0, 0, 0, 0]
WORST = [4, 4, 4, 4, 4, 4, 4]
MIDDLING = [1, 2, 0, 3, 1, 2, 1]        # raw 10


# ---------------------------------------------------------------------------
# The instrument itself
# ---------------------------------------------------------------------------

def test_the_lookup_table_covers_every_possible_raw_sum():
    assert len(om._KOOS_JR_INTERVAL) == om.MAX_RAW_SUM + 1 == 29


def test_the_scale_runs_the_full_range_at_its_ends():
    """
    Seven items answered "none" is a knee with no symptoms; seven answered
    "extreme" is total knee disability. If either end drifts, every score in
    between is quietly on a different scale from everybody else's.
    """
    assert om.score(PERFECT)["interval_score"] == 100.0
    assert om.score(WORST)["interval_score"] == 0.0


def test_more_symptoms_always_means_a_lower_score():
    """
    The two scales run opposite ways — raw sum up is worse, interval score up is
    better — which is the single easiest thing to get backwards in this file.
    """
    scores = list(om._KOOS_JR_INTERVAL)
    assert scores == sorted(scores, reverse=True)
    assert all(a > b for a, b in pairwise(scores)), "no two raw sums may share a score"


def test_the_table_is_not_a_straight_line():
    """
    The irregular spacing is the Rasch calibration, and it is the entire reason
    this instrument is worth using. A well-meaning simplification to a linear
    map would still produce plausible-looking scores — and none of them would
    match what a registry computes from the same seven answers.
    """
    first_step = om._KOOS_JR_INTERVAL[0] - om._KOOS_JR_INTERVAL[1]
    last_step = om._KOOS_JR_INTERVAL[27] - om._KOOS_JR_INTERVAL[28]
    assert abs(last_step - first_step) > 0.1


@pytest.mark.parametrize("raw,expected", [
    (0, 100.0), (1, 92.0), (7, 64.8), (14, 47.4), (20, 31.5), (27, 8.3), (28, 0.0),
])
def test_known_values_from_the_published_table(raw, expected):
    """Spot checks against Lyman et al. 2016, Table 4."""
    responses = [0] * 7
    for i in range(raw):
        responses[i % 7] += 1
    assert sum(responses) == raw
    assert om.score(responses)["interval_score"] == pytest.approx(expected, abs=0.05)


def test_every_item_carries_its_koos_identifier():
    """
    The seven items are a named subset of the full KOOS. Keeping the codes means
    a score recorded here can be traced back to the parent instrument.
    """
    codes = [item.code for item in om.KOOS_JR_ITEMS]
    assert codes == ["S1", "P2", "P3", "P5", "P6", "A4", "A6"]


def test_the_scale_has_five_options_running_none_to_extreme():
    assert [o["value"] for o in om.RESPONSE_OPTIONS] == [0, 1, 2, 3, 4]
    assert om.RESPONSE_OPTIONS[0]["label"] == "None"
    assert om.RESPONSE_OPTIONS[-1]["label"] == "Extreme"


# ---------------------------------------------------------------------------
# Refusing what cannot be scored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("responses", [
    [0, 0, 0, 0, 0, 0],           # one short
    [0, 0, 0, 0, 0, 0, 0, 0],     # one too many
    [],
])
def test_a_partly_completed_form_is_refused(responses):
    """
    Not imputed. There is no published rule for a missing KOOS-JR item, and a
    guess would be indistinguishable from an answer.
    """
    with pytest.raises(ValueError, match="all 7 answers"):
        om.score(responses)


@pytest.mark.parametrize("bad", [5, -1, 100])
def test_answers_outside_the_scale_are_refused(bad):
    with pytest.raises(ValueError, match="scale runs"):
        om.score([bad, 0, 0, 0, 0, 0, 0])


def test_a_boolean_is_not_an_answer():
    """bool subclasses int, so True would otherwise quietly score as 'Mild'."""
    with pytest.raises(ValueError, match="whole number"):
        om.score([True, 0, 0, 0, 0, 0, 0])


def test_an_unknown_instrument_says_what_is_available():
    with pytest.raises(ValueError, match=r"oxford|Oxford|koos_jr"):
        om.score(PERFECT, instrument="oxford_knee_score")


# ---------------------------------------------------------------------------
# Change, which is the part that carries meaning
# ---------------------------------------------------------------------------

def test_the_first_score_is_a_baseline_not_a_change():
    change = om.change(58.0, None)
    assert change["direction"] == "first"
    assert change["delta"] is None
    assert change["meaningful"] is False


def test_a_change_the_instrument_cannot_detect_reads_as_no_change():
    """
    Three points is inside the measurement noise. Reporting it as "slightly
    worse" would invite a conclusion the questionnaire cannot support.
    """
    change = om.change(58.0, 61.0)
    assert change["direction"] == "unchanged"
    assert change["meaningful"] is False


def test_an_improvement_past_the_mcid_is_one_the_patient_would_feel():
    change = om.change(58.0, 58.0 - om.MCID)
    assert change["direction"] == "improved"
    assert change["meaningful"] is True


def test_a_fall_past_the_mcid_is_worth_raising():
    change = om.change(58.0, 58.0 + om.MCID)
    assert change["direction"] == "declined"
    assert change["meaningful"] is True
    assert "physiotherapist" in change["summary"]


def test_a_real_but_small_move_is_reported_without_overstating_it():
    """Between MDC and MCID: the instrument can see it; the patient may not."""
    change = om.change(58.0, 58.0 - (om.MDC + om.MCID) / 2)
    assert change["direction"] == "improved"
    assert change["meaningful"] is False


# ---------------------------------------------------------------------------
# When to ask again
# ---------------------------------------------------------------------------

def now():
    return datetime.now(timezone.utc)


def test_the_first_questionnaire_is_always_due():
    plan = om.schedule(None)
    assert plan["due"] is True and plan["can_record"] is True


def test_asking_again_the_next_morning_is_refused_with_a_reason():
    plan = om.schedule(now() - timedelta(days=1))
    assert plan["can_record"] is False
    assert plan["due"] is False
    assert str(om.MIN_INTERVAL_DAYS) in plan["reason"]


def test_between_the_floor_and_the_interval_it_is_allowed_but_not_asked_for():
    plan = om.schedule(now() - timedelta(days=20))
    assert plan["can_record"] is True
    assert plan["due"] is False


def test_after_the_interval_it_is_due():
    plan = om.schedule(now() - timedelta(days=40))
    assert plan["due"] is True and plan["can_record"] is True
    assert plan["days_since"] == 40


def test_a_naive_timestamp_is_read_as_the_utc_it_was_stored_as():
    """SQLite hands back naive datetimes; treating them as local would shift the interval."""
    naive = (now() - timedelta(days=30)).replace(tzinfo=None)
    assert om.schedule(naive)["due"] is True


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="outcome tests need sqlalchemy")

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier

import main
from db import SessionLocal
from models import OutcomeScore, Patient

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


def submit(client, tok, responses=MIDDLING, knee_side="right"):
    return client.post(
        "/me/outcome-scores",
        json={"responses": responses, "knee_side": knee_side},
        headers=bearer(tok),
    )


def backdate(email, days, side="right"):
    """Move a patient's scores back in time, so an interval can be crossed."""
    with SessionLocal() as db:
        patient = db.query(Patient).filter_by(email=email).one()
        rows = (db.query(OutcomeScore)
                  .filter_by(patient_id=patient.id, knee_side=side)
                  .all())
        for row in rows:
            row.recorded_at = row.recorded_at - timedelta(days=days)
        db.commit()


# ── The questionnaire ──────────────────────────────────────────────────────

def test_the_questionnaire_is_served_so_the_form_has_one_source_of_truth(client):
    body = client.get("/outcome-measures").json()
    assert body["instrument"] == "koos_jr"
    assert len(body["items"]) == 7
    assert len(body["response_options"]) == 5
    assert body["higher_is_better"] is True
    assert body["min_score"] == 0 and body["max_score"] == 100


def test_the_questionnaire_needs_no_account(client):
    """It is not private, and login.html has a guest route."""
    assert client.get("/outcome-measures").status_code == 200


def test_an_acl_patient_is_told_the_instrument_is_a_poor_fit(client, token):
    """
    KOOS-JR asks nothing about sport, pivoting under load, or confidence in the
    knee — which is most of what has gone wrong after a reconstruction. Surfaced
    rather than enforced: a score with a caveat beats no score at all.
    """
    client.patch("/me/surgery", json={"surgery_date": "2026-08-01", "surgery_type": "acl"},
                 headers=bearer(token))
    assert client.get("/outcome-measures", headers=bearer(token)).json()["caveat"] is not None


def test_a_replacement_patient_gets_no_caveat(client, token):
    """The population the instrument was built and validated for."""
    client.patch("/me/surgery", json={"surgery_date": "2026-08-01", "surgery_type": "tkr"},
                 headers=bearer(token))
    assert client.get("/outcome-measures", headers=bearer(token)).json()["caveat"] is None


# ── Recording ──────────────────────────────────────────────────────────────

def test_a_completed_questionnaire_is_scored_and_kept(client, token):
    r = submit(client, token)
    assert r.status_code == 201
    body = r.json()
    assert body["raw_sum"] == 10
    assert body["interval_score"] == 56.9
    assert body["band"] == "fair"
    assert body["responses"] == MIDDLING
    assert body["change"]["direction"] == "first"


def test_the_answers_are_kept_alongside_the_score(client, token):
    """
    The score is what gets charted; the answers are the evidence. A corrected
    lookup table or a second instrument can be recomputed from them, and neither
    is possible from a total.
    """
    submit(client, token)
    with SessionLocal() as db:
        row = db.query(OutcomeScore).one()
        assert row.raw_sum == sum(MIDDLING)
        assert row.interval_score == pytest.approx(56.9, abs=0.05)


def test_recording_requires_an_account(client):
    assert client.post("/me/outcome-scores", json={"responses": MIDDLING}).status_code == 401


@pytest.mark.parametrize("responses", [
    [0, 0, 0, 0, 0, 0],            # short
    [0, 0, 0, 0, 0, 0, 0, 0],      # long
    [5, 0, 0, 0, 0, 0, 0],         # off the scale
    [-1, 0, 0, 0, 0, 0, 0],
])
def test_an_unscoreable_form_is_refused(client, token, responses):
    assert submit(client, token, responses).status_code == 422


def test_answering_again_the_next_day_is_refused(client, token):
    """
    Answered weekly this becomes a mood reading: genuine week-to-week movement
    is smaller than the instrument can detect, and a chart of that noise invites
    conclusions it cannot support.
    """
    assert submit(client, token).status_code == 201
    r = submit(client, token)
    assert r.status_code == 409
    assert "noise" in r.json()["detail"]


def test_answering_after_the_minimum_interval_is_allowed(client, token):
    submit(client, token)
    backdate("ada@example.com", om.MIN_INTERVAL_DAYS + 1)
    assert submit(client, token, PERFECT).status_code == 201


def test_the_second_score_carries_the_change_from_the_first(client, token):
    submit(client, token, MIDDLING)                 # 56.9
    backdate("ada@example.com", 30)
    change = submit(client, token, PERFECT).json()["change"]   # 100.0
    assert change["direction"] == "improved"
    assert change["delta"] == pytest.approx(43.1, abs=0.05)
    assert change["meaningful"] is True


# ── Two knees are two questionnaires ───────────────────────────────────────

def test_the_other_knee_can_be_answered_in_the_same_sitting(client, token):
    """
    KOOS-JR asks about "your knee", singular. Someone with two bad knees answers
    twice, and the interval that stops repeat answers is per knee.
    """
    assert submit(client, token, knee_side="right").status_code == 201
    assert submit(client, token, knee_side="left").status_code == 201


def test_one_knees_score_is_never_compared_against_the_others(client, token):
    submit(client, token, WORST, knee_side="right")        # 0.0
    left = submit(client, token, PERFECT, knee_side="left")  # 100.0
    # Not "up 100 points" — this is the left knee's first answer.
    assert left.json()["change"]["direction"] == "first"


def test_each_knee_gets_its_own_series(client, token):
    submit(client, token, WORST, knee_side="right")
    submit(client, token, PERFECT, knee_side="left")

    series = client.get("/me/progress", headers=bearer(token)).json()["outcome_measures"]
    by_side = {s["knee_side"]: s for s in series}
    assert by_side["right"]["latest"] == 0.0
    assert by_side["left"]["latest"] == 100.0


# ── History ────────────────────────────────────────────────────────────────

def test_history_comes_back_newest_first_with_the_schedule(client, token):
    submit(client, token, MIDDLING)
    backdate("ada@example.com", 30)
    submit(client, token, PERFECT)

    body = client.get("/me/outcome-scores", headers=bearer(token)).json()
    assert body["count"] == 2
    assert body["scores"][0]["interval_score"] == 100.0
    assert body["schedule"]["can_record"] is False      # just answered


def test_history_is_empty_and_due_for_someone_who_has_never_answered(client, token):
    body = client.get("/me/outcome-scores", headers=bearer(token)).json()
    assert body["count"] == 0
    assert body["schedule"]["due"] is True
    assert body["scores"] == []


def test_history_can_be_narrowed_to_one_knee(client, token):
    submit(client, token, WORST, knee_side="right")
    submit(client, token, PERFECT, knee_side="left")

    body = client.get("/me/outcome-scores", params={"knee_side": "left"}, headers=bearer(token)).json()
    assert body["count"] == 1
    assert body["scores"][0]["knee_side"] == "left"


def test_one_patient_cannot_see_anothers_scores(client, token):
    submit(client, token)
    other = client.post(
        "/auth/register", json={"email": "bob@example.com", "password": PASSWORD}
    ).json()["access_token"]
    assert client.get("/me/outcome-scores", headers=bearer(other)).json()["count"] == 0


# ── Progress ───────────────────────────────────────────────────────────────

def test_progress_carries_the_latest_score_alongside_what_was_measured(client, token):
    submit(client, token, MIDDLING)
    summary = client.get("/me/progress", headers=bearer(token)).json()["summary"]
    assert summary["latest_outcome_score"] == 56.9
    assert summary["outcome_band"] == "fair"


def test_progress_is_unbothered_by_a_patient_who_has_never_answered(client, token):
    body = client.get("/me/progress", headers=bearer(token)).json()
    assert body["outcome_measures"] == []
    assert body["summary"]["latest_outcome_score"] is None


def test_the_baseline_survives_a_narrow_time_range(client, token):
    """
    The series is deliberately not clipped to the requested window. A
    questionnaire answered monthly has its baseline further back than 30 days,
    and a trend that dropped its own starting point would be worse than none.
    """
    submit(client, token, WORST)
    backdate("ada@example.com", 120)
    submit(client, token, PERFECT)

    series = client.get("/me/progress", params={"days": 30}, headers=bearer(token)).json()["outcome_measures"]
    assert series[0]["count"] == 2
    assert series[0]["baseline"] == 0.0
    assert series[0]["change_from_baseline"]["direction"] == "improved"


# ── Triage ─────────────────────────────────────────────────────────────────

def test_a_knee_going_backwards_raises_a_flag(client, token):
    """
    The case nothing else here can see: the exercises can be going perfectly and
    the knee still getting worse to live with.
    """
    submit(client, token, PERFECT)                  # 100
    backdate("ada@example.com", 40)
    submit(client, token, MIDDLING)                 # 56.9

    flags = client.get("/me/flags", headers=bearer(token)).json()
    codes = [f["code"] for f in flags["flags"]]
    assert "outcome_declining" in codes


def test_a_small_fall_raises_nothing(client, token):
    """Inside the MCID it is not a signal, and a flag nobody trusts gets muted."""
    submit(client, token, [0, 0, 0, 0, 0, 0, 0])    # 100
    backdate("ada@example.com", 40)
    submit(client, token, [1, 0, 0, 0, 0, 0, 0])    # 91.975 — an 8-point fall

    flags = client.get("/me/flags", headers=bearer(token)).json()
    assert "outcome_declining" not in [f["code"] for f in flags["flags"]]


def test_a_single_score_raises_nothing(client, token):
    submit(client, token, WORST)
    flags = client.get("/me/flags", headers=bearer(token)).json()
    assert "outcome_declining" not in [f["code"] for f in flags["flags"]]


def test_a_slow_slide_is_caught_even_though_each_step_is_small(client, token):
    """
    Measured against the best score so far, not the previous one. Someone going
    71 → 62 → 53 never trips a previous-to-latest comparison, and is exactly the
    patient this rule exists for.
    """
    submit(client, token, [0, 0, 0, 0, 0, 0, 0])    # 100.0
    backdate("ada@example.com", 40)
    submit(client, token, [1, 0, 0, 0, 0, 0, 0])    # 91.975
    backdate("ada@example.com", 40)
    submit(client, token, [1, 1, 0, 0, 0, 0, 0])    # 84.775 — 15.2 below the best

    flags = client.get("/me/flags", headers=bearer(token)).json()
    assert "outcome_declining" in [f["code"] for f in flags["flags"]]


# ── The clinician's view ───────────────────────────────────────────────────

def clinician_token(client, email="phys@example.com"):
    return client.post(
        "/clinician/register", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]


def link(client, clin_tok, patient_tok):
    code = client.post("/clinician/invites", json={}, headers=bearer(clin_tok)).json()["code"]
    client.post("/me/clinicians/redeem", json={"code": code}, headers=bearer(patient_tok))


def test_a_linked_clinician_sees_the_scores(client, token):
    submit(client, token, MIDDLING)
    clin = clinician_token(client)
    link(client, clin, token)

    with SessionLocal() as db:
        patient_id = db.query(Patient).filter_by(email="ada@example.com").one().id

    body = client.get(f"/clinician/patients/{patient_id}/outcome-scores", headers=bearer(clin)).json()
    assert body["count"] == 1
    assert body["scores"][0]["interval_score"] == 56.9


def test_an_unlinked_clinician_sees_nothing(client, token):
    submit(client, token)
    clin = clinician_token(client, "stranger@example.com")

    with SessionLocal() as db:
        patient_id = db.query(Patient).filter_by(email="ada@example.com").one().id

    # 404 rather than 403: confirming the id exists confirms the patient does.
    r = client.get(f"/clinician/patients/{patient_id}/outcome-scores", headers=bearer(clin))
    assert r.status_code == 404


def test_the_caseload_carries_the_score_and_its_direction(client, token):
    """
    On the list itself, because a score that has fallen since last month is the
    best single reason to open a record — and it is invisible in adherence,
    which often looks fine right up until someone gives up.
    """
    submit(client, token, PERFECT)
    backdate("ada@example.com", 40)
    submit(client, token, MIDDLING)

    clin = clinician_token(client)
    link(client, clin, token)

    entry = client.get("/clinician/patients", headers=bearer(clin)).json()["patients"][0]
    assert entry["latest_outcome_score"] == 56.9
    assert entry["outcome_change"] == pytest.approx(-43.1, abs=0.05)
    assert entry["worst_flag"] in ("warning", "urgent")


def test_a_clinician_cannot_answer_on_the_patients_behalf(client, token):
    """
    The whole value of a PROM is whose report it is. A score filled in by
    somebody else is an opinion wearing a registry-comparable number.
    """
    clin = clinician_token(client)
    link(client, clin, token)
    assert client.post("/me/outcome-scores", json={"responses": MIDDLING},
                       headers=bearer(clin)).status_code == 401


# ── It is never a gate ─────────────────────────────────────────────────────

def test_nothing_is_withheld_from_someone_who_never_answers(client, token):
    """
    No reminder here can become a wall. A patient who does not want to fill in
    seven questions keeps every other part of the app.
    """
    assert client.get("/me/progress", headers=bearer(token)).status_code == 200
    assert client.get("/sessions", headers=bearer(token)).status_code == 200
    assert client.get("/exercises", params={"surgery_type": "none"}).status_code == 200


def test_a_score_does_not_move_a_movement_ceiling(client, token):
    """
    The ceiling comes from the radiograph and the surgical protocol. A
    questionnaire — however standard — is not evidence about what a joint can
    safely withstand, and nothing downstream of a score is allowed to touch it.
    """
    before = client.get("/exercises", params={"surgery_type": "tkr", "weeks_post_op": 6}).json()
    submit(client, token, WORST)
    after = client.get("/exercises", params={"surgery_type": "tkr", "weeks_post_op": 6}).json()
    assert before["max_angle"] == after["max_angle"]
