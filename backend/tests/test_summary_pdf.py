"""
The one-page appointment summary.

Tests read the generated PDF back as text, so they assert what somebody holding
the sheet can actually see. Asserting that `render()` was *asked* to draw a
number would pass just as happily when the number lands under the footer, off
the right edge, or on a second page nobody prints.

Two properties carry most of the weight here:

  * **It is one page**, whatever it is given. A summary is a summary because it
    fits on the thing you hand across a desk.
  * **Nothing disappears silently.** Anything dropped for want of room has to be
    named on the page, because the reader cannot tell an exercise the patient
    never did from one that fell off the bottom.

Run:  python -m pytest backend/tests -q
"""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("reportlab", reason="the summary needs reportlab")
pypdf = pytest.importorskip("pypdf", reason="reading the PDF back needs pypdf")

import summary_pdf as sp

NOW = datetime(2026, 9, 4, 10, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read(pdf: bytes):
    """(page count, all text on page one)."""
    reader = pypdf.PdfReader(__import__("io").BytesIO(pdf))
    return len(reader.pages), reader.pages[0].extract_text()


def points(n=6, start=60.0, step=5.0, limit=105):
    return [(date(2026, 7, 1) + timedelta(days=7 * i), start + step * i, limit) for i in range(n)]


def rom(name="Heel slides", best=88.0, latest=88.0, limit=105, pts=None):
    return sp.RomRow(name, best, latest, limit, len(pts or []), pts or [])


def data(**kw):
    base = {
        "generated_at": NOW,
        "range_days": 90,
        "patient_name": "Aftab Sheikh",
        "surgery_type": "tkr",
        "surgery_date": date(2026, 6, 10),
        "weeks_post_op": 12,
        "knee_side": "left",
        "ceiling_deg": 105,
        "has_plan": True,
        "plan_reviewed": True,
        "reviewed_by": "Dr R Mehta",
        "reviewed_at": datetime(2026, 8, 2, tzinfo=timezone.utc),
        "sessions": 34,
        "active_days": 21,
        "current_streak_days": 5,
        "longest_streak_days": 9,
        "rom": [rom(pts=points())],
    }
    base.update(kw)
    return sp.SummaryData(**base)


# ---------------------------------------------------------------------------
# It is one page
# ---------------------------------------------------------------------------

def test_a_normal_summary_is_one_page():
    pages, _ = read(sp.render(data()))
    assert pages == 1


def test_a_patient_with_a_great_deal_of_history_still_gets_one_page():
    """
    Twenty exercises, twelve flags and two questionnaires. Nothing about this is
    unusual after a year of rehab, and it must not become a four-page document.
    """
    pages, _ = read(sp.render(data(
        rom=[rom(f"Exercise number {i}", 90 - i, 88 - i, 105, points()) for i in range(20)],
        flags=[sp.FlagRow("warning", f"Something worth mentioning, number {i}, at some length.")
               for i in range(12)],
        outcomes=[
            sp.OutcomeRow("KOOS-JR", side, 56.0, 71.0, "Moderate", "Down 15 points.", date(2026, 9, 1))
            for side in ("left", "right")
        ],
        unverified_sessions=4,
        demo_mode=True,
    )))
    assert pages == 1


def test_an_empty_account_still_renders_a_page():
    """Someone who signed up and did nothing yet gets a valid sheet, not a crash."""
    pages, text = read(sp.render(sp.SummaryData(generated_at=NOW, range_days=90)))
    assert pages == 1
    assert "No verified range-of-motion measurements" in text


# ---------------------------------------------------------------------------
# Nothing disappears silently
# ---------------------------------------------------------------------------

# Around thirty rows fit beneath the chart, which no real prescription
# approaches — these use forty to reach the limit deliberately.
MORE_THAN_FITS = 40


def test_exercises_that_do_not_fit_are_counted_on_the_page():
    rows = [rom(f"Exercise number {i:02d}", 90, 88, 105, points() if i == 0 else [])
            for i in range(MORE_THAN_FITS)]
    pages, text = read(sp.render(data(rom=rows)))

    assert pages == 1
    assert "not listed here" in text, "a truncated table must say it was truncated"


def test_the_most_practised_exercises_are_the_ones_kept():
    """
    The caller sorts most-measured first, so truncation must take from the tail.
    Dropping the exercise with the most history would invert the whole point.
    """
    rows = [rom(f"Exercise number {i:02d}", 90, 88, 105, points() if i == 0 else [])
            for i in range(MORE_THAN_FITS)]
    _, text = read(sp.render(data(rom=rows)))

    assert "Exercise number 00" in text
    assert f"Exercise number {MORE_THAN_FITS - 1:02d}" not in text


def test_a_short_table_says_nothing_about_truncation():
    _, text = read(sp.render(data(rom=[rom(pts=points()), rom("Straight-leg raise", 41, 38, 105)])))
    assert "not listed here" not in text


# ---------------------------------------------------------------------------
# What has to be on it
# ---------------------------------------------------------------------------

def test_the_headline_facts_are_present():
    _, text = read(sp.render(data()))

    assert "Aftab Sheikh" in text
    assert "Covering the last 90 days" in text
    assert "12" in text                      # weeks post-op
    assert "105" in text                     # the ceiling
    assert "34" in text and "21" in text     # sessions, active days


def test_the_measurement_caveat_is_always_printed():
    """On paper there is no tooltip. It is printed or it does not exist."""
    _, text = read(sp.render(data()))
    assert "not a goniometer" in text
    assert "not a diagnosis" in text


def test_an_unreviewed_plan_says_so():
    _, text = read(sp.render(data(plan_reviewed=False, reviewed_by=None, reviewed_at=None)))
    assert "has not been reviewed" in text


def test_a_reviewed_plan_names_who_reviewed_it():
    _, text = read(sp.render(data()))
    assert "Dr R Mehta" in text
    assert "has not been reviewed" not in text


def test_demo_mode_is_disclosed():
    _, text = read(sp.render(data(demo_mode=True)))
    assert "demonstration mode" in text


def test_unverified_sessions_are_explained_not_just_counted():
    """
    The number alone invites the wrong conclusion — that some sessions did not
    count. They counted for adherence and not for the angles, and the difference
    is the whole reason the figure is disclosed.
    """
    _, text = read(sp.render(data(unverified_sessions=3)))
    assert "3 sessions had an unverified camera angle" in text
    assert "adherence" in text


def test_one_unverified_session_is_not_pluralised():
    _, text = read(sp.render(data(unverified_sessions=1)))
    assert "1 session had" in text


def test_flags_appear_under_a_heading_that_says_what_they_are_for():
    _, text = read(sp.render(data(
        flags=[sp.FlagRow("warning", "KOOS-JR down 15 points (71 to 56) over 4 weeks.")],
    )))
    assert "WORTH RAISING" in text
    assert "KOOS-JR down 15 points" in text


def test_the_outcome_caveat_is_printed_when_the_instrument_does_not_fit():
    """
    KOOS-JR after an ACL reconstruction is a score that reads better than the
    knee is. The sheet has to say so where it prints the number.
    """
    _, text = read(sp.render(data(
        surgery_type="acl",
        outcomes=[sp.OutcomeRow("KOOS-JR", "left", 82.0, 60.0, "Good", "Up 22 points.",
                                date(2026, 9, 1))],
        outcome_caveat="KOOS-JR was developed for knee osteoarthritis and joint replacement.",
    )))
    assert "developed for knee osteoarthritis" in text


def test_the_caveat_is_not_printed_when_there_is_no_score_to_qualify():
    _, text = read(sp.render(data(
        outcomes=[], outcome_caveat="KOOS-JR was developed for knee osteoarthritis.",
    )))
    assert "developed for knee osteoarthritis" not in text


# ---------------------------------------------------------------------------
# Thin and awkward data
# ---------------------------------------------------------------------------

def test_a_single_measurement_does_not_break_the_chart():
    """A one-point series has zero date span, which is a division waiting to happen."""
    pages, text = read(sp.render(data(rom=[rom(pts=points(n=1))])))
    assert pages == 1
    assert "Heel slides" in text


def test_a_flat_series_still_gets_a_readable_scale():
    """Every reading identical collapses the y-range to zero unless it is padded."""
    pts = [(date(2026, 7, 1) + timedelta(days=i), 90.0, 90) for i in range(5)]
    pages, _ = read(sp.render(data(rom=[rom(best=90, latest=90, limit=90, pts=pts)])))
    assert pages == 1


def test_an_exercise_with_no_verified_measurements_says_so_in_the_chart():
    _, text = read(sp.render(data(rom=[rom(pts=[])])))
    assert "No verified measurements" in text


def test_a_missing_name_does_not_leave_the_sheet_anonymous():
    _, text = read(sp.render(data(patient_name=None)))
    assert "Patient" in text


def test_missing_context_renders_as_a_dash_not_as_none():
    _, text = read(sp.render(data(
        surgery_type=None, surgery_date=None, weeks_post_op=None, knee_side=None, ceiling_deg=None,
    )))
    assert "None" not in text


def test_pain_is_omitted_entirely_when_it_was_never_recorded():
    _, text = read(sp.render(data(latest_pain_after=None, mean_pain_change=None)))
    assert "PAIN" not in text


def test_a_rise_in_pain_is_signed_so_the_direction_is_unambiguous():
    _, text = read(sp.render(data(latest_pain_after=6, mean_pain_change=1.4)))
    assert "+1.4" in text


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def test_a_name_the_font_cannot_draw_is_flagged_rather_than_silently_blanked():
    """
    Base-14 PDF fonts stop at Latin-1. Dropping the characters would leave a
    plausible-looking name that is not the patient's; the sheet says what
    happened and how to fix it.
    """
    _, text = read(sp.render(data(patient_name="अफ़ताब शेख़")))
    assert "could not be drawn" in text
    assert sp.FONT_ENV in text


def test_a_latin_name_with_accents_needs_no_apology():
    _, text = read(sp.render(data(patient_name="Renée Fauré")))
    assert "Renée Fauré" in text
    assert "could not be drawn" not in text


def test_an_unreadable_font_setting_falls_back_instead_of_failing_the_download(monkeypatch, tmp_path):
    """A misconfigured font must not be the reason a patient cannot print anything."""
    missing = tmp_path / "not-a-font.ttf"
    monkeypatch.setenv(sp.FONT_ENV, str(missing))
    sp._registered.pop(str(missing), None)

    pages, text = read(sp.render(data()))
    assert pages == 1
    assert "Aftab Sheikh" in text


# ---------------------------------------------------------------------------
# The document itself
# ---------------------------------------------------------------------------

def test_the_pdf_declares_a_useful_title():
    """It becomes the browser tab and the filename suggestion in most readers."""
    reader = pypdf.PdfReader(__import__("io").BytesIO(sp.render(data())))
    assert "Aftab Sheikh" in (reader.metadata.title or "")


def test_it_is_a_real_pdf():
    assert sp.render(data()).startswith(b"%PDF-")


# ---------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------
#
# Three routes render the same sheet for three different readers, and the
# interesting difference between them is what each is allowed to put at the top
# of the page. A share link is a bearer token: anybody holding it is inside, so
# what it discloses is the thing worth pinning down.

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="the endpoints need sqlalchemy")

import io

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier

import main
from db import SessionLocal

PASSWORD = "correct-horse-battery"
EMAIL = "ada@example.com"


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
def token(client):
    return client.post(
        "/auth/register",
        json={"email": EMAIL, "password": PASSWORD, "display_name": "Ada Lovelace"},
    ).json()["access_token"]


def text_of(response):
    reader = pypdf.PdfReader(io.BytesIO(response.content))
    return len(reader.pages), reader.pages[0].extract_text()


def test_the_patient_can_download_their_own_summary(client, token):
    response = client.get("/me/summary.pdf", headers=bearer(token))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    pages, text = text_of(response)
    assert pages == 1
    assert "Ada Lovelace" in text


def test_the_summary_needs_an_account(client):
    assert client.get("/me/summary.pdf").status_code == 401


def test_it_arrives_as_a_download_named_for_the_patient_and_the_day(client, token):
    disposition = client.get("/me/summary.pdf", headers=bearer(token)).headers["content-disposition"]

    assert disposition.startswith("attachment;")
    assert "ada-lovelace" in disposition
    assert ".pdf" in disposition


def test_a_clinical_document_is_not_left_in_a_shared_cache(client, token):
    cache = client.get("/me/summary.pdf", headers=bearer(token)).headers["cache-control"]
    assert "no-store" in cache and "private" in cache


def test_a_name_that_could_forge_a_header_cannot(client):
    """
    A display name is user input on its way into a response header. The ASCII
    filename comes from a strict allow-list, and the UTF-8 one is percent-encoded,
    so a newline arrives as %0D%0A and stays inert. Control characters are
    stripped before encoding too, so what a browser decodes for its save dialog
    is a filename rather than a smuggled header.
    """
    tok = client.post("/auth/register", json={
        "email": "mallory@example.com", "password": PASSWORD,
        "display_name": 'evil"\r\nX-Injected: yes',
    }).json()["access_token"]

    disposition = client.get("/me/summary.pdf", headers=bearer(tok)).headers["content-disposition"]

    assert "\r" not in disposition and "\n" not in disposition
    assert "%0D" not in disposition and "%0A" not in disposition
    # The quoted ASCII filename must contain no quote of its own to close early.
    assert disposition.split('filename="')[1].split('"')[0].count('"') == 0


def test_a_share_link_renders_the_same_sheet_without_an_account(client, token):
    share = client.post("/me/share-links", json={"label": "Physio", "days": 30},
                        headers=bearer(token)).json()

    response = client.get(f"/share/{share['token']}/summary.pdf")

    assert response.status_code == 200
    pages, text = text_of(response)
    assert pages == 1
    assert "Ada Lovelace" in text


def test_a_share_link_never_discloses_the_email_address(client, token):
    """
    The link is the credential, so whoever holds it is inside. An account
    identifier printed on the sheet turns a read-only link into the first half
    of an attack on the account itself.
    """
    from models import Patient
    with SessionLocal() as db:
        db.query(Patient).filter_by(email=EMAIL).one().display_name = None
        db.commit()

    share = client.post("/me/share-links", json={"days": 30}, headers=bearer(token)).json()
    _, text = text_of(client.get(f"/share/{share['token']}/summary.pdf"))

    assert EMAIL not in text
    assert "Patient" in text


def test_a_revoked_share_link_stops_serving_the_pdf(client, token):
    created = client.post("/me/share-links", json={"days": 30}, headers=bearer(token)).json()
    client.delete(f"/me/share-links/{created['id']}", headers=bearer(token))

    assert client.get(f"/share/{created['token']}/summary.pdf").status_code == 404


def test_an_invented_share_token_is_a_flat_404(client):
    assert client.get("/share/not-a-real-token/summary.pdf").status_code == 404


def test_a_clinician_can_download_a_summary_for_a_patient_on_their_caseload(client, token):
    clinician = client.post("/clinician/register", json={
        "email": "physio@clinic.example", "password": PASSWORD,
        "display_name": "Dr R Mehta", "registration_number": "PT-4471",
    }).json()["access_token"]
    code = client.post("/clinician/invites", json={"patient_label": "Ada L", "days": 7},
                       headers=bearer(clinician)).json()["code"]
    client.post("/me/clinicians/redeem", json={"code": code}, headers=bearer(token))

    patient_id = client.get("/auth/me", headers=bearer(token)).json()["id"]
    response = client.get(f"/clinician/patients/{patient_id}/summary.pdf", headers=bearer(clinician))

    assert response.status_code == 200
    pages, text = text_of(response)
    assert pages == 1
    assert "Ada Lovelace" in text


def test_a_clinician_cannot_download_a_summary_for_someone_elses_patient(client, token):
    clinician = client.post("/clinician/register", json={
        "email": "stranger@clinic.example", "password": PASSWORD,
        "display_name": "Dr Nobody", "registration_number": "PT-0000",
    }).json()["access_token"]
    patient_id = client.get("/auth/me", headers=bearer(token)).json()["id"]

    response = client.get(f"/clinician/patients/{patient_id}/summary.pdf", headers=bearer(clinician))
    assert response.status_code == 404


def test_a_patient_token_is_not_a_clinician_token(client, token):
    patient_id = client.get("/auth/me", headers=bearer(token)).json()["id"]
    response = client.get(f"/clinician/patients/{patient_id}/summary.pdf", headers=bearer(token))
    assert response.status_code in (401, 403)
