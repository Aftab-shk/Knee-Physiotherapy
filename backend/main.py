"""
Physiotherapy AI — FastAPI Backend v2
======================================
Endpoints
---------
  GET  /              → redirects to /docs (Swagger UI)
  GET  /health        → liveness + model status
  POST /analyse-xray  → KL grade + rehab prescription from X-ray
  GET  /exercises     → exercise list by surgery type + weeks (no X-ray needed)

Run locally:
  uvicorn main:app --reload --port 8000

Environment variables:
  MODEL_PATH    path to trained weights (default: model/efficientnet_b4_kl_v2.pt)
  CORS_ORIGINS  comma-separated allow-list (default: localhost:5173, localhost:3000)
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date, timedelta, timezone
from itertools import pairwise
from typing import TYPE_CHECKING, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from model.image_checks import validate_image
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import auth
import triage
from clinical_logic import build_prescription, get_exercises_only
from db import get_db, init_db
from exercise_protocols import all_exercises
from models import (
    CareLink,
    Clinician,
    ExerciseSession,
    ExerciseSet,
    Patient,
    Prescription,
    PrescriptionAudit,
    ShareLink,
)
from models import utcnow as models_utcnow
from schemas import (
    AdherenceDay,
    AnalyseXrayResponse,
    AuditEntry,
    CareLinkOut,
    Caseload,
    CaseloadEntry,
    CatalogueExercise,
    ClinicianOut,
    ClinicianRegister,
    ClinicianToken,
    ExerciseBreakdown,
    ExercisesResponse,
    FlagOut,
    HealthResponse,
    InviteCreate,
    InviteCreated,
    InviteOut,
    KneeSide,
    LoginRequest,
    PainPoint,
    PatientFlag,
    PatientFlags,
    PatientFlagsForClinician,
    PatientOut,
    PrescriptionDetail,
    PrescriptionEffective,
    PrescriptionHistory,
    PrescriptionSummaryForClinician,
    ProgressResponse,
    ProgressSummary,
    RedeemInvite,
    RegisterRequest,
    ReviewRequest,
    RomPoint,
    RomSeries,
    SessionHistory,
    SessionOut,
    SessionReport,
    SetRecord,
    SharedProgress,
    ShareLinkCreate,
    ShareLinkCreated,
    ShareLinkOut,
    SurgeryType,
    SurgeryUpdate,
    TokenResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("physio-backend")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover - typing only
    from model.inference import KneeClassifier

# model.inference imports torch, which costs seconds and hundreds of megabytes.
# Loading it lazily is what lets the API tests, the clinical logic and the
# schemas be imported without it — and what keeps the "no torch needed" claim in
# tests/test_api_security.py true.
classifier: "Optional[KneeClassifier]" = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    # Before the model: an instance that cannot reach its database should fail
    # at startup, not on the first request that tries to save something.
    init_db()
    logger.info("Loading KneeClassifier…")
    try:
        from model.inference import KneeClassifier
    except ImportError:
        # torch is not installed. Serve anyway: /health reports model_loaded
        # false and /analyse-xray already answers 503, which surfaces the
        # problem far more usefully than refusing to boot. It is also what lets
        # the API be tested without a 2 GB dependency.
        # Logged as an error, not an exception: the traceback for a missing
        # import adds nothing the message does not already say, and the
        # install instruction is the useful part.
        logger.error(  # noqa: TRY400
            "❌  torch is not installed — the API is up but cannot grade X-rays. "
            "Install it with: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        yield
        logger.info("Shutting down AI Knee Physiotherapy backend.")
        return

    classifier = KneeClassifier()
    if classifier.demo_mode:
        # Name the path actually searched, not a stale default that never
        # existed — MODEL_PATH resolves relative to backend/model/.
        logger.warning(
            "⚠️  DEMO MODE — no usable checkpoint at %s. Predictions are deterministic "
            "mocks, not readings. See backend/model/fetch_weights.py.",
            os.getenv("MODEL_PATH", "model/best_model.pth"),
        )
    else:
        # classifier.model_version is derived from the checkpoint; MODEL_VERSION
        # is only the fallback constant, so logging it here reported the wrong
        # model whenever real weights were loaded.
        logger.info(
            "✅  Model loaded — real inference active. Version: %s (calibrated=%s, ood_screening=%s)",
            classifier.model_version,
            classifier.calibration["calibrated"],
            classifier.calibration["reject_threshold"] is not None,
        )
    yield
    logger.info("Shutting down AI Knee Physiotherapy backend.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# The "null" origin was removed deliberately. It is what a sandboxed iframe and
# a file:// page send, so allowing it alongside allow_credentials lets any local
# HTML file — or a sandboxed iframe on a hostile page — make credentialed
# requests to this API. Serve the frontend over http:// instead (the README says
# to, and tracker.html needs a real origin for camera permissions anyway).
CORS_ORIGINS = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,"
        "http://localhost:8080,http://127.0.0.1:8080,"
        "http://localhost:5500,http://127.0.0.1:5500,"
        "http://localhost:5501,http://127.0.0.1:5501",
    ).split(",") if o.strip()
]

if "null" in CORS_ORIGINS or "*" in CORS_ORIGINS:
    logger.warning(
        "CORS_ORIGINS contains %r. Combined with credentials this allows any "
        "sandboxed iframe or file:// page to call this API.",
        "null" if "null" in CORS_ORIGINS else "*",
    )

# Request-size ceiling, enforced while streaming rather than after buffering.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# Sliding-window rate limit for the inference endpoint. A B4 forward pass on CPU
# is ~300 ms, so an unthrottled endpoint is trivially exhausted.
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_S = int(os.getenv("RATE_LIMIT_WINDOW_S", "60"))

# Bucket-table bounds. Sweeping starts at the soft cap; above the hard cap the
# oldest entries are evicted outright so memory cannot grow with attacker-chosen
# source addresses.
_RATE_BUCKET_SOFT_CAP = 4096
_RATE_BUCKET_HARD_CAP = 8192

app = FastAPI(
    title       = "AI Knee Physiotherapy API",
    description = (
        "Knee X-ray KL grading + personalised rehab exercise prescription. "
        "⚠️ For informational purposes only — not a substitute for professional medical advice."
    ),
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ORIGINS,
    # No cookies or Authorization headers are used by this API today. Turning
    # credentials off means a stolen origin cannot ride a browser session, and
    # it removes the "null" origin footgun entirely.
    allow_credentials = False,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    # Authorization carries the bearer token from POST /auth/login. It is not
    # a "credential" in the CORS sense — no cookie rides along — so
    # allow_credentials stays off and the "null" origin footgun stays shut.
    allow_headers     = ["Content-Type", "Authorization"],
    expose_headers    = ["X-Request-ID", "X-Response-Time"],
    max_age           = 600,
)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_rate_buckets: dict[str, deque] = defaultdict(deque)
_rate_lock = asyncio.Lock()


def _client_key(request: Request) -> str:
    """
    Client identity for rate limiting.

    X-Forwarded-For is only trusted when TRUST_PROXY is set — otherwise any
    caller could spoof the header and get a fresh bucket per request, which
    would make the limiter worse than useless.
    """
    if os.getenv("TRUST_PROXY", "").lower() in ("1", "true", "yes"):
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def enforce_rate_limit(request: Request) -> None:
    """
    Sliding-window limiter for the expensive endpoint.

    NOTE: state is per-process. The Dockerfile runs `--workers 2`, so the
    effective limit is RATE_LIMIT_REQUESTS x worker count. That is fine as a
    backstop against accidental hammering; a real deployment should enforce this
    at the reverse proxy (nginx `limit_req`) or with a shared Redis bucket.
    """
    if RATE_LIMIT_REQUESTS <= 0:
        return

    key = _client_key(request)
    now = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW_S

    async with _rate_lock:
        bucket = _rate_buckets[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_REQUESTS:
            retry_after = max(1, int(bucket[0] + RATE_LIMIT_WINDOW_S - now) + 1)
            logger.warning("Rate limit hit by %s (%d in %ds)", key, len(bucket), RATE_LIMIT_WINDOW_S)
            raise HTTPException(
                status_code = status.HTTP_429_TOO_MANY_REQUESTS,
                detail      = f"Too many requests. Try again in {retry_after}s.",
                headers     = {"Retry-After": str(retry_after)},
            )

        bucket.append(now)

        # Bound the bucket table.
        #
        # Entries are only trimmed on a key's own next request, so a client that
        # calls once and never returns leaves a non-empty deque behind. The
        # original cleanup deleted only already-empty buckets, which reclaimed
        # nothing at all. Sweeping expired entries fixes that, but is still not
        # enough on its own: a spread-out flood whose buckets are all *live*
        # sweeps to nothing and keeps growing, which turns a DoS defence into a
        # memory-exhaustion vector. So sweep first, then hard-evict by age.
        if len(_rate_buckets) > _RATE_BUCKET_SOFT_CAP:
            for k in list(_rate_buckets):
                stale = _rate_buckets[k]
                while stale and stale[0] < cutoff:
                    stale.popleft()
                if not stale:
                    del _rate_buckets[k]

            if len(_rate_buckets) > _RATE_BUCKET_HARD_CAP:
                # Evict least-recently-seen first. Dropping a bucket only ever
                # forgives a client, never penalises one, so the worst case is
                # that an attacker at this scale gets their window reset — by
                # which point the proxy-level limit is the real defence anyway.
                by_age = sorted(_rate_buckets, key=lambda k: _rate_buckets[k][-1])
                for k in by_age[: len(_rate_buckets) - _RATE_BUCKET_HARD_CAP]:
                    del _rate_buckets[k]
                logger.warning(
                    "Rate-limit table hit its cap (%d clients); evicted oldest entries. "
                    "Enforce rate limiting at the reverse proxy for traffic at this scale.",
                    _RATE_BUCKET_HARD_CAP,
                )


async def read_capped(upload: UploadFile, limit: int) -> bytes:
    """
    Read an upload, aborting as soon as it exceeds `limit`.

    The previous version did `await upload.read()` and checked the length
    afterwards, which buffers the whole body first — a handful of concurrent
    multi-gigabyte POSTs would exhaust memory before any check ran. Declared
    Content-Length is rejected up front, but it is only a hint, so the streaming
    loop is what actually enforces the ceiling.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail      = f"Image exceeds {limit // (1024 * 1024)} MB. Please compress and retry.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    t0         = time.perf_counter()
    response   = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"]    = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms:.0f}ms"
    if request.url.path not in ("/health", "/"):
        logger.info("%s %s → %d  (%.0f ms) [%s]",
                    request.method, request.url.path,
                    response.status_code, elapsed_ms, request_id)
    return response


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------
#
# These endpoints are declared `def`, not `async def`: SQLAlchemy's driver
# blocks, and FastAPI runs a sync endpoint in a threadpool rather than on the
# event loop. Argon2 is deliberately expensive too, which would stall every
# other request if it ran inline.

# One response for "no such account" and "wrong password" alike. Saying which
# one it was turns the login form into a way to test whether an address is
# registered — which, for a medical app, leaks that someone is a patient.
_BAD_CREDENTIALS = HTTPException(
    status_code = status.HTTP_401_UNAUTHORIZED,
    detail      = "Incorrect email or password.",
    headers     = {"WWW-Authenticate": "Bearer"},
)


def _issue_token(patient: Patient) -> TokenResponse:
    token, expires_in = auth.create_access_token(patient.id)
    return TokenResponse(
        access_token = token,
        expires_in   = expires_in,
        patient      = PatientOut.model_validate(patient),
    )


@app.post(
    "/auth/register",
    response_model = TokenResponse,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Create an account and sign in",
    tags           = ["accounts"],
)
def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> TokenResponse:
    email = auth.normalise_email(body.email)

    patient = Patient(
        email         = email,
        password_hash = auth.hash_password(body.password),
        display_name  = (body.display_name or "").strip() or None,
    )
    db.add(patient)
    try:
        db.commit()
    except IntegrityError:
        # Checking first and inserting second is a race: two simultaneous
        # sign-ups with the same address both pass the check. The unique index
        # is what actually decides, so the collision is handled here.
        db.rollback()
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = "An account already exists for that email address.",
        )

    logger.info("Registered patient %s", patient.id)
    return _issue_token(patient)


@app.post(
    "/auth/login",
    response_model = TokenResponse,
    summary        = "Exchange email and password for a bearer token",
    tags           = ["accounts"],
)
def login(
    body: LoginRequest,
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> TokenResponse:
    patient = auth.find_by_email(db, body.email)

    if patient is None:
        # Verify against a dummy hash anyway. Returning early here would make a
        # login for an unknown address measurably faster than one for a known
        # address, which turns this endpoint into an account-enumeration oracle.
        auth.waste_time_like_a_real_verify()
        raise _BAD_CREDENTIALS

    if not auth.verify_password(patient.password_hash, body.password):
        raise _BAD_CREDENTIALS

    # Argon2 parameters get stronger over time; a correct password is the only
    # moment the plaintext is available to upgrade the stored hash.
    if auth.needs_rehash(patient.password_hash):
        patient.password_hash = auth.hash_password(body.password)

    auth.touch_last_login(db, patient)
    return _issue_token(patient)


@app.get(
    "/auth/me",
    response_model = PatientOut,
    summary        = "The signed-in patient",
    tags           = ["accounts"],
)
def me(patient: Patient = Depends(auth.current_patient)) -> PatientOut:
    return PatientOut.model_validate(patient)


@app.patch(
    "/me/surgery",
    response_model = PatientOut,
    summary        = "Record or clear the operation being recovered from",
    tags           = ["accounts"],
)
def set_surgery(
    body: SurgeryUpdate,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> PatientOut:
    """
    Store the operation date once, so weeks-post-op stops being retyped.

    That number selects the whole rehab protocol. Asking a patient to recompute
    it from memory every visit is how someone ends up in the wrong phase — and a
    plausible wrong number is indistinguishable from a right one, so nothing
    catches it.
    """
    if body.surgery_date is not None and body.surgery_date > date.today():
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = "That surgery date is in the future.",
        )

    patient.surgery_date = body.surgery_date
    patient.surgery_type = body.surgery_type.value if body.surgery_type else None
    db.commit()
    db.refresh(patient)

    logger.info(
        "Surgery recorded | patient=%s type=%s date=%s (week %s)",
        patient.id, patient.surgery_type, patient.surgery_date, patient.weeks_post_op,
    )
    return PatientOut.model_validate(patient)


@app.get(
    "/me/flags",
    response_model = PatientFlags,
    summary        = "Anything worth raising with your physiotherapist",
    tags           = ["accounts"],
)
def my_flags(
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> PatientFlags:
    """
    The same assessment the clinician sees, worded for the person it is about.

    Findings about the clinician's own workflow — an unread prescription, say —
    carry no patient message and are left out here. Telling someone their
    physiotherapist has not read their notes yet is alarming and not theirs to
    act on.
    """
    flags = [f for f in _evaluate_patient(db, patient) if f.patient_message]
    return PatientFlags(
        flags = [PatientFlag(code=f.code, severity=f.severity, message=f.patient_message)
                 for f in flags],
        worst = triage.worst_severity(flags),
    )


@app.get(
    "/me/prescriptions/{prescription_id}",
    response_model = PrescriptionEffective,
    summary        = "One of your prescriptions, as you should follow it",
    tags           = ["accounts"],
)
def my_prescription(
    prescription_id: str,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> PrescriptionEffective:
    """
    Returns the clinician's version when there is one, the model's otherwise.

    The patient follows one document, not two, and which one it is is not their
    problem to work out — that is what `status` is for.
    """
    prescription = db.scalar(
        select(Prescription).where(
            Prescription.id == prescription_id,
            Prescription.patient_id == patient.id,
        )
    )
    if prescription is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such prescription.")

    return PrescriptionEffective(
        id          = prescription.id,
        created_at  = prescription.created_at,
        status      = prescription.status,
        reviewed_at = prescription.reviewed_at,
        reviewed_by = (prescription.reviewed_by.display_name or prescription.reviewed_by.email)
                      if prescription.reviewed_by else None,
        review_note = prescription.review_note,
        payload     = json.loads(prescription.effective_payload),
    )


@app.get(
    "/me/prescriptions",
    response_model = PrescriptionHistory,
    summary        = "Past analyses, newest first",
    tags           = ["accounts"],
)
def my_prescriptions(
    limit: int = Query(20, ge=1, le=100),
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> PrescriptionHistory:
    rows = db.scalars(
        select(Prescription)
        .where(Prescription.patient_id == patient.id)
        .order_by(Prescription.created_at.desc())
        .limit(limit)
    ).all()
    return PrescriptionHistory(count=len(rows), prescriptions=list(rows))


# ---------------------------------------------------------------------------
# Exercise sessions
# ---------------------------------------------------------------------------

@app.post(
    "/sessions/sets",
    response_model = SessionOut,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Record one completed set from the webcam tracker",
    tags           = ["sessions"],
)
def record_set(
    body: SetRecord,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> SessionOut:
    """
    Store what the tracker measured while one set was performed.

    Posted per set rather than once at the end: a patient who shuts the laptop
    after two of three sets is the normal case, and their two sets should
    already be saved with nothing left to reconcile.

    Re-posting the same (session, set_index) is a no-op rather than an error.
    The browser retries on a dropped connection, and a retry must not turn one
    set into two.
    """
    session = db.scalar(
        select(ExerciseSession).where(
            ExerciseSession.patient_id == patient.id,
            ExerciseSession.client_session_id == body.client_session_id,
        )
    )

    if session is None:
        # Only link a prescription the caller actually owns. Without this check
        # a client could file its sessions against someone else's analysis.
        prescription_id = None
        if body.prescription_id:
            owned = db.scalar(
                select(Prescription.id).where(
                    Prescription.id == body.prescription_id,
                    Prescription.patient_id == patient.id,
                )
            )
            if owned is None:
                logger.warning(
                    "Patient %s referenced prescription %s, which is not theirs; "
                    "recording the session unlinked.",
                    patient.id, body.prescription_id,
                )
            prescription_id = owned

        session = ExerciseSession(
            patient_id        = patient.id,
            client_session_id = body.client_session_id,
            prescription_id   = prescription_id,
            exercise_name     = body.exercise_name,
            tracked_joint     = body.tracked_joint.value,
            knee_side         = body.knee_side.value,
            angle_limit       = body.angle_limit,
            target_reps       = body.target_reps,
            target_sets       = body.target_sets,
            hold_seconds      = body.hold_seconds,
            view_verified     = body.view_verified,
            pain_before       = body.pain_before,
        )
        db.add(session)
        db.flush()

    existing = db.scalar(
        select(ExerciseSet).where(
            ExerciseSet.session_id == session.id,
            ExerciseSet.set_index == body.set_index,
        )
    )

    if existing is None:
        db.add(ExerciseSet(
            session_id            = session.id,
            set_index             = body.set_index,
            reps_completed        = body.reps_completed,
            duration_seconds      = body.duration_seconds,
            peak_flexion_deg      = body.peak_flexion_deg,
            breach_count          = body.breach_count,
            breach_seconds        = body.breach_seconds,
            hold_seconds_achieved = body.hold_seconds_achieved,
            mean_visibility       = body.mean_visibility,
            suspended_seconds     = body.suspended_seconds,
        ))
        session.sets_completed += 1
        session.last_set_at = models_utcnow()

        # One unverified set taints the session: the trend cannot treat any of
        # it as a properly measured angle.
        if not body.view_verified:
            session.view_verified = False

    session.completed = session.sets_completed >= session.target_sets

    db.commit()
    db.refresh(session)

    logger.info(
        "Set recorded | patient=%s exercise=%s set=%d/%d reps=%d peak=%.0f° breaches=%d",
        patient.id, session.exercise_name, body.set_index, session.target_sets,
        body.reps_completed, body.peak_flexion_deg, body.breach_count,
    )
    return SessionOut.model_validate(session)


@app.post(
    "/sessions/report",
    response_model = SessionOut,
    summary        = "Record how a finished session felt",
    tags           = ["sessions"],
)
def record_report(
    body: SessionReport,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> SessionOut:
    """
    Attach pain-afterwards and perceived exertion to a session already recorded.

    404 rather than creating one: a rating with no completed set behind it is not
    a session, and counting it as one would inflate an adherence record with work
    that never happened.
    """
    session = db.scalar(
        select(ExerciseSession).where(
            ExerciseSession.patient_id == patient.id,
            ExerciseSession.client_session_id == body.client_session_id,
        )
    )
    if session is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "No recorded session to attach this to.",
        )

    # Only overwrite what was actually sent, so a second call adding exertion
    # does not wipe a pain score from the first.
    if body.pain_after is not None:
        session.pain_after = body.pain_after
    if body.rpe is not None:
        session.rpe = body.rpe

    db.commit()
    db.refresh(session)
    logger.info(
        "Session report | patient=%s exercise=%s pain %s→%s rpe=%s",
        patient.id, session.exercise_name, session.pain_before, session.pain_after, session.rpe,
    )
    return SessionOut.model_validate(session)


@app.get(
    "/sessions",
    response_model = SessionHistory,
    summary        = "Recent exercise sessions, newest first",
    tags           = ["sessions"],
)
def my_sessions(
    limit: int = Query(20, ge=1, le=100),
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> SessionHistory:
    rows = db.scalars(
        select(ExerciseSession)
        .where(ExerciseSession.patient_id == patient.id)
        .order_by(ExerciseSession.started_at.desc())
        .limit(limit)
    ).all()
    return SessionHistory(count=len(rows), sessions=list(rows))


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
#
# Aggregated in Python rather than SQL. The volume is small — a year of daily
# rehab is a few thousand rows — and date bucketing is the one thing SQLite and
# Postgres disagree about most, so doing it here keeps the two backends
# identical and the shifting-into-local-time step trivial.

# The rule that runs through all of this: a session performed with the camera
# gate overridden still counts as work done, but none of its angles are
# trustworthy. A knee measured square-on to the camera reads close to straight
# no matter how far it is bent (see frontend/assets/pose-gate.js). So unverified
# sessions are counted for adherence and excluded from every flexion figure —
# charting them would show a recovery that never happened.

def _as_utc(dt):
    """SQLite hands back naive datetimes; treat those as the UTC they were stored as."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _local_date(dt, tz_offset_minutes: int):
    """The calendar day this happened on, from the patient's point of view."""
    return (_as_utc(dt) + timedelta(minutes=tz_offset_minutes)).date()


def _streaks(days: set, today):
    """
    (current, longest) run of consecutive active days.

    The current streak may end yesterday rather than today: someone who has not
    exercised *yet* today has not broken anything, and zeroing their streak at
    midnight would be both wrong and discouraging.
    """
    if not days:
        return 0, 0

    ordered = sorted(days)
    longest = run = 1
    for prev, cur in pairwise(ordered):
        run = run + 1 if (cur - prev).days == 1 else 1
        longest = max(longest, run)

    current = 0
    if ordered[-1] in (today, today - timedelta(days=1)):
        cursor = ordered[-1]
        while cursor in days:
            current += 1
            cursor -= timedelta(days=1)

    return current, longest


def build_progress(
    db: Session,
    patient: Patient,
    days: int,
    tz_offset_minutes: int,
) -> ProgressResponse:
    """
    Aggregate one patient's sessions into the progress view.

    Factored out of the endpoint so a share link can serve exactly the same
    figures without a second implementation drifting away from this one.
    """
    now = models_utcnow()
    since = now - timedelta(days=days)

    sessions = db.scalars(
        select(ExerciseSession)
        .where(
            ExerciseSession.patient_id == patient.id,
            ExerciseSession.started_at >= since,
        )
        .order_by(ExerciseSession.started_at)
    ).all()

    total_sets = total_reps = 0
    active_days: set = set()
    unverified = 0

    # Keyed by (exercise, day). Peak flexion only means something against what
    # the exercise asked for — see RomSeries — so the two are never pooled.
    rom_by_exercise_day: dict = defaultdict(dict)
    pain_by_day: dict = defaultdict(lambda: {"before": [], "after": [], "sessions": 0})
    pain_changes: list = []
    latest_pain_after = None
    adherence_by_day: dict = defaultdict(lambda: {"sessions": 0, "sets": 0, "completed_sessions": 0})
    by_exercise: dict = defaultdict(lambda: {
        "sessions": 0, "sets": 0, "reps": 0, "peak": None,
        "breach_count": 0, "breach_seconds": 0.0, "vis_sum": 0.0, "vis_n": 0,
    })

    for session in sessions:
        day = _local_date(session.started_at, tz_offset_minutes)
        active_days.add(day)

        session_sets = session.sets
        session_reps = sum(s.reps_completed for s in session_sets)
        total_sets += len(session_sets)
        total_reps += session_reps

        entry = adherence_by_day[day]
        entry["sessions"] += 1
        entry["sets"] += len(session_sets)
        entry["completed_sessions"] += 1 if session.completed else 0

        ex = by_exercise[session.exercise_name]
        ex["sessions"] += 1
        ex["sets"] += len(session_sets)
        ex["reps"] += session_reps
        ex["breach_count"] += sum(s.breach_count for s in session_sets)
        ex["breach_seconds"] += sum(s.breach_seconds for s in session_sets)
        for s in session_sets:
            if s.mean_visibility is not None:
                ex["vis_sum"] += s.mean_visibility
                ex["vis_n"] += 1

        # Pain is reported by the patient, so an unverified camera angle says
        # nothing about it — unlike the measured angles below, these are kept
        # whatever the view was.
        if session.pain_before is not None or session.pain_after is not None:
            bucket = pain_by_day[day]
            bucket["sessions"] += 1
            if session.pain_before is not None:
                bucket["before"].append(session.pain_before)
            if session.pain_after is not None:
                bucket["after"].append(session.pain_after)
                latest_pain_after = session.pain_after
            if session.pain_before is not None and session.pain_after is not None:
                pain_changes.append(session.pain_after - session.pain_before)

        if not session.view_verified:
            unverified += 1
            continue

        # Angles from here down: verified sessions only.
        peak = max((s.peak_flexion_deg for s in session_sets), default=None)
        if peak is None:
            continue

        ex["peak"] = peak if ex["peak"] is None else max(ex["peak"], peak)

        series = rom_by_exercise_day[session.exercise_name]
        point = series.get(day)
        if point is None:
            series[day] = {"peak": peak, "angle_limit": session.angle_limit, "sessions": 1}
        else:
            point["peak"] = max(point["peak"], peak)
            # The later session's ceiling is the one that applied by end of day.
            point["angle_limit"] = session.angle_limit
            point["sessions"] += 1

    def _mean(values):
        return round(sum(values) / len(values), 1) if values else None

    pain_trend = [
        PainPoint(
            date        = d,
            pain_before = _mean(v["before"]),
            pain_after  = _mean(v["after"]),
            sessions    = v["sessions"],
        )
        for d, v in sorted(pain_by_day.items())
    ]

    today = _local_date(now, tz_offset_minutes)
    current_streak, longest_streak = _streaks(active_days, today)

    rom_by_exercise = []
    for name, by_day in rom_by_exercise_day.items():
        points = [
            RomPoint(date=d, peak_flexion_deg=round(v["peak"], 1),
                     angle_limit=v["angle_limit"], sessions=v["sessions"])
            for d, v in sorted(by_day.items())
        ]
        rom_by_exercise.append(RomSeries(
            exercise_name = name,
            days_measured = len(points),
            latest_deg    = points[-1].peak_flexion_deg,
            best_deg      = max(p.peak_flexion_deg for p in points),
            angle_limit   = points[-1].angle_limit,
            points        = points,
        ))

    # Most-measured first: that is the series worth charting by default, and the
    # one the headline figures describe.
    rom_by_exercise.sort(key=lambda r: (-r.days_measured, r.exercise_name))
    primary = rom_by_exercise[0] if rom_by_exercise else None

    return ProgressResponse(
        range_days        = days,
        tz_offset_minutes = tz_offset_minutes,
        generated_at      = now,
        summary = ProgressSummary(
            sessions            = len(sessions),
            sets                = total_sets,
            reps                = total_reps,
            active_days         = len(active_days),
            current_streak_days = current_streak,
            longest_streak_days = longest_streak,
            primary_exercise    = primary.exercise_name if primary else None,
            best_flexion_deg    = primary.best_deg if primary else None,
            latest_flexion_deg  = primary.latest_deg if primary else None,
            latest_pain_after   = latest_pain_after,
            mean_pain_change    = round(sum(pain_changes) / len(pain_changes), 1) if pain_changes else None,
            sessions_with_pain  = sum(v["sessions"] for v in pain_by_day.values()),
            unverified_sessions = unverified,
        ),
        pain_trend = pain_trend,
        rom_by_exercise = rom_by_exercise,
        adherence = [
            AdherenceDay(date=d, **v) for d, v in sorted(adherence_by_day.items())
        ],
        by_exercise = [
            ExerciseBreakdown(
                exercise_name    = name,
                sessions         = v["sessions"],
                sets             = v["sets"],
                reps             = v["reps"],
                peak_flexion_deg = round(v["peak"], 1) if v["peak"] is not None else None,
                breach_count     = v["breach_count"],
                breach_seconds   = round(v["breach_seconds"], 1),
                mean_visibility  = round(v["vis_sum"] / v["vis_n"], 3) if v["vis_n"] else None,
            )
            # Most-practised first: that is the order a clinician scans.
            for name, v in sorted(by_exercise.items(), key=lambda kv: (-kv[1]["sessions"], kv[0]))
        ],
    )


@app.get(
    "/me/progress",
    response_model = ProgressResponse,
    summary        = "Range of motion, adherence and per-exercise history",
    tags           = ["sessions"],
)
def my_progress(
    days: int = Query(90, ge=1, le=365, description="How far back to look"),
    tz_offset_minutes: int = Query(
        0, ge=-840, le=840,
        description="Minutes east of UTC, i.e. -new Date().getTimezoneOffset(). "
                    "Days are bucketed in this offset so an evening session lands on "
                    "the evening it happened.",
    ),
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> ProgressResponse:
    return build_progress(db, patient, days, tz_offset_minutes)


# ---------------------------------------------------------------------------
# Share links
# ---------------------------------------------------------------------------
#
# A clinician will click a link. A clinician will not create an account in an app
# their patient uses, and insisting they do is where this feature dies — so the
# link itself is the credential.
#
# Which puts the weight on being able to take it back. Tokens are stored hashed
# and looked up by hash, so a leaked database yields no working links, and a
# revoked row stops working immediately rather than whenever a signed expiry
# happens to run out.

# Where the share page lives, relative to wherever the frontend is served. Kept
# as a path rather than a full URL because the backend does not know the
# frontend's origin — that pairing is a deployment decision, not this file's.
SHARE_PATH = "/share.html?t={token}"


def _hash_share_token(token: str) -> str:
    """
    SHA-256, not Argon2.

    The token is 256 bits from secrets.token_urlsafe, so there is no dictionary
    for a slow hash to protect against — and this runs on every page view of a
    shared link, where a deliberately expensive hash would be a denial-of-service
    surface rather than a defence.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@app.post(
    "/me/share-links",
    response_model = ShareLinkCreated,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Create a read-only link to your progress",
    tags           = ["sharing"],
)
def create_share_link(
    body: ShareLinkCreate,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> ShareLinkCreated:
    token = secrets.token_urlsafe(32)

    link = ShareLink(
        patient_id = patient.id,
        token_hash = _hash_share_token(token),
        label      = (body.label or "").strip() or None,
        expires_at = models_utcnow() + timedelta(days=body.days),
    )
    db.add(link)
    db.commit()
    db.refresh(link)

    logger.info("Share link created | patient=%s expires=%s", patient.id, link.expires_at)

    # The only time the token exists outside the browser that asked for it.
    return ShareLinkCreated(
        **ShareLinkOut.model_validate(link).model_dump(),
        token = token,
        path  = SHARE_PATH.format(token=token),
    )


@app.get(
    "/me/share-links",
    response_model = list[ShareLinkOut],
    summary        = "Links you have shared",
    tags           = ["sharing"],
)
def list_share_links(
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> list[ShareLinkOut]:
    rows = db.scalars(
        select(ShareLink)
        .where(ShareLink.patient_id == patient.id)
        .order_by(ShareLink.created_at.desc())
    ).all()
    return [ShareLinkOut.model_validate(r) for r in rows]


@app.delete(
    "/me/share-links/{link_id}",
    response_model = ShareLinkOut,
    summary        = "Revoke a share link",
    tags           = ["sharing"],
)
def revoke_share_link(
    link_id: str,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> ShareLinkOut:
    link = db.scalar(
        select(ShareLink).where(ShareLink.id == link_id, ShareLink.patient_id == patient.id)
    )
    if link is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such share link.")

    # Revoking twice is not an error — the patient wants it off, and it is off.
    if link.revoked_at is None:
        link.revoked_at = models_utcnow()
        db.commit()
        db.refresh(link)
        logger.info("Share link revoked | patient=%s link=%s", patient.id, link.id)

    return ShareLinkOut.model_validate(link)


@app.get(
    "/share/{token}",
    response_model = SharedProgress,
    summary        = "Open a shared progress view (no account needed)",
    tags           = ["sharing"],
)
def read_shared_progress(
    token: str,
    request: Request,
    days: int = Query(90, ge=1, le=365),
    tz_offset_minutes: int = Query(0, ge=-840, le=840),
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> SharedProgress:
    """
    The clinician's view. No account, no password — the link is the credential.

    Every failure returns the same 404. Distinguishing "expired", "revoked" and
    "never existed" would confirm to anyone holding an old link that it was once
    real, and whose it was.
    """
    link = db.scalar(select(ShareLink).where(ShareLink.token_hash == _hash_share_token(token)))

    if link is None or not link.is_active:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "This link is not valid. It may have expired or been withdrawn.",
        )

    # Recorded so the patient can see whether it was ever opened. Not who opened
    # it — the link carries no identity, and inventing one from an IP address
    # would be a guess dressed up as a fact.
    link.access_count += 1
    link.last_accessed_at = models_utcnow()
    db.commit()

    patient = link.patient
    return SharedProgress(
        patient_name = patient.display_name,
        shared_at    = link.created_at,
        expires_at   = link.expires_at,
        label        = link.label,
        progress     = build_progress(db, patient, days, tz_offset_minutes),
    )


# ---------------------------------------------------------------------------
# Clinicians
# ---------------------------------------------------------------------------
#
# A clinician issues a code; the patient redeems it. That direction is the point:
# redeeming is the patient's own act, and that act is the consent. Nobody can
# attach themselves to a patient's record without the patient doing something.
#
# Either side can end the link afterwards. A patient must be able to stop
# sharing, and a clinician must be able to discharge.

# Unambiguous when read aloud or written down: no O/0, no I/1, no U (which is
# heard as "you"). Ten characters from this alphabet is about 10^15 codes, and
# they expire in days behind a rate limiter.
_INVITE_ALPHABET = "ABCDEFGHJKLMNPQRSTVWXYZ23456789"
_INVITE_LENGTH = 10


def _new_invite_code() -> str:
    raw = "".join(secrets.choice(_INVITE_ALPHABET) for _ in range(_INVITE_LENGTH))
    return f"{raw[:5]}-{raw[5:]}"


def _normalise_invite_code(code: str) -> str:
    """Dashes, spaces and case are cosmetic — people retype codes how they like."""
    return "".join(ch for ch in code.upper() if ch.isalnum())


def _hash_invite_code(code: str) -> str:
    return hashlib.sha256(_normalise_invite_code(code).encode("utf-8")).hexdigest()


def _issue_clinician_token(clinician: Clinician) -> ClinicianToken:
    token, expires_in = auth.create_access_token(clinician.id, role=auth.ROLE_CLINICIAN)
    return ClinicianToken(
        access_token = token,
        expires_in   = expires_in,
        clinician    = ClinicianOut.model_validate(clinician),
    )


@app.post(
    "/clinician/register",
    response_model = ClinicianToken,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Create a clinician account",
    tags           = ["clinicians"],
)
def clinician_register(
    body: ClinicianRegister,
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> ClinicianToken:
    clinician = Clinician(
        email         = auth.normalise_email(body.email),
        password_hash = auth.hash_password(body.password),
        display_name  = (body.display_name or "").strip() or None,
        registration  = (body.registration or "").strip() or None,
    )
    db.add(clinician)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = "An account already exists for that email address.",
        )

    logger.info("Registered clinician %s", clinician.id)
    return _issue_clinician_token(clinician)


@app.post(
    "/clinician/login",
    response_model = ClinicianToken,
    summary        = "Sign in as a clinician",
    tags           = ["clinicians"],
)
def clinician_login(
    body: LoginRequest,
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> ClinicianToken:
    clinician = auth.find_clinician_by_email(db, body.email)

    if clinician is None:
        # Same dummy verify as the patient login, for the same reason: without
        # it, response time alone says whether an address is registered.
        auth.waste_time_like_a_real_verify()
        raise _BAD_CREDENTIALS

    if not auth.verify_password(clinician.password_hash, body.password):
        raise _BAD_CREDENTIALS

    if auth.needs_rehash(clinician.password_hash):
        clinician.password_hash = auth.hash_password(body.password)

    auth.touch_last_login(db, clinician)
    return _issue_clinician_token(clinician)


@app.get(
    "/clinician/me",
    response_model = ClinicianOut,
    summary        = "The signed-in clinician",
    tags           = ["clinicians"],
)
def clinician_me(clinician: Clinician = Depends(auth.current_clinician)) -> ClinicianOut:
    return ClinicianOut.model_validate(clinician)


# ── Invites ────────────────────────────────────────────────────────────────

@app.post(
    "/clinician/invites",
    response_model = InviteCreated,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Create a code for a patient to redeem",
    tags           = ["clinicians"],
)
def create_invite(
    body: InviteCreate,
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> InviteCreated:
    code = _new_invite_code()
    link = CareLink(
        clinician_id     = clinician.id,
        invite_code_hash = _hash_invite_code(code),
        # Enough to tell two pending codes apart on screen, far too little to
        # narrow a guess: seven of ten characters remain unknown.
        invite_hint      = code[:3],
        patient_label    = (body.patient_label or "").strip() or None,
        expires_at       = models_utcnow() + timedelta(days=body.days),
    )
    db.add(link)
    db.commit()
    db.refresh(link)

    logger.info("Invite created | clinician=%s expires=%s", clinician.id, link.expires_at)
    return InviteCreated(**InviteOut.model_validate(link).model_dump(), code=code)


@app.get(
    "/clinician/invites",
    response_model = list[InviteOut],
    summary        = "Codes issued but not yet redeemed",
    tags           = ["clinicians"],
)
def list_invites(
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> list[InviteOut]:
    rows = db.scalars(
        select(CareLink)
        .where(
            CareLink.clinician_id == clinician.id,
            CareLink.patient_id.is_(None),
            CareLink.revoked_at.is_(None),
        )
        .order_by(CareLink.created_at.desc())
    ).all()
    return [InviteOut.model_validate(r) for r in rows]


@app.post(
    "/me/clinicians/redeem",
    response_model = CareLinkOut,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Give a clinician access to your progress",
    tags           = ["clinicians"],
)
def redeem_invite(
    body: RedeemInvite,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> CareLinkOut:
    """
    Redeeming is the consent. Everything a clinician can see follows from this.
    """
    link = db.scalar(
        select(CareLink).where(CareLink.invite_code_hash == _hash_invite_code(body.code))
    )

    # One message for wrong, expired and already-used alike. Anything more
    # specific turns this into an oracle for guessing codes.
    if link is None or not link.is_pending:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "That code is not valid. Ask your physiotherapist for a new one.",
        )

    existing = db.scalar(
        select(CareLink).where(
            CareLink.clinician_id == link.clinician_id,
            CareLink.patient_id == patient.id,
            CareLink.revoked_at.is_(None),
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = "That physiotherapist already has access to your progress.",
        )

    link.patient_id = patient.id
    link.accepted_at = models_utcnow()
    db.commit()
    db.refresh(link)

    logger.info("Care link accepted | patient=%s clinician=%s", patient.id, link.clinician_id)
    return _care_link_out(link)


def _care_link_out(link: CareLink) -> CareLinkOut:
    return CareLinkOut(
        id              = link.id,
        clinician_name  = link.clinician.display_name if link.clinician else None,
        clinician_email = link.clinician.email if link.clinician else None,
        patient_name    = link.patient.display_name if link.patient else None,
        accepted_at     = link.accepted_at,
        revoked_at      = link.revoked_at,
        revoked_by      = link.revoked_by,
        is_active       = link.is_active,
    )


@app.get(
    "/me/clinicians",
    response_model = list[CareLinkOut],
    summary        = "Who can see your progress",
    tags           = ["clinicians"],
)
def my_clinicians(
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> list[CareLinkOut]:
    rows = db.scalars(
        select(CareLink)
        .where(CareLink.patient_id == patient.id)
        .order_by(CareLink.accepted_at.desc())
    ).all()
    return [_care_link_out(r) for r in rows]


@app.delete(
    "/me/clinicians/{link_id}",
    response_model = CareLinkOut,
    summary        = "Withdraw a clinician's access",
    tags           = ["clinicians"],
)
def withdraw_clinician(
    link_id: str,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> CareLinkOut:
    link = db.scalar(
        select(CareLink).where(CareLink.id == link_id, CareLink.patient_id == patient.id)
    )
    if link is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such link.")

    if link.revoked_at is None:
        link.revoked_at = models_utcnow()
        link.revoked_by = "patient"
        db.commit()
        db.refresh(link)
        logger.info("Care link withdrawn by patient | link=%s", link.id)

    return _care_link_out(link)


@app.delete(
    "/clinician/patients/{link_id}",
    response_model = CareLinkOut,
    summary        = "Discharge a patient from your caseload",
    tags           = ["clinicians"],
)
def discharge_patient(
    link_id: str,
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> CareLinkOut:
    link = db.scalar(
        select(CareLink).where(CareLink.id == link_id, CareLink.clinician_id == clinician.id)
    )
    if link is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such link.")

    if link.revoked_at is None:
        link.revoked_at = models_utcnow()
        # Recorded separately from a patient withdrawal: "I discharged them" and
        # "they withdrew" are different events, and a caseload that cannot tell
        # them apart tells the clinician something false.
        link.revoked_by = "clinician"
        db.commit()
        db.refresh(link)
        logger.info("Patient discharged | link=%s clinician=%s", link.id, clinician.id)

    return _care_link_out(link)


# ── The caseload ───────────────────────────────────────────────────────────

@app.get(
    "/clinician/patients",
    response_model = Caseload,
    summary        = "Your caseload",
    tags           = ["clinicians"],
)
def caseload(
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> Caseload:
    """
    The screen a clinician scans, so it carries what tells them who to look at.

    Sessions for the whole caseload are fetched in one query and aggregated here,
    rather than a query per patient — a list of thirty would otherwise be thirty
    round trips to render one page.
    """
    links = db.scalars(
        select(CareLink)
        .where(
            CareLink.clinician_id == clinician.id,
            CareLink.patient_id.is_not(None),
            CareLink.revoked_at.is_(None),
        )
        .order_by(CareLink.accepted_at.desc())
    ).all()

    if not links:
        return Caseload(count=0, patients=[])

    patient_ids = [link.patient_id for link in links]
    since = models_utcnow() - timedelta(days=30)
    week_ago = models_utcnow() - timedelta(days=7)

    sessions = db.scalars(
        select(ExerciseSession)
        .where(
            ExerciseSession.patient_id.in_(patient_ids),
            ExerciseSession.started_at >= since,
        )
        .order_by(ExerciseSession.started_at)
    ).all()

    by_patient: dict = defaultdict(lambda: {
        "last": None, "week": 0, "flexion": None, "pain": None, "breaches": 0,
    })
    for session in sessions:
        started = _as_utc(session.started_at)
        entry = by_patient[session.patient_id]
        entry["last"] = started

        if started >= week_ago:
            entry["week"] += 1
            entry["breaches"] += sum(s.breach_count for s in session.sets)

        if session.pain_after is not None:
            entry["pain"] = session.pain_after

        # Angles only from a verified camera view — an unverified one reads far
        # too low, and a clinician scanning this column would read a collapse in
        # range of motion that never happened.
        if session.view_verified:
            peak = max((s.peak_flexion_deg for s in session.sets), default=None)
            if peak is not None:
                entry["flexion"] = round(peak, 1)

    # Assessed per patient rather than in one pass: the rules need each
    # patient's whole recent history, and a caseload is tens of people, not
    # thousands.
    flags_by_patient = {
        link.patient_id: _evaluate_patient(db, link.patient) for link in links
    }

    return Caseload(
        count = len(links),
        patients = [
            CaseloadEntry(
                link_id              = link.id,
                patient_id           = link.patient_id,
                patient_name         = link.patient.display_name,
                patient_label        = link.patient_label,
                surgery_type         = link.patient.surgery_type,
                weeks_post_op        = link.patient.weeks_post_op,
                linked_at            = link.accepted_at,
                last_session_at      = by_patient[link.patient_id]["last"],
                sessions_last_7_days = by_patient[link.patient_id]["week"],
                latest_flexion_deg   = by_patient[link.patient_id]["flexion"],
                latest_pain_after    = by_patient[link.patient_id]["pain"],
                breaches_last_7_days = by_patient[link.patient_id]["breaches"],
                flags = [
                    FlagOut(code=f.code, severity=f.severity, summary=f.summary, evidence=f.evidence)
                    for f in flags_by_patient[link.patient_id]
                ],
                worst_flag = triage.worst_severity(flags_by_patient[link.patient_id]),
            )
            for link in links
        ],
    )


def _evaluate_patient(db: Session, patient: Patient) -> list:
    """
    Run the triage rules over one patient's recent history.

    The window is the widest any rule looks at, so every rule sees the same
    data — a flag that fired on a different slice than its neighbour would be
    impossible to reason about.
    """
    since = models_utcnow() - timedelta(days=triage.BASELINE_DAYS + 1)

    sessions = db.scalars(
        select(ExerciseSession).where(
            ExerciseSession.patient_id == patient.id,
            ExerciseSession.started_at >= since,
        )
    ).all()
    prescriptions = db.scalars(
        select(Prescription).where(
            Prescription.patient_id == patient.id,
            Prescription.created_at >= since,
        )
    ).all()
    return triage.evaluate(sessions, prescriptions)


def _linked_patient(db: Session, clinician: Clinician, patient_id: str) -> Patient:
    """
    Resolve a patient this clinician is actually looking after.

    404 rather than 403 for a patient they are not linked to: telling a clinician
    that an id exists but is not theirs confirms the account is real.
    """
    link = db.scalar(
        select(CareLink).where(
            CareLink.clinician_id == clinician.id,
            CareLink.patient_id == patient_id,
            CareLink.revoked_at.is_(None),
        )
    )
    if link is None or link.patient is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "That patient is not on your caseload.",
        )
    return link.patient


@app.get(
    "/clinician/patients/{patient_id}/progress",
    response_model = ProgressResponse,
    summary        = "One patient's progress in full",
    tags           = ["clinicians"],
)
def patient_progress(
    patient_id: str,
    days: int = Query(90, ge=1, le=365),
    tz_offset_minutes: int = Query(0, ge=-840, le=840),
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> ProgressResponse:
    patient = _linked_patient(db, clinician, patient_id)
    return build_progress(db, patient, days, tz_offset_minutes)


# ---------------------------------------------------------------------------
# Clinical review
# ---------------------------------------------------------------------------
#
# This is the change that alters what the system is. Until now the model set a
# movement ceiling and the patient followed it, with nobody accountable for the
# number. Now the model drafts and a clinician approves.
#
# Two rules run through everything below.
#
# 1. The clinician's judgement wins. They operated on the knee; the radiograph
#    did not. A system that refused to let them raise a limit would simply be
#    ignored, and then the ceiling would be enforced by nothing at all.
#
# 2. It does not get to be invisible. Every value the patient is asked to keep
#    their knee under is recorded with where it came from, what it replaced, and
#    — when a human loosened it — why. Loosening requires a reason; tightening
#    does not, because tightening is not the direction that hurts.

def _catalogue_by_name() -> dict:
    return {ex["name"]: ex for ex in all_exercises()}


def _prescription_for_clinician(db: Session, clinician: Clinician, prescription_id: str) -> Prescription:
    """
    Resolve a prescription belonging to a patient on this clinician's caseload.

    404 rather than 403 for anything else: confirming that an id exists but is
    not theirs confirms the patient exists.
    """
    prescription = db.get(Prescription, prescription_id)
    if prescription is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such prescription.")
    # Raises 404 itself when the patient is not on the caseload.
    _linked_patient(db, clinician, prescription.patient_id)
    return prescription


def _audit_out(rows: list) -> list[AuditEntry]:
    return [
        AuditEntry(
            exercise_name  = r.exercise_name,
            field          = r.field,
            previous_value = r.previous_value,
            new_value      = r.new_value,
            actor          = r.actor,
            clinician_name = None,
            reason         = r.reason,
            created_at     = r.created_at,
        )
        for r in rows
    ]


def _detail(db: Session, prescription: Prescription) -> PrescriptionDetail:
    draft = json.loads(prescription.payload)
    effective = json.loads(prescription.effective_payload)
    return PrescriptionDetail(
        id              = prescription.id,
        patient_id      = prescription.patient_id,
        patient_name    = prescription.patient.display_name if prescription.patient else None,
        created_at      = prescription.created_at,
        status          = prescription.status,
        reviewed_at     = prescription.reviewed_at,
        reviewed_by     = (prescription.reviewed_by.display_name or prescription.reviewed_by.email)
                          if prescription.reviewed_by else None,
        review_note     = prescription.review_note,
        kl_grade        = prescription.kl_grade,
        max_angle       = effective.get("max_angle", prescription.max_angle),
        model_max_angle = draft.get("max_angle", prescription.max_angle),
        model_version   = prescription.model_version,
        demo_mode       = prescription.demo_mode,
        confidence_band = draft.get("confidence_band"),
        calibrated      = bool(draft.get("calibrated")),
        draft           = draft,
        effective       = effective,
        audit           = _audit_out(prescription.audit),
    )


@app.get(
    "/exercises/catalogue",
    response_model = list[CatalogueExercise],
    summary        = "Every exercise a clinician can prescribe",
    tags           = ["review"],
)
def exercise_catalogue() -> list[CatalogueExercise]:
    """
    The same catalogue the protocols draw from.

    A clinician adding something the model did not prescribe picks from here, so
    the tracker still knows how to count it and which joint to watch. A free-text
    exercise would be untrackable, which is worse than not offering it.
    """
    return [
        CatalogueExercise(
            name          = ex["name"],
            description   = ex["description"],
            target_reps   = ex["target_reps"],
            target_sets   = ex["target_sets"],
            angle_limit   = ex.get("protocol_angle_limit", 120),
            hold_seconds  = ex.get("hold_seconds"),
            tracked_joint = ex.get("tracked_joint", "knee"),
            hold_target   = ex.get("hold_target"),
            start_angle   = ex.get("start_angle"),
            instructions  = ex["instructions"],
            cautions      = ex.get("cautions"),
        )
        for ex in all_exercises()
    ]


@app.get(
    "/clinician/patients/{patient_id}/flags",
    response_model = PatientFlagsForClinician,
    summary        = "Why this patient might need looking at",
    tags           = ["review"],
)
def patient_flags(
    patient_id: str,
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> PatientFlagsForClinician:
    patient = _linked_patient(db, clinician, patient_id)
    flags = _evaluate_patient(db, patient)
    return PatientFlagsForClinician(
        patient_id = patient.id,
        flags = [FlagOut(code=f.code, severity=f.severity, summary=f.summary, evidence=f.evidence)
                 for f in flags],
        worst = triage.worst_severity(flags),
    )


@app.get(
    "/clinician/patients/{patient_id}/prescriptions",
    response_model = list[PrescriptionSummaryForClinician],
    summary        = "A patient's prescriptions, newest first",
    tags           = ["review"],
)
def patient_prescriptions(
    patient_id: str,
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> list[PrescriptionSummaryForClinician]:
    patient = _linked_patient(db, clinician, patient_id)
    rows = db.scalars(
        select(Prescription)
        .where(Prescription.patient_id == patient.id)
        .order_by(Prescription.created_at.desc())
    ).all()
    return [
        PrescriptionSummaryForClinician(
            id=r.id, created_at=r.created_at, status=r.status, kl_grade=r.kl_grade,
            max_angle=json.loads(r.effective_payload).get("max_angle", r.max_angle),
            surgery_type=r.surgery_type, weeks_post_op=r.weeks_post_op,
            rehab_phase=r.rehab_phase, reviewed_at=r.reviewed_at,
        )
        for r in rows
    ]


@app.get(
    "/clinician/prescriptions/{prescription_id}",
    response_model = PrescriptionDetail,
    summary        = "One prescription, draft and effective side by side",
    tags           = ["review"],
)
def prescription_detail(
    prescription_id: str,
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> PrescriptionDetail:
    return _detail(db, _prescription_for_clinician(db, clinician, prescription_id))


@app.post(
    "/clinician/prescriptions/{prescription_id}/review",
    response_model = PrescriptionDetail,
    summary        = "Approve a prescription, with or without changes",
    tags           = ["review"],
)
def review_prescription(
    prescription_id: str,
    body: ReviewRequest,
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> PrescriptionDetail:
    prescription = _prescription_for_clinician(db, clinician, prescription_id)
    draft = json.loads(prescription.payload)
    catalogue = _catalogue_by_name()

    drafted = {ex["name"]: ex for ex in draft.get("exercise_list", [])}
    model_ceiling = draft.get("max_angle", prescription.max_angle)
    ceiling = model_ceiling
    audit: list[PrescriptionAudit] = []

    def record(field, previous, new, reason=None, exercise=None):
        audit.append(PrescriptionAudit(
            prescription_id = prescription.id,
            exercise_name   = exercise,
            field           = field,
            previous_value  = None if previous is None else str(previous),
            new_value       = None if new is None else str(new),
            actor           = clinician.id,
            clinician_id    = clinician.id,
            reason          = reason,
        ))

    def require_reason(reason, what):
        """
        A loosening without a stated reason is refused.

        Not paperwork: this number is the one thing in the system that can hurt
        someone, and "the physiotherapist raised it" is not an answer the patient
        or the next clinician can weigh. Tightening passes without comment.
        """
        if not (reason or "").strip():
            raise HTTPException(
                status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail      = f"{what} loosens a movement restriction, so it needs a reason.",
            )

    # ── The prescription-wide ceiling ────────────────────────────────────────
    if body.ceiling is not None and body.ceiling != model_ceiling:
        if body.ceiling > model_ceiling:
            require_reason(body.ceiling_reason, f"Raising the safe ceiling to {body.ceiling}°")
        ceiling = body.ceiling
        record("ceiling", model_ceiling, ceiling, body.ceiling_reason)

    # ── Exercise by exercise ─────────────────────────────────────────────────
    decisions = {d.name: d for d in body.decisions}
    unknown = [
        name for name, d in decisions.items()
        if name not in drafted and (d.action != "add" or name not in catalogue)
    ]
    if unknown:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = f"Not in this prescription or the catalogue: {', '.join(sorted(unknown))}.",
        )

    result: list[dict] = []

    for name, drafted_exercise in drafted.items():
        decision = decisions.get(name)

        # No decision means no change. A clinician who edits three exercises out
        # of nine has approved the other six, not silently deleted them.
        if decision is None or decision.action == "keep":
            result.append({**drafted_exercise, "angle_limit": min(drafted_exercise["angle_limit"], ceiling)})
            continue

        if decision.action == "remove":
            record("removed", drafted_exercise["angle_limit"], None, decision.reason, name)
            continue

        adjusted = dict(drafted_exercise)
        original_limit = drafted_exercise["angle_limit"]

        if decision.angle_limit is not None and decision.angle_limit != original_limit:
            if decision.angle_limit > original_limit:
                require_reason(decision.reason, f"Raising {name} to {decision.angle_limit}°")
            adjusted["angle_limit"] = decision.angle_limit
            record("angle_limit", original_limit, decision.angle_limit, decision.reason, name)
        else:
            adjusted["angle_limit"] = min(original_limit, ceiling)

        for field, value in (("target_reps", decision.target_reps),
                             ("target_sets", decision.target_sets),
                             ("hold_seconds", decision.hold_seconds)):
            if value is not None and value != drafted_exercise.get(field):
                record(field, drafted_exercise.get(field), value, decision.reason, name)
                adjusted[field] = value

        # Carried on the exercise so the patient's own screen can say who changed
        # it. A silent adjustment is indistinguishable from the model's output.
        adjusted["override"] = {
            "by": clinician.display_name or clinician.email,
            "original_angle_limit": original_limit,
            "reason": (decision.reason or "").strip() or None,
        }
        result.append(adjusted)

    # ── Additions ────────────────────────────────────────────────────────────
    for name, decision in decisions.items():
        if decision.action != "add" or name in drafted:
            continue
        source = catalogue[name]
        protocol_limit = source.get("protocol_angle_limit", 120)
        limit = decision.angle_limit if decision.angle_limit is not None else min(protocol_limit, ceiling)

        # An exercise the model withheld, or one whose limit exceeds the ceiling,
        # is a loosening however it is spelled.
        if limit > ceiling:
            require_reason(decision.reason, f"Adding {name} at {limit}°, above the {ceiling}° ceiling")

        result.append({
            "name":          source["name"],
            "description":   source["description"],
            "target_reps":   decision.target_reps or source["target_reps"],
            "target_sets":   decision.target_sets or source["target_sets"],
            "angle_limit":   limit,
            "hold_seconds":  decision.hold_seconds or source.get("hold_seconds"),
            "tracked_joint": source.get("tracked_joint", "knee"),
            "hold_target":   source.get("hold_target"),
            "start_angle":   source.get("start_angle"),
            "hold_angle":    source.get("hold_angle"),
            "instructions":  source["instructions"],
            "cautions":      source.get("cautions"),
            "angle_capped":  limit < protocol_limit,
            "override": {
                "by": clinician.display_name or clinician.email,
                "original_angle_limit": None,
                "reason": (decision.reason or "").strip() or None,
            },
        })
        record("added", None, limit, decision.reason, name)

    # ── Commit ───────────────────────────────────────────────────────────────
    changed = bool(audit)

    if changed:
        approved = dict(draft)
        approved["exercise_list"] = result
        approved["max_angle"] = ceiling
        approved["reviewed_by"] = clinician.display_name or clinician.email
        prescription.approved_payload = json.dumps(approved, default=str)
        prescription.status = "clinician_modified"
    else:
        # Read and agreed with. The draft stands, and now someone has put their
        # name to it — which is the difference between an unreviewed machine
        # output and a prescription.
        prescription.approved_payload = None
        prescription.status = "clinician_approved"

    prescription.reviewed_by_clinician_id = clinician.id
    prescription.reviewed_at = models_utcnow()
    prescription.review_note = (body.note or "").strip() or None

    for row in audit:
        db.add(row)
    db.commit()
    db.refresh(prescription)

    logger.info(
        "Prescription reviewed | id=%s clinician=%s status=%s changes=%d ceiling=%s→%s",
        prescription.id, clinician.id, prescription.status, len(audit), model_ceiling, ceiling,
    )
    return _detail(db, prescription)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


def _fallback_model_version() -> str:
    """The constant to report when no checkpoint has been loaded."""
    try:
        from model.inference import MODEL_VERSION
        return MODEL_VERSION
    except ImportError:
        # torch is not installed. The instance cannot serve predictions, but
        # /health answering is exactly how that gets noticed.
        return "unavailable"


@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Liveness and model status",
    tags           = ["meta"],
)
async def health() -> HealthResponse:
    return HealthResponse(
        status        = "ok",
        model_loaded  = classifier is not None,
        # Importing here rather than at module scope: /health has to answer
        # even on an instance where the model failed to load.
        model_version = classifier.model_version if classifier else _fallback_model_version(),
        demo_mode     = classifier.demo_mode if classifier else True,
        # Surfaced so a deployment can be checked for these without reading logs:
        # an uncalibrated or unscreened model is servable but should not be
        # quoting confidence percentages at patients.
        calibrated    = bool(classifier and classifier.calibration["calibrated"]),
        ood_screening = bool(classifier and classifier.calibration["reject_threshold"] is not None),
    )


@app.get(
    "/exercises",
    response_model = ExercisesResponse,
    summary        = "Get exercise list for a surgery type and phase (no X-ray needed)",
    tags           = ["exercises"],
)
async def get_exercises(
    surgery_type: SurgeryType = Query(
        ...,
        description = "Surgery or injury type",
    ),
    weeks_post_op: Optional[int] = Query(
        None,
        ge          = 0,
        le          = 520,
        description = "Weeks since surgery. Omit for surgery_type=none",
    ),
    kl_grade: int = Query(
        0,
        ge          = 0,
        le          = 4,
        description = "KL Grade 0–4 (used to cap exercise angle limits). Defaults to 0 (no restriction).",
    ),
) -> ExercisesResponse:
    """
    Returns the exercise protocol for a given surgery type, recovery week, and
    KL grade — without requiring an X-ray upload.

    Useful for:
      - Frontend development and testing
      - Physiotherapists who want to browse protocols
      - Patients who don't have an X-ray available
    """
    if surgery_type != SurgeryType.none and weeks_post_op is None:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = f"'weeks_post_op' is required when surgery_type is '{surgery_type.value}'.",
        )
    if surgery_type == SurgeryType.none:
        weeks_post_op = None

    return get_exercises_only(
        surgery_type  = surgery_type.value,
        weeks_post_op = weeks_post_op,
        kl_grade      = kl_grade,
    )


@app.post(
    "/analyse-xray",
    response_model = AnalyseXrayResponse,
    summary        = "Analyse a knee X-ray and return a personalised rehab prescription",
    tags           = ["analysis"],
)
async def analyse_xray(
    request: Request,
    # Optional on purpose: login.html offers a guest route, and an anonymous
    # analysis is still worth returning. Signing in is what makes it persist.
    patient: Optional[Patient] = Depends(auth.optional_patient),
    db: Session = Depends(get_db),
    image: UploadFile = File(
        ...,
        description = "Knee X-ray image — JPEG or PNG, ≤ 10 MB.",
    ),
    knee_side: KneeSide = Form(
        ...,
        description = "Which knee: left, right, or both.",
    ),
    surgery_type: SurgeryType = Form(
        ...,
        description = "Surgery/injury type: acl | tkr | meniscus | arthroscopy | none.",
    ),
    weeks_post_op: Optional[int] = Form(
        None,
        ge          = 0,
        le          = 520,
        description = "Weeks since surgery. Required if surgery_type is not 'none'.",
    ),
) -> AnalyseXrayResponse:

    # ── 0. Rate limit ────────────────────────────────────────────────────────
    # Before any work: this endpoint runs a B4 forward pass per request.
    await enforce_rate_limit(request)

    # ── 1. Model ready guard ─────────────────────────────────────────────────
    if classifier is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Model is still loading. Retry in a moment.",
        )

    # ── 2. File type check ───────────────────────────────────────────────────
    allowed_types = ("image/jpeg", "image/png", "image/jpg")
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = f"Unsupported file type '{image.content_type}'. Upload JPEG or PNG.",
        )

    # ── 3. File size check ───────────────────────────────────────────────────
    # Reject an oversized declaration before reading a byte, then enforce the
    # real ceiling while streaming — Content-Length is a hint, not a guarantee.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES * 2:
        raise HTTPException(
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail      = f"Request exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    image_bytes = await read_capped(image, MAX_UPLOAD_BYTES)

    # ── 4. Image quality check ───────────────────────────────────────────────
    ok, quality_msg = validate_image(image_bytes)
    if not ok:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = quality_msg,
        )

    # ── 5. Input logic validation ────────────────────────────────────────────
    # A signed-in patient with a recorded operation does not have to send this:
    # the date is on file, so the week is arithmetic rather than something to
    # remember. An explicit value still wins — the caller may be correcting it.
    derived_weeks = False
    if weeks_post_op is None and patient is not None and patient.surgery_date is not None:
        weeks_post_op = patient.weeks_post_op
        derived_weeks = weeks_post_op is not None

    if surgery_type != SurgeryType.none and weeks_post_op is None:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = (
                f"'weeks_post_op' is required when surgery_type is '{surgery_type.value}'. "
                "Enter 0 if you are in the first week post-op."
            ),
        )
    if surgery_type == SurgeryType.none:
        weeks_post_op = None

    # ── 6. Model inference ───────────────────────────────────────────────────
    try:
        result = classifier.predict(image_bytes)
    except Exception:
        logger.exception("Inference failed for an uploaded image")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "Inference failed. Please try a different image.",
        )

    # ── 6b. Out-of-distribution guard ────────────────────────────────────────
    # The classifier has five outputs and no "not a knee" class, so nothing else
    # stops a chest X-ray or a photo of a wall returning a confident grade that
    # then sets a movement ceiling. Inert until the checkpoint carries an energy
    # reference (see model/inference.load_calibration).
    if result.get("ood_reject"):
        logger.warning(
            "OOD reject | energy=%s exceeds the reference for this checkpoint",
            result.get("energy"),
        )
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = (
                "This image does not look like a knee X-ray to the model, so it has not "
                "been graded. Please check you uploaded the right file — a plain "
                "anteroposterior (front-on) knee radiograph works best."
            ),
        )

    # ── 7. Build prescription ────────────────────────────────────────────────
    prescription = build_prescription(
        kl_grade        = result["kl_grade"],
        health_score    = result["health_score"],
        max_angle       = result["max_angle"],
        confidence      = result["confidence"],
        confidence_band = result["confidence_band"],
        calibrated      = result["calibrated"],
        ood_suspected   = result["ood_suspected"],
        demo_mode       = result["demo_mode"],
        knee_side       = knee_side.value,
        surgery_type    = surgery_type.value,
        weeks_post_op   = weeks_post_op,
        model_version   = classifier.model_version,
    )

    logger.info(
        "Prescription | knee=%s surgery=%s weeks=%s%s kl=%d angle=%d° phase=%s demo=%s patient=%s",
        knee_side.value, surgery_type.value, weeks_post_op,
        " (from surgery date)" if derived_weeks else "",
        result["kl_grade"], result["max_angle"],
        prescription["rehab_phase"], result["demo_mode"],
        patient.id if patient else "guest",
    )

    # ── 8. Persist, for a signed-in patient ──────────────────────────────────
    # The X-ray itself is not stored — only what was read off it. Keeping the
    # image would turn this into a system holding diagnostic scans, which is a
    # far larger promise about retention and access than the app makes today.
    if patient is not None:
        prescription["prescription_id"] = await run_in_threadpool(
            _save_prescription, db, patient, prescription
        )

    return AnalyseXrayResponse(**prescription)


def _save_prescription(db: Session, patient: Patient, prescription: dict) -> Optional[str]:
    """
    Store one analysis and return its id, or None if it could not be stored.

    A database problem must not cost the patient the reading they just waited
    for, so this never raises — but it does not lie about it either: the id
    comes back null and the response says the result was not kept.
    """
    try:
        row = Prescription(
            patient_id    = patient.id,
            kl_grade      = prescription["kl_grade"],
            health_score  = prescription["health_score"],
            max_angle     = prescription["max_angle"],
            knee_side     = prescription["knee_side"],
            surgery_type  = prescription["surgery_type"],
            weeks_post_op = prescription["weeks_post_op"],
            rehab_phase   = prescription["rehab_phase"],
            model_version = prescription["model_version"],
            demo_mode     = prescription["demo_mode"],
            # Verbatim, so a stored analysis still renders correctly after the
            # exercise database or the wording of the rationale changes.
            payload       = json.dumps(prescription, default=str),
        )
        db.add(row)
        db.flush()

        # The first entry in the audit trail is the model's. A clinician reading
        # it later should not have to infer where the original ceiling came
        # from — "the machine decided" is a decision, and it is recorded as one.
        db.add(PrescriptionAudit(
            prescription_id = row.id,
            field           = "ceiling",
            previous_value  = None,
            new_value       = str(prescription["max_angle"]),
            actor           = "model",
            reason          = (
                f"KL grade {prescription['kl_grade']} read by {prescription['model_version']}"
                + (" (demo mode — not a real reading)" if prescription["demo_mode"] else "")
            ),
        ))
        db.commit()
        return row.id
    except Exception:
        logger.exception("Could not save the prescription for patient %s", patient.id)
        db.rollback()
        return None


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)