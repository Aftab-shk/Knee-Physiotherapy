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
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
import unicodedata
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

import mailer
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from model.image_checks import validate_image
from model.prosthesis import detect_hardware
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import auth
import outcome_measures
import summary_pdf
import triage
from clinical_logic import build_prescription, get_exercises_only, merge_bilateral
from db import get_db, init_db
from exercise_protocols import all_exercises
from models import (
    CareLink,
    Clinician,
    ExerciseSession,
    ExerciseSet,
    OutcomeScore,
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
    ChangePassword,
    ClinicianOut,
    ClinicianRegister,
    ClinicianToken,
    DeleteAccount,
    DeletionReceipt,
    ExerciseBreakdown,
    ExercisesResponse,
    FlagOut,
    ForgotPassword,
    HealthResponse,
    InviteCreate,
    InviteCreated,
    InviteOut,
    KneeSide,
    LoginRequest,
    OutcomeChange,
    OutcomeHistory,
    OutcomeInstrument,
    OutcomePoint,
    OutcomeSchedule,
    OutcomeScoreCreate,
    OutcomeScoreOut,
    OutcomeSeries,
    PainPoint,
    PatientFlag,
    PatientFlags,
    PatientFlagsForClinician,
    PatientOut,
    PrescriptionDetail,
    PrescriptionEffective,
    PrescriptionHistory,
    PrescriptionSummary,
    PrescriptionSummaryForClinician,
    ProgressResponse,
    ProgressSummary,
    RedeemInvite,
    RegisterRequest,
    ResetPassword,
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
    SimpleMessage,
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


def refuse_unsafe_production() -> None:
    """
    Two states that are worse than not starting at all, once ENV=production.

    A generated JWT_SECRET signs every user out on each restart — and each
    worker in the same container signs its tokens with a different key, so
    logins fail at random rather than consistently. Demo mode returns
    deterministic mock KL grades through the same fields, with the same shape,
    as a real reading: nobody looking at the app can tell the difference.

    Both are warnings in development, where they are the point. In production
    they are a refusal — the failure is silent otherwise, and the thing being
    got wrong is a clinical number.
    """
    if os.getenv("ENV", "").lower() not in ("production", "prod"):
        return
    problems = []
    if auth.JWT_SECRET_IS_EPHEMERAL:
        problems.append("JWT_SECRET is not set")
    if classifier is None:
        problems.append("no classifier — torch is not installed")
    elif classifier.demo_mode:
        problems.append(
            "no usable checkpoint: demo mode returns mock grades that look like readings"
        )
    if mailer.backend() == "log":
        # The log backend writes a working password-reset link into the
        # application log. That is the right behaviour on a laptop and an
        # account takeover waiting to happen anywhere a log is shipped,
        # aggregated or read by more than one person.
        problems.append(
            "MAIL_BACKEND is 'log', which prints password reset links into the log; "
            "set MAIL_BACKEND=smtp and the SMTP_* variables"
        )
    if problems:
        raise RuntimeError("Refusing to start with ENV=production — " + "; ".join(problems))


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
        refuse_unsafe_production()
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
    refuse_unsafe_production()
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

# Where the frontend lives, when this process is the thing serving it.
#
# Serving the pages from the API makes them same-origin, which removes three
# separate deployment failures at once: a CORS allow-list that has to be kept in
# step with the site's hostname, config.js guessing at an API port from the
# page's hostname, and the browser blocking that guess as mixed content when the
# page is https and the guess was http. Unset (or pointed elsewhere) in
# development, where a separate static server holds the pages.
_frontend_dir = Path(os.getenv("FRONTEND_DIR", Path(__file__).resolve().parent.parent / "frontend"))
FRONTEND_DIR = _frontend_dir if (_frontend_dir / "index.html").is_file() else None

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
    # PATCH and DELETE are not optional extras here: without them the browser
    # refuses the preflight, and recording a surgery date, revoking a share link,
    # withdrawing a clinician's access and discharging a patient all fail from
    # the frontend while working perfectly from curl. Every method this API
    # actually routes has to appear, or the endpoint may as well not exist.
    allow_methods     = ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    # Authorization carries the bearer token from POST /auth/login. It is not
    # a "credential" in the CORS sense — no cookie rides along — so
    # allow_credentials stays off and the "null" origin footgun stays shut.
    allow_headers     = ["Content-Type", "Authorization"],
    expose_headers    = ["X-Request-ID", "X-Response-Time"],
    max_age           = 600,
)


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

# Now that this process serves the pages as well as the API, these are the app's
# headers, not just an API's. The bearer token lives in localStorage, so the
# policy below is what limits the blast radius if a script ever does get in.
#
# The allowances are all load-bearing: MediaPipe's pose code and wasm come from
# jsDelivr and its model from Google's storage, the fonts come from Google, the
# tracker builds blob: workers, and upload.html previews the chosen X-ray from a
# blob URL. 'unsafe-inline' is there because the pages are written as inline
# script and style throughout; removing it is a refactor, not a header change,
# and is the one real gap left in this policy.
_CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data: blob:",
    "media-src 'self' blob:",
    "worker-src 'self' blob:",
    "connect-src 'self' https://cdn.jsdelivr.net https://storage.googleapis.com",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "object-src 'none'",
])

# Set only when the request already arrived over https, so a plain-http
# development server does not pin a browser to a scheme it cannot serve.
_HSTS = "max-age=31536000; includeSubDomains"


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", _CSP)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # The tracker needs the camera; nothing here needs anything else.
    response.headers.setdefault(
        "Permissions-Policy", "camera=(self), microphone=(), geolocation=(), interest-cohort=()"
    )
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    if request.url.scheme == "https" or forwarded_proto == "https":
        response.headers.setdefault("Strict-Transport-Security", _HSTS)
    return response


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
    token, expires_in = auth.create_access_token(patient.id, token_version=patient.token_version)
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


# ---------------------------------------------------------------------------
# Keeping, changing and ending an account
# ---------------------------------------------------------------------------
#
# A JWT is valid until it expires, so "sign out" used to mean the browser threw
# its copy away while the token carried on working for the rest of its
# fortnight. Everything below that ends a session does it by bumping the
# account's token_version, which takes back every token issued before that
# moment — on every device, which is the only granularity worth having after a
# password change or a lost phone.


def _revoke_and_reissue(db: Session, patient: Patient) -> TokenResponse:
    """End every existing session, then hand this caller a fresh token."""
    auth.revoke_tokens(patient)
    db.commit()
    return _issue_token(patient)


@app.post(
    "/auth/logout",
    response_model = SimpleMessage,
    summary        = "Sign out of every device",
    tags           = ["accounts"],
)
def logout(
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> SimpleMessage:
    """
    Signs out everywhere, not just here: there is no per-device identity in the
    token to sign out of, and after a lost phone "everywhere" is the answer
    anyone actually wants.
    """
    auth.revoke_tokens(patient)
    db.commit()
    logger.info("Patient %s signed out of all sessions", patient.id)
    return SimpleMessage(detail="Signed out on every device.")


@app.post(
    "/auth/change-password",
    response_model = TokenResponse,
    summary        = "Change your password",
    tags           = ["accounts"],
)
def change_password(
    body: ChangePassword,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> TokenResponse:
    """
    The current password is required even though the caller is already signed
    in: the case this endpoint exists for is a session someone else left open.
    """
    if not auth.verify_password(patient.password_hash, body.current_password):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "That is not your current password.",
        )
    patient.password_hash = auth.hash_password(body.new_password)
    # Whoever knew the old password is signed out by this, which is the point.
    logger.info("Patient %s changed their password", patient.id)
    return _revoke_and_reissue(db, patient)


@app.post(
    "/auth/forgot-password",
    response_model = SimpleMessage,
    status_code    = status.HTTP_202_ACCEPTED,
    summary        = "Ask for a password reset link",
    tags           = ["accounts"],
)
def forgot_password(
    body: ForgotPassword,
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> SimpleMessage:
    """
    Always 202, whether or not the address has an account.

    The more helpful-looking alternative — "no account with that address" —
    turns this form into a way to ask the server whether a particular person is
    a patient here, which is exactly the fact a medical app must not confirm to
    a stranger.
    """
    email = auth.normalise_email(body.email)
    patient = auth.find_by_email(db, email)

    if patient is not None:
        token, token_hash = auth.new_reset_token()
        patient.reset_token_hash = token_hash
        patient.reset_requested_at = models_utcnow()
        db.commit()
        # Sent after the commit: a delivery that fails must not leave a live
        # token the database never recorded.
        mailer.send_password_reset(patient.email, token, patient.display_name)
    else:
        # The same shape of work, so the response time does not answer the
        # question the status code refuses to.
        auth.waste_time_like_a_real_verify()

    return SimpleMessage(
        detail="If that address has an account, a reset link is on its way. It expires in an hour."
    )


@app.post(
    "/auth/reset-password",
    response_model = TokenResponse,
    summary        = "Set a new password using a reset link",
    tags           = ["accounts"],
)
def reset_password(
    body: ResetPassword,
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> TokenResponse:
    token_hash = auth.hash_reset_token(body.token)
    patient = db.scalar(select(Patient).where(Patient.reset_token_hash == token_hash))

    if patient is None or not auth.reset_token_is_live(patient):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "That reset link is no longer valid. Ask for a new one.",
        )

    patient.password_hash = auth.hash_password(body.new_password)
    # Single use. Clearing it here is what stops the same link being replayed
    # out of a mailbox months later.
    patient.reset_token_hash = None
    patient.reset_requested_at = None
    logger.info("Patient %s completed a password reset", patient.id)
    # Anyone still holding a session from before the reset loses it, which is
    # the whole point when the reset was prompted by someone else having one.
    return _revoke_and_reissue(db, patient)


@app.get(
    "/me/export",
    summary = "Download everything held about you",
    tags    = ["accounts"],
)
def export_my_data(
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> Response:
    """
    Everything this system holds about one patient, as JSON.

    Built by reading each row's own columns rather than by listing fields here,
    so a column added later is exported without anyone remembering to come back
    and add it. The X-ray images are not in it because they were never stored —
    only what was read off them.
    """
    def rows(instances) -> list:
        out = []
        for obj in instances:
            record = {}
            for column in obj.__table__.columns:
                value = getattr(obj, column.name)
                record[column.name] = value.isoformat() if hasattr(value, "isoformat") else value
            out.append(record)
        return out

    prescriptions = list(patient.prescriptions)
    sessions = db.scalars(
        select(ExerciseSession).where(ExerciseSession.patient_id == patient.id)
    ).all()

    payload = {
        "exported_at":        models_utcnow().isoformat(),
        "account":            rows([patient])[0],
        "prescriptions":      rows(prescriptions),
        "prescription_audit": rows([a for pres in prescriptions for a in pres.audit]),
        "exercise_sessions":  rows(sessions),
        "exercise_sets":      rows([st for sess in sessions for st in sess.sets]),
        "outcome_scores":     rows(list(patient.outcome_scores)),
        "share_links":        rows(list(patient.share_links)),
        "clinician_links":    rows(list(patient.care_links)),
    }
    # Credentials are not facts about the patient, and a downloaded copy of one
    # is a liability to whoever downloaded it.
    for secret in ("password_hash", "reset_token_hash"):
        payload["account"].pop(secret, None)
    for link in payload["share_links"]:
        link.pop("token_hash", None)
    for link in payload["clinician_links"]:
        link.pop("invite_code_hash", None)

    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    stamp = models_utcnow().strftime("%Y-%m-%d")
    return Response(
        content    = body,
        media_type = "application/json",
        headers    = {"Content-Disposition": f'attachment; filename="physio-data-{stamp}.json"'},
    )


@app.post(
    "/me/delete",
    response_model = DeletionReceipt,
    summary        = "Delete your account and everything in it",
    tags           = ["accounts"],
)
def delete_my_account(
    body: DeleteAccount,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> DeletionReceipt:
    """
    Immediate and complete. No grace period, no tombstone, no anonymised
    remainder kept for analytics.

    That is a retention policy, and it is the one that matches what this app
    already promises: the X-ray was never stored, and the readings taken off it
    belong to the patient. Every child row goes with the account —
    prescriptions and their audit trail, sessions, sets, outcome scores, share
    links, and the access any clinician had. A clinician's notes live inside the
    prescription they annotated and go too, so nothing is left pointing at
    someone who asked to be forgotten.

    Anyone deploying this where clinical records must be retained for a fixed
    number of years has a different policy to implement, and this is the
    function to change.

    POST rather than DELETE because it carries a body, and a body on DELETE is
    allowed by the spec but dropped by enough proxies to be a poor bet on the
    one request that must not half-happen.
    """
    if not auth.verify_password(patient.password_hash, body.password):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "That is not your password.",
        )

    # Counted before the delete, so the receipt says what actually went.
    session_count = db.scalar(
        select(func.count()).select_from(ExerciseSession)
        .where(ExerciseSession.patient_id == patient.id)
    ) or 0
    receipt = DeletionReceipt(
        detail          = "Your account and everything in it has been deleted.",
        prescriptions   = len(patient.prescriptions),
        sessions        = session_count,
        outcome_scores  = len(patient.outcome_scores),
        share_links     = len(patient.share_links),
        clinician_links = len(patient.care_links),
    )

    patient_id = patient.id
    db.delete(patient)
    db.commit()
    logger.info(
        "Deleted patient %s: %d prescriptions, %d sessions, %d outcome scores",
        patient_id, receipt.prescriptions, receipt.sessions, receipt.outcome_scores,
    )
    return receipt


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
    # Built field by field rather than validated straight off the ORM row, for
    # one reason: `max_angle` on the row is what the MODEL read off the X-ray,
    # and a clinician who lowered it afterwards is the number the patient must
    # actually follow. That override lives in the payload, so reading the column
    # would show this patient a ceiling nobody approved — the same mistake the
    # service worker refuses to make by never caching an API response.
    return PrescriptionHistory(
        count = len(rows),
        prescriptions = [
            PrescriptionSummary(
                id            = r.id,
                created_at    = r.created_at,
                kl_grade      = r.kl_grade,
                health_score  = r.health_score,
                max_angle     = json.loads(r.effective_payload).get("max_angle", r.max_angle),
                knee_side     = r.knee_side,
                surgery_type  = r.surgery_type,
                weeks_post_op = r.weeks_post_op,
                rehab_phase   = r.rehab_phase,
                model_version = r.model_version,
                demo_mode     = r.demo_mode,
                status        = r.status,
                reviewed_at   = r.reviewed_at,
            )
            for r in rows
        ],
    )


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
    # The exercise name and its ceiling both arrive from the browser. The name
    # is what a clinician reads back later, and an unknown one made the history
    # describe work that does not exist in any protocol. Match it against the
    # catalogue the prescription was drawn from.
    known = {e["name"]: e for e in all_exercises()}
    if body.exercise_name not in known:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = f"'{body.exercise_name}' is not an exercise in the catalogue.",
        )

    # The ceiling is a clinical number and the page is not where it is decided.
    # A tracker running an out-of-date prescription, or a hand-written request,
    # must not be able to file a set claiming a limit nobody prescribed. Only the
    # upper bound is checked: a patient's own ceiling is often lower than the
    # protocol's, because the KL grade capped it.
    catalogue_limit = known[body.exercise_name].get("protocol_angle_limit")
    if catalogue_limit is not None and body.angle_limit > catalogue_limit:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = (
                f"{body.exercise_name} has a ceiling of {catalogue_limit}°; "
                f"this set reported {body.angle_limit}°."
            ),
        )

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
# Patient-reported outcome measures
# ---------------------------------------------------------------------------
#
# Every other number in this API is something the app measured. This is the one
# the patient supplies, and it is the only one that answers the question they
# came with — whether the knee is actually getting better to live with. A joint
# that flexes to 120° and hurts on every stair is not a success, and no amount of
# goniometry says so.
#
# Two rules, both of which the endpoints below enforce rather than assume:
#
# 1. It changes nothing clinical on its own. A score never moves an exercise
#    ceiling. That number comes from the radiograph and the surgical protocol,
#    and a questionnaire is not evidence about what a joint can withstand. What a
#    falling score does is put a patient in front of a human — see triage.py.
#
# 2. It is never a gate. Nothing is withheld from someone who does not want to
#    answer seven questions, and there is no reminder that cannot be ignored.

def _score_out(row: OutcomeScore, previous: Optional[float] = None) -> OutcomeScoreOut:
    label, text = outcome_measures.band(row.interval_score)
    return OutcomeScoreOut(
        id             = row.id,
        instrument     = row.instrument,
        knee_side      = row.knee_side,
        recorded_at    = row.recorded_at,
        weeks_post_op  = row.weeks_post_op,
        raw_sum        = row.raw_sum,
        interval_score = row.interval_score,
        band           = label,
        band_text      = text,
        responses      = json.loads(row.responses),
        change         = OutcomeChange(**outcome_measures.change(row.interval_score, previous)),
    )


def _outcome_rows(db: Session, patient: Patient) -> list[OutcomeScore]:
    """
    Every questionnaire this patient has completed, oldest first.

    Not clipped to a date range, unlike the session queries. A questionnaire
    answered monthly produces three or four points in ninety days, and the
    baseline they are all read against is usually older than the window — a
    trend that dropped its own starting point would be worse than no trend. The
    table is a few rows per patient per year; there is nothing to save here.
    """
    return list(db.scalars(
        select(OutcomeScore)
        .where(OutcomeScore.patient_id == patient.id)
        .order_by(OutcomeScore.recorded_at)
    ).all())


def _outcome_series(rows: list[OutcomeScore], tz_offset_minutes: int) -> list[OutcomeSeries]:
    """
    One series per knee.

    Per side for the same reason range of motion is per exercise: KOOS-JR asks
    about "your knee", singular. Someone with two bad knees has two different
    answers, and averaging them hides the difference worth seeing.
    """
    by_side: dict = defaultdict(list)
    for row in rows:
        by_side[row.knee_side].append(row)

    series = []
    for side, side_rows in by_side.items():
        latest = side_rows[-1]
        baseline = side_rows[0]
        label, text = outcome_measures.band(latest.interval_score)

        series.append(OutcomeSeries(
            instrument = latest.instrument,
            knee_side  = side,
            count      = len(side_rows),
            baseline   = round(baseline.interval_score, 1),
            latest     = round(latest.interval_score, 1),
            best       = round(max(r.interval_score for r in side_rows), 1),
            latest_at  = latest.recorded_at,
            band       = label,
            band_text  = text,
            change_from_previous = OutcomeChange(**outcome_measures.change(
                latest.interval_score,
                side_rows[-2].interval_score if len(side_rows) > 1 else None,
            )),
            # With one score, baseline and latest are the same row, so this
            # reports "first" rather than a change of zero — which would read as
            # "no progress" for someone who has only just started.
            change_from_baseline = OutcomeChange(**outcome_measures.change(
                latest.interval_score,
                baseline.interval_score if len(side_rows) > 1 else None,
            )),
            points = [
                OutcomePoint(
                    date           = _local_date(r.recorded_at, tz_offset_minutes),
                    interval_score = round(r.interval_score, 1),
                    raw_sum        = r.raw_sum,
                    weeks_post_op  = r.weeks_post_op,
                )
                for r in side_rows
            ],
        ))

    # Most-answered first, matching how the range-of-motion series are ordered:
    # the one with the most history is the one worth charting by default.
    series.sort(key=lambda s: (-s.count, s.knee_side))
    return series


@app.get(
    "/outcome-measures",
    response_model = OutcomeInstrument,
    summary        = "The KOOS-JR questionnaire, as it should be asked",
    tags           = ["outcomes"],
)
def outcome_instrument(
    patient: Optional[Patient] = Depends(auth.optional_patient),
) -> OutcomeInstrument:
    """
    Serves the item wording, so the form has one source of truth.

    A frontend holding its own copy of the questions is a frontend that drifts,
    and a KOOS-JR whose items have been reworded is not a KOOS-JR — it is a
    bespoke survey whose scores only look comparable to everyone else's.

    Open without a token: the questionnaire is not private, and login.html has a
    guest route. Signing in only adds the applicability caveat, which needs to
    know what operation the patient had.
    """
    return OutcomeInstrument(**outcome_measures.definition(
        surgery_type = patient.surgery_type if patient else None,
    ))


@app.post(
    "/me/outcome-scores",
    response_model = OutcomeScoreOut,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Record a completed questionnaire",
    tags           = ["outcomes"],
)
def record_outcome_score(
    body: OutcomeScoreCreate,
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> OutcomeScoreOut:
    """
    Score the seven answers and keep both them and the result.

    The minimum interval is enforced here rather than left to the UI. Answered
    weekly, KOOS-JR becomes a mood reading — genuine week-to-week movement is
    smaller than the instrument can detect — and a chart of that noise would
    invite exactly the conclusions it cannot support.
    """
    try:
        scored = outcome_measures.score(body.responses, body.instrument.value)
    except ValueError as err:
        # The schema already bounds the list length and the integer range, so
        # this is the belt-and-braces path: outcome_measures is the authority on
        # what a valid response set is, and it says so in its own words.
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err))

    # Same knee only. Two knees are two questionnaires, and one answered today
    # must not suppress the other.
    previous = db.scalar(
        select(OutcomeScore)
        .where(
            OutcomeScore.patient_id == patient.id,
            OutcomeScore.instrument == body.instrument.value,
            OutcomeScore.knee_side == body.knee_side.value,
        )
        .order_by(OutcomeScore.recorded_at.desc())
    )

    plan = outcome_measures.schedule(previous.recorded_at if previous else None)
    if not plan["can_record"]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=plan["reason"])

    row = OutcomeScore(
        patient_id     = patient.id,
        instrument     = scored["instrument"],
        knee_side      = body.knee_side.value,
        responses      = json.dumps(list(body.responses)),
        raw_sum        = scored["raw_sum"],
        interval_score = scored["interval_score"],
        # Frozen, not derived on read: a patient who corrects their surgery date
        # a year from now must not retroactively move every questionnaire they
        # have answered to a different point in their recovery.
        weeks_post_op  = patient.weeks_post_op,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    logger.info(
        "Outcome score | patient=%s instrument=%s side=%s raw=%d score=%.1f (%s)",
        patient.id, row.instrument, row.knee_side, row.raw_sum, row.interval_score, scored["band"],
    )
    return _score_out(row, previous.interval_score if previous else None)


@app.get(
    "/me/outcome-scores",
    response_model = OutcomeHistory,
    summary        = "Your questionnaire scores, and whether another is due",
    tags           = ["outcomes"],
)
def my_outcome_scores(
    knee_side: Optional[KneeSide] = Query(
        None, description="Limit to one knee. Omit for every score on record."
    ),
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> OutcomeHistory:
    """
    Newest first, each carrying the change from the one before it.

    `schedule` is the part the UI acts on: it says whether to offer the
    questionnaire, and if not, why not in words a patient can read.
    """
    rows = _outcome_rows(db, patient)
    if knee_side is not None:
        rows = [r for r in rows if r.knee_side == knee_side.value]

    # Against the previous score for the SAME knee — a right-knee questionnaire
    # says nothing about the left.
    previous_by_side: dict = {}
    with_change: list[OutcomeScoreOut] = []
    for row in rows:
        with_change.append(_score_out(row, previous_by_side.get(row.knee_side)))
        previous_by_side[row.knee_side] = row.interval_score

    return OutcomeHistory(
        instrument = outcome_measures.KOOS_JR,
        # Follows whatever was asked for: filtered to one knee it is that knee's
        # schedule, unfiltered it is the most recent answer for any of them. The
        # POST is per knee, so a bilateral patient asks per knee here too.
        schedule   = OutcomeSchedule(**outcome_measures.schedule(
            rows[-1].recorded_at if rows else None
        )),
        count      = len(with_change),
        scores     = list(reversed(with_change)),
    )


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

    # What the patient says, alongside what the app measured. Deliberately whole
    # history rather than the selected range — see _outcome_rows.
    outcomes = _outcome_series(_outcome_rows(db, patient), tz_offset_minutes)
    # The headline figure follows the most recently answered knee, not the
    # longest series: "how is it now" is a question about the latest answer.
    newest_outcome = max(outcomes, key=lambda s: s.latest_at) if outcomes else None

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
            latest_outcome_score = newest_outcome.latest if newest_outcome else None,
            outcome_band         = newest_outcome.band if newest_outcome else None,
            outcome_recorded_at  = newest_outcome.latest_at if newest_outcome else None,
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
        outcome_measures = outcomes,
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
    token, expires_in = auth.create_access_token(
        clinician.id, role=auth.ROLE_CLINICIAN, token_version=clinician.token_version
    )
    return ClinicianToken(
        access_token = token,
        expires_in   = expires_in,
        clinician    = ClinicianOut.model_validate(clinician),
    )


# A clinician holds other people's records, so the same credential controls
# apply — arguably more so. The flows are the patient ones with the other table
# underneath; what is deliberately absent is self-deletion, because a caseload
# is not a clinician's own data to erase. Discharging each patient first is the
# route out, and that is already the DELETE on /clinician/patients/{link_id}.


@app.post(
    "/clinician/logout",
    response_model = SimpleMessage,
    summary        = "Sign out of every device",
    tags           = ["clinicians"],
)
def clinician_logout(
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> SimpleMessage:
    auth.revoke_tokens(clinician)
    db.commit()
    logger.info("Clinician %s signed out of all sessions", clinician.id)
    return SimpleMessage(detail="Signed out on every device.")


@app.post(
    "/clinician/change-password",
    response_model = ClinicianToken,
    summary        = "Change your password",
    tags           = ["clinicians"],
)
def clinician_change_password(
    body: ChangePassword,
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> ClinicianToken:
    if not auth.verify_password(clinician.password_hash, body.current_password):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "That is not your current password.",
        )
    clinician.password_hash = auth.hash_password(body.new_password)
    auth.revoke_tokens(clinician)
    db.commit()
    logger.info("Clinician %s changed their password", clinician.id)
    return _issue_clinician_token(clinician)


@app.post(
    "/clinician/forgot-password",
    response_model = SimpleMessage,
    status_code    = status.HTTP_202_ACCEPTED,
    summary        = "Ask for a password reset link",
    tags           = ["clinicians"],
)
def clinician_forgot_password(
    body: ForgotPassword,
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> SimpleMessage:
    clinician = auth.find_clinician_by_email(db, body.email)
    if clinician is not None:
        token, token_hash = auth.new_reset_token()
        clinician.reset_token_hash = token_hash
        clinician.reset_requested_at = models_utcnow()
        db.commit()
        mailer.send_password_reset(clinician.email, token, clinician.display_name)
    else:
        auth.waste_time_like_a_real_verify()
    return SimpleMessage(
        detail="If that address has an account, a reset link is on its way. It expires in an hour."
    )


@app.post(
    "/clinician/reset-password",
    response_model = ClinicianToken,
    summary        = "Set a new password using a reset link",
    tags           = ["clinicians"],
)
def clinician_reset_password(
    body: ResetPassword,
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> ClinicianToken:
    token_hash = auth.hash_reset_token(body.token)
    clinician = db.scalar(select(Clinician).where(Clinician.reset_token_hash == token_hash))

    if clinician is None or not auth.reset_token_is_live(clinician):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "That reset link is no longer valid. Ask for a new one.",
        )

    clinician.password_hash = auth.hash_password(body.new_password)
    clinician.reset_token_hash = None
    clinician.reset_requested_at = None
    auth.revoke_tokens(clinician)
    db.commit()
    logger.info("Clinician %s completed a password reset", clinician.id)
    return _issue_clinician_token(clinician)


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

    # The patient's own verdict, and which way it has moved. One query for the
    # whole caseload rather than one per patient, for the same reason the
    # sessions above are fetched in one go.
    outcome_rows = db.scalars(
        select(OutcomeScore)
        .where(OutcomeScore.patient_id.in_(patient_ids))
        .order_by(OutcomeScore.recorded_at)
    ).all()

    # Grouped by knee so a right-knee score is only ever compared against the
    # previous right-knee score. Rows arrive oldest-first, so each list's last
    # entry is that knee's latest.
    outcome_by_patient: dict = defaultdict(lambda: defaultdict(list))
    for row in outcome_rows:
        outcome_by_patient[row.patient_id][row.knee_side].append(row)

    def outcome_columns(patient_id: str) -> dict:
        """
        The caseload's three outcome fields for one patient.

        Reports the knee answered most recently rather than the longest series:
        on a list a clinician is scanning, "how is it now" is a question about
        the latest answer, whichever side it came from.
        """
        by_side = outcome_by_patient.get(patient_id)
        if not by_side:
            return {"latest_outcome_score": None, "outcome_recorded_at": None, "outcome_change": None}

        rows = max(by_side.values(), key=lambda r: _as_utc(r[-1].recorded_at))
        return {
            "latest_outcome_score": round(rows[-1].interval_score, 1),
            "outcome_recorded_at":  rows[-1].recorded_at,
            "outcome_change": (
                round(rows[-1].interval_score - rows[-2].interval_score, 1)
                if len(rows) > 1 else None
            ),
        }

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
                **outcome_columns(link.patient_id),
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
    # Outcome scores are exempt from the window on purpose. They are answered
    # every few weeks, so a 29-day slice would routinely hold one — and a rule
    # comparing a score against the best one before it needs both of them.
    return triage.evaluate(sessions, prescriptions, _outcome_rows(db, patient))


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
    "/clinician/patients/{patient_id}/outcome-scores",
    response_model = OutcomeHistory,
    summary        = "A patient's questionnaire scores",
    tags           = ["review"],
)
def patient_outcome_scores(
    patient_id: str,
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> OutcomeHistory:
    """
    Read-only, exactly like every other clinician view of a patient's record.

    A clinician cannot answer the questionnaire on the patient's behalf. The
    whole value of a PROM is whose report it is, and a score filled in by
    somebody else is not a patient-reported outcome — it is an opinion wearing
    a registry-comparable number.
    """
    patient = _linked_patient(db, clinician, patient_id)
    rows = _outcome_rows(db, patient)

    previous_by_side: dict = {}
    with_change: list[OutcomeScoreOut] = []
    for row in rows:
        with_change.append(_score_out(row, previous_by_side.get(row.knee_side)))
        previous_by_side[row.knee_side] = row.interval_score

    return OutcomeHistory(
        instrument = outcome_measures.KOOS_JR,
        schedule   = OutcomeSchedule(**outcome_measures.schedule(
            rows[-1].recorded_at if rows else None
        )),
        count      = len(with_change),
        scores     = list(reversed(with_change)),
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
# Appointment summary PDF
# ---------------------------------------------------------------------------
#
# The offline half of the share link. A link works when the physiotherapist has
# a screen and a spare hand; a great many appointments have neither, and the
# patient arrives with a phone on 4% and no signal. This is the sheet they print
# the night before.
#
# Everything on it already exists — build_progress, the triage rules, the
# prescription in force. The only new decisions are what to leave out, since it
# has to fit on one page, and who is allowed to see a name.
#
# Which flags to print is the one judgement worth writing down. The patient's own
# list is used, not the clinician's: the two differ only in that the clinician's
# includes workflow findings like "this draft has been sitting unread for three
# days", and an unreviewed plan is already stated at the foot of the page,
# straight from the prescription rather than from a rule with a timeout in it.

# Deliberately ASCII-only by construction. A display name is user input on its
# way into a response header, and a header is exactly where a stray newline
# stops being cosmetic.
_FILENAME_STRIP = re.compile(r"[^A-Za-z0-9]+")


def _summary_filename(name: Optional[str], when: date) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    )
    slug = _FILENAME_STRIP.sub("-", ascii_name).strip("-").lower()[:40].strip("-")
    return f"physio-summary-{slug}-{when:%Y-%m-%d}.pdf" if slug else f"physio-summary-{when:%Y-%m-%d}.pdf"


def _pdf_response(pdf: bytes, name: Optional[str], when: date) -> Response:
    """
    Attach the PDF under a filename that says whose it is and when it was made.

    Two filenames, per RFC 6266: an ASCII one every client understands, and a
    percent-encoded UTF-8 one for those that do. The second is worth the four
    lines — it is the only part of this feature that can carry a name the
    embedded font has no glyphs for.
    """
    filename = _summary_filename(name, when)
    disposition = f'attachment; filename="{filename}"'
    if name:
        # Percent-encoding already makes this inert as a header — a newline
        # arrives as %0D%0A and stays there. Control characters are stripped
        # anyway so that what the browser *decodes* is a sane filename rather
        # than a smuggled header the user has to look at in a save dialog.
        clean = "".join(ch for ch in name if ch.isprintable())
        encoded = quote(f"physio-summary-{clean}-{when:%Y-%m-%d}.pdf", safe="")
        disposition += f"; filename*=UTF-8''{encoded}"

    return Response(
        content    = pdf,
        media_type = "application/pdf",
        headers    = {
            "Content-Disposition": disposition,
            # It contains somebody's clinical history; it should not sit in a
            # shared cache or a proxy on the way back.
            "Cache-Control": "private, no-store",
        },
    )


def _current_plan(db: Session, patient: Patient) -> Optional[Prescription]:
    return db.scalar(
        select(Prescription)
        .where(Prescription.patient_id == patient.id)
        .order_by(Prescription.created_at.desc())
        .limit(1)
    )


def _summary_data(
    db: Session,
    patient: Patient,
    progress: ProgressResponse,
    *,
    patient_name: Optional[str],
) -> summary_pdf.SummaryData:
    """
    Assemble the sheet from what the app already knows.

    `patient_name` is passed in rather than read off the patient, because who is
    allowed to see a name differs by route: a share link discloses only what its
    own JSON view does, and never an email address.
    """
    plan = _current_plan(db, patient)
    ceiling = None
    if plan is not None:
        try:
            ceiling = json.loads(plan.effective_payload).get("max_angle")
        except (ValueError, AttributeError):
            # A payload that will not parse is a broken row, not a reason to
            # refuse the download; the sheet simply shows no ceiling.
            logger.warning("Prescription %s has an unreadable payload", plan.id)

    flags = [f for f in _evaluate_patient(db, patient) if f.patient_message]

    return summary_pdf.SummaryData(
        generated_at  = progress.generated_at,
        range_days    = progress.range_days,
        patient_name  = patient_name,
        surgery_type  = patient.surgery_type,
        surgery_date  = patient.surgery_date,
        weeks_post_op = patient.weeks_post_op,
        knee_side     = plan.knee_side if plan else None,

        ceiling_deg   = ceiling,
        has_plan      = plan is not None,
        plan_reviewed = bool(plan and plan.reviewed_at),
        reviewed_by   = (plan.reviewed_by.display_name or plan.reviewed_by.email)
                        if plan and plan.reviewed_by else None,
        reviewed_at   = plan.reviewed_at if plan else None,
        demo_mode     = bool(plan and plan.demo_mode),

        sessions            = progress.summary.sessions,
        active_days         = progress.summary.active_days,
        current_streak_days = progress.summary.current_streak_days,
        longest_streak_days = progress.summary.longest_streak_days,
        unverified_sessions = progress.summary.unverified_sessions,

        latest_pain_after = progress.summary.latest_pain_after,
        mean_pain_change  = progress.summary.mean_pain_change,

        rom = [
            summary_pdf.RomRow(
                exercise      = series.exercise_name,
                best_deg      = series.best_deg,
                latest_deg    = series.latest_deg,
                angle_limit   = series.angle_limit,
                days_measured = series.days_measured,
                points        = [(p.date, p.peak_flexion_deg, p.angle_limit) for p in series.points],
            )
            for series in progress.rom_by_exercise
        ],
        flags = [summary_pdf.FlagRow(severity=f.severity, summary=f.summary) for f in flags],
        outcomes = [
            summary_pdf.OutcomeRow(
                instrument  = outcome_measures.display_name(series.instrument),
                knee_side   = series.knee_side,
                latest      = series.latest,
                baseline    = series.baseline,
                band        = series.band_text,
                change_note = series.change_from_baseline.summary,
                recorded_on = series.points[-1].date,
            )
            for series in progress.outcome_measures
        ],
        # KOOS-JR was validated in osteoarthritis and joint replacement. Printing
        # a score from it after an ACL reconstruction without saying what it does
        # not ask about would be the sheet overstating its own evidence.
        outcome_caveat = outcome_measures.applicability_caveat(patient.surgery_type),
    )


def _render_summary(
    db: Session,
    patient: Patient,
    days: int,
    tz_offset_minutes: int,
    patient_name: Optional[str],
) -> Response:
    progress = build_progress(db, patient, days, tz_offset_minutes)
    data = _summary_data(db, patient, progress, patient_name=patient_name)
    return _pdf_response(summary_pdf.render(data), patient_name, progress.generated_at.date())


@app.get(
    "/me/summary.pdf",
    summary        = "One-page summary to bring to an appointment",
    tags           = ["sessions"],
    response_class = Response,
    responses      = {200: {"content": {"application/pdf": {}}, "description": "The summary"}},
)
def my_summary_pdf(
    days: int = Query(90, ge=1, le=365),
    tz_offset_minutes: int = Query(0, ge=-840, le=840),
    patient: Patient = Depends(auth.current_patient),
    db: Session = Depends(get_db),
) -> Response:
    """
    The patient's own copy.

    Falls back to the email address when no display name is set: this route is
    reachable only by the account holder, it is their own document, and a summary
    sheet with nobody's name on it is no use in a waiting room.
    """
    return _render_summary(
        db, patient, days, tz_offset_minutes,
        patient.display_name or patient.email,
    )


@app.get(
    "/share/{token}/summary.pdf",
    summary        = "Download a shared summary (no account needed)",
    tags           = ["sharing"],
    response_class = Response,
    responses      = {200: {"content": {"application/pdf": {}}, "description": "The summary"}},
)
def shared_summary_pdf(
    token: str,
    days: int = Query(90, ge=1, le=365),
    tz_offset_minutes: int = Query(0, ge=-840, le=840),
    db: Session = Depends(get_db),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> Response:
    """
    The same sheet, for whoever holds the link — so a clinician who opened it can
    put a copy in the patient's file.

    Discloses exactly what the shared JSON view does, which is a display name or
    nothing at all. Never the email address: the link is a bearer token, and
    handing an account identifier to anyone who finds it is how a read-only link
    turns into the start of an attack on the account.

    The same flat 404 as the JSON view, for the same reason.
    """
    link = db.scalar(select(ShareLink).where(ShareLink.token_hash == _hash_share_token(token)))
    if link is None or not link.is_active:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "This link is not valid. It may have expired or been withdrawn.",
        )

    link.access_count += 1
    link.last_accessed_at = models_utcnow()
    db.commit()

    return _render_summary(db, link.patient, days, tz_offset_minutes, link.patient.display_name)


@app.get(
    "/clinician/patients/{patient_id}/summary.pdf",
    summary        = "A patient's one-page summary",
    tags           = ["clinicians"],
    response_class = Response,
    responses      = {200: {"content": {"application/pdf": {}}, "description": "The summary"}},
)
def clinician_summary_pdf(
    patient_id: str,
    days: int = Query(90, ge=1, le=365),
    tz_offset_minutes: int = Query(0, ge=-840, le=840),
    clinician: Clinician = Depends(auth.current_clinician),
    db: Session = Depends(get_db),
) -> Response:
    """
    For the paper file, or to hand back at the end of the appointment.

    Names the patient the way the caseload does — display name, else the label
    this clinician gave them — and not by email, which the caseload deliberately
    withholds.
    """
    patient = _linked_patient(db, clinician, patient_id)
    link = db.scalar(
        select(CareLink).where(
            CareLink.clinician_id == clinician.id,
            CareLink.patient_id == patient.id,
        )
    )
    return _render_summary(
        db, patient, days, tz_offset_minutes,
        patient.display_name or (link.patient_label if link else None),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """The app when its pages are bundled here, the API docs when they are not."""
    if FRONTEND_DIR is not None:
        return FileResponse(FRONTEND_DIR / "index.html")
    return RedirectResponse(url="/docs")


def gradcam_explain(classifier, image_bytes: bytes):
    """
    Grad-CAM, imported at call time.

    Same reason the classifier is: model.gradcam imports torch, and this module
    has to stay importable without it (see the note on `classifier` above).
    """
    try:
        from model.gradcam import explain
    except ImportError:
        return None
    return explain(classifier, image_bytes)


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
        description = (
            "Knee X-ray image — JPEG or PNG, ≤ 10 MB. When knee_side is 'both', this is "
            "the LEFT knee and image_right carries the other."
        ),
    ),
    image_right: Optional[UploadFile] = File(
        None,
        description = (
            "The right knee's X-ray. Required when knee_side is 'both', ignored otherwise. "
            "Two knees are graded independently — one film cannot answer for both."
        ),
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
    explain: bool = Form(
        False,
        description = (
            "Return a Grad-CAM overlay showing where the model was looking. Off by "
            "default because it costs a backward pass — roughly doubling inference "
            "time — and nothing should pay that by accident."
        ),
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

    # ── 4. Input logic validation ────────────────────────────────────────────
    if knee_side == KneeSide.both and image_right is None:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = (
                "Assessing both knees needs two X-rays: send the left knee as 'image' and "
                "the right as 'image_right'. One film cannot be graded for both — two knees "
                "routinely differ by two grades and 45° of permitted flexion."
            ),
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

    # ── 6. Grade each knee ───────────────────────────────────────────────────
    # Once for one knee, twice for two. Each film is read, screened and
    # prescribed for on its own: two knees routinely differ by two grades, and
    # holding one to the other's ceiling is either unsafe or pointlessly
    # restrictive depending on which way round it is.
    if knee_side == KneeSide.both:
        left = await _grade_one_side(
            image, KneeSide.left, surgery_type, weeks_post_op, explain, label="Left X-ray",
        )
        right = await _grade_one_side(
            image_right, KneeSide.right, surgery_type, weeks_post_op, explain, label="Right X-ray",
        )
        prescription = merge_bilateral(left, right)
    else:
        prescription = await _grade_one_side(
            image, knee_side, surgery_type, weeks_post_op, explain,
        )

    logger.info(
        "Prescription | knee=%s surgery=%s weeks=%s%s kl=%d%s angle=%d° phase=%s demo=%s patient=%s",
        knee_side.value, surgery_type.value, weeks_post_op,
        " (from surgery date)" if derived_weeks else "",
        prescription["kl_grade"],
        "" if prescription["kl_applicable"] else " (not applied — replaced joint)",
        prescription["max_angle"],
        prescription["rehab_phase"], prescription["demo_mode"],
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


async def _grade_one_side(
    upload: UploadFile,
    knee_side: KneeSide,
    surgery_type: SurgeryType,
    weeks_post_op: Optional[int],
    explain: bool,
    label: str = "",
) -> dict:
    """
    Read one X-ray and turn it into a prescription for that knee.

    Everything from the quality check to the exercise list, for a single film.
    Split out of the endpoint so a bilateral request runs it twice rather than
    duplicating it — and so `label` can say which film a complaint is about,
    since "image contrast is too low" is not much help when two were sent.
    """
    prefix = f"{label}: " if label else ""

    image_bytes = await read_capped(upload, MAX_UPLOAD_BYTES)

    ok, quality_msg = validate_image(image_bytes)
    if not ok:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = prefix + quality_msg,
        )

    try:
        result = classifier.predict(image_bytes)
    except Exception:
        logger.exception("Inference failed for an uploaded image")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "Inference failed. Please try a different image.",
        )

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
            detail      = prefix + (
                "This image does not look like a knee X-ray to the model, so it has not "
                "been graded. Please check you uploaded the right file — a plain "
                "anteroposterior (front-on) knee radiograph works best."
            ),
        )

    # A declared TKR already tells us the joint is replaced; this is the backstop
    # for one nobody declared. Advisory only — see model/prosthesis.py.
    hardware = detect_hardware(image_bytes)
    if hardware["suspected"] and surgery_type != SurgeryType.tkr:
        logger.warning(
            "Possible undeclared joint replacement | knee=%s surgery=%s saturated=%.3f solidity=%.2f",
            knee_side.value, surgery_type.value,
            hardware["saturated_fraction"], hardware["solidity"],
        )

    # Opt-in, and never load-bearing: an overlay that fails must not cost the
    # patient the reading they waited for. gradcam.explain returns None rather
    # than raising, and the response simply carries no picture.
    explanation_uri = None
    if explain:
        overlay = await run_in_threadpool(gradcam_explain, classifier, image_bytes)
        if overlay is not None:
            explanation_uri = (
                "data:image/png;base64,"
                + base64.b64encode(overlay["overlay_png"]).decode("ascii")
            )

    prescription = build_prescription(
        kl_grade        = result["kl_grade"],
        health_score    = result["health_score"],
        max_angle       = result["max_angle"],
        confidence      = result["confidence"],
        confidence_band = result["confidence_band"],
        calibrated      = result["calibrated"],
        ood_suspected   = result["ood_suspected"],
        demo_mode       = result["demo_mode"],
        grade_probabilities = result.get("grade_probabilities"),
        within_one_grade    = result.get("within_one_grade", 0.0),
        hardware_suspected = hardware["suspected"],
        hardware_reason    = hardware["reason"] if hardware["suspected"] else None,
        knee_side       = knee_side.value,
        surgery_type    = surgery_type.value,
        weeks_post_op   = weeks_post_op,
        model_version   = classifier.model_version,
    )
    prescription["explanation"] = explanation_uri
    return prescription


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
# The frontend
# ---------------------------------------------------------------------------

# Registered last, deliberately: a mount at "/" matches by prefix and would
# shadow every route declared after it. Everything above wins, and only what no
# endpoint claimed falls through to a file on disk.
if FRONTEND_DIR is not None:
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info("Serving the frontend from %s", FRONTEND_DIR)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)