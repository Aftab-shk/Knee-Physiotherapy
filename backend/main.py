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
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from model.inference import MODEL_VERSION, KneeClassifier, validate_image

from clinical_logic import build_prescription, get_exercises_only
from schemas import (
    AnalyseXrayResponse,
    ExercisesResponse,
    HealthResponse,
    KneeSide,
    SurgeryType,
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

classifier: Optional[KneeClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    logger.info("Loading KneeClassifier…")
    classifier = KneeClassifier()
    if classifier.demo_mode:
        logger.warning("⚠️  DEMO MODE — place trained weights at %s to enable real inference.",
                       os.getenv("MODEL_PATH", "model/efficientnet_b4_kl_v2.pt"))
    else:
        logger.info("✅  Model loaded — real inference active. Version: %s", MODEL_VERSION)
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
        "http://localhost:5500,http://127.0.0.1:5500",
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
    allow_headers     = ["Content-Type"],
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
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


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
        model_version = classifier.model_version if classifier else MODEL_VERSION,
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
        "Prescription | knee=%s surgery=%s weeks=%s kl=%d angle=%d° phase=%s demo=%s",
        knee_side.value, surgery_type.value, weeks_post_op,
        result["kl_grade"], result["max_angle"],
        prescription["rehab_phase"], result["demo_mode"],
    )

    return AnalyseXrayResponse(**prescription)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)