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

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from clinical_logic import build_prescription, get_exercises_only
from model.inference import KneeClassifier, MODEL_VERSION, validate_image
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
    logger.info("Shutting down PhysioAI backend.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,"
    "http://localhost:8080,http://127.0.0.1:8080,"
    "http://localhost:5500,http://127.0.0.1:5500,"
    "null",
).split(",")

app = FastAPI(
    title       = "PhysioAI API",
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
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


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
        model_version = MODEL_VERSION,
        demo_mode     = classifier.demo_mode if classifier else True,
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

    # ── 3. File size check (10 MB) ───────────────────────────────────────────
    image_bytes = await image.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail      = "Image exceeds 10 MB. Please compress and retry.",
        )

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
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "Inference failed. Please try a different image.",
        )

    # ── 7. Build prescription ────────────────────────────────────────────────
    prescription = build_prescription(
        kl_grade      = result["kl_grade"],
        health_score  = result["health_score"],
        max_angle     = result["max_angle"],
        confidence    = result["confidence"],
        demo_mode     = result["demo_mode"],
        knee_side     = knee_side.value,
        surgery_type  = surgery_type.value,
        weeks_post_op = weeks_post_op,
        model_version = MODEL_VERSION,
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