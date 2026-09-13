# ── Base image ──────────────────────────────────────────────────────────────
# Built from the repository root, not backend/: the image serves the frontend
# too, so both directories have to be in the build context.
#
#   docker build -t physio-backend .
#
FROM python:3.10-slim

WORKDIR /app

# ── System dependencies for OpenCV headless ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
# Install torch + torchvision CPU-only first (saves ~1.5 GB vs CUDA build)
RUN pip install --no-cache-dir \
        torch==2.3.0 \
        torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
# Install everything else (torch lines in requirements.txt are skipped by pip
# because the packages are already satisfied from the step above)
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
# .dockerignore keeps the ~70 MB checkpoint, datasets and local venvs out of
# these layers. Without it the weights get baked into the image.
COPY backend/ /app/
# The pages are served by this process, from FRONTEND_DIR. Same origin as the
# API, so there is no CORS list to keep in step with the site's hostname and no
# guessing at an API port from the page's URL.
COPY frontend/ /frontend/

# ── Trained weights ──────────────────────────────────────────────────────────
# Not baked into the image — they are ~70 MB, change every training run, and are
# distributed via GitHub Releases (see model/fetch_weights.py). Mount at runtime:
#
#   docker run -v /path/to/weights:/weights \
#              -e MODEL_PATH=/weights/best_model.pth \
#              -e JWT_SECRET="$(openssl rand -base64 48)" \
#              -e DATABASE_URL=postgresql+psycopg://user:pw@host/db \
#              -e ENV=production \
#              -p 8000:8000 physio-backend
#
# With ENV=production the server refuses to start without them, rather than
# quietly serving mock grades that look exactly like readings.

# ── Runtime ──────────────────────────────────────────────────────────────────
# Drop root: nothing here needs it, and the process handles user uploads.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app /frontend
USER appuser

EXPOSE 8000

# Resolved relative to backend/model/ when not absolute.
ENV MODEL_PATH=best_model.pth
ENV FRONTEND_DIR=/frontend
# Same-origin now, so the allow-list only matters if the pages are hosted
# elsewhere as well.
ENV CORS_ORIGINS=http://localhost:5173,http://localhost:3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# Migrations run once, here, before any worker exists. The app calls init_db()
# at startup too, but with --workers 2 that is two processes running `upgrade
# head` against the same database at the same time; doing it first makes each of
# those a no-op. --proxy-headers so request.client and the scheme come from the
# reverse proxy rather than reading as the proxy's own address — which assumes
# the container is only reachable through that proxy, not published directly.
CMD ["sh", "-c", "python -c 'from db import init_db; init_db()' && exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2 --proxy-headers --forwarded-allow-ips='*'"]
