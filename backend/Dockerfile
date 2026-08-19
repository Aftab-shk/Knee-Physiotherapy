# ── Base image ──────────────────────────────────────────────────────────────
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

COPY requirements.txt .
# Install everything else (torch lines in requirements.txt are skipped by pip
# because the packages are already satisfied from the step above)
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
# .dockerignore keeps the ~70 MB checkpoint, datasets and local venvs out of
# this layer. Without it `COPY . .` bakes the weights into the image.
COPY . .

# ── Trained weights ──────────────────────────────────────────────────────────
# Not baked into the image — they are ~70 MB, change every training run, and are
# distributed via GitHub Releases (see model/fetch_weights.py). Mount at runtime:
#
#   docker run -v /path/to/weights:/weights \
#              -e MODEL_PATH=/weights/best_model.pth \
#              -p 8000:8000 physio-backend
#
# Without them the server starts in demo mode and says so on /health.

# ── Runtime ──────────────────────────────────────────────────────────────────
# Drop root: nothing here needs it, and the process handles user uploads.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Resolved relative to backend/model/ when not absolute.
ENV MODEL_PATH=best_model.pth
ENV CORS_ORIGINS=http://localhost:5173,http://localhost:3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
