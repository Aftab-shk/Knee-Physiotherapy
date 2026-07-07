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
COPY . .

# ── Trained weights (optional at build time) ─────────────────────────────────
# To include weights in the image, uncomment:
#   COPY model/resnet50_kl_v1.pt model/resnet50_kl_v1.pt
# Otherwise mount at runtime:
#   docker run -v /path/to/weights:/app/model physio-backend

# ── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8000

ENV MODEL_PATH=model/resnet50_kl_v1.pt
ENV CORS_ORIGINS=http://localhost:5173,http://localhost:3000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
