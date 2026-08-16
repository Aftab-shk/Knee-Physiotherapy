# AI Knee Physiotherapy

AI-assisted knee rehabilitation system. Upload a knee X-ray, enter your surgery history, and get a personalised exercise programme — then follow along with a webcam-based safety tracker that alerts you in real time if you exceed a safe joint angle.

⚠️ **For informational purposes only.** Not a substitute for professional medical advice. Always consult a physiotherapist or surgeon before starting any exercise programme.

---

## How it works

Two independent inputs combine to build every prescription:

```
Knee X-ray (KL Grade 0–4)        →  sets the SAFE ANGLE CEILING (degrees)
Surgery type + weeks post-op     →  selects WHICH EXERCISES to prescribe
```

The ceiling caps every exercise's angle limit:
```
effective_angle_limit = min(protocol_angle_limit, kl_derived_max_angle)
```

A KL Grade 4 knee 2 weeks post-TKR and the same knee 12 weeks post-TKR have identical X-rays but need completely different exercises — the surgery timeline picks the exercises, the X-ray restricts how far to move.

---

## File structure

```
knee-physiotherapy/
│
├── backend/
│   ├── main.py                    FastAPI app — all endpoints
│   ├── schemas.py                 Pydantic request/response models
│   ├── clinical_logic.py          Core prescription-building logic
│   ├── exercise_protocols.py      Exercise database (4 surgery types + none)
│   │
│   ├── model/
│   │   ├── inference.py           EfficientNet (B0–B5) inference + demo-mode fallback
│   │   ├── train.py               Training script (run on GPU / Kaggle)
│   │   ├── prepare_dataset.py     Splits raw dataset → train/val/test
│   │   ├── TRAINING.md            Training recipe, flags, and how to read results
│   │   └── best_model.pth   ⚠️ NOT in git — see Model Weights below
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .gitignore
│   └── README.md                  Backend-specific docs
│
└── frontend/
    ├── index.html                 Landing page
    ├── login.html                 Login / guest access
    ├── upload.html                X-ray upload + results + exercise plan
    ├── tracker.html                Webcam safety tracker (MediaPipe pose landmarker)
    ├── logo.png / logo-dark.png   Branding assets
```

> Note: the frontend lives at the repo root (`frontend/`), not inside `backend/`.

---

## Quick start

### 1. Backend

```bash
cd backend
python -m venv physio-env

# Windows
physio-env\Scripts\activate
# Mac/Linux
source physio-env/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

Visit `http://127.0.0.1:8000/docs` for the interactive API docs.

The server runs in **demo mode** until trained weights are placed at `backend/model/best_model.pth` — demo mode returns deterministic mock KL grades so the rest of the system (frontend, clinical logic, exercise selection) can be developed and tested without a trained model.

### 2. Frontend

```bash
cd frontend
python -m http.server 8080
```

Open `http://localhost:8080/index.html` in your browser. Do **not** open the HTML files by double-clicking them — the browser blocks API requests from `file://` origins in some configurations. Always serve via a local HTTP server.

The webcam tracker (`tracker.html`) needs camera permission and loads its pose-detection model (MediaPipe Tasks Vision, pose landmarker lite) from a CDN — it runs entirely client-side and does not call the backend.

### 3. Model weights

Not included in this repo (model files are large and don't belong in git). See **[backend/model/TRAINING.md](backend/model/TRAINING.md)** for the current training recipe (EfficientNet-B4, tuned for this dataset) and step-by-step Kaggle instructions. Once trained, place the checkpoint at:
```
backend/model/best_model.pth
```
Restart the server — it auto-detects the file and switches out of demo mode. The checkpoint carries its own architecture, input resolution, and preprocessing flags, so nothing else needs to be configured to match it.

---

## API reference

### `POST /analyse-xray`
Multipart form: `image` (file, JPEG/PNG ≤10MB), `knee_side` (left/right/both), `surgery_type` (acl/tkr/meniscus/arthroscopy/none), `weeks_post_op` (int, required unless surgery_type=none).

Returns KL grade, health score, safe angle ceiling, confidence, rehab phase, a full exercise list with per-exercise angle limits, and a plain-English rationale.

### `GET /exercises`
Query params: `surgery_type` (required), `weeks_post_op` (required unless surgery_type=none), `kl_grade` (optional, default 0). Returns the exercise list without requiring an X-ray upload — useful for browsing protocols or frontend testing.

### `GET /health`
Returns `{ status, model_loaded, model_version, demo_mode }`.

Full schema and interactive testing at `/docs` once the server is running.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn, Pydantic v2 |
| Model | EfficientNet (torchvision), B4 by default, fine-tuned on KL grade data |
| Preprocessing | Optional OpenCV CLAHE, ImageNet normalisation, native per-architecture resolution |
| Frontend | Plain HTML/CSS/JS (no framework); MediaPipe Tasks Vision for pose tracking |
| Training | PyTorch, Kaggle (T4/P100 GPU) — MixUp/CutMix, layer-wise LR decay, EMA, ordinal loss, post-hoc class-prior correction |
| Container | Docker (CPU-only PyTorch build) |

---

## Project status

| Component | Status |
|---|---|
| Backend API | ✅ Complete |
| Clinical logic + exercise database | ✅ Complete |
| Frontend (landing, login, upload/results) | ✅ Complete |
| Webcam safety tracker (live angle + red-screen alert) | ✅ Complete |
| Model training pipeline | ✅ Complete — see [backend/model/TRAINING.md](backend/model/TRAINING.md) |
| Trained model weights | 🔄 In progress — current best ~71% test accuracy, iterating |
| Session analytics / progress reports | ⬜ Not yet built |

---

## License

Not yet decided — add a LICENSE file before making this public if you intend to open-source it.
