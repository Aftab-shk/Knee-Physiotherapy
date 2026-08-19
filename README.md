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
│   │   ├── inference.py           EfficientNet (B0–B5) inference, calibration, OOD screen
│   │   ├── train.py               Training script (run on GPU / Kaggle)
│   │   ├── prepare_dataset.py     Splits raw dataset → train/val/test
│   │   ├── fetch_weights.py       Downloads + verifies the checkpoint
│   │   ├── weights.json           Checkpoint manifest (url, size, sha256)
│   │   ├── TRAINING.md            Training recipe, flags, and how to read results
│   │   └── best_model.pth   ⚠️ NOT in git — see Model weights below
│   │
│   ├── kl_constants.py            KL grade tables — single source of truth, torch-free
│   ├── tests/                     pytest — clinical logic, tracker, calibration, API
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── README.md                  Backend-specific docs
│
├── .github/workflows/ci.yml       lint + tests + large-file guard
├── ruff.toml                      lint config
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

Not in git — ~70 MB of binary that changes every training run. Fetch them:

```bash
python backend/model/fetch_weights.py
```

This reads [backend/model/weights.json](backend/model/weights.json), downloads the
checkpoint, and verifies its SHA-256. Override the source with `WEIGHTS_URL=<url>`,
or point the server at a checkpoint anywhere with `MODEL_PATH=/path/to/weights.pth`.

To train your own instead, see **[backend/model/TRAINING.md](backend/model/TRAINING.md)**
(EfficientNet-B4, Kaggle instructions) and drop the result at `backend/model/best_model.pth`.

Restart the server — it auto-detects the file and leaves demo mode. The checkpoint
carries its own architecture, input resolution, preprocessing flags, class priors,
and calibration, so nothing else needs configuring to match it.

**Publishing a new checkpoint:**
```bash
gh release create weights-v2 backend/model/best_model.pth --notes "EfficientNet-B4, calibrated"
python backend/model/fetch_weights.py --manifest backend/model/best_model.pth   # refresh size + sha256
# then set "url" in weights.json to the release asset
```

<details>
<summary><b>Purging the old weights from git history</b> (optional, rewrites history)</summary>

Weights are untracked going forward, but three checkpoints are still in past
commits — 173 MB of blobs that every `git clone` still downloads:

| Blob | Size |
|---|---|
| `backend/model/resnet50_kl_v1.pt` | 90.0 MB |
| `backend/model/best_model.pth` (v2) | 67.7 MB |
| `backend/model/best_model.pth` (v1) | 15.6 MB |

Removing them shrinks `.git` from ~163 MB to a few MB, but **rewrites every
commit SHA** and needs a force-push. Only worth it while you are the sole
contributor — anyone else with a clone must re-clone afterwards.

```bash
pip install git-filter-repo

# from a fresh clone, or pass --force
git filter-repo \
  --path backend/model/resnet50_kl_v1.pt \
  --path backend/model/best_model.pth \
  --invert-paths

git remote add origin https://github.com/Aftab-shk/Knee-Physiotherapy
git push --force --all
git push --force --tags
```

Back up the repo first — `git filter-repo` is not reversible.

</details>

---

## API reference

### `POST /analyse-xray`
Multipart form: `image` (file, JPEG/PNG ≤10MB), `knee_side` (left/right/both), `surgery_type` (acl/tkr/meniscus/arthroscopy/none), `weeks_post_op` (int, required unless surgery_type=none).

Returns KL grade, health score, safe angle ceiling, confidence, rehab phase, a full exercise list with per-exercise angle limits, and a plain-English rationale.

### `GET /exercises`
Query params: `surgery_type` (required), `weeks_post_op` (required unless surgery_type=none), `kl_grade` (optional, default 0). Returns the exercise list without requiring an X-ray upload — useful for browsing protocols or frontend testing.

### `GET /health`
Returns `{ status, model_loaded, model_version, demo_mode, calibrated, ood_screening }`.

`calibrated` and `ood_screening` report whether the loaded checkpoint carries a
fitted temperature and an OOD energy reference. Both false means the instance is
serving raw softmax confidence and cannot screen non-radiograph uploads.

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
| Serving | Flip-TTA, temperature-scaled confidence, energy-based OOD screening |
| Tests | pytest — clinical logic, tracker contract, calibration invariants |
| Container | Docker (CPU-only PyTorch build, non-root, healthcheck) |

---

## Project status

| Component | Status |
|---|---|
| Backend API | ✅ Complete |
| Clinical logic + exercise database | ✅ Complete |
| Frontend (landing, login, upload/results) | ✅ Complete |
| Webcam safety tracker (live angle + red-screen alert) | ✅ Complete |
| Model training pipeline | ✅ Complete — see [backend/model/TRAINING.md](backend/model/TRAINING.md) |
| Confidence calibration + OOD screening | ✅ Complete — temperature scaling, energy screen |
| Test suite | ✅ 253 tests — clinical logic, tracker contract, calibration, API security |
| CI | ✅ GitHub Actions — ruff + pytest, and a guard against large tracked files |
| Trained model weights | ✅ Calibrated — 70.3% test accuracy, 95.3% within-one-grade, ECE 0.041 |
| Real authentication | ⬜ Not built — `login.html` is a labelled demo, any credentials pass |
| Rate limiting / request-size limits | ⬜ Not built |
| Session analytics / progress reports | ⬜ Not yet built |

> **Known gap:** a post-TKR X-ray shows a prosthesis, not a native knee. The
> classifier is trained on native knees, so grading a replaced joint is
> out-of-distribution — which makes the safe ceiling unreliable for exactly the
> TKR patients the rehab protocols target. The energy screen will flag these
> once weights carry a reference, but the underlying modelling question is open.

---

## License

Not yet decided — add a LICENSE file before making this public if you intend to open-source it.
