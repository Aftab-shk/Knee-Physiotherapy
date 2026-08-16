# Physiotherapy AI — Backend

FastAPI backend for knee X-ray KL-grade classification and personalised rehab exercise prescription.

---

## Architecture

```
POST /analyse-xray
  │
  ├── validate_image()          image quality gate (contrast, exposure)
  ├── KneeClassifier.predict()  EfficientNet (B0–B5, B4 by default) → KL Grade 0–4 + confidence
  └── build_prescription()
        ├── get_phase()         surgery_type + weeks_post_op → rehab phase
        ├── _cap_exercises()    protocol angle limits capped by X-ray max_angle
        └── _build_rationale()  plain-English explanation
```

---

## Quick Start (local dev)

```bash
# 1. Create virtualenv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install CPU torch (avoids downloading the CUDA build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Run (demo mode — no weights needed)
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

> **Demo mode** is active by default until you place trained weights at
> `model/best_model.pth`. All clinical logic (exercise selection, angle capping,
> rationale) runs normally in demo mode — only the KL grade becomes a
> deterministic mock value derived from the image hash.

---

## Docker

```bash
# Build
docker build -t physio-backend .

# Run (demo mode — no weights)
docker run -p 8000:8000 physio-backend

# Run with trained weights mounted
docker run -p 8000:8000 \
  -v /path/to/best_model.pth:/app/model/best_model.pth \
  physio-backend
```

---

## API

### `POST /analyse-xray`

Multipart form data:

| Field          | Type     | Required | Description |
|----------------|----------|----------|-------------|
| `image`        | file     | ✅        | JPEG or PNG, ≤ 10 MB |
| `knee_side`    | string   | ✅        | `left` · `right` · `both` |
| `surgery_type` | string   | ✅        | `acl` · `tkr` · `meniscus` · `arthroscopy` · `none` |
| `weeks_post_op`| integer  | ✅ (surgical) | Weeks since surgery. Omit for `surgery_type=none` |

**Example with curl:**
```bash
curl -X POST http://localhost:8000/analyse-xray \
  -F "image=@knee.jpg" \
  -F "knee_side=left" \
  -F "surgery_type=acl" \
  -F "weeks_post_op=4"
```

**Response (trimmed):**
```json
{
  "kl_grade": 2,
  "health_score": 60,
  "max_angle": 90,
  "confidence": 0.812,
  "knee_side": "left",
  "surgery_type": "acl",
  "weeks_post_op": 4,
  "rehab_phase": "phase_2",
  "rehab_phase_label": "Phase II — Early Strengthening (Weeks 3–6)",
  "rehab_phase_goal": "Restore 0–90° ROM, begin weight-bearing strengthening.",
  "exercise_list": [
    {
      "name": "Mini Squat",
      "description": "Partial squat to 45–60° for safe quad loading.",
      "target_reps": 10,
      "target_sets": 3,
      "angle_limit": 60,
      "hold_seconds": null,
      "instructions": ["..."],
      "cautions": null,
      "angle_capped": false
    }
  ],
  "rationale": "Your X-ray shows KL Grade 2 ...",
  "disclaimer": "For informational purposes only...",
  "model_version": "efficientnet_b4_v2",
  "demo_mode": false
}
```

### `GET /exercises`

Query params: `surgery_type` (required), `weeks_post_op` (required unless `surgery_type=none`), `kl_grade` (optional, 0–4, default 0). Returns the same `exercise_list` shape without requiring an image upload.

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "model_version": "efficientnet_b4_v2", "demo_mode": false }
```

---

## Clinical Logic Summary

| KL Grade | Health Score | Safe Flexion Ceiling |
|----------|-------------|---------------------|
| 0        | 95/100      | 120°                |
| 1        | 80/100      | 120°                |
| 2        | 60/100      | 90°                 |
| 3        | 35/100      | 60°                 |
| 4        | 15/100      | 45°                 |

The X-ray-derived ceiling **caps** individual exercise angle limits but does not
select exercises. Surgery type + weeks post-op selects the exercise protocol phase.
If a patient's ceiling (e.g. 60°) is below the normal protocol limit for an
exercise (e.g. 90°), the exercise is included with its limit reduced and a caution
note added. This mirrors how a clinical physiotherapist would adapt a standard
protocol to a more severe presentation.

---

## Training the Model

The training recipe (EfficientNet-B4 at its native 380px resolution, ordinal-aware
loss, MixUp/CutMix, layer-wise LR decay, EMA weight averaging, and a post-hoc
class-prior correction) lives entirely in `model/train.py` and is documented in
**[model/TRAINING.md](model/TRAINING.md)**, including the exact Kaggle command,
how to resume a run that stopped early, and how to read the confusion matrix /
per-class report to decide what to tune next.

Short version:
```bash
python model/train.py --data_dir "<path-to-kaggle-dataset>" --arch b4 --num_workers 4 --tta
```
Then copy the result into place:
```bash
cp outputs/best_model.pth backend/model/best_model.pth
```
Restart the server — it detects the file and switches out of demo mode automatically.
The checkpoint embeds its own architecture, input resolution, CLAHE flag, and
class-prior correction, so nothing else needs to be kept in sync manually.

---

## Environment Variables

| Variable       | Default | Description |
|----------------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080,http://localhost:5500,http://127.0.0.1:5500,null` | Comma-separated allowed origins |

> `MODEL_PATH` appears in `main.py`'s docstring and startup log message, but
> nothing currently reads it to choose where weights are loaded from —
> `model/inference.py` always loads `backend/model/best_model.pth`. Setting
> the env var has no effect until that's wired up; place weights at that
> fixed path.
