# Physiotherapy AI — Backend

FastAPI backend for knee X-ray KL-grade classification and personalised rehab exercise prescription.

---

## Architecture

```
POST /analyse-xray
  │
  ├── validate_image()          image quality gate (contrast, exposure)
  ├── KneeClassifier.predict()  ResNet50 → KL Grade 0–4 + confidence
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
> `model/resnet50_kl_v1.pt`. All clinical logic (exercise selection, angle capping,
> rationale) runs normally in demo mode.

---

## Docker

```bash
# Build
docker build -t physio-backend .

# Run (demo mode — no weights)
docker run -p 8000:8000 physio-backend

# Run with trained weights mounted
docker run -p 8000:8000 \
  -v /path/to/resnet50_kl_v1.pt:/app/model/resnet50_kl_v1.pt \
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
  "model_version": "resnet50_kl_v1",
  "demo_mode": true
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "model_version": "resnet50_kl_v1", "demo_mode": true }
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

1. Download the **Knee Osteoarthritis Dataset with Severity Grading** from Kaggle.
2. Train a ResNet50 (`torchvision.models.resnet50`) with its final FC replaced by
   `nn.Linear(2048, 5)` for 5-class output (KL 0–4).
3. Preprocessing during training must match inference:
   - Convert to grayscale → CLAHE (clipLimit=2.0, tileGridSize=8×8) → RGB
   - Resize to 224×224
   - Normalize with ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
4. Save weights: `torch.save(model.state_dict(), "model/resnet50_kl_v1.pt")`
5. Place the file at `model/resnet50_kl_v1.pt` (or set `MODEL_PATH` env var).
6. Restart the server — it will detect and load the weights automatically.

---

## Environment Variables

| Variable       | Default | Description |
|----------------|---------|-------------|
| `MODEL_PATH`   | `model/resnet50_kl_v1.pt` | Path to trained weights |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allowed origins |
