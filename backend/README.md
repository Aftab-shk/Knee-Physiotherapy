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
docker build -t physio-backend ..   # context is the repo root: the image serves frontend/ too

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
      "tracked_joint": "knee",
      "hold_target": null,
      "start_angle": null,
      "hold_angle": null,
      "instructions": ["..."],
      "cautions": null,
      "angle_capped": false
    }
  ],
  "rationale": "Your X-ray shows KL Grade 2 ...",
  "disclaimer": "For informational purposes only...",
  "model_version": "efficientnet_b4_20260815_cal",
  "demo_mode": false
}
```

#### Tracker fields

`tracked_joint` and `hold_target` tell `tracker.html` how to score an exercise.

| Field | Values | Meaning |
|---|---|---|
| `tracked_joint` | `knee` \| `ankle` | Which joint drives rep/hold counting. Only Ankle Pumps uses `ankle`. The **safety check always watches the knee** against `angle_limit` — the ceiling is a knee flexion ceiling regardless of this field. |
| `hold_target` | `straight` \| `flexed` \| `null` | Where the hold sits relative to `angle_limit`. `straight` = held at or below it (knee locked out: quad sets, SLR, patellar mobilisation, single-leg balance, seated extension). `flexed` = held near it (flexion end-range: seated/standing knee bend). `null` for rep-counted exercises. |
| `start_angle` | int \| `null` | Unavoidable passive knee flexion the exercise begins from — sitting in a chair puts the knee at ~90° before the patient moves. `null` when it starts from extension. Used for screening, see below. |
| `hold_angle` | int \| `null` | Angle to reach and hold, when it differs from `angle_limit`. `null` → derive from `angle_limit`. |

`angle_limit` is a **safety ceiling** (never exceed); `hold_angle` is a
**performance target** (try to reach). They coincide for most exercises. Seated
Knee Extension is the exception: performed seated the knee sits at 90° (its real
ceiling) while the goal is to extend to 45°.

All four are required for the tracker to behave correctly. A timed hold with no
`hold_target` falls back to treating "in position" as *bent past 15°*, which is
unreachable for any exercise whose limit is below 15° — the patient would trip
the safety alarm while performing it correctly. `backend/tests/test_tracker_contract.py`
pins this invariant.

#### Start-position screening

An exercise whose `start_angle` exceeds the patient's ceiling is **excluded from
the prescription** and reported in `excluded_exercises`:

```json
"excluded_exercises": [
  {
    "name": "Seated Knee Extension (Gravity Only)",
    "start_angle": 90,
    "max_angle": 45,
    "reason": "This exercise is performed with the knee already at about 90°, which is beyond your 45° safe ceiling..."
  }
]
```

Capping the target angle cannot rescue these — the *starting* position is the
problem, so a KL4 patient with a 45° ceiling would breach it simply by sitting
down. They are reported rather than silently dropped so a physiotherapist can see
what was withheld and prescribe a lying or standing alternative. `exercise_list`
is never left empty by screening.

### `GET /exercises`

Query params: `surgery_type` (required), `weeks_post_op` (required unless `surgery_type=none`), `kl_grade` (optional, 0–4, default 0). Returns the same `exercise_list` shape without requiring an image upload.

### Confidence and OOD screening

`confidence` is only a probability of being right when `calibrated` is `true`.
Until then the API reports a band and the UI leads with it.

| Field | Meaning |
|---|---|
| `confidence_band` | `low` (<0.50) · `moderate` (0.50–0.75) · `high` (≥0.75). **Display this, not the raw number.** |
| `calibrated` | `true` when the checkpoint carries a temperature fitted on val. `false` → raw softmax, which overstates certainty. |
| `ood_suspected` | Image sits in the tail of the in-distribution energy range — grade with caution. |

Two post-hoc corrections are fitted on the validation split at the end of
training and stored in the checkpoint:

- **Temperature scaling** (Guo et al., 2017) — one scalar dividing the logits.
  Monotonic per forward pass, so on a single pass it cannot change `argmax`.
  With flip-TTA the branches are averaged *after* softmax, which is not
  monotonic in T, and a borderline case can flip — measured here at 1 image in
  1656 (0.06%). `train.py` prints ECE before/after and warns above 0.5% drift.
- **Energy OOD reference** (Liu et al., 2020) — `E(x) = -logsumexp(logits)`,
  low for inputs the model recognises. Validation percentiles become the
  screening thresholds: above p95 warns (`ood_suspected`), beyond p99 plus one
  inter-percentile spread rejects with a 422.

Both degrade safely. A checkpoint without them serves normally, reports
`calibrated: false` / `ood_screening: false`, logs a warning at startup, and the
UI says the certainty is uncalibrated rather than implying otherwise.

Measured on the held-out test split (1656 images), fitted on val only:

| pipeline | acc | macro-F1 | within-1 | ECE |
|---|---|---|---|---|
| raw softmax | 0.6987 | 0.6839 | 0.9517 | 0.0810 |
| + prior correction (τ=0.05) | 0.7035 | 0.6910 | 0.9529 | 0.0804 |
| **+ temperature (T=0.734)** | 0.7029 | 0.6902 | 0.9529 | **0.0407** |

The model was **under**-confident (0.636 mean confidence against 0.699
accuracy), so T came out below 1 and calibration sharpened it. Calibration error
halves. Temperature moved 1 prediction in 1656 — dividing logits is monotonic
per-branch, but flip-TTA averages after softmax, which is not.

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "efficientnet_b4_20260818_cal",
  "demo_mode": false,
  "calibrated": true,
  "ood_screening": true
}
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

## Tests

```bash
pip install pytest
python -m pytest backend/tests -q
```

`tests/test_tracker_contract.py` pins the contract between the protocol database
and the webcam tracker. None of it needs torch or a trained checkpoint — the
clinical logic is pure functions over dicts.

The load-bearing assertion is `test_hold_position_never_breaches_the_safety_limit`:
for every timed hold, the position the tracker asks the patient to reach must sit
at or below that exercise's own angle limit. Three exercises used to violate it.

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
| `MODEL_PATH` | `best_model.pth` | Checkpoint to load. Relative paths resolve against `backend/model/`; absolute paths are used as-is, which is how the container mounts weights. |
| `WEIGHTS_URL` | — | Overrides the download url in `model/weights.json` for `fetch_weights.py`. |

## Model weights

Not in git. They are ~70 MB, change every training run, and previously bloated
this repo's history to 163 MB before being removed.

```bash
python model/fetch_weights.py          # download + verify sha256
python model/fetch_weights.py --force  # re-download
```

Without a checkpoint the server starts in **demo mode**: deterministic mock KL
grades so the frontend and clinical logic can be developed without one.
`/health` reports `demo_mode`, `calibrated` and `ood_screening` so you can tell
exactly what a running instance is doing.

## Development

```bash
pip install ruff pytest
ruff check backend/          # lint  (config: ruff.toml at the repo root)
python -m pytest backend/tests -q
```

Both run in CI on every push and pull request, alongside a guard that fails the
build if any tracked file exceeds 5 MB — the check that would have caught the
70 MB checkpoint this repo carried for four commits.
