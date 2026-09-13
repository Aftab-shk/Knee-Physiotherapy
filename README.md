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
│   ├── db.py                      SQLAlchemy engine, session, ORM base
│   ├── models.py                  Patient, Prescription, PrescriptionAudit, ExerciseSession,
│   │                              ExerciseSet, ShareLink, Clinician, CareLink
│   ├── auth.py                    Argon2id hashing, JWT, request dependencies
│   ├── triage.py                  Red-flag rules — torch-free, database-free, unit tested
│   ├── outcome_measures.py        KOOS-JR items, Rasch lookup, MCID/MDC — torch-free, DB-free
│   ├── summary_pdf.py             The one-page appointment sheet — layout only, no DB, no torch
│   ├── alembic.ini                Migration config — URL comes from db.py, not here
│   ├── migrations/                Schema history; init_db() runs it at startup
│   │
│   ├── model/
│   │   ├── inference.py           EfficientNet (B0–B5) inference, calibration, OOD screen
│   │   ├── image_checks.py        Upload quality checks — torch-free on purpose
│   │   ├── prosthesis.py          Metalwork screen — heuristic, advisory, torch-free
│   │   ├── gradcam.py             Where the model looked — opt-in, never load-bearing
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
│   └── README.md                  Backend-specific docs
│
├── Dockerfile                     one image: API + pages, built from this directory
├── .github/workflows/ci.yml       lint + tests + large-file guard
├── ruff.toml                      lint config
│
└── frontend/
    ├── index.html                 Landing page
    ├── login.html                 Log in / create account / guest access
    ├── upload.html                X-ray upload + results + exercise plan
    ├── progress.html              Range of motion, consistency, per-exercise history
    ├── tracker.html               Webcam safety tracker (MediaPipe pose landmarker)
    ├── share.html                 What a clinician sees through a share link
    ├── manifest.webmanifest       Installable to a phone's home screen
    ├── sw.js                      App-shell cache — never caches an API response
    ├── progress.html              Range of motion, consistency, pain
    ├── assets/
    │   ├── theme.css              Brand tokens, type stacks, reset — shared by every page
    │   ├── progress-view.css      Charts, tiles and tables — shared by progress + share
    │   ├── progress-view.js       Renders a progress payload; used by both pages
    │   ├── config.js              Resolves the API base URL
    │   ├── api.js                 The only place the frontend calls the backend
    │   ├── pose-gate.js           Joint geometry + camera-view validation
    │   ├── voice.js               Spoken cues — what to say, decided apart from saying it
    │   ├── form-check.js          Sagittal-plane form faults; see its header on valgus
    │   └── outcome-form.js        Renders the questionnaire from the server's own definition
    ├── tests/pose-gate.test.mjs   `node --test` — run from frontend/
    ├── package.json               No build step; marks assets/*.js as ES modules
    └── logo.png / logo-dark.png   Branding assets
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

**Configuration.** Everything has a working default, so nothing below is required
for local development:

| Variable | Default | What it does |
|---|---|---|
| `DATABASE_URL` | `sqlite:///backend/physio.db` | Point at Postgres for a real deployment |
| `JWT_SECRET` | *random per process* | **Set this before deploying.** Unset means every restart signs everyone out |
| `JWT_EXPIRY_HOURS` | `336` (14 days) | How long a sign-in lasts |
| `MODEL_PATH` | `model/best_model.pth` | Trained weights |
| `CORS_ORIGINS` | localhost ports | Comma-separated allow-list |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload ceiling, enforced while streaming |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_S` | `20` / `60` | Per-IP sliding window |
| `SUMMARY_PDF_FONT` | *unset — Helvetica* | TTF for the summary PDF. Set it to serve names outside Latin-1 |
| `ENV` | *unset* | Set to `production` and the server refuses to start without `JWT_SECRET` and real weights |
| `FRONTEND_DIR` | `frontend/` beside `backend/` | Pages served by the API itself, same-origin. Unset it to serve them elsewhere |
| `TRUST_PROXY` | *unset* | Take the client IP from `X-Forwarded-For`. Set it only behind a proxy you control |

The database file is created on first start and is git-ignored — it holds real
patient rows and password hashes.

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

`knee_side=both` requires a second file, `image_right` — one film cannot be
graded for two joints. Each knee is read, screened and prescribed for
independently, and `sides` carries both. The flat fields carry **the more
restrictive knee**, so a client that ignores `sides` still gets a safe answer.

Returns KL grade, health score, safe angle ceiling, confidence, rehab phase, a full exercise list with per-exercise angle limits, and a plain-English rationale.

`kl_applicable` is false for a declared total knee replacement: the grade is
still reported, but it sets nothing. `hardware_suspected` warns about a
replacement that was not declared — advisory only, and it never raises a limit.

`grade_probabilities` carries the calibrated probability for every KL grade, and
`within_one_grade` the mass at the chosen grade or either neighbour. The model
was trained with an ordinal loss and scores 70.3% exact against 95.3% within one
grade, so a single confidence figure describes it as far less useful than it is —
and hides the readings where it is genuinely torn.

Send `explain=true` for a Grad-CAM overlay showing which part of the image drove
the grade, returned as a PNG data URI. Off by default: it costs a backward pass,
roughly doubling inference time (measured 750 ms → 1.7 s). It never fails the
analysis — if the overlay cannot be produced, the reading comes back without it.

Authentication is optional. With a bearer token the analysis is saved to the
account and `prescription_id` names the stored record; as a guest it is returned
and forgotten. The X-ray image itself is never stored either way.

`weeks_post_op` may be omitted by a signed-in patient with a surgery date on
file — it is derived from the date. An explicitly supplied value always wins,
since the caller may be correcting the record.

### `GET /exercises`
Query params: `surgery_type` (required), `weeks_post_op` (required unless surgery_type=none), `kl_grade` (optional, default 0). Returns the exercise list without requiring an X-ray upload — useful for browsing protocols or frontend testing.

### `POST /auth/register` · `POST /auth/login`
JSON `{ email, password }` (plus optional `display_name` on register). Both return
`{ access_token, token_type, expires_in, patient }`. Send the token as
`Authorization: Bearer <token>` on later requests.

Passwords are hashed with Argon2id and must be at least 8 characters. A failed
login returns the same 401 whether the address is unknown or the password is
wrong — telling those apart would reveal who has an account.

### `GET /auth/me`
The signed-in patient, including `surgery_date` and the `weeks_post_op` derived
from it. 401 without a valid token.

### `PATCH /me/surgery`
Records the operation being recovered from: `{ surgery_date, surgery_type }`.
Send null for either to clear it. A future date is refused.

Stored once so weeks-post-op stops being retyped on every visit — that number
selects the entire rehab protocol, and a plausible wrong week is
indistinguishable from a right one. The week is derived on every read, never
stored, so it cannot go stale overnight.

### `GET /me/prescriptions`
Past analyses, newest first. `limit` defaults to 20.

### `POST /sessions/sets`
Records one completed set from the webcam tracker: reps, peak flexion, how long
the safe ceiling was exceeded and how often, hold time against target, mean
landmark visibility, and time spent with no usable measurement.

Posted per set rather than once at the end, so a patient who stops after two of
three sets keeps those two. Re-posting the same set is a no-op — the browser
retries on a dropped connection, and a duplicate would inflate an adherence
record. Sets from one sitting are grouped by a browser-generated
`client_session_id`.

Sessions performed with the camera-view gate overridden are flagged
`view_verified: false`. Angles measured square-on to the camera read far too
low, so anything charting range of motion has to be able to leave them out.

### `POST /sessions/report`
Records how a finished session felt: pain afterwards (0–10 NPRS) and perceived
exertion (0–10 Borg CR10). Pain *before* rides along with the set posts, since it
is taken before the session starts.

Merges rather than replaces, so adding an exertion score later cannot wipe a pain
score already recorded. Returns 404 for a sitting with no recorded set — a rating
with no work behind it is not a session, and counting it would inflate adherence.

Every rating is optional. Nothing in the app is blocked without one.

### `GET /sessions`
Recent exercise sessions with their sets, newest first.

### `POST /clinician/register` · `POST /clinician/login` · `GET /clinician/me`
Clinician accounts. A separate table from patients, and a separate role stamped
into the token — a clinician's token is refused on a patient's endpoints and the
reverse. The same email may hold both kinds of account.

### `POST /clinician/invites` · `GET /clinician/invites`
Issues a short code for a patient to redeem. Returned once; only its SHA-256 is
kept. Codes are drawn from an alphabet with no O/0, I/1 or U, so they survive
being read aloud.

### `POST /me/clinicians/redeem`
The patient enters the code. **This is the consent** — the direction is
deliberate, so nobody can attach themselves to a medical record without the
patient acting. Case, spaces and dashes are normalised away. A wrong, expired or
already-used code all return the same 404.

### `GET /clinician/patients` · `GET /clinician/patients/{id}/progress`
The caseload, and one patient in full. Sessions for the whole list are fetched in
one query. Flexion comes only from verified camera views — an unverified one
reads far too low, and a clinician scanning that column would see a collapse in
range of motion that never happened.

### `DELETE /me/clinicians/{id}` · `DELETE /clinician/patients/{id}`
Either side can end the link, and `revoked_by` records which. "My physiotherapist
discharged me" and "I withdrew access" are different events.

### `GET /outcome-measures`
The KOOS-JR questionnaire as it should be asked: the seven items, their wording,
and the response options. The frontend renders from this rather than hardcoding
the items, so the wording has one source.

### `POST /me/outcome-scores` · `GET /me/outcome-scores`
Records a completed questionnaire and returns the derived score, or lists the
history with whether another is due.

All seven answers are required — **a partly completed form is refused, not
imputed.** KOOS-JR has no published rule for a missing item, and averaging the
six that were answered produces something indistinguishable from a real score
that is not one.

Re-answering within **14 days** returns 409: real week-to-week movement is
smaller than the instrument can detect, so a score that soon would be noise
presented as progress. Two knees are two questionnaires — KOOS-JR asks about
"your knee", singular.

A score is **never a gate and never moves a movement ceiling.** That stays with
the radiograph and the surgical protocol; there is a test asserting it.

### `GET /clinician/patients/{id}/outcome-scores`
The same history for a patient on the clinician's caseload. The trend also rides
on `/me/progress` and the caseload row, and flows through to a share link.

### `GET /me/summary.pdf` · `GET /share/{token}/summary.pdf` · `GET /clinician/patients/{id}/summary.pdf`
The offline counterpart to the share link: **one page**, to print or to show on a
phone with no signal. A link works when the physiotherapist has a screen and a
spare hand, and a good many appointments have neither.

Same sheet, three readers — and what differs is only what each may put at the
top of it. The patient's own copy falls back to their email address when no
display name is set; the shared and clinician copies never do. A share link is a
bearer token, so an account identifier printed on it would turn a read-only link
into the first half of an attack on the account.

**One page is enforced, not hoped for.** Vertical space is handed out by a cursor
that can refuse, and anything dropped for want of room is *named in the notes* —
a reader cannot otherwise tell an exercise the patient never did from one that
fell off the bottom.

**Every caveat is printed**, because paper has no tooltips: the count of sessions
excluded from the angles and why, whether a clinician has actually approved the
limits, whether the grade came from demonstration mode, and that a webcam is not
a goniometer. `?days=` and `?tz_offset_minutes=` match `/me/progress`.

### `GET /me/flags` · `GET /clinician/patients/{id}/flags`
Why someone might need looking at: severe or rising pain, range of motion going
backwards, repeated ceiling breaches, a patient who was going and stopped, and —
for the clinician only — prescriptions nobody has reviewed.

**Triage signals, not findings.** They move a patient up a list and decide
nothing. Each carries the numbers it was raised on so a clinician can disagree
with it at a glance, and the rules stay deliberately quiet: nothing fires from a
single reading, a patient who never started is not deteriorating, and angles from
an unverified camera view raise nothing at all. Flags also ride on the caseload,
so it can be sorted by them.

The patient sees the same assessment worded for them. Findings about the
clinician's own workflow are left out — telling someone their physiotherapist has
not read their notes is alarming and not theirs to act on.

### `GET /exercises/catalogue`
Every exercise a clinician may add. Additions come from here so the tracker still
knows how to count them and which joint to watch — a free-text exercise would be
untrackable, which is worse than not offering it.

### `GET /clinician/prescriptions/{id}` · `POST /clinician/prescriptions/{id}/review`
The model drafts; a clinician approves. The review keeps, removes or adjusts each
exercise, adds from the catalogue, and can raise or lower the safe ceiling.

**Loosening any restriction requires a reason; tightening does not.** The
clinician's judgement wins — they operated on the knee, the radiograph did not —
but it is never silent. Every ceiling decision, the model's included, is recorded
with what it replaced and why.

Exercises left out of the request are kept as drafted. A clinician editing one of
nine has approved the other eight, not deleted them.

### `GET /me/prescriptions/{id}`
What the patient should actually follow: the clinician's version if there is one,
the model's otherwise, with `status` saying which. Overridden exercises carry who
changed them and what the limit was before, so the patient's own screen can say
so — and the tracker enforces the approved number.

### `POST /me/share-links` · `GET /me/share-links` · `DELETE /me/share-links/{id}`
Creates, lists and withdraws read-only links to your progress. The token is
returned **once** — only its SHA-256 is stored — and a link carries its own
access count so you can see whether it was ever opened.

### `GET /share/{token}`
The clinician's view. No account, no password: the link is the credential,
because a clinician will click a link and will not register for an app their
patient uses.

Shows progress only — no email, no X-ray, no stored prescriptions. Every failure
is the same 404: telling an expired link apart from a withdrawn one would confirm
to whoever holds it that it was real, and whose it was.

### `GET /me/progress`
Range of motion per exercise, adherence by day, pain before and after, and a
per-exercise breakdown. Takes `days` (default 90) and `tz_offset_minutes`, so an
evening session lands on the evening it happened rather than the next UTC day.

Sessions performed with the camera gate overridden are counted for adherence and
excluded from every angle — but their pain scores are kept, since the camera
angle says nothing about what the patient felt.

### `GET /me/progress`
Query params: `days` (1–365, default 90), `tz_offset_minutes` (minutes east of UTC).

Returns headline figures, a range-of-motion series **per exercise**, day-by-day
adherence, and a per-exercise breakdown.

Two rules shape the numbers:

- **Range of motion is kept per exercise.** Peak flexion only means something
  against what the exercise asked for — Quad Sets are held with the knee locked
  straight at about 4°, Mini Squats bend past 50°. Pooled into one line, a day of
  quad sets plots as a collapse in range of motion and the patient reads their own
  chart as a relapse.
- **Unverified sessions count for adherence and for nothing else.** A session
  recorded with the camera-view gate overridden is real work, but a knee filmed
  square-on reads far straighter than it is, so its angles are excluded from every
  flexion figure.

Days are bucketed in the caller's own UTC offset, so an evening session lands on
the evening it happened rather than the next UTC day — which would split a streak
the patient never broke.

### `GET /health`
Returns `{ status, model_loaded, model_version, demo_mode, calibrated, ood_screening }`.

`calibrated` and `ood_screening` report whether the loaded checkpoint carries a
fitted temperature and an OOD energy reference. Both false means the instance is
serving raw softmax confidence and cannot screen non-radiograph uploads.

Full schema and interactive testing at `/docs` once the server is running.

---

## Installing it on a phone

Serve the frontend over **https** (a service worker will not register otherwise)
and open it in a mobile browser — the install prompt appears from the browser
menu. The app shell is cached so it opens without a connection; **no API response
is ever cached**, because a stale prescription is a stale movement ceiling.

The tracker offers a camera switcher when the device has more than one, keeps the
screen awake during a session, and re-fits the pose overlay when the phone is
rotated.

> The launcher icons are upscaled from a 68×68 logo, so they are soft at 512px.
> A vector or high-resolution source would fix that; nothing else is blocked on it.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn, Pydantic v2 |
| Model | EfficientNet (torchvision), B4 by default, fine-tuned on KL grade data |
| Preprocessing | Optional OpenCV CLAHE, ImageNet normalisation, native per-architecture resolution |
| Frontend | Plain HTML/CSS/JS (no framework); MediaPipe Tasks Vision for pose tracking |
| Training | PyTorch, Kaggle (T4/P100 GPU) — MixUp/CutMix, layer-wise LR decay, EMA, ordinal loss, post-hoc class-prior correction |
| Serving | Flip-TTA, temperature-scaled confidence, energy-based OOD screening |
| Accounts | Argon2id password hashing, stateless JWT bearer tokens |
| Persistence | SQLAlchemy 2.0 — SQLite by default, Postgres via `DATABASE_URL` |
| Tests | pytest — clinical logic, tracker contract, calibration, API, accounts; `node --test` for the pose gate |
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
| Test suite | ✅ 656 pytest + 117 node — clinical logic, tracker, calibration, API security, accounts, sessions, progress, pain, surgery dates, sharing, clinicians, review, triage, outcome measures, appointment summary, prosthesis gate, Grad-CAM, grade distribution, bilateral, migrations, pose gate, voice cues, form checks, service-worker routing |
| CI | ✅ GitHub Actions — ruff + pytest, and a guard against large tracked files |
| Trained model weights | ✅ Calibrated — 70.3% test accuracy, 95.3% within-one-grade, ECE 0.041 |
| Replaced-joint gate | ✅ A declared TKR is never graded; undeclared metalwork warns but never loosens |
| Explainability | ✅ Grad-CAM overlay on the uploaded X-ray, with a fade control |
| Grade distribution | ✅ All five grades with calibrated probability, plus within-one-grade |
| Bilateral assessment | ✅ Two X-rays, two grades, two ceilings; the summary shows the stricter knee |
| Voice cues + rest timers | ✅ Spoken counts, countdowns and safety warnings; a real 45s rest between sets |
| Form coaching | ✅ Trunk lean, hip lift, unlocked knee, sway — sagittal only (valgus needs a front-on camera) |
| Installable / phone | ✅ Manifest, offline app shell, camera switcher, screen wake lock, rotation handling |
| Outcome measures | ✅ KOOS-JR — Rasch-scored 0–100, MCID/MDC-aware, charted and shared; never a gate |
| Appointment summary | ✅ One-page PDF for patient, share link and clinician — one page enforced, omissions named |
| Real authentication | ✅ Argon2id + JWT — register, log in, guest access still supported |
| Persistence | ✅ SQLAlchemy + SQLite (Postgres via `DATABASE_URL`); analyses saved per account |
| Rate limiting / request-size limits | ✅ Sliding-window limiter + streaming upload cap |
| Camera-view validation in the tracker | ✅ Session will not start until the camera sees the leg side-on |
| Exercise session logging | ✅ Every set stored — peak flexion, breaches, hold time, tracking quality |
| Progress view | ✅ ROM trend per exercise, consistency heatmap, streaks, per-exercise breakdown |
| Pain + exertion capture | ✅ NPRS before/after, Borg CR10 — optional throughout |
| Surgery date on file | ✅ Weeks post-op derived, not retyped; floored so a phase never starts early |
| Share with a clinician | ✅ Signed-out, expiring, revocable links; tokens stored hashed |
| Clinician accounts + caseload | ✅ Separate role, invite codes redeemed by the patient, either side can end it |
| Schema migrations | ✅ Alembic; `init_db()` upgrades at startup and adopts pre-migration databases |
| Clinical review / prescription override | ✅ AI drafts, clinician approves; loosening needs a stated reason |
| Audit trail | ✅ Every ceiling decision recorded — the model's included — with what it replaced and why |
| Red-flag triage | ✅ Pain, ROM regression, breaches and lapses — on the caseload and on the patient's own page |

> **Replaced joints don't get graded.** A post-TKR X-ray shows a prosthesis, not
> a native knee, so a Kellgren–Lawrence grade — a measure of wear on a natural
> joint surface — describes something that is no longer there. For a declared
> TKR the grade is reported but sets nothing: the ceiling comes from the surgical
> protocol and the week instead. An undeclared replacement is caught by a
> brightness heuristic in `model/prosthesis.py`, which warns and is never allowed
> to loosen a restriction on its own.
>
> **The summary PDF cannot print a name outside Latin-1 without a font.** The
> base-14 PDF fonts stop at Latin-1, and the faces bundled with reportlab cover
> no more — no Devanagari, no CJK, no Arabic. So a patient named in one of those
> scripts gets "?" where the characters should be, *with a line on the sheet
> saying so and naming the fix*, rather than a silently mangled name on their own
> medical summary. The fix is one variable:
>
> ```
> SUMMARY_PDF_FONT=/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf
> ```
>
> Point it at a font covering the scripts you serve and names render properly.
> An unreadable path falls back to Helvetica and logs a warning rather than
> failing the download. The RFC 5987 `filename*` is unaffected either way, so the
> saved file keeps the real name even when the page cannot draw it.
>
> **The Oxford Knee Score is not included, for licensing reasons.** The roadmap
> item said "KOOS-JR or Oxford Knee Score". The OKS is © Oxford University
> Innovation and needs a licence beyond non-commercial research, so its twelve
> items cannot ship in this repository. KOOS-JR is free to use, is seven items
> rather than twelve for something asked repeatedly over a year, is
> Rasch-calibrated onto an interval 0–100 scale (so a ten-point change means the
> same thing at either end, which an ordinal 0–48 sum does not), and is what the
> American Joint Replacement Registry and CMS collect — so a score here lines up
> with the national comparison. Everything downstream of `score()` is
> instrument-agnostic, so a licensed deployment can add the OKS as a second
> instrument without touching the storage, the charts or the triage rule.
>
> **Knee valgus is not detected, and cannot be from this camera position.**
> Valgus and hip hiking are frontal-plane movements — seeing them needs a camera
> in front of the patient. The tracker requires the opposite: a knee angle from
> 2D landmarks is only valid side-on, so the session will not start until the
> camera is in the one position from which valgus is invisible. Detecting it
> properly means a second camera position and a separate assessment mode.
> `assets/form-check.js` covers the sagittal-plane faults instead — trunk
> compensation, hips lifting, a "straight" hold performed bent, and sway during
> a balance exercise.
>
> **Still open:** a learned native-versus-replaced classifier. That needs a
> labelled set of post-arthroplasty radiographs, which this project does not
> have — the heuristic is a backstop, not a substitute.

---

## License

Not yet decided — add a LICENSE file before making this public if you intend to open-source it.
