# AI Knee Physiotherapy — Comprehensive System Architecture & Technical Notes

This document provides a detailed breakdown of how the entire **AI Knee Physiotherapy** project works, covering the backend, machine learning model, frontend interface, and real-time webcam exercise safety tracking system.

---

## 💡 1. Overview: What is AI Knee Physiotherapy?

**AI Knee Physiotherapy** is an AI-assisted knee rehabilitation application designed to generate safe, personalized exercise programs for knee surgery and osteoarthritis patients.

### The Problem It Solves
Two patients may have had the exact same surgery (e.g., knee replacement 4 weeks ago), but one patient's joint has severe wear-and-tear (osteoarthritis) while the other's joint is relatively healthy. 

AI Knee Physiotherapy combines two independent streams of input:
1. **Knee X-ray Image:** Analyzed by an AI computer vision model to determine joint damage severity using the **Kellgren-Lawrence (KL) Grade (0 to 4)**. This establishes a **Safe Angle Ceiling** (the maximum safe flexion angle).
2. **Surgery Type & Recovery Duration:** Determines **which exercises** are suitable for the patient's current stage of recovery.

During exercise sessions, AI Knee Physiotherapy tracks leg joint angles in real time via the user's webcam and **triggers visual and audible alerts if the joint angle exceeds the safe limit**.

---

## 🧠 2. Machine Learning Model (How & Why)

### Why Train a Model?
Medical professionals evaluate knee osteoarthritis using the **Kellgren-Lawrence (KL) scale**:
* **Grade 0:** Normal, healthy knee ($120^\circ$ safe flexion ceiling).
* **Grade 1:** Doubtful joint changes ($120^\circ$ safe flexion ceiling).
* **Grade 2:** Minimal arthritis ($90^\circ$ safe flexion ceiling).
* **Grade 3:** Moderate arthritis ($60^\circ$ safe flexion ceiling).
* **Grade 4:** Severe arthritis / bone-on-bone ($45^\circ$ safe flexion ceiling).

Training a deep learning classification model automates KL grading directly from X-ray uploads.

### Model Architecture & Training Pipeline
* **Dataset Preprocessing ([prepare_dataset.py](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/model/prepare_dataset.py)):**
  * X-ray images categorized into grades `0` through `4`.
  * Images are resized to $224 \times 224$, normalized, and enhanced using OpenCV CLAHE (Contrast Limited Adaptive Histogram Equalization).
* **Model Architecture ([train.py](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/model/train.py#L72)):**
  * Uses **EfficientNet** (B0 / B3) pre-trained on ImageNet.
  * Replaces the 1000-class output head with a 5-class linear classifier with dropout ($p=0.4$).
* **Two-Phase Training Strategy:**
  1. *Phase 1 (Warm-up):* Pre-trained backbone is frozen; only the new classifier head is trained for 5 epochs.
  2. *Phase 2 (Fine-tuning):* All layers are unfrozen and trained end-to-end using AdamW optimizer with cosine learning rate annealing and class-imbalance weighting.
* **Inference & Demo Mode ([inference.py](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/model/inference.py)):**
  * Saves weights to `best_model.pth`.
  * Performs Test-Time Augmentation (TTA) combining regular and horizontally flipped inputs.
  * If weights are missing, the system gracefully falls back to **Demo Mode** with deterministic mock predictions so full system testing can occur without trained weights.

---

## ⚙️ 3. Backend Architecture (`FastAPI`)

The backend is built in Python using **FastAPI** and comprises three primary files:

### 📄 [main.py](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/main.py) — API Server & Routes
* **`POST /analyse-xray`** ([L195](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/main.py#L195)): Accepts multipart form upload (`image`, `knee_side`, `surgery_type`, `weeks_post_op`). Validates image format/contrast, executes AI classification, generates prescription details, and returns JSON output.
* **`GET /exercises`** ([L147](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/main.py#L147)): Returns protocol exercise lists without requiring an X-ray upload (useful for testing and physiotherapy browsing).
* **`GET /health`** ([L132](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/main.py#L132)): Provides service liveness status, model load state, and demo mode flags.

### 📄 [clinical_logic.py](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/clinical_logic.py) — Clinical Rules Engine
* Receives raw model prediction and patient inputs.
* Selects appropriate rehab protocol from `exercise_protocols.py`.
* Enforces angle capping logic:
  $$\text{effective\_angle\_limit} = \min(\text{protocol\_angle\_limit}, \text{xray\_max\_angle})$$
* Flags `angle_capped = True` and appends explanatory clinical warnings if an exercise angle is reduced due to joint severity.

### 📄 [exercise_protocols.py](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/exercise_protocols.py) — Exercise Database
* Contains 16 protocol sets and 58 distinct exercises categorized across 5 surgery types (`acl`, `tkr`, `meniscus`, `arthroscopy`, `none`) and multi-stage recovery timeframes (Weeks 0–2, 2–6, 6–12, 12+).

---

## 🖥️ 4. Frontend Application

The frontend is implemented using vanilla HTML5, modern CSS design tokens, and JavaScript:

* **[index.html](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/frontend/index.html):** Public landing page detailing system capabilities and medical rationale.
* **[login.html](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/frontend/login.html):** User access and authentication portal.
* **[upload.html](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/frontend/upload.html):**
  * Interactive drag-and-drop X-ray upload interface with real-time validation.
  * Communicates with backend endpoint `POST /analyse-xray`.
  * Dynamically populates diagnostic summaries (KL Grade, Health Score, Safe Ceiling, Phase Goals, Exercise Cards).
  * Launches exercise tracking sessions by storing current exercise data in browser `sessionStorage` and redirecting to `tracker.html`.

---

## 📹 5. Real-Time Webcam Exercise Safety Tracker

The tracking module ([tracker.html](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/frontend/tracker.html)) runs entirely inside the client browser without sending video data to external servers.

```
 Webcam Feed  ──►  Google MediaPipe Pose  ──►  Knee Angle Calculation  ──►  Safety & Rep Check
```

1. **Pose Landmarker:** Loads Google **MediaPipe Tasks Vision** via WebAssembly (WASM), detecting 33 3D body pose landmarks at 30+ FPS.
2. **Knee Angle Calculation ([calcAngle](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/frontend/tracker.html#L962)):**
   * Maps 3 key leg coordinates: **Hip** (23/24), **Knee** (25/26), and **Ankle** (27/28).
   * Computes vector angle:
     $$\text{Flexion Angle} = 180^\circ - \arccos\left(\frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|}\right)$$
   * Applies rolling 5-frame moving average smoothing to eliminate landmark jitter.
3. **Repetition & Hold Logic:**
   * Dynamic exercises: State machine (`EXTENDED` $\rightarrow$ `FLEXING` $\rightarrow$ `FLEXED` $\rightarrow$ `EXTENDING`) tracks completed repetitions.
   * Isometric exercises: Ring timer monitors target position hold duration.
4. **Real-Time Safety System ([handleSafetyCheck](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/frontend/tracker.html#L980)):**
   * Continually compares real-time knee flexion angle against `exercise.angle_limit`.
   * If limit is breached:
     * Triggers a full-screen red warning overlay.
     * Plays an audible alert using the browser's native **Web Audio API** oscillator.
     * Auto-pauses exercise session if breach persists beyond $500\text{ ms}$.

---

## 📊 6. System Execution Summary

| Pipeline Step | Module Path | Technology | Function |
|---|---|---|---|
| 1. Input Collection | [upload.html](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/frontend/upload.html) | HTML5 / JavaScript | Collects X-ray image file and surgical timeline |
| 2. Image Classification | [inference.py](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/model/inference.py) | PyTorch / EfficientNet | Predicts KL Grade (0–4) from X-ray |
| 3. Clinical Rules | [clinical_logic.py](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/backend/clinical_logic.py) | Python | Computes safe angle ceiling & caps exercise limits |
| 4. Routine Presentation | [upload.html](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/frontend/upload.html) | CSS / JavaScript | Displays personalized rehab plan and exercise instructions |
| 5. Live Pose Safety | [tracker.html](file:///c:/Users/aftab/OneDrive/Desktop/knee-physiotherapy/frontend/tracker.html) | MediaPipe / Web Audio | Tracks webcam pose, counts reps, alerts on angle breach |