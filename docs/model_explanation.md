# AI Knee Physiotherapy — ML Model Folder Explanation

Complete plain-English explanation of every file inside `backend/model/`, what algorithm is used, why it's used, and how the pieces fit together.

> **⚠️ Partially out of date.** The training recipe was upgraded to EfficientNet-B4
> with a new default configuration. The sections below still describe the older
> B0/B3 setup (224px input, 5 warm-up + 25 fine-tune epochs, inverse-frequency
> class weights, `--arch b0` default). The *concepts* — transfer learning, the
> two-phase warm-up/fine-tune structure, KL grading, the output files — are all
> still accurate; only the specific defaults and flags in the tables have changed.
> See **[`backend/model/TRAINING.md`](backend/model/TRAINING.md)** for the current
> command, defaults, and the reasoning behind each change.

---

## 📁 Files in `backend/model/`

| File | Purpose in One Line |
|---|---|
| `prepare_dataset.py` | Finds the X-ray dataset, organises it into train/val/test splits, and creates image loaders |
| `train.py` | Trains the AI model (EfficientNet) to classify knee X-rays into KL Grades 0–4 |
| `inference.py` | Uses the trained model to predict the KL Grade of a new X-ray image |
| `best_model.pth` | The saved brain of the trained model (binary weights file, ~16 MB) |

---

## 🧠 What Algorithm Is Used and Why?

### Algorithm: **EfficientNet (B0 or B3) with Transfer Learning**

| Question | Answer |
|---|---|
| What is EfficientNet? | A family of image classification neural networks designed by Google that are both accurate and efficient (fast, small) |
| Why EfficientNet? | It gives very high accuracy on image tasks while using fewer parameters than older networks like ResNet or VGG — perfect for medical images where datasets are small |
| What is Transfer Learning? | Instead of training from scratch (which needs millions of images), we start with a model that has already learned to recognise shapes, edges, and textures from 1.2 million everyday photos (ImageNet). We then teach it the *specific* skill of reading knee X-rays |
| What are B0 and B3? | Different sizes of EfficientNet. B0 is smaller and faster; B3 is bigger and slightly more accurate. The code supports both — you choose via a command-line flag |
| What does the model output? | 5 probabilities — one for each KL Grade (0, 1, 2, 3, 4). The grade with the highest probability is the prediction |

### What Is KL Grading?

The **Kellgren-Lawrence (KL) scale** is the standard medical grading system for knee osteoarthritis severity:

| KL Grade | Meaning | What It Looks Like on X-ray | Safe Angle Ceiling | Health Score |
|---|---|---|---|---|
| 0 | Normal | Clean joint space, no bony growths | 120° | 95/100 |
| 1 | Doubtful | Possible tiny bony spurs, joint space still normal | 120° | 80/100 |
| 2 | Mild | Definite bony spurs, slight joint narrowing | 90° | 60/100 |
| 3 | Moderate | Multiple bony spurs, obvious narrowing, some bone hardening | 60° | 35/100 |
| 4 | Severe | Large bony spurs, very narrow/gone joint space, bone-on-bone | 45° | 15/100 |

---

## 📄 File 1: `prepare_dataset.py` — Data Preparation

### What Does This File Do?
Think of it as the **kitchen prep** before cooking. It doesn't train anything itself — it finds the raw X-ray images on your computer, organises them neatly, and hands them to the training script in the right format.

### Step-by-Step Walkthrough

#### 1. Finding the Dataset
The function `find_dataset_root()` searches for a folder that contains a `train/` subfolder. It looks in three places:
- The path you give it via `--data_dir`
- Kaggle notebook input mounts (if running on Kaggle)
- The current working directory

> **Why?** The Kaggle dataset folder structure has changed over the years, so the code searches multiple levels deep rather than hardcoding one exact path.

#### 2. Expected Folder Layout
```
Dataset Root/
├── train/
│   ├── 0/   ← hundreds of Grade 0 X-ray images (PNG)
│   ├── 1/   ← Grade 1 images
│   ├── 2/   ← Grade 2 images
│   ├── 3/   ← Grade 3 images
│   └── 4/   ← Grade 4 images
├── val/
│   ├── 0/ 1/ 2/ 3/ 4/
└── test/
    ├── 0/ 1/ 2/ 3/ 4/
```

Each subfolder name IS the grade label. PyTorch's `ImageFolder` class reads this layout automatically.

#### 3. Auto-Creating Validation Split
If the dataset only has `train/` and `test/` (no `val/`), the code automatically moves 15% of the training images into a new `val/` folder. It does this **once** using a fixed random seed (42) so it's reproducible.

> **Why?** We need `val/` to check how the model is doing *during* training without touching the `test/` set (which is reserved for the final grade).

#### 4. Image Transforms (Preprocessing)

| Transform | Used During | What It Does | Why |
|---|---|---|---|
| `Resize(224×224)` | Training + Evaluation | Shrinks/stretches all images to the same size | Neural networks need fixed-size inputs |
| `RandomHorizontalFlip` | Training only | Randomly mirrors the image left↔right | Left knee ≈ mirrored right knee — doubles the effective data |
| `RandomAffine` | Training only | Small random rotation (±10°), shift (±5%), and zoom (90–110%) | Teaches the model to handle slightly misaligned X-rays |
| `ColorJitter` | Training only | Slightly changes brightness and contrast | Makes the model robust to different X-ray machines |
| `ToTensor` | Both | Converts the image from a picture to numbers (0.0–1.0) | PyTorch only works with numbers (tensors) |
| `Normalize(ImageNet)` | Both | Subtracts the ImageNet mean and divides by std | The pre-trained model was trained on ImageNet-normalised images, so we must match that |

> **Key design choice:** Augmentation is *deliberately mild*. Aggressive random cropping could cut out the joint space — the very feature the KL grade depends on.

#### 5. Handling Class Imbalance
Real-world X-ray datasets are imbalanced — lots of Grade 0/1 images but few Grade 3/4. The function `compute_class_weights()` calculates **inverse-frequency weights**:

```
If Grade 4 has only 200 images and Grade 0 has 2000 images,
Grade 4 gets a 10× higher weight in the loss function
so the model pays equal attention to rare severe cases.
```

#### 6. DataLoaders
The function `get_dataloaders()` wraps everything into PyTorch `DataLoader` objects that feed batches of 32 images at a time to the model during training. It optionally uses a `WeightedRandomSampler` to oversample rare classes.

---

## 📄 File 2: `train.py` — Model Training

### What Does This File Do?
This is the **main training script**. It takes the prepared data, builds the EfficientNet model, and teaches it to distinguish KL Grades 0–4 through two training phases.

### Training Pipeline (Step by Step)

```
┌───────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ prepare_dataset│────▶│  Phase 1: Warmup  │────▶│ Phase 2: Finetune│
│ (load images) │     │ (head only, 5 ep) │     │ (all layers, 25) │
└───────────────┘     └──────────────────┘     └──────────────────┘
                                                        │
                                   ┌────────────────────┘
                                   ▼
                          ┌─────────────────┐
                          │ Save best_model  │
                          │ Evaluate on test │
                          │ Plot curves      │
                          └─────────────────┘
```

#### Step 1: Build the Model (`build_model()`)

| Part | What Happens |
|---|---|
| Load EfficientNet B0/B3 | Download a pre-trained model that already knows how to see edges, shapes, textures from ImageNet (1.2M photos) |
| Replace the head | The original model outputs 1000 classes (dog, cat, car…). We replace the last layer with: `Dropout(0.4)` → `Linear(→ 5 classes)` so it outputs 5 KL grades instead |
| Move to GPU | If a GPU is available, the model is moved there for faster training |

#### Step 2: Phase 1 — Warm-Up (5 epochs)
```
Backbone: ❄️ FROZEN (all pre-trained layers locked)
Training: Only the new 5-class head
Learning rate: 0.001 (relatively high)
```

> **Why freeze?** The new head starts with random numbers. If we let gradients flow back through the entire network right away, those random-noise gradients would destroy the good features the backbone already learned from ImageNet. So we first train *only* the head until it produces sensible outputs.

#### Step 3: Phase 2 — Fine-Tuning (up to 25 epochs)
```
Backbone: 🔥 UNFROZEN (all layers trainable)
Training: Entire network end-to-end
Learning rate: 0.0001 (10× lower than Phase 1)
LR schedule: Cosine Annealing (smoothly decays to near zero)
Early stopping: If accuracy doesn't improve for 6 epochs, stop
```

> **Why cosine annealing?** The learning rate starts at 0.0001 and smoothly decreases like a cosine wave. This helps the model settle into a good solution without jumping around.

#### Step 4: Loss Function & Optimiser

| Component | Choice | Why |
|---|---|---|
| Loss function | `CrossEntropyLoss` with class weights + label smoothing (0.05) | Class weights fix the imbalance problem; label smoothing prevents the model from being overconfident |
| Optimiser | `AdamW` | Modern optimizer that adapts learning rates per-parameter and includes weight decay (regularisation) to prevent overfitting |
| Mixed precision (AMP) | Enabled on GPU | Runs some calculations in 16-bit instead of 32-bit — trains ~2× faster with almost no accuracy loss |

#### Step 5: Tracking & Early Stopping
After each epoch, the model is evaluated on the validation set. The code tracks:
- **Training loss & accuracy** — how well it fits the training data
- **Validation loss, accuracy & macro F1** — how well it generalises to unseen data

If validation accuracy doesn't improve for 6 consecutive epochs (`patience=6`), training stops early to avoid wasting time and overfitting.

The **best model** (highest val accuracy) is saved to `best_model.pth`.

#### Step 6: Outputs Saved

| Output File | What It Contains |
|---|---|
| `best_model.pth` | Model weights + architecture name + label mappings + image size — everything needed to reload the model |
| `label_map.json` | Human-readable mapping: `{0: "Grade 0 - Normal", 1: "Grade 1 - Doubtful", ...}` |
| `training_curves.png` | Plot of loss and accuracy over all epochs, with a vertical line showing where Phase 2 starts |
| `confusion_matrix_test.png` | Grid showing how many images of each true grade were predicted as each grade |
| `test_report.json` | Precision, recall, and F1 score for each grade on the test set |

### Training Hyperparameters Summary

| Parameter | Default Value | What It Controls |
|---|---|---|
| `--arch` | `b0` | EfficientNet variant (b0 = smaller/faster, b3 = bigger/better) |
| `--img_size` | 224 | Input image dimensions (224×224 pixels) |
| `--batch_size` | 32 | Number of images processed at once |
| `--warmup_epochs` | 5 | Epochs training only the classifier head |
| `--finetune_epochs` | 25 | Max epochs fine-tuning the whole network |
| `--head_lr` | 0.001 | Learning rate for Phase 1 (head only) |
| `--finetune_lr` | 0.0001 | Learning rate for Phase 2 (full network) |
| `--patience` | 6 | Stop early if no improvement for this many epochs |
| `--seed` | 42 | Random seed for reproducibility |

---

## 📄 File 3: `inference.py` — Using the Trained Model

### What Does This File Do?
This is the **prediction engine**. Once the model is trained, this file loads the saved weights and uses them to classify new X-ray images. It serves two roles:
1. **Standalone CLI tool** — classify a single image, a folder of images, or evaluate a labeled test set from the command line.
2. **Library imported by the web backend** — the `KneeClassifier` class is imported by `main.py` to serve predictions via the API.

### Key Components

#### 1. Image Validation (`validate_image()`)
Before running any prediction, the image goes through a quality check:

| Check | Condition | What It Catches |
|---|---|---|
| Contrast too low | `std < 15` | Blank/flat images that aren't real X-rays |
| Completely black | `mean < 10` | Corrupted or empty files |
| Overexposed/blank | `mean > 245` | White/blank uploads |

> If any check fails, the user gets a clear error message instead of a garbage prediction.

#### 2. `KneeClassifier` Class — The Main Prediction Wrapper

**On startup (`__init__`):**
1. Checks if `best_model.pth` exists next to this file
2. If YES → loads the trained model into memory (**real mode**)
3. If NO → switches to **demo mode** (no crash, just fake but deterministic predictions)

**On each prediction (`predict()`):**

##### Real Mode (`_real_predict`)

| Step | What Happens |
|---|---|
| 1. Open image | Converts uploaded bytes → PIL Image → RGB |
| 2. Preprocess | Resize to 224×224, normalize with ImageNet mean/std |
| 3. Test-Time Augmentation (TTA) | Creates a horizontally flipped copy of the image |
| 4. Forward pass × 2 | Runs both the original and flipped image through the model |
| 5. Average probabilities | Averages the two sets of probabilities for a more stable prediction |
| 6. Pick winner | The class with the highest average probability is the predicted KL Grade |
| 7. Map to clinical values | Looks up the health score and safe angle ceiling from the grade |

> **Why TTA?** Flipping the image and averaging predictions makes the result more robust — it reduces the chance of the model being confused by whether it's a left or right knee.

##### Demo Mode (`_demo_predict`)
When no trained weights are available, the system doesn't crash. Instead:
1. Takes the first 2048 bytes of the uploaded image
2. Computes an MD5 hash → converts to a number 0–99
3. Maps that number to a KL Grade using approximate real-world prevalence:

| Hash Range | Assigned Grade | Approximate Prevalence |
|---|---|---|
| 0–24 | Grade 0 (Normal) | 25% |
| 25–49 | Grade 1 (Doubtful) | 25% |
| 50–69 | Grade 2 (Mild) | 20% |
| 70–86 | Grade 3 (Moderate) | 17% |
| 87–99 | Grade 4 (Severe) | 13% |

> **Key property:** The same image always returns the same grade (deterministic). This lets frontend developers test the full upload → results → exercises → tracker flow without needing a trained model.

#### 3. Checkpoint Loading (`load_checkpoint()`)
The saved `best_model.pth` file contains everything needed to recreate the model:

| Saved Field | What It Stores |
|---|---|
| `model_state_dict` | All the learned weights and biases |
| `arch` | Which EfficientNet variant was used (`"b0"` or `"b3"`) |
| `num_classes` | Number of output classes (5) |
| `idx_to_grade` | Maps model output index → KL Grade number |
| `idx_to_name` | Maps model output index → human-readable label |
| `img_size` | Image size the model was trained on (224) |

> Because the architecture name is saved *inside* the checkpoint, there's zero risk of accidentally loading B3 weights into a B0 model.

#### 4. CLI Usage Modes

| Command | What It Does |
|---|---|
| `python inference.py --image xray.png` | Predicts one image and prints the grade + confidence |
| `python inference.py --image_dir folder/` | Predicts every image in a folder, saves results to `predictions.json` |
| `python inference.py --eval_dir test/` | Evaluates a labeled folder (subfolders 0–4), prints accuracy + classification report |

---

## 🔗 How the Three Files Connect

```
┌──────────────────────┐
│  prepare_dataset.py  │
│  (data loading)      │
│                      │
│  • find dataset      │
│  • create val split  │
│  • image transforms  │
│  • class weights     │
│  • DataLoaders       │
└──────────┬───────────┘
           │ imports
           ▼
┌──────────────────────┐         ┌──────────────────────┐
│     train.py         │         │    inference.py       │
│  (model training)    │         │  (model prediction)   │
│                      │         │                       │
│  • build EfficientNet│         │  • load checkpoint    │
│  • Phase 1: warmup   │────────▶│  • validate image     │
│  • Phase 2: finetune │  saves  │  • real predict (TTA) │
│  • early stopping    │  .pth   │  • demo fallback      │
│  • save best model   │         │  • CLI + API class    │
└──────────────────────┘         └───────────┬──────────┘
                                             │ imported by
                                             ▼
                                    ┌─────────────────┐
                                    │    main.py       │
                                    │  (FastAPI server)│
                                    └─────────────────┘
```

---

## 📊 Algorithms & Techniques Summary Table

| Technique | Where Used | Simple Explanation |
|---|---|---|
| **EfficientNet** | `train.py`, `inference.py` | A neural network architecture that classifies images accurately while being small and fast |
| **Transfer Learning** | `train.py` | Starting from a model pre-trained on ImageNet instead of training from zero — needs far less medical data |
| **Two-Phase Training** | `train.py` | First train only the new head (safe), then fine-tune everything (powerful) |
| **AdamW Optimizer** | `train.py` | An optimiser that adapts learning rates per-parameter and includes weight decay regularisation |
| **Cosine Annealing LR** | `train.py` | Learning rate smoothly decreases like a cosine wave over training for stable convergence |
| **CrossEntropyLoss** | `train.py` | Standard loss function for multi-class classification (5 KL grades) |
| **Class Weighting** | `prepare_dataset.py`, `train.py` | Gives rare grades (3, 4) higher importance so the model doesn't just learn to always predict Grade 0 |
| **Label Smoothing** | `train.py` | Prevents overconfidence by slightly softening the target labels (e.g., target becomes 0.95 instead of 1.0) |
| **Mixed Precision (AMP)** | `train.py` | Uses 16-bit floats for some GPU calculations — ~2× faster training with negligible accuracy loss |
| **Early Stopping** | `train.py` | Stops training when validation accuracy stops improving for 6 epochs to prevent overfitting |
| **Data Augmentation** | `prepare_dataset.py` | Random flips, rotations, brightness changes during training to artificially expand the dataset |
| **CLAHE** | `prepare_dataset.py` (via OpenCV) | Contrast enhancement that improves visibility of joint structures in X-rays |
| **ImageNet Normalisation** | `prepare_dataset.py` | Subtracts ImageNet mean/std from pixel values so they match what the pre-trained model expects |
| **Test-Time Augmentation** | `inference.py` | Predicts on both the original and flipped image, averages the result for higher confidence |
| **Softmax** | `inference.py` | Converts raw model outputs (logits) into probabilities that sum to 1.0 |
| **Weighted Random Sampling** | `prepare_dataset.py` | Optionally oversamples rare classes so each training batch has a balanced mix of all grades |
| **Macro F1 Score** | `train.py` | Evaluation metric that averages F1 across all classes equally — fair even when classes are imbalanced |
| **Confusion Matrix** | `train.py` | Grid visualisation showing correct vs. incorrect predictions for every grade pair |
| **Demo Mode (MD5 Hash)** | `inference.py` | Deterministic fake predictions when no trained model exists, allowing full system testing |
