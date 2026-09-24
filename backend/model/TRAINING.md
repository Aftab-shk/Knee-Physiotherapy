# Training the Knee-OA Classifier

Quick reference for the improved EfficientNet-B4 recipe. For the plain-English
walkthrough of the whole pipeline, see `../../docs/model_explanation.md`.

## The command

On Kaggle (GPU T4/P100), with the dataset attached:

```bash
python train.py \
  --data_dir "/kaggle/input/knee-osteoarthritis-dataset-with-severity" \
  --arch b4 \
  --num_workers 4 \
  --tta
```

Everything else is already defaulted to the tuned recipe. On a 16 GB T4 this
takes roughly 2.5–4 hours for the full 3 + 40 epochs; early stopping usually
ends it sooner.

If you hit **CUDA out of memory**, lower the batch size first — it is the only
knob that needs changing:

```bash
--batch_size 16      # or 12 if 16 still OOMs
```

To reproduce the old behaviour exactly:

```bash
python train.py --arch b3 --img_size 224 --mild_aug \
  --mixup 0 --cutmix 0 --ema_decay 0 --ordinal_weight 0 \
  --llrd 1.0 --class_weight_mode inverse --label_smoothing 0.05 \
  --warmup_epochs 5 --finetune_epochs 25 --patience 6 --finetune_lr 1e-4
```

## What to expect

Published results for 5-class KL grading on this dataset sit around **68–75%**
test accuracy. Radiologist inter-observer agreement on KL grading is itself
only about 60–70% — Grade 0 vs Grade 1 differ by "doubtful osteophytes" and are
genuinely ambiguous — so this is a hard ceiling, not a tuning failure.

Watch **within-one-grade accuracy** (printed after training, and stored in
`test_report.json`) alongside raw accuracy. It should land around 92–97%, and
it is the number that reflects whether the downstream exercise protocol would
have been roughly right.

## Why each change helps

| Change | Approx. gain | Reasoning |
|---|---|---|
| Native resolution (380px for B4, not 224px) | +3 to 6 pts | EfficientNet's compound scaling only pays off at the resolution it was designed for. Running B3/B4 at B0's 224px is the most common reason a "bigger" model fails to beat B0. |
| MixUp / CutMix + stochastic depth | +2 to 4 pts | A 17.6M-parameter model on ~5.8k training images will otherwise memorize them. |
| `sqrt_inverse` instead of `inverse` class weights | +1 to 3 pts | Full inverse frequency weights Grade 4 (~173 images) about 13× Grade 0 (~2286). That maximizes balanced recall at the direct cost of overall accuracy. |
| Ordinal loss term | +1 to 2 pts | KL grades are ordered; plain cross-entropy treats 0→4 as no worse than 0→1. Pulls residual errors onto the diagonal's neighbours. |
| Layer-wise LR decay | +1 to 2 pts | Early blocks hold generic edge/texture filters ImageNet already got right; late blocks need to re-specialize for radiographs. One global LR forces a bad compromise between the two. |
| EMA weight averaging | +1 to 2 pts | The average of the last few thousand steps beats wherever the final step happened to land. |
| Flip TTA | +0.5 to 1 pt | One extra forward pass; a mirrored knee is a real, label-preserving variation. |

## Options worth trying

- `--clahe` — contrast-equalizes each radiograph tile-by-tile. Helps when
  exposure varies between source machines. Recorded in the checkpoint, so
  `inference.py` re-applies it automatically. Worth an A/B run.
- `--select_metric f1` — pick the checkpoint by macro-F1 instead of accuracy if
  you care more about the rare Grade 4 than about the headline number.
- `--weighted_sampler` — oversamples rare grades per batch instead of weighting
  the loss. The script automatically drops loss weights to uniform when this is
  on, so the imbalance is not corrected twice.

## Diagnosing a finished run

Read the per-class report, not just the headline accuracy. A run that scored
69.9% overall looked like this:

| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 – Normal | 0.74 | **0.88** | 0.81 | 639 |
| 1 – Doubtful | **0.35** | **0.22** | **0.27** | 296 |
| 2 – Mild | 0.70 | 0.68 | 0.69 | 447 |
| 3 – Moderate | 0.82 | 0.79 | 0.81 | 223 |
| 4 – Severe | 0.79 | 0.90 | 0.84 | 51 |

Two things to notice, both actionable:

**1. One class carries almost all the loss.** Grade 1 recall of 0.22 means the
model finds barely a fifth of Doubtful cases, while Grade 0 has high recall
(0.88) but weak precision (0.74) — Grade 1 images are being absorbed into
Grade 0. With 296 Grade 1 samples, lifting recall to 0.50 is worth **+5 points
of overall accuracy**, more than every other available change combined. Attack
it with `--class_weights` and the logit adjustment below.

**2. Check whether the LR schedule actually finished.** The cosine schedule is
sized by `--finetune_epochs`. If early stopping fires well before that number,
the LR never annealed to near-zero and you stopped mid-descent — models
typically make their sharpest gains in those final low-LR epochs. In the run
above, `--finetune_epochs 40` stopped at epoch 24, so the LR was still at ~35%
of peak. Two fixes: set `--finetune_epochs` to where it actually stops and
raise `--patience` to match, or `--resume` and anneal from there.

## Continuing a run that stopped early

`--resume` loads existing weights, skips the warm-up phase, and restarts a
fresh cosine schedule — so a run that early-stopped mid-anneal can be finished
in ~20 minutes instead of retraining for hours:

```bash
python train.py --data_dir "..." --arch b4 \
  --resume outputs/best_model.pth \
  --finetune_epochs 10 --finetune_lr 6e-5 --patience 10 \
  --mix_prob 0.3 --num_workers 4 --tta \
  --output_dir outputs_v2
```

The resumed model is scored on val first, and the checkpoint is only
overwritten if training beats that baseline — so this can't lose you ground.
Use a lower `--finetune_lr` than the original run (it is an anneal, not a
restart) and ease off the regularization with `--mix_prob 0.3`.

> On Kaggle, `/kaggle/working` is wiped when the session ends. To resume in a
> later session, upload the previous `best_model.pth` as a Kaggle Dataset and
> point `--resume` at `/kaggle/input/<your-upload>/best_model.pth`.

## Logit adjustment (on by default)

After training, the script sweeps a class-prior correction on the validation
split — `adjusted_logits = logits - tau * log(class_prior)` — and stores the
best `tau` in the checkpoint. The model is trained where Grade 0 outnumbers
Grade 4 ~13×, so it leans on that prior and over-predicts common grades; this
cancels the lean back out, which is exactly the Grade 0 / Grade 1 failure
above. `tau=0` is always in the sweep, so it can never make validation worse.

`inference.py` reads `tau` out of the checkpoint automatically, so serving
predictions match the reported test accuracy. Disable with `--logit_adjust off`
or pin a value with `--logit_adjust 0.5`.

## Calibration and OOD screening (on by default)

Two more things are fitted on the validation split once training finishes. Both
are post-hoc, cost one forward pass, and neither can change a prediction.

**Temperature scaling.** A single scalar dividing the logits, fitted to minimise
val NLL, so the confidence shown to a patient beside a movement restriction
means what it says.

Measured on this dataset (826 val / 1656 test, fitted on val only):

| pipeline | acc | macro-F1 | within-1 | ECE | mean conf |
|---|---|---|---|---|---|
| raw softmax | 0.6987 | 0.6839 | 0.9517 | 0.0810 | 0.6355 |
| + prior correction (τ=0.05) | 0.7035 | 0.6910 | 0.9529 | 0.0804 | 0.6318 |
| **+ temperature (T=0.734)** | 0.7029 | 0.6902 | 0.9529 | **0.0407** | 0.7177 |

**T came out below 1.0** — this model was *under*-confident, not over-confident.
It reported 0.636 mean confidence against 0.699 accuracy, so calibration
*sharpened* it. That is the opposite of the usual deep-network failure, which is
why the fit brackets both sides of 1.0 instead of assuming T>1. Calibration
error halves; the confidence gap goes from −0.063 to +0.015.

**On argmax:** dividing logits is monotonic, so on a single forward pass
temperature cannot change a prediction. With flip-TTA the two branches are
averaged *after* softmax, and that average is not monotonic in T — a sharper T
lets the more confident branch dominate. Measured drift: **1 image in 1656
(0.06%)**. Small, but not zero. `train.py` warns above 0.5%.

**Energy OOD reference.** `E(x) = -logsumexp(logits)`, low for inputs the model
recognises and high for ones it does not. The classifier has five outputs and no
"not a knee" class, so nothing else stops a chest film or a photo of a wall from
receiving a confident grade that then sets a movement ceiling. Validation
percentiles are stored and become the serving thresholds — above p95 warns, well
beyond p99 rejects with a 422. Energy far *below* p50 is refused too: that is
where photos and screenshots land, because the network answers them with huge
activations (see `outside_energy_range()` in inference.py).

Disable both with `--no_calibrate` (debugging only). A checkpoint trained without
them still serves: `inference.py` falls back to raw softmax, reports
`calibrated: false` and `ood_screening: false`, and logs a warning at startup
instead of quietly implying a certainty it never validated.

The shipped checkpoint now carries all of it: `logit_tau=0.05`, `class_priors`,
`temperature=0.7344`, and an energy reference (`p50=-2.344 p95=-1.978
p99=-1.828`). Weights are byte-identical to before — only metadata was added.

## Targeting a specific weak class

`--class_weights` overrides the automatic weighting with explicit per-class
values. To push harder on Grade 1 without disturbing the rest:

```bash
--class_weights "1.0,1.7,1.1,1.0,1.0"
```

Raise the Grade 1 value gradually (1.5 → 1.7 → 2.0) and watch its recall in the
report. Push too far and Grade 0 precision collapses instead — the errors move
rather than disappear.

## Squeezing out more

Ensembling is more reliable than a bigger single model. Train three seeds:

```bash
for s in 42 43 44; do
  python train.py --arch b4 --seed $s --output_dir outputs/seed$s --tta
done
```

Averaging their softmax outputs is typically worth another 1–3 points.

## Deploying

Copy the trained checkpoint next to `inference.py`:

```bash
cp outputs/best_model.pth backend/model/best_model.pth
```

The checkpoint records its own `arch`, `img_size`, and `clahe` flag, so
inference preprocessing always matches training — no flags to keep in sync.

**Serving cost.** The backend runs on CPU. Measured latency per image,
including flip TTA:

| Model | Params | CPU latency |
|---|---|---|
| B0 @ 224 | 4.0M | ~110 ms |
| B3 @ 300 | 10.7M | ~250 ms |
| B4 @ 380 | 17.6M | ~490 ms |

~0.5 s per upload is fine for one-X-ray-at-a-time use. If you later need
higher throughput, that is the tradeoff to revisit.

## Training-only dependencies

`requirements.txt` covers serving only. Training additionally needs:

```bash
pip install matplotlib scikit-learn
```

Both are preinstalled on Kaggle. `opencv-python-headless` (already in
`requirements.txt`) is only needed if you use `--clahe`.
