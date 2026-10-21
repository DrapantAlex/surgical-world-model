# Decision record — 2026-10-20: next direction after the first STAR-IL diagnostics

**Status:** decided, before any new training run; E0 done (see §4.1)
**Reference run:** `star-il-simplecnn-latent512-seed0` (simple CNN encoder, latent 512, seed 0)
**Dataset:** STAR-IL (`Surgical/jhu/imerse/star_il/star_il`), 409 train / 51 val episodes; the test split is not used
**Evidence:** `outputs/star-il-simplecnn-latent512-seed0/val_metrics_reeval.json`,
`.../check_orientation_error.json`, `scripts/check_quaternion_sign.py` output (all below)

---

## 1. Context: what we observed

### 1.1 Actor: position is learned, orientation is not
Per-dimension val MSE (normalised units, ≈ 1 − R²), 20,111 val pairs:

| Action dim | Model | Mean-action baseline |
|---|---|---|
| `x`, `y`, `z` | 0.034, 0.029, 0.011 | 1.04, 1.09, 0.80 |
| `ori.x`, `ori.y`, `ori.z`, `ori.w` | 0.181, 0.121, 0.201, 0.157 | 1.04, 0.97, 1.10, 1.01 |
| **all 7** | **0.105** | 1.008 |

The quaternion accounts for ~90% of the summed action MSE.

### 1.2 Generalisation gap: on every dimension, much larger for orientation
Train subset (20 episodes evenly spaced over the train split) vs val, same weights:

| | Train | Val | Val / train |
|---|---|---|---|
| Action MSE | 0.0009 | 0.105 | ×116 |
| Position dims | 0.0003–0.0011 | 0.011–0.034 | ×27–49 |
| Orientation dims | 0.0006–0.0015 | 0.12–0.20 | ×100–335 |
| Rotation error, median / mean / p90 (deg) | 0.75 / 0.87 / 1.6 | 3.6 / 6.6 / 14.9 | |

Training history: train action loss falls ×80 (0.033 → 0.0004), val only −38% (0.072 → 0.045, best at
epoch 13 of 24) and is flat from epoch 4. The model memorises the training episodes.

### 1.3 Orientation is an episode-level quantity
Within an episode the tool orientation barely changes; across episodes it varies widely
(rotation from the train-mean orientation: median 33.5°, p90 104°, max 122.5°).
At 30 fps the ~174k training frames therefore contain only ~409 distinct orientations — one per
episode — which invites a lookup shortcut ("this scene looks like training episode k → its orientation").

### 1.4 The quaternion sign in the labels is inconsistent (`check_quaternion_sign.py`, all 409 train episodes)
q and −q are the same rotation, but far apart for an MSE loss.

| Finding | Value |
|---|---|
| Sign flips between consecutive frames (label jumps, not robot motion) | 71 in 174,035 frames, in 41 episodes |
| Episodes on the + / − side of the train-mean rotation | 259 / 150 |
| Opposite-sign neighbours, **nearest-neighbour definition**: episodes whose single most similar episode in rotation (\|dot\|, sign ignored) is within 10° AND stored with the opposite sign | 33 of 405 (8%) |
| Opposite-sign neighbours, **any-neighbour definition**: episodes with at least one episode (any, not only the nearest) within 10° stored with the opposite sign | 123 of 409 (30%) |
| Frames with w < 0 / with \|w\| < 0.1 | 56% / 79% (the usual "w ≥ 0" rule cannot work) |
| Quaternion variance (sum of 4 components), stored → sign-canonicalised | 0.89 → 0.20 (**78% from the sign**) |

The two neighbour definitions differ by ~4× because each episode has a median of 24 episodes within 10°;
the nearest-neighbour count only looks at one of them. Each episode's orientation is the mean of its frames
(after aligning them to the sign of its first frame).

### 1.5 The sign inconsistency measurably hurts the trained actor (`check_orientation_error.py`)

| Val | Value |
|---|---|
| Frames predicted with the wrong sign (dot(q_pred, q_true) < 0) | 4.9% (train: 0%) |
| Frames with \|q_pred\| < 0.7 ("averaging" +q and −q) | 10.5% (train: 0%) |
| Orientation MSE → after flipping wrong-sign predictions | 0.165 → 0.071 (**−57%**) |
| Action MSE → after flipping wrong-sign predictions | 0.105 → 0.051 |

The worst val episodes are of two kinds:
- **sign confusion** (433, 434, 423, 422, 441): high wrong-sign rate, |q_pred| ≈ 0.4–0.8;
- **real rotation error** (409, 428, 427, 447, 457): correct sign, |q_pred| ≈ 1, 12–21° off.

### 1.6 World-model heads (from 2026-10-19, unchanged)
Both heads are worse than copying the input: next state ×75–9600 per joint vs copy-state,
next image 0.0018 vs 0.0006 copy-frame.

### Summary
Three causes act on the actor's val error:
1. **Inconsistent quaternion sign** in the labels: ~57% of the val orientation MSE, measured.
2. **Few distinct orientations** (~409): the remaining real rotation errors.
3. **General overfitting** on all dimensions (position ×27–49 too).

---

## 2. Options considered

| # | Option | Pros | Cons |
|---|---|---|---|
| A | **Canonicalise the quaternion sign at load time**: flip q so that dot(q, q_ref) > 0, q_ref = Markley mean of the train labels | Cheapest fix for cause 1; Open-H files stay unmodified; written once in `load_split`, applies to every dataset; safe for STAR-IL (all orientations ≤ 122.5° from q_ref, so no remaining discontinuity) | Changes the label statistics, so normalised MSE is no longer comparable with older runs; needs the quaternion position per dataset; must be off for older runs |
| B | Sign-invariant loss: min(‖p − q‖², ‖p + q‖²) | Labels unchanged | Custom loss that must work in raw units (normalisation breaks q ↔ −q); metrics and baselines stay sign-dependent; more code to defend |
| C | 6D rotation representation (Zhou et al., 2019) | Continuous by construction, standard in robot learning, no sign at all | Action dim 7 → 9; conversion needed for metrics and baselines; a larger change than needed to test the sign hypothesis |
| D | Regularisation first (AdamW weight decay, dropout, augmentation) | Targets cause 3, which is real | Does not address cause 1; its effect would be mixed with the label noise |
| E | Scale to more Open-H subsets | Targets cause 2; the planned thesis direction | Does not fix cause 1 and may make it worse (each lab/robot may log a different sign convention); not the current phase |
| F | Relative (delta) actions w.r.t. the current end-effector pose | Common in imitation learning; removes the absolute orientation | STAR-IL state is joint angles, not the end-effector pose; would need forward kinematics |
| G | State-only baseline now (A2) | Already planned | Trained on the same inconsistent labels, so its result would be confounded |

---

## 3. Decision

**Option A first (canonicalise the sign at load time), then the state-only baseline and regularisation on the
canonicalised labels. Option C is kept as a later follow-up.**

Why:
- Cause 1 is the only one with a **measured** size (~57% of the val orientation MSE), and A is the cheapest fix.
- It is a **prerequisite** for everything after it: a state-only baseline, a regularisation experiment, or scaling to more
  datasets would all be measured through the same label noise.
- It keeps the project's principle: **the dataset stays read-only; representation choices live in the code**
  (like normalisation and the 224 px resize).
- One variable per experiment: the first canonicalised run changes **only** the labels, with the same
  hyperparameters as the reference run, so any difference is attributable to it.

**Comparison rule from now on:** normalised MSE is not comparable across label conventions
(canonicalisation shrinks the std of `ori.y` from 0.71 to 0.17, so the same physical error gives a larger
normalised MSE). Runs with different conventions are compared on **rotation error in degrees**
(sign-invariant) and **position error in raw units**.

---

## 4. Planned experiments

| # | Experiment | Hypothesis | Confirmed if | Rejected if |
|---|---|---|---|---|
| E0 | Re-run the val baselines and `check_quaternion_sign.py` on canonicalised labels (no training) | Canonicalisation removes all sign inconsistency | 0 consecutive-frame flips, 0 opposite-sign neighbours, quaternion variance ≈ 0.20 | any flips remain |
| E1 | **Reference run with canonicalised labels**, otherwise identical to `star-il-simplecnn-latent512-seed0` | The sign convention is a major cause of the val orientation error | val wrong-sign ≈ 0%, \|q_pred\| < 0.7 ≈ 0%, and val rotation error **clearly below 6.6° mean / 3.6° median**; position error ≈ unchanged (≈ 0.0057 raw, control) | val rotation error not lower than 6.6° mean |
| E1b | (prediction, tested by E1) | The remaining causes (2, 3) keep val far from train | val rotation error stays well above train (0.87°) | val ≈ train: then causes 2–3 were overestimated |
| E2 | **State-only baseline** (MLP: joint angles → action) on canonicalised labels | The joints determine the end-effector pose (forward kinematics), so the state alone predicts most of the action | state-only val rotation and position error ≤ E1's | clearly worse than E1: the image contributes information the state lacks |
| E3 | **Weight decay (AdamW)** on top of E1, everything else unchanged | Regularisation reduces memorisation (cause 3) | lower val action loss and rotation error than E1, with a smaller train/val gap | no val improvement, or both train and val get worse |
| E4 | `check_quaternion_sign.py` on tissue lift (both arms, action and state) | The sign issue is not specific to STAR-IL | flips / opposite-sign neighbours found | none found: canonicalisation is a no-op there |
| E5 | (conditional on E1) 6D rotation representation | A continuous representation helps where canonicalisation is not enough | lower val rotation error than E1 | no gain: keep canonicalised quaternions |

E2 is especially important for the thesis: if a state-only MLP matches E1, the image encoder (the final product)
currently contributes little to the actor.

### 4.1 E0 result (2026-10-20): confirmed
`python -m scripts.check_quaternion_sign --split {train,val} [--canonical-quaternion]`. The reference rotation is
always the train Markley mean, `[-0.587, 0.808, 0.009, 0.039]`, also for val.

| | Train, stored | Train, canonical | Val, stored | Val, canonical |
|---|---|---|---|---|
| Within-episode sign flips | 71 / 174,035 frames | **0** | 3 / 20,162 frames | **0** |
| Opposite-sign nearest neighbours (within 10°) | 33 of 405 | **0** of 405 | 3 of 44 | **0** of 44 |
| Frames on the opposite side of the reference | 68,308 | **0** | 7,608 | **0** |
| `ori.x` / `ori.y` std | 0.612 / 0.710 | 0.399 / 0.174 | 0.624 / 0.699 | 0.390 / 0.169 |
| Largest angle of any frame from the reference | 123.5° | 123.5° | 120.8° | 120.8° |

All three criteria are met: 0 flips, 0 opposite-sign neighbours, and every angle well below the 150° safety limit
(180° would make the sign ambiguous again). Val's raw labels had the same inconsistency as train (38% of frames on
the other side of the reference); the train reference fixes them without using val statistics.
The canonical-signs val baselines (`scripts/baselines.py --canonical-quaternion`) are still to be run.

---

## 5. Open questions

1. **Units:** are STAR-IL positions in metres and joints in radians? Not documented in `info.json`; all physical
   numbers above are "raw units" until confirmed.
2. **Where the quaternion is, per dataset:** detect it from the `info.json` names (`*ori.x…w`, `*orientation.x…w`)
   or declare an explicit per-dataset layout (the existing TODO in `models/metrics.py`)? Tissue lift has two
   quaternions per vector (one per arm), in both action and state.
3. **Reference rotation:** the train Markley mean works for STAR-IL (max 122.5° from it). Should we check, for every
   new dataset, that all orientations are < 180° from the reference, and what do we do if not (go to 6D)?
4. **Backward compatibility:** a config key (e.g. `canonical_quaternion`) that defaults to off when missing, so
   older runs still load and re-evaluate unchanged. Should the reference rotation be saved with the run, next to
   `norm_stats.json`?
5. **Checkpoint selection:** `MONITOR = val_action_output_loss` is in normalised units; within a run it stays
   valid, but should we also log the rotation error in degrees per epoch?
6. **Permanent metric:** should the rotation error in degrees become a core metric in `models/metrics.py`
   (requires the per-dataset layout from question 2)?
7. **Seeds:** all comparisons so far use one seed. How large is seed-to-seed variation? Should E1 and E3 use
   2–3 seeds before drawing conclusions?
8. **Origin of the sign convention:** why does the STAR-IL logging switch the sign (KUKA controller, ROS
   conversion)? Not needed for the fix, but worth one sentence in the thesis.
9. **World model:** residual targets (Δstate, Δimage) for the copy-baseline problem — still open, to discuss with Michael.
10. **E4 on the tissue lift state quaternions (deferred):** E4 (2026-10-21) checked only the action quaternions
    (`--quat-start 3` and `11`, train and val): 0 flips, 0 opposite-sign neighbours, w ≥ 0 in every frame, so
    tissue lift has a consistent sign convention. `check_quaternion_sign.py` cannot read the state
    (`psm*_pose.orientation`) yet. Add a `--key {action,observation.state}` flag and run it **before the first
    tissue lift training with `--canonical-quaternion`**. Also at that point: the Markley reference for PSM1
    came out with w < 0 (eigenvector sign is arbitrary), so canonicalisation would flip *every* PSM1 label to −q.
    That is harmless but surprising, so consider choosing the reference sign with w ≥ 0.
