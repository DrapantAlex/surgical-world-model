# Experiment comparison (STAR-IL, val split)

All numbers are on the **val split** (51 episodes), mean over **3 seeds** unless a range is given (min–max over seeds).
The test split was never used. Sources: research diary (2026-10-19 to 2026-11-13) and the scripts named in the last section.
"Trimmed" = only the frames where the tool is in view (`configs/keep_ranges/star_il_tool_visible.csv`): same 51 val
episodes, fewer pairs.

**How to read it.** Only **mm and degrees** compare all experiments, because every actor ends up predicting the command a(t)
(E8/E9: chunk step 1). Losses (Huber, MSE) compare models **only on the same target** (tables 4 and 5).

---

## 1. Overview: what each experiment changed and what it showed

| ID | Change (vs the previous reference) | Question | Result | Verdict |
| --- | --- | --- | --- | --- |
| First run | professor's model on STAR-IL | does it work? | rotation 6.6°, position 5.7 mm | orientation bad: inconsistent quaternion signs |
| **E1** | canonical quaternion signs | is the sign the problem? | 0.92°, 2.90 mm | big gain; new reference |
| **E2** | state-only MLP (no image, no WM) | does the image help? | 0.41°, 1.24 mm | state alone is better (FK shortcut; size confound) |
| **E6** | E1 with WM loss weights = 0 | does the WM help? | 0.89°, 2.92 mm | no detectable effect |
| **E7** | WM state head predicts Δstate | does a motion target help the WM? | actor 6.81°, 11.8 mm (best epoch 1); WM worse than "no change" | memorisation + loss scale ~900× |
| **E8** | actor predicts 16 offsets from a(t−1) | does a motion target help the actor? | ≈ stand-still (≤ 1% better) | no motion learned |
| **E9** | E8 with WM off | does the WM help the chunked actor? | 8–11% worse than stand-still | WM = mild regulariser |
| **E8-trim** | E8 trained on tool-visible frames only | does removing the out-of-view tail help? | = stand-still (1.001 pos., 1.006 rot.); E8 on the same pairs 1.016 / 1.008 | cleaner, lower variance; still no motion learned |

---

## 2. Actor error on the command a(t), val (mm and degrees)

Every actor's prediction of a(t) vs the true command. E8/E9: chunk step 1, a(t) = a(t−1) + o_0.

| Model | Position (mm) | Rotation (°) | Position ÷ stand-still | Rotation ÷ stand-still |
| --- | --- | --- | --- | --- |
| **Stand-still** (rule: a(t) = a(t−1)), E1–E7 pairs | **0.42** | **0.09** | 1.0 | 1.0 |
| **Stand-still**, chunk pairs (E8/E9) | **0.419** | **0.0908** | 1.0 | 1.0 |
| First run (raw quaternion signs) | 5.7 | 6.6 | 13.6× | 73× |
| E1 (full model) | 2.90 | 0.92 | 6.9× | 10.2× |
| E2 (state-only MLP) | 1.24 | 0.41 | 3.0× | 4.6× |
| E6 (WM off) | 2.92 | 0.89 | 7.0× | 9.9× |
| E7 (Δstate WM)¹ | 11.8 (seeds 10.7–12.5) | 6.81 (seeds 6.00–7.63) | 28× | 76× |
| E8 (chunk, WM on)² | 0.414–0.421 | 0.091–0.092 | 0.99–1.00× | 1.00–1.01× |
| E9 (chunk, WM off)² | 0.453–0.485 | 0.094–0.096 | 1.08–1.16× | 1.04–1.06× |
| **Stand-still**, trimmed chunk pairs | **0.390** | **0.0886** | 1.0 | 1.0 |
| E8 on trimmed val (same weights)²³ | 0.391–0.397 | 0.0887–0.0895 | 1.00–1.02× | 1.00–1.01× |
| E8-trim (chunk, WM on, trimmed)²³ | 0.390–0.391 | 0.0889–0.0890 | 1.00× | 1.00–1.01× |

¹ **E7 best epoch = 1 in all three seeds**: these numbers come from epoch-1 weights (early stopping ended each run at
epoch 11). The actor is under-trained rather than overfitted (train/val gap 1.6× vs 7–9× for E1/E6; train rotation
error 4.4°). Paired episode test vs E6: +5.9° (95% CI [+5.3, +6.7]) and +8.9 mm (95% CI [+8.1, +9.6]), so the
Δstate world model hurts the actor on both metrics. Its worst episodes (409, 427) are the same as E1/E6's, 2–3× worse.

² **E8/E9 = chunk step 1** (the prediction of a(t)), from `val_metrics.json`: `chunk/kuka_pose_mm/step_01` and
`chunk/kuka_pose.ori_deg/step_01`. Degrees are computed on the rebuilt quaternion (anchor + offset) with the same
sign-invariant `quaternion_angle_deg`, so they mean the same as the E1–E7 numbers. Chunk runs skip the first frame and
the last 16 positions of each episode, so their stand-still reference is measured on their own pairs
(`chunk_baselines.json`). **The ratio columns are the fairer comparison:** each model is divided by stand-still on its
own pairs. Rotation is shown with two significant figures, otherwise the E8/E9 difference (~0.004°) disappears.

³ **Trimmed rows** are on the 17,692 trimmed val chunk pairs (untrimmed: 19,346), so they are compared only with the
trimmed stand-still row. "E8 on trimmed val" = the E8 weights re-evaluated with `--eval-keep-ranges`
(`val_metrics_keep_star_il_tool_visible.json`).

Note: E8's low error comes from the anchor a(t−1) built into its target, not from learning (its predicted offsets ≈ 0).
E1, E2 and E6 do not see a(t−1), so for them stand-still is a physical reference, not an equal contest.

---

## 3. Chunk error per step (E8, E9 and the chunk baselines), val, 19,346 pairs

| Predictor | Val Huber (best) | Step 1 (mm) | Step 4 (mm) | Step 8 (mm) | Step 16 (mm) | Step 16 (°) |
| --- | --- | --- | --- | --- | --- | --- |
| Stand-still (offset 0) | **0.670** | 0.42 | 1.57 | 2.93 | 5.22 | 1.01 |
| Constant velocity | 0.781 | 0.20 | 0.96 | 2.33 | 5.24 | 1.48 |
| E8 (WM on) | 0.663–0.668 | 0.41–0.42 | 1.55–1.58 | 2.89–2.95 | 5.15–5.26 | 1.01–1.03 |
| E9 (WM off) | 0.720–0.747 | 0.45–0.49 | 1.71–1.84 | 3.21–3.48 | 5.82–6.40 | 1.12–1.16 |

Training dynamics (every seed): best epoch 1 (E8: 1–2); train loss 0.66 → 0.25 while val rises 0.67 → 0.95 (memorisation).

### 3b. Trimmed val (tool visible), 17,692 pairs

| Predictor | Huber ÷ stand-still | Step 1 (mm) | Step 4 (mm) | Step 8 (mm) | Step 16 (mm) | Step 16 (°) | Mean 16 steps (mm) | Mean 16 steps (°) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stand-still (offset 0) | 1.0 | 0.39 | 1.46 | 2.71 | 4.84 | 0.98 | 2.764 | 0.557 |
| Constant velocity | 1.19–1.20⁴ | 0.19 | 0.92 | 2.23 | 4.98 | 1.46 | 2.452 | 0.761 |
| E8 (same weights, re-evaluated) | 0.995–1.003 | 0.39–0.40 | 1.46–1.49 | 2.72–2.77 | 4.86–4.96 | 0.98–1.00 | 2.774–2.825 | 0.558–0.564 |
| E8-trim | 1.000–1.002 | 0.39 | 1.46 | 2.71–2.72 | 4.84–4.85 | 0.99 | 2.765–2.771 | 0.559–0.561 |

⁴ Huber is in each run's own normalisation (trimming changes the stds: action z 29.6 → 20.6 mm), so only the ratio to
stand-still on the same pairs and normalisation is shown. Constant velocity: 0.769 / 0.639 (E8 norm.), 0.924 / 0.780
(E8-trim norm.).

- Mean of 16 steps ÷ stand-still: E8 1.016 ± 0.010 (pos.), 1.008 ± 0.006 (rot.); E8-trim 1.001 ± 0.001, 1.006 ± 0.002.
  The difference is not significant with 3 seeds (Welch t ≈ 2.4), but the seed spread is 10× smaller: every E8-trim seed
  is the "offset ≈ 0" predictor.
- E8 was 0.998 on untrimmed val and 1.016 on trimmed val. Hypothesis (not checked): it learned a small drift towards the
  tool-withdrawal motion of the tail, which helps on tail pairs and hurts elsewhere.
- Training dynamics of E8-trim: best epoch 1 in all seeds; train 0.76 → 0.28 while val rises 0.78 → 1.11–1.19 (same
  pattern as E8).
- World model on the same trimmed pairs: next-image MSE E8 0.00306 ± 0.00021, E8-trim 0.00314 ± 0.00003 (same). Both are
  epoch-1 weights (checkpoint chosen on action loss).

---

## 4. Losses on the same target (only comparable within a row group)

**Actor, absolute target a(t) (z-score units):**

| Run | Val action Huber (best) | Best epoch / epochs run |
| --- | --- | --- |
| E1 | 0.0027–0.0031 | 45–100 |
| E6 | 0.0026–0.0035 | 74–100 |
| E7 | 0.032–0.041 | 1 / 11 |

**World model, state head:**

| Run | Metric | Model | Baseline |
| --- | --- | --- | --- |
| E1 | next-state MSE (normalised) | 0.0018–0.0029 | copy_state ≈ 0.00016–0.0002 |
| E7 | Huber on f = Δ / scale | 3.07–3.11 | f = 0 ("no change"): 2.955 |

E7 loss balance: state loss ≈ 200× (epoch 1) to 900× (epoch 11) the action loss; in E1 about 1 : 1.

---

## 5. Diagnostics and probes (no image; fitted on train, scored on val)

**D1: "no change" (f = 0) on each split, E7 target**

| Split | f = 0 | E7 head, epoch 1 | E7 head, epoch 11 |
| --- | --- | --- | --- |
| train | 3.59 | 3.49 | 2.10 (below f = 0) |
| val | 2.955 | 3.1 | 4.0 (above f = 0) |

→ memorisation, not a train/val difference.

**D2: is the one-frame Δstate predictable? (target f = Δs / scale, 8-D; val Huber ÷ f = 0, below 1 = better)**

| Inputs | Ridge | MLP (2 × 256) |
| --- | --- | --- |
| s, a | 1.13 | 1.23 (memorises) |
| s, a + a(t) − a(t−1) | 1.08 | **0.63** |
| s, a + f(t−1) (last state change) | **0.44** | **0.42** |
| constant velocity f(t−1), no learning | 0.50 | – |

**Chunk probe: are the 16 offsets predictable? (E8 target, option C units; 50 train episodes held out)**

| Inputs | Probe | Train | Held-out train | Val | vs stand-still (0.670) |
| --- | --- | --- | --- | --- | --- |
| s(t) | ridge | 0.673 | 0.652 | 0.670 | equal |
| s(t) | MLP | 0.537 | 0.831 | 0.852 | worse (memorises) |
| a(t−1) − a(t−2) | ridge | 0.491 | 0.493 | **0.502** | −25% |
| s(t) + a(t−1) − a(t−2) | MLP | 0.399 | 0.432 | **0.453** | **−33%** |

→ without history every probe fails; with one step of history the target is learnable and generalises.

---

## 6. Where each number comes from

| Numbers | Script |
| --- | --- |
| E1, E2, E6, E7 (mm, degrees; paired test) | `scripts/check_orientation_error.py` → `scripts/compare_runs.py` |
| Stand-still per horizon | `scripts/look_standstill_baseline.py` |
| E7 losses, WM vs f = 0 | `scripts/train.py` (`evaluate`, `metrics.state_delta_metrics`) |
| E8, E9, chunk baselines | `scripts/train.py` (`metrics.chunk_metrics`, `metrics.chunk_baselines`) |
| E8-trim; E8 on trimmed val | `scripts/train.py --keep-ranges`; `scripts/train.py --eval-only --eval-keep-ranges` |
| Keep ranges (tool visible) | `scripts/look_tool_projection.py --all --export-keep-ranges` |
| copy_state | `scripts/baselines.py` |
| D1, D2 | `scripts/look_delta_reference.py`, `scripts/look_delta_predictability.py` |
| Chunk probe | `chunk_probe.py` in the DGX scratchpad (not in the repository yet) |

**Missing:** nothing in table 2. The chunk-probe script still has to be saved into the repository.
