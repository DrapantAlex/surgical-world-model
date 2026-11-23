# Analysis scripts

Inspection, check and figure scripts: they measure, verify or plot something, but never
train a model for the thesis. Run them from the repo root:

```bash
python -m scripts.analysis.<name> [args]
```

Moved here from `scripts/` on 2026-11-19; diary entries before that date use the old paths
(`scripts/<name>.py`). "Supports" = the diary entry (or decision record) that reports the result.

## Checks (`check_*`): does the code or the data do what I think?

| Script | What it does | Supports |
|---|---|---|
| `check_camera_duplicates.py` | Are the left and right endoscope videos identical, per episode? | diary 10-12, 10-15 (left camera only) |
| `check_eval_unchanged.py` | Did a change move any number? `save <label>` re-evaluates runs (`--eval-only`, CPU) before it, `compare <label>` after it, every key exactly (default: E7, E8, 1c) | diary 11-19 (refactor) |
| `check_model.py` | One batch of one episode through the model: shape check, no training | — (sanity tool) |
| `check_orientation_error.py` | Actor error in mm and degrees per run (`check_orientation_error.json`) | decision 10-20; `experiment-comparison.md` §2, §6 (E1, E2, E6, E7) |
| `check_quaternion_sign.py` | Is the action quaternion sign consistent across episodes? | decision 10-20 → `--canonical-quaternion` |
| `check_repeat_batches.py` | `--val-every`: does `.repeat()` give the same batches as plain passes? | diary 11-16 |
| `check_val_every_weights.py` | `--val-every`: weights bitwise identical to a normal run after epoch 1? | diary 11-16 |
| `check_wm_balance_equivalence.py` | (c): the balancing `train_step` vs Keras' on CPU: level 1 (Keras' step in the subclass) bitwise, level 2a (split vs combined gradient at step 0) ≤ 1e-5, level 2b (20 steps through the split path) ≤ 2× a 1-ulp control | — (experiment c, 11-23; diary entry to come) |
| `check_wm_balance_push.py` | (c): for a `--wm-balance` run, the realised push of each balanced term at the best check on fresh train / val batches (mean, median, of means, Σ); band 0.2–0.45 (×ρ/0.3) on train | — (experiment c, 11-23; diary entry to come) |
| `check_wm_off_gradient.py` | E6: with both WM weights at 0, no gradient reaches the encoder from the WM heads | decision 10-20 (E6) |

## Inspections (`look_*`): what do the data or the runs show?

| Script | What it does | Supports |
|---|---|---|
| `look_chunk_baselines.py` | E8: no-learning chunk predictors on val, before training | diary 10-29, E8 decision (not cited by name) |
| `look_delta_predictability.py` | E7 D2: can the one-frame state change be predicted without the image? | diary 10-29; `experiment-comparison.md` §5 |
| `look_delta_reference.py` | E7 D1: reference losses of the Δ head on train and val | diary 10-29 |
| `look_encoder_gradients.py` | Encoder gradients of the actor, state and image losses (norms, weighted ratios, cosines, per layer) on 20 val + 20 train batches, best weights and untrained; checks that the weighted terms sum to the compiled loss's gradient (TF32 off) → `outputs/analysis/encoder_gradients/` | — (WM gradient share, 11-23; diary entry to come) |
| `look_endo360.py` | endo360 checks 1–3: glitches, sample-and-hold, share of the error | diary 11-17 |
| `look_frame_difference.py` | From which gap k does frame t − k clearly differ from frame t? | — (output in `outputs/analysis/frame_difference/`) |
| `look_history_ablation.py` | History inputs: best weights of the `--history` runs evaluated with Δs, Δa or both zeroed (GPU; "full" vs `val_metrics.json` within 1e-4 relative) → `outputs/analysis/history_ablation/` | 11-20 (history experiment: which half of h(t) the actor uses) |
| `look_standstill_baseline.py` | Stand-still error of a(t) vs a(t+h), h = 1..60 | diary 10-23; `experiment-comparison.md` |
| `look_state_delta.py` | E7: size of the one-frame state change per dimension | diary 10-28 |
| `look_state_groups.py` | KUKA / endo360 state curves of finished runs, from `history.csv` | — (commit `6eabc07`, 11-18; for runs before the in-training group logs) |
| `look_static_vs_moving.py` | E1 / E2 / E6 errors split by phase and speed | diary 10-27 |
| `look_tool_projection.py` | Projects the commanded tool position into the image; exports the keep ranges | diary 11-12; `configs/keep_ranges/README.md` |

## Figures (`plot_*`): figures for the notes, from finished runs

| Script | What it does | Supports |
|---|---|---|
| `plot_step1_summary.py` | Step 1 (1a, 1b, 1c) curves from `history.csv`: KUKA ÷ copy, image ÷ copy_frame, actor Huber → `outputs/analysis/step1_summary/` | — (notes, step 1) |

## Shared code

Some scripts import from others; move or rename them together:
`look_state_delta` → `look_chunk_baselines`, `look_delta_reference`, `look_delta_predictability`;
`look_delta_reference` → `look_delta_predictability`;
`look_standstill_baseline` → `look_frame_difference`, `look_static_vs_moving`;
`check_quaternion_sign` → `check_orientation_error` → `look_static_vs_moving`, `scripts/train_state_only.py`.
