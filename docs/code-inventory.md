# Code inventory (2026-11-19, commit `2089c8a`)

Read-only survey before any refactoring: nothing was changed, moved or deleted.
Method: `git log` per file, a grep of `docs/` and `CLAUDE.md` for each file name, every
internal `from models/scripts ... import`, and every `outputs/*/config.json` (42 runs, 30 of
them non-smoke) for flag usage. "Not run" means I did not run the script.

Categories: **core** (training / evaluation) · **option** (experiment option still in use) ·
**reload** (needed only to reload old runs) · **evidence** (analysis cited in the diary or in
`experiment-comparison.md`) · **obsolete**.

---

## 1. Files

### models/

| File | Lines | What it does | Category | Cited in | Imported by |
|---|---|---|---|---|---|
| `data.py` | 571 | Loads LeRobot episodes; splits, frame cache, train-only norm stats, quaternion canonicalisation, keep ranges, chunk targets, `get_pairs` | core | CLAUDE.md; whole diary | `pipeline`, `metrics`, `train`, `baselines`, `train_overfit`, and almost every `check_*`/`look_*` |
| `encoders.py` | 61 | `simple_cnn` + Keras Applications backbones by name | core (backbones: next phase) | CLAUDE.md; diary 10-14, 10-15 | `swm.py`, `train.py` |
| `metrics.py` | 199 | Shared val metrics (core, `extra/`, `per_dim/`, chunk, Δ-state), `ImageMSE`, `results.csv` | core | CLAUDE.md; `decisions/2026-10-20` | `train`, `baselines`, `check_orientation_error`, `train_state_only`, several `look_*` |
| `pipeline.py` | 48 | `make_dataset`: `tf.data` over `pair_index` → `get_pairs` | core | CLAUDE.md; diary 10-16, 11-16 | `train`, `check_orientation_error` |
| `simple_world_model/swm.py` | 135 | The actor–world model: E7 delta head, E8 chunk, 1a skip, 1b `state_layers`, 1c mixer | core | CLAUDE.md; diary | `train`, `train_overfit`, `check_model` |
| `state_only.py` | 20 | E2 state-only MLP actor | evidence (E2) | `experiment-comparison.md` §1–2 (as "E2") | `train_state_only`, `look_static_vs_moving` |
| `visualization.py` | 74 | `WorldModelTimelineCallback`: GT/PRED timeline (TensorBoard + W&B) | core | — | `train`, `train_overfit` |
| `__init__.py` ×2 | 0 | package markers | core | — | — |

### scripts/: entry points

| File | Lines | What it does | Category | Cited in | Imported by |
|---|---|---|---|---|---|
| `train.py` | 796 | Training, `--eval-only`, reference lines, final evaluation, `results.csv` | core | CLAUDE.md; whole diary | `check_orientation_error`, `check_wm_off_gradient`, `check_val_every_weights`, `compare_runs`, `look_state_groups`, `look_static_vs_moving`, `train_state_only` |
| `baselines.py` | 129 | Trivial predictors on val (same metrics as training) | core | CLAUDE.md; `decisions/2026-10-20` | — |
| `train_overfit.py` | 98 | Overfit one episode (pipeline sanity check) | core (sanity) | CLAUDE.md; diary 10-13 | — |
| `download_dataset.py` | 14 | Downloads tissue lift (left camera only) | core (data) | CLAUDE.md | — |
| `download_star_il.py` | 32 | Downloads STAR-IL: one episode or `--full` | core (data) | CLAUDE.md | — |
| `download_episode.py` | 14 | Downloads tissue-lift episode 0 (all cameras) | obsolete (superseded by the two above) | — | — |
| `train_state_only.py` | 153 | E2: trains the state-only actor | evidence (E2) | `experiment-comparison.md` (E2 rows) | — |
| `compare_runs.py` | 240 | Groups of seeds compared in mm/deg, paired episode bootstrap | evidence (E1/E2/E6/E7) | `decisions/2026-10-20`; `experiment-comparison.md` §6 | — |
| `plot_step1_summary.py` | 134 | The three step-1 figures (untracked, made today) | evidence (step 1) | — (not yet) | — |

### scripts/: checks (`check_*`)

| File | Lines | What it does | Category | Cited in | Imported by |
|---|---|---|---|---|---|
| `check_camera_duplicates.py` | 157 | Are the left and right videos identical, per episode? | evidence (the "left only" choice) | diary 10-12, 10-15 (the result, not the file name) | — |
| `check_data.py` | 4 | Prints the shapes of `load_episode_triplets(0)` | obsolete (tissue lift only) | — | — |
| `check_data_full.py` | 23 | Frames/episodes/pairs/RAM of train + val (tissue lift `ROOT`) | obsolete (tissue lift only, before the cache) | — | — |
| `check_model.py` | 27 | One batch through the model (shape check) | core (sanity), small | — | — |
| `check_orientation_error.py` | 182 | Actor error in degrees + `predict_actions`, `check_run` | evidence (E1/E2/E6) **and library** | `decisions/2026-10-20`; `experiment-comparison.md` §6 | `train_state_only`, `look_static_vs_moving` |
| `check_quaternion_sign.py` | 130 | Is the quaternion sign consistent? (led to `--canonical-quaternion`) | evidence + library | `decisions/2026-10-20` (×6) | `check_orientation_error` |
| `check_repeat_batches.py` | 46 | `--val-every`: `.repeat()` gives the same batches | evidence (11-16) | diary 11-16 | — |
| `check_val_every_weights.py` | 92 | `--val-every`: bitwise-identical weights in epoch 1 | evidence (11-16) | diary 11-16; comment in `train.py` | — |
| `check_wm_off_gradient.py` | 79 | E6: zero WM weights → no gradient from the WM heads | evidence (E6) | `decisions/2026-10-20` | — |

### scripts/: inspections (`look_*`)

| File | Lines | What it does | Category | Cited in | Imported by |
|---|---|---|---|---|---|
| `look_action_dimension.py` | 10 | Prints the state/action names of an `info.json` | obsolete (one-off) | — | — |
| `look_chunk_baselines.py` | 105 | E8: chunk baselines before training | evidence (E8), now inside `train.py` | — | — |
| `look_dataset_size.py` | 25 | Hugging Face sizes of tissue lift | obsolete (one-off) | — | — |
| `look_delta_predictability.py` | 148 | E7 D2: is Δ predictable without the image? | evidence (E7) | diary 10-29; `experiment-comparison.md` | — |
| `look_delta_reference.py` | 66 | E7 D1: reference losses of the Δ head | evidence (E7) + library | diary 10-29 | `look_delta_predictability` |
| `look_endo360.py` | 240 | endo360 checks 1–3 | evidence (11-17) | diary 11-17 | — |
| `look_episode_one_arm.py` | 5 | Columns of STAR-IL episode 0 | obsolete (one-off) | — | — |
| `look_frame_difference.py` | 198 | From which gap k does frame t − k differ from t? | evidence? (not cited by name) | — (output in `outputs/analysis/frame_difference`) | — |
| `look_kinematics.py` | 14 | Columns of tissue-lift episode 0 (polars) | obsolete (one-off) | — | — |
| `look_parquet.py` | 40 | Parquet schema of tissue-lift episode 0 | obsolete (one-off) | — | — |
| `look_standstill_baseline.py` | 173 | Stand-still error per horizon h = 1..60 | evidence (10-23) + library | diary 10-23; `experiment-comparison.md` | `look_frame_difference`, `look_static_vs_moving` |
| `look_state_delta.py` | 64 | E7: size of Δ per dimension; `STAR_IL` constant | evidence (E7) + library | diary 10-28 | `look_delta_reference`, `look_delta_predictability`, `look_chunk_baselines` |
| `look_state_groups.py` | 83 | Per-group state curves of old runs from `history.csv` | evidence (11-18), superseded for new runs by the in-training group logs | — | — |
| `look_static_vs_moving.py` | 199 | Errors split by phase and speed (E1/E2/E6) | evidence (10-27) | diary 10-27 | — |
| `look_tool_projection.py` | 596 | Tool projection → `star_il_tool_visible.csv` (keep ranges) | evidence (11-12); **generates a config** | diary 11-12; `experiment-comparison.md` | — |
| `look_video.py` | 36 | Decodes one tissue-lift video to PNGs | obsolete (one-off) | — | — |

### bash_scripts/, configs/

| File | What it does | Category | Cited in | Used by |
|---|---|---|---|---|
| `bash_scripts/env_dgx.sh` | venv, TF memory growth, least-used GPU | core | CLAUDE.md | you, in tmux |
| `configs/keep_ranges/README.md` | Format of the keep-ranges files | core (docs) | diary 11-12 | — |
| `configs/keep_ranges/star_il_tool_visible.csv` | Kept frames per episode (tool visible) | core: 11 runs train on it, hash-checked on reload | diary 11-12 → 11-17 | `--keep-ranges`; copied into each run |

**Hidden library dependencies.** Seven scripts import from `scripts/train.py`, and four `look_*`/`check_*`
files are imported by other scripts (`look_state_delta.STAR_IL`, `look_standstill_baseline`,
`check_quaternion_sign`, `check_orientation_error`). Moving or deleting any of them breaks the importers.

---

## 2. Flags of `scripts/train.py`

"Used" = in the 30 non-smoke runs in `outputs/`. "Reload" = `build_model` / `eval_only` read the key
from `config.json`, so old runs need the code path behind the flag to stay.

| Flag | What it does | Added by | Used | Reload needs it? |
|---|---|---|---|---|
| `--run-name` | Output folder name (refuses to reuse one) | base | all | — |
| `--dataset-root` | LeRobot folder (default tissue lift) | STAR-IL (10-16) | all (STAR-IL) | yes (`dataset_root()`, default tissue lift) |
| `--epochs` | max epochs | base | 100 everywhere | no |
| `--batch-size` | | base | 32 everywhere | yes (eval batch size) |
| `--lr` | | base | 3e-4 everywhere | compile only |
| `--seed` | seeds + 1b/1c extra-layer initialisers | base | 0/1/2 | yes (`init_seed`, harmless) |
| `--patience` | early stopping, epochs | base | 10 everywhere | no |
| `--latent-dim` | 512 or `none` | base | 512 everywhere | yes |
| `--encoder` | `simple_cnn` or a backbone | base | `simple_cnn` only | yes; backbones are the next phase |
| `--freeze-encoder`, `--unfreeze-last-n` | backbone freezing | base | never | yes (key read); next phase |
| `--max-episodes` | smoke subset | base | smoke runs only | yes (`eval_only` uses `cfg["max_episodes"]`) |
| `--canonical-quaternion` | flip quaternion labels to the train-mean sign | D1 / E1 (10-20) | 29 of 30 | **yes** (quaternion reference) |
| `--w-next-state`, `--w-next-image` | WM loss weights (0 = WM off) | E6 (10-21) | 1.0/0.1, or 0/0 in 6 runs (E6, E9) | compile only (default if absent) |
| `--state-delta` | E7 Δ head + scaled Huber | E7 (10-28) | 3 runs (E7) | **yes** (different architecture + loss) |
| `--state-skip` | 1a: s + zero-initialised Dense | 1a (11-13) | 8 runs (1a/1b/1c) | **yes** |
| `--state-layers` | 1b: K hidden layers | 1b (11-16) | 2 runs (K = 2, 3) | **yes** |
| `--wm-mixer-blocks` | 1c: ConvMixer blocks in the WM branch | 1c (11-18) | 2 runs (D = 1, 3) | **yes** |
| `--chunk` | E8: k offsets from a(t−1) | E8 (10-29) | 17 runs | **yes** (output shape, scale, metrics) |
| `--keep-ranges` | train/eval only on kept frames | B1 trimming (11-12) | 11 runs | **yes** (hash-checked copy) |
| `--eval-keep-ranges` | `--eval-only` on another keep-ranges file | 11-12 | 3 E8 runs re-evaluated | no (evaluation only) |
| `--vis-every` | timeline every N epochs | base | 5 everywhere | no |
| `--val-every` | one val check every N batches | 11-16 | 5 runs (1a-val500, 1b, 1c) | no for loading; `best_batch` in `results.csv` |
| `--wandb-mode` | online/offline/disabled | base | online | no |
| `--deterministic` | deterministic TF ops | base | **never** | no |
| `--eval-only` | re-evaluate the saved best weights | base | 9 re-evals | — |

Other things old runs need to stay loadable: `run_quat_ref`, `run_keep_ranges`, `dataset_root`,
`loss_weights`, `delta_factor` (+ `scaled_huber`), and the `cfg.get(..., default)` calls in
`build_model`. The 3 state-only runs are not loaded by `train.py`; they need
`train_state_only.py` + `models/state_only.py`.

---

## 3. Map of `scripts/train.py` (796 lines)

| Lines | Section | What it does |
|---|---|---|
| 1–23 | docstring | usage examples, one line per option |
| 24–42 | imports | |
| 44–56 | **0: fixed choices** | `LOSS_WEIGHTS`, `MONITOR`, `WANDB_PROJECT`, `DELTA_FLOOR_C`, `CHUNK_FLOOR_C`, `GLITCH_G`, paths |
| 59–140 | **A: command line** | `parse_args`: 27 flags + 10 consistency checks (keep-ranges validated before the run folder exists) |
| 143–201 | B: config / backward-compat helpers | `freeze_label`, `dataset_root`, `run_quat_ref`, `run_keep_ranges`, `baselines_json`, `shuffle_order` (**never called**), `episodes_for`, `loss_weights`, `delta_factor`, `scaled_huber` |
| 203–250 | B: model + metrics | `metric_name`, `state_dim_metrics`, `build_model` (compile with losses, weights, metrics) |
| 252–357 | B: reference lines | `copy_state_mse(_per_dim)`, `copy_frame_mse`, `prev_in_window`, `constvel_state(_mse_per_dim)`, `glitch_thresholds`, `glitch_frames`, `state_group_metrics`, `trainmean_huber`, `standstill_huber` |
| 359–361 | B: schedule | `checks` (epochs → `--val-every` checks) |
| 363–421 | B: final evaluation | `evaluate`, `chunk_baseline_metrics`, `comparison_table` |
| 423–441 | B: utilities | `git_commit`, `save_json`, `log_tensorboard` |
| 443–518 | **C: callbacks** | `EpochTimer`, `TrainingProgress`, `CheckMonitor`, `ConstantLogs`, `StateGroupLogs`, `WandbLogger`, `timeline_callback` |
| 521–680 | **D: training** | run folder → seeds → keep ranges, quaternion reference, stats → splits → schedule (`--val-every`) → datasets → model → reference lines → 1a copy assertion → `config.json` → W&B → callbacks → `fit` |
| 681–726 | E: final evaluation (inside `train`) | best weights → `evaluate` → JSON, TensorBoard, W&B summary → comparison table → `results.csv` |
| 729–788 | **F: `eval_only`** | rebuild from config + stats; glitch thresholds from the run's train tables; optional `--eval-keep-ranges`; group ratios printed |
| 791–796 | main | |

Section B alone is about 300 lines and holds four different kinds of code.

---

## 4. Proposal

**safe** = no behaviour change, checkable by a smoke run plus a reload of an old run.
**decide** = needs your decision.

### 4a. Keep as they are
- **safe:** everything in `models/`, `train.py` (until 4d), `baselines.py`, `train_overfit.py`,
  `download_dataset.py`, `download_star_il.py`, `env_dgx.sh`, `configs/`.
- **safe:** every train flag that a saved run uses (table 2). Removing any of them makes those runs non-reloadable.

### 4b. Move to `scripts/analysis/` (evidence, cited)
`check_camera_duplicates`, `check_orientation_error`, `check_quaternion_sign`, `check_repeat_batches`,
`check_val_every_weights`, `check_wm_off_gradient`, `compare_runs`, `train_state_only`,
`look_chunk_baselines`, `look_delta_predictability`, `look_delta_reference`, `look_endo360`,
`look_frame_difference`, `look_standstill_baseline`, `look_state_delta`, `look_state_groups`,
`look_static_vs_moving`, `look_tool_projection`, `plot_step1_summary`.

- **decide:** this changes their commands (`python -m scripts.analysis.look_endo360`) and their
  imports. Four of them are imported by others (see the end of §1), so the move must update those
  imports in the same commit. The diary cites the old paths, so either leave the diary as a dated
  record or add a one-line redirect note. My recommendation: move them, and add one line at the
  top of the diary's next entry. Don't rewrite old entries.
- **decide:** `look_tool_projection.py` generates `configs/keep_ranges/star_il_tool_visible.csv`,
  so it is closer to a data tool than to a one-off. Alternatives: `scripts/analysis/`
  (simplest) or `scripts/data/`. Recommendation: `scripts/analysis/`, with a note in `configs/keep_ranges/README.md`.
- **decide:** `models/state_only.py` is only E2 evidence. Keep it in `models/` (it is a model)
  or move it next to `train_state_only.py`. Recommendation: keep it in `models/`.
- **safe:** `check_model.py` stays in `scripts/` as a sanity tool.

### 4c. Candidates for deletion
All of these are in git history, so a deletion is recoverable with `git show <commit>:<path>`.

| File | What we would lose | Mark |
|---|---|---|
| `download_episode.py` | the "episode 0, all cameras" download (superseded) | safe |
| `check_data.py` | a 4-line shape print, tissue lift only | safe |
| `look_action_dimension.py`, `look_episode_one_arm.py`, `look_kinematics.py`, `look_parquet.py`, `look_video.py`, `look_dataset_size.py` | the first-week data exploration (10-09 → 10-16). Diary entries describe the results but don't cite the files | **decide** (they document how you first read the data; maybe keep under `scripts/analysis/exploration/`) |
| `check_data_full.py` | a tissue-lift RAM/pairs count from before the frame cache | **decide** (not run; probably still works on tissue lift) |
| `train.py: shuffle_order()` | nothing (never called; the value is written inline in `cfg`) | safe |
| `--deterministic` | the option of deterministic TF ops (never used) | **decide** (cheap to keep; useful for a seed-variance study) |

### 4d. Splitting `train.py`
Goal: `train.py` keeps only the command line, the training run and `--eval-only`. The pieces
other scripts import move to `models/`, so analysis scripts stop importing a training script.

| New module | Comes from (lines) | Content |
|---|---|---|
| `models/run_config.py` | 44–56 (part), 144–201, 224–250 | `LOSS_WEIGHTS`, `OUTPUTS`, the backward-compat readers (`dataset_root`, `run_quat_ref`, `run_keep_ranges`, `loss_weights`, `delta_factor`, `freeze_label`, `episodes_for`), `scaled_huber`, `metric_name`, `state_dim_metrics`, `build_model`: "how to rebuild a run from its folder" |
| `models/references.py` | 252–357 | copy_state, copy_frame, constant velocity, glitches, state groups, stand-still, train-mean: every no-learning reference line (could also go into `metrics.py`, which is already "shared by baselines and training") |
| `models/evaluation.py` | 363–421 | `evaluate`, `chunk_baseline_metrics`, `comparison_table`, `baselines_json` |
| `models/callbacks.py` | 443–518 | the six callbacks + `timeline_callback` (+ `visualization.py`'s callback could join them) |
| `scripts/train.py` | 59–140, 359–361, 423–441, 521–796 | `parse_args`, `checks`, `git_commit`/`save_json`/`log_tensorboard`, `train`, `eval_only`, main: about 400 lines |

- **decide:** the split itself (one module per row, or fewer, e.g. merge `references` into `metrics.py`).
- **decide:** the order. My recommendation is one module per commit, each followed by:
  1. a smoke run with a new name,
  2. `--eval-only` on one old run of each kind (E7 delta, E8 chunk untrimmed, trimmed 1c) and a check that
     `val_metrics_reeval.json` matches the saved one (`largest difference` printed by `eval_only`),
  3. a run of the seven importers listed in §1 (or at least `python -c "import ..."`).
- **safe** (if done as re-exports): `train.py` can keep `from models.run_config import OUTPUTS, build_model, ...`
  for a while, so the seven importers keep working until they are updated.
- **decide:** `MONITOR`, `WANDB_PROJECT` and the floor constants stay at the top of `train.py`
  ("fixed on purpose", CLAUDE.md), or move with the code that uses them. Recommendation: `MONITOR` and
  `WANDB_PROJECT` stay in `train.py`; `LOSS_WEIGHTS` moves to `run_config.py`, because reloading needs it.

### 4e. Not part of the code, noticed in passing
- `docs/` has no index of the analysis outputs; `outputs/analysis/` has 12 folders. A short table
  (folder → script → diary date) would help. **decide**
- The 1a and 1b runs don't have the KUKA/endo360 group columns. That is not a code issue, just a reason
  `look_state_groups.py` still matters.
