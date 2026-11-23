# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An actor–world model (TensorFlow/Keras) trained on surgical-robot demonstrations from NVIDIA's Open-H dataset (LeRobot v2.1 format). One network has two branches that share an image encoder:
- **Actor**: (frame t, state t, language) → action t (imitation learning).
- **World model**: (frame t, state t, language, action t) → state t+1 and frame t+1.

The model was adapted from the professor's Open-X script in `reference/simple_world_model.py`. Leave that file as it is; the working version is `models/simple_world_model/swm.py`. `docs/research-diary.md` records the dated findings and reasoning behind the design choices.

## Thesis context
This repo is for my M.Sc. thesis "Physical Deep Learning for Surgical Robotics" (RICAM Linz / PoliTo).
The final product is encoder + actor: at inference time only they are used. The world model is an auxiliary task during training that shapes the shared encoder.
Current phase: training on STAR-IL (single-arm). Next: ablation studies on visual encoders (Keras Applications) and model variants, then scaling to more arms and cameras.

## How to work with me
- I'm a student and I must be able to explain and defend every line of this code. Explain the concept and your plan before changing code.
- Make small changes, one at a time, and explain them section by section.
- For design choices, present alternatives with pros and cons and a recommendation, then wait for my decision.
- Ask me when information is missing instead of making silent assumptions.
- Never launch long training runs. Prepare the exact command and let me run it in tmux.
- Don't load full datasets to inspect data: use `--max-episodes` or a few samples.
- After every change, run a smoke run with a new run name. Use the full-option STAR-IL command, so the chunk, skip, mixer, keep-ranges, `--val-every` and canonical-quaternion paths are all exercised:
  ```bash
  python -m scripts.train --run-name smoke-<short-description> \
      --dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il --max-episodes 2 --epochs 2 \
      --wandb-mode offline --canonical-quaternion --chunk 16 --state-skip --wm-mixer-blocks 1 \
      --keep-ranges configs/keep_ranges/star_il_tool_visible.csv --val-every 50
  ```
- If a code change must not change any number (a refactor, a move, a fix elsewhere), also run `python -m scripts.analysis.check_eval_unchanged compare pre-refactor`: it re-evaluates the E7, E8 and 1c runs on CPU and compares every key exactly against the saved pre-refactor values (exit 1 if anything differs). After a change that is meant to move numbers, save a new baseline with `save <label>` and compare later changes against that one.
- Never touch the test split and never modify `reference/simple_world_model.py`.

## Environment and commands

On the DGX, set up the shell with `source bash_scripts/env_dgx.sh [GPU]`. It activates `.venv`, turns on TF memory growth, and picks the least-used free GPU (or the one you pass). If no GPU is free it falls back to CPU. Dependencies are in `requirements.txt`.

Run every script as a module from the repo root, because imports are `from models import ...`:

```bash
python -m scripts.download_dataset                     # tissue lift (left endoscope only) -> data/open_h
python -m scripts.download_star_il [--full]            # STAR-IL: episode 0, or the full sub-dataset
python -m scripts.baselines [--max-episodes 2]         # trivial predictors on val; no TF/GPU needed; run before training
python -m scripts.train --run-name smoke --max-episodes 2 --epochs 2 --wandb-mode offline   # smoke test
python -m scripts.train --run-name smoke --eval-only   # re-evaluate saved best weights
python -m scripts.train --dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il
python -m scripts.train_overfit                        # overfit one episode (pipeline sanity check)
```

The repo has no test suite and no linter. To validate a change, run the full-option smoke run above and, for changes that must not move any number, `check_eval_unchanged compare pre-refactor`. The `scripts/analysis/check_*.py` and `scripts/analysis/look_*.py` files are inspection scripts (run as `python -m scripts.analysis.<name>`), listed with their diary entries in `scripts/analysis/README.md`.

## Architecture

`models/` holds the library code, one module per concern; `scripts/` holds the entry points. `README.md` has the folder tree.

**Data (`models/data.py`)**
- The default dataset is `srt_tissue_lift`. Every entry point accepts `--dataset-root`, and the code must stay dataset-agnostic: read paths, splits (`"train": "0:180"`), and state/action dims from `meta/info.json` and `episodes.jsonl`. Never hard-code them.
- Only the left endoscope camera is used, because the right video is an exact copy.
- Episodes without a video for that camera are skipped and recorded as `skipped_episodes`.
- `load_split` returns flat arrays for a whole split:
  - frames as uint8;
  - normalised states and actions;
  - `episode_index`;
  - `pair_index`, the positions t whose t+1 is in the same episode (pairs never cross episode boundaries).
- Frames for a full split come from a memory-mapped cache, `data/cache/<dataset>/<split>_<camera>_224.npy`. It is built atomically on first use and rebuilt whenever its JSON manifest stops matching the metadata. Subsets (`--max-episodes`) decode the videos directly and bypass the cache.
- **Only `train` and `val` are allowed** (`ALLOWED_SPLITS`). The test split is deliberately never loaded.
- **Normalisation stats come from the train split only**, computed by `compute_norm_stats`, which raises on missing files and never falls back. They are saved with each run as `norm_stats.json`.
- `get_pairs` is the single place where indices become model inputs and targets, with images converted to float in [0, 1] per batch. The overfit test, the baselines, and training all use it.
- `language_input` is a zero vector of size 512, since there is only one task per dataset.

**Pipeline (`models/pipeline.py`)**: `make_dataset` wraps `get_pairs` in `tf.numpy_function` over shuffled, batched `pair_index`.

**Model (`models/simple_world_model/swm.py`, `models/encoders.py`, `models/state_only.py`)**
- The encoder is selected by name: `simple_cnn` (the original 4 convs) or a Keras Applications backbone.
- To add a backbone, add one line to `APPLICATIONS`. Backbone-specific preprocessing goes inside the encoder as layers, so the data pipeline stays in [0, 1].
- In backbones, BatchNorm is always frozen and runs in inference mode.
- `latent_dim=512` adds a bottleneck before the image decoder, giving about 72M parameters in total. `latent_dim=None` is the original decoder, with about 2.5B parameters (roughly 40 GB of GPU memory).
- The decoder starts at img/16 and upsamples by 2 four times.
- The experiment variants are arguments of `build_actor_world_model`, each marked `NEW (<experiment>):` in the code: `delta_factor` (E7), `action_chunk` (E8), `state_skip` (1a), `state_layers` (1b), `wm_mixer_blocks` (1c). Their defaults give the original model.
- `models/state_only.py` is the state-only actor of E2 (no image), trained by `scripts/train_state_only.py`; with `--chunk`, `--history`, `--keep-ranges` and `--val-every` it is experiment B.
- `models/balance.py` is experiment (c), `--wm-balance RHO`: `build_model` re-wraps the same graph as `BalancedWorldModel`, whose `train_step` divides the WM losses by train copy references and adapts their weights so each term's push on the encoder stays at RHO × the actor's. compile, `test_step` and the weight file are unchanged; the controller state is in untracked `tf.Variable`s.

**Metrics and reference lines (`models/metrics.py`, `models/references.py`)**
- `scripts/baselines.py` and `scripts/train.py` both use the metrics functions, so baselines and models are always measured the same way.
- The core metrics (`action_mse`, `action_huber`, `next_state_mse`, `next_image_mse`) are in normalised units and exist for every dataset.
- The physical `extra/` metrics (mm, rotation, jaw per arm) exist only for tissue lift, whose 16-D layout `[x y z | quat | jaw] × 2 arms` is known. The check for it is the dataset folder name.
- The no-learning reference lines used during training and evaluation are in `models/references.py`: copy_state (per dim), copy_frame, constant velocity, stand-still and train-mean Huber, and the glitch rule (`GLITCH_G`) with the per-group state report.

**Evaluation (`models/evaluation.py`)**: the final evaluation of the best weights (`evaluate`), the E8 chunk baselines, and the comparison table next to the `scripts/baselines.py` results, joined only when the dataset, quaternion convention and keep ranges match.

**Rebuilding a run (`models/run_config.py`)**
- `build_model(cfg, stats)` builds and compiles the model from a run's saved `config.json` and `norm_stats.json`. Training, `--eval-only`, and the analysis scripts that reload saved weights all use it; pass the run's own `stats`.
- The config readers there (`dataset_root`, `loss_weights`, `delta_factor`, `run_quat_ref`, `run_keep_ranges`, …) give older runs the defaults they were trained with. `dataset_root` also turns an absolute path inside the repo into the repo-relative one.
- `OUTPUTS` (the `outputs/` folder) lives here too, with `git_commit` and `save_json`, which write a run's records.

**Callbacks and visualisation (`models/callbacks.py`, `models/visualization.py`)**
- All Keras callbacks are in `models/callbacks.py`: the ones that add values to the logs or send them to W&B, `CheckMonitor` (it gets `MONITOR` from `scripts/train.py` as an argument) and `timeline_callback`.
- `models/visualization.py` draws the GT/PRED timeline for TensorBoard and W&B.

**Training (`scripts/train.py`)**
- It holds the command line, `train()`, `eval_only()` and two small helpers (`checks`, `log_tensorboard`). Everything else it imports from `models/`.
- Settings that are fixed on purpose are constants rather than CLI flags:
  - the loss weights (action 1.0, state 1.0, image 0.1) are `LOSS_WEIGHTS` in `models/run_config.py`, because rebuilding a run needs them;
  - `MONITOR = val_action_output_loss` (checkpointing and early stopping) and the W&B project are at the top of `scripts/train.py`.
- Each run writes to `outputs/<run>/`: `config.json` (including git commit, params, and episodes), `norm_stats.json`, `best.weights.h5` (weights only), `history.csv`, and `val_metrics.json`.
- TensorBoard logs go to `logs/<run>/`, and W&B is also enabled.
- Full runs (no `--max-episodes`) append a row to `outputs/results.csv`. Smoke runs do not.
- The script refuses to reuse an existing run name.
- `--eval-only` rebuilds the model from the saved config and stats.
- When changing config keys or metric names, keep older runs loadable. For example, runs without a `dataset_root` key default to tissue lift (`dataset_root` in `models/run_config.py`).

**Other scripts**
- `scripts/baselines.py` (trivial predictors), `scripts/train_state_only.py` (E2), `scripts/train_overfit.py` (overfit one episode) and `scripts/compare_runs.py` (seed groups, bootstrap CIs, the paired test).
- `scripts/analysis/`: checks (`check_*`), inspections (`look_*`) and figures (`plot_*`). They never train a thesis model. Each is listed with the diary entry it supports in `scripts/analysis/README.md`.

## Conventions

- Code is organised into lettered sections (`# --- A: ... ---`) with short, explanatory comments. The `CHANGED:`/`NEW:` markers in `swm.py` record differences from the reference model.
- `data/`, `outputs/`, `logs/`, `wandb/`, and all weight files are gitignored.
