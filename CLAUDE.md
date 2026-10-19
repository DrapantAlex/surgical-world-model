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
- After every change, run a smoke run with a new run name, for example
  `python -m scripts.train --run-name smoke-<short-description> --max-episodes 2 --epochs 2 --wandb-mode offline`.
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

The repo has no test suite and no linter. To validate a change, run a `--max-episodes` smoke run. The `scripts/check_*.py` and `scripts/look_*.py` files are throwaway inspection scripts.

## Architecture

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

**Model (`models/simple_world_model/swm.py`, `models/encoders.py`)**
- The encoder is selected by name: `simple_cnn` (the original 4 convs) or a Keras Applications backbone.
- To add a backbone, add one line to `APPLICATIONS`. Backbone-specific preprocessing goes inside the encoder as layers, so the data pipeline stays in [0, 1].
- In backbones, BatchNorm is always frozen and runs in inference mode.
- `latent_dim=512` adds a bottleneck before the image decoder, giving about 72M parameters in total. `latent_dim=None` is the original decoder, with about 2.5B parameters (roughly 40 GB of GPU memory).
- The decoder starts at img/16 and upsamples by 2 four times.

**Metrics (`models/metrics.py`)**
- `scripts/baselines.py` and `scripts/train.py` both use these functions, so baselines and models are always measured the same way.
- The core metrics (`action_mse`, `action_huber`, `next_state_mse`, `next_image_mse`) are in normalised units and exist for every dataset.
- The physical `extra/` metrics (mm, rotation, jaw per arm) exist only for tissue lift, whose 16-D layout `[x y z | quat | jaw] × 2 arms` is known. The check for it is the dataset folder name.

**Training (`scripts/train.py`)**
- Settings that are fixed on purpose live at the top of the file rather than as CLI flags: the loss weights (action 1.0, state 1.0, image 0.1), `MONITOR = val_action_output_loss` for checkpointing and early stopping, and the W&B project.
- Each run writes to `outputs/<run>/`: `config.json` (including git commit, params, and episodes), `norm_stats.json`, `best.weights.h5` (weights only), `history.csv`, and `val_metrics.json`.
- TensorBoard logs go to `logs/<run>/`, and W&B is also enabled.
- Full runs (no `--max-episodes`) append a row to `outputs/results.csv`. Smoke runs do not.
- The script refuses to reuse an existing run name.
- `--eval-only` rebuilds the model from the saved config and stats.
- When changing config keys or metric names, keep older runs loadable. For example, runs without a `dataset_root` key default to tissue lift.

## Conventions

- Code is organised into lettered sections (`# --- A: ... ---`) with short, explanatory comments. The `CHANGED:`/`NEW:` markers in `swm.py` record differences from the reference model.
- `data/`, `outputs/`, `logs/`, `wandb/`, and all weight files are gitignored.
