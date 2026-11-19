# Surgical world model

An actor–world model in TensorFlow/Keras, trained on surgical-robot demonstrations from
NVIDIA's [Open-H dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
(LeRobot v2.1 format). It is part of the M.Sc. thesis "Physical Deep Learning for Surgical
Robotics" (RICAM Linz / PoliTo).

One network has two branches that share an image encoder:

- **Actor**: (frame t, state t, language) → action t, learned by imitation.
- **World model**: (frame t, state t, language, action t) → state t+1 and frame t+1.

The final product is the encoder plus the actor; only they are used at inference time. The
world model is an auxiliary task during training that shapes the shared encoder.

The model is adapted from an Open-X script, kept unchanged in `reference/simple_world_model.py`.
The dated findings and the reasoning behind each design choice are in
[`docs/research-diary.md`](docs/research-diary.md).

## Setup

Python 3.12. Dependencies are listed in `requirements.txt` (not pinned).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the DGX, start every shell with:

```bash
source bash_scripts/env_dgx.sh        # activates .venv, picks the least-used free GPU (or CPU)
source bash_scripts/env_dgx.sh 3      # or force GPU 3
```

## Running

Run every script as a module from the repo root (imports are `from models import ...`):

```bash
# data -> data/open_h
python -m scripts.download_dataset                 # tissue lift, left endoscope only
python -m scripts.download_star_il [--full]        # STAR-IL: episode 0, or the full sub-dataset

# trivial predictors on val (no TF/GPU needed); run before training
python -m scripts.baselines --dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il

# smoke test (2 episodes, 2 epochs, W&B offline)
python -m scripts.train --run-name smoke-test --max-episodes 2 --epochs 2 --wandb-mode offline

# a full run on STAR-IL, then re-evaluate its saved best weights
python -m scripts.train --run-name <run> --dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il
python -m scripts.train --run-name <run> --eval-only

# pipeline sanity check: overfit one episode
python -m scripts.train_overfit
```

`python -m scripts.train --help` lists every option, for example `--encoder`,
`--canonical-quaternion` and the model variants of the experiments. The docstring at the top
of `scripts/train.py` shows an example command for each variant (`--state-delta`,
`--state-skip`, `--state-layers`, `--wm-mixer-blocks`, `--chunk`, `--keep-ranges`,
`--val-every`).

Each run writes to `outputs/<run>/`:
- `config.json`: every option, the git commit, the episodes and the reference values;
- `norm_stats.json`: statistics from the train split only;
- `best.weights.h5`;
- `history.csv`;
- `val_metrics.json`.

TensorBoard logs go to `logs/<run>/`. Full runs also append a row to
`outputs/results.csv`. Only the `train` and `val` splits are ever loaded; the test split
is never used.

## Repository structure

```
models/                      library code (imported, never run directly)
  data.py                    Open-H episodes -> splits, normalisation, frame cache, get_pairs
  pipeline.py                tf.data batches from a split (make_dataset)
  encoders.py                image encoders by name: simple_cnn or a Keras Applications backbone
  simple_world_model/swm.py  the actor-world model (CHANGED:/NEW: = differences from the reference)
  state_only.py              state-only actor (E2), no image
  metrics.py                 validation metrics shared by baselines and training
  references.py              no-learning reference lines (copy_state, copy_frame, constant
                             velocity, stand-still, train-mean) and the glitch rule
  evaluation.py              final evaluation of the best weights and the comparison table
  run_config.py              a run folder: rebuild a run from config.json (build_model, readers
                             with defaults for older runs) and write its records
  callbacks.py               Keras callbacks: epoch time, x-axis, reference lines, W&B
  visualization.py           GT/PRED timeline for TensorBoard and W&B

scripts/                     entry points (python -m scripts.<name>)
  download_dataset.py, download_star_il.py
  baselines.py               trivial predictors, measured like the model
  train.py                   training, final evaluation, --eval-only
  train_state_only.py        E2: state-only actor
  train_overfit.py           overfit one episode
  compare_runs.py            groups of runs (seeds), bootstrap CIs, paired test
  analysis/                  checks (check_*), inspections (look_*) and figures (plot_*);
                             one line each, with the diary entry it supports, in its README

configs/keep_ranges/         which frames of each episode are kept (tool visible), see its README
docs/
  research-diary.md          dated findings and decisions
  experiment-comparison.md   the experiments side by side, and where each number comes from
  decisions/                 decision records
reference/                   the original Open-X script (unchanged)
bash_scripts/env_dgx.sh      shell setup on the DGX
```

`data/`, `outputs/`, `logs/`, `wandb/` and all weight files are gitignored.

## Validating a change

There is no test suite. After a change, run a smoke run with a new run name. If the change
must not affect any number (a refactor, for example), re-evaluate a few finished runs with
`--eval-only` and compare `val_metrics_reeval.json` exactly against the values from before
the change.
