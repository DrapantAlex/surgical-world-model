"""Everything about a run folder outputs/<run>/: rebuilding a run, and writing its records.
Rebuilding: the backward-compatible readers of config.json (older runs lack newer keys and
get the value they were trained with), the run's own files (quaternion reference, keep
ranges) and build_model. Records: git_commit (saved in config.json) and save_json.
Shared by scripts/train.py (training and --eval-only), scripts/train_state_only.py and the
analysis scripts that reload saved weights. Moved here from scripts/train.py on 2026-11-19
(refactor steps 3 and 4a), unchanged.
"""
import json
import re
import subprocess
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from models import data
from models.simple_world_model.swm import build_actor_world_model

# --- A: fixed choices needed to rebuild a run ---
# the action weight is fixed; the two world-model weights are only the DEFAULTS of
# --w-next-state / --w-next-image (and the values of runs saved before those options)
LOSS_WEIGHTS = {"action_output": 1.0, "next_state_output": 1.0, "next_image_output": 0.1}
OUTPUTS = Path("outputs")
REPO = Path(__file__).resolve().parents[1]   # models/run_config.py -> the repo root


# --- B: readers of a run's config.json and files (defaults = older runs) ---
def freeze_label(cfg):
    if not cfg["freeze_encoder"]:
        return "trainable"
    return f"frozen+last{cfg['unfreeze_last_n']}" if cfg["unfreeze_last_n"] else "frozen"

def dataset_root(cfg):
    """Runs saved before --dataset-root existed were all tissue lift. An absolute path inside
    the repo (the first STAR-IL runs, 2026-10-16/20) becomes the repo-relative one that every
    other run, baselines.json and the frame-cache manifest use: Path equality compares the
    spelling, not the folder. Scripts run from the repo root, so both name the same files."""
    root = Path(cfg.get("dataset_root", data.ROOT))
    if root.is_absolute() and root.is_relative_to(REPO):
        return root.relative_to(REPO)
    return root

def run_quat_ref(cfg, out):
    """The run's saved reference rotations, or None (runs before the option: raw signs)."""
    if not cfg.get("canonical_quaternion", False):
        return None
    return data.load_quat_reference(out / "quat_reference.json")

def run_keep_ranges(cfg, out):
    """The run's own copy of the keep-ranges file, checked against the saved hash;
    None for runs without --keep-ranges (and runs saved before the option)."""
    if not cfg.get("keep_ranges"):
        return None
    copy = out / "keep_ranges.csv"
    if data.file_sha256(copy) != cfg["keep_ranges_sha256"]:
        raise ValueError(f"{copy} does not match the hash in config.json")
    return data.load_keep_ranges(copy)

def episodes_for(split, max_episodes, root):
    """None = the whole split (read from the frame cache); else its first N episodes."""
    return None if max_episodes is None else data.get_split_episodes(split, root)[:max_episodes]

def loss_weights(cfg):
    """Runs saved before --w-next-state / --w-next-image used the LOSS_WEIGHTS defaults."""
    return {"action_output": LOSS_WEIGHTS["action_output"],
            "next_state_output": cfg.get("w_next_state", LOSS_WEIGHTS["next_state_output"]),
            "next_image_output": cfg.get("w_next_image", LOSS_WEIGHTS["next_image_output"])}

def delta_factor(cfg, stats):
    """E7 runs: k = scale / std (see data.state_delta_factor); None for runs without
    --state-delta (and runs saved before the option)."""
    return data.state_delta_factor(stats) if cfg.get("state_delta", False) else None

def run_history(cfg):
    """History inputs on? Runs saved before --history: off."""
    return cfg.get("history", False)


# --- C: model, losses and metrics as compiled for training ---
def scaled_huber(factor):
    """Huber(next_state / k) = Huber(f, Δ / scale): next_state = state + k * f, so
    (pred - true) / k = f - Δ / scale. The loss on the change, with the s(t+1) target."""
    k = tf.constant(factor, tf.float32)
    def loss(y_true, y_pred):
        return keras.losses.huber(y_true / k, y_pred / k)
    return loss

def metric_name(feature):
    """A feature name as a Keras metric name: Keras would silently drop "[", "]", "." itself,
    so every run of other characters becomes one "_" here ("kuka_joint_pos[0]" ->
    "kuka_joint_pos_0") and the metric and its _copy line share exactly this name."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", feature).strip("_")

def state_dim_metrics(state_names):
    """One val metric per state dimension: the MSE of that dimension only, logged as
    (val_)next_state_output_mse_<name>. Keras averages metrics per SAMPLE, so the mean over
    dimensions = the per-sample MSE of next_state_output (= next_state_mse in val_metrics.json),
    not exactly the logged loss (an unweighted mean of batch means, see standstill_huber)."""
    names = [metric_name(n) for n in state_names]
    if len(set(names)) != len(names):
        raise ValueError(f"state names are not unique as metric names: {names}")
    def one(i, name):
        def fn(y_true, y_pred):
            return keras.ops.square(y_true[:, i] - y_pred[:, i])
        fn.__name__ = f"mse_{name}"
        return fn
    return [one(i, n) for i, n in enumerate(names)]

def build_model(cfg, stats):
    state_dim, action_dim = data.feature_dims(dataset_root(cfg))
    factor = delta_factor(cfg, stats)
    model = build_actor_world_model(state_dim, action_dim,
                                    latent_dim=cfg["latent_dim"], encoder=cfg["encoder"],
                                    freeze_encoder=cfg["freeze_encoder"],
                                    unfreeze_last_n=cfg["unfreeze_last_n"],
                                    delta_factor=factor, action_chunk=cfg.get("chunk"),
                                    state_skip=cfg.get("state_skip", False),   # 1a; older runs: off
                                    state_layers=cfg.get("state_layers", 1),   # 1b; older runs: 1
                                    init_seed=cfg.get("seed", 0),   # 1b/1c: seeds of the extra layers
                                    wm_mixer_blocks=cfg.get("wm_mixer_blocks", 0),   # 1c; older runs: 0
                                    history_dim=state_dim + action_dim if run_history(cfg) else 0)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]),
                  loss={"action_output": keras.losses.Huber(),
                        "next_state_output": "mse" if factor is None else scaled_huber(factor),
                        "next_image_output": "mse"},
                  loss_weights=loss_weights(cfg),
                  # metrics only (no gradient): the exact per-pair actor Huber and image MSE (Keras
                  # logs the LOSS as an unweighted mean of batch means, see standstill_huber;
                  # metrics are averaged per sample) and the per-dimension state error
                  metrics={"action_output":
                           [keras.metrics.MeanMetricWrapper(keras.losses.huber, name="huber")],
                           "next_state_output":
                           state_dim_metrics(data.feature_names(dataset_root(cfg))[0]),
                           "next_image_output":
                           [keras.metrics.MeanMetricWrapper(keras.losses.mse, name="mse")]})
    return model


# --- D: writing a run's records ---
def git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+uncommitted" if dirty else "")
    except OSError:
        return "unknown"

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))
