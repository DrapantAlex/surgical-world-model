"""Everything about a run folder outputs/<run>/: rebuilding a run, and writing its records.
Rebuilding: the backward-compatible readers of config.json (older runs lack newer keys and
get the value they were trained with), the run's own files (quaternion reference, keep
ranges) and build_model. Records: git_commit (saved in config.json) and save_json.
Setting up a new run: prepare_stats (the training-only statistics) and check_schedule
(--val-every). Shared by scripts/train.py (training and --eval-only),
scripts/train_state_only.py and the analysis scripts that reload saved weights. Moved here
from scripts/train.py on 2026-11-19 (refactor steps 3 and 4a) and 2026-11-23 (section E),
unchanged.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import data
from models.balance import BalancedWorldModel
from models.simple_world_model.swm import build_actor_world_model

# --- A: fixed choices needed to rebuild a run ---
# the action weight is fixed; the two world-model weights are only the DEFAULTS of
# --w-next-state / --w-next-image (and the values of runs saved before those options)
LOSS_WEIGHTS = {"action_output": 1.0, "next_state_output": 1.0, "next_image_output": 0.1}
DELTA_FLOOR_C = 0.25   # --state-delta: scale = max(median|Δ|, DELTA_FLOOR_C * mean|Δ|) per dim
CHUNK_FLOOR_C = 0.25   # --chunk: scale = max(median, CHUNK_FLOOR_C * mean) of each group's offset norm
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

def run_wm_balance(cfg):
    """NEW (c): the target share ρ of --wm-balance, or None (off; runs saved before the option)."""
    return cfg.get("wm_balance")


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

def balanced_model(model, cfg, stats, mode="balance", refs=None, state_dims=None):
    """NEW (c): re-wrap the functional `model` (same layers and variables, so the same weight
    file) as balance.BalancedWorldModel. refs / state_dims default to stats["wm_balance"];
    `mode`, `refs` and `state_dims` are overridden only by the equivalence check."""
    if refs is None or state_dims is None:
        wb = stats["wm_balance"]
        refs = refs or {"image": wb["image_copy_mse"], "state": wb["state_copy_mse"]}
        state_dims = wb["state_dims"] if state_dims is None else state_dims
    w = loss_weights(cfg)   # the raw history-run weights (--w-next-* are rejected with --wm-balance)
    return BalancedWorldModel(
        inputs=model.inputs, outputs=model.outputs, name=model.name,
        rho=cfg["wm_balance"], balanced=cfg["wm_balance_terms"],
        refs=refs, state_dims=state_dims,
        raw_weights={"image": w["next_image_output"], "state": w["next_state_output"]},
        every=cfg["wm_balance_every"], mode=mode)

def build_model(cfg, stats, balance_overrides=None):
    """balance_overrides (NEW (c)): keyword arguments for balanced_model (mode, refs,
    state_dims); only scripts/analysis/check_wm_balance_equivalence.py passes them."""
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
    if run_wm_balance(cfg) is not None:   # NEW (c): the same graph with the balancing train_step
        model = balanced_model(model, cfg, stats, **(balance_overrides or {}))
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


# --- E: setting up a new run: its statistics and its evaluation schedule ---
# Moved here from scripts/train.py on 2026-11-23 (shared with scripts/train_state_only.py), unchanged.
def prepare_stats(root, out, canonical_quaternion=False, keep_ranges_path=None, chunk=None,
                  history=False, state_delta=False, wm_balance_groups=None):
    """The run's training-only statistics, saved in the run folder `out`; never the test split.
    compute_norm_stats raises if the train split is incomplete: no fallback here.
    Returns (stats, quat_ref, keep_ranges, keep_hash); quat_ref / keep_ranges / keep_hash are
    None when the option is off.
    NEW (c) wm_balance_groups = [state group names] (--wm-balance): + stats["wm_balance"], the
    copy references of the balanced WM losses, from the FULL train split's pairs with frames
    (memory-mapped cache, also in smoke runs, like the norm stats; ≈ 2 min on STAR-IL)."""
    root = Path(root)
    train_eps, _ = data.drop_missing_videos("train", data.get_split_episodes("train", root),
                                            root=root, verbose=False)
    # optional keep ranges: copied into the run (eval-only reads the copy, checked by hash);
    # used for every statistic below AND both splits' pairs
    keep_ranges, keep_hash = None, None
    if keep_ranges_path:
        shutil.copyfile(keep_ranges_path, out / "keep_ranges.csv")
        keep_hash = data.file_sha256(out / "keep_ranges.csv")
        keep_ranges = data.load_keep_ranges(out / "keep_ranges.csv")   # checked by the caller
    # optional canonical quaternion signs: reference from the full train split, saved
    # with the run, and used for the stats AND both splits (all through load_table)
    quat_ref = None
    if canonical_quaternion:
        quat_ref = data.compute_quat_reference(train_eps, root, keep_ranges)
        data.save_quat_reference(quat_ref, out / "quat_reference.json")
    stats = data.compute_norm_stats(train_eps, root, quat_ref, keep_ranges)
    if state_delta:   # E7: Δ scale from the full train split, like the norm stats
        stats["state_delta"] = data.compute_delta_scale(train_eps, root, quat_ref, DELTA_FLOOR_C,
                                                        keep_ranges)
    if chunk:   # E8: option-C chunk scale from the full train split, like the norm stats
        stats["action_chunk"] = data.compute_chunk_scale(train_eps, root, quat_ref, chunk,
                                                         CHUNK_FLOOR_C, keep_ranges)
    if history:   # history: per-dim std of h(t) over the full train split's pairs
        stats["history_scale"] = data.compute_history_scale(train_eps, root, stats, quat_ref,
                                                            chunk, keep_ranges)
    if wm_balance_groups:   # NEW (c): copy references for the balanced WM losses
        # imported here: models/references.py imports metric_name from this module
        from models.references import wm_balance_references
        groups = data.state_groups(data.feature_names(root)[0])
        dims = sorted(d for g in wm_balance_groups for d in groups[g])
        train_full = data.load_split("train", root=root, stats=stats, quat_ref=quat_ref,
                                     episodes=train_eps, chunk=chunk, keep_ranges=keep_ranges)
        stats["wm_balance"] = {"state_groups": list(wm_balance_groups),
                               **wm_balance_references(train_full, dims)}
        del train_full
        print(f"--wm-balance references (train): copy_frame {stats['wm_balance']['image_copy_mse']:.6g}, "
              f"copy_state on {wm_balance_groups} {stats['wm_balance']['state_copy_mse']:.6g}")
    data.save_norm_stats(stats, out / "norm_stats.json")
    return stats, quat_ref, keep_ranges, keep_hash

def checks(x_epochs, batches_per_epoch, val_every):
    """--val-every: a count in real epochs -> the number of checks (N batches each) it spans."""
    return int(np.ceil(x_epochs * batches_per_epoch / val_every))

def check_schedule(n_train_pairs, batch_size, epochs, patience, vis_every, val_every=None):
    """Evaluation schedule: one Keras "epoch" = one check. Normal runs: a check per real epoch.
    --val-every N: the passes repeated endlessly, cut into checks of N batches;
    --epochs/--patience/--vis-every converted from real epochs."""
    batches_per_epoch = int(np.ceil(n_train_pairs / batch_size))
    if val_every:
        batches_per_check = val_every
        max_checks, patience_checks, vis_every_checks = (
            checks(x, batches_per_epoch, val_every) for x in (epochs, patience, vis_every))
        print(f"--val-every {val_every}: {batches_per_epoch} batches per epoch -> "
              f"{max_checks} checks max (= {epochs} epochs), patience {patience_checks} "
              f"checks (= {patience} epochs), timeline every {vis_every_checks} checks")
    else:
        batches_per_check = batches_per_epoch
        max_checks, patience_checks, vis_every_checks = epochs, patience, vis_every
    return {"batches_per_epoch": batches_per_epoch, "batches_per_check": batches_per_check,
            "max_checks": max_checks, "patience_checks": patience_checks,
            "vis_every_checks": vis_every_checks}
