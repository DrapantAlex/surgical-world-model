"""File 2: train the actor-world model on the train split, validate on the val split.

- normalisation statistics from the TRAIN split only, saved with the run
- best weights chosen by the validation action loss (weights only, no model.keras)
- logs to TensorBoard (logs/<run>) and Weights & Biases, plus the GT/PRED timeline
- at the end, the best weights are evaluated on val with models/metrics.py (the same
  code as scripts/baselines.py), saved as JSON and appended to outputs/results.csv
The test split is never loaded.

  python -m scripts.train --run-name smoke --max-episodes 2 --epochs 2 --wandb-mode offline
  python -m scripts.train --run-name smoke --eval-only
  python -m scripts.train --run-name <run> --eval-only --eval-keep-ranges <file>   # other val pairs
  python -m scripts.train --dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il
  python -m scripts.train ... --w-next-state 0 --w-next-image 0   # world model off (actor only)
  python -m scripts.train ... --state-delta   # E7: WM state head predicts the change (skip)
  python -m scripts.train ... --state-skip    # 1a: state + zero-initialised Dense, MSE (starts = copy)
  python -m scripts.train ... --state-layers 3   # 1b: 3 hidden Dense(128) in the state head
  python -m scripts.train ... --chunk 16      # E8: actor predicts 16 offsets from a(t-1)
  python -m scripts.train ... --keep-ranges configs/keep_ranges/star_il_tool_visible.csv
                                              # only the kept frames (pairs + statistics)
  python -m scripts.train ... --val-every 500 # one check (full val) every 500 train batches
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from tensorflow import keras

from models import data, metrics
from models.encoders import ENCODERS
from models.pipeline import make_dataset
from models.simple_world_model.swm import build_actor_world_model
from models.visualization import WorldModelTimelineCallback

# --- 0: fixed choices (deliberately not command-line options) ---
# the action weight is fixed; the two world-model weights are only the DEFAULTS of
# --w-next-state / --w-next-image (and the values of runs saved before those options)
LOSS_WEIGHTS = {"action_output": 1.0, "next_state_output": 1.0, "next_image_output": 0.1}
MONITOR = "val_action_output_loss"     # best checkpoint and early stopping (kept as in every
                                       # earlier run; the exact per-pair value is val_action_output_huber)
WANDB_PROJECT = "surgical-world-model"
DELTA_FLOOR_C = 0.25   # --state-delta: scale = max(median|Δ|, DELTA_FLOOR_C * mean|Δ|) per dim
CHUNK_FLOOR_C = 0.25   # --chunk: scale = max(median, CHUNK_FLOOR_C * mean) of each group's offset norm
GLITCH_G = 5           # glitch report (evaluation only): a one-frame jump > GLITCH_G x the train p99
                       # of |Δ| of that dim that comes back (glitch_frames); STAR-IL endo360: the
                       # 0.11-0.126 rad spikes of scripts/look_endo360.py
OUTPUTS, LOGS = Path("outputs"), Path("logs")


# --- A: command line ---
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", help="default: built from encoder, freeze, latent, seed, time")
    p.add_argument("--dataset-root", default=data.ROOT.as_posix(),
                   help="LeRobot dataset folder (default: tissue lift)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10, help="early stopping, in epochs")
    p.add_argument("--latent-dim", type=lambda v: None if v.lower() == "none" else int(v),
                   default=512, help="512 or none")
    p.add_argument("--encoder", choices=ENCODERS, default="simple_cnn")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--unfreeze-last-n", type=int, default=0,
                   help="with --freeze-encoder: train the last N backbone layers again")
    p.add_argument("--max-episodes", type=int,
                   help="only the first N train and N val episodes (smoke test)")
    p.add_argument("--canonical-quaternion", action="store_true",
                   help="flip every quaternion label to the sign of the train mean rotation "
                        "(saved as quat_reference.json); off = the dataset's own signs")
    p.add_argument("--w-next-state", type=float, default=LOSS_WEIGHTS["next_state_output"],
                   help="loss weight of the next-state head (0 = no gradient from it)")
    p.add_argument("--w-next-image", type=float, default=LOSS_WEIGHTS["next_image_output"],
                   help="loss weight of the next-image head (0 = no gradient from it)")
    p.add_argument("--state-delta", action="store_true",
                   help="E7: state head predicts the change f, next_state = state + f (skip); "
                        "loss = Huber on Δ / scale (scale saved in norm_stats.json)")
    p.add_argument("--state-skip", action="store_true",
                   help="1a: next_state = state + Dense(...) zero-initialised (starts as "
                        "copy_state); loss stays MSE on the normalised next state")
    p.add_argument("--state-layers", type=int, default=1,
                   help="1b: hidden Dense(128, relu) layers in the WM state head (1 = original)")
    p.add_argument("--chunk", type=int, default=None,
                   help="E8: actor predicts this many offsets a(t+i) - a(t-1), option-C scale "
                        "per step and group (saved in norm_stats.json); default: a(t) only")
    p.add_argument("--keep-ranges", default=None,
                   help="CSV episode, first_visible, last_visible: use only those frames for "
                        "pairs and statistics (copied into the run folder, with its hash)")
    p.add_argument("--eval-keep-ranges", default=None,
                   help="with --eval-only: evaluate on the pairs of THIS keep-ranges file (the "
                        "run's own statistics), saved as val_metrics_keep_<file stem>.json")
    p.add_argument("--vis-every",type=int, default=5, help="timeline image every N epochs")
    p.add_argument("--val-every", type=int, default=None,
                   help="validate (full val split) every N training batches instead of once per "
                        "epoch; --epochs, --patience, --vis-every stay in real epochs")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--deterministic", action="store_true", help="deterministic TF ops (slower)")
    p.add_argument("--eval-only", action="store_true",
                   help="re-evaluate the saved best weights of outputs/<run-name> on val")
    args = p.parse_args()
    if args.unfreeze_last_n and not args.freeze_encoder:
        p.error("--unfreeze-last-n needs --freeze-encoder")
    if args.eval_only and not args.run_name:
        p.error("--eval-only needs --run-name")
    if args.w_next_state < 0 or args.w_next_image < 0:
        p.error("loss weights must be >= 0")
    if args.val_every is not None and args.val_every <= 0:
        p.error("--val-every must be a positive number of batches")
    if args.state_layers < 1:
        p.error("--state-layers must be >= 1")
    if args.state_skip and args.state_delta:
        p.error("--state-skip and --state-delta are two different state heads: choose one")
    # keep-ranges files are checked here, before the run folder exists
    if args.eval_keep_ranges and not args.eval_only:
        p.error("--eval-keep-ranges needs --eval-only (use --keep-ranges for training)")
    for flag, path in (("--keep-ranges", args.keep_ranges), ("--eval-keep-ranges", args.eval_keep_ranges)):
        if path and not Path(path).is_file():
            p.error(f"{flag}: {path} not found")
    if args.keep_ranges and not args.eval_only:   # exactly train + val of this dataset
        try:
            data.check_keep_ranges(data.load_keep_ranges(args.keep_ranges), Path(args.dataset_root))
        except ValueError as e:
            p.error(f"--keep-ranges: {e}")
    return args


# --- B: helpers shared by training and --eval-only ---
def freeze_label(cfg):
    if not cfg["freeze_encoder"]:
        return "trainable"
    return f"frozen+last{cfg['unfreeze_last_n']}" if cfg["unfreeze_last_n"] else "frozen"

def dataset_root(cfg):
    """Runs saved before --dataset-root existed were all tissue lift."""
    return Path(cfg.get("dataset_root", data.ROOT))

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

def baselines_json(canonical, trimmed=False):
    """Baselines measured with the same quaternion convention (and keep ranges) as the run:
    normalised MSE is not comparable across conventions (different label std)."""
    name = "baselines" + ("_canonical" if canonical else "") + ("_trimmed" if trimmed else "")
    return OUTPUTS / name / "baselines.json"

def shuffle_order(cfg):
    """Which reshuffles the run trained on (see section D); runs saved before --val-every
    were all normal Keras epochs."""
    return cfg.get("shuffle_order", "keras_epochs")

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
                                    init_seed=cfg.get("seed", 0))   # 1b: seeds of the extra state layers
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

def copy_state_mse(split):
    """Val MSE of "next state = current state" (copy_state), normalised units: the mean of
    (s(t+1) - s(t))² over every pair t and every state dim. Same definition as Keras "mse" on
    next_state_output (mean over dims, then over samples; equal counts per sample)."""
    t = split["pair_index"]
    diff = split["states"][t + 1].astype(np.float64) - split["states"][t].astype(np.float64)
    return float(np.mean(diff ** 2))

def copy_state_mse_per_dim(split):
    """copy_state_mse for each state dimension separately (mean over pairs only); the mean of
    these values is copy_state_mse (equal counts per dimension)."""
    t = split["pair_index"]
    diff = split["states"][t + 1].astype(np.float64) - split["states"][t].astype(np.float64)
    return np.mean(diff ** 2, axis=0)

def copy_frame_mse(split, chunk=256):
    """Val MSE of "next frame = current frame" (copy_frame): images in [0, 1] through get_pairs
    (the model's own inputs and targets), in chunks so the split never exists as float. Every
    pair has the same pixel count, so the overall mean = mean over pixels, then over pairs
    (= the per-pair metric val_next_image_output_mse). Same code as scripts/baselines.py."""
    t, image = split["pair_index"], metrics.ImageMSE()
    for i in range(0, len(t), chunk):
        inputs, targets = data.get_pairs(split, t[i:i + chunk])
        image.update(inputs["image_input"], targets["next_image_output"])
    return image.result()

def prev_in_window(split, keep_ranges):
    """For each pair t: is s(t-1) in the same episode AND inside its kept window? (Frames
    outside the window exist in the split, but the run never uses them.)"""
    t, ep = split["pair_index"], split["episode_index"]
    episodes, starts, lengths = np.unique(ep, return_index=True, return_counts=True)
    first_kept = np.empty(len(ep), np.int64)   # first kept row of each row's episode
    for e, s, T in zip(episodes, starts, lengths):
        first_kept[s:s + T] = s + data.kept_window(int(e), int(T), keep_ranges)[0]
    return t - 1 >= first_kept[t]

def constvel_state(split, keep_ranges):
    """Constant velocity s(t) + (s(t) - s(t-1)) for every pair (normalised units); copy (Δ = 0)
    where s(t-1) is not available (prev_in_window). Returns (prediction, n_fallback_pairs)."""
    t, s = split["pair_index"], split["states"]
    ok = prev_in_window(split, keep_ranges)
    vel = np.where(ok[:, None], s[t].astype(np.float64) - s[t - 1], 0.0)
    return s[t] + vel, int((~ok).sum())

def constvel_state_mse_per_dim(split, keep_ranges):
    """Constant-velocity val MSE per state dim (mean over pairs), and the fallback count."""
    pred, n_fallback = constvel_state(split, keep_ranges)
    t = split["pair_index"]
    return np.mean((pred - split["states"][t + 1]) ** 2, axis=0), n_fallback

def glitch_thresholds(train_split):
    """Per state dim: GLITCH_G x the p99 of |s(t+1) - s(t)| over the TRAIN pairs (normalised)."""
    t, s = train_split["pair_index"], train_split["states"].astype(np.float64)
    return GLITCH_G * np.percentile(np.abs(s[t + 1] - s[t]), 99, axis=0)

def glitch_frames(split, thresholds):
    """(N, D) bool: frame t of dim d is a one-frame glitch if, within one episode,
    |s(t) - s(t-1)| > threshold (and threshold > 0), the next step goes the opposite way, and
    the value comes back: |s(t+1) - s(t-1)| < 0.2 |s(t) - s(t-1)| (this also makes the two steps
    of similar size, 0.8-1.2x). Uses the episode's full rows, not only the kept window."""
    s, ep = split["states"].astype(np.float64), split["episode_index"]
    before, after, back = s[1:-1] - s[:-2], s[2:] - s[1:-1], s[2:] - s[:-2]
    same = ((ep[:-2] == ep[1:-1]) & (ep[1:-1] == ep[2:]))[:, None]
    g = (same & (thresholds > 0) & (np.abs(before) > thresholds)
         & (np.sign(after) == -np.sign(before)) & (np.abs(back) < 0.2 * np.abs(before)))
    out = np.zeros(s.shape, bool)
    out[1:-1] = g
    return out

def state_group_metrics(pred_state, split, names, keep_ranges, thresholds):
    """Per state group (data.state_groups): model, copy_state and constant-velocity MSE
    (normalised), with all val pairs and without the glitch pairs (state t or t+1 is a glitch
    frame in any dim of the group), plus the glitch counts. Measurement only."""
    t = split["pair_index"]
    cur, true = split["states"][t].astype(np.float64), split["states"][t + 1]
    constvel, n_fallback = constvel_state(split, keep_ranges)
    preds = {"model": pred_state, "copy": cur, "constvel": constvel}
    glitch = glitch_frames(split, thresholds)
    used = np.unique(np.concatenate([t, t + 1]))   # frames the val pairs touch
    m = {"state_group/n_constvel_fallback_pairs": float(n_fallback)}
    for g, dims in data.state_groups(names).items():
        key = f"state_group/{metric_name(g)}"
        bad = glitch[t][:, dims].any(1) | glitch[t + 1][:, dims].any(1)
        m[f"{key}/n_glitch_frames"] = float(glitch[used][:, dims].any(1).sum())
        m[f"{key}/n_glitch_pairs"] = float(bad.sum())
        for who, p in preds.items():
            err = (np.asarray(p, np.float64)[:, dims] - true[:, dims]) ** 2
            m[f"{key}/{who}_mse"] = float(err.mean())
            m[f"{key}/{who}_mse_noglitch"] = float(err[~bad].mean()) if (~bad).any() else float("nan")
    return m

def trainmean_huber(train, val):
    """E8: val action loss of the input-free rule "always predict the mean TRAIN chunk target"
    (normalised units), with the compiled Huber() like standstill_huber. The bar a model must
    beat to show that it uses its inputs (not only the average drift)."""
    mean = data.chunk_targets(train, train["pair_index"]).astype(np.float64).mean(axis=0)
    true = data.chunk_targets(val, val["pair_index"])
    return float(keras.losses.Huber()(true, np.broadcast_to(mean, true.shape).astype(np.float32)))

def standstill_huber(val):
    """E8: val action loss of stand-still (every offset 0): the compiled Huber() on all val
    chunk targets at once = the mean over every pair. Note: Keras logs val losses as the
    UNWEIGHTED mean of the batch means (the batch size is None in our tf.data graph), so the
    partial last batch is over-weighted; logged curves can differ from this line by ~1e-4."""
    true = data.chunk_targets(val, val["pair_index"])   # the get_pairs target, no frames
    return float(keras.losses.Huber()(true, np.zeros_like(true)))

def checks(x_epochs, batches_per_epoch, val_every):
    """--val-every: a count in real epochs -> the number of checks (N batches each) it spans."""
    return int(np.ceil(x_epochs * batches_per_epoch / val_every))

def evaluate(model, val, stats, batch_size, root, factor=None, keep_ranges=None, glitch_thr=None):
    """Predict every val pair (fixed order) and measure it with models/metrics.py.
    glitch_thr (glitch_thresholds of the train split): adds the per-group state report."""
    pred_action, true_action, pred_state, image = [], [], [], metrics.ImageMSE()
    for inputs, targets in make_dataset(val, batch_size, shuffle=False):
        a, s, img = model(inputs, training=False)
        pred_action.append(a.numpy())
        true_action.append(targets["action_output"].numpy())
        pred_state.append(s.numpy())
        image.update(img.numpy(), targets["next_image_output"].numpy())
    t = val["pair_index"]
    physical = metrics.has_tissue_lift_layout(root)   # else core metrics only, no "extra/"
    state_names, action_names = data.feature_names(root)
    if "chunk_factor" in val:   # E8: offsets in option-C units, per step, in mm / deg
        m = metrics.chunk_metrics(np.concatenate(pred_action), np.concatenate(true_action),
                                  data.chunk_anchors(val, t)[0], stats, action_names)
    else:
        m = metrics.action_metrics(np.concatenate(pred_action), val["actions"][t], stats,
                                   physical, action_names)
    m.update(metrics.next_state_metrics(np.concatenate(pred_state), val["states"][t + 1], stats,
                                        physical, state_names))
    m["next_state_mse_copy"] = copy_state_mse(val)   # 1a: reference for next_state_mse, same pairs
    if glitch_thr is not None:   # per group: model / copy / constant velocity, with and without glitches
        m.update(state_group_metrics(np.concatenate(pred_state), val, state_names, keep_ranges,
                                     glitch_thr))
    if factor is not None:   # E7: the loss on the change, model vs f = 0
        m.update(metrics.state_delta_metrics(np.concatenate(pred_state), val["states"][t],
                                             val["states"][t + 1], factor))
    m.update(metrics.next_image_metrics(image))
    return m

def chunk_baseline_metrics(val, stats, root):
    """E8: stand-still and constant velocity on the same val pairs, through the same
    metrics.chunk_metrics as the model (labels only)."""
    t = val["pair_index"]
    true = data.chunk_targets(val, t)   # the get_pairs target, without loading frames
    anchor_n, prev_n = data.chunk_anchors(val, t)
    names = data.feature_names(root)[1]
    return {who: metrics.chunk_metrics(pred, true, anchor_n, stats, names)
            for who, pred in metrics.chunk_baselines(anchor_n, prev_n, stats).items()}

def comparison_table(model_name, model_metrics, root, canonical=False, chunk_baselines=None,
                     keep_hash=None):
    """The model's metrics next to the baselines (if scripts/baselines.py has been run
    on the same dataset, quaternion convention and keep ranges; baselines.json files
    without "dataset_root" are tissue lift, without "keep_ranges_sha256" untrimmed).
    E8 chunk runs: next to their own chunk baselines, because the one-step baselines
    measure a different target."""
    columns = {model_name: model_metrics}
    if chunk_baselines is not None:
        columns.update(chunk_baselines)
        return metrics.format_table(columns)
    path = baselines_json(canonical, trimmed=keep_hash is not None)
    if path.exists():
        baselines = json.loads(path.read_text())
        if (Path(baselines.get("dataset_root", data.ROOT)) == Path(root)
                and baselines.get("keep_ranges_sha256") == keep_hash):
            columns.update(baselines["results"])
    return metrics.format_table(columns)

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

def log_tensorboard(val_metrics, log_dir, step):
    """Final metrics of the best weights, same names as results.csv ("extra/..." its own group)."""
    with tf.summary.create_file_writer(str(Path(log_dir) / "best_val")).as_default():
        for k, v in val_metrics.items():
            tf.summary.scalar(k, v, step=step)


# --- C: callbacks ---
class EpochTimer(keras.callbacks.Callback):
    """Adds the duration of each epoch to the logs, so every later callback records it."""
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["epoch_time_s"] = time.time() - self.t0

class TrainingProgress(keras.callbacks.Callback):
    """Adds the real x-axis to each check's logs: training batches seen so far and the
    same in real epochs. One Keras "epoch" = one check = batches_per_check batches
    (--val-every N: N; normal runs: one full epoch)."""
    def __init__(self, batches_per_check, batches_per_epoch):
        super().__init__()
        self.batches_per_check, self.batches_per_epoch = batches_per_check, batches_per_epoch

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["batches_seen"] = (epoch + 1) * self.batches_per_check
            logs["epochs_seen"] = logs["batches_seen"] / self.batches_per_epoch

class CheckMonitor(keras.callbacks.Callback):
    """Stops at once if Keras does not log MONITOR (loss names differ between versions)."""
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 and MONITOR not in (logs or {}):
            raise KeyError(f"{MONITOR!r} is not in the Keras logs: {sorted(logs or {})}")

class ConstantLogs(keras.callbacks.Callback):
    """Adds fixed reference values (e.g. the f = 0 val loss) to every epoch's logs,
    so they appear as flat lines next to the real curves (CSV, TensorBoard, W&B)."""
    def __init__(self, values):
        super().__init__()
        self.values = values

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs.update(self.values)

class StateGroupLogs(keras.callbacks.Callback):
    """Adds (val_)next_state_<group>_mse = the mean of the group's per-dimension metrics
    (state_dim_metrics). Every per-dim metric averages over the same samples, so this is
    exactly the group's MSE. groups: {group metric name: [dim metric names]}."""
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        for prefix in ("", "val_"):
            for g, dims in self.groups.items():
                keys = [f"{prefix}next_state_output_mse_{d}" for d in dims]
                if all(k in logs for k in keys):
                    logs[f"{prefix}next_state_{g}_mse"] = float(np.mean([logs[k] for k in keys]))

class WandbLogger(keras.callbacks.Callback):
    """Every log value except the per-dimension state columns and their _copy lines
    (kept in history.csv and TensorBoard; W&B gets the per-group columns instead)."""
    def __init__(self, run):
        super().__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        self.run.log({k: float(v) for k, v in (logs or {}).items()
                      if "next_state_output_mse_" not in k}, step=epoch)

def timeline_callback(val, run, log_dir, every, root):
    """The existing GT/PRED timeline, on the first validation episode (8 frames, 10 apart)."""
    ep = int(val["episode_index"][0])
    t = val["pair_index"][val["episode_index"][val["pair_index"]] == ep]
    inputs, targets = data.get_pairs(val, t)
    return WorldModelTimelineCallback(log_dir, inputs, targets, data.load_instruction(ep, root),
                                      wandb_run=run, num_frames=8, start=0, stride=10,
                                      every=every)


# --- D: training ---
def train(args):
    run_name = args.run_name or (f"{args.encoder}-{freeze_label(vars(args))}-lat{args.latent_dim}"
                                 f"-s{args.seed}-{datetime.now():%Y%m%d-%H%M}")
    out, log_dir = OUTPUTS / run_name, LOGS / run_name
    if out.exists():
        raise SystemExit(f"{out} already exists: choose another --run-name")
    out.mkdir(parents=True)

    # reproducibility: Python, NumPy and TensorFlow seeds (+ the tf.data shuffle seed)
    keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    # data: training-only statistics, saved with the run; never the test split.
    # compute_norm_stats raises if the train split is incomplete: no fallback here.
    root = Path(args.dataset_root)
    train_eps, _ = data.drop_missing_videos("train", data.get_split_episodes("train", root),
                                            root=root, verbose=False)
    # optional keep ranges: copied into the run (eval-only reads the copy, checked by hash);
    # used for every statistic below AND both splits' pairs
    keep_ranges, keep_hash = None, None
    if args.keep_ranges:
        shutil.copyfile(args.keep_ranges, out / "keep_ranges.csv")
        keep_hash = data.file_sha256(out / "keep_ranges.csv")
        keep_ranges = data.load_keep_ranges(out / "keep_ranges.csv")   # checked in parse_args
    # optional canonical quaternion signs: reference from the full train split, saved
    # with the run, and used for the stats AND both splits (all through load_table)
    quat_ref = None
    if args.canonical_quaternion:
        quat_ref = data.compute_quat_reference(train_eps, root, keep_ranges)
        data.save_quat_reference(quat_ref, out / "quat_reference.json")
    stats = data.compute_norm_stats(train_eps, root, quat_ref, keep_ranges)
    if args.state_delta:   # E7: Δ scale from the full train split, like the norm stats
        stats["state_delta"] = data.compute_delta_scale(train_eps, root, quat_ref, DELTA_FLOOR_C,
                                                        keep_ranges)
    if args.chunk:   # E8: option-C chunk scale from the full train split, like the norm stats
        stats["action_chunk"] = data.compute_chunk_scale(train_eps, root, quat_ref, args.chunk,
                                                         CHUNK_FLOOR_C, keep_ranges)
    data.save_norm_stats(stats, out / "norm_stats.json")
    train_split = data.load_split("train", root=root, stats=stats, quat_ref=quat_ref,
                                  episodes=episodes_for("train", args.max_episodes, root),
                                  chunk=args.chunk, keep_ranges=keep_ranges)
    val = data.load_split("val", root=root, stats=stats, quat_ref=quat_ref,
                          episodes=episodes_for("val", args.max_episodes, root), chunk=args.chunk,
                          keep_ranges=keep_ranges)
    # evaluation schedule: one Keras "epoch" = one check. Normal runs: a check per real epoch.
    # --val-every N: the passes repeated endlessly, cut into checks of N batches (shuffle
    # order: see below); --epochs/--patience/--vis-every converted from real epochs
    batches_per_epoch = int(np.ceil(len(train_split["pair_index"]) / args.batch_size))
    if args.val_every:
        batches_per_check = args.val_every
        max_checks, patience_checks, vis_every_checks = (
            checks(x, batches_per_epoch, args.val_every)
            for x in (args.epochs, args.patience, args.vis_every))
        print(f"--val-every {args.val_every}: {batches_per_epoch} batches per epoch -> "
              f"{max_checks} checks max (= {args.epochs} epochs), patience {patience_checks} "
              f"checks (= {args.patience} epochs), timeline every {vis_every_checks} checks")
    else:
        batches_per_check = batches_per_epoch
        max_checks, patience_checks, vis_every_checks = args.epochs, args.patience, args.vis_every
    # shuffle order: every new iterator draws the next reshuffle. Normal epochs: Keras 3.15
    # creates two iterators between epochs and uses only the second (shuffles #1, #3, #5...,
    # "keras_epochs"); .repeat() uses every one (#1, #2, #3..., "repeat"). So epoch 1 is
    # identical to old runs (checked bitwise: scripts/check_val_every_weights.py), later
    # epochs see a different but equally random order.
    train_ds = make_dataset(train_split, args.batch_size, shuffle=True, seed=args.seed,
                            repeat=bool(args.val_every))
    val_ds = make_dataset(val, args.batch_size, shuffle=False)

    # model + full configuration, saved before training starts
    cfg = {k: v for k, v in vars(args).items() if k != "eval_only"}
    cfg["run_name"] = run_name
    model = build_model(cfg, stats)
    factor = delta_factor(cfg, stats)
    # reference lines for the state branch, flat in every epoch's logs and saved in config.json:
    # copy_state (every run; 1a: the untrained --state-skip model) and, for E7, f = 0
    reference_logs = {"val_next_state_output_loss_copy": copy_state_mse(val)}
    # the image branch's reference: copy_frame (one pass over the val frames, in chunks)
    reference_logs["val_next_image_output_loss_copyframe"] = copy_frame_mse(val)
    print(f"copy_frame val image MSE: {reference_logs['val_next_image_output_loss_copyframe']:.6g}")
    # copy_state per state dimension, next to the per-dimension val metrics (same names)
    state_names = data.feature_names(root)[0]
    copy_dim = copy_state_mse_per_dim(val)
    for name, value in zip(state_names, copy_dim):
        reference_logs[f"val_next_state_output_mse_{metric_name(name)}_copy"] = float(value)
    # per state group (data.state_groups): copy_state and constant velocity, next to the
    # group curves of StateGroupLogs (the mean of the group's per-dim values)
    groups = data.state_groups(state_names)
    constvel_dim, n_fallback = constvel_state_mse_per_dim(val, keep_ranges)
    for g, dims in groups.items():
        reference_logs[f"val_next_state_{metric_name(g)}_mse_copy"] = float(copy_dim[dims].mean())
        reference_logs[f"val_next_state_{metric_name(g)}_mse_constvel"] = float(constvel_dim[dims].mean())
    glitch_thr = glitch_thresholds(train_split)   # for the glitch report of the final evaluation
    if args.chunk:   # E8: the actor's reference lines, in the loss units:
        # stand-still (offsets 0) and the input-free mean train offset
        reference_logs["val_action_output_loss_standstill"] = standstill_huber(val)
        reference_logs["val_action_output_loss_trainmean"] = trainmean_huber(train_split, val)
    if factor is not None:
        reference_logs["val_next_state_output_loss_zero"] = metrics.state_delta_metrics(
            val["states"][val["pair_index"]], val["states"][val["pair_index"]],
            val["states"][val["pair_index"] + 1], factor)["next_state_delta_huber_zero"]
    cfg.update(**reference_logs)
    if args.state_skip:   # 1a: before any training, the state head must be exactly copy_state
        inputs, _ = next(iter(val_ds))
        pred_state = model(inputs, training=False)[1].numpy()
        gap = float(np.max(np.abs(pred_state - inputs["state_input"].numpy())))
        print(f"state skip check: max |next_state - state_input| = {gap} (must be 0)")
        if gap != 0.0:
            raise AssertionError(f"untrained --state-skip model is not copy_state (gap {gap})")
    cfg.update(
        freeze=freeze_label(cfg), monitor=MONITOR, loss_weights=loss_weights(cfg),
        params_total=int(model.count_params()),
        params_trainable=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        n_train_pairs=int(len(train_split["pair_index"])), n_val_pairs=int(len(val["pair_index"])),
        n_train_pairs_untrimmed=int(train_split["n_pairs_untrimmed"]),
        n_val_pairs_untrimmed=int(val["n_pairs_untrimmed"]),
        batches_per_epoch=batches_per_epoch, batches_per_check=batches_per_check,
        max_checks=max_checks, patience_checks=patience_checks,
        vis_every_checks=vis_every_checks,
        shuffle_order="repeat" if args.val_every else "keras_epochs",
        n_val_constvel_fallback=n_fallback,   # pairs without s(t-1) in the kept window: copy
        glitch_threshold=glitch_thr.tolist(),  # per state dim, normalised (GLITCH_G x train p99 |Δ|)
        keep_ranges_sha256=keep_hash,
        train_episodes=np.unique(train_split["episode_index"]).tolist(),
        val_episodes=np.unique(val["episode_index"]).tolist(),
        train_skipped_episodes=train_split["skipped_episodes"],
        val_skipped_episodes=val["skipped_episodes"],
        state_dim=data.feature_dims(root)[0], action_dim=data.feature_dims(root)[1],
        tensorflow=tf.__version__, keras=getattr(keras, "__version__", "unknown"),
        git_commit=git_commit(), started=datetime.now().isoformat(timespec="seconds"))
    save_json(cfg, out / "config.json")
    print(f"run {run_name}: {cfg['params_total']:,} parameters "
          f"({cfg['params_trainable']:,} trainable), {cfg['n_train_pairs']} train pairs, "
          f"{cfg['n_val_pairs']} val pairs")

    run = wandb.init(project=WANDB_PROJECT, name=run_name, config=cfg, mode=args.wandb_mode)
    callbacks = [
        EpochTimer(),
        TrainingProgress(batches_per_check, batches_per_epoch),
        CheckMonitor(),
        ConstantLogs(reference_logs),
        StateGroupLogs({metric_name(g): [metric_name(state_names[i]) for i in dims]
                        for g, dims in groups.items()}),
        keras.callbacks.ModelCheckpoint(str(out / "best.weights.h5"), monitor=MONITOR, mode="min",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=MONITOR, mode="min", patience=patience_checks,
                                      verbose=1),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
        timeline_callback(val, run, log_dir, vis_every_checks, root),
        WandbLogger(run),
    ]
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=max_checks,
                        steps_per_epoch=args.val_every,   # None = one full pass per "epoch"
                        callbacks=callbacks, verbose=2)
    train_time_min = (time.time() - t0) / 60

    # --- E: evaluate the BEST weights with the same metrics as the baselines ---
    model.load_weights(str(out / "best.weights.h5"))
    val_metrics = evaluate(model, val, stats, args.batch_size, root, factor, keep_ranges, glitch_thr)
    chunk_base = chunk_baseline_metrics(val, stats, root) if args.chunk else None
    if chunk_base is not None:
        save_json(chunk_base, out / "chunk_baselines.json")
    monitored = history.history[MONITOR]
    # best_epoch = index of the best check (= the real epoch for normal runs, as before)
    cfg.update(epochs_run=len(monitored), best_epoch=int(np.argmin(monitored)) + 1,
               train_time_min=round(train_time_min, 2))
    cfg["best_batch"] = cfg["best_epoch"] * batches_per_check
    cfg["best_epoch_equiv"] = cfg["best_batch"] / batches_per_epoch
    save_json(cfg, out / "config.json")
    save_json(val_metrics, out / "val_metrics.json")
    log_tensorboard(val_metrics, log_dir, cfg["best_epoch"])
    run.summary.update({f"best/{k}": v for k, v in val_metrics.items()})
    run.summary.update({"best_epoch": cfg["best_epoch"], "best_batch": cfg["best_batch"],
                        "best_epoch_equiv": cfg["best_epoch_equiv"],
                        "train_time_min": cfg["train_time_min"]})

    unit = "check" if args.val_every else "epoch"
    print(f"\nbest {unit} {cfg['best_epoch']} of {cfg['epochs_run']} (batch {cfg['best_batch']}, "
          f"{cfg['best_epoch_equiv']:.2f} epochs), training {train_time_min:.1f} min\n")
    print(comparison_table(run_name, val_metrics, root, args.canonical_quaternion, chunk_base,
                           keep_hash))
    if args.max_episodes is None:
        metrics.append_results({"run_name": run_name, "dataset": root.name,
                                "encoder": args.encoder,
                                "canonical_quaternion": args.canonical_quaternion,
                                "freeze": cfg["freeze"], "seed": args.seed,
                                "latent_dim": args.latent_dim,
                                "w_next_state": args.w_next_state,
                                "w_next_image": args.w_next_image,
                                "state_delta": args.state_delta, "state_skip": args.state_skip,
                                "state_layers": args.state_layers,
                                "chunk": args.chunk,
                                "keep_ranges": Path(args.keep_ranges).name if args.keep_ranges else None,
                                "epochs": cfg["epochs_run"],
                                "best_epoch": cfg["best_epoch"],
                                "val_every": args.val_every, "best_batch": cfg["best_batch"],
                                "train_time_min": cfg["train_time_min"], **val_metrics})
        print(f"\nadded a row to {metrics.RESULTS_CSV}")
    else:
        print("\nsmoke run (--max-episodes): not added to results.csv")
    run.finish()


# --- F: re-evaluate a finished run with ITS saved statistics ---
def eval_only(args):
    out = OUTPUTS / args.run_name
    cfg = json.loads((out / "config.json").read_text())
    stats = data.load_norm_stats(out / "norm_stats.json")
    root = dataset_root(cfg)
    quat_ref = run_quat_ref(cfg, out)
    keep_ranges = run_keep_ranges(cfg, out)        # None: untrimmed (and older runs)
    keep_hash = cfg.get("keep_ranges_sha256")
    # glitch thresholds from the run's own train split (tables only), as in train(); always the
    # run's keep ranges, also with --eval-keep-ranges (older runs did not save them)
    train_tables = data.load_split("train", root=root, stats=stats, quat_ref=quat_ref,
                                   episodes=episodes_for("train", cfg["max_episodes"], root),
                                   chunk=cfg.get("chunk"), keep_ranges=keep_ranges,
                                   with_frames=False)
    glitch_thr = glitch_thresholds(train_tables)
    if args.eval_keep_ranges:   # other pairs, same weights and statistics (nothing recomputed)
        keep_ranges = data.load_keep_ranges(args.eval_keep_ranges)
        keep_hash = data.file_sha256(args.eval_keep_ranges)
        data.check_keep_ranges(keep_ranges, root)
    val = data.load_split("val", root=root, stats=stats, quat_ref=quat_ref,
                          episodes=episodes_for("val", cfg["max_episodes"], root),
                          chunk=cfg.get("chunk"),   # runs before E8: no "chunk" key
                          keep_ranges=keep_ranges)
    model = build_model(cfg, stats)
    model.load_weights(str(out / "best.weights.h5"))
    val_metrics = evaluate(model, val, stats, cfg["batch_size"], root, delta_factor(cfg, stats),
                           keep_ranges, glitch_thr)
    chunk_base = chunk_baseline_metrics(val, stats, root) if cfg.get("chunk") else None
    model_mse, copy_mse = val_metrics["next_state_mse"], val_metrics["next_state_mse_copy"]
    copy_line = (f"next_state_mse: model {model_mse:.3g} vs copy_state {copy_mse:.3g} "
                 f"(model / copy = {model_mse / copy_mse:.2f})")
    for g in data.state_groups(data.feature_names(root)[0]):
        k = f"state_group/{metric_name(g)}"
        copy_line += "\n" + " ".join(
            f"{g} {who}: model / copy = {val_metrics[f'{k}/model_mse{sfx}'] / val_metrics[f'{k}/copy_mse{sfx}']:.3f}, "
            f"constvel / copy = {val_metrics[f'{k}/constvel_mse{sfx}'] / val_metrics[f'{k}/copy_mse{sfx}']:.3f};"
            for who, sfx in (("all pairs", ""), ("no glitch", "_noglitch")))
    if args.eval_keep_ranges:   # its own JSON: val_metrics*.json of the run stay untouched
        path = out / f"val_metrics_keep_{Path(args.eval_keep_ranges).stem}.json"
        save_json({"keep_ranges": args.eval_keep_ranges, "keep_ranges_sha256": keep_hash,
                   "n_val_pairs": int(len(val["pair_index"])),
                   "n_val_pairs_untrimmed": int(val["n_pairs_untrimmed"]),
                   "metrics": val_metrics, "chunk_baselines": chunk_base}, path)
        print(comparison_table(f"{args.run_name} (keep ranges)", val_metrics, root,
                               quat_ref is not None, chunk_base, keep_hash))
        print(f"\n{copy_line}")
        print(f"saved {path}")
        return
    save_json(val_metrics, out / "val_metrics_reeval.json")
    print(comparison_table(f"{args.run_name} (re-eval)", val_metrics, root, quat_ref is not None,
                           chunk_base, keep_hash))
    print(f"\n{copy_line}")

    saved = out / "val_metrics.json"
    if saved.exists():
        before = json.loads(saved.read_text())
        common = before.keys() & val_metrics.keys()   # metric names changed over time
        diff = max(abs(before[k] - val_metrics[k]) for k in common)
        print(f"\nlargest difference from val_metrics.json: {diff:.2e}")


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        train(args)
