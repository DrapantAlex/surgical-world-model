"""Shared validation metrics, used by scripts/baselines.py AND scripts/train.py.

Both scripts call these same functions, so a trivial baseline and a trained
model are always measured in exactly the same way. Inputs are in normalised
units (what the model sees); physical errors are computed after un-normalising
with the run's training statistics.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from models.data import chunk_scale_per_dim, quaternion_starts, unnormalise

# --- A: units and layout of the 16-D state / action vectors (tissue lift ONLY) ---
POSITION_TO_MM = 1000.0          # ASSUMPTION: positions in the dataset are in metres
HUBER_DELTA = 1.0                # default of keras.losses.Huber(), used in training
ARMS = {"psm1": 0, "psm2": 8}    # where each arm starts in the 16-D vector
POS, QUAT, JAW = slice(0, 3), slice(3, 7), 7   # inside one arm: [x y z | qx qy qz qw | jaw]
RESULTS_CSV = Path("outputs/results.csv")
TISSUE_LIFT = "srt_tissue_lift"  # the only dataset whose layout is known (see ARMS)

# Core metrics: normalised units, the SAME names for every dataset (logs, TensorBoard,
# results.csv). Physical errors are extras, under EXTRA, and exist for tissue lift only.
CORE_METRICS = ("action_mse", "action_huber", "next_state_mse", "next_image_mse")
EXTRA = "extra/"
PER_DIM = "per_dim/"   # one MSE per vector dimension, named from info.json (every dataset)


# TODO: replace with an explicit per-dataset layout config (which slices are
# position / quaternion / jaw, per arm), instead of recognising tissue lift by name.
def has_tissue_lift_layout(root):
    """True only for tissue lift, the one dataset whose ARMS / POS / QUAT / JAW
    layout is known: only then are the physical "extra/" metrics computed."""
    return Path(root).name == TISSUE_LIFT


# --- B: errors in normalised units (the same formulas as the training losses) ---
def huber_per_element(pred, true, delta=HUBER_DELTA):
    """keras.losses.Huber per element: 0.5*e^2 for |e| <= delta, delta*(|e| - delta/2) above."""
    err = np.abs(np.asarray(pred, np.float64) - true)
    small = np.minimum(err, delta)
    return 0.5 * small ** 2 + delta * (err - small)

def huber(pred, true, delta=HUBER_DELTA):
    return float(np.mean(huber_per_element(pred, true, delta)))

def mse(pred, true):
    return float(np.mean((np.asarray(pred, np.float64) - true) ** 2))

def per_dim_mse(pred, true, names, prefix):
    """MSE of each dimension separately: {"per_dim/<prefix>_mse/<name>": value}.
    In normalised units ~1.0 = no better than the train mean for that dimension."""
    err = np.mean((np.asarray(pred, np.float64) - true) ** 2, axis=0)   # (D,): mean over samples
    return {f"{PER_DIM}{prefix}_mse/{name}": float(e) for name, e in zip(names, err)}

class ImageMSE:
    """Image MSE added up batch by batch, so a whole split never exists as float."""
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, pred, true):
        diff = np.asarray(pred, np.float32) - true
        self.sum += float(np.sum(diff * diff, dtype=np.float64))
        self.count += diff.size

    def result(self):
        return self.sum / self.count


# --- C: errors in physical units ---
def quaternion_angle_deg(q_pred, q_true):
    """Angle (degrees) of the rotation between two orientations.
    Both are scaled to unit length first; |dot| makes q and -q the same rotation."""
    q1 = q_pred / np.maximum(np.linalg.norm(q_pred, axis=-1, keepdims=True), 1e-12)
    q2 = q_true / np.maximum(np.linalg.norm(q_true, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))

def physical_errors(pred_raw, true_raw, prefix):
    """Mean position (mm), rotation (deg) and jaw (raw units) error, per arm and both arms."""
    out, per_arm = {}, {"pos_mm": [], "rot_deg": [], "jaw_raw": []}
    for arm, s in ARMS.items():
        p = np.asarray(pred_raw[:, s:s + 8], np.float64)
        t = np.asarray(true_raw[:, s:s + 8], np.float64)
        errors = {"pos_mm": np.linalg.norm(p[:, POS] - t[:, POS], axis=1) * POSITION_TO_MM,
                  "rot_deg": quaternion_angle_deg(p[:, QUAT], t[:, QUAT]),
                  "jaw_raw": np.abs(p[:, JAW] - t[:, JAW])}
        for name, e in errors.items():
            out[f"{EXTRA}{prefix}_{name}_{arm}"] = float(e.mean())
            per_arm[name].append(e)
    for name, es in per_arm.items():
        out[f"{EXTRA}{prefix}_{name}"] = float(np.concatenate(es).mean())
    return out


# --- D: one function per model output (core always; "extra/" only if physical) ---
def action_metrics(pred, true, stats, physical, names):
    """names: the action dimension names (data.feature_names), for the per_dim/ metrics."""
    m = {"action_mse": mse(pred, true), "action_huber": huber(pred, true)}
    m.update(per_dim_mse(pred, true, names, "action"))
    if physical:
        m.update(physical_errors(unnormalise(pred, stats, "action"),
                                 unnormalise(true, stats, "action"), "action"))
    return m

def next_state_metrics(pred, true, stats, physical, names):
    """names: the state dimension names (data.feature_names), for the per_dim/ metrics."""
    key = "observation.state"
    m = {"next_state_mse": mse(pred, true)}
    m.update(per_dim_mse(pred, true, names, "next_state"))
    if physical:
        m.update(physical_errors(unnormalise(pred, stats, key),
                                 unnormalise(true, stats, key), "next_state"))
    return m

def state_delta_metrics(pred, state, true, factor):
    """E7: Huber on the change f = (next state - state) / factor, i.e. Δ / scale (the training
    loss), for the model and for f = 0 ("no change" = copy the state). All in normalised units."""
    true_f = (np.asarray(true, np.float64) - state) / factor
    pred_f = (np.asarray(pred, np.float64) - state) / factor
    return {"next_state_delta_huber": huber(pred_f, true_f),
            "next_state_delta_huber_zero": huber(np.zeros_like(true_f), true_f)}

def next_image_metrics(image_mse):
    """image_mse: a finished ImageMSE accumulator."""
    return {"next_image_mse": image_mse.result()}


# --- D2: action chunks (E8): offsets from a(t-1), option-C units ---
CHUNK_STEPS = (1, 2, 4, 8, 12, 16)   # steps reported one by one (1 = a(t)); "mean" = all k

def chunk_baselines(anchor_n, prev_n, stats):
    """No-learning predictions in target units, shape (N, k, D):
    stand-still (offset 0: a(t+i) = a(t-1)) and constant velocity
    (offset_i = (i+1) * (a(t-1) - a(t-2)))."""
    scale = chunk_scale_per_dim(stats)
    velocity = (unnormalise(anchor_n, stats, "action").astype(np.float64)
                - unnormalise(prev_n, stats, "action"))
    steps = np.arange(1, len(scale) + 1)[None, :, None]
    return {"standstill": np.zeros((len(anchor_n),) + scale.shape),
            "constvel": steps * velocity[:, None, :] / scale}

def group_name(names, dims):
    """Shared prefix of a group ("kuka_pose", "kuka_pose.ori"), or the single dim's name."""
    return names[dims[0]].rsplit(".", 1)[0] if len(dims) > 1 else names[dims[0]]

def chunk_metrics(pred, true, anchor_n, stats, names, steps=CHUNK_STEPS):
    """One predictor, (N, k, D) in target units. Huber/MSE in target units (the training
    loss) and per-group errors after reconstruction: offsets x scale + a(t-1), raw units.
    Quaternion groups -> degrees (renormalised by quaternion_angle_deg), other 3-D groups
    -> mm (POSITION_TO_MM assumption), anything else -> mean |error| in raw units."""
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    m = {"action_mse": mse(pred, true), "action_huber": huber(pred, true)}
    errors = {"huber": huber_per_element(pred, true).mean(axis=-1)}   # (N, k)
    scale = chunk_scale_per_dim(stats)
    off_pred, off_true = pred * scale, true * scale
    anchor = unnormalise(anchor_n, stats, "action").astype(np.float64)[:, None, :]
    quats = quaternion_starts(names)
    for dims in stats["action_chunk"]["groups"]:
        name = group_name(names, dims)
        p, t = off_pred[..., dims], off_true[..., dims]
        if len(dims) == 4 and dims[0] in quats:
            a = anchor[..., dims]
            errors[f"{name}_deg"] = quaternion_angle_deg(a + p, a + t)
        elif len(dims) == 3:
            errors[f"{name}_mm"] = np.linalg.norm(p - t, axis=-1) * POSITION_TO_MM
        else:
            errors[f"{name}_raw"] = np.abs(p - t).mean(axis=-1)
    for key, e in errors.items():
        for s in steps:
            if s <= e.shape[1]:
                m[f"chunk/{key}/step_{s:02d}"] = float(e[:, s - 1].mean())
        m[f"chunk/{key}/mean"] = float(e.mean())
    return m


# --- E: reporting ---
def format_table(columns):
    """{predictor: {metric: value}} -> text table, one metric per row, '-' if missing."""
    names = list(columns)
    rows = list(dict.fromkeys(k for c in columns.values() for k in c))
    w = max(len(r) for r in rows) + 2
    cw = max(12, max(len(n) for n in names) + 2)
    lines = ["metric".ljust(w) + "".join(n.rjust(cw) for n in names)]
    for r in rows:
        cells = [f"{columns[n][r]:.4f}" if r in columns[n] else "-" for n in names]
        lines.append(r.ljust(w) + "".join(c.rjust(cw) for c in cells))
    return "\n".join(lines)

def append_results(row, path=RESULTS_CSV):
    """Add one row to the shared results table. New columns are added; old rows get blanks."""
    path = Path(path)
    table = pd.DataFrame([{"date": datetime.now().isoformat(timespec="seconds"), **row}])
    if path.exists():
        table = pd.concat([pd.read_csv(path), table], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
