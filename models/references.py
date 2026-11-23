"""No-learning reference lines, measured on the same pairs as the model: copy_state (per
dim), copy_frame, constant velocity, stand-still and train-mean Huber (E8), and the glitch
rule (thresholds from the train split) with the per-group state report. Shared by training
(reference lines in every epoch's logs) and the final evaluation / --eval-only.
Moved here from scripts/train.py on 2026-11-19 (refactor step 4b), unchanged.
"""
import numpy as np
from tensorflow import keras

from models import data, metrics
from models.run_config import metric_name

# --- A: fixed choice of the glitch rule ---
GLITCH_G = 5           # glitch report (evaluation only): a one-frame jump > GLITCH_G x the train p99
                       # of |Δ| of that dim that comes back (glitch_frames); STAR-IL endo360: the
                       # 0.11-0.126 rad spikes of scripts/analysis/look_endo360.py


# --- B: reference lines and the glitch report ---
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

def wm_balance_references(train_split, state_dims):
    """NEW (c): the copy references the balanced WM losses are divided by (--wm-balance), from
    the TRAIN split's pairs: the image MSE of copy_frame and the state MSE of copy_state over
    `state_dims` only (the mean of the per-dim values: every dim has the same number of pairs).
    The same functions as the val reference lines, so 1.0 means "as good as copying"."""
    return {"image_copy_mse": copy_frame_mse(train_split),
            "state_copy_mse": float(np.mean(copy_state_mse_per_dim(train_split)[state_dims])),
            "state_dims": [int(d) for d in state_dims],
            "n_pairs": int(len(train_split["pair_index"]))}

def constvel_state(split, keep_ranges):
    """Constant velocity s(t) + (s(t) - s(t-1)) for every pair (normalised units); copy (Δ = 0)
    where s(t-1) is not available (data.prev_in_window). Returns (prediction, n_fallback_pairs)."""
    t, s = split["pair_index"], split["states"]
    ok = data.prev_in_window(split, keep_ranges)
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

def constvel_huber(val, stats):
    """Val action loss of constant velocity, offset_i = (i+1) x (a(t-1) - a(t-2)), in the
    option-C target units, with the compiled Huber() like standstill_huber. The prediction
    is metrics.chunk_baselines on data.chunk_anchors (velocity 0 where t-2 is not in t's
    episode): the same functions as chunk_baselines.json, so both always agree. The rule
    that uses the last command change (the --history experiment's actor bar)."""
    t = val["pair_index"]
    true = data.chunk_targets(val, t)
    pred = metrics.chunk_baselines(*data.chunk_anchors(val, t), stats)["constvel"]
    return float(keras.losses.Huber()(true, pred.astype(np.float32)))
