"""
look_frame_difference.py

Question: from which gap k does the change between frame t and frame t-k clearly
exceed the change between two frames where the robot is still (the image noise)?

Hypothesis: the tool moves ~0.3 mm/frame ~ 0.7 px, below one pixel at k = 1, so the
motion should stand out from the noise after a few frames. Prediction: k = 2.

STAR-IL val split (a subset: --max-episodes). Frames and actions via data.load_split,
normalisation stats from the train split, then unnormalise -> raw commands (metres +
quaternion). Motion is measured on the commands (the action is the Cartesian pose; the
state is joint angles). No canonical quaternions: the rotation angle is sign-invariant.

  d_k(t) = mean over all pixels and channels of |frame(t) - frame(t-k)|   (0-255)

A step s -> s+1 is "still" if position < 0.1 mm AND rotation < 0.1 deg (the thresholds
of look_standstill_baseline). A pair (t-k, t) is
  still (simple): every step from frame t-k to frame t is still;
  still (safe):   every step from frame t-k-3 to frame t+3 is still (video lags the
                  kinematics by ~100 ms ~ 3 frames); pairs whose window leaves the episode
                  are excluded from this rule;
  moving:         not still under the simple rule.
"Clearly visible" = mean d_k (moving) / mean d_k (still) >= 2.

Output: outputs/analysis/frame_difference/frame_difference_val_<n>ep.csv (n = episodes)

Usage:
  python -m scripts.look_frame_difference --max-episodes 2    # quick test
  python -m scripts.look_frame_difference                     # 50 episodes
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models import data, metrics
from scripts.look_standstill_baseline import POS, QUAT, STATIC_MM, STATIC_DEG, STAR_IL

OUT = Path("outputs/analysis/frame_difference")
GAPS = (1, 2, 4, 8, 16)   # the gaps k between the two frames
LAG = 3                   # video lag in frames (~100 ms at 30 fps, Open-H README)
CLEAR_RATIO = 2.0         # "clearly visible": moving at least twice the noise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=STAR_IL)
    p.add_argument("--max-episodes", type=int, default=50)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


# --- A: frames and raw commands of the val subset ---
def load_val(root, max_episodes):
    """The val split restricted to its first `max_episodes` episodes with a video.
    Returns (frames uint8 (N,224,224,3), raw actions float64 (N,7), episode_index (N,))."""
    train_eps, _ = data.drop_missing_videos("train", data.get_split_episodes("train", root),
                                            root=root, verbose=False)
    stats = data.compute_norm_stats(train_eps, root)        # train split only
    val_eps, _ = data.drop_missing_videos("val", data.get_split_episodes("val", root),
                                          root=root, verbose=False)
    split = data.load_split("val", root, stats=stats, episodes=val_eps[:max_episodes])
    # back to metres + quaternion (float32 round trip: ~1e-5 mm, far below 0.1 mm)
    actions = data.unnormalise(split["actions"], stats, "action").astype(np.float64)
    return split["frames"], actions, split["episode_index"]


# --- B: which steps s -> s+1 are moving (commands) ---
def moving_steps(actions):
    """actions: (T, 7) raw, one episode. Returns bool (T-1,): True = step s -> s+1 moves."""
    pos_mm = np.linalg.norm(np.diff(actions[:, POS], axis=0), axis=1) * metrics.POSITION_TO_MM
    rot_deg = metrics.quaternion_angle_deg(actions[:-1, QUAT], actions[1:, QUAT])
    still = (pos_mm < STATIC_MM) & (rot_deg < STATIC_DEG)
    return ~still


# --- C: is the window around a pair (t-k, t) still? ---
def still_pairs(moving, k, margin):
    """Classify every pair (t-k, t) of one episode, for t = k .. T-1.

    moving: bool (T-1,), moving[s] = step s -> s+1 moves (from B).
    margin: frames added on both sides of the pair. 0 = simple rule, LAG = safe rule.
    The window covers frames t-k-margin .. t+margin, i.e. steps s = t-k-margin .. t+margin-1
    (the step s goes from frame s to frame s+1).

    Returns (still, valid), both bool (T-k,), aligned with t = k .. T-1:
      valid: the whole window is inside the episode (always True for margin = 0);
      still: valid AND no moving step in the window.

    Trick: count[i] = number of moving steps among steps 0 .. i-1 (a cumulative sum,
    count[0] = 0). The number of moving steps in steps a .. b is count[b+1] - count[a],
    one subtraction per pair instead of a loop over the window.

    Example: T = 8 frames, 7 steps, only step 2 (frame 2 -> 3) moves.
      moving = [0 0 1 0 0 0 0]
      count  = [0 0 0 1 1 1 1 1]          (length T)
      Pair (3, 5), k = 2, t = 5:
        simple (margin 0): steps 3..4 -> count[5] - count[3] = 1 - 1 = 0 -> still
        safe   (margin 3): frames 0..8, but frame 8 does not exist (T-1 = 7) -> not valid
      Pair (3, 4), k = 1, t = 4:
        simple (margin 0): step 3 only -> count[4] - count[3] = 1 - 1 = 0 -> still
        safe   (margin 3): frames 0..7, steps 0..6 -> count[7] - count[0] = 1 - 0 = 1 -> moving
        The motion (step 2 -> 3) ends exactly at frame t-k = 3: with a 3-frame video lag
        it can still show up in the image, so the safe rule does not call this pair noise.
    """
    T = len(moving) + 1
    count = np.concatenate([[0], np.cumsum(moving)])   # count[i] = moving steps in 0 .. i-1
    t = np.arange(k, T)
    first = t - k - margin          # first step of the window
    last = t + margin - 1           # last step of the window (inclusive)
    valid = (first >= 0) & (last <= T - 2)              # T-2 = the last step
    # clip only so the indexing works; invalid pairs are thrown away by `valid`
    n_moving = count[np.clip(last + 1, 0, T - 1)] - count[np.clip(first, 0, T - 1)]
    return valid & (n_moving == 0), valid


# --- D: image difference d_k for every pair of one episode ---
def frame_difference(frames, k):
    """frames: (T, H, W, 3) uint8. Returns (T-k,) float64, entry i = d_k for pair (i, i+k),
    i.e. aligned with t = k .. T-1 like still_pairs.
    Cast to int16 BEFORE subtracting: uint8 wraps around (3 - 5 = 254)."""
    f = frames.astype(np.int16)
    return np.abs(f[k:] - f[:-k]).mean(axis=(1, 2, 3))


# --- E: one table row per k ---
def collect(frames, actions, episode_index):
    """For each k: d_k, simple-still, safe-still, safe-valid over all episodes (concatenated)."""
    per_k = {k: {"d": [], "simple": [], "safe": [], "safe_valid": []} for k in GAPS}
    for ep in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == ep)        # one episode is contiguous
        f, moving = frames[rows[0]:rows[-1] + 1], moving_steps(actions[rows])
        for k in GAPS:
            if len(rows) <= k:
                continue
            simple, _ = still_pairs(moving, k, 0)
            safe, safe_valid = still_pairs(moving, k, LAG)
            for key, v in [("d", frame_difference(f, k)), ("simple", simple),
                           ("safe", safe), ("safe_valid", safe_valid)]:
                per_k[k][key].append(v)
    return {k: {key: np.concatenate(v) for key, v in g.items()} for k, g in per_k.items()}


def summarise(per_k):
    def mean(x): return float(x.mean()) if len(x) else np.nan
    def median(x): return float(np.median(x)) if len(x) else np.nan
    rows = []
    for k, g in per_k.items():
        d, moving = g["d"], ~g["simple"]
        groups = {"moving": d[moving], "still_simple": d[g["simple"]], "still_safe": d[g["safe"]]}
        row = {"k": k, "n_pairs": len(d), "n_safe_valid": int(g["safe_valid"].sum())}
        row.update({f"n_{name}": len(x) for name, x in groups.items()})
        for name, x in groups.items():
            row[f"mean_{name}"], row[f"median_{name}"] = mean(x), median(x)
        for rule in ("simple", "safe"):
            row[f"ratio_{rule}"] = row["mean_moving"] / row[f"mean_still_{rule}"]
            row[f"ratio_median_{rule}"] = row["median_moving"] / row[f"median_still_{rule}"]
        rows.append(row)
    return pd.DataFrame(rows)


# --- F: reading ---
def smallest_clear_k(table, column):
    hits = table.loc[table[column] >= CLEAR_RATIO, "k"]
    return int(hits.iloc[0]) if len(hits) else None

def report(table):
    print("\nReading ('clearly visible' = ratio of means >= 2):")
    for rule in ("simple", "safe"):
        k = smallest_clear_k(table, f"ratio_{rule}")
        k_med = smallest_clear_k(table, f"ratio_median_{rule}")
        print(f"  {rule:6s}: smallest k = {k if k else f'none up to {GAPS[-1]}'}"
              f"   (ratio of medians: {k_med if k_med else 'none'})")
    print("  noise (mean d of still pairs), simple vs safe:")
    for _, r in table.iterrows():
        print(f"    k={int(r.k):2d}: {r.mean_still_simple:.3f} vs {r.mean_still_safe:.3f}")


def main():
    args = parse_args()
    frames, actions, episode_index = load_val(args.dataset_root, args.max_episodes)
    n_eps = len(np.unique(episode_index))
    print(f"{n_eps} val episodes, {len(frames)} frames")
    table = summarise(collect(frames, actions, episode_index))
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(table.T)
    report(table)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"frame_difference_val_{n_eps}ep.csv"
    table.to_csv(path, index=False)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
