"""
look_standstill_baseline.py

Question: how far does the commanded pose drift in h frames? If the actor just
repeated a(t) ("stand still"), how wrong would it be at t+h, for h = 1..60?
And how much of the data is the robot not moving at all?

Train or val split (parquet tables, no videos), loaded exactly like train.py with
--canonical-quaternion: drop episodes without the endoscope video, quaternion
reference from the train split, load_table(ep, root, quat_ref). Raw units
(no normalisation): position in metres -> mm, rotation as the angle between
quaternions (sign-invariant, so the canonical sign does not change it).

Time = frame_index / 30 (the timestamps in STAR-IL are constant per episode).

Outputs (outputs/analysis/standstill/):
  standstill_vs_h_<split>.csv      per h: n_pairs, mean/median position (mm) and rotation (deg)
  static_per_episode_<split>.csv   per episode: frames, static frames, static fraction
  standstill_vs_h_<split>.png      error vs h, position and rotation side by side

Usage:
  python -m scripts.look_standstill_baseline                    # train
  python -m scripts.look_standstill_baseline --split val        # same frames as E1/E2 val
  python -m scripts.look_standstill_baseline --max-episodes 2   # quick test
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import data

STAR_IL = Path("data/open_h/Surgical/jhu/imerse/star_il/star_il")
OUT = Path("outputs/analysis/standstill")
POS, QUAT = slice(0, 3), slice(3, 7)   # STAR-IL action: [x y z | qx qy qz qw]
STATIC_MM, STATIC_DEG = 0.1, 0.1       # "static": a(t) -> a(t+1) below both


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=STAR_IL)
    p.add_argument("--split", default="train", choices=data.ALLOWED_SPLITS)
    p.add_argument("--max-h", type=int, default=60)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


# --- A: actions, loaded like train.py (canonical quaternion sign) ---
def load_actions(split, root, max_episodes=None):
    """(fps, {episode: actions (T, 7) raw, float64}). The quaternion reference always
    comes from the full train split, as in train.py, whatever the split or subset."""
    data.check_split(split)   # never the test split
    train_eps, _ = data.drop_missing_videos("train", data.get_split_episodes("train", root),
                                            root=root, verbose=False)
    quat_ref = data.compute_quat_reference(train_eps, root)
    episodes, _ = data.drop_missing_videos(split, data.get_split_episodes(split, root),
                                           root=root, verbose=False)
    episodes = episodes[:max_episodes] if max_episodes else episodes
    fps = data.load_info(root)["fps"]
    actions = {}
    for ep in episodes:
        frame_index = pd.read_parquet(data.table_path(ep, root), columns=["frame_index"])
        frame_index = frame_index["frame_index"].to_numpy()
        # t+h is the row h further down only if frames are contiguous: check, don't assume
        if not np.array_equal(frame_index, np.arange(len(frame_index))):
            raise ValueError(f"episode {ep}: frame_index is not 0..T-1")
        _, a = data.load_table(ep, root, quat_ref)
        actions[ep] = a.astype(np.float64)
    return fps, actions


# --- B: errors between a(t) and a(t+h) ---
def unit(q):
    return q / np.linalg.norm(q, axis=1, keepdims=True)

def rotation_deg(q1, q2):
    """Angle of the rotation from q1 to q2, in degrees, for unit quaternions [x y z w].
    2*atan2(|vec(r)|, |w(r)|) with r = conj(q1) q2: well-conditioned near 0, unlike
    2*arccos(|q1.q2|), whose error there is ~sqrt(machine eps) (~0.05 deg in float32,
    half the static threshold; harmless in float64, used here). |w| makes it sign-invariant."""
    v1, w1, v2, w2 = q1[:, :3], q1[:, 3:], q2[:, :3], q2[:, 3:]
    vec = w1 * v2 - w2 * v1 - np.cross(v1, v2)   # vector part of conj(q1) * q2
    w = np.sum(q1 * q2, axis=1)                  # scalar part of conj(q1) * q2
    return np.degrees(2 * np.arctan2(np.linalg.norm(vec, axis=1), np.abs(w)))

def errors_at_h(a, h):
    """(position mm, rotation deg) for every t of one episode with t+h in the episode."""
    pos_mm = 1000 * np.linalg.norm(a[h:, POS] - a[:-h, POS], axis=1)
    rot = rotation_deg(unit(a[:-h, QUAT]), unit(a[h:, QUAT]))
    return pos_mm, rot


# --- C: stand-still baseline per h, pooled over all frames of all episodes ---
def standstill_table(actions, max_h, fps):
    rows = []
    for h in range(1, max_h + 1):
        pos, rot = zip(*(errors_at_h(a, h) for a in actions.values() if len(a) > h))
        pos, rot = np.concatenate(pos), np.concatenate(rot)
        rows.append({"h": h, "seconds": h / fps, "n_pairs": len(pos),
                     "pos_mm_mean": pos.mean(), "pos_mm_median": np.median(pos),
                     "rot_deg_mean": rot.mean(), "rot_deg_median": np.median(rot)})
    return pd.DataFrame(rows)


# --- D: static frames (h = 1 below both thresholds) ---
def static_table(actions):
    rows = []
    for ep, a in actions.items():
        pos, rot = errors_at_h(a, 1)
        static = (pos < STATIC_MM) & (rot < STATIC_DEG)
        rows.append({"episode": ep, "n_frames": len(static), "n_static": int(static.sum()),
                     "static_fraction": static.mean()})
    return pd.DataFrame(rows)


# --- E: plot ---
def plot(table, split, path):
    colors = {"mean": "#2a78d6", "median": "#eb6834"}   # categorical slots 1, 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("pos_mm", "Position error (mm)"), ("rot_deg", "Rotation error (deg)")]
    for ax, (key, title) in zip(axes, panels):
        for stat, color in colors.items():
            y = table[f"{key}_{stat}"]
            ax.plot(table["h"], y, color=color, lw=2, label=stat)
            ax.annotate(f"{stat} {y.iloc[-1]:.2f}", (table["h"].iloc[-1], y.iloc[-1]),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color="#52514e")
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_xlabel("horizon h (frames)", color="#52514e")
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, table["h"].max() * 1.15)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        sec = ax.secondary_xaxis("top", functions=(lambda h: h / 30, lambda s: s * 30))
        sec.set_xlabel("seconds (frame_index / 30)", color="#52514e", fontsize=9)
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle(f"Stand-still baseline on STAR-IL {split}: error of a(t) as a prediction of a(t+h)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    args = parse_args()
    fps, actions = load_actions(args.split, args.dataset_root, args.max_episodes)
    assert fps == 30, f"fps {fps}: the plot's seconds axis assumes 30"
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.split}" + (f"_max{args.max_episodes}" if args.max_episodes else "")

    table = standstill_table(actions, args.max_h, fps)
    static = static_table(actions)
    table.to_csv(args.out / f"standstill_vs_h{suffix}.csv", index=False, float_format="%.5f")
    static.to_csv(args.out / f"static_per_episode{suffix}.csv", index=False, float_format="%.5f")
    plot(table, args.split, args.out / f"standstill_vs_h{suffix}.png")

    print(f"{len(actions)} {args.split} episodes, {sum(map(len, actions.values()))} frames")
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(table[table["h"].isin([1, 2, 5, 10, 15, 30, 45, 60])].to_string(index=False))
    frac = static["n_static"].sum() / static["n_frames"].sum()
    q = static["static_fraction"].quantile([0, .25, .5, .75, 1]).round(3).tolist()
    print(f"\nstatic frames (< {STATIC_MM} mm and < {STATIC_DEG} deg, t -> t+1): {frac:.1%}")
    print(f"per episode: min / q25 / median / q75 / max = {q}")
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
