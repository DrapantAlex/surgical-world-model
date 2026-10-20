"""
check_quaternion_sign.py

Question: is the sign of the action quaternion consistent across the episodes of
a split, or do physically similar orientations appear as q in one episode and
-q in another? (q and -q are the same rotation, but far apart for an MSE loss.)
With --canonical-quaternion the labels are read exactly as training reads them
(models/data.py D1), to check that canonicalisation removed the inconsistency.
The reference rotation always comes from the TRAIN split, also for --split val.

Four pieces of evidence, all from the parquet tables (no videos):
  1. Within-episode flips - consecutive frames with dot(q_t, q_t+1) < 0
  2. Across-episode flips - each episode's nearest episode in ROTATION (|dot|,
                            sign ignored): does it have the OPPOSITE sign (dot < 0)?
  3. Variance from sign   - per-component std before / after putting every
                            quaternion in the hemisphere of one reference rotation
  4. Real rotation spread - angle of each episode (and the largest of any frame)
                            from the reference rotation

Usage:
  python -m scripts.check_quaternion_sign
  python -m scripts.check_quaternion_sign --max-episodes 50
  python -m scripts.check_quaternion_sign --canonical-quaternion --split val
"""
import argparse
from pathlib import Path

import numpy as np

from models import data

STAR_IL = "data/open_h/Surgical/jhu/imerse/star_il/star_il"
CLOSE_DEG = 10.0   # two episodes count as "the same orientation" below this angle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=STAR_IL)
    p.add_argument("--quat-start", type=int, default=3,
                   help="first action index of the quaternion [x y z w] (STAR-IL: 3)")
    p.add_argument("--split", choices=data.ALLOWED_SPLITS, default="train")
    p.add_argument("--canonical-quaternion", action="store_true",
                   help="read the labels with canonical signs, as train.py --canonical-quaternion")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="first N episodes of the split (the reference always uses all of train)")
    return p.parse_args()


# --------------------------------------------------------------------------
# Section 1 - helpers
# --------------------------------------------------------------------------
def unit(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def angle_deg(abs_dot):
    """Rotation angle between two orientations, from |dot| of their unit quaternions."""
    return np.degrees(2.0 * np.arccos(np.clip(abs_dot, 0.0, 1.0)))

def average_quaternion(qs):
    """Markley average: the eigenvector of sum(q q^T) with the largest eigenvalue.
    q q^T is the same for q and -q, so the average does not depend on the signs."""
    _, vecs = np.linalg.eigh(qs.T @ qs)
    return vecs[:, -1]


# --------------------------------------------------------------------------
# Section 2 - the reference (always from train) and the action quaternions of the split
# --------------------------------------------------------------------------
def split_episodes(split, root):
    """The episodes training actually uses (those with a video)."""
    return data.drop_missing_videos(split, data.get_split_episodes(split, root),
                                    root=root, verbose=False)[0]

def main():
    args = parse_args()
    root = Path(args.dataset_root)
    # the same reference as train.py --canonical-quaternion (all train episodes)
    quat_ref = data.compute_quat_reference(split_episodes("train", root), root)
    entry = next(e for e in quat_ref["action"] if e["start"] == args.quat_start)
    ref = np.asarray(entry["q"])
    episodes = split_episodes(args.split, root)[:args.max_episodes]
    labels = "canonical" if args.canonical_quaternion else "stored (dataset)"
    print(f"\n{len(episodes)} {args.split} episodes, {labels} signs, quaternion = {entry['names']}\n")
    names = entry["names"]

    per_episode, frames, inner_flips = [], [], 0
    for ep in episodes:
        _, actions = data.load_table(ep, root, quat_ref if args.canonical_quaternion else None)
        q = unit(actions[:, args.quat_start:args.quat_start + 4].astype(np.float64))
        inner_flips += int(np.sum(np.sum(q[1:] * q[:-1], axis=1) < 0))
        aligned = q * np.sign(q @ q[0])[:, None]     # same hemisphere as frame 0
        per_episode.append(unit(aligned.mean(axis=0)))
        frames.append(q)
    E, frames = np.stack(per_episode), np.concatenate(frames)

    # ----------------------------------------------------------------------
    # Section 3 - the four pieces of evidence
    # ----------------------------------------------------------------------
    print(f"1. within-episode sign flips: {inner_flips} over {len(frames)} frames")

    dots = E @ E.T
    np.fill_diagonal(dots, 0.0)
    nn = np.argmax(np.abs(dots), axis=1)              # nearest episode, sign ignored
    nn_dot = dots[np.arange(len(E)), nn]
    close = angle_deg(np.abs(nn_dot)) < CLOSE_DEG
    opposite = close & (nn_dot < 0)
    print(f"2. episodes whose nearest episode is within {CLOSE_DEG:g} deg: {close.sum()}"
          f"\n   ... and with the OPPOSITE sign: {opposite.sum()}"
          f" ({100 * opposite.sum() / max(close.sum(), 1):.0f}%)")

    canon = frames * np.sign(frames @ ref)[:, None]   # every frame in ref's hemisphere
    print(f"3. per-component std over all frames (train reference q = {np.round(ref, 3)}):")
    w = max(len(n) for n in names) + 2
    print(f"   {'':{w}s}{'loaded':>10s}{'canonical':>12s}")
    for i, n in enumerate(names):
        print(f"   {n:{w}s}{frames[:, i].std():10.3f}{canon[:, i].std():12.3f}")
    print(f"   frames with w < 0: {100 * np.mean(frames[:, 3] < 0):.0f}%,"
          f" with |w| < 0.1: {100 * np.mean(np.abs(frames[:, 3]) < 0.1):.0f}%")
    print(f"   frames on the OPPOSITE side of the reference (dot < 0): {int(np.sum(frames @ ref < 0))}")

    ang = angle_deg(np.abs(E @ ref))
    frame_ang = angle_deg(np.abs(frames @ ref))
    print(f"4. episode angle from the reference rotation (deg): "
          f"median {np.median(ang):.1f}, 90th pct {np.percentile(ang, 90):.1f}, max {ang.max():.1f}"
          f"\n   largest angle of any frame: {frame_ang.max():.1f} deg"
          f" (safe limit {data.QUAT_SAFE_DEG:g}, sign ambiguous at 180)")


if __name__ == "__main__":
    main()
