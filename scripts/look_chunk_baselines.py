"""
look_chunk_baselines.py

Question (E8, before any training): how wrong are the no-learning chunk predictors on val,
step by step, in the exact units and metric the chunked actor will be measured with?
  - stand-still:       a(t+i) = a(t-1)                    (offset 0)
  - constant velocity: a(t+i) = a(t-1) + (i+1)(a(t-1) - a(t-2))
Both go through models/metrics.chunk_metrics, the function train.py will use for the model.

Labels only (no frames), the same preprocessing as a training run: train-only canonical
quaternions, norm stats and option-C chunk scale. Checks:
  - round trip: a(t-1) + y * scale reproduces a(t+i) (the targets lose nothing)
  - stand-still at step h = the diary's stand-still table at h (2026-10-23), up to the
    smaller pair set (1 <= t <= T-k instead of every t)

Outputs: outputs/analysis/chunk/baselines_val.csv (one row per predictor x step)

Usage:
  python -m scripts.look_chunk_baselines                    # val
  python -m scripts.look_chunk_baselines --max-episodes 2   # quick test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models import data, metrics
from scripts.look_state_delta import STAR_IL

OUT = Path("outputs/analysis/chunk")
DIARY_STANDSTILL = {1: (0.42, 0.09), 5: (1.92, 0.36), 10: (3.53, 0.67)}   # h: (mm, deg), val mean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=STAR_IL)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--floor-c", type=float, default=0.25)
    p.add_argument("--max-episodes", type=int, default=None)
    return p.parse_args()


# --- A: training-run preprocessing, val labels ---
def load_val(args):
    root = args.dataset_root
    train_eps, _ = data.drop_missing_videos("train", data.get_split_episodes("train", root),
                                            root=root, verbose=False)
    quat_ref = data.compute_quat_reference(train_eps, root)
    stats = data.compute_norm_stats(train_eps, root, quat_ref)
    stats["action_chunk"] = data.compute_chunk_scale(train_eps, root, quat_ref, args.chunk,
                                                     args.floor_c)
    episodes = data.get_split_episodes("val", root)[:args.max_episodes]
    val = data.load_split("val", root=root, stats=stats, quat_ref=quat_ref, episodes=episodes,
                          with_frames=False, chunk=args.chunk)
    return val, stats


# --- B: checks and table ---
def main():
    args = parse_args()
    val, stats = load_val(args)
    names = data.feature_names(args.dataset_root)[1]
    t = val["pair_index"]
    true = data.chunk_targets(val, t).astype(np.float64)   # the get_pairs target
    anchor_n, prev_n = data.chunk_anchors(val, t)

    # round trip in raw units
    scale = data.chunk_scale_per_dim(stats)
    raw = data.unnormalise(val["actions"], stats, "action").astype(np.float64)
    rebuilt = raw[t - 1][:, None] + true * scale
    err = np.abs(rebuilt - raw[t[:, None] + np.arange(args.chunk)]).max()
    assert err < 1e-5, f"round trip error {err}"
    print(f"\nround trip a(t-1) + y * scale -> a(t+i): max error {err:.1e} (raw units)")

    c = stats["action_chunk"]
    print(f"\noption-C scale (train), groups {[metrics.group_name(names, g) for g in c['groups']]}")
    for i in (0, 3, 7, 15):
        print(f"  step {i + 1:>2}: " + "  ".join(f"{v:.3e}" for v in c["scale"][i]))

    steps = tuple(range(1, args.chunk + 1))
    rows = []
    for who, pred in metrics.chunk_baselines(anchor_n, prev_n, stats).items():
        m = metrics.chunk_metrics(pred, true, anchor_n, stats, names, steps)
        for s in steps:
            rows.append({"predictor": who, "step": s,
                         **{k.split("/")[1]: v for k, v in m.items()
                            if k.startswith("chunk/") and k.endswith(f"step_{s:02d}")}})
    df = pd.DataFrame(rows)
    print(f"\nval: {len(t)} pairs, {len(set(val['episode_index'][t]))} episodes\n")
    print(df.pivot(index="step", columns="predictor").round(3).to_string())

    pos, rot = [k for k in df.columns if k.endswith("_mm")][0], [k for k in df.columns if k.endswith("_deg")][0]
    print("\nstand-still vs diary 2026-10-23 (val mean, every t):")
    for h, (mm, deg) in DIARY_STANDSTILL.items():
        r = df[(df.predictor == "standstill") & (df.step == h)].iloc[0]
        print(f"  h={h:>2}: {r[pos]:.2f} mm (diary {mm}), {r[rot]:.2f} deg (diary {deg})")
    if args.max_episodes is None:
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "baselines_val.csv", index=False)
        print(f"\nsaved {OUT / 'baselines_val.csv'}")


if __name__ == "__main__":
    main()
