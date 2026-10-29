"""
look_delta_reference.py

Question (E7 analysis, D1): what Huber loss does the E7 state head get on train and
val without learning anything? E7's train state loss fell 3.5 -> 2.1 while val rose
3.1 -> 4.0 (f = 0 on val: 2.955). If f = 0 on train is ~2.9, the head learned
something train-only (memorisation); if it is ~2.1, train and val simply move differently.

Labels only (no frames), raw units. Target f = Δ / scale with the scale saved in the
E7 run's norm_stats.json, so this is exactly the training target
(Huber(next_state / k) = Huber(f, Δ / scale), see scripts/train.py scaled_huber).
  - zero:      f = 0 ("no motion" = copy the state)
  - train mean: f = per-joint mean of the train targets (best input-free constant)

Usage:
  python -m scripts.look_delta_reference                    # train + val
  python -m scripts.look_delta_reference --max-episodes 2   # quick test
"""
import argparse
import json
from pathlib import Path

import numpy as np

from models import data, metrics
from scripts.look_state_delta import STAR_IL, state_deltas

E7_RUN = Path("outputs/star-il-simplecnn-latent512-canonq-delta-seed0")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=STAR_IL)
    p.add_argument("--run", type=Path, default=E7_RUN, help="E7 run with state_delta in norm_stats.json")
    p.add_argument("--max-episodes", type=int, default=None)
    return p.parse_args()


# --- A: E7 targets f = Δ / scale, per split ---
def targets(split, root, scale, max_episodes):
    _, d = state_deltas(split, root, max_episodes)
    return d / scale


# --- B: reference losses, overall and per joint ---
def main():
    args = parse_args()
    scale = np.asarray(json.loads((args.run / "norm_stats.json").read_text())["state_delta"]["scale"])
    names = data.feature_names(args.dataset_root)[0]
    f = {s: targets(s, args.dataset_root, scale, args.max_episodes) for s in ("train", "val")}
    mean_f = f["train"].mean(axis=0)

    print(f"\nE7 target f = Δ / scale (scale from {args.run.name})\n")
    print(f"{'':<22}{'train f=0':>11}{'val f=0':>11}{'train mean':>12}{'val mean':>11}")
    for i, n in enumerate(names + ["ALL"]):
        col = slice(None) if n == "ALL" else slice(i, i + 1)
        row = [metrics.huber(np.zeros_like(f[s][:, col]), f[s][:, col]) for s in ("train", "val")]
        row += [metrics.huber(np.broadcast_to(mean_f[col], f[s][:, col].shape), f[s][:, col])
                for s in ("train", "val")]
        print(f"{n:<22}{row[0]:>11.3f}{row[1]:>11.3f}{row[2]:>12.3f}{row[3]:>11.3f}")
    print(f"\npairs: train {len(f['train'])}, val {len(f['val'])}"
          f"   (E7 logged val f=0: 2.955 on the full split)")


if __name__ == "__main__":
    main()
