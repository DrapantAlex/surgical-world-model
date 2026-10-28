"""
look_state_delta.py

Question (E7): how big is the one-frame state change Δ = s(t+1) - s(t), per dimension?
The E7 world-model target is Δ / max(median|Δ|, floor), so the floor must be chosen
from these numbers.

Raw units (no normalisation), pairs within an episode only, episodes without the
endoscope video dropped (same pairs as train.py). STAR-IL state has no quaternion,
so --canonical-quaternion would not change it.

Usage:
  python -m scripts.look_state_delta                    # train split (tables only)
  python -m scripts.look_state_delta --max-episodes 2   # quick test
"""
import argparse
from pathlib import Path

import numpy as np

from models import data

STAR_IL = Path("data/open_h/Surgical/jhu/imerse/star_il/star_il")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=STAR_IL)
    p.add_argument("--split", default="train", choices=data.ALLOWED_SPLITS)
    p.add_argument("--max-episodes", type=int, default=None)
    return p.parse_args()


# --- A: Δ per pair, raw units ---
def state_deltas(split, root, max_episodes=None):
    """(states (N, D), deltas (P, D)): every in-episode pair, float64."""
    episodes, _ = data.drop_missing_videos(split, data.get_split_episodes(split, root)[:max_episodes],
                                           root=root)
    states, deltas = [], []
    for ep in episodes:
        s, _ = data.load_table(ep, root)
        s = s.astype(np.float64)
        states.append(s)
        deltas.append(s[1:] - s[:-1])
    return np.concatenate(states), np.concatenate(deltas)


# --- B: table ---
def main():
    args = parse_args()
    states, d = state_deltas(args.split, args.dataset_root, args.max_episodes)
    names = data.feature_names(args.dataset_root)[0]
    a = np.abs(d)
    print(f"\n{args.split}: {len(d)} pairs, raw units\n")
    print(f"{'dim':<22}{'median|Δ|':>11}{'mean|Δ|':>11}{'std Δ':>11}{'p90|Δ|':>11}"
          f"{'Δ == 0':>8}{'std s':>10}")
    for i, n in enumerate(names):
        print(f"{n:<22}{np.median(a[:, i]):>11.2e}{a[:, i].mean():>11.2e}{d[:, i].std():>11.2e}"
              f"{np.percentile(a[:, i], 90):>11.2e}{(d[:, i] == 0).mean():>8.1%}"
              f"{states[:, i].std():>10.2e}")


if __name__ == "__main__":
    main()
