"""
look_delta_predictability.py

Question (E7 analysis, D2): can the one-frame state change be predicted from the
world model's non-image inputs at all? E7's head never beat f = 0 on val (D1).
If no probe here beats f = 0 either, the single-frame Δ is (mostly) unpredictable
noise; if one does, the target is learnable and E7's problem is elsewhere (D3).

Labels only (no frames). Same preprocessing as the E7 run: its quaternion reference,
norm stats and Δ scale. Target f = Δ / scale (the E7 training target).
Pairs: t with t-1 and t+1 in the same episode (one pair per episode fewer than E7,
so f = 0 is recomputed on exactly these pairs).

  references (no learning): f = 0; constant velocity f(t) = f(t-1)
  feature sets:  s,a     state + action (what the WM sees, minus the image)
                 s,a,da  + last action change a(t) - a(t-1)
                 s,a,ds  + last state change f(t-1)
  probes: ridge (closed form, MSE) and a small MLP (2 x 256, Huber, fixed epochs,
          last epoch reported: val is never used to choose anything)
  score: val Huber, as the E7 loss

Outputs: outputs/analysis/state_delta/predictability.csv

Usage:
  python -m scripts.look_delta_predictability                    # train + val
  python -m scripts.look_delta_predictability --max-episodes 2   # quick test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import data, metrics
from scripts.look_delta_reference import E7_RUN
from scripts.look_state_delta import STAR_IL

OUT = Path("outputs/analysis/state_delta")
FEATURE_SETS = ("s,a", "s,a,da", "s,a,ds")
RIDGE_LAMBDA = 1.0                      # on z-scored features: a tiny regulariser
MLP_EPOCHS, MLP_BATCH, MLP_LR = 20, 512, 1e-3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=STAR_IL)
    p.add_argument("--run", type=Path, default=E7_RUN)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --- A: per-episode features and targets (E7 preprocessing) ---
def episode_features(s_n, a_n, f_all, feature_set):
    """One episode. s_n (T, 8) and a_n (T, 7): normalised state / action;
    f_all (T-1, 8): f_all[t] = (s[t+1] - s[t]) / scale, the change *from* t.
    Returns X (T-2, F) for the pairs t = 1 .. T-2, row i <-> t = i + 1,
    aligned with the target y = f_all[1:] built in load_split_xy()."""
    cols = [s_n[1:-1], a_n[1:-1]]                  # s(t), a(t) for t = 1 .. T-2
    if feature_set == "s,a,da":
        cols.append(a_n[1:-1] - a_n[:-2])          # a(t) - a(t-1)
    elif feature_set == "s,a,ds":
        cols.append(f_all[:-1])                    # f(t-1): the change from t-1 to t
    elif feature_set != "s,a":
        raise ValueError(f"unknown feature set {feature_set!r}")
    return np.hstack(cols)


def load_split_xy(split, root, stats, quat_ref, scale, max_episodes):
    """{feature set: X}, y, f_prev (= the constant-velocity prediction)."""
    episodes, _ = data.drop_missing_videos(split, data.get_split_episodes(split, root)[:max_episodes],
                                           root=root, verbose=False)
    xs, ys, prev = {k: [] for k in FEATURE_SETS}, [], []
    for ep in episodes:
        s, a = (x.astype(np.float64) for x in data.load_table(ep, root, quat_ref))
        s_n = data.normalise(s, stats, "observation.state")
        a_n = data.normalise(a, stats, "action")
        f_all = (s[1:] - s[:-1]) / scale
        for k in FEATURE_SETS:
            xs[k].append(episode_features(s_n, a_n, f_all, k))
        ys.append(f_all[1:])
        prev.append(f_all[:-1])
    return {k: np.concatenate(v) for k, v in xs.items()}, np.concatenate(ys), np.concatenate(prev)


# --- B: probes ---
def zscore(x_train, x_val):
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8
    return (x_train - mu) / sd, (x_val - mu) / sd

def ridge(x_train, y_train, x_val):
    x1 = np.hstack([x_train, np.ones((len(x_train), 1))])
    reg = RIDGE_LAMBDA * np.eye(x1.shape[1])
    reg[-1, -1] = 0.0                                   # no penalty on the bias
    w = np.linalg.solve(x1.T @ x1 + reg, x1.T @ y_train)
    return np.hstack([x_val, np.ones((len(x_val), 1))]) @ w

def mlp(x_train, y_train, x_val, seed):
    from tensorflow import keras                        # only here: ridge needs no TF
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input((x_train.shape[1],)),
                              keras.layers.Dense(256, activation="relu"),
                              keras.layers.Dense(256, activation="relu"),
                              keras.layers.Dense(y_train.shape[1])])
    model.compile(optimizer=keras.optimizers.Adam(MLP_LR), loss=keras.losses.Huber())
    model.fit(x_train.astype(np.float32), y_train.astype(np.float32),
              batch_size=MLP_BATCH, epochs=MLP_EPOCHS, verbose=0)
    return model.predict(x_val.astype(np.float32), batch_size=4096, verbose=0)


# --- C: table ---
def main():
    args = parse_args()
    stats = data.load_norm_stats(args.run / "norm_stats.json")
    quat_ref = data.load_quat_reference(args.run / "quat_reference.json")
    scale = np.asarray(stats["state_delta"]["scale"])
    load = lambda split: load_split_xy(split, args.dataset_root, stats, quat_ref, scale,
                                       args.max_episodes)
    (x_tr, y_tr, prev_tr), (x_va, y_va, prev_va) = load("train"), load("val")

    rows = [{"features": "-", "probe": "f = 0", "train": metrics.huber(np.zeros_like(y_tr), y_tr),
             "val": metrics.huber(np.zeros_like(y_va), y_va)},
            {"features": "f(t-1)", "probe": "constant velocity", "train": metrics.huber(prev_tr, y_tr),
             "val": metrics.huber(prev_va, y_va)}]
    for k in FEATURE_SETS:
        a, b = zscore(x_tr[k], x_va[k])
        for name in ("ridge", "mlp"):
            # one fit per probe; it predicts train and val stacked, then split again
            probe = ridge if name == "ridge" else lambda *x: mlp(*x, args.seed)
            both = probe(a, y_tr, np.vstack([a, b]))
            p_tr, p_va = both[:len(a)], both[len(a):]
            rows.append({"features": k, "probe": name, "train": metrics.huber(p_tr, y_tr),
                         "val": metrics.huber(p_va, y_va)})
            print(f"done: {k} / {name}")

    df = pd.DataFrame(rows)
    df["val vs f=0"] = df["val"] / df.loc[0, "val"]
    print(f"\npairs: train {len(y_tr)}, val {len(y_va)}   (Huber on f = Δ / scale)\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.max_episodes is None:
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "predictability.csv", index=False)
        print(f"\nsaved {OUT / 'predictability.csv'}")


if __name__ == "__main__":
    main()
