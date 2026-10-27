"""
look_static_vs_moving.py

Question: are E1 (image + WM), E2 (state-only) and E6 (image, WM off) wrong in the
same way on frames where the robot stands still and on frames where it moves?

Inference only, from each run's best.weights.h5 (nothing is trained or saved into the run).
  - static / moving: the label of scripts/look_standstill_baseline.py, unchanged
    (a(t) -> a(t+1) below STATIC_MM and STATIC_DEG, raw actions, float64)
  - errors: as in scripts/check_orientation_error.py (position mm, sign-invariant
    rotation deg); the all-frames mean is checked against the run's saved
    check_orientation_error.json, so these are the numbers of the diary
  - moving frames are split into 3 equal-size speed bins (slow / medium / fast) by
    speed(): the commanded pose change a(t) -> a(t+1)
  - per run: mean, median and p90 per phase; per group: mean +- std (ddof=1) over seeds

Outputs (outputs/analysis/static_moving/):
  per_run_<split>.csv    one row per run x phase
  summary_<split>.csv    one row per group x phase (mean, std over seeds)

Usage:
  python -m scripts.look_static_vs_moving                    # val, E1 / E2 / E6
  python -m scripts.look_static_vs_moving --max-episodes 2   # quick test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import data, metrics
from models.state_only import build_state_only_actor
from scripts.check_orientation_error import predict_actions
from scripts.look_standstill_baseline import STATIC_DEG, STATIC_MM, errors_at_h, load_actions
from scripts.train import OUTPUTS, build_model, dataset_root, run_quat_ref

OUT = Path("outputs/analysis/static_moving")
POS, QUAT = slice(0, 3), slice(3, 7)   # STAR-IL action: [x y z | qx qy qz qw]
SEEDS = (0, 1, 2)
GROUPS = {"E1 image + WM": "star-il-simplecnn-latent512-canonq-seed{}",
          "E6 image, WM off": "star-il-simplecnn-latent512-canonq-nowm-seed{}",
          "E2 state-only": "star-il-stateonly-h256-canonq-seed{}"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=data.ALLOWED_SPLITS)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--speed", default="position", choices=["position", "max-rank"],
                   help="speed measure for the slow / medium / fast bins")
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


SPEED_BINS = ("slow", "medium", "fast")


# --- A: commanded pose change a(t) -> a(t+1) per pair position t ---
def pose_change(split, loaded, root, max_episodes):
    """(position mm, rotation deg) of a(t) -> a(t+1), aligned with loaded["pair_index"],
    computed exactly as in look_standstill_baseline (raw actions, float64)."""
    _, actions = load_actions(split, root, max_episodes)
    t = loaded["pair_index"]
    ep = loaded["episode_index"]
    first = {e: np.flatnonzero(ep == e)[0] for e in np.unique(ep)}   # episode start in the flat array
    assert set(actions) == set(first), "standstill and load_split use different episodes"
    change = {e: errors_at_h(a, 1) for e, a in actions.items()}
    local = t - np.array([first[e] for e in ep[t]])
    dpos = np.array([change[e][0][i] for e, i in zip(ep[t], local)])
    drot = np.array([change[e][1][i] for e, i in zip(ep[t], local)])
    return dpos, drot, actions, local


# --- B: speed of the moving frames, and equal-size bins ---
def speed(dpos_mm, drot_deg, moving, measure):
    """One number per frame: how fast the commanded pose changes from t to t+1.
    Only the ORDER matters (the bins are tertiles of it), not the units.
      position  dpos_mm (main result: "mm per frame")
      max-rank  the larger of the two ranks (position, rotation) among the MOVING frames:
                no units and no mm-vs-deg exchange rate (robustness check)"""
    if measure == "position":
        return dpos_mm
    spd = np.full(len(dpos_mm), np.nan)   # static frames are never binned
    ranks = [pd.Series(d[moving]).rank().to_numpy() for d in (dpos_mm, drot_deg)]
    spd[moving] = np.maximum(*ranks)
    return spd

def speed_bins(spd, moving):
    """Equal-size bins of the moving frames, split on the speed RANK, so ties at a bin
    edge cannot make the bins unequal (as value quantiles could)."""
    idx = np.flatnonzero(moving)
    order = idx[np.argsort(spd[idx], kind="stable")]
    bins = {}
    for name, part in zip(SPEED_BINS, np.array_split(order, len(SPEED_BINS))):
        bins[name] = np.zeros_like(moving)
        bins[name][part] = True
    return bins


# --- C: predictions of one run, from its best weights ---
def predict(run, loaded, cfg):
    out = OUTPUTS / run
    if cfg.get("model_type") == "state_only":
        state_dim, action_dim = data.feature_dims(dataset_root(cfg))
        model = build_state_only_actor(state_dim, action_dim, cfg["hidden"])
        model.load_weights(str(out / "best.weights.h5"))
        return model.predict(loaded["states"][loaded["pair_index"]], batch_size=1024, verbose=0)
    model = build_model(cfg)
    model.load_weights(str(out / "best.weights.h5"))
    return predict_actions(model, loaded, cfg["batch_size"])


# --- D: errors per frame, as in check_orientation_error.measure ---
def frame_errors(pred_n, loaded, stats):
    true_n = loaded["actions"][loaded["pair_index"]]
    pred = data.unnormalise(pred_n, stats, "action").astype(np.float64)
    true = data.unnormalise(true_n, stats, "action").astype(np.float64)
    pos_mm = 1000 * np.linalg.norm(pred[:, POS] - true[:, POS], axis=1)
    rot_deg = metrics.quaternion_angle_deg(pred[:, QUAT], true[:, QUAT])
    return pos_mm, rot_deg, true


def main():
    args = parse_args()
    runs = {g: [p.format(s) for s in SEEDS] for g, p in GROUPS.items()}
    first = OUTPUTS / runs["E1 image + WM"][0]
    cfg0 = json.loads((first / "config.json").read_text())
    root = dataset_root(cfg0)
    stats = data.load_norm_stats(first / "norm_stats.json")
    quat_ref = run_quat_ref(cfg0, first)
    eps = None if args.max_episodes is None else data.get_split_episodes(args.split, root)[:args.max_episodes]
    loaded = data.load_split(args.split, root=root, stats=stats, episodes=eps, quat_ref=quat_ref)

    dpos, drot, raw_actions, local = pose_change(args.split, loaded, root, args.max_episodes)
    static = (dpos < STATIC_MM) & (drot < STATIC_DEG)   # the label of look_standstill_baseline
    spd = speed(dpos, drot, ~static, args.speed)
    bins = speed_bins(spd, ~static)
    phases = {"static": static, **bins, "moving": ~static, "all": np.ones_like(static)}
    print(f"{args.split}: {len(static)} pairs, {static.sum()} static ({static.mean():.1%}), "
          f"{(~static).sum()} moving")
    for name, sel in bins.items():   # what each bin means physically
        print(f"  {name:6s} {sel.sum():6d} frames: speed {spd[sel].min():.3f}-{spd[sel].max():.3f}, "
              f"median da {np.median(dpos[sel]):.3f} mm, {np.median(drot[sel]):.3f} deg per frame")

    rows = []
    for group, names in runs.items():
        for run in names:
            out = OUTPUTS / run
            cfg = json.loads((out / "config.json").read_text())
            # one loaded split serves every run only if they share the label convention
            assert (out / "norm_stats.json").read_text() == (first / "norm_stats.json").read_text(), run
            assert (out / "quat_reference.json").read_text() == (first / "quat_reference.json").read_text(), run

            pos, rot, true = frame_errors(predict(run, loaded, cfg), loaded, stats)
            if args.max_episodes is None:   # same numbers as the diary (check_orientation_error)
                saved = json.loads((out / "check_orientation_error.json").read_text())["results"]["val"]
                assert np.isclose(pos.mean(), 1000 * saved["pos_err_raw_mean"], rtol=1e-3), (run, pos.mean())
                assert np.isclose(rot.mean(), saved["rot_err_deg_mean"], rtol=1e-3), (run, rot.mean())
            for phase, sel in phases.items():
                row = {"group": group, "run": run, "seed": cfg["seed"], "phase": phase,
                       "n_frames": int(sel.sum())}
                for key, err in (("pos_mm", pos[sel]), ("rot_deg", rot[sel])):
                    row.update({f"{key}_mean": err.mean(), f"{key}_median": np.median(err),
                                f"{key}_p90": np.percentile(err, 90)})
                rows.append(row)
            print(f"  {run}: done")

    # the labels and the evaluated targets are the same actions (float32 vs float64)
    a_raw = np.concatenate([raw_actions[e][i][None] for e, i in
                            zip(loaded["episode_index"][loaded["pair_index"]], local)])
    assert np.allclose(a_raw[:, POS], true[:, POS], atol=1e-6), "labels not aligned with targets"

    per_run = pd.DataFrame(rows)
    metrics_cols = [f"{k}_{s}" for k in ("pos_mm", "rot_deg") for s in ("mean", "median", "p90")]
    summary = per_run.groupby(["group", "phase"], sort=False).agg(
        n_frames=("n_frames", "first"), n_seeds=("seed", "count"),
        **{f"{c}_{s}": (c, s) for c in metrics_cols for s in ("mean", "std")}).reset_index()

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = (f"_{args.split}_speed-{args.speed}"
              + (f"_max{args.max_episodes}" if args.max_episodes else ""))
    per_run.to_csv(args.out / f"per_run{suffix}.csv", index=False, float_format="%.5f")
    summary.to_csv(args.out / f"summary{suffix}.csv", index=False, float_format="%.5f")

    print(f"\nmean +- std over {len(SEEDS)} seeds (ddof=1); static = a(t)->a(t+1) "
          f"< {STATIC_MM} mm and < {STATIC_DEG} deg")
    for key, unit in (("pos_mm", "position error (mm)"), ("rot_deg", "rotation error (deg)")):
        cols = [f"{key}_{s}" for s in ("mean", "median", "p90")]
        print(f"\n{unit}\n{'model':18s} {'phase':7s} {'frames':>7s}"
              + "".join(s.rjust(15) for s in ("mean", "median", "p90")))
        for _, r in summary.iterrows():
            cells = "".join(f"{r[c + '_mean']:.3f}±{r[c + '_std']:.3f}".rjust(15) for c in cols)
            print(f"{r['group']:18s} {r['phase']:7s} {r['n_frames']:7d}{cells}")
    print(f"\nsaved to {args.out}/")


if __name__ == "__main__":
    main()
