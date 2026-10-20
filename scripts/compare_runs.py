"""
compare_runs.py

Compare groups of runs (e.g. the same configuration with seeds 0, 1, 2) in
PHYSICAL units only: rotation error in degrees and position error in raw units.
Normalised MSE is never compared here, because it depends on the label
convention (--canonical-quaternion changes the label std).

Reads, for every run, outputs/<run>/config.json and the output of
scripts/check_orientation_error.py (outputs/<run>/check_orientation_error.json).

  1. Per run: rotation error (median / mean / p90) on the train subset and val,
     position error, val wrong-sign % and |q_pred| < 0.7 %, best epoch, train/val gap,
     and a 95% bootstrap CI of the val mean rotation error
  2. Per group: mean and min-max across runs (seeds)
  3. Per group: 95% bootstrap CI of the seed-averaged val mean rotation error
  4. Per group: the 5 worst val episodes (rotation error averaged over seeds)

Bootstrap: resample whole val EPISODES with replacement (frames of one episode
are strongly correlated), frame-weighted mean, B resamples, fixed seed.

Usage:
  python -m scripts.compare_runs \
      --group old  star-il-simplecnn-latent512-seed0 star-il-simplecnn-latent512-seed1 ... \
      --group canonq star-il-simplecnn-latent512-canonq-seed0 ...
"""
import argparse
import json

import numpy as np

from scripts.train import OUTPUTS

B = 10_000            # bootstrap resamples
BOOT_SEED = 0
# episodes A3 (2026-10-20) classified as "real rotation error" in the old reference run
REAL_ROTATION_EPISODES = {409, 428, 427, 447, 457}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--group", nargs="+", action="append", required=True,
                   metavar=("NAME", "RUN"), help="a group name followed by its run names")
    p.add_argument("--out", help="optional JSON file for the numbers")
    return p.parse_args()


# --------------------------------------------------------------------------
# Section 1 - one run
# --------------------------------------------------------------------------
def load_run(run):
    out = OUTPUTS / run
    cfg = json.loads((out / "config.json").read_text())
    check = json.loads((out / "check_orientation_error.json").read_text())
    return cfg, check

def val_episodes(check):
    """(episode ids, frames per episode, mean rotation error per episode), sorted by id."""
    eps = sorted(check["per_episode"]["val"].items(), key=lambda kv: int(kv[0]))
    if any("n_pairs" not in v for _, v in eps):
        raise SystemExit("check_orientation_error.json has no per-episode n_pairs: re-run "
                         "scripts.check_orientation_error for this run")
    ids = np.array([int(e) for e, _ in eps])
    n = np.array([v["n_pairs"] for _, v in eps], float)
    err = np.array([v["rot_err_deg"] for _, v in eps])
    return ids, n, err

def bootstrap_ci(n, err, rng):
    """95% CI of the frame-weighted mean error, resampling whole episodes."""
    idx = rng.integers(0, len(n), size=(B, len(n)))
    means = (n[idx] * err[idx]).sum(axis=1) / n[idx].sum(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()

def run_row(run, rng):
    cfg, check = load_run(run)
    tr, va = check["results"]["train_subset"], check["results"]["val"]
    _, n, err = val_episodes(check)
    return {
        "canonical": bool(cfg.get("canonical_quaternion", False)), "seed": cfg["seed"],
        "best_epoch": cfg.get("best_epoch"), "epochs_run": cfg.get("epochs_run"),
        "train_rot_median": tr["rot_err_deg_median"], "train_rot_mean": tr["rot_err_deg_mean"],
        "train_rot_p90": tr["rot_err_deg_p90"],
        "val_rot_median": va["rot_err_deg_median"], "val_rot_mean": va["rot_err_deg_mean"],
        "val_rot_p90": va["rot_err_deg_p90"],
        "val_rot_mean_ci95": bootstrap_ci(n, err, rng),
        "train_pos": tr["pos_err_raw_mean"], "val_pos": va["pos_err_raw_mean"],
        "val_wrong_sign_pct": va["wrong_sign_pct"],
        "val_qnorm_below_0.7_pct": va["q_pred_norm_below_0.7_pct"],
        # train/val gap, in physical units (comparable across label conventions)
        "gap_rot_ratio": va["rot_err_deg_mean"] / tr["rot_err_deg_mean"],
        "gap_pos_ratio": va["pos_err_raw_mean"] / tr["pos_err_raw_mean"],
    }


# --------------------------------------------------------------------------
# Section 2 - report
# --------------------------------------------------------------------------
COLUMNS = [("train_rot_median", "{:.2f}"), ("train_rot_mean", "{:.2f}"), ("train_rot_p90", "{:.2f}"),
           ("val_rot_median", "{:.2f}"), ("val_rot_mean", "{:.2f}"), ("val_rot_p90", "{:.2f}"),
           ("train_pos", "{:.5f}"), ("val_pos", "{:.5f}"),
           ("val_wrong_sign_pct", "{:.2f}"), ("val_qnorm_below_0.7_pct", "{:.2f}"),
           ("best_epoch", "{:.0f}"), ("epochs_run", "{:.0f}"),
           ("gap_rot_ratio", "{:.1f}"), ("gap_pos_ratio", "{:.1f}")]

def print_group(name, runs, rows):
    print(f"\n=== group {name}: {len(runs)} runs ===")
    w = max(len(c) for c, _ in COLUMNS) + 2
    cw = 17   # fits "[12.34, 56.78]" with spaces between cells
    header = "".join(f"seed {rows[r]['seed']}".rjust(cw) for r in runs)
    print("metric".ljust(w) + header + "mean".rjust(cw) + "min-max".rjust(2 * cw))
    for col, fmt in COLUMNS:
        vals = [rows[r][col] for r in runs]
        if any(v is None for v in vals):
            continue
        cells = "".join(fmt.format(v).rjust(cw) for v in vals)
        spread = f"{fmt.format(min(vals))} - {fmt.format(max(vals))}"
        print(col.ljust(w) + cells + fmt.format(np.mean(vals)).rjust(cw) + spread.rjust(2 * cw))
    cis = "".join(f"[{a:.2f}, {b:.2f}]".rjust(cw) for a, b in (rows[r]["val_rot_mean_ci95"] for r in runs))
    print("val_rot_mean CI95".ljust(w) + cis + "   (per run, episode bootstrap)")

def group_summary(name, runs, rng):
    """Seed-averaged per-episode error -> group CI and worst episodes."""
    per_run = [val_episodes(load_run(r)[1]) for r in runs]
    ids, n = per_run[0][0], per_run[0][1]
    for other_ids, other_n, _ in per_run[1:]:
        assert np.array_equal(ids, other_ids) and np.array_equal(n, other_n), "different val episodes"
    err = np.mean([e for _, _, e in per_run], axis=0)
    ci = bootstrap_ci(n, err, rng)
    mean = float((n * err).sum() / n.sum())
    print(f"seed-averaged val mean rotation error: {mean:.2f} deg, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"
          f" (episode bootstrap, B={B})")
    worst = np.argsort(-err)[:5]
    print("5 worst val episodes (seed-averaged rotation error):")
    for i in worst:
        tag = "  <- 'real rotation error' in A3" if ids[i] in REAL_ROTATION_EPISODES else ""
        per_seed = ", ".join(f"{e[i]:.1f}" for _, _, e in per_run)
        print(f"  ep {ids[i]}: {err[i]:.1f} deg (seeds: {per_seed}){tag}")
    return {"seed_avg_val_rot_mean": mean, "seed_avg_val_rot_mean_ci95": ci,
            "worst5": [int(ids[i]) for i in worst]}


def main():
    args = parse_args()
    rng = np.random.default_rng(BOOT_SEED)
    report = {}
    for name, *runs in args.group:
        rows = {r: run_row(r, rng) for r in runs}
        print_group(name, runs, rows)
        report[name] = {"runs": rows, **group_summary(name, runs, rng)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
