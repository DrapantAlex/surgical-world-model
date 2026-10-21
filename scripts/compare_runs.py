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
  5. Optional --paired A B: paired episode test of A - B on val rotation and
     position (seed-averaged per-episode error), with each group's seed range
     and the verdict of the pre-registered rule (decision record, E6 in 4.2)

Bootstrap: resample whole val EPISODES with replacement (frames of one episode
are strongly correlated), frame-weighted mean, B resamples, fixed seed.

Usage:
  python -m scripts.compare_runs \
      --group old  star-il-simplecnn-latent512-seed0 star-il-simplecnn-latent512-seed1 ... \
      --group canonq star-il-simplecnn-latent512-canonq-seed0 ... \
      --paired canonq old
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
    p.add_argument("--paired", nargs=2, metavar=("A", "B"),
                   help="two group names: paired episode test of A - B (rotation and position)")
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


# --------------------------------------------------------------------------
# Section 3 - paired test between two groups (same val episodes)
# --------------------------------------------------------------------------
PAIRED_METRICS = [("rotation (deg)", "rot_err_deg", "val_rot_mean"),
                  ("position (raw)", "pos_err_raw", "val_pos")]

def seed_averaged(runs, key):
    """(episode ids, frames per episode, per-episode error averaged over the runs' seeds)."""
    per_run = []
    for r in runs:
        eps = sorted(load_run(r)[1]["per_episode"]["val"].items(), key=lambda kv: int(kv[0]))
        if any(key not in v for _, v in eps):
            raise SystemExit(f"{r}: no per-episode {key!r}: re-run scripts.check_orientation_error")
        per_run.append((np.array([int(e) for e, _ in eps]),
                        np.array([v["n_pairs"] for _, v in eps], float),
                        np.array([v[key] for _, v in eps])))
    ids, n = per_run[0][0], per_run[0][1]
    for other_ids, other_n, _ in per_run[1:]:
        assert np.array_equal(ids, other_ids) and np.array_equal(n, other_n), "different val episodes"
    return ids, n, np.mean([e for _, _, e in per_run], axis=0)

def compare_one_metric(range_a, range_b, ci):
    """The pre-registered rule for ONE metric (lower error = better).
    range_a, range_b: (min, max) over the seeds of each group; ci: 95% CI of A - B.
    Returns "A better", "B better", "overlap" or "unclear"."""
    # 1) seed ranges first: if they overlap (touching counts), training randomness alone
    #    could explain the difference, whatever the episode CI says
    if not (range_a[1] < range_b[0] or range_b[1] < range_a[0]):
        return "overlap"
    # 2) ranges separated: the paired CI must exclude 0 AND agree in direction
    if range_a[1] < range_b[0] and ci[1] < 0:
        return "A better"
    if range_b[1] < range_a[0] and ci[0] > 0:
        return "B better"
    return "unclear"   # separated, but the CI contains 0 or points the other way

def combined_verdict(a, b, per_metric):
    """Decision record 4.2: both metrics must agree for "helps"; one is enough for "hurts"."""
    outcomes = list(per_metric.values())
    if all(o == "A better" for o in outcomes):
        return f"{a} better on both metrics ({a} helps)"
    if all(o == "overlap" for o in outcomes):
        return "no effect (seed ranges overlap on both metrics)"
    if any(o == "B better" for o in outcomes):
        return f"{b} clearly better on at least one metric ({a} hurts)"
    return f"mixed / inconclusive: {per_metric}"

def paired_test(a, b, groups, rows, rng):
    print(f"\n=== paired test {a} - {b} (val, seed-averaged per episode, B={B}) ===")
    result = {}
    for label, key, row_key in PAIRED_METRICS:
        ids_a, n, err_a = seed_averaged(groups[a], key)
        ids_b, n_b, err_b = seed_averaged(groups[b], key)
        assert np.array_equal(ids_a, ids_b) and np.array_equal(n, n_b), "different val episodes"
        d = err_a - err_b
        # resampling whole episodes and averaging their DIFFERENCES = paired bootstrap
        mean_d, ci = float((n * d).sum() / n.sum()), bootstrap_ci(n, d, rng)
        ranges = {g: (min(rows[g][r][row_key] for r in groups[g]),
                      max(rows[g][r][row_key] for r in groups[g])) for g in (a, b)}
        outcome = compare_one_metric(ranges[a], ranges[b], ci)
        scale = 1000 if key == "pos_err_raw" else 1    # x1000 = mm IF the raw unit is metres
        unit = " (x1000: mm, assuming metres)" if scale != 1 else ""
        print(f"{label}{unit}: {a} seeds {ranges[a][0]*scale:.3f}-{ranges[a][1]*scale:.3f}, "
              f"{b} seeds {ranges[b][0]*scale:.3f}-{ranges[b][1]*scale:.3f}; "
              f"{a} - {b} = {mean_d*scale:+.3f}, 95% CI [{ci[0]*scale:+.3f}, {ci[1]*scale:+.3f}]"
              f"  -> {outcome}")
        result[key] = {"ranges": ranges, "mean_diff": mean_d, "ci95": ci, "outcome": outcome,
                       "n_episodes_a_better": int((d < 0).sum()), "n_episodes": int(len(d))}
    verdict = combined_verdict(a, b, {k: v["outcome"] for k, v in result.items()})
    print(f"verdict: {verdict}")
    return {**result, "verdict": verdict}


def main():
    args = parse_args()
    rng = np.random.default_rng(BOOT_SEED)
    report, groups, all_rows = {}, {}, {}
    for name, *runs in args.group:
        rows = {r: run_row(r, rng) for r in runs}
        print_group(name, runs, rows)
        report[name] = {"runs": rows, **group_summary(name, runs, rng)}
        groups[name], all_rows[name] = runs, rows
    if args.paired:
        a, b = args.paired
        report[f"paired {a} - {b}"] = paired_test(a, b, groups, all_rows, rng)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
