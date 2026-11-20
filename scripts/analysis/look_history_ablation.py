"""History inputs: which half of h(t) = [Δs(t), Δa(t-1)] do the trained models use?
Eval-only ablation on the best weights of --history runs: the same evaluate() as training and
--eval-only, on four versions of the val history input:
  full      the run's own history (must reproduce val_metrics.json, see below)
  no_ds     Δs(t) set to 0 (the first state_dim columns)
  no_da     Δa(t-1) set to 0 (the last action_dim columns)
  none      both set to 0
0 is the fallback value of the history ("change unknown"), so the model has seen it on
train pairs, but zeroing a change is not the same as removing it: the drop is how much the
model relies on that input, an upper bound on what it gets from it. Nothing is trained.

Check of "full": every key of val_metrics.json within FULL_RTOL (1e-4) relative, not exactly.
GPU inference is not bit-reproducible: repeated evaluations of the same weights on the same
GPU differed by up to ~1.2e-5 relative (only the model-prediction keys; cuDNN may pick other
kernels each run), largest on the near-zero per-dim MSEs (e.g. one KUKA joint, ~7e-7). 1e-4
is still >= 1000x below the smallest ablation effect (~10%, no_da). Bit-exact checks are the
job of check_eval_unchanged.py, on CPU.

  python -m scripts.analysis.look_history_ablation                # the 3 history seeds
  python -m scripts.analysis.look_history_ablation <run> [<run> ...]

Saves outputs/analysis/history_ablation/<run>.json (the run folders stay untouched).
"""
import argparse
import json
import math

import numpy as np

from models import data
from models.evaluation import evaluate
from models.references import glitch_thresholds
from models.run_config import (OUTPUTS, build_model, dataset_root, delta_factor, episodes_for,
                               run_history, run_keep_ranges, run_quat_ref, save_json)

# --- A: runs, variants and the metrics shown ---
DEFAULT_RUNS = [f"star-il-simplecnn-latent512-canonq-chunk16-trim-skip-hist-val1000-seed{s}"
                for s in range(3)]
VARIANTS = {"full": (False, False), "no_ds": (True, False), "no_da": (False, True),
            "none": (True, True)}   # (zero Δs, zero Δa)
SHOWN = {"actor Huber": "action_huber",
         "Huber step 1": "chunk/huber/step_01", "Huber step 16": "chunk/huber/step_16",
         "pos mm (mean)": "chunk/kuka_pose_mm/mean", "ori deg (mean)": "chunk/kuka_pose.ori_deg/mean",
         "next_state_mse": "next_state_mse",
         "KUKA mse": "state_group/kuka_joint_pos/model_mse",
         "endo360 mse": "state_group/endo360_joint_pos/model_mse",
         "endo360 mse no glitch": "state_group/endo360_joint_pos/model_mse_noglitch",
         "image mse": "next_image_mse"}
OUT = OUTPUTS / "analysis" / "history_ablation"
FULL_RTOL = 1e-4   # "full" vs val_metrics.json, relative, every key (GPU: see the docstring)


# --- B: one run, four evaluations ---
def ablate(run):
    out = OUTPUTS / run
    cfg = json.loads((out / "config.json").read_text())
    if not run_history(cfg):
        raise SystemExit(f"{run} was trained without --history")
    stats = data.load_norm_stats(out / "norm_stats.json")
    root, quat_ref, keep_ranges = dataset_root(cfg), run_quat_ref(cfg, out), run_keep_ranges(cfg, out)
    # glitch thresholds from the run's own train tables, exactly as in --eval-only
    train_tables = data.load_split("train", root=root, stats=stats, quat_ref=quat_ref,
                                   episodes=episodes_for("train", cfg["max_episodes"], root),
                                   chunk=cfg.get("chunk"), keep_ranges=keep_ranges,
                                   with_frames=False)
    glitch_thr = glitch_thresholds(train_tables)
    val = data.load_split("val", root=root, stats=stats, quat_ref=quat_ref,
                          episodes=episodes_for("val", cfg["max_episodes"], root),
                          chunk=cfg.get("chunk"), keep_ranges=keep_ranges, history=True)
    model = build_model(cfg, stats)
    model.load_weights(str(out / "best.weights.h5"))
    state_dim = data.feature_dims(root)[0]

    results = {}
    for name, (zero_ds, zero_da) in VARIANTS.items():
        history = val["history"].copy()
        if zero_ds:
            history[:, :state_dim] = 0.0
        if zero_da:
            history[:, state_dim:] = 0.0
        print(f"{run}: {name} ...", flush=True)
        results[name] = evaluate(model, dict(val, history=history), stats, cfg["batch_size"],
                                 root, delta_factor(cfg, stats), keep_ranges, glitch_thr)
    return results


def rel_gap(a, b):
    """|a - b| relative to the larger value; 0 if equal (also both NaN or both 0)."""
    if a == b or (math.isnan(a) and math.isnan(b)):
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))

def check_full(run, full):
    """Every key of the run's val_metrics.json against the "full" evaluation, within FULL_RTOL."""
    saved = json.loads((OUTPUTS / run / "val_metrics.json").read_text())
    gaps = {k: rel_gap(saved[k], full[k]) for k in saved if k in full}
    worst = max(gaps, key=gaps.get)
    ok = saved.keys() == full.keys() and gaps[worst] <= FULL_RTOL
    print(f"  full vs val_metrics.json: {len(saved)} keys, same key set: {saved.keys() == full.keys()}, "
          f"{sum(g > 0 for g in gaps.values())} not bit-equal, largest relative gap "
          f"{gaps[worst]:.1e} ({worst}) -> {'OK' if ok else 'FAILED'} (tolerance {FULL_RTOL:g})")
    return ok


# --- C: tables ---
def table(rows, title):
    """rows: {variant: {shown name: value}}, plus the ratio to "full"."""
    print(f"\n{title}")
    print(f"{'metric':24s}" + "".join(f"{v:>12s}" for v in VARIANTS) + "".join(
        f"{v + '/full':>12s}" for v in list(VARIANTS)[1:]))
    for label in SHOWN:
        vals = [rows[v][label] for v in VARIANTS]
        print(f"{label:24s}" + "".join(f"{x:12.4g}" for x in vals)
              + "".join(f"{x / vals[0]:12.3f}" for x in vals[1:]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per_run, exact = {}, {}
    for run in args.runs:
        results = ablate(run)
        exact[run] = check_full(run, results["full"])
        save_json({"full_matches_val_metrics": exact[run], "full_rtol": FULL_RTOL,
                   "variants": results}, OUT / f"{run}.json")
        per_run[run] = {v: {k: results[v][key] for k, key in SHOWN.items()} for v in VARIANTS}
        table(per_run[run], run)
    if len(per_run) > 1:   # the seed mean of every number
        mean = {v: {k: float(np.mean([r[v][k] for r in per_run.values()])) for k in SHOWN}
                for v in VARIANTS}
        table(mean, f"mean of {len(per_run)} runs")
    print(f"\nfull reproduces val_metrics.json (rtol {FULL_RTOL:g}): {all(exact.values())} "
          f"({sum(exact.values())}/{len(exact)} runs); saved in {OUT}/")


if __name__ == "__main__":
    main()
