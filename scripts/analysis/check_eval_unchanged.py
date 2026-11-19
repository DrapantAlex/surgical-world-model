"""Check that a change did not change any computed number: re-evaluate finished runs with
`scripts.train --eval-only` on CPU and compare every key of val_metrics_reeval.json exactly
against a baseline saved before the change.

  python -m scripts.analysis.check_eval_unchanged save <label> [run ...]      # before the change
  python -m scripts.analysis.check_eval_unchanged compare <label> [run ...]   # after it

Default runs: one E7 (delta), one E8 (chunk, untrimmed) and one 1c (chunk, trimmed, skip,
mixer) run, as in the 2026-11-19 refactor. Baselines: outputs/analysis/eval_unchanged/<label>/.
Exit code 1 if any value or key set differs. Side effect: like --eval-only itself, each run's
own val_metrics_reeval.json is overwritten.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from models.run_config import OUTPUTS, git_commit

# --- A: runs and folders ---
DEFAULT_RUNS = ["star-il-simplecnn-latent512-canonq-delta-seed0",
                "star-il-simplecnn-latent512-canonq-chunk16-seed0",
                "star-il-simplecnn-latent512-canonq-chunk16-trim-skip-mix3-val1000-seed0"]
BASELINES = OUTPUTS / "analysis" / "eval_unchanged"


# --- B: one re-evaluation ---
def reevaluate(run):
    """--eval-only of one run in its own process, on CPU (every evaluation on the same
    device, no TF state shared between runs); returns its val_metrics_reeval.json."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    done = subprocess.run([sys.executable, "-m", "scripts.train", "--run-name", run, "--eval-only"],
                          env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"{run}: --eval-only failed\n{done.stderr[-2000:]}")
    return OUTPUTS / run / "val_metrics_reeval.json"


def same_value(a, b):
    """Exact equality; NaN (a group without glitch-free pairs) equals NaN."""
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


# --- C: the two commands ---
def save(label, runs):
    folder = BASELINES / label
    if folder.exists():
        raise SystemExit(f"{folder} exists: choose a new label")
    folder.mkdir(parents=True)
    for run in runs:
        print(f"re-evaluating {run} ...", flush=True)
        shutil.copy(reevaluate(run), folder / f"{run}.json")
    (folder / "meta.json").write_text(json.dumps({"git_commit": git_commit(), "runs": runs}, indent=2))
    print(f"baseline {label!r} saved: {len(runs)} runs, commit {git_commit()}")


def compare(label, runs):
    folder = BASELINES / label
    meta = json.loads((folder / "meta.json").read_text())
    runs = runs or meta["runs"]
    ok = True
    for run in runs:
        print(f"re-evaluating {run} ...", flush=True)
        before = json.loads((folder / f"{run}.json").read_text())
        now = json.loads(reevaluate(run).read_text())
        differ = [k for k in before if k in now and not same_value(before[k], now[k])]
        same_keys = before.keys() == now.keys()
        ok &= same_keys and not differ
        print(f"  {len(before)} keys, same key set: {same_keys}, values that differ: {len(differ)}"
              + (f" {differ[:5]}" if differ else ""))
        if not same_keys:
            print(f"  only in baseline: {sorted(before.keys() - now.keys())[:5]}, "
                  f"only now: {sorted(now.keys() - before.keys())[:5]}")
    print(f"\nbaseline {label!r} (commit {meta['git_commit']}) vs now (commit {git_commit()}): "
          + ("UNCHANGED" if ok else "CHANGED"))
    if not ok:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["save", "compare"])
    p.add_argument("label")
    p.add_argument("runs", nargs="*", help="save: default the E7/E8/1c runs; compare: default the baseline's runs")
    args = p.parse_args()
    if args.command == "save":
        save(args.label, args.runs or DEFAULT_RUNS)
    else:
        compare(args.label, args.runs)


if __name__ == "__main__":
    main()
