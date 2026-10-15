"""File 1: how good are trivial predictors? Run this BEFORE any training.

Measures four "no learning" predictors on the validation pairs, with the same
metric code as scripts/train.py (models/metrics.py), so their numbers can be
compared directly with a trained model. Needs no TensorFlow and no GPU.

  python -m scripts.baselines                    # full validation split
  python -m scripts.baselines --max-episodes 2   # quick check (not added to results.csv)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from models import data, metrics

CHUNK = 64   # pairs per chunk for the image baseline, keeps the float images small


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-episodes", type=int,
                   help="use only the first N validation episodes (smoke test)")
    p.add_argument("--out", help="default: outputs/baselines (outputs/baselines_smoke with --max-episodes)")
    return p.parse_args()


def main():
    args = parse_args()
    smoke = args.max_episodes is not None
    out = Path(args.out or ("outputs/baselines_smoke" if smoke else "outputs/baselines"))
    out.mkdir(parents=True, exist_ok=True)

    # --- A: data: statistics from the TRAINING split, pairs from the VALIDATION split ---
    stats = data.compute_norm_stats(data.get_split_episodes("train"))
    data.save_norm_stats(stats, out / "norm_stats.json")
    episodes = data.get_split_episodes("val")[:args.max_episodes] if smoke else None
    val = data.load_split("val", stats=stats, episodes=episodes)
    t = val["pair_index"]
    state_t, state_t1, action_t = val["states"][t], val["states"][t + 1], val["actions"][t]

    # --- B: the four predictors (all in normalised units, like the model) ---
    results = {}

    # "mean action": always the training mean action (0 after normalisation)
    mean_action = data.normalise(np.asarray(stats["action"]["mean"], np.float32), stats, "action")
    results["mean_action"] = metrics.action_metrics(
        np.broadcast_to(mean_action, action_t.shape), action_t, stats)

    # "action = current state": state t, converted from state units to action units
    state_as_action = data.normalise(
        data.unnormalise(state_t, stats, "observation.state"), stats, "action")
    results["action_eq_state"] = metrics.action_metrics(state_as_action, action_t, stats)

    # "copy state": state t+1 = state t
    results["copy_state"] = metrics.next_state_metrics(state_t, state_t1, stats)

    # "copy frame": frame t+1 = frame t, through get_pairs = the model's own preprocessing
    image = metrics.ImageMSE()
    for i in range(0, len(t), CHUNK):
        inputs, targets = data.get_pairs(val, t[i:i + CHUNK])
        image.update(inputs["image_input"], targets["next_image_output"])
    results["copy_frame"] = {"image_mse": image.result()}

    # --- C: report and save ---
    print(f"\nValidation pairs: {len(t)}  (position unit -> mm: x{metrics.POSITION_TO_MM:g})\n")
    print(metrics.format_table(results))
    summary = {"split": "val",
               "episodes": np.unique(val["episode_index"]).tolist(),
               "n_pairs": int(len(t)),
               "position_to_mm": metrics.POSITION_TO_MM,
               "results": results}
    (out / "baselines.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out / 'baselines.json'}")

    if smoke:
        print("smoke run: not added to results.csv")
    else:
        for name, m in results.items():
            metrics.append_results({"run_name": f"baseline:{name}", "encoder": "baseline", **m})
        print(f"added {len(results)} rows to {metrics.RESULTS_CSV}")


if __name__ == "__main__":
    main()
