"""
check_orientation_error.py

Question: when the trained actor gets the orientation wrong (most of its val
action MSE), is it the right rotation with the wrong quaternion sign, an
"average" of +q and -q, or a really different rotation? And is the train/val
gap specific to orientation?

For the saved best weights of one run, on val and on a train subset
(inference only, nothing is trained or saved into the run's metrics):
  1. Per-dimension action MSE (normalised units), train vs val
  2. Position error (raw units) and sign-invariant rotation error (degrees)
  3. Wrong-sign rate: frames with dot(q_pred, q_true) < 0
  4. Norm of the predicted quaternion: well below 1 = averaging +q and -q
  5. Per episode: rotation error by sign group (+/- side of the train mean rotation)
  6. Sign-aligned MSE: the same MSE after flipping q_pred -> -q_pred where the sign
     is wrong = how much orientation error would remain without the sign problem

Usage:
  python -m scripts.check_orientation_error
  python -m scripts.check_orientation_error --train-episodes 2 --val-episodes 2   # quick test
"""
import argparse
import json

import numpy as np

from models import data, metrics
from models.pipeline import make_dataset
from scripts.check_quaternion_sign import average_quaternion, unit
from scripts.train import OUTPUTS, build_model, dataset_root, run_quat_ref

POS = slice(0, 3)   # STAR-IL action: [x y z | qx qy qz qw]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="star-il-simplecnn-latent512-seed0")
    p.add_argument("--quat-start", type=int, default=3)
    p.add_argument("--train-episodes", type=int, default=20,
                   help="train subset: this many episodes, evenly spaced over the split")
    p.add_argument("--val-episodes", type=int, default=None, help="default: the whole val split")
    return p.parse_args()


# --------------------------------------------------------------------------
# Section 1 - which episodes, and the train mean rotation (labels only)
# --------------------------------------------------------------------------
def evenly_spaced(episodes, n):
    """n episodes spread over the whole split (episode order may follow recording time)."""
    return [episodes[i] for i in np.linspace(0, len(episodes) - 1, n).round().astype(int)]

def train_reference(root, q):
    """Markley mean of every train action quaternion (parquet only, sign-independent)."""
    eps, _ = data.drop_missing_videos("train", data.get_split_episodes("train", root),
                                      root=root, verbose=False)
    return average_quaternion(np.concatenate(
        [unit(data.load_table(ep, root)[1][:, q].astype(np.float64)) for ep in eps]))


# --------------------------------------------------------------------------
# Section 2 - predict and measure one split
# --------------------------------------------------------------------------
def predict_actions(model, split, batch_size):
    out = [model(inputs, training=False)[0].numpy()
           for inputs, _ in make_dataset(split, batch_size, shuffle=False)]
    return np.concatenate(out)

def measure(model, split, stats, names, q, ref, batch_size):
    t = split["pair_index"]
    pred_n, true_n = predict_actions(model, split, batch_size), split["actions"][t]
    pred = data.unnormalise(pred_n, stats, "action").astype(np.float64)
    true = data.unnormalise(true_n, stats, "action").astype(np.float64)
    qp, qt = pred[:, q], true[:, q]

    angle = metrics.quaternion_angle_deg(qp, qt)          # |dot|: sign ignored
    wrong_sign = np.sum(qp * qt, axis=1) < 0
    norm = np.linalg.norm(qp, axis=1)
    side = np.sign(unit(qt) @ ref)                        # sign group of the label
    ep = split["episode_index"][t]

    m = {"n_pairs": int(len(t)), "action_mse": metrics.mse(pred_n, true_n),
         **metrics.per_dim_mse(pred_n, true_n, names, "action"),
         "pos_err_raw_mean": float(np.linalg.norm(pred[:, POS] - true[:, POS], axis=1).mean()),
         "rot_err_deg_mean": float(angle.mean()), "rot_err_deg_median": float(np.median(angle)),
         "rot_err_deg_p90": float(np.percentile(angle, 90)),
         "wrong_sign_pct": float(100 * wrong_sign.mean()),
         "q_pred_norm_median": float(np.median(norm)),
         "q_pred_norm_below_0.7_pct": float(100 * np.mean(norm < 0.7))}

    # 6: flip in RAW units (q and -q are the same rotation only before normalisation),
    # then normalise again so the MSE is comparable with the numbers above
    aligned = pred.copy()
    aligned[wrong_sign, q] = -aligned[wrong_sign, q]
    aligned_n = data.normalise(aligned, stats, "action")
    ori_mse, ori_aligned = metrics.mse(pred_n[:, q], true_n[:, q]), metrics.mse(aligned_n[:, q], true_n[:, q])
    m.update({"ori_mse": ori_mse, "ori_mse_sign_aligned": ori_aligned,
              "ori_mse_due_to_sign_pct": float(100 * (1 - ori_aligned / ori_mse)),
              "action_mse_sign_aligned": metrics.mse(aligned_n, true_n),
              **metrics.per_dim_mse(aligned_n, true_n, names, "action_sign_aligned")})

    for s, g in ((1, "+"), (-1, "-")):
        sel = side == s
        m[f"group{g}_frames_pct"] = float(100 * sel.mean())
        m[f"group{g}_rot_err_deg_mean"] = float(angle[sel].mean()) if sel.any() else float("nan")
        m[f"group{g}_wrong_sign_pct"] = float(100 * wrong_sign[sel].mean()) if sel.any() else float("nan")

    episodes = {}
    for e in np.unique(ep):
        sel = ep == e
        episodes[int(e)] = {"n_pairs": int(sel.sum()),   # weight for frame-level means
                            "group": "+" if side[sel].mean() > 0 else "-",
                            "rot_err_deg": float(angle[sel].mean()),
                            "wrong_sign_pct": float(100 * wrong_sign[sel].mean()),
                            "q_pred_norm": float(norm[sel].mean())}
    return m, episodes


# --------------------------------------------------------------------------
# Section 3 - run, report, save next to the run (not in val_metrics / results.csv)
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    out = OUTPUTS / args.run_name
    cfg = json.loads((out / "config.json").read_text())
    stats = data.load_norm_stats(out / "norm_stats.json")
    root = dataset_root(cfg)
    names = data.feature_names(root)[1]
    q = slice(args.quat_start, args.quat_start + 4)
    ref = train_reference(root, q)

    model = build_model(cfg)
    model.load_weights(str(out / "best.weights.h5"))

    train_eps, _ = data.drop_missing_videos("train", data.get_split_episodes("train", root),
                                            root=root, verbose=False)
    val_eps = None if args.val_episodes is None else data.get_split_episodes("val", root)[:args.val_episodes]
    splits = {"train_subset": ("train", evenly_spaced(train_eps, args.train_episodes)),
              "val": ("val", val_eps)}

    results, per_episode = {}, {}
    for name, (split, eps) in splits.items():
        loaded = data.load_split(split, root=root, stats=stats, episodes=eps,
                                 quat_ref=run_quat_ref(cfg, out))   # the run's own labels
        results[name], per_episode[name] = measure(model, loaded, stats, names, q, ref,
                                                   cfg["batch_size"])
        del loaded

    print(f"\nreference rotation (train mean) = {np.round(ref, 3)}\n")
    print(metrics.format_table(results))
    worst = sorted(per_episode["val"].items(), key=lambda kv: -kv[1]["rot_err_deg"])[:10]
    print("\nworst 10 val episodes by rotation error:")
    print(f"  {'ep':>4s} {'group':>5s} {'rot_deg':>8s} {'wrong_sign%':>12s} {'|q_pred|':>9s}")
    for e, v in worst:
        print(f"  {e:4d} {v['group']:>5s} {v['rot_err_deg']:8.1f} {v['wrong_sign_pct']:12.1f}"
              f" {v['q_pred_norm']:9.3f}")

    subset = "" if args.val_episodes is None else "_val_subset"   # quick tests: own file
    path = out / f"check_orientation_error{subset}.json"
    path.write_text(json.dumps({"reference_q": ref.tolist(), "train_subset_episodes":
                                splits["train_subset"][1], "results": results,
                                "per_episode": per_episode}, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
