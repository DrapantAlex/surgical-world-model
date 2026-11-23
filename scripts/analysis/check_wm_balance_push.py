"""Experiment (c): does the controller really hold each balanced WM term's push at RHO?

Independent of the controller's own logs: for a --wm-balance run, the best weights, the WM
weights logged at the best check (history.csv, wm_bal_w_*), and fresh batches (N_BATCHES x BATCH
pairs, seed SAMPLE_SEED) of the run's train and val pairs. Per batch, the encoder gradient norms
of the actor loss and of each balanced (normalised) WM loss, through the model's own term_losses
and models.balance.encoder_model, with this script's own gradient and norm code. Reported per term:
  mean      the mean of the per-batch ratios w_k‖g_k‖ / ‖g_a‖
  median    their median
  of_means  w_k mean‖g_k‖ / mean‖g_a‖ (what the EMA controller targets)
  summed    w_k‖Σ g_k‖ / ‖Σ g_a‖ (the gradients summed over the batches)
PASS if of_means on TRAIN (the controller measures on train batches) lies in BAND x RHO / 0.3
(BAND = 0.2-0.45 for RHO = 0.3). Exit 1 otherwise. TF32 off (see look_encoder_gradients).

  python -m scripts.analysis.check_wm_balance_push <run>
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from models import data
from models.balance import encoder_model
from models.run_config import (OUTPUTS, build_model, dataset_root, episodes_for, run_history,
                               run_keep_ranges, run_quat_ref, run_wm_balance, save_json)

# --- A: fixed choices ---
N_BATCHES, BATCH, SAMPLE_SEED = 20, 32, 0
BAND = (0.2, 0.45)          # for RHO = 0.3; scaled with RHO for other targets


# --- B: per-batch encoder norms ---
def batch_norms(model, enc_vars, x, y):
    with tf.GradientTape(persistent=True) as tape:
        L = model.term_losses(y, model(x, training=True))
    flat = lambda gs: np.concatenate([np.zeros(v.shape).ravel() if g is None else
                                      g.numpy().astype(np.float64).ravel()
                                      for v, g in zip(enc_vars, gs)])
    g = {"actor": flat(tape.gradient(L["actor"], enc_vars))}
    for k in model.bal_terms:
        g[k] = flat(tape.gradient(L[f"{k}_norm"], enc_vars))
    del tape
    return g


def measure(model, split, idx, w):
    enc_vars = list(encoder_model(model).trainable_variables)
    norms, summed = [], None
    for b in range(N_BATCHES):
        x, y = data.get_pairs(split, idx[b * BATCH:(b + 1) * BATCH])
        g = batch_norms(model, enc_vars, x, y)
        norms.append({k: float(np.linalg.norm(v)) for k, v in g.items()})
        summed = g if summed is None else {k: summed[k] + g[k] for k in g}
    out = {}
    for k in model.bal_terms:
        r = np.array([w[k] * n[k] / n["actor"] for n in norms])
        out[k] = {"mean": float(r.mean()), "median": float(np.median(r)),
                  "min": float(r.min()), "max": float(r.max()),
                  "of_means": float(w[k] * np.mean([n[k] for n in norms])
                                    / np.mean([n["actor"] for n in norms])),
                  "summed": float(w[k] * np.linalg.norm(summed[k]) / np.linalg.norm(summed["actor"]))}
    return out


# --- C: main ---
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    args = p.parse_args()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    out = OUTPUTS / args.run
    cfg = json.loads((out / "config.json").read_text())
    rho = run_wm_balance(cfg)
    if rho is None:
        raise SystemExit(f"{args.run} is not a --wm-balance run")
    stats = data.load_norm_stats(out / "norm_stats.json")
    root = dataset_root(cfg)
    model = build_model(cfg, stats)
    model.load_weights(str(out / "best.weights.h5"))
    hist = pd.read_csv(out / "history.csv")
    best = hist.iloc[cfg["best_epoch"] - 1]
    w = {k: float(best[f"wm_bal_w_{k}"]) for k in model.bal_terms}
    for k in model.bal_terms:   # the weights in effect at the best check
        model.v[k]["w"].assign(w[k])
    rng = np.random.default_rng(SAMPLE_SEED)
    res = {"rho": rho, "best_check": int(cfg["best_epoch"]), "weights_at_best": w}
    lo, hi = BAND[0] * rho / 0.3, BAND[1] * rho / 0.3
    for s in ("train", "val"):
        split = data.load_split(s, root=root, stats=stats, quat_ref=run_quat_ref(cfg, out),
                                episodes=episodes_for(s, cfg["max_episodes"], root),
                                chunk=cfg.get("chunk"), keep_ranges=run_keep_ranges(cfg, out),
                                history=run_history(cfg))
        idx = rng.choice(split["pair_index"], min(N_BATCHES * BATCH, len(split["pair_index"])),
                         replace=False)
        res[s] = measure(model, split, idx, w)
    print(f"\n{args.run}: rho = {rho}, best check {res['best_check']}, weights at best {w}")
    print(f"{'':8s}{'':6s}{'mean':>9s}{'median':>9s}{'of_means':>10s}{'summed':>9s}{'min':>9s}{'max':>9s}")
    ok = True
    for s in ("train", "val"):
        for k, m in res[s].items():
            print(f"{s:8s}{k:6s}" + "".join(f"{m[c]:9.3f}" for c in ("mean", "median")) +
                  f"{m['of_means']:10.3f}{m['summed']:9.3f}{m['min']:9.3f}{m['max']:9.3f}")
            if s == "train":
                ok &= lo <= m["of_means"] <= hi
    save_json(res, out / "check_wm_balance_push.json")
    print(f"\nband for of_means on train: [{lo:.3f}, {hi:.3f}] -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
