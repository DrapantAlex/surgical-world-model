"""Experiment (c), check e: does the controller hold each balanced WM term's push at RHO?

The push w_k‖g_k‖ / ‖g_a‖ on the encoder oscillates with the WEIGHTS (11-24, test 3: CV 0.69,
period ≈ 4-5 steps, also with the WM weights frozen), so one frozen weight state cannot measure
it. This check measures it over a REPLAY WINDOW:
  1. resume the run at its best checkpoint: weights AND Adam state from best.weights.h5
     (verified: iterations = the best check's batch, every slot equal to the file), the
     controller state from history.csv at the best check;
  2. continue on the run's real training stream (same seed, shuffle, repeat) for K_STEPS
     steps with the controller active (the model's own train_step); alignment verified
     against wm_balance_trace.csv at the first measurement step;
  3. before every step, eager encoder norms (actor, each balanced normalised WM loss) on a
     FIXED set of N_FIXED train batches, with this script's own gradient and norm code.
PASS statistic per balanced term: the RATIO OF TIME MEANS, mean_t(w_t mean‖g_k‖) / mean_t(mean‖g_a‖),
in BAND x RHO / 0.3 (BAND = 0.2-0.45 for RHO = 0.3, fixed before the data). Also reported: the time
mean of the per-step push, its range and CV, and w over the window. Train batches only (the
controller measures on train). TF32 as in training. Measurement only: nothing is saved but
check_wm_balance_push.json. Exit 1 if a term is outside the band or the resume / alignment fails.

  python -m scripts.analysis.check_wm_balance_push <run>
"""
import argparse
import json
import sys

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from models import data
from models.balance import encoder_model
from models.pipeline import make_dataset
from models.run_config import (OUTPUTS, build_model, dataset_root, episodes_for, run_history,
                               run_keep_ranges, run_quat_ref, run_wm_balance, save_json)

# --- A: fixed choices ---
K_STEPS, N_FIXED, BATCH, SAMPLE_SEED = 100, 10, 32, 0
BAND = (0.2, 0.45)          # for RHO = 0.3; scaled with RHO for other targets
ALIGN_RTOL = 0.02           # replayed vs original controller measurement (GPU nondeterminism)


# --- B: per-batch encoder norms (eager; this script's own gradient code) ---
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


def fixed_norms(model, enc_vars, batches):
    """Mean over the fixed batches of each term's encoder gradient norm."""
    rows = [{k: float(np.linalg.norm(v)) for k, v in batch_norms(model, enc_vars, x, y).items()}
            for x, y in batches]
    return pd.DataFrame(rows).mean()


# --- C: resume at the best checkpoint (weights, Adam, controller) and the training stream ---
def resume(cfg, stats, out, best):
    model = build_model(cfg, stats)
    model.optimizer.build(model.trainable_weights)   # so load_weights restores Adam too
    model.load_weights(str(out / "best.weights.h5"))
    start = int(cfg["best_batch"])
    with h5py.File(out / "best.weights.h5", "r") as f:
        saved = [f[f"optimizer/vars/{i}"][()] for i in range(len(f["optimizer/vars"]))]
    restored = [v.numpy() for v in model.optimizer.variables]
    if int(model.optimizer.iterations.numpy()) != start or len(restored) != len(saved) or \
            not all(np.array_equal(a, b) for a, b in zip(restored, saved)):
        raise SystemExit(f"cannot resume: optimizer iterations {int(model.optimizer.iterations.numpy())} "
                         f"(expected {start}) or slots not restored: replay from step 0 instead")
    model.v_ema_a.assign(best["wm_bal_ema_actor"])
    model.v_init_a.assign(True)
    for k in model.bal_terms:
        v = model.v[k]
        for name, col in (("w", "w"), ("ema", "ema"), ("push_ema", "push"), ("cos_ema", "cos"),
                          ("init_step", "init_step")):
            v[name].assign(best[f"wm_bal_{col}_{k}"])
        for flag in ("init", "push_init", "cos_init"):
            v[flag].assign(True)
    return model, start, len(saved)


def training_stream(split, cfg, start):
    """The run's training batches from batch `start` on (as scripts/train.py builds them)."""
    keras.utils.set_random_seed(cfg["seed"])
    it = iter(make_dataset(split, cfg["batch_size"], shuffle=True, seed=cfg["seed"],
                           repeat=bool(cfg["val_every"])))
    for _ in range(start):
        next(it)
    return it


# --- D: the replay window ---
def replay(model, split, cfg, start, fixed, trace):
    enc_vars = list(encoder_model(model).trainable_variables)
    it = training_stream(split, cfg, start)
    rows, aligned = [], None
    for s in range(K_STEPS + 1):
        n = fixed_norms(model, enc_vars, fixed)          # at the weights before step start + s
        row = {"step": start + s, "n_actor": n["actor"]}
        for k in model.bal_terms:
            row[f"w_{k}"] = float(model.v[k]["w"].numpy())
            row[f"n_{k}"] = n[k]
        rows.append(row)
        if s == K_STEPS:
            break
        x, y = next(it)
        logs = model.train_on_batch(x, y, return_dict=True)
        if aligned is None and float(logs["wm_bal_measured"]) == 1.0:   # the first measurement
            orig = trace[trace.step == start + s]
            if orig.empty:
                raise SystemExit(f"no trace row at step {start + s}: cannot verify the stream")
            got = {"actor": float(logs["wm_bal_last_n_actor"]),
                   **{k: float(logs[f"wm_bal_last_n_{k}"]) for k in model.bal_terms}}
            want = {"actor": float(orig.n_actor.iloc[0]),
                    **{k: float(orig[f"n_{k}"].iloc[0]) for k in model.bal_terms}}
            err = max(abs(got[k] / want[k] - 1) for k in got)
            aligned = {"step": start + s, "replay": got, "original": want, "max_rel_err": err}
            if err > ALIGN_RTOL:
                raise SystemExit(f"stream not aligned at step {start + s}: {got} vs {want}")
    return pd.DataFrame(rows), aligned


def statistics(df, terms):
    out = {}
    for k in terms:
        wn = df[f"w_{k}"] * df[f"n_{k}"]
        push = wn / df["n_actor"]
        out[k] = {"ratio_of_time_means": float(wn.mean() / df["n_actor"].mean()),
                  "time_mean_push": float(push.mean()), "push_min": float(push.min()),
                  "push_max": float(push.max()), "push_cv": float(push.std() / push.mean()),
                  "w_start": float(df[f"w_{k}"].iloc[0]), "w_end": float(df[f"w_{k}"].iloc[-1])}
    return out


# --- E: main ---
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    args = p.parse_args()
    out = OUTPUTS / args.run
    cfg = json.loads((out / "config.json").read_text())
    rho = run_wm_balance(cfg)
    if rho is None:
        raise SystemExit(f"{args.run} is not a --wm-balance run")
    stats = data.load_norm_stats(out / "norm_stats.json")
    root = dataset_root(cfg)
    best = pd.read_csv(out / "history.csv").iloc[cfg["best_epoch"] - 1]
    trace = pd.read_csv(out / "wm_balance_trace.csv")
    split = data.load_split("train", root=root, stats=stats, quat_ref=run_quat_ref(cfg, out),
                            episodes=episodes_for("train", cfg["max_episodes"], root),
                            chunk=cfg.get("chunk"), keep_ranges=run_keep_ranges(cfg, out),
                            history=run_history(cfg))
    idx = np.random.default_rng(SAMPLE_SEED).choice(split["pair_index"], N_FIXED * BATCH, replace=False)
    fixed = [data.get_pairs(split, idx[b * BATCH:(b + 1) * BATCH]) for b in range(N_FIXED)]

    model, start, n_opt = resume(cfg, stats, out, best)
    df, aligned = replay(model, split, cfg, start, fixed, trace)
    res = {"statistic": "replay_window", "rho": rho, "every": cfg["wm_balance_every"],
           "best_check": int(cfg["best_epoch"]), "start_step": start, "k_steps": K_STEPS,
           "n_fixed_batches": N_FIXED, "optimizer_variables_restored": n_opt,
           "alignment": aligned, "terms": statistics(df, model.bal_terms)}
    lo, hi = BAND[0] * rho / 0.3, BAND[1] * rho / 0.3
    print(f"\n{args.run}: rho {rho}, every {cfg['wm_balance_every']}, resumed at step {start} "
          f"(best check {res['best_check']}; Adam restored, {n_opt} variables), {K_STEPS} steps")
    print(f"alignment at step {aligned['step']}: max relative difference {aligned['max_rel_err']:.1e}")
    print(f"{'term':6s}{'ratio of time means':>21s}{'time-mean push':>16s}{'min':>8s}{'max':>8s}"
          f"{'CV':>6s}{'w start -> end':>24s}")
    ok = True
    for k, m in res["terms"].items():
        print(f"{k:6s}{m['ratio_of_time_means']:21.3f}{m['time_mean_push']:16.3f}{m['push_min']:8.3f}"
              f"{m['push_max']:8.3f}{m['push_cv']:6.2f}{m['w_start']:12.5f} -> {m['w_end']:.5f}")
        ok &= lo <= m["ratio_of_time_means"] <= hi
    res["band"], res["pass"] = [lo, hi], bool(ok)
    save_json(res, out / "check_wm_balance_push.json")
    print(f"\nband for the ratio of time means: [{lo:.3f}, {hi:.3f}] -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
