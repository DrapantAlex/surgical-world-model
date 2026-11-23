"""How strongly, and in which direction, does each loss push the shared encoder?

During training the encoder gets one gradient, g = 1.0 g_a + 1.0 g_s + 0.1 g_i, with
  g_a = ∇_enc L_actor (Huber),  g_s = ∇_enc L_state (MSE),  g_i = ∇_enc L_image (MSE),
the loss weights of the run (config.json). For each batch this script measures the three
encoder gradients separately: their norms (unweighted and weighted), the weighted WM / actor
ratios, the cosines between them, and how much the WM bends the actor's update
(‖g - g_a‖ / ‖g_a‖, cos(g, g_a)), in total and per encoder layer. Nothing is trained.

- weights: the best weights of the run ("best") and an untrained model with the run's seed
  ("untrained"; a fresh init, not guaranteed bitwise equal to the run's own start). Untrained
  --state-skip runs: state_change is zero-initialised, so g_s is exactly 0 (reported as 0, and
  its cosines as undefined, with that reason).
- data: N_BATCHES x BATCH pairs drawn without replacement (seed SAMPLE_SEED) from all val
  pairs and, separately, all train pairs of the run (its keep ranges, chunk and history).
- cosines: per batch (mean [min, max]: what one training step sees, noisy) and of the
  gradients SUMMED over the batches (the consistent direction, batch noise averaged out).
- check: w_a g_a + w_s g_s + w_i g_i must equal the gradient of model.compute_loss (the
  compiled training loss) within CHECK_RTOL, else the script stops.
Adam divides each parameter's step by the running RMS of its TOTAL gradient: the ratios
measure the push on the encoder, not exactly the step.

  python -m scripts.analysis.look_encoder_gradients                 # the 3 WM-on history seeds
  python -m scripts.analysis.look_encoder_gradients <run> [<run> ...]

Saves outputs/analysis/encoder_gradients/<run>.json (the run folders stay untouched).
"""
import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import data
from models.balance import encoder_model
from models.run_config import (OUTPUTS, build_model, dataset_root, episodes_for, loss_weights,
                               run_history, run_keep_ranges, run_quat_ref, save_json)

# --- A: fixed choices ---
DEFAULT_RUNS = [f"star-il-simplecnn-latent512-canonq-chunk16-trim-skip-hist-val1000-seed{s}"
                for s in range(3)]
N_BATCHES, BATCH, SAMPLE_SEED = 20, 32, 0
CHECK_RTOL = 1e-5      # sum of the weighted terms vs the compiled loss's gradient
TERMS = ("actor", "state", "image")
OUTPUT_OF = {"actor": "action_output", "state": "next_state_output", "image": "next_image_output"}
ZERO_STATE_REASON = "state_change is zero-initialised (--state-skip): no gradient reaches the encoder"


# --- B: the encoder variables (encoder_model: models/balance.py, shared with training) ---
def encoder_layers(enc):
    """[(layer name, [variables])] of the encoder's trainable layers, in graph order."""
    return [(l.name, list(l.trainable_variables)) for l in enc.layers if l.trainable_variables]


# --- C: the three encoder gradients of one batch ---
def batch_gradients(model, enc_vars, weights, inputs, targets):
    """{term: flat encoder gradient (float64)} for actor / state / image, unweighted, plus the
    flat gradient of the compiled total loss; raises if the weighted sum differs from it."""
    losses = {"actor": keras.losses.Huber(), "state": keras.losses.MeanSquaredError(),
              "image": keras.losses.MeanSquaredError()}   # as compiled in run_config.build_model
    with tf.GradientTape(persistent=True) as tape:
        pred = dict(zip(("action_output", "next_state_output", "next_image_output"),
                        model(inputs, training=True)))
        values = {k: losses[k](targets[OUTPUT_OF[k]], pred[OUTPUT_OF[k]]) for k in TERMS}
        total = model.compute_loss(x=inputs, y=targets, y_pred=list(pred.values()), training=True)
    flat = lambda gs: np.concatenate([np.zeros(v.shape).ravel() if g is None else
                                      g.numpy().astype(np.float64).ravel()
                                      for v, g in zip(enc_vars, gs)])
    grads = {k: flat(tape.gradient(values[k], enc_vars)) for k in TERMS}
    g_total = flat(tape.gradient(total, enc_vars))
    del tape
    weighted_sum = sum(weights[OUTPUT_OF[k]] * grads[k] for k in TERMS)
    err = np.linalg.norm(weighted_sum - g_total) / max(np.linalg.norm(g_total), 1e-30)
    if err > CHECK_RTOL:
        raise AssertionError(f"weighted terms != compiled loss gradient (relative {err:.2e})")
    return grads, g_total, {k: float(v) for k, v in values.items()}, err


# --- D: norms, ratios and cosines ---
def cosine(u, v):
    """None if either vector is exactly 0 (the cosine is undefined there)."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return None if nu == 0 or nv == 0 else float(u @ v / (nu * nv))

def measures(g, w, g_total):
    """All measures for one set of encoder gradients g = {term: vector} (one batch or a sum)."""
    na = np.linalg.norm(g["actor"])
    m = {f"norm_{k}": float(np.linalg.norm(g[k])) for k in TERMS}
    m.update({f"wnorm_{k}": float(w[OUTPUT_OF[k]] * np.linalg.norm(g[k])) for k in TERMS})
    m["ratio_state"] = m["wnorm_state"] / na      # weighted WM term / actor term
    m["ratio_image"] = m["wnorm_image"] / na
    m["cos_actor_state"] = cosine(g["actor"], g["state"])
    m["cos_actor_image"] = cosine(g["actor"], g["image"])
    m["cos_state_image"] = cosine(g["state"], g["image"])
    m["bend"] = float(np.linalg.norm(g_total - w["action_output"] * g["actor"]) / na)
    m["cos_total_actor"] = cosine(g_total, g["actor"])
    return m

def summarise(per_batch):
    """mean [min, max] over the batches; None where a value is undefined in every batch."""
    out = {}
    for k in per_batch[0]:
        vals = [b[k] for b in per_batch if b[k] is not None]
        out[k] = None if not vals else {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                                        "max": float(np.max(vals)), "n": len(vals)}
    return out

def measure_split(model, split, weights, idx):
    enc = encoder_model(model)
    layers = encoder_layers(enc)
    enc_vars = [v for _, vs in layers for v in vs]
    sizes = [int(np.prod(v.shape)) for v in enc_vars]
    bounds = np.cumsum([0] + [sum(int(np.prod(v.shape)) for v in vs) for _, vs in layers])
    per_batch, summed, summed_total, loss_vals, max_err = [], None, 0.0, [], 0.0
    for b in range(N_BATCHES):
        t = idx[b * BATCH:(b + 1) * BATCH]
        inputs, targets = data.get_pairs(split, t)
        g, g_total, values, err = batch_gradients(model, enc_vars, weights, inputs, targets)
        max_err = max(max_err, err)
        per_batch.append(measures(g, weights, g_total))
        loss_vals.append(values)
        summed = g if summed is None else {k: summed[k] + g[k] for k in TERMS}
        summed_total = summed_total + g_total
    by_layer = {}
    for (name, _), lo, hi in zip(layers, bounds[:-1], bounds[1:]):
        by_layer[name] = measures({k: summed[k][lo:hi] for k in TERMS}, weights,
                                  summed_total[lo:hi])
    return {"n_encoder_params": int(sum(sizes)), "n_batches": N_BATCHES, "batch": BATCH,
            "check_max_rel_err": max_err,
            "losses_mean": {k: float(np.mean([v[k] for v in loss_vals])) for k in TERMS},
            "per_batch": summarise(per_batch),
            "summed": measures(summed, weights, summed_total),
            "summed_by_layer": by_layer}


# --- E: runs, tables and output ---
def fmt(x, nd=3):
    return "undefined" if x is None else f"{x:.{nd}g}" if abs(x) < 1e-2 and x != 0 else f"{x:.{nd}f}"

def print_tables(run, res):
    print(f"\n=== {run}")
    rows = [("‖g_a‖ (unweighted)", "norm_actor"), ("‖g_s‖", "norm_state"), ("‖g_i‖", "norm_image"),
            ("w_s‖g_s‖ / ‖g_a‖", "ratio_state"), ("w_i‖g_i‖ / ‖g_a‖", "ratio_image"),
            ("cos(g_a, g_s)", "cos_actor_state"), ("cos(g_a, g_i)", "cos_actor_image"),
            ("cos(g_s, g_i)", "cos_state_image"),
            ("‖g - g_a‖ / ‖g_a‖", "bend"), ("cos(g, g_a)", "cos_total_actor")]
    cols = [(f"{w}/{s}", res[w][s]) for w in ("untrained", "best") for s in ("val", "train")]
    print(f"{'':22s}" + "".join(f"  {c:>36s}" for c, _ in cols))
    for label, k in rows:
        cells = []
        for _, r in cols:
            pb, sm = r["per_batch"][k], r["summed"][k]
            pb_s = "undefined" if pb is None else f"{fmt(pb['mean'])} [{fmt(pb['min'])}, {fmt(pb['max'])}]"
            cells.append(f"{pb_s} | Σ {fmt(sm)}")
        print(f"{label:22s}" + "".join(f"  {c:>36s}" for c in cells))
    print("(per batch: mean [min, max] over the batches | Σ: of the gradients summed over the batches)")
    for w in ("untrained", "best"):
        if res[w]["val"]["summed"]["norm_state"] == 0.0:
            print(f"{w}: g_s = 0 exactly: {ZERO_STATE_REASON}")
    print("\nper encoder layer (best weights, val, Σ):  ‖g_a‖   w_s‖g_s‖/‖g_a‖   w_i‖g_i‖/‖g_a‖   "
          "cos(g_a,g_s)   cos(g_a,g_i)")
    for name, m in res["best"]["val"]["summed_by_layer"].items():
        print(f"  {name:12s} {fmt(m['norm_actor']):>10s} {fmt(m['ratio_state']):>16s} "
              f"{fmt(m['ratio_image']):>16s} {fmt(m['cos_actor_state']):>14s} {fmt(m['cos_actor_image']):>14s}")
    for w in ("untrained", "best"):
        lm = {s: res[w][s]["losses_mean"] for s in ("val", "train")}
        print(f"losses ({w}, mean of the batches): " + "; ".join(
            f"{s} actor {fmt(l['actor'], 4)} state {fmt(l['state'], 4)} image {fmt(l['image'], 4)}"
            for s, l in lm.items()) + f"; check max rel err {max(res[w][s]['check_max_rel_err'] for s in ('val', 'train')):.1e}")

def run_one(run):
    out = OUTPUTS / run
    cfg = json.loads((out / "config.json").read_text())
    stats = data.load_norm_stats(out / "norm_stats.json")
    root, quat_ref, keep = dataset_root(cfg), run_quat_ref(cfg, out), run_keep_ranges(cfg, out)
    splits = {s: data.load_split(s, root=root, stats=stats, quat_ref=quat_ref,
                                 episodes=episodes_for(s, cfg["max_episodes"], root),
                                 chunk=cfg.get("chunk"), keep_ranges=keep, history=run_history(cfg))
              for s in ("val", "train")}
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = {s: rng.choice(sp["pair_index"], N_BATCHES * BATCH, replace=False)
           for s, sp in splits.items()}
    weights = loss_weights(cfg)
    res = {}
    for which in ("untrained", "best"):
        keras.utils.set_random_seed(cfg["seed"])
        model = build_model(cfg, stats)
        if which == "best":
            model.load_weights(str(out / "best.weights.h5"))
        elif cfg.get("state_skip"):   # the reason for g_s = 0 must hold, not be assumed
            assert not np.any(model.get_layer("state_change").kernel.numpy()), "state_change != 0"
        res[which] = {s: measure_split(model, splits[s], weights, idx[s]) for s in splits}
    res["loss_weights"] = weights
    res["sample"] = {"seed": SAMPLE_SEED, "n_pairs": N_BATCHES * BATCH}
    res["zero_state_reason"] = ZERO_STATE_REASON if res["untrained"]["val"]["summed"]["norm_state"] == 0.0 else None
    return res

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    args = p.parse_args()
    tf.config.experimental.enable_op_determinism()   # repeated runs give the same numbers
    # full float32 on GPU: TF32 (the default on Ampere+) rounds matmul / conv inputs to a 10-bit
    # mantissa, ~1e-4 relative: the check above failed with it (9.5e-5), and the smallest ratios
    # measured here (~3e-5) are below that noise. Training itself ran with TF32 on.
    tf.config.experimental.enable_tensor_float_32_execution(False)
    save_dir = OUTPUTS / "analysis" / "encoder_gradients"
    save_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for run in args.runs:
        all_res[run] = run_one(run)
        save_json(all_res[run], save_dir / f"{run}.json")
        print_tables(run, all_res[run])
    if len(all_res) > 1:   # seed summary of the Σ values
        print(f"\n=== mean over {len(all_res)} runs (Σ over the batches)  [min, max]")
        for k in ("ratio_state", "ratio_image", "cos_actor_state", "cos_actor_image", "bend", "cos_total_actor"):
            line = f"{k:18s}"
            for w in ("untrained", "best"):
                for s in ("val", "train"):
                    v = [r[w][s]["summed"][k] for r in all_res.values()]
                    line += (f"  {w}/{s}: undefined" if any(x is None for x in v) else
                             f"  {w}/{s}: {fmt(np.mean(v))} [{fmt(min(v))}, {fmt(max(v))}]")
            print(line)
    print(f"\nsaved {save_dir}/")


if __name__ == "__main__":
    main()
