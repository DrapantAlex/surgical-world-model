"""Experiment (c): is the balancing train_step (models/balance.py) the same training as Keras'?

On CPU, deterministic ops, the history run's configuration (2 train episodes, the run's own
statistics), the same initial weights and the same 20 batches, against the plain model:
  level 1  mode "fixed_compiled" (Keras' train_step, copied): every weight after 20 steps
           identical (tolerance 0.0) -> the subclass, its tape, optimizer step and metrics
           are Keras' own.
  level 2  mode "fixed_split" (the balancer's split path on every step, weights frozen at the
           history-run values 1.0 / 0.1, state on all dims, references 1):
           (a) step 0, before any update: the summed split gradient vs the combined gradient
               of the compiled loss, relative norm difference <= GRAD_RTOL (Adam can amplify
               rounding on near-zero gradients into weight differences up to ~lr, so the
               gradients are compared directly);
           (b) after 20 steps: max |Δw| and the per-step loss difference at most CONTROL_X x
               those of a control, the plain model with its loss x (1 + 2^-23) (~1 ulp per
               gradient: Adam amplifies any rounding change, so a fixed bound is meaningless).
Exit 1 if any level fails. The default path (--wm-balance off) never builds the subclass:
that is checked by check_eval_unchanged and the smoke run.

  python -m scripts.analysis.check_wm_balance_equivalence
"""
import json
import sys
import types

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import data
from models.run_config import (OUTPUTS, build_model, dataset_root, run_keep_ranges, run_quat_ref)

# --- A: fixed choices ---
REF_RUN = "star-il-simplecnn-latent512-canonq-chunk16-trim-skip-hist-val1000-seed0"
N_EPISODES, N_STEPS, BATCH, SEED = 2, 20, 32, 0
GRAD_RTOL = 1e-5   # level 2a
CONTROL_X = 2.0    # level 2b: at most 2 x the divergence of the 1-ulp control


# --- B: models and data ---
def make(cfg, stats, mode=None, state_dim=None):
    """The plain model (mode None) or the subclass in a check mode, same seed -> same init."""
    keras.utils.set_random_seed(SEED)
    if mode is None:
        return build_model(cfg, stats)
    cfg_b = {**cfg, "wm_balance": 0.3, "wm_balance_terms": ["image", "state"],
             "wm_balance_every": 1}
    return build_model(cfg_b, stats, balance_overrides={
        "mode": mode, "refs": {"image": 1.0, "state": 1.0},
        "state_dims": list(range(state_dim))})

def batches(cfg, stats, out):
    root = dataset_root(cfg)
    split = data.load_split("train", root=root, stats=stats, quat_ref=run_quat_ref(cfg, out),
                            episodes=data.get_split_episodes("train", root)[:N_EPISODES],
                            chunk=cfg.get("chunk"), keep_ranges=run_keep_ranges(cfg, out),
                            history=cfg.get("history", False))
    idx = np.random.default_rng(SEED).choice(split["pair_index"], N_STEPS * BATCH, replace=False)
    return [data.get_pairs(split, idx[i * BATCH:(i + 1) * BATCH]) for i in range(N_STEPS)]

def max_weight_diff(a, b):
    return max(float(np.max(np.abs(p.numpy() - q.numpy()))) for p, q in zip(a.weights, b.weights))

def ulp_train_step(self, data_):
    """Level-2b control: Keras' train_step with the loss x (1 + 2^-23) (~1 ulp per gradient)."""
    x, y = data_[0], data_[1]
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self._compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=None, training=True)
        self._loss_tracker.update_state(loss, sample_weight=tf.shape(x["state_input"])[0])
        loss = loss * (1.0 + 2.0 ** -23)
    grads = tape.gradient(loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    return self.compute_metrics(x, y, y_pred, sample_weight=None)

def train(model, data_batches):
    return [float(model.train_on_batch(x, y, return_dict=True)["loss"]) for x, y in data_batches]


# --- C: the checks ---
def main():
    tf.config.experimental.enable_op_determinism()
    out = OUTPUTS / REF_RUN
    cfg = json.loads((out / "config.json").read_text())
    stats = data.load_norm_stats(out / "norm_stats.json")
    state_dim = data.feature_dims(dataset_root(cfg))[0]
    xb = batches(cfg, stats, out)
    ok = True

    plain = make(cfg, stats)
    loss_plain = train(plain, xb)

    # level 1: Keras' own train_step, in the subclass
    fc = make(cfg, stats, "fixed_compiled", state_dim)
    fresh = make(cfg, stats)
    d0 = max_weight_diff(fc, fresh)
    loss_fc = train(fc, xb)
    d1 = max_weight_diff(fc, plain)
    print(f"level 1: initial weights max |Δw| = {d0} ; after {N_STEPS} steps max |Δw| = {d1} "
          f"(must be 0.0); losses identical: {loss_fc == loss_plain}")
    ok &= d0 == 0.0 and d1 == 0.0 and loss_fc == loss_plain

    # level 2a: step 0, the summed split gradient vs the combined gradient of the compiled loss
    fs = make(cfg, stats, "fixed_split", state_dim)
    x, y = xb[0]
    tf_vars = [v.value for v in fs.trainable_weights]
    with tf.GradientTape(persistent=True) as tape:
        y_pred = fs(x, training=True)
        compiled = fs.compute_loss(x=x, y=y, y_pred=y_pred, training=True)
        L = fs.term_losses(y, y_pred)
    g_comb = tape.gradient(compiled, tf_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    g_split = fs.split_gradients(tape, L, tf_vars)
    del tape
    diff = np.sqrt(sum(float(tf.reduce_sum((a - b) ** 2)) for a, b in zip(g_split, g_comb)))
    norm = np.sqrt(sum(float(tf.reduce_sum(b ** 2)) for b in g_comb))
    per_var = [float(tf.norm(a - b) / tf.maximum(tf.norm(b), 1e-30))
               for a, b in zip(g_split, g_comb)]
    worst = int(np.argmax(per_var))
    print(f"level 2a: step-0 gradient, split sum vs combined: relative {diff / norm:.2e} "
          f"(<= {GRAD_RTOL}); worst variable {fs.trainable_weights[worst].path} "
          f"{per_var[worst]:.2e}")
    ok &= diff / norm <= GRAD_RTOL

    # level 2b: 20 steps through the split path (weights frozen) vs the plain model, judged
    # against a CONTROL: the plain model with its loss x (1 + 2^-23), i.e. ~1 ulp on every
    # gradient. Adam turns such rounding into weight differences up to ~lr (measured 11-23:
    # control 1.07e-3 after 20 steps), so a fixed 1e-6 bound cannot be met by ANY change of
    # summation order; the split path must not diverge more than CONTROL_X x the control.
    fs = make(cfg, stats, "fixed_split", state_dim)
    loss_fs = train(fs, xb)
    d2 = max_weight_diff(fs, plain)
    lrel = max(abs(a - b) / abs(b) for a, b in zip(loss_fs, loss_plain))
    ctrl = make(cfg, stats)
    ctrl.train_step = types.MethodType(ulp_train_step, ctrl)   # before the first call
    loss_ctrl = train(ctrl, xb)
    d_ctrl = max_weight_diff(ctrl, plain)
    l_ctrl = max(abs(a - b) / abs(b) for a, b in zip(loss_ctrl, loss_plain))
    print(f"level 2b: after {N_STEPS} steps max |Δw| = {d2:.2e} (control {d_ctrl:.2e}, "
          f"<= {CONTROL_X} x); max per-step loss relative difference {lrel:.2e} "
          f"(control {l_ctrl:.2e}, <= {CONTROL_X} x)")
    ok &= d_ctrl > 0 and d2 <= CONTROL_X * d_ctrl and lrel <= CONTROL_X * l_ctrl

    print("\nOK: the balancing train_step is Keras' training" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
