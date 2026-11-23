"""Experiment (c): balanced world-model losses (--wm-balance RHO).

The actor's weight stays 1.0 (the reference). Each balanced WM term k (image, state) is divided
by its train copy reference c_k (stats["wm_balance"], run_config.prepare_stats), and its weight
w_k is set so that its push on the shared encoder stays at RHO x the actor's:

    w_k = min(RHO * EMA‖g_a‖ / EMA‖g_k‖, W_CAP)        (encoder weights only)

The push ratio w_k‖g_k‖ / ‖g_a‖ is exactly proportional to w_k at a given step, so this closed
form needs no learning rate. The norms are measured on step 0, then every N steps (and on every
step while a balanced term is still uninitialised), and smoothed by an EMA (BETA); the weights
are held fixed in between. Before its first nonzero measurement a term keeps its raw history-run
weight (LOSS_WEIGHTS; the state term: on the KUKA-only loss of --wm-state-groups), so nothing is
pushed hard before it is measured. A term that is NOT balanced keeps its raw history-run term
(the 8-dim state MSE x 1.0, or the image MSE x 0.1).

compile() is unchanged (run_config.build_model): the logged per-output losses, "loss" and every
val_ value mean what they mean in every other run; the optimised objective is logged as
wm_bal_objective. test_step is Keras' own. The controller state lives in raw tf.Variables, which
Keras does not track: best.weights.h5 holds exactly the variables of a normal run.
"""
import csv

import tensorflow as tf
from tensorflow import keras

# --- A: fixed choices ---
TERMS = ("image", "state")
BETA = 0.9                  # EMA of the encoder gradient norms, per measurement
W_CAP = 1e4                 # upper bound of a balanced weight (normalised units)
OVERSHOOT_X, OVERSHOOT_N = 2.0, 20     # alarm: realised push EMA > 2 RHO for 20 measurements
CONFLICT_COS, CONFLICT_N = -0.3, 50    # alarm: EMA cos(g_a, g_k) < -0.3 for 50 measurements
MODES = ("balance", "fixed_compiled", "fixed_split")   # the last two: equivalence checks only
Z = tf.UnconnectedGradients.ZERO


# --- B: the encoder variables, without relying on layer names ---
def encoder_model(model):
    """image_input -> encoded_img as a sub-model sharing the layers (and variables) of `model`.
    encoded_img = the first input of the actor's Concatenate (swm.py C). Keras builds this
    sub-model only if the output depends on image_input alone, so it holds exactly the encoder.
    Moved here from scripts/analysis/look_encoder_gradients.py (2026-11-23), unchanged."""
    concat = next(l for l in model.layers if isinstance(l, keras.layers.Concatenate))
    return keras.Model(model.get_layer("image_input").output, concat.input[0])


# --- C: the model with the balancing train_step ---
class BalancedWorldModel(keras.Model):
    """The actor-world model of swm.py, re-wrapped (same layers and variables) with a
    train_step that balances the WM terms. Built by run_config.build_model for --wm-balance runs.

    rho: target push share; balanced: the balanced terms (subset of TERMS); refs: {term: c_k};
    state_dims: the state dims of the balanced state term; raw_weights: {term: history-run
    weight}; every: N. mode: "balance"; "fixed_compiled" = Keras' own train_step (equivalence
    level 1); "fixed_split" = the split path with the weights frozen (level 2)."""

    def __init__(self, inputs, outputs, *, rho, balanced, refs, state_dims, raw_weights,
                 every=10, mode="balance", name=None):
        super().__init__(inputs=inputs, outputs=outputs, name=name)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.bal_mode, self.bal_rho, self.bal_every = mode, float(rho), int(every)
        self.bal_terms = tuple(k for k in TERMS if k in balanced)
        self.bal_unbalanced = tuple(k for k in TERMS if k not in balanced)
        self.bal_raw = {k: float(raw_weights[k]) for k in TERMS}
        self.bal_c = {k: tf.constant(float(refs[k]), tf.float32) for k in TERMS}
        self.bal_dims = tf.constant(list(state_dims), tf.int32)
        enc_ids = {id(v) for v in encoder_model(self).trainable_variables}
        self.bal_enc_mask = [id(v) in enc_ids for v in self.trainable_weights]
        self.huber, self.mse = keras.losses.Huber(), keras.losses.MeanSquaredError()
        var = lambda v, dt=tf.float32: tf.Variable(v, trainable=False, dtype=dt)
        # controller state (raw tf.Variables: not tracked by Keras, not in best.weights.h5)
        self.v_ema_a, self.v_init_a = var(0.0), var(False, tf.bool)
        self.v = {k: {"w": var(self.bal_raw[k] * float(refs[k])),   # raw weight, normalised units
                      "ema": var(0.0), "init": var(False, tf.bool), "init_step": var(-1.0),
                      "push_ema": var(0.0), "push_init": var(False, tf.bool),
                      "cos_ema": var(0.0), "cos_init": var(False, tf.bool),
                      "n_over": var(0.0), "n_conflict": var(0.0),
                      "last_n": var(0.0), "last_cos": var(0.0), "last_push": var(0.0)}
                  for k in self.bal_terms}
        self.v_last_n_a = var(0.0)
        self.v_obj_sum, self.v_obj_n = var(0.0), var(0.0)

    # --- C1: the loss terms (the loss objects of run_config.build_model's compile) ---
    def term_losses(self, y, y_pred):
        a, s, img = y_pred
        y_s = y["next_state_output"]
        return {"actor": self.huber(y["action_output"], a),
                "image_raw": self.mse(y["next_image_output"], img),
                "state_raw": self.mse(y_s, s),                                  # all dims
                "image_norm": self.mse(y["next_image_output"], img) / self.bal_c["image"],
                "state_norm": self.mse(tf.gather(y_s, self.bal_dims, axis=-1),
                                       tf.gather(s, self.bal_dims, axis=-1)) / self.bal_c["state"]}

    def objective(self, L):
        """The optimised loss with the current weights."""
        total = L["actor"]
        for k in self.bal_terms:
            total += self.v[k]["w"] * L[f"{k}_norm"]
        for k in self.bal_unbalanced:
            total += self.bal_raw[k] * L[f"{k}_raw"]
        return total

    # --- C2: encoder norms and dot products of flat gradient lists ---
    def _enc(self, grads):
        return [g for g, m in zip(grads, self.bal_enc_mask) if m]

    def _norm(self, grads):
        return tf.linalg.global_norm(self._enc(grads))

    def _dot(self, ga, gb):
        return tf.add_n([tf.reduce_sum(p * q) for p, q in zip(self._enc(ga), self._enc(gb))])

    # --- C3: the training step ---
    def _measure_now(self):
        if self.bal_mode == "fixed_split":
            return tf.constant(True)
        due = tf.equal(self.optimizer.iterations % self.bal_every, 0)
        for k in self.bal_terms:   # every step while a balanced term is uninitialised
            due = due | tf.logical_not(self.v[k]["init"])
        return due

    def split_gradients(self, tape, L, tf_vars):
        """Measurement step: per-term gradients, the controller update, then
        g = g_a + sum_k w_k(new) grad L'_k + grad(unbalanced raw terms)."""
        ga = tape.gradient(L["actor"], tf_vars, unconnected_gradients=Z)
        parts = {k: tape.gradient(L[f"{k}_norm"], tf_vars, unconnected_gradients=Z)
                 for k in self.bal_terms}
        rest = None
        if self.bal_unbalanced:
            rest = tape.gradient(tf.add_n([self.bal_raw[k] * L[f"{k}_raw"]
                                           for k in self.bal_unbalanced]),
                                 tf_vars, unconnected_gradients=Z)
        n_a = self._norm(ga)
        stats = {k: (self._norm(g), self._dot(ga, g)) for k, g in parts.items()}
        if self.bal_mode == "balance":
            self._update(n_a, stats)
        else:   # fixed_split: weights frozen, only the last measurement recorded
            self.v_last_n_a.assign(n_a)
        total = list(ga)
        for k, g in parts.items():
            total = [t + self.v[k]["w"] * p for t, p in zip(total, g)]
        if rest is not None:
            total = [t + r for t, r in zip(total, rest)]
        return total

    def _update(self, n_a, stats):
        ok_a = tf.math.is_finite(n_a) & (n_a > 0)
        self.v_last_n_a.assign(n_a)
        self.v_ema_a.assign(tf.where(ok_a, tf.where(self.v_init_a,
                                                    BETA * self.v_ema_a + (1 - BETA) * n_a, n_a),
                                     self.v_ema_a))
        self.v_init_a.assign(self.v_init_a | ok_a)
        step = tf.cast(self.optimizer.iterations, tf.float32)
        for k, (n_k, dot) in stats.items():
            v = self.v[k]
            ok = ok_a & tf.math.is_finite(n_k) & (n_k > 0)
            cos = tf.math.divide_no_nan(dot, n_a * n_k)
            push = tf.math.divide_no_nan(v["w"] * n_k, n_a)   # realised: the weight in effect
            v["last_n"].assign(n_k)
            v["last_cos"].assign(cos)
            v["last_push"].assign(push)
            # push EMA only under controller-set weights (from the 2nd valid measurement on)
            ok_push = ok & v["init"]
            v["push_ema"].assign(tf.where(ok_push, tf.where(
                v["push_init"], BETA * v["push_ema"] + (1 - BETA) * push, push), v["push_ema"]))
            v["push_init"].assign(v["push_init"] | ok_push)
            v["cos_ema"].assign(tf.where(ok, tf.where(
                v["cos_init"], BETA * v["cos_ema"] + (1 - BETA) * cos, cos), v["cos_ema"]))
            v["cos_init"].assign(v["cos_init"] | ok)
            # the norm EMA; the first valid measurement initialises it (and records the step)
            v["init_step"].assign(tf.where(ok & tf.logical_not(v["init"]), step, v["init_step"]))
            v["ema"].assign(tf.where(ok, tf.where(
                v["init"], BETA * v["ema"] + (1 - BETA) * n_k, n_k), v["ema"]))
            v["init"].assign(v["init"] | ok)
            # the closed form, capped; unchanged until the term is initialised
            w_new = tf.minimum(self.bal_rho * tf.math.divide_no_nan(self.v_ema_a, v["ema"]), W_CAP)
            v["w"].assign(tf.where(v["init"] & self.v_init_a, w_new, v["w"]))
            # alarm counters (in measurements; reset when the condition is false)
            v["n_over"].assign(tf.where(v["push_init"] & (v["push_ema"] > OVERSHOOT_X * self.bal_rho),
                                        v["n_over"] + 1, 0.0))
            v["n_conflict"].assign(tf.where(v["cos_init"] & (v["cos_ema"] < CONFLICT_COS),
                                            v["n_conflict"] + 1, 0.0))

    def train_step(self, data):
        x, y = data[0], data[1]
        batch = tf.shape(x["state_input"])[0]
        if self.bal_mode == "fixed_compiled":   # equivalence level 1: Keras' own train_step
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self._compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=None,
                                          training=True)
                self._loss_tracker.update_state(loss, sample_weight=batch)
                loss = self.optimizer.scale_loss(loss)
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return self.compute_metrics(x, y, y_pred, sample_weight=None)

        tf_vars = [v.value for v in self.trainable_weights]   # tape.gradient inside tf.cond
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)
            compiled = self._compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=None,
                                          training=True)   # per-output logs and "loss", as today
            self._loss_tracker.update_state(compiled, sample_weight=batch)
            L = self.term_losses(y, y_pred)
            obj = self.objective(L)
        measure = self._measure_now()
        grads = tf.cond(measure, lambda: self.split_gradients(tape, L, tf_vars),
                        lambda: tape.gradient(obj, tf_vars, unconnected_gradients=Z))
        del tape
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.v_obj_sum.assign_add(obj)
        self.v_obj_n.assign_add(1.0)
        logs = self.compute_metrics(x, y, y_pred, sample_weight=None)
        logs.update(self.log_values(measure))
        return logs

    # --- C4: logs and the controller state ---
    def log_values(self, measure):
        """wm_bal_*: per check = the state at the end of the check (CSVLogger, TensorBoard, W&B);
        wm_bal_last_* and wm_bal_measured: per step, read by WMBalanceTrace."""
        out = {"wm_bal_objective": tf.math.divide_no_nan(self.v_obj_sum, self.v_obj_n),
               "wm_bal_ema_actor": self.v_ema_a, "wm_bal_last_n_actor": self.v_last_n_a,
               "wm_bal_measured": tf.cast(measure, tf.float32)}
        for k in self.bal_terms:
            v = self.v[k]
            out.update({f"wm_bal_w_{k}": v["w"], f"wm_bal_ema_{k}": v["ema"],
                        f"wm_bal_push_{k}": v["push_ema"], f"wm_bal_cos_{k}": v["cos_ema"],
                        f"wm_bal_init_step_{k}": v["init_step"],
                        f"wm_bal_alarm_overshoot_{k}": tf.cast(v["n_over"] >= OVERSHOOT_N, tf.float32),
                        f"wm_bal_alarm_conflict_{k}": tf.cast(v["n_conflict"] >= CONFLICT_N, tf.float32),
                        f"wm_bal_alarm_cap_{k}": tf.cast(v["w"] >= W_CAP, tf.float32),
                        f"wm_bal_last_n_{k}": v["last_n"], f"wm_bal_last_cos_{k}": v["last_cos"],
                        f"wm_bal_last_push_{k}": v["last_push"]})
        return {k: tf.identity(v) for k, v in out.items()}   # values read now, not handles

    def reset_objective(self):
        self.v_obj_sum.assign(0.0)
        self.v_obj_n.assign(0.0)

    def controller_state(self):
        """Plain floats, for config.json at the end of training."""
        out = {"ema_actor": float(self.v_ema_a.numpy())}
        for k in self.bal_terms:
            out[k] = {n: float(self.v[k][n].numpy())
                      for n in ("w", "ema", "init_step", "push_ema", "cos_ema")}
        return out


# --- D: the per-measurement trace ---
class WMBalanceTrace(keras.callbacks.Callback):
    """wm_balance_trace.csv: one row per measurement step (step = training steps before it),
    with the raw per-batch norms, cosines, realised pushes and the weights after the update.
    Also resets the objective's running mean at the start of every check."""
    def __init__(self, path, terms):
        super().__init__()
        self.path, self.terms, self.step = path, tuple(terms), 0
        self.cols = (["step", "n_actor"] + [f"{p}_{k}" for k in self.terms
                                            for p in ("n", "cos", "push", "w")])

    def on_train_begin(self, logs=None):
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.cols)

    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_objective()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if float(logs.get("wm_bal_measured", 0.0)) == 1.0:
            row = [self.step, float(logs["wm_bal_last_n_actor"])]
            for k in self.terms:
                row += [float(logs[f"wm_bal_last_{p}_{k}"]) for p in ("n", "cos", "push")]
                row.append(float(logs[f"wm_bal_w_{k}"]))
            self.writer.writerow(row)
        self.step += 1

    def on_train_end(self, logs=None):
        self.file.close()
