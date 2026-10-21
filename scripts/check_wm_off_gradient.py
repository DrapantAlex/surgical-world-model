"""
check_wm_off_gradient.py

Question: with --w-next-state 0 --w-next-image 0, does the world model still
send any gradient into the shared encoder (or move its own heads)?

On one batch, for every trainable variable, compare the gradient of
  (a) the compiled total loss (Keras loss weights, as in training) and
  (b) the action loss alone (Huber).
With weights (1, 0, 0) they must be identical: encoder/actor gradients equal,
world-model-only gradients zero. With the default weights (1, 1, 0.1) they must
differ (control: the check is able to fail).

Labels/statistics come from a finished canonical run (no full-split loading).

  python -m scripts.check_wm_off_gradient
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import data
from scripts.train import OUTPUTS, build_model

REF_RUN = "star-il-simplecnn-latent512-canonq-seed0"
ROOT = Path("data/open_h/Surgical/jhu/imerse/star_il/star_il")


def max_grad_diff(w_state, w_image, inputs, targets):
    keras.utils.set_random_seed(0)   # same initial weights for both settings
    cfg = {"latent_dim": 512, "encoder": "simple_cnn", "freeze_encoder": False,
           "unfreeze_last_n": 0, "lr": 3e-4, "dataset_root": str(ROOT),
           "w_next_state": w_state, "w_next_image": w_image}
    model = build_model(cfg)
    with tf.GradientTape(persistent=True) as tape:
        pred = model(inputs, training=True)
        total = model.compute_loss(x=inputs, y=targets, y_pred=pred, training=True)
        action = keras.losses.Huber()(targets["action_output"], pred[0])
    variables = model.trainable_variables
    g_total = tape.gradient(total, variables)
    g_action = tape.gradient(action, variables)
    diffs = {}
    for v, gt, ga in zip(variables, g_total, g_action):
        gt = np.zeros(v.shape) if gt is None else gt.numpy()
        ga = np.zeros(v.shape) if ga is None else ga.numpy()   # None = not on the actor path
        diffs[v.path] = float(np.abs(gt - ga).max())
    print("Keras per-output losses (logged unweighted):",
          [m.name for m in model.metrics if "loss" in m.name])
    return diffs


def main():
    # GPU kernels sum in a non-fixed order (e.g. conv bias gradients): without this, two
    # backward passes of the SAME loss can differ by ~1e-8, which would hide an exact 0
    tf.config.experimental.enable_op_determinism()
    out = OUTPUTS / REF_RUN
    stats = data.load_norm_stats(out / "norm_stats.json")
    quat_ref = data.load_quat_reference(out / "quat_reference.json")
    eps = data.get_split_episodes("train", ROOT)[:1]
    split = data.load_split("train", root=ROOT, stats=stats, quat_ref=quat_ref, episodes=eps)
    inputs, targets = data.get_pairs(split, split["pair_index"][:16])

    off = max_grad_diff(0.0, 0.0, inputs, targets)
    worst = max(off.values())
    print(f"\nweights (1, 0, 0): max |grad_total - grad_action| over {len(off)} variables = {worst:.3e}")
    assert worst == 0.0, sorted(off.items(), key=lambda kv: -kv[1])[:5]

    on = max_grad_diff(1.0, 0.1, inputs, targets)
    changed = [k for k, d in on.items() if d > 0]
    print(f"weights (1, 1, 0.1): {len(changed)} of {len(on)} variables get a different gradient "
          f"(control), e.g. {changed[:3]}")
    assert changed, "control failed: the world model never changes the gradient"
    print("\nOK: with both world-model weights at 0, only the action loss trains the network.")


if __name__ == "__main__":
    main()
