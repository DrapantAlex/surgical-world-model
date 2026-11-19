"""Throwaway check: does validating every N batches change training?
Runs the real scripts.train (one process per run) and additionally saves the weights
after every check; then compares the runs bitwise at the same number of batches.

Result (2026-11-16), deterministic smokes (STAR-IL trimmed, 2 episodes, --chunk 16
--canonical-quaternion --state-skip, 22 batches/epoch), --val-every 11 vs none:
- after epoch 1 (batch 22): all 86 weight arrays bitwise identical, despite the extra
  full validation at batch 11 (no effect on random or optimizer state)
- after epoch 2 (batch 44): different (max |diff| 6.6e-3). Not the validation: the data
  order. Keras 3.15 (trainers/epoch_iterator.py, _enumerate_iterator) creates two
  iterators between normal epochs (end of epoch + start of next) and each one draws a
  reshuffle, so normal epochs use shuffles #1, #3, #5...; .repeat() uses #1, #2, #3...
  -> config.json "shuffle_order": "keras_epochs" / "repeat".

  python -m scripts.analysis.check_val_every_weights <train args incl. --deterministic>   # one run
  python -m scripts.analysis.check_val_every_weights compare <run with --val-every> <normal run>
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from models.callbacks import TrainingProgress
from scripts import train as T


def weights_path(run, batches):
    return T.OUTPUTS / run / f"after_batch_{batches:05d}.weights.h5"


def save_every_check(run):
    """TrainingProgress + save_weights after every check (reads the weights, changes nothing)."""
    class SaveEveryCheck(TrainingProgress):
        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            self.model.save_weights(str(weights_path(run, (epoch + 1) * self.batches_per_check)))
    return SaveEveryCheck


def read_h5(path):
    """Every dataset in the weights file, by its path inside the file."""
    arrays = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: arrays.__setitem__(name, obj[()])
                     if isinstance(obj, h5py.Dataset) else None)
    return arrays


def same(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def compare(run_checks, run_epochs):
    files = sorted((T.OUTPUTS / run_epochs).glob("after_batch_*.weights.h5"))
    ok = True
    for f in files:
        batches = int(f.stem.split("_")[2].split(".")[0])
        a, b = read_h5(weights_path(run_checks, batches)), read_h5(f)
        equal = same(a, b)
        ok &= equal
        max_diff = max(float(np.max(np.abs(a[k].astype(np.float64) - b[k]))) for k in a if a[k].size)
        print(f"after {batches} batches: {len(a)} weight arrays, bitwise equal: {equal} "
              f"(max |diff| {max_diff:.1e})")
    # control: the test is not trivial (training really moves the weights between checks)
    w11, w22 = read_h5(weights_path(run_checks, 11)), read_h5(weights_path(run_checks, 22))
    print(f"control: batch 11 vs batch 22 differ: {not same(w11, w22)}")
    ok &= not same(w11, w22)
    # the val losses of the matching rows
    h_c = pd.read_csv(T.OUTPUTS / run_checks / "history.csv").set_index("batches_seen")
    h_e = pd.read_csv(T.OUTPUTS / run_epochs / "history.csv").set_index("batches_seen")
    cols = [c for c in h_e.columns if c.startswith("val_")]
    rows = h_c.loc[h_e.index, cols]
    print(rows[["val_action_output_loss", "val_loss"]].assign(
        normal_action=h_e["val_action_output_loss"], normal_total=h_e["val_loss"]).to_string())
    val_equal = bool((rows.values == h_e[cols].values).all())
    print(f"all {len(cols)} val_* columns equal at those rows: {val_equal}")
    ok &= val_equal
    print("SAME TRAINING" if ok else "DIFFERENT")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        args = T.parse_args()
        if not args.deterministic:
            raise SystemExit("this check needs --deterministic")
        # replace the name INSIDE scripts.train: train() looks it up there at call time.
        # Not models.callbacks.TrainingProgress: train.py holds its own reference (from its
        # import), so changing the name in models.callbacks would not reach train()
        T.TrainingProgress = save_every_check(args.run_name)
        T.train(args)
