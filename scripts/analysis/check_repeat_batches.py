"""Throwaway check for --val-every: does .repeat() after .batch() give exactly the batches
of plain tf.data passes (new iterator per pass; same order, same smaller last batch)?
Indices only (no frames, no model), same shuffle + batch as models/pipeline.make_dataset.

Result (2026-11-16): IDENTICAL, 64 vs 64 batches over 2 passes, last batch of 11 kept per
pass, pass 2 reshuffled.
Caveat: plain passes are NOT what Keras fit does. Keras 3.15 discards every second
iterator between epochs (shuffles #1, #3, ...), so with Keras only epoch 1 matches:
see scripts/analysis/check_val_every_weights.py.

  python -m scripts.analysis.check_repeat_batches
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

N_PAIRS, BATCH, SEED, PASSES = 1003, 32, 0, 2   # 1003 % 32 = 11: a smaller last batch


def index_batches(repeat):
    """The index part of make_dataset (section C), with the global seed set as in train()."""
    keras.utils.set_random_seed(SEED)
    ds = tf.data.Dataset.from_tensor_slices(np.arange(N_PAIRS, dtype=np.int64))
    ds = ds.shuffle(N_PAIRS, seed=SEED, reshuffle_each_iteration=True).batch(BATCH)
    return ds.repeat() if repeat else ds


def main():
    # normal training: Keras creates a new iterator of the same dataset every epoch
    ds = index_batches(repeat=False)
    per_epoch = [b.numpy() for _ in range(PASSES) for b in ds]

    # --val-every: one iterator over the repeated dataset, never restarted
    per_batches_per_pass = int(np.ceil(N_PAIRS / BATCH))
    it = iter(index_batches(repeat=True))
    repeated = [next(it).numpy() for _ in range(PASSES * per_batches_per_pass)]

    print(f"{len(per_epoch)} vs {len(repeated)} batches, last batch of pass 1: "
          f"{len(per_epoch[per_batches_per_pass - 1])} vs {len(repeated[per_batches_per_pass - 1])}")
    same = len(per_epoch) == len(repeated) and all(np.array_equal(a, b)
                                                   for a, b in zip(per_epoch, repeated))
    pass1, pass2 = np.concatenate(per_epoch[:per_batches_per_pass]), np.concatenate(per_epoch[per_batches_per_pass:])
    print(f"each pass a permutation: {sorted(pass1) == list(range(N_PAIRS)) == sorted(pass2)}, "
          f"pass 2 reshuffled: {not np.array_equal(pass1, pass2)}")
    print("IDENTICAL" if same else "DIFFERENT")
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
