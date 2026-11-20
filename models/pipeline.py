"""tf.data pipeline: turns a split from models.data.load_split into model batches.

Frames stay uint8 (in RAM, or memory-mapped from data/cache/). Each batch is
built by models.data.get_pairs, which converts only that batch to float, so the
model sees exactly the same pairs as the overfit test and the baselines.
"""
import numpy as np
import tensorflow as tf

from models.data import get_pairs

TARGET_KEYS = ("action_output", "next_state_output", "next_image_output")
# CHANGED (history): the input keys are no longer fixed: they are the keys get_pairs returns
# for this split (image, state, language, action; + history_input when the split has it).


def make_dataset(split, batch_size, shuffle=False, seed=0, repeat=False):
    """Batches of (inputs, targets) dicts over split["pair_index"].
    shuffle=True: a new (seeded) order every epoch; shuffle=False: fixed order.
    repeat=True: the passes one after another, endless (for --val-every: Keras
    steps_per_epoch then cuts this stream into checks of N batches)."""
    pair_index = split["pair_index"].astype(np.int64)
    input_keys = tuple(get_pairs(split, pair_index[:1])[0])   # in get_pairs' order

    # --- A: one batch in NumPy (reuses get_pairs) ---
    def numpy_batch(t):
        inputs, targets = get_pairs(split, t)
        return tuple(inputs[k] for k in input_keys) + tuple(targets[k] for k in TARGET_KEYS)

    # shapes of every array, read from one real example (None = batch size)
    example = numpy_batch(pair_index[:1])
    shapes = [(None,) + a.shape[1:] for a in example]

    # --- B: the same batch inside the TensorFlow graph ---
    def tf_batch(t):
        arrays = tf.numpy_function(numpy_batch, [t], [tf.float32] * len(shapes))
        for a, shape in zip(arrays, shapes):
            a.set_shape(shape)
        n = len(input_keys)
        return dict(zip(input_keys, arrays[:n])), dict(zip(TARGET_KEYS, arrays[n:]))

    # --- C: indices -> shuffle -> batch -> load -> prefetch ---
    ds = tf.data.Dataset.from_tensor_slices(pair_index)
    if shuffle:
        ds = ds.shuffle(len(pair_index), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if repeat:   # NEW: after batch, so every pass keeps its own (smaller) last batch
        ds = ds.repeat()
    ds = ds.map(tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
