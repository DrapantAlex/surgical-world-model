"""Keras callbacks that add values to each check's logs (time, real x-axis, reference
lines, state-group curves) or send them to W&B. Shared by scripts/train.py and
scripts/train_state_only.py. Moved here from scripts/train.py on 2026-11-19 (refactor
step 4a), unchanged. CheckMonitor stays in train.py, next to MONITOR.
"""
import time

import numpy as np
from tensorflow import keras


class EpochTimer(keras.callbacks.Callback):
    """Adds the duration of each epoch to the logs, so every later callback records it."""
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["epoch_time_s"] = time.time() - self.t0

class TrainingProgress(keras.callbacks.Callback):
    """Adds the real x-axis to each check's logs: training batches seen so far and the
    same in real epochs. One Keras "epoch" = one check = batches_per_check batches
    (--val-every N: N; normal runs: one full epoch)."""
    def __init__(self, batches_per_check, batches_per_epoch):
        super().__init__()
        self.batches_per_check, self.batches_per_epoch = batches_per_check, batches_per_epoch

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["batches_seen"] = (epoch + 1) * self.batches_per_check
            logs["epochs_seen"] = logs["batches_seen"] / self.batches_per_epoch

class ConstantLogs(keras.callbacks.Callback):
    """Adds fixed reference values (e.g. the f = 0 val loss) to every epoch's logs,
    so they appear as flat lines next to the real curves (CSV, TensorBoard, W&B)."""
    def __init__(self, values):
        super().__init__()
        self.values = values

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs.update(self.values)

class StateGroupLogs(keras.callbacks.Callback):
    """Adds (val_)next_state_<group>_mse = the mean of the group's per-dimension metrics
    (state_dim_metrics). Every per-dim metric averages over the same samples, so this is
    exactly the group's MSE. groups: {group metric name: [dim metric names]}."""
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        for prefix in ("", "val_"):
            for g, dims in self.groups.items():
                keys = [f"{prefix}next_state_output_mse_{d}" for d in dims]
                if all(k in logs for k in keys):
                    logs[f"{prefix}next_state_{g}_mse"] = float(np.mean([logs[k] for k in keys]))

class WandbLogger(keras.callbacks.Callback):
    """Every log value except the per-dimension state columns and their _copy lines
    (kept in history.csv and TensorBoard; W&B gets the per-group columns instead)."""
    def __init__(self, run):
        super().__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        self.run.log({k: float(v) for k, v in (logs or {}).items()
                      if "next_state_output_mse_" not in k}, step=epoch)
