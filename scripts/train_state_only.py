"""E2 / B: train the state-only actor (state t [+ history] -> action t or chunk, no image).

Everything except the model is the same as scripts/train.py: train-only normalisation
statistics (run_config.prepare_stats), optional canonical quaternions, the same episodes and
pairs (episodes without video are dropped here too), Huber loss, Adam, early stopping on the
val action loss. Frames are never loaded.

E2 (no new flag): action t from state t; at the end the best weights are measured with
scripts/analysis/check_orientation_error.py (check_orientation_error.json, read by compare_runs.py).
NEW (B): --chunk, --history, --keep-ranges, --val-every with the meaning of scripts/train.py:
the history runs' actor without the image and the WM. With --chunk / --history the logs use
the names of scripts/train.py (val_action_output_loss, ..._huber, the reference lines).

  python -m scripts.train_state_only --run-name smoke-state-only --max-episodes 2 --epochs 2 \
      --dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il --canonical-quaternion --wandb-mode offline
  python -m scripts.train_state_only ... --chunk 16 --history \
      --keep-ranges configs/keep_ranges/star_il_tool_visible.csv --val-every 1000   # experiment B
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from tensorflow import keras

from models import data, metrics
from models.callbacks import (AliasLogs, CheckMonitor, ConstantLogs, EpochTimer,
                              TrainingProgress, WandbLogger)
from models.evaluation import chunk_baseline_metrics, comparison_table
from models.references import constvel_huber, standstill_huber, trainmean_huber
from models.run_config import (OUTPUTS, check_schedule, episodes_for, git_commit,
                               prepare_stats, save_json)
from models.state_only import HIDDEN, build_state_only_actor
from scripts.analysis.check_orientation_error import check_run
from scripts.train import WANDB_PROJECT

MONITOR_E2 = "val_loss"                 # old E2: one output only, the loss IS the action loss
MONITOR = "val_action_output_loss"      # NEW (B): the same value under train.py's name (AliasLogs)
ALIASES = {"action_output_loss": "loss", "action_output_huber": "huber"}   # NEW (B)


# --- A: command line ---
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--dataset-root", default=data.ROOT.as_posix())
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10, help="early stopping, in epochs")
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--max-episodes", type=int,
                   help="only the first N train and N val episodes (smoke test)")
    p.add_argument("--canonical-quaternion", action="store_true")
    # NEW (B): the same options as scripts/train.py
    p.add_argument("--chunk", type=int, default=None,
                   help="E8: actor predicts this many offsets a(t+i) - a(t-1), option-C scale "
                        "per step and group (saved in norm_stats.json); default: a(t) only")
    p.add_argument("--history", action="store_true",
                   help="history input: h(t) = [Δs(t), Δa(t-1)] / train std (scale saved in "
                        "norm_stats.json); 0 where the earlier row is missing")
    p.add_argument("--keep-ranges", default=None,
                   help="CSV episode, first_visible, last_visible: use only those frames for "
                        "pairs and statistics (copied into the run folder, with its hash)")
    p.add_argument("--val-every", type=int, default=None,
                   help="validate (full val split) every N training batches instead of once per "
                        "epoch; --epochs and --patience stay in real epochs")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = p.parse_args()
    if args.val_every is not None and args.val_every <= 0:
        p.error("--val-every must be a positive number of batches")
    # keep-ranges file checked here, before the run folder exists (as in scripts/train.py)
    if args.keep_ranges:
        if not Path(args.keep_ranges).is_file():
            p.error(f"--keep-ranges: {args.keep_ranges} not found")
        try:   # exactly train + val of this dataset
            data.check_keep_ranges(data.load_keep_ranges(args.keep_ranges), Path(args.dataset_root))
        except ValueError as e:
            p.error(f"--keep-ranges: {e}")
    return args

def new_mode(args):
    """NEW (B): the history runs' setup (chunk and/or history): train.py's log names."""
    return bool(args.chunk or args.history)


# --- B: data: the model's inputs and targets for every pair t, as in get_pairs (no image) ---
def xy(split):
    t = split["pair_index"]
    if "chunk_factor" not in split and "history" not in split:
        return split["states"][t], split["actions"][t]          # old E2: unchanged
    x = {"state_input": split["states"][t]}                     # NEW (B): named inputs
    if "history" in split:
        x["history_input"] = split["history"][t]                # [Δs(t), Δa(t-1)] / scale
    y = data.chunk_targets(split, t) if "chunk_factor" in split else split["actions"][t]
    return x, y

def make_dataset(split, batch_size, shuffle=False, seed=0, repeat=False):
    x, y = xy(split)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(split["pair_index"]), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if repeat:   # NEW (B), --val-every: after batch, as models/pipeline.py
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)

def prepare_data(args, out):
    """Statistics (saved in `out`), both splits (tables only) and the actor's reference lines.
    Shared by main() and the section C check, so both run exactly this code."""
    root = Path(args.dataset_root)
    stats, quat_ref, keep_ranges, keep_hash = prepare_stats(
        root, out, args.canonical_quaternion, args.keep_ranges, args.chunk, args.history)
    splits = {s: data.load_split(s, root=root, stats=stats, quat_ref=quat_ref, with_frames=False,
                                 episodes=episodes_for(s, args.max_episodes, root),
                                 chunk=args.chunk, keep_ranges=keep_ranges, history=args.history)
              for s in ("train", "val")}
    reference_logs = {}
    if args.chunk:   # E8: stand-still, input-free mean train offset, constant velocity
        reference_logs["val_action_output_loss_standstill"] = standstill_huber(splits["val"])
        reference_logs["val_action_output_loss_trainmean"] = trainmean_huber(splits["train"],
                                                                             splits["val"])
        reference_logs["val_action_output_loss_constvel"] = constvel_huber(splits["val"], stats)
    return root, stats, keep_hash, splits, reference_logs


# --- C: training ---
def main():
    args = parse_args()
    out = OUTPUTS / args.run_name
    if out.exists():
        raise SystemExit(f"{out} already exists: choose another --run-name")
    out.mkdir(parents=True)
    keras.utils.set_random_seed(args.seed)

    # same statistics, pairs and labels as scripts/train.py (train split only)
    root, stats, keep_hash, splits, reference_logs = prepare_data(args, out)
    train_split, val = splits["train"], splits["val"]
    # vis_every=1: no timeline here (no image); only the other schedule values are used
    schedule = check_schedule(len(train_split["pair_index"]), args.batch_size, args.epochs,
                              args.patience, 1, args.val_every)
    schedule.pop("vis_every_checks")

    state_dim, action_dim = data.feature_dims(root)
    model = build_state_only_actor(state_dim, action_dim, args.hidden, action_chunk=args.chunk,
                                   history_dim=state_dim + action_dim if args.history else 0)
    monitor = MONITOR if new_mode(args) else MONITOR_E2
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                  loss=keras.losses.Huber(),   # the actor loss of scripts/train.py
                  # NEW (B): the exact per-pair Huber, as run_config.build_model
                  metrics=([keras.metrics.MeanMetricWrapper(keras.losses.huber, name="huber")]
                           if new_mode(args) else None))

    cfg = {**vars(args), "model_type": "state_only", "encoder": "state_only",
           "monitor": monitor, "params_total": int(model.count_params()),
           "n_train_pairs": int(len(train_split["pair_index"])),
           "n_val_pairs": int(len(val["pair_index"])),
           "train_episodes": np.unique(train_split["episode_index"]).tolist(),
           "val_episodes": np.unique(val["episode_index"]).tolist(),
           "state_dim": state_dim, "action_dim": action_dim,
           "tensorflow": tf.__version__, "git_commit": git_commit(),
           "started": datetime.now().isoformat(timespec="seconds")}
    cfg.update(   # NEW (B): the records of scripts/train.py that exist without a WM
        **reference_logs, **schedule,
        shuffle_order="repeat" if args.val_every else "keras_epochs",
        n_train_pairs_untrimmed=int(train_split["n_pairs_untrimmed"]),
        n_val_pairs_untrimmed=int(val["n_pairs_untrimmed"]),
        n_train_history_fallback=train_split.get("n_history_fallback"),
        n_val_history_fallback=val.get("n_history_fallback"),
        keep_ranges_sha256=keep_hash,
        train_skipped_episodes=train_split["skipped_episodes"],
        val_skipped_episodes=val["skipped_episodes"],
        keras=getattr(keras, "__version__", "unknown"))
    save_json(cfg, out / "config.json")
    print(f"run {args.run_name}: {cfg['params_total']:,} parameters, "
          f"{cfg['n_train_pairs']} train pairs, {cfg['n_val_pairs']} val pairs")

    run = wandb.init(project=WANDB_PROJECT, name=args.run_name, config=cfg, mode=args.wandb_mode)
    # order matters: the aliases and reference lines are in the logs before checkpointing
    new = new_mode(args)
    callbacks = [
        EpochTimer(),
        *([TrainingProgress(schedule["batches_per_check"], schedule["batches_per_epoch"]),
           AliasLogs(ALIASES), CheckMonitor(monitor)] if new else []),   # NEW (B)
        *([ConstantLogs(reference_logs)] if reference_logs else []),     # NEW (B): chunk only
        keras.callbacks.ModelCheckpoint(str(out / "best.weights.h5"), monitor=monitor, mode="min",
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.EarlyStopping(monitor=monitor, mode="min",
                                      patience=schedule["patience_checks"], verbose=1),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
        WandbLogger(run),
    ]
    t0 = time.time()
    history = model.fit(make_dataset(train_split, args.batch_size, shuffle=True, seed=args.seed,
                                     repeat=bool(args.val_every)),
                        validation_data=make_dataset(val, args.batch_size),
                        epochs=schedule["max_checks"],
                        steps_per_epoch=args.val_every,   # None = one full pass per "epoch"
                        callbacks=callbacks, verbose=2)
    monitored = history.history[monitor]
    # best_epoch = index of the best check (= the real epoch for normal runs, as before)
    cfg.update(epochs_run=len(monitored), best_epoch=int(np.argmin(monitored)) + 1,
               train_time_min=round((time.time() - t0) / 60, 2))
    cfg["best_batch"] = cfg["best_epoch"] * schedule["batches_per_check"]
    cfg["best_epoch_equiv"] = cfg["best_batch"] / schedule["batches_per_epoch"]
    save_json(cfg, out / "config.json")

    # --- D: evaluate the BEST weights: normalised metrics + mm / degrees (same code as train.py) ---
    model.load_weights(str(out / "best.weights.h5"))
    predict = lambda split: model.predict(xy(split)[0], batch_size=1024, verbose=0)
    action_names = data.feature_names(root)[1]
    t = val["pair_index"]
    chunk_base = None
    if args.chunk:   # NEW (B): the chunk branch of models/evaluation.evaluate, + its baselines
        val_metrics = metrics.chunk_metrics(predict(val), data.chunk_targets(val, t),
                                            data.chunk_anchors(val, t)[0], stats, action_names)
        chunk_base = chunk_baseline_metrics(val, stats, root)
        save_json(chunk_base, out / "chunk_baselines.json")
    else:
        val_metrics = metrics.action_metrics(predict(val), val["actions"][t], stats,
                                             metrics.has_tissue_lift_layout(root), action_names)
    save_json(val_metrics, out / "val_metrics.json")
    run.summary.update({f"best/{k}": v for k, v in val_metrics.items()})
    run.summary.update({"best_epoch": cfg["best_epoch"], "best_batch": cfg["best_batch"],
                        "best_epoch_equiv": cfg["best_epoch_equiv"],
                        "train_time_min": cfg["train_time_min"]})
    unit = "check" if args.val_every else "epoch"
    print(f"\nbest {unit} {cfg['best_epoch']} of {cfg['epochs_run']} (batch {cfg['best_batch']}, "
          f"{cfg['best_epoch_equiv']:.2f} epochs)\n")
    if args.chunk:
        print(comparison_table(args.run_name, val_metrics, root, args.canonical_quaternion,
                               chunk_base, keep_hash))
    else:
        print(metrics.format_table({args.run_name: val_metrics}))

    smoke = args.max_episodes is not None
    # check_orientation_error loads its own splits (no chunk, history or keep ranges): old E2 only
    if not (new or args.keep_ranges):
        check_run(out, cfg, stats, predict, with_frames=False,
                  train_episodes=min(20, len(cfg["train_episodes"])),
                  val_episodes=args.max_episodes)
    else:
        print("\ncheck_orientation_error skipped (not chunk / history / keep-ranges aware yet)")

    if smoke:
        print("\nsmoke run (--max-episodes): not added to results.csv")
    else:
        metrics.append_results({"run_name": args.run_name, "dataset": root.name,
                                "encoder": "state_only",
                                "canonical_quaternion": args.canonical_quaternion,
                                "seed": args.seed, "chunk": args.chunk, "history": args.history,
                                "keep_ranges": Path(args.keep_ranges).name if args.keep_ranges else None,
                                "epochs": cfg["epochs_run"], "best_epoch": cfg["best_epoch"],
                                "val_every": args.val_every, "best_batch": cfg["best_batch"],
                                "train_time_min": cfg["train_time_min"], **val_metrics})
        print(f"\nadded a row to {metrics.RESULTS_CSV}")
    run.finish()


if __name__ == "__main__":
    main()
