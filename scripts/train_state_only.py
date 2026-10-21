"""E2: train the state-only actor (state t -> action t, no image) and measure it in degrees.

Everything except the model is the same as scripts/train.py: train-only normalisation
statistics, optional canonical quaternions, the same episodes and pairs (episodes without
video are dropped here too), Huber loss, Adam, early stopping on the val action loss.
Frames are never loaded. At the end the best weights are measured with
scripts/check_orientation_error.py (check_orientation_error.json, read by compare_runs.py).

  python -m scripts.train_state_only --run-name smoke-state-only --max-episodes 2 --epochs 2 \
      --dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il --canonical-quaternion --wandb-mode offline
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
from models.state_only import HIDDEN, build_state_only_actor
from scripts.check_orientation_error import check_run
from scripts.train import (OUTPUTS, WANDB_PROJECT, EpochTimer, WandbLogger, episodes_for,
                           git_commit, save_json)

MONITOR = "val_loss"   # one output only: the loss IS the action loss


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
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


# --- B: data: (state t, action t) for every pair t, as in get_pairs ---
def xy(split):
    t = split["pair_index"]
    return split["states"][t], split["actions"][t]

def make_dataset(split, batch_size, shuffle=False, seed=0):
    x, y = xy(split)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --- C: training ---
def main():
    args = parse_args()
    out = OUTPUTS / args.run_name
    if out.exists():
        raise SystemExit(f"{out} already exists: choose another --run-name")
    out.mkdir(parents=True)
    keras.utils.set_random_seed(args.seed)

    # same statistics and labels as scripts/train.py (train split only)
    root = Path(args.dataset_root)
    train_eps, _ = data.drop_missing_videos("train", data.get_split_episodes("train", root),
                                            root=root, verbose=False)
    quat_ref = None
    if args.canonical_quaternion:
        quat_ref = data.compute_quat_reference(train_eps, root)
        data.save_quat_reference(quat_ref, out / "quat_reference.json")
    stats = data.compute_norm_stats(train_eps, root, quat_ref)
    data.save_norm_stats(stats, out / "norm_stats.json")
    splits = {s: data.load_split(s, root=root, stats=stats, quat_ref=quat_ref, with_frames=False,
                                 episodes=episodes_for(s, args.max_episodes, root))
              for s in ("train", "val")}

    state_dim, action_dim = data.feature_dims(root)
    model = build_state_only_actor(state_dim, action_dim, args.hidden)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                  loss=keras.losses.Huber())   # the actor loss of scripts/train.py

    cfg = {**vars(args), "model_type": "state_only", "encoder": "state_only",
           "monitor": MONITOR, "params_total": int(model.count_params()),
           "n_train_pairs": int(len(splits["train"]["pair_index"])),
           "n_val_pairs": int(len(splits["val"]["pair_index"])),
           "train_episodes": np.unique(splits["train"]["episode_index"]).tolist(),
           "val_episodes": np.unique(splits["val"]["episode_index"]).tolist(),
           "state_dim": state_dim, "action_dim": action_dim,
           "tensorflow": tf.__version__, "git_commit": git_commit(),
           "started": datetime.now().isoformat(timespec="seconds")}
    save_json(cfg, out / "config.json")
    print(f"run {args.run_name}: {cfg['params_total']:,} parameters, "
          f"{cfg['n_train_pairs']} train pairs, {cfg['n_val_pairs']} val pairs")

    run = wandb.init(project=WANDB_PROJECT, name=args.run_name, config=cfg, mode=args.wandb_mode)
    callbacks = [
        EpochTimer(),
        keras.callbacks.ModelCheckpoint(str(out / "best.weights.h5"), monitor=MONITOR, mode="min",
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.EarlyStopping(monitor=MONITOR, mode="min", patience=args.patience,
                                      verbose=1),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
        WandbLogger(run),
    ]
    t0 = time.time()
    history = model.fit(make_dataset(splits["train"], args.batch_size, shuffle=True, seed=args.seed),
                        validation_data=make_dataset(splits["val"], args.batch_size),
                        epochs=args.epochs, callbacks=callbacks, verbose=2)
    monitored = history.history[MONITOR]
    cfg.update(epochs_run=len(monitored), best_epoch=int(np.argmin(monitored)) + 1,
               train_time_min=round((time.time() - t0) / 60, 2))
    save_json(cfg, out / "config.json")

    # --- D: evaluate the BEST weights: normalised metrics + degrees (same code as E1) ---
    model.load_weights(str(out / "best.weights.h5"))
    predict = lambda split: model.predict(xy(split)[0], batch_size=1024, verbose=0)
    val_metrics = metrics.action_metrics(predict(splits["val"]), xy(splits["val"])[1], stats,
                                         metrics.has_tissue_lift_layout(root),
                                         data.feature_names(root)[1])
    save_json(val_metrics, out / "val_metrics.json")
    run.summary.update({f"best/{k}": v for k, v in val_metrics.items()})
    print(f"\nbest epoch {cfg['best_epoch']} of {cfg['epochs_run']}\n")
    print(metrics.format_table({args.run_name: val_metrics}))

    smoke = args.max_episodes is not None
    check_run(out, cfg, stats, predict, with_frames=False,
              train_episodes=min(20, len(cfg["train_episodes"])),
              val_episodes=args.max_episodes)

    if smoke:
        print("\nsmoke run (--max-episodes): not added to results.csv")
    else:
        metrics.append_results({"run_name": args.run_name, "dataset": root.name,
                                "encoder": "state_only",
                                "canonical_quaternion": args.canonical_quaternion,
                                "seed": args.seed, "epochs": cfg["epochs_run"],
                                "best_epoch": cfg["best_epoch"],
                                "train_time_min": cfg["train_time_min"], **val_metrics})
        print(f"\nadded a row to {metrics.RESULTS_CSV}")
    run.finish()


if __name__ == "__main__":
    main()
