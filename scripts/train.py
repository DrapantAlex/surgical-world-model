"""File 2: train the actor-world model on the train split, validate on the val split.

- normalisation statistics from the TRAIN split only, saved with the run
- best weights chosen by the validation action loss (weights only, no model.keras)
- logs to TensorBoard (logs/<run>) and Weights & Biases, plus the GT/PRED timeline
- at the end, the best weights are evaluated on val with models/metrics.py (the same
  code as scripts/baselines.py), saved as JSON and appended to outputs/results.csv
The test split is never loaded.

  python -m scripts.train --run-name smoke --max-episodes 2 --epochs 2 --wandb-mode offline
  python -m scripts.train --run-name smoke --eval-only
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from tensorflow import keras

from models import data, metrics
from models.encoders import ENCODERS
from models.pipeline import make_dataset
from models.simple_world_model.swm import build_actor_world_model
from models.visualization import WorldModelTimelineCallback

# --- 0: fixed choices (deliberately not command-line options) ---
LOSS_WEIGHTS = {"action_output": 1.0, "next_state_output": 1.0, "next_image_output": 0.1}
MONITOR = "val_action_output_loss"     # best checkpoint and early stopping
WANDB_PROJECT = "surgical-world-model"
OUTPUTS, LOGS = Path("outputs"), Path("logs")
BASELINES_JSON = OUTPUTS / "baselines" / "baselines.json"


# --- A: command line ---
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", help="default: built from encoder, freeze, latent, seed, time")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10, help="early stopping, in epochs")
    p.add_argument("--latent-dim", type=lambda v: None if v.lower() == "none" else int(v),
                   default=512, help="512 or none")
    p.add_argument("--encoder", choices=ENCODERS, default="simple_cnn")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--unfreeze-last-n", type=int, default=0,
                   help="with --freeze-encoder: train the last N backbone layers again")
    p.add_argument("--max-episodes", type=int,
                   help="only the first N train and N val episodes (smoke test)")
    p.add_argument("--vis-every", type=int, default=5, help="timeline image every N epochs")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--deterministic", action="store_true", help="deterministic TF ops (slower)")
    p.add_argument("--eval-only", action="store_true",
                   help="re-evaluate the saved best weights of outputs/<run-name> on val")
    args = p.parse_args()
    if args.unfreeze_last_n and not args.freeze_encoder:
        p.error("--unfreeze-last-n needs --freeze-encoder")
    if args.eval_only and not args.run_name:
        p.error("--eval-only needs --run-name")
    return args


# --- B: helpers shared by training and --eval-only ---
def freeze_label(cfg):
    if not cfg["freeze_encoder"]:
        return "trainable"
    return f"frozen+last{cfg['unfreeze_last_n']}" if cfg["unfreeze_last_n"] else "frozen"

def episodes_for(split, max_episodes):
    """None = the whole split (read from the frame cache); else its first N episodes."""
    return None if max_episodes is None else data.get_split_episodes(split)[:max_episodes]

def build_model(cfg):
    model = build_actor_world_model(latent_dim=cfg["latent_dim"], encoder=cfg["encoder"],
                                    freeze_encoder=cfg["freeze_encoder"],
                                    unfreeze_last_n=cfg["unfreeze_last_n"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]),
                  loss={"action_output": keras.losses.Huber(),
                        "next_state_output": "mse",
                        "next_image_output": "mse"},
                  loss_weights=LOSS_WEIGHTS)
    return model

def evaluate(model, val, stats, batch_size):
    """Predict every val pair (fixed order) and measure it with models/metrics.py."""
    pred_action, pred_state, image = [], [], metrics.ImageMSE()
    for inputs, targets in make_dataset(val, batch_size, shuffle=False):
        a, s, img = model(inputs, training=False)
        pred_action.append(a.numpy())
        pred_state.append(s.numpy())
        image.update(img.numpy(), targets["next_image_output"].numpy())
    t = val["pair_index"]
    m = metrics.action_metrics(np.concatenate(pred_action), val["actions"][t], stats)
    m.update(metrics.next_state_metrics(np.concatenate(pred_state), val["states"][t + 1], stats))
    m["image_mse"] = image.result()
    return m

def comparison_table(model_name, model_metrics):
    """The model's metrics next to the baselines (if scripts/baselines.py has been run)."""
    columns = {model_name: model_metrics}
    if BASELINES_JSON.exists():
        columns.update(json.loads(BASELINES_JSON.read_text())["results"])
    return metrics.format_table(columns)

def git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+uncommitted" if dirty else "")
    except OSError:
        return "unknown"

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


# --- C: callbacks ---
class EpochTimer(keras.callbacks.Callback):
    """Adds the duration of each epoch to the logs, so every later callback records it."""
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["epoch_time_s"] = time.time() - self.t0

class CheckMonitor(keras.callbacks.Callback):
    """Stops at once if Keras does not log MONITOR (loss names differ between versions)."""
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 and MONITOR not in (logs or {}):
            raise KeyError(f"{MONITOR!r} is not in the Keras logs: {sorted(logs or {})}")

class WandbLogger(keras.callbacks.Callback):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        self.run.log({k: float(v) for k, v in (logs or {}).items()}, step=epoch)

def timeline_callback(val, run, log_dir, every):
    """The existing GT/PRED timeline, on the first validation episode (8 frames, 10 apart)."""
    ep = int(val["episode_index"][0])
    t = val["pair_index"][val["episode_index"][val["pair_index"]] == ep]
    inputs, targets = data.get_pairs(val, t)
    return WorldModelTimelineCallback(log_dir, inputs, targets, data.load_instruction(ep),
                                      wandb_run=run, num_frames=8, start=0, stride=10,
                                      every=every)


# --- D: training ---
def train(args):
    run_name = args.run_name or (f"{args.encoder}-{freeze_label(vars(args))}-lat{args.latent_dim}"
                                 f"-s{args.seed}-{datetime.now():%Y%m%d-%H%M}")
    out, log_dir = OUTPUTS / run_name, LOGS / run_name
    if out.exists():
        raise SystemExit(f"{out} already exists: choose another --run-name")
    out.mkdir(parents=True)

    # reproducibility: Python, NumPy and TensorFlow seeds (+ the tf.data shuffle seed)
    keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    # data: training-only statistics, saved with the run; never the test split
    stats = data.compute_norm_stats(data.get_split_episodes("train"))
    data.save_norm_stats(stats, out / "norm_stats.json")
    train_split = data.load_split("train", stats=stats,
                                  episodes=episodes_for("train", args.max_episodes))
    val = data.load_split("val", stats=stats, episodes=episodes_for("val", args.max_episodes))
    train_ds = make_dataset(train_split, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(val, args.batch_size, shuffle=False)

    # model + full configuration, saved before training starts
    cfg = {k: v for k, v in vars(args).items() if k != "eval_only"}
    cfg["run_name"] = run_name
    model = build_model(cfg)
    cfg.update(
        freeze=freeze_label(cfg), monitor=MONITOR, loss_weights=LOSS_WEIGHTS,
        params_total=int(model.count_params()),
        params_trainable=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        n_train_pairs=int(len(train_split["pair_index"])), n_val_pairs=int(len(val["pair_index"])),
        train_episodes=np.unique(train_split["episode_index"]).tolist(),
        val_episodes=np.unique(val["episode_index"]).tolist(),
        tensorflow=tf.__version__, keras=getattr(keras, "__version__", "unknown"),
        git_commit=git_commit(), started=datetime.now().isoformat(timespec="seconds"))
    save_json(cfg, out / "config.json")
    print(f"run {run_name}: {cfg['params_total']:,} parameters "
          f"({cfg['params_trainable']:,} trainable), {cfg['n_train_pairs']} train pairs, "
          f"{cfg['n_val_pairs']} val pairs")

    run = wandb.init(project=WANDB_PROJECT, name=run_name, config=cfg, mode=args.wandb_mode)
    callbacks = [
        EpochTimer(),
        CheckMonitor(),
        keras.callbacks.ModelCheckpoint(str(out / "best.weights.h5"), monitor=MONITOR, mode="min",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=MONITOR, mode="min", patience=args.patience,
                                      verbose=1),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
        timeline_callback(val, run, log_dir, args.vis_every),
        WandbLogger(run),
    ]
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, verbose=2)
    train_time_min = (time.time() - t0) / 60

    # --- E: evaluate the BEST weights with the same metrics as the baselines ---
    model.load_weights(str(out / "best.weights.h5"))
    val_metrics = evaluate(model, val, stats, args.batch_size)
    monitored = history.history[MONITOR]
    cfg.update(epochs_run=len(monitored), best_epoch=int(np.argmin(monitored)) + 1,
               train_time_min=round(train_time_min, 2))
    save_json(cfg, out / "config.json")
    save_json(val_metrics, out / "val_metrics.json")
    run.summary.update({f"best/{k}": v for k, v in val_metrics.items()})
    run.summary.update({"best_epoch": cfg["best_epoch"], "train_time_min": cfg["train_time_min"]})

    print(f"\nbest epoch {cfg['best_epoch']} of {cfg['epochs_run']}, "
          f"training {train_time_min:.1f} min\n")
    print(comparison_table(run_name, val_metrics))
    if args.max_episodes is None:
        metrics.append_results({"run_name": run_name, "encoder": args.encoder,
                                "freeze": cfg["freeze"], "seed": args.seed,
                                "latent_dim": args.latent_dim, "epochs": cfg["epochs_run"],
                                "best_epoch": cfg["best_epoch"],
                                "train_time_min": cfg["train_time_min"], **val_metrics})
        print(f"\nadded a row to {metrics.RESULTS_CSV}")
    else:
        print("\nsmoke run (--max-episodes): not added to results.csv")
    run.finish()


# --- F: re-evaluate a finished run with ITS saved statistics ---
def eval_only(args):
    out = OUTPUTS / args.run_name
    cfg = json.loads((out / "config.json").read_text())
    stats = data.load_norm_stats(out / "norm_stats.json")
    val = data.load_split("val", stats=stats, episodes=episodes_for("val", cfg["max_episodes"]))
    model = build_model(cfg)
    model.load_weights(str(out / "best.weights.h5"))
    val_metrics = evaluate(model, val, stats, cfg["batch_size"])
    save_json(val_metrics, out / "val_metrics_reeval.json")
    print(comparison_table(f"{args.run_name} (re-eval)", val_metrics))

    saved = out / "val_metrics.json"
    if saved.exists():
        before = json.loads(saved.read_text())
        diff = max(abs(before[k] - val_metrics[k]) for k in before)
        print(f"\nlargest difference from val_metrics.json: {diff:.2e}")


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        train(args)
