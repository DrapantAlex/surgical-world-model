"""File 2: train the actor-world model on the train split, validate on the val split.

- normalisation statistics from the TRAIN split only, saved with the run
- best weights chosen by the validation action loss (weights only, no model.keras)
- logs to TensorBoard (logs/<run>) and Weights & Biases, plus the GT/PRED timeline
- at the end, the best weights are evaluated on val with models/metrics.py (the same
  code as scripts/baselines.py), saved as JSON and appended to outputs/results.csv
The test split is never loaded.

  python -m scripts.train --run-name smoke --max-episodes 2 --epochs 2 --wandb-mode offline
  python -m scripts.train --run-name smoke --eval-only
  python -m scripts.train --dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il
  python -m scripts.train ... --w-next-state 0 --w-next-image 0   # world model off (actor only)
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
# the action weight is fixed; the two world-model weights are only the DEFAULTS of
# --w-next-state / --w-next-image (and the values of runs saved before those options)
LOSS_WEIGHTS = {"action_output": 1.0, "next_state_output": 1.0, "next_image_output": 0.1}
MONITOR = "val_action_output_loss"     # best checkpoint and early stopping
WANDB_PROJECT = "surgical-world-model"
OUTPUTS, LOGS = Path("outputs"), Path("logs")


# --- A: command line ---
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", help="default: built from encoder, freeze, latent, seed, time")
    p.add_argument("--dataset-root", default=data.ROOT.as_posix(),
                   help="LeRobot dataset folder (default: tissue lift)")
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
    p.add_argument("--canonical-quaternion", action="store_true",
                   help="flip every quaternion label to the sign of the train mean rotation "
                        "(saved as quat_reference.json); off = the dataset's own signs")
    p.add_argument("--w-next-state", type=float, default=LOSS_WEIGHTS["next_state_output"],
                   help="loss weight of the next-state head (0 = no gradient from it)")
    p.add_argument("--w-next-image", type=float, default=LOSS_WEIGHTS["next_image_output"],
                   help="loss weight of the next-image head (0 = no gradient from it)")
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
    if args.w_next_state < 0 or args.w_next_image < 0:
        p.error("loss weights must be >= 0")
    return args


# --- B: helpers shared by training and --eval-only ---
def freeze_label(cfg):
    if not cfg["freeze_encoder"]:
        return "trainable"
    return f"frozen+last{cfg['unfreeze_last_n']}" if cfg["unfreeze_last_n"] else "frozen"

def dataset_root(cfg):
    """Runs saved before --dataset-root existed were all tissue lift."""
    return Path(cfg.get("dataset_root", data.ROOT))

def run_quat_ref(cfg, out):
    """The run's saved reference rotations, or None (runs before the option: raw signs)."""
    if not cfg.get("canonical_quaternion", False):
        return None
    return data.load_quat_reference(out / "quat_reference.json")

def baselines_json(canonical):
    """Baselines measured with the same quaternion convention as the run: normalised
    MSE is not comparable across conventions (different label std)."""
    return OUTPUTS / ("baselines_canonical" if canonical else "baselines") / "baselines.json"

def episodes_for(split, max_episodes, root):
    """None = the whole split (read from the frame cache); else its first N episodes."""
    return None if max_episodes is None else data.get_split_episodes(split, root)[:max_episodes]

def loss_weights(cfg):
    """Runs saved before --w-next-state / --w-next-image used the LOSS_WEIGHTS defaults."""
    return {"action_output": LOSS_WEIGHTS["action_output"],
            "next_state_output": cfg.get("w_next_state", LOSS_WEIGHTS["next_state_output"]),
            "next_image_output": cfg.get("w_next_image", LOSS_WEIGHTS["next_image_output"])}

def build_model(cfg):
    state_dim, action_dim = data.feature_dims(dataset_root(cfg))
    model = build_actor_world_model(state_dim, action_dim,
                                    latent_dim=cfg["latent_dim"], encoder=cfg["encoder"],
                                    freeze_encoder=cfg["freeze_encoder"],
                                    unfreeze_last_n=cfg["unfreeze_last_n"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]),
                  loss={"action_output": keras.losses.Huber(),
                        "next_state_output": "mse",
                        "next_image_output": "mse"},
                  loss_weights=loss_weights(cfg))
    return model

def evaluate(model, val, stats, batch_size, root):
    """Predict every val pair (fixed order) and measure it with models/metrics.py."""
    pred_action, pred_state, image = [], [], metrics.ImageMSE()
    for inputs, targets in make_dataset(val, batch_size, shuffle=False):
        a, s, img = model(inputs, training=False)
        pred_action.append(a.numpy())
        pred_state.append(s.numpy())
        image.update(img.numpy(), targets["next_image_output"].numpy())
    t = val["pair_index"]
    physical = metrics.has_tissue_lift_layout(root)   # else core metrics only, no "extra/"
    state_names, action_names = data.feature_names(root)
    m = metrics.action_metrics(np.concatenate(pred_action), val["actions"][t], stats, physical,
                               action_names)
    m.update(metrics.next_state_metrics(np.concatenate(pred_state), val["states"][t + 1], stats,
                                        physical, state_names))
    m.update(metrics.next_image_metrics(image))
    return m

def comparison_table(model_name, model_metrics, root, canonical=False):
    """The model's metrics next to the baselines (if scripts/baselines.py has been run
    on the same dataset and quaternion convention; baselines.json files without
    "dataset_root" are tissue lift)."""
    columns = {model_name: model_metrics}
    path = baselines_json(canonical)
    if path.exists():
        baselines = json.loads(path.read_text())
        if Path(baselines.get("dataset_root", data.ROOT)) == Path(root):
            columns.update(baselines["results"])
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

def log_tensorboard(val_metrics, log_dir, step):
    """Final metrics of the best weights, same names as results.csv ("extra/..." its own group)."""
    with tf.summary.create_file_writer(str(Path(log_dir) / "best_val")).as_default():
        for k, v in val_metrics.items():
            tf.summary.scalar(k, v, step=step)


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

def timeline_callback(val, run, log_dir, every, root):
    """The existing GT/PRED timeline, on the first validation episode (8 frames, 10 apart)."""
    ep = int(val["episode_index"][0])
    t = val["pair_index"][val["episode_index"][val["pair_index"]] == ep]
    inputs, targets = data.get_pairs(val, t)
    return WorldModelTimelineCallback(log_dir, inputs, targets, data.load_instruction(ep, root),
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

    # data: training-only statistics, saved with the run; never the test split.
    # compute_norm_stats raises if the train split is incomplete: no fallback here.
    root = Path(args.dataset_root)
    train_eps, _ = data.drop_missing_videos("train", data.get_split_episodes("train", root),
                                            root=root, verbose=False)
    # optional canonical quaternion signs: reference from the full train split, saved
    # with the run, and used for the stats AND both splits (all through load_table)
    quat_ref = None
    if args.canonical_quaternion:
        quat_ref = data.compute_quat_reference(train_eps, root)
        data.save_quat_reference(quat_ref, out / "quat_reference.json")
    stats = data.compute_norm_stats(train_eps, root, quat_ref)
    data.save_norm_stats(stats, out / "norm_stats.json")
    train_split = data.load_split("train", root=root, stats=stats, quat_ref=quat_ref,
                                  episodes=episodes_for("train", args.max_episodes, root))
    val = data.load_split("val", root=root, stats=stats, quat_ref=quat_ref,
                          episodes=episodes_for("val", args.max_episodes, root))
    train_ds = make_dataset(train_split, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(val, args.batch_size, shuffle=False)

    # model + full configuration, saved before training starts
    cfg = {k: v for k, v in vars(args).items() if k != "eval_only"}
    cfg["run_name"] = run_name
    model = build_model(cfg)
    cfg.update(
        freeze=freeze_label(cfg), monitor=MONITOR, loss_weights=loss_weights(cfg),
        params_total=int(model.count_params()),
        params_trainable=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        n_train_pairs=int(len(train_split["pair_index"])), n_val_pairs=int(len(val["pair_index"])),
        train_episodes=np.unique(train_split["episode_index"]).tolist(),
        val_episodes=np.unique(val["episode_index"]).tolist(),
        train_skipped_episodes=train_split["skipped_episodes"],
        val_skipped_episodes=val["skipped_episodes"],
        state_dim=data.feature_dims(root)[0], action_dim=data.feature_dims(root)[1],
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
        timeline_callback(val, run, log_dir, args.vis_every, root),
        WandbLogger(run),
    ]
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, verbose=2)
    train_time_min = (time.time() - t0) / 60

    # --- E: evaluate the BEST weights with the same metrics as the baselines ---
    model.load_weights(str(out / "best.weights.h5"))
    val_metrics = evaluate(model, val, stats, args.batch_size, root)
    monitored = history.history[MONITOR]
    cfg.update(epochs_run=len(monitored), best_epoch=int(np.argmin(monitored)) + 1,
               train_time_min=round(train_time_min, 2))
    save_json(cfg, out / "config.json")
    save_json(val_metrics, out / "val_metrics.json")
    log_tensorboard(val_metrics, log_dir, cfg["best_epoch"])
    run.summary.update({f"best/{k}": v for k, v in val_metrics.items()})
    run.summary.update({"best_epoch": cfg["best_epoch"], "train_time_min": cfg["train_time_min"]})

    print(f"\nbest epoch {cfg['best_epoch']} of {cfg['epochs_run']}, "
          f"training {train_time_min:.1f} min\n")
    print(comparison_table(run_name, val_metrics, root, args.canonical_quaternion))
    if args.max_episodes is None:
        metrics.append_results({"run_name": run_name, "dataset": root.name,
                                "encoder": args.encoder,
                                "canonical_quaternion": args.canonical_quaternion,
                                "freeze": cfg["freeze"], "seed": args.seed,
                                "latent_dim": args.latent_dim,
                                "w_next_state": args.w_next_state,
                                "w_next_image": args.w_next_image, "epochs": cfg["epochs_run"],
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
    root = dataset_root(cfg)
    quat_ref = run_quat_ref(cfg, out)
    val = data.load_split("val", root=root, stats=stats, quat_ref=quat_ref,
                          episodes=episodes_for("val", cfg["max_episodes"], root))
    model = build_model(cfg)
    model.load_weights(str(out / "best.weights.h5"))
    val_metrics = evaluate(model, val, stats, cfg["batch_size"], root)
    save_json(val_metrics, out / "val_metrics_reeval.json")
    print(comparison_table(f"{args.run_name} (re-eval)", val_metrics, root, quat_ref is not None))

    saved = out / "val_metrics.json"
    if saved.exists():
        before = json.loads(saved.read_text())
        common = before.keys() & val_metrics.keys()   # metric names changed over time
        diff = max(abs(before[k] - val_metrics[k]) for k in common)
        print(f"\nlargest difference from val_metrics.json: {diff:.2e}")


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        train(args)
