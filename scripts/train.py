"""File 2: train the actor-world model on the train split, validate on the val split.

- normalisation statistics from the TRAIN split only, saved with the run
- best weights chosen by the validation action loss (weights only, no model.keras)
- logs to TensorBoard (logs/<run>) and Weights & Biases, plus the GT/PRED timeline
- at the end, the best weights are evaluated on val with models/metrics.py (the same
  code as scripts/baselines.py), saved as JSON and appended to outputs/results.csv
The test split is never loaded.

  python -m scripts.train --run-name smoke --max-episodes 2 --epochs 2 --wandb-mode offline
  python -m scripts.train --run-name smoke --eval-only
  python -m scripts.train --run-name <run> --eval-only --eval-keep-ranges <file>   # other val pairs
  python -m scripts.train --dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il
  python -m scripts.train ... --w-next-state 0 --w-next-image 0   # world model off (actor only)
  python -m scripts.train ... --state-delta   # E7: WM state head predicts the change (skip)
  python -m scripts.train ... --state-skip    # 1a: state + zero-initialised Dense, MSE (starts = copy)
  python -m scripts.train ... --state-layers 3   # 1b: 3 hidden Dense(128) in the state head
  python -m scripts.train ... --wm-mixer-blocks 3   # 1c: 3 ConvMixer blocks in the WM branch
  python -m scripts.train ... --chunk 16      # E8: actor predicts 16 offsets from a(t-1)
  python -m scripts.train ... --keep-ranges configs/keep_ranges/star_il_tool_visible.csv
                                              # only the kept frames (pairs + statistics)
  python -m scripts.train ... --val-every 500 # one check (full val) every 500 train batches
  python -m scripts.train ... --history       # + [Δs(t), Δa(t-1)] as input to both branches
  python -m scripts.train ... --wm-balance 0.3 --wm-state-groups kuka_joint_pos
                                              # (c): each WM term's encoder push held at 0.3 x the actor's
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from tensorflow import keras

from models import data, metrics
from models.balance import WMBalanceTrace
from models.callbacks import (CheckMonitor, ConstantLogs, EpochTimer, StateGroupLogs,
                              TrainingProgress, WandbLogger, timeline_callback)
from models.encoders import ENCODERS
from models.evaluation import chunk_baseline_metrics, comparison_table, evaluate
from models.pipeline import make_dataset
from models.references import (constvel_huber, constvel_state_mse_per_dim, copy_frame_mse,
                               copy_state_mse, copy_state_mse_per_dim, glitch_thresholds,
                               standstill_huber, trainmean_huber)
from models.run_config import (LOSS_WEIGHTS, OUTPUTS, build_model, check_schedule,
                               dataset_root, delta_factor, episodes_for, freeze_label,
                               git_commit, loss_weights, metric_name, prepare_stats,
                               run_history, run_keep_ranges, run_quat_ref, save_json)
from models.simple_world_model.swm import WM_MIXER_KERNEL

# --- 0: fixed choices (deliberately not command-line options) ---
# the loss weights (LOSS_WEIGHTS) live in models/run_config.py: rebuilding a run needs them
MONITOR = "val_action_output_loss"     # best checkpoint and early stopping (kept as in every
                                       # earlier run; the exact per-pair value is val_action_output_huber)
WANDB_PROJECT = "surgical-world-model"
# DELTA_FLOOR_C / CHUNK_FLOOR_C (the Δ and chunk scale floors) live in models/run_config.py
LOGS = Path("logs")


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
    # default None = "not given" (needed to reject them with --wm-balance); filled with the
    # LOSS_WEIGHTS values right after parsing, so config.json stores the same numbers as before
    p.add_argument("--w-next-state", type=float, default=None,
                   help="loss weight of the next-state head (0 = no gradient from it; "
                        f"default {LOSS_WEIGHTS['next_state_output']})")
    p.add_argument("--w-next-image", type=float, default=None,
                   help="loss weight of the next-image head (0 = no gradient from it; "
                        f"default {LOSS_WEIGHTS['next_image_output']})")
    # NEW (c): balanced WM losses (replaces the fixed --w-next-* weights)
    p.add_argument("--wm-balance", type=float, default=None, metavar="RHO",
                   help="(c): WM losses ÷ their train copy references, and each WM term's weight "
                        "adapted so its push on the encoder stays at RHO x the actor's (off by default)")
    p.add_argument("--wm-balance-terms", type=lambda v: v.split(","), default=["image", "state"],
                   help="(c): the balanced WM terms, comma-separated (the others keep their "
                        "history-run term: state MSE x 1.0 on all dims, image MSE x 0.1)")
    p.add_argument("--wm-state-groups", type=lambda v: v.split(","), default=None,
                   help="(c): state groups (data.state_groups) in the balanced state loss, "
                        "comma-separated; default: all (STAR-IL: kuka_joint_pos)")
    p.add_argument("--wm-balance-every", type=int, default=10,
                   help="(c): measure the pushes and update the weights every N training steps")
    p.add_argument("--state-delta", action="store_true",
                   help="E7: state head predicts the change f, next_state = state + f (skip); "
                        "loss = Huber on Δ / scale (scale saved in norm_stats.json)")
    p.add_argument("--state-skip", action="store_true",
                   help="1a: next_state = state + Dense(...) zero-initialised (starts as "
                        "copy_state); loss stays MSE on the normalised next state")
    p.add_argument("--state-layers", type=int, default=1,
                   help="1b: hidden Dense(128, relu) layers in the WM state head (1 = original)")
    p.add_argument("--wm-mixer-blocks", type=int, default=0,
                   help="1c: residual ConvMixer blocks on the encoder map, WM branch only "
                        "(both heads; 0 = original; simple_cnn only)")
    p.add_argument("--chunk", type=int, default=None,
                   help="E8: actor predicts this many offsets a(t+i) - a(t-1), option-C scale "
                        "per step and group (saved in norm_stats.json); default: a(t) only")
    p.add_argument("--history", action="store_true",
                   help="history inputs: h(t) = [Δs(t), Δa(t-1)] / train std, to the actor and "
                        "the WM (scale saved in norm_stats.json); 0 where the earlier row is missing")
    p.add_argument("--keep-ranges", default=None,
                   help="CSV episode, first_visible, last_visible: use only those frames for "
                        "pairs and statistics (copied into the run folder, with its hash)")
    p.add_argument("--eval-keep-ranges", default=None,
                   help="with --eval-only: evaluate on the pairs of THIS keep-ranges file (the "
                        "run's own statistics), saved as val_metrics_keep_<file stem>.json")
    p.add_argument("--vis-every",type=int, default=5, help="timeline image every N epochs")
    p.add_argument("--val-every", type=int, default=None,
                   help="validate (full val split) every N training batches instead of once per "
                        "epoch; --epochs, --patience, --vis-every stay in real epochs")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--deterministic", action="store_true", help="deterministic TF ops (slower)")
    p.add_argument("--eval-only", action="store_true",
                   help="re-evaluate the saved best weights of outputs/<run-name> on val")
    args = p.parse_args()
    if args.unfreeze_last_n and not args.freeze_encoder:
        p.error("--unfreeze-last-n needs --freeze-encoder")
    if args.eval_only and not args.run_name:
        p.error("--eval-only needs --run-name")
    # NEW (c): --wm-balance sets the WM weights itself; its options only make sense with it
    if args.wm_balance is not None:
        if args.w_next_state is not None or args.w_next_image is not None:
            p.error("--wm-balance adapts the WM weights itself: drop --w-next-state / --w-next-image")
        if args.wm_balance <= 0:
            p.error("--wm-balance must be > 0 (to switch the WM off: --w-next-state 0 --w-next-image 0)")
        if args.state_delta:
            p.error("--wm-balance uses the MSE state loss: not with --state-delta (E7)")
        if not args.wm_balance_terms or set(args.wm_balance_terms) - {"image", "state"}:
            p.error(f"--wm-balance-terms: a subset of image,state, got {args.wm_balance_terms}")
        if args.wm_balance_every < 1:
            p.error("--wm-balance-every must be >= 1")
        groups = data.state_groups(data.feature_names(Path(args.dataset_root))[0])
        if args.wm_state_groups is None:
            args.wm_state_groups = list(groups)
        unknown = set(args.wm_state_groups) - set(groups)
        if unknown:
            p.error(f"--wm-state-groups: {sorted(unknown)} not in this dataset's groups {list(groups)}")
    elif args.wm_state_groups is not None:
        p.error("--wm-state-groups needs --wm-balance")
    if args.w_next_state is None:
        args.w_next_state = LOSS_WEIGHTS["next_state_output"]
    if args.w_next_image is None:
        args.w_next_image = LOSS_WEIGHTS["next_image_output"]
    if args.w_next_state < 0 or args.w_next_image < 0:
        p.error("loss weights must be >= 0")
    if args.val_every is not None and args.val_every <= 0:
        p.error("--val-every must be a positive number of batches")
    if args.state_layers < 1:
        p.error("--state-layers must be >= 1")
    if args.wm_mixer_blocks < 0:
        p.error("--wm-mixer-blocks must be >= 0")
    if args.state_skip and args.state_delta:
        p.error("--state-skip and --state-delta are two different state heads: choose one")
    # keep-ranges files are checked here, before the run folder exists
    if args.eval_keep_ranges and not args.eval_only:
        p.error("--eval-keep-ranges needs --eval-only (use --keep-ranges for training)")
    for flag, path in (("--keep-ranges", args.keep_ranges), ("--eval-keep-ranges", args.eval_keep_ranges)):
        if path and not Path(path).is_file():
            p.error(f"{flag}: {path} not found")
    if args.keep_ranges and not args.eval_only:   # exactly train + val of this dataset
        try:
            data.check_keep_ranges(data.load_keep_ranges(args.keep_ranges), Path(args.dataset_root))
        except ValueError as e:
            p.error(f"--keep-ranges: {e}")
    return args


# --- B: helpers (the --val-every schedule: run_config.check_schedule) ---
def log_tensorboard(val_metrics, log_dir, step):
    """Final metrics of the best weights, same names as results.csv ("extra/..." its own group)."""
    with tf.summary.create_file_writer(str(Path(log_dir) / "best_val")).as_default():
        for k, v in val_metrics.items():
            tf.summary.scalar(k, v, step=step)


# --- C: training ---
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

    # data: training-only statistics (keep ranges, quaternion reference, norm stats, Δ / chunk /
    # history scales), saved with the run; never the test split (run_config.prepare_stats)
    root = Path(args.dataset_root)
    stats, quat_ref, keep_ranges, keep_hash = prepare_stats(
        root, out, args.canonical_quaternion, args.keep_ranges, args.chunk, args.history,
        args.state_delta,   # the keep-ranges file was checked in parse_args
        wm_balance_groups=args.wm_state_groups if args.wm_balance else None)   # NEW (c)
    train_split = data.load_split("train", root=root, stats=stats, quat_ref=quat_ref,
                                  episodes=episodes_for("train", args.max_episodes, root),
                                  chunk=args.chunk, keep_ranges=keep_ranges, history=args.history)
    val = data.load_split("val", root=root, stats=stats, quat_ref=quat_ref,
                          episodes=episodes_for("val", args.max_episodes, root), chunk=args.chunk,
                          keep_ranges=keep_ranges, history=args.history)
    # evaluation schedule: one Keras "epoch" = one check (run_config.check_schedule; shuffle
    # order: see below)
    schedule = check_schedule(len(train_split["pair_index"]), args.batch_size, args.epochs,
                              args.patience, args.vis_every, args.val_every)
    batches_per_epoch, batches_per_check, max_checks, patience_checks, vis_every_checks = (
        schedule[k] for k in ("batches_per_epoch", "batches_per_check", "max_checks",
                              "patience_checks", "vis_every_checks"))
    # shuffle order: every new iterator draws the next reshuffle. Normal epochs: Keras 3.15
    # creates two iterators between epochs and uses only the second (shuffles #1, #3, #5...,
    # "keras_epochs"); .repeat() uses every one (#1, #2, #3..., "repeat"). So epoch 1 is
    # identical to old runs (checked bitwise: scripts/analysis/check_val_every_weights.py), later
    # epochs see a different but equally random order.
    train_ds = make_dataset(train_split, args.batch_size, shuffle=True, seed=args.seed,
                            repeat=bool(args.val_every))
    val_ds = make_dataset(val, args.batch_size, shuffle=False)

    # model + full configuration, saved before training starts
    cfg = {k: v for k, v in vars(args).items() if k != "eval_only"}
    cfg["run_name"] = run_name
    model = build_model(cfg, stats)
    factor = delta_factor(cfg, stats)
    # reference lines for the state branch, flat in every epoch's logs and saved in config.json:
    # copy_state (every run; 1a: the untrained --state-skip model) and, for E7, f = 0
    reference_logs = {"val_next_state_output_loss_copy": copy_state_mse(val)}
    # the image branch's reference: copy_frame (one pass over the val frames, in chunks)
    reference_logs["val_next_image_output_loss_copyframe"] = copy_frame_mse(val)
    print(f"copy_frame val image MSE: {reference_logs['val_next_image_output_loss_copyframe']:.6g}")
    # copy_state per state dimension, next to the per-dimension val metrics (same names)
    state_names = data.feature_names(root)[0]
    copy_dim = copy_state_mse_per_dim(val)
    for name, value in zip(state_names, copy_dim):
        reference_logs[f"val_next_state_output_mse_{metric_name(name)}_copy"] = float(value)
    # per state group (data.state_groups): copy_state and constant velocity, next to the
    # group curves of StateGroupLogs (the mean of the group's per-dim values)
    groups = data.state_groups(state_names)
    constvel_dim, n_fallback = constvel_state_mse_per_dim(val, keep_ranges)
    for g, dims in groups.items():
        reference_logs[f"val_next_state_{metric_name(g)}_mse_copy"] = float(copy_dim[dims].mean())
        reference_logs[f"val_next_state_{metric_name(g)}_mse_constvel"] = float(constvel_dim[dims].mean())
    glitch_thr = glitch_thresholds(train_split)   # for the glitch report of the final evaluation
    if args.chunk:   # E8: the actor's reference lines, in the loss units:
        # stand-still (offsets 0), the input-free mean train offset and constant velocity
        reference_logs["val_action_output_loss_standstill"] = standstill_huber(val)
        reference_logs["val_action_output_loss_trainmean"] = trainmean_huber(train_split, val)
        reference_logs["val_action_output_loss_constvel"] = constvel_huber(val, stats)
    if factor is not None:
        reference_logs["val_next_state_output_loss_zero"] = metrics.state_delta_metrics(
            val["states"][val["pair_index"]], val["states"][val["pair_index"]],
            val["states"][val["pair_index"] + 1], factor)["next_state_delta_huber_zero"]
    cfg.update(**reference_logs)
    if args.state_skip:   # 1a: before any training, the state head must be exactly copy_state
        inputs, _ = next(iter(val_ds))
        pred_state = model(inputs, training=False)[1].numpy()
        gap = float(np.max(np.abs(pred_state - inputs["state_input"].numpy())))
        print(f"state skip check: max |next_state - state_input| = {gap} (must be 0)")
        if gap != 0.0:
            raise AssertionError(f"untrained --state-skip model is not copy_state (gap {gap})")
    cfg.update(
        freeze=freeze_label(cfg), monitor=MONITOR, loss_weights=loss_weights(cfg),
        wm_mixer_kernel=WM_MIXER_KERNEL,   # 1c: fixed in swm.py, recorded with every run
        params_total=int(model.count_params()),
        params_trainable=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        n_train_pairs=int(len(train_split["pair_index"])), n_val_pairs=int(len(val["pair_index"])),
        n_train_pairs_untrimmed=int(train_split["n_pairs_untrimmed"]),
        n_val_pairs_untrimmed=int(val["n_pairs_untrimmed"]),
        batches_per_epoch=batches_per_epoch, batches_per_check=batches_per_check,
        max_checks=max_checks, patience_checks=patience_checks,
        vis_every_checks=vis_every_checks,
        shuffle_order="repeat" if args.val_every else "keras_epochs",
        n_val_constvel_fallback=n_fallback,   # pairs without s(t-1) in the kept window: copy
        # history: pairs whose Δs / Δa is 0 because the earlier row is missing (None: off)
        n_train_history_fallback=train_split.get("n_history_fallback"),
        n_val_history_fallback=val.get("n_history_fallback"),
        glitch_threshold=glitch_thr.tolist(),  # per state dim, normalised (references.GLITCH_G x train p99 |Δ|)
        keep_ranges_sha256=keep_hash,
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
    if args.wm_balance:   # NEW (c)
        wb = stats["wm_balance"]
        print(f"--wm-balance {args.wm_balance}: balanced {list(model.bal_terms)} (others: "
              f"history-run weights), state groups {args.wm_state_groups}, measured every "
              f"{args.wm_balance_every} steps; references image {wb['image_copy_mse']:.4g}, "
              f"state {wb['state_copy_mse']:.4g}")

    run = wandb.init(project=WANDB_PROJECT, name=run_name, config=cfg, mode=args.wandb_mode)
    callbacks = [
        EpochTimer(),
        TrainingProgress(batches_per_check, batches_per_epoch),
        CheckMonitor(MONITOR),
        ConstantLogs(reference_logs),
        StateGroupLogs({metric_name(g): [metric_name(state_names[i]) for i in dims]
                        for g, dims in groups.items()}),
        keras.callbacks.ModelCheckpoint(str(out / "best.weights.h5"), monitor=MONITOR, mode="min",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=MONITOR, mode="min", patience=patience_checks,
                                      verbose=1),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
        timeline_callback(val, run, log_dir, vis_every_checks, root),
        WandbLogger(run),
    ]
    if args.wm_balance:   # NEW (c): per-measurement trace; stop on NaN (the only automatic stop)
        callbacks += [WMBalanceTrace(out / "wm_balance_trace.csv", model.bal_terms),
                      keras.callbacks.TerminateOnNaN()]
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=max_checks,
                        steps_per_epoch=args.val_every,   # None = one full pass per "epoch"
                        callbacks=callbacks, verbose=2)
    train_time_min = (time.time() - t0) / 60
    if args.wm_balance:   # NEW (c): the controller at the END (at the best check: history.csv)
        cfg["wm_balance_final"] = model.controller_state()

    # --- D: evaluate the BEST weights with the same metrics as the baselines ---
    model.load_weights(str(out / "best.weights.h5"))
    val_metrics = evaluate(model, val, stats, args.batch_size, root, factor, keep_ranges, glitch_thr)
    chunk_base = chunk_baseline_metrics(val, stats, root) if args.chunk else None
    if chunk_base is not None:
        save_json(chunk_base, out / "chunk_baselines.json")
    monitored = history.history[MONITOR]
    # best_epoch = index of the best check (= the real epoch for normal runs, as before)
    cfg.update(epochs_run=len(monitored), best_epoch=int(np.argmin(monitored)) + 1,
               train_time_min=round(train_time_min, 2))
    cfg["best_batch"] = cfg["best_epoch"] * batches_per_check
    cfg["best_epoch_equiv"] = cfg["best_batch"] / batches_per_epoch
    save_json(cfg, out / "config.json")
    save_json(val_metrics, out / "val_metrics.json")
    log_tensorboard(val_metrics, log_dir, cfg["best_epoch"])
    run.summary.update({f"best/{k}": v for k, v in val_metrics.items()})
    run.summary.update({"best_epoch": cfg["best_epoch"], "best_batch": cfg["best_batch"],
                        "best_epoch_equiv": cfg["best_epoch_equiv"],
                        "train_time_min": cfg["train_time_min"]})

    unit = "check" if args.val_every else "epoch"
    print(f"\nbest {unit} {cfg['best_epoch']} of {cfg['epochs_run']} (batch {cfg['best_batch']}, "
          f"{cfg['best_epoch_equiv']:.2f} epochs), training {train_time_min:.1f} min\n")
    print(comparison_table(run_name, val_metrics, root, args.canonical_quaternion, chunk_base,
                           keep_hash))
    if args.max_episodes is None:
        metrics.append_results({"run_name": run_name, "dataset": root.name,
                                "encoder": args.encoder,
                                "canonical_quaternion": args.canonical_quaternion,
                                "freeze": cfg["freeze"], "seed": args.seed,
                                "latent_dim": args.latent_dim,
                                "w_next_state": args.w_next_state,
                                "w_next_image": args.w_next_image,
                                "state_delta": args.state_delta, "state_skip": args.state_skip,
                                "state_layers": args.state_layers,
                                "wm_mixer_blocks": args.wm_mixer_blocks,
                                "wm_balance": args.wm_balance,   # NEW (c)
                                "wm_balance_terms": (",".join(args.wm_balance_terms)
                                                     if args.wm_balance else None),
                                "wm_state_groups": (",".join(args.wm_state_groups)
                                                    if args.wm_balance else None),
                                "chunk": args.chunk, "history": args.history,
                                "keep_ranges": Path(args.keep_ranges).name if args.keep_ranges else None,
                                "epochs": cfg["epochs_run"],
                                "best_epoch": cfg["best_epoch"],
                                "val_every": args.val_every, "best_batch": cfg["best_batch"],
                                "train_time_min": cfg["train_time_min"], **val_metrics})
        print(f"\nadded a row to {metrics.RESULTS_CSV}")
    else:
        print("\nsmoke run (--max-episodes): not added to results.csv")
    run.finish()


# --- E: re-evaluate a finished run with ITS saved statistics ---
def eval_only(args):
    out = OUTPUTS / args.run_name
    cfg = json.loads((out / "config.json").read_text())
    stats = data.load_norm_stats(out / "norm_stats.json")
    root = dataset_root(cfg)
    quat_ref = run_quat_ref(cfg, out)
    keep_ranges = run_keep_ranges(cfg, out)        # None: untrimmed (and older runs)
    keep_hash = cfg.get("keep_ranges_sha256")
    # glitch thresholds from the run's own train split (tables only), as in train(); always the
    # run's keep ranges, also with --eval-keep-ranges (older runs did not save them)
    train_tables = data.load_split("train", root=root, stats=stats, quat_ref=quat_ref,
                                   episodes=episodes_for("train", cfg["max_episodes"], root),
                                   chunk=cfg.get("chunk"), keep_ranges=keep_ranges,
                                   with_frames=False)
    glitch_thr = glitch_thresholds(train_tables)
    if args.eval_keep_ranges:   # other pairs, same weights and statistics (nothing recomputed)
        keep_ranges = data.load_keep_ranges(args.eval_keep_ranges)
        keep_hash = data.file_sha256(args.eval_keep_ranges)
        data.check_keep_ranges(keep_ranges, root)
    val = data.load_split("val", root=root, stats=stats, quat_ref=quat_ref,
                          episodes=episodes_for("val", cfg["max_episodes"], root),
                          chunk=cfg.get("chunk"),   # runs before E8: no "chunk" key
                          keep_ranges=keep_ranges, history=run_history(cfg))
    model = build_model(cfg, stats)
    model.load_weights(str(out / "best.weights.h5"))
    val_metrics = evaluate(model, val, stats, cfg["batch_size"], root, delta_factor(cfg, stats),
                           keep_ranges, glitch_thr)
    chunk_base = chunk_baseline_metrics(val, stats, root) if cfg.get("chunk") else None
    model_mse, copy_mse = val_metrics["next_state_mse"], val_metrics["next_state_mse_copy"]
    copy_line = (f"next_state_mse: model {model_mse:.3g} vs copy_state {copy_mse:.3g} "
                 f"(model / copy = {model_mse / copy_mse:.2f})")
    for g in data.state_groups(data.feature_names(root)[0]):
        k = f"state_group/{metric_name(g)}"
        copy_line += "\n" + " ".join(
            f"{g} {who}: model / copy = {val_metrics[f'{k}/model_mse{sfx}'] / val_metrics[f'{k}/copy_mse{sfx}']:.3f}, "
            f"constvel / copy = {val_metrics[f'{k}/constvel_mse{sfx}'] / val_metrics[f'{k}/copy_mse{sfx}']:.3f};"
            for who, sfx in (("all pairs", ""), ("no glitch", "_noglitch")))
    if args.eval_keep_ranges:   # its own JSON: val_metrics*.json of the run stay untouched
        path = out / f"val_metrics_keep_{Path(args.eval_keep_ranges).stem}.json"
        save_json({"keep_ranges": args.eval_keep_ranges, "keep_ranges_sha256": keep_hash,
                   "n_val_pairs": int(len(val["pair_index"])),
                   "n_val_pairs_untrimmed": int(val["n_pairs_untrimmed"]),
                   "metrics": val_metrics, "chunk_baselines": chunk_base}, path)
        print(comparison_table(f"{args.run_name} (keep ranges)", val_metrics, root,
                               quat_ref is not None, chunk_base, keep_hash))
        print(f"\n{copy_line}")
        print(f"saved {path}")
        return
    save_json(val_metrics, out / "val_metrics_reeval.json")
    print(comparison_table(f"{args.run_name} (re-eval)", val_metrics, root, quat_ref is not None,
                           chunk_base, keep_hash))
    print(f"\n{copy_line}")

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
