"""Step 1 summary figures (1a, 1b, 1c) from each run's history.csv. No TF, no data loading.

    python -m scripts.plot_step1_summary   ->  outputs/analysis/step1_summary/*.png

1. KUKA state / copy_state (log y), with copy (1) and constant velocity
2. image / copy_frame (log y), with copy_frame (1)
3. actor val Huber, with stand-still and train-mean
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- A: runs, colours and reference values ---
PREFIX = "outputs/star-il-simplecnn-latent512-canonq-chunk16-trim-skip-"
# hue = step (1a grey, 1b orange, 1c blue); lighter + dashed = smaller K / D.
# The same run has the same colour and style in every figure.
RUNS = {
    "1a  K=1, D=0":  dict(path="val500-seed0",       color="#5f5e5a", ls="-"),
    "1b  K=2":       dict(path="sl2-val1000-seed0",  color="#f0a07a", ls="--"),
    "1b  K=3":       dict(path="sl3-val1000-seed0",  color="#b4461a", ls="-"),
    "1c  D=1":       dict(path="mix1-val1000-seed0", color="#6da7ec", ls="--"),
    "1c  D=3":       dict(path="mix3-val1000-seed0", color="#184f95", ls="-"),
}
COPY_FRAME = 0.000595          # val_next_image_output_loss_copyframe as logged by the 1c runs
CONSTVEL_RATIO = 0.107         # KUKA constvel / copy on val (1.1555e-6 / 1.0791e-5, 1c runs)
REF_COLOR = "#8a8984"          # reference lines: neutral, never a run colour
OUT = Path("outputs/analysis/step1_summary")
N_KUKA = 7


def load(path):
    with open(PREFIX + path + "/history.csv") as f:
        return list(csv.DictReader(f))


def col(rows, name):
    """Column as floats, or None if this run never logged it."""
    if not rows or rows[0].get(name) in (None, ""):
        return None
    return [float(r[name]) for r in rows]


# --- B: the three ratios ---
def kuka_ratio(rows):
    """KUKA group MSE / its copy line. 1c logs the group; 1b only the per-joint columns, whose
    mean is the same quantity (group MSE = mean over the 7 joints of the per-joint MSE).
    1a logged neither -> None."""
    model, copy = col(rows, "val_next_state_kuka_joint_pos_mse"), col(rows, "val_next_state_kuka_joint_pos_mse_copy")
    if model is None:
        per = [col(rows, f"val_next_state_output_mse_kuka_joint_pos_{i}") for i in range(N_KUKA)]
        per_c = [col(rows, f"val_next_state_output_mse_kuka_joint_pos_{i}_copy") for i in range(N_KUKA)]
        if any(p is None for p in per + per_c):
            return None
        model = [sum(v) / N_KUKA for v in zip(*per)]
        copy = [sum(v) / N_KUKA for v in zip(*per_c)]
    return [m / c for m, c in zip(model, copy)]


def image_ratio(rows):
    return [v / COPY_FRAME for v in col(rows, "val_next_image_output_loss")]


def actor(rows):
    """Exact per-pair Huber (since 1b); 1a only has the batch-weighted loss (~1e-3 relative)."""
    return col(rows, "val_action_output_huber") or col(rows, "val_action_output_loss")


# --- C: plotting ---
def ref_line(ax, y, text):
    ax.axhline(y, color=REF_COLOR, lw=1, ls=":", zorder=1)
    ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8, color="#55544f")


def figure(name, title, ylabel, series_fn, refs, logy, note=None):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    shown = 0
    for label, r in RUNS.items():
        rows = load(r["path"])
        y = series_fn(rows)
        if y is None:
            continue
        x = col(rows, "batches_seen")
        ax.plot(x, y, color=r["color"], ls=r["ls"], lw=1.6, marker="o", ms=2.5, label=label, zorder=3)
        shown += 1
    for y, text in refs:
        ref_line(ax, y, text)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("batches seen")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=shown, frameon=False, fontsize=8,
              handlelength=2.5)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color="#55544f")
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_1b = load(RUNS["1b  K=2"]["path"])
    standstill = col(rows_1b, "val_action_output_loss_standstill")[0]
    trainmean = col(rows_1b, "val_action_output_loss_trainmean")[0]

    figure("1_kuka_state_vs_copy.png",
           "WM state on the KUKA arm, val  (below 1 = beats copy_state)",
           "KUKA MSE ÷ copy_state", kuka_ratio,
           refs=[(1.0, "copy"), (CONSTVEL_RATIO, "const. velocity")], logy=True,
           note="1a not shown: it predates the per-joint metrics (no KUKA columns in its history.csv).")
    figure("2_image_vs_copyframe.png",
           "WM next image, val  (below 1 = beats copy_frame)",
           "image MSE ÷ copy_frame", image_ratio,
           refs=[(1.0, "copy_frame")], logy=True)
    figure("3_actor_val_huber.png",
           "Actor, val  (lower is better)",
           "action Huber", actor,
           # the two lines are 0.001 apart (indistinguishable here): one shared label
           refs=[(standstill, f"stand-still {standstill:.4f}\ntrain-mean {trainmean:.4f}"), (trainmean, "")],
           logy=False,
           note="1a: batch-weighted val_action_output_loss (no exact Huber logged); others: val_action_output_huber.")


if __name__ == "__main__":
    main()
