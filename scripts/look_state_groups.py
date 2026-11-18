"""
look_state_groups.py  (throwaway, no model, no data loading)

Per-group WM state curves of finished runs, computed after the fact from history.csv: the
group's val MSE = the mean of its per-dimension columns (val_next_state_output_mse_<dim>),
divided by the group's copy_state line (the mean of the per-dim _copy values in config.json).
The constant-velocity line comes from the run's val_metrics_reeval.json (--eval-only).

  python -m scripts.look_state_groups <run> [<run> ...]    # first run solid, second dashed, ...
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import data
from scripts.train import OUTPUTS, metric_name

OUT = Path("outputs/analysis/state_groups")
PANELS = {"kuka_joint_pos": ("KUKA arm (7 joints)", "#2a78d6"),
          "endo360_joint_pos": ("endo360 needle driver", "#eb6834")}
STYLES = ["-", "--", ":"]


def group_curves(run):
    """{group: (epochs_seen, val ÷ copy per check, constvel ÷ copy or None)} and the run's K."""
    out = OUTPUTS / run
    cfg = json.loads((out / "config.json").read_text())
    h = pd.read_csv(out / "history.csv")
    reeval = out / "val_metrics_reeval.json"
    m = json.loads(reeval.read_text()) if reeval.exists() else {}
    res = {}
    for g, dims in data.state_groups(data.feature_names(Path(cfg["dataset_root"]))[0]).items():
        cols = [metric_name(data.feature_names(Path(cfg["dataset_root"]))[0][i]) for i in dims]
        model = h[[f"val_next_state_output_mse_{c}" for c in cols]].mean(axis=1)
        copy = np.mean([cfg[f"val_next_state_output_mse_{c}_copy"] for c in cols])
        k = f"state_group/{metric_name(g)}"
        cv = m[f"{k}/constvel_mse"] / m[f"{k}/copy_mse"] if f"{k}/copy_mse" in m else None
        res[metric_name(g)] = (h["epochs_seen"].to_numpy(), (model / copy).to_numpy(), cv)
    return res, cfg.get("state_layers", 1)


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    for j, run in enumerate(runs):
        curves, K = group_curves(run)
        for ax, (g, (title, color)) in zip(axes, PANELS.items()):
            x, r, cv = curves[g]
            ax.plot(x, r, STYLES[j], color=color, lw=2, label=f"K = {K}")
            if g == "kuka_joint_pos" and cv is not None and j == 0:
                ax.axhline(cv, color="#6b6b6b", lw=1.5, ls="-.", label=f"constant velocity ({cv:.3f})")
            rows.append(dict(run=run, K=K, group=g, min_ratio=r.min(),
                             at_epochs=x[r.argmin()], n_checks=len(r),
                             checks_below_copy=int((r < 1).sum()),
                             checks_below_constvel=int((r < cv).sum()) if cv else None,
                             last=r[-1]))
    for ax, (g, (title, _)) in zip(axes, PANELS.items()):
        ax.axhline(1.0, color="#1f1f1f", lw=1, label="copy_state (= 1)")
        ax.set_title(title, loc="left")
        ax.set_xlabel("epochs seen")
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    axes[0].set_ylabel("val state MSE ÷ copy_state")
    for ax in axes:   # one legend per panel at the top (the line sets differ)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=4, frameon=False, fontsize=9)
    plt.tight_layout()
    path = OUT / "state_groups_vs_copy.png"
    plt.savefig(path, dpi=130)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "state_groups_summary.csv", index=False)
    print(table.to_string(index=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
