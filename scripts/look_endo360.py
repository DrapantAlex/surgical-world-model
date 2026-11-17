"""
look_endo360.py  (throwaway inspection script, tables only)

Three checks on the STAR-IL endo360 joint (needle driver, state dim 7):
  1. what are the large one-frame jumps (real motion, wrap-around, glitch)?
  2. is the staircase a slower sampling rate (sample-and-hold)?
  3. is "endo360 = 94% of the copy_state error" real in radians, or a normalisation effect?

Train + val episodes only (after drop_missing_videos, as in the runs); never test.
Raw units (rad) unless stated; differences only within an episode.
Pair sets per episode (t = first frame of the pair (t, t+1)):
  all  : t = 0 .. T-2
  kept : t, t+1 inside the keep-ranges window [lo, hi]
  run  : data.pair_positions(lo, hi, chunk) = the exact pairs of the 1b runs (check 3, 1a)
Only check 1d decodes video (7 frames around each of the 3 largest jumps).

Usage:
  python -m scripts.look_endo360
"""
import json
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import data

RUN = Path("outputs/star-il-simplecnn-latent512-canonq-chunk16-trim-skip-sl2-val1000-seed0")
OUT = Path("outputs/analysis/endo360_checks")
EPS = 1e-9          # "changes" = |Δ| > EPS
E = 7               # endo360 index in observation.state


# --- A: load every train + val table once (raw radians) ---
def load_all(cfg):
    root = Path(cfg["dataset_root"])
    keep = data.load_keep_ranges(cfg["keep_ranges"])
    names_s, names_a = data.feature_names(root)
    eps = []
    for split in data.ALLOWED_SPLITS:
        kept_eps, _ = data.drop_missing_videos(split, data.get_split_episodes(split, root),
                                               data.CAMERA, root, verbose=False)
        for ep in kept_eps:
            s, a = data.load_table(ep, root)                 # raw, no quaternion canonicalising
            T = len(s)
            lo, hi = data.kept_window(ep, T, keep)
            eps.append(dict(ep=ep, split=split, s=s.astype(np.float64), a=a.astype(np.float64),
                            T=T, lo=lo, hi=hi,
                            all=np.arange(0, T - 1),
                            kept=np.arange(lo, hi) if hi > lo else np.arange(0),
                            run=data.pair_positions(lo, hi, cfg["chunk"])))
    return root, names_s, names_a, eps


def deltas(eps, pairs, split=None, dim=E):
    """Concatenated Δ = x(t+1) - x(t) of one state dim over the chosen pair set."""
    return np.concatenate([e["s"][e[pairs] + 1, dim] - e["s"][e[pairs], dim]
                           for e in eps if split in (None, e["split"])])


def top1_share(d):
    sq = np.sort(d ** 2)[::-1]
    return sq[:max(1, int(np.ceil(0.01 * len(sq))))].sum() / sq.sum()


# --- B: check 1, the large jumps ---
def check1(eps, root):
    rows = []
    for split in (None, "val"):
        for pairs in ("all", "kept", "run"):
            d = np.abs(deltas(eps, pairs, split))
            r = dict(split=split or "train+val", pairs=pairs, n=len(d))
            r.update({f"p{q}": np.percentile(d, q) for q in (50, 90, 99, 99.9)})
            r["max"] = d.max()
            r.update({f"n_gt_{t}": int((d > t).sum()) for t in (0.01, 0.1, 0.5, 1, 3)})
            r["top1pct_share_sq"] = top1_share(d)
            r["share_still"] = np.mean(d <= EPS)
            rows.append(r)
    t1a = pd.DataFrame(rows)
    t1a.to_csv(OUT / "check1a_delta_distribution.csv", index=False)

    # b) the 20 largest |Δ| over all frames (train + val)
    jumps = []
    for e in eps:
        x = e["s"][:, E]
        d = np.diff(x)
        for t in np.argsort(-np.abs(d))[:20]:
            nxt = [d[t + k] if t + k < len(d) else np.nan for k in (1, 2, 3)]
            back = any(np.sign(n) == -np.sign(d[t]) and abs(n) >= 0.5 * abs(d[t])
                       for n in nxt if not np.isnan(n))
            jumps.append(dict(episode=e["ep"], split=e["split"], frame=int(t),
                              before=x[t], after=x[t + 1], delta=d[t], delta_over_pi=d[t] / np.pi,
                              in_kept=bool(e["lo"] <= t and t + 1 <= e["hi"]),
                              next1=nxt[0], next2=nxt[1], next3=nxt[2], comes_back=back))
    t1b = pd.DataFrame(jumps).reindex(
        pd.DataFrame(jumps)["delta"].abs().sort_values(ascending=False).index).head(20)
    t1b.to_csv(OUT / "check1b_top20_jumps.csv", index=False)

    # c) per episode: range vs net change vs path length
    per = []
    for e in eps:
        for pairs in ("all", "kept"):
            p = e[pairs]
            if len(p) == 0:
                continue
            x = e["s"][p[0]:p[-1] + 2, E]
            per.append(dict(episode=e["ep"], split=e["split"], pairs=pairs, frames=len(x),
                            min=x.min(), max=x.max(), range=x.max() - x.min(),
                            net=x[-1] - x[0], sum_abs_delta=np.abs(np.diff(x)).sum()))
    t1c = pd.DataFrame(per)
    t1c["path_over_range"] = t1c["sum_abs_delta"] / t1c["range"].where(t1c["range"] > 0)
    t1c.to_csv(OUT / "check1c_per_episode.csv", index=False)

    # d) frame strips t-3 .. t+3 around the 3 largest jumps
    for i, j in enumerate(t1b.head(3).itertuples()):
        e = next(x for x in eps if x["ep"] == j.episode)
        want = set(range(max(0, j.frame - 3), min(e["T"], j.frame + 5)))   # t-3 .. t+4 (= (t+1)+3)
        frames = {}
        with av.open(str(data.video_path(j.episode, data.CAMERA, root))) as c:
            for k, f in enumerate(c.decode(video=0)):
                if k in want:
                    frames[k] = f.to_ndarray(format="rgb24")
                if k >= max(want):
                    break
        ks = sorted(frames)
        fig, ax = plt.subplots(1, len(ks), figsize=(3 * len(ks), 2.4))
        for a_, k in zip(ax, ks):
            a_.imshow(frames[k]); a_.axis("off")
            a_.set_title(f"f{k}  endo={e['s'][k, E]:+.3f}", fontsize=8,
                         color="red" if k in (j.frame, j.frame + 1) else "black")
        fig.suptitle(f"episode {j.episode} ({j.split}): Δ = {j.delta:+.3f} rad between "
                     f"f{j.frame} and f{j.frame + 1} (red)", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT / f"check1d_jump{i + 1}_ep{j.episode}_f{j.frame}.png", dpi=90)
        plt.close(fig)
    return t1a, t1b, t1c


# --- C: check 2, sample-and-hold? ---
def check2(eps, names_s, names_a):
    hist = []
    for pairs in ("all", "kept"):
        gaps = []
        for e in eps:
            p = e[pairs]
            ch = p[np.abs(e["s"][p + 1, E] - e["s"][p, E]) > EPS]
            gaps.append(np.diff(ch))
        g = np.concatenate(gaps)
        row = dict(pairs=pairs, n_gaps=len(g))
        row.update({f"gap_{k}": int((g == k).sum()) for k in range(1, 16)})
        row["gap_gt15"] = int((g > 15).sum())
        moving = g[g <= 15]
        row["moving_mode_gap"] = int(np.bincount(moving).argmax())
        row["moving_share_mode"] = np.mean(moving == row["moving_mode_gap"])
        hist.append(row)
    t2a = pd.DataFrame(hist)
    t2a.to_csv(OUT / "check2a_gap_histogram.csv", index=False)

    share = []
    for pairs in ("all", "kept"):
        r = {"pairs": pairs}
        for arr, names in (("s", names_s), ("a", names_a)):
            for i, n in enumerate(names):
                ch = np.concatenate([np.abs(e[arr][e[pairs] + 1, i] - e[arr][e[pairs], i]) > EPS
                                     for e in eps])
                r[f"{arr}:{n}"] = ch.mean()
        share.append(r)
    t2b = pd.DataFrame(share).set_index("pairs").T
    t2b.to_csv(OUT / "check2b_change_share.csv")

    q = []
    for pairs in ("all", "kept"):
        d = np.abs(deltas(eps, pairs))
        d = d[d > EPS]
        vals, counts = np.unique(np.round(d, 7), return_counts=True)
        top = np.argsort(-counts)[:10]
        base = d.min()
        r = d / base
        q.append(dict(pairs=pairs, n_nonzero=len(d), smallest=base,
                      top10=[(float(vals[i]), int(counts[i])) for i in top],
                      share_within_0p1_of_integer_multiple=np.mean(np.abs(r - np.round(r)) < 0.1),
                      share_within_0p05=np.mean(np.abs(r - np.round(r)) < 0.05)))
    t2c = pd.DataFrame(q)
    t2c.to_csv(OUT / "check2c_quantisation.csv", index=False)
    return t2a, t2b, t2c


# --- D: check 3, error shares raw vs normalised (exact val pairs of the runs) ---
def check3(eps, names_s, cfg):
    stats = data.load_norm_stats(RUN / "norm_stats.json")
    std = np.asarray(stats["observation.state"]["std"], np.float64) + 1e-8
    val = [e for e in eps if e["split"] == "val"]
    d = np.concatenate([e["s"][e["run"] + 1] - e["s"][e["run"]] for e in val])   # (P, 8)
    assert len(d) == cfg["n_val_pairs"], (len(d), cfg["n_val_pairs"])

    def table(dd, tag):
        raw = (dd ** 2).mean(0)
        norm = ((dd / std) ** 2).mean(0)
        return pd.DataFrame({"joint": names_s, "subset": tag, "n_pairs": len(dd),
                             "mse_rad2": raw, "share_raw": raw / raw.sum(),
                             "mse_norm": norm, "share_norm": norm / norm.sum(),
                             "rmse_deg": np.degrees(np.sqrt(raw)),
                             "train_std_rad": std, "train_std_deg": np.degrees(std)})
    keep = np.abs(d[:, E]) < np.quantile(np.abs(d[:, E]), 0.99)
    t3 = pd.concat([table(d, "val_run_pairs"), table(d[keep], "without_top1pct_endo360")])
    ref = [cfg[f"val_next_state_output_mse_{data_name}_copy"] for data_name in
           [n.replace("[", "_").replace("]", "") for n in names_s]]
    t3.loc[t3.subset == "val_run_pairs", "config_copy_norm"] = ref
    t3.to_csv(OUT / "check3_error_shares.csv", index=False)
    return t3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RUN / "config.json").read_text())
    root, names_s, names_a, eps = load_all(cfg)
    print(f"{len(eps)} episodes (train {sum(e['split'] == 'train' for e in eps)}, "
          f"val {sum(e['split'] == 'val' for e in eps)}); endo360 overall range "
          f"[{min(e['s'][:, E].min() for e in eps):.3f}, {max(e['s'][:, E].max() for e in eps):.3f}] rad")
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 4)
    t1a, t1b, t1c = check1(eps, root)
    print("\n=== 1a |Δ| endo360 (rad) ===\n", t1a.to_string(index=False))
    print("\n=== 1b 20 largest jumps ===\n", t1b.to_string(index=False))
    print("\n=== 1c per episode (median over episodes) ===\n",
          t1c.groupby("pairs")[["range", "net", "sum_abs_delta", "path_over_range"]].median())
    t2a, t2b, t2c = check2(eps, names_s, names_a)
    print("\n=== 2a gaps between endo360 change frames ===\n", t2a.T.to_string())
    print("\n=== 2b share of pairs where each dim changes ===\n", t2b.to_string())
    print("\n=== 2c quantisation ===\n", t2c.T.to_string())
    t3 = check3(eps, names_s, cfg)
    print("\n=== 3 copy_state error shares ===\n", t3.to_string(index=False))
    print(f"\nCSVs and frame strips in {OUT}/")


if __name__ == "__main__":
    main()
