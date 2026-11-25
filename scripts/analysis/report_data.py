"""Data card: every input and target of the actor-world model, from the raw files to the model.

Measurement and documentation only: nothing is trained, models/ and scripts/train.py are only
imported. For one run (default: the history seed-0 run) it writes docs/data-card.md: a summary
table, then one section per item with the same template (role, shape, raw source and units, raw
values, transformations with file:line, check, final values, range, special cases, one worked
example, model side), then the three model outputs with the measured ranges of the predictions.

Data sources
  tables (state, action, chunk targets, history): load_split(with_frames=False) over the FULL
      train and val splits with the run's options (quaternion reference, keep ranges, chunk,
      history), then the real get_pairs; the raw values are the parquet columns of the same rows
      (load_table without quat_ref, i.e. the file values).
  images: only the first --image-episodes episodes of train and val, decoded directly (the
      frame cache is read for one comparison if it exists, never built); raw = the native-
      resolution frames at every --raw-every-th pair row.
  statistics: the run's own norm_stats.json, quat_reference.json and keep_ranges.csv (train only).
  predictions: the run's best.weights.h5 on the val image episodes.
The test split is never loaded (load_split refuses it).

Every transformation is recomputed here from the raw values with its formula, independently of
models/data.py, and compared with what get_pairs gives the model (the "Check" line of each
section). The script writes the card anyway and exits 1 if any check fails. The file:line
references are found in the source at run time (where()), so they follow the code.

  python -m scripts.analysis.report_data
  python -m scripts.analysis.report_data --run <run> --image-episodes 3 --example-t 1234
"""
import argparse
import datetime
import inspect
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

from models import data, encoders
from models.pipeline import make_dataset
from models.references import GLITCH_G, glitch_frames, glitch_thresholds
from models.run_config import (OUTPUTS, REPO, build_model, dataset_root, git_commit, run_history,
                               run_keep_ranges, run_quat_ref)
from models.simple_world_model import swm
from models import run_config

# --- A: fixed choices ---
DEFAULT_RUN = "star-il-simplecnn-latent512-canonq-chunk16-trim-skip-hist-val1000-seed0"
SPLITS = data.ALLOWED_SPLITS     # ("train", "val"): the test split is never touched
CHUNK_STEPS = (1, 8, 16)         # chunk steps reported, 1-based: step 1 = a(t) - a(t-1) (code i = 0)
# formula (float64, here) vs get_pairs (float32): a value passes if |diff| <= F32_C x eps32 x
# (mag + |y|), mag = the size of the float32 numbers that were rounded, in final units (e.g. a
# chunk offset subtracts two normalised values of size (|x| + |mean|) / std, then multiplies by
# std / scale: mag = (|x1| + |x2| + 2 |mean|) / scale). A wrong formula is orders of magnitude off.
EPS32 = float(np.finfo(np.float32).eps)
F32_C = 4.0
PRED_BINS = 10_000               # histogram bins on [0, 1] for the predicted image (p1/p99 to 1e-4)
STAT_KEYS = ("min", "max", "mean", "std", "p1", "p99")
# physical meaning and unit per info.json name; source: the dataset README (STAR-IL). Any other
# name is reported as undocumented, never guessed
DIM_INFO = [
    (r"^kuka_joint_pos\[(\d+)\]$", "KUKA LBR iiwa joint {0} angle", "rad"),
    (r"^endo360_joint_pos\[(\d+)\]$", "Endo360 needle-driver joint {0} (the tool, not the camera)", "rad"),
    (r"^kuka_pose\.([xyz])$", "commanded end-effector position {0}", "m"),
    (r"^kuka_pose\.ori\.([xyzw])$", "commanded end-effector orientation, quaternion {0}", "1 (unit quaternion)"),
]
LINK_BASE = "."                  # repo root relative to the card's folder (set in main)
STATIC_M = 1e-4                  # worked example: "moving" = commanded position step > 0.1 mm / frame
TAILS = (5, 10)                 # "Tails" line: share of final values with |y| above these
# the summary table's one-line transformation per item (the sections give the full steps)
SHORT = {"image_input": "decode, bilinear resize to 224 × 224 (uint8), / 255",
         "next_image": "decode, bilinear resize to 224 × 224 (uint8), / 255",
         "state_input": "(s − mean) / std",
         "next_state": "(s(t+1) − mean) / std",
         "action_input": "canonical sign, (a − mean) / std",
         "chunk": "canonical sign, (a(t+i) − a(t−1)) / scale[i, group]",
         "history_ds": "(s(t) − s(t−1)) / std_s / std_Δs",
         "history_da": "canonical sign, (a(t−1) − a(t−2)) / std_a / std_Δa",
         "language": "constant 0"}


def dim_info(name):
    for pattern, meaning, unit in DIM_INFO:
        m = re.match(pattern, name)
        if m:
            return meaning.format(*m.groups()), unit
    return "(not documented in this script)", "?"

def where(obj, needle):
    """Markdown link 'path:line' to the first source line of obj that contains needle. Raises if
    it is not found: the card's references are read from the code, never typed by hand."""
    lines, start = inspect.getsourcelines(obj)
    path = Path(inspect.getsourcefile(obj)).resolve().relative_to(REPO).as_posix()
    for i, line in enumerate(lines):
        if needle in line:
            return f"[{path}:{start + i}]({LINK_BASE}/{path}#L{start + i})"
    raise ValueError(f"{needle!r} not found in {obj.__qualname__}: update report_data.py")


# --- B: statistics and their tables ---
def dim_stats(x):
    """Per column of x (n, d): min, max, mean, std, p1, p99 (numpy linear), % negative, plus the
    share of all values outside [0, 1] and outside [-1, 1]."""
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    return {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0),
            "p1": np.percentile(x, 1, axis=0), "p99": np.percentile(x, 99, axis=0),
            "neg": 100.0 * (x < 0).mean(0), "n": len(x),
            "out01": float(((x < 0) | (x > 1)).mean()), "out11": float((np.abs(x) > 1).mean())}

def hist_stats(counts, values):
    """The same from per-channel histograms counts (C, B) over bin values (B,): exact for uint8
    pixels without holding them. p1/p99 by nearest rank (the first value with CDF >= 1% / 99%)."""
    out = {k: np.empty(len(counts)) for k in (*STAT_KEYS, "neg")}
    for c, h in enumerate(counts):
        n, nz, cdf = h.sum(), np.flatnonzero(h), np.cumsum(h)
        mean = (h * values).sum() / n
        out["min"][c], out["max"][c] = values[nz[0]], values[nz[-1]]
        out["mean"][c], out["std"][c] = mean, np.sqrt((h * (values - mean) ** 2).sum() / n)
        out["p1"][c] = values[np.searchsorted(cdf, 0.01 * n)]
        out["p99"][c] = values[np.searchsorted(cdf, 0.99 * n)]
        out["neg"][c] = 100.0 * h[values < 0].sum() / n
    total = counts.sum()
    out["n"] = int(counts[0].sum())
    out["out01"] = float(counts[:, (values < 0) | (values > 1)].sum() / total)
    out["out11"] = float(counts[:, np.abs(values) > 1].sum() / total)
    return out

def g(v):
    return f"{v:.4g}"

def stats_table(names, per_split, units=None, label="split"):
    head = ["dim", *(["unit"] if units else []), label, *STAT_KEYS, "% neg"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for d, name in enumerate(names):
        for split, s in per_split.items():
            cells = [f"`{name}`", *([units[d]] if units else []), split,
                     *(g(s[k][d]) for k in STAT_KEYS), f"{s['neg'][d]:.1f}"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def measured_range(per_split):
    lo = min(float(s["min"].min()) for s in per_split.values())
    hi = max(float(s["max"].max()) for s in per_split.values())
    if lo >= 0 and hi <= 1:
        verdict = "in [0, 1]"
    elif lo >= -1 and hi <= 1:
        verdict = "in [-1, 1], not [0, 1]"
    else:
        out = max(s["out11"] for s in per_split.values())
        verdict = f"not in [-1, 1]: {100 * out:.1f}% of values outside (worst split)"
    return lo, hi, verdict


# --- C: loading (tables: full splits; images: a few episodes) ---
def split_episodes(split, root, n=None):
    eps, _ = data.drop_missing_videos(split, data.get_split_episodes(split, root), root=root,
                                      verbose=False)
    return eps if n is None else eps[:n]

def load_tables(cfg, stats, quat_ref, keep_ranges):
    """{split: load_split(with_frames=False) + raw tables + get_pairs(inputs, targets)}."""
    root, out = dataset_root(cfg), {}
    for split in SPLITS:
        sp = data.load_split(split, root=root, stats=stats, quat_ref=quat_ref, with_frames=False,
                             chunk=cfg.get("chunk"), keep_ranges=keep_ranges,
                             history=run_history(cfg))
        raw = [data.load_table(ep, root) for ep in split_episodes(split, root)]   # file values
        sp["raw_states"] = np.concatenate([r[0] for r in raw]).astype(np.float64)
        sp["raw_actions"] = np.concatenate([r[1] for r in raw]).astype(np.float64)
        N = len(sp["states"])
        assert len(sp["raw_states"]) == N and np.all(np.diff(sp["episode_index"]) >= 0)
        # the real get_pairs: frames replaced by a zero-size broadcast (no image is read here)
        fake = dict(sp, frames=np.broadcast_to(np.zeros((1, 1, 1, 3), np.uint8), (N, 1, 1, 3)))
        sp["inputs"], sp["targets"] = data.get_pairs(fake, sp["pair_index"])
        del sp["inputs"]["image_input"], sp["targets"]["next_image_output"]
        out[split] = sp
    return out

def load_image_subsets(cfg, stats, quat_ref, keep_ranges, n):
    root = dataset_root(cfg)
    return {split: data.load_split(split, root=root, stats=stats, quat_ref=quat_ref,
                                   episodes=split_episodes(split, root, n), chunk=cfg.get("chunk"),
                                   keep_ranges=keep_ranges, history=run_history(cfg))
            for split in SPLITS}

def instruction_counts(root, episodes):
    texts = Counter()
    for ep in episodes:
        col = pd.read_parquet(data.table_path(ep, root), columns=["instruction.text"])
        texts.update(col["instruction.text"].map(str))
    return texts


# --- D: the worked example ---
def pick_example_pair(val, image_episodes):
    """The one val pair t shown in every section's worked example (the same t for every item).
    val: the full val tables (pair_index, episode_index, states, actions: normalised, flat rows).
    image_episodes: the val episodes that have frames here (the first --image-episodes).
    Returns a flat row t with t in val["pair_index"] and val["episode_index"][t] in image_episodes."""
    # rule: the median-speed MOVING pair of the image episodes, so offsets and changes are not ~0;
    # deterministic (no randomness), so the card reproduces
    t, ep = val["pair_index"], val["episode_index"]
    first = np.searchsorted(ep, ep[t])                         # first row of each pair's episode
    ok = np.isin(ep[t], image_episodes) & (t - 2 >= first)    # has frames; Δa is a real change
    std = np.asarray(val["stats"]["action"]["std"])[:3]
    speed = np.linalg.norm((val["actions"][t, :3] - val["actions"][t - 1, :3]) * std, axis=1)  # m / frame
    cand = np.flatnonzero(ok & (speed > STATIC_M))            # moving: > 0.1 mm per frame (diary 10-23)
    if len(cand) == 0:
        raise ValueError("no moving pair in the image episodes: pass --example-t")
    order = cand[np.argsort(speed[cand], kind="stable")]      # ties: the earlier pair
    return int(t[order[len(order) // 2]])


# --- E: independent re-implementation of the transformations (the checks) ---
def canon(x, quat_ref, key):
    """Sign flip of every quaternion where dot(q, ref) < 0 (written again, not data.py's)."""
    x = x.copy()
    for e in (quat_ref or {}).get(key, []):
        q = x[:, e["start"]:e["start"] + 4]                  # a view into x
        q[q @ np.asarray(e["q"]) < 0] *= -1
    return x

def prev_masks(sp, keep_ranges):
    """Per pair t: is s(t-1) in the episode's kept window (Δs), is a(t-2) in the episode (Δa)."""
    t, ep = sp["pair_index"], sp["episode_index"]
    first = np.searchsorted(ep, ep[t])                       # first row of t's episode (ep sorted)
    lo = np.array([0 if keep_ranges is None else keep_ranges[int(e)][0] for e in ep[t]])
    return t - 1 >= first + lo, t - 2 >= first, t - 2 >= first + lo


# --- F: the items ---
def f32_check(formula, pipe, mag):
    """(max |diff|, max |diff| / float32 bound, number of values, passed) over the splits."""
    diff = ratio = 0.0
    for s in SPLITS:
        d = np.abs(np.asarray(formula[s], np.float64) - pipe[s])
        bound = F32_C * EPS32 * (np.asarray(mag[s]) + np.abs(pipe[s])) + 1e-30
        diff, ratio = max(diff, float(d.max())), max(ratio, float((d / bound).max()))
    return diff, ratio, sum(pipe[s].size for s in SPLITS), ratio <= 1.0

def make_item(key, title, role, names, raw, formula, pipe, mag, steps, special, model, bound,
              example_j, units=None, meanings=None, source=""):
    """raw / formula / pipe / mag: {split: (P, d)}. Final statistics = pipe (what the model gets).
    "worst" = (|y|, split, pair position j, dim) of the largest |final| value, for the Tails line."""
    final = {s: dim_stats(pipe[s]) for s in SPLITS}
    worst = max((float(np.abs(pipe[s]).max()), s, *np.unravel_index(np.abs(pipe[s]).argmax(), pipe[s].shape))
                for s in SPLITS)
    tails = {s: [100.0 * float((np.abs(pipe[s]) > c).mean()) for c in TAILS] for s in SPLITS}
    return {"key": key, "title": title, "role": role, "names": names, "units": units,
            "meanings": meanings, "source": source,
            "shape": f"{tuple(pipe['val'].shape[1:])} {pipe['val'].dtype}",
            "raw": {s: dim_stats(raw[s]) for s in SPLITS} if raw else None,
            "final": final, "range": measured_range(final), "bound": bound,
            "steps": steps, "check": f32_check(formula, pipe, mag), "worst": worst, "tails": tails,
            "short": SHORT["chunk" if key.startswith("chunk") else key],
            "special": special, "model": model,
            "example": [(names[d], None if raw is None else raw["val"][example_j][d],
                         pipe["val"][example_j][d]) for d in range(len(names))]}

def row_label(sp, t):
    """'episode e, frame f' of flat row t of a split."""
    ep = int(sp["episode_index"][t])
    return f"episode {ep}, frame {t - int(np.searchsorted(sp['episode_index'], ep))}"

def add_tails(it, T):
    """Special-case line: how heavy the tails of the final values are, and where the largest is."""
    v, s, j, d = it["worst"][:4]
    shares = "; ".join(f"{s2}: " + ", ".join(f"> {c}: {p:.2f}%" for c, p in zip(TAILS, it["tails"][s2]))
                       for s2 in SPLITS)
    it["special"].append(f"Tails (|final|): {shares}. Largest |value| {g(v)}: `{it['names'][d]}`, {s} "
                         f"{row_label(T[s], int(T[s]['pair_index'][j]))}.")

def table_items(T, cfg, stats, quat_ref, keep_ranges, ex_t, model):
    root = dataset_root(cfg)
    s_names, a_names = data.feature_names(root)
    sd, ad = len(s_names), len(a_names)
    s_info, a_info = [dim_info(n) for n in s_names], [dim_info(n) for n in a_names]
    ms, ss = (np.asarray(stats["observation.state"][k]) for k in ("mean", "std"))
    ma, sa = (np.asarray(stats["action"][k]) for k in ("mean", "std"))
    ex_j = int(np.searchsorted(T["val"]["pair_index"], ex_t))
    assert T["val"]["pair_index"][ex_j] == ex_t, f"--example-t {ex_t} is not a val pair"

    P = {s: T[s]["pair_index"] for s in SPLITS}
    rs = {s: T[s]["raw_states"] for s in SPLITS}
    ra = {s: T[s]["raw_actions"] for s in SPLITS}
    cs = {s: canon(rs[s], quat_ref, "observation.state") for s in SPLITS}
    ca = {s: canon(ra[s], quat_ref, "action") for s in SPLITS}
    norm_s = lambda x: (x - ms) / (ss + 1e-8)
    norm_a = lambda x: (x - ma) / (sa + 1e-8)
    # size of a value as float32 rounds it, in normalised units (see F32_C): (|x| + |mean|) / std
    mag_s = lambda x: (np.abs(x) + np.abs(ms)) / ss
    mag_a = lambda x: (np.abs(x) + np.abs(ma)) / sa
    inp = {s: T[s]["inputs"] for s in SPLITS}
    tgt = {s: T[s]["targets"] for s in SPLITS}
    count = lambda f: ", ".join(f"{s} {f(s)}" for s in SPLITS)

    # common steps and special cases
    read = lambda col: (f"read the parquet column `{col}` (float32)",
                        where(data.load_table, f'df["{col}"]'), "none")
    q_entries = (quat_ref or {}).get("action", [])
    canon_a_step = (f"canonical quaternion sign: q -> -q where q·q_ref < 0, q_ref = "
                    f"{np.round(q_entries[0]['q'], 4).tolist() if q_entries else '—'}",
                    where(data.canonicalise_quaternions, "np.where(q @") + ", called in "
                    + where(data.load_table, "canonicalise_quaternions(actions"),
                    "q_ref = Markley mean of the train kept rows ("
                    + where(data.compute_quat_reference, "average_quaternion(q)") + "), quat_reference.json")
    norm_step = lambda key, short: (
        f"normalise: {short}_n = ({short} - mean) / (std + 1e-8), std floored at 1e-6",
        where(data.normalise, "return (x - mean)") + ", called in "
        + where(data.load_split, f"normalise(ep_{'states' if key == 'observation.state' else 'actions'}"),
        f"norm_stats.json `{key}` mean/std: train episodes, kept rows only ("
        + where(data.compute_norm_stats, "x.std(axis=0)") + ")")
    pair_step = ("keep ranges + pair rule: t only where the whole window (a(t-1) .. a(t+k-1), "
                 "s(t+1)) lies in the episode's kept rows; selects rows, changes no value",
                 where(data.pair_positions, "np.arange(lo + 1") + ", " + where(data.load_split, "pair_positions(lo, hi, chunk)"),
                 "keep_ranges.csv (copied into the run, hash-checked)")
    n_pairs = count(lambda s: f"{len(P[s])} of {T[s]['n_pairs_untrimmed']} untrimmed")
    flipped = count(lambda s: f"{100 * (ra[s][P[s], 3:7] @ np.asarray(q_entries[0]['q']) < 0).mean():.1f}%"
                    if q_entries else "—")
    rows_vs_pairs = ("Normalised mean / std are not exactly 0 / 1: the norm statistics use every kept "
                     "row of the train episodes, these tables only the pair rows t (and val is another set of episodes).")
    thr = glitch_thresholds(T["train"])
    glitches = {s: glitch_frames(T[s], thr).sum(0) for s in SPLITS}
    gl_text = "; ".join(f"`{s_names[d]}`: " + count(lambda s: int(glitches[s][d]))
                        for d in range(sd) if any(glitches[s][d] for s in SPLITS))
    state_special = [
        f"Pairs: {n_pairs}.",
        "No quaternion in `observation.state` (quat_reference.json lists none): the canonical-sign step is a no-op here.",
        "`endo360_joint_pos[0]` is the Endo360 needle driver (the tool), not the camera; it is not commanded by the action.",
        f"One-frame glitches (`references.glitch_frames`, jump > {GLITCH_G} × train p99 that comes back), frames per split: {gl_text or 'none'}.",
        rows_vs_pairs]
    state_model = [
        f"`state_input` ({where(swm.build_actor_world_model, 'name=\"state_input\"')}) is read by the actor's "
        f"Concatenate ({where(swm.build_actor_world_model, 'actor_features = layers.Concatenate')}) and the WM's "
        f"Concatenate ({where(swm.build_actor_world_model, 'wm_features = layers.Concatenate')}): raw, no layer in between.",
        *([f"State skip (1a): also added directly to the state head's output, next_state = state_input + Dense(s_x) "
           f"({where(swm.build_actor_world_model, 'layers.Add(name=\"next_state_output\")')})."]
          if cfg.get("state_skip") else [])]
    items = []

    # 1-2: state t and state t+1
    for key, title, role, shift, pipe in [
            ("state_input", "state_input (state t)", "input of both branches (actor and world model)",
             0, {s: inp[s]["state_input"] for s in SPLITS}),
            ("next_state", "next_state target (state t+1)", "target of the world model's state head "
             "(`next_state_output`, MSE, weight " + str(run_config.loss_weights(cfg)["next_state_output"]) + ")",
             1, {s: tgt[s]["next_state_output"] for s in SPLITS})]:
        items.append(make_item(
            key, title, role, s_names, {s: rs[s][P[s] + shift] for s in SPLITS},
            {s: norm_s(cs[s][P[s] + shift]) for s in SPLITS}, pipe,
            {s: mag_s(cs[s][P[s] + shift]) for s in SPLITS},
            [read("observation.state"), pair_step, norm_step("observation.state", "s"),
             (f"get_pairs: `data[\"states\"][t{' + 1' if shift else ''}]`",
              where(data.get_pairs, f'data["states"][t{" + 1" if shift else ""}]'), "none")],
            state_special, state_model if shift == 0 else
            [f"Compared with `next_state_output` ({where(swm.build_actor_world_model, 'next_state_output = layers.Add') if cfg.get('state_skip') else where(swm.build_actor_world_model, 'name=\"next_state_output\"')}); "
             f"loss MSE ({where(run_config.build_model, '\"next_state_output\": \"mse\"')})."],
            "unbounded (z-score of an angle in rad)", ex_j,
            units=[u for _, u in s_info], meanings=[m for m, _ in s_info],
            source=f"`observation.state` at row t{' + 1' if shift else ''}"))

    # 3: action t, the WM's input
    a_special = [f"Pairs: {n_pairs}.",
                 f"Canonical quaternion: share of pair rows whose sign is flipped: {flipped}; the furthest train "
                 f"orientation is {q_entries[0]['max_angle_deg']:.1f}° from q_ref (sign ambiguous only near 180°)."
                 if q_entries else "No canonical quaternion in this run.",
                 "The action is a commanded pose (7-D, one arm); the Endo360 joint has no command.", rows_vs_pairs]
    items.append(make_item(
        "action_input", "action_input (action t, world model)", "input of the world model only",
        a_names, {s: ra[s][P[s]] for s in SPLITS}, {s: norm_a(ca[s][P[s]]) for s in SPLITS},
        {s: inp[s]["action_input"] for s in SPLITS}, {s: mag_a(ca[s][P[s]]) for s in SPLITS},
        [read("action"), canon_a_step, pair_step, norm_step("action", "a"),
         ("get_pairs: `data[\"actions\"][t]`", where(data.get_pairs, '"action_input":'), "none")],
        a_special,
        [f"`action_input` ({where(swm.build_actor_world_model, 'name=\"action_input\"')}) is read only by the WM's "
         f"Concatenate ({where(swm.build_actor_world_model, 'wm_features = layers.Concatenate')}); the actor never sees a(t)."],
        "unbounded (z-score)", ex_j, units=[u for _, u in a_info], meanings=[m for m, _ in a_info],
        source="`action` at row t"))

    # 4-6: the actor's chunk target, steps 1 / 8 / 16
    if cfg.get("chunk"):
        c = stats["action_chunk"]
        scale_dim = np.empty((c["k"], ad))
        for gi, dims in enumerate(c["groups"]):
            scale_dim[:, dims] = np.asarray(c["scale"])[:, gi:gi + 1]
        group_names = ["+".join(a_names[d].rsplit(".", 1)[-1] for d in dims) for dims in c["groups"]]
        for j in CHUNK_STEPS:
            i = j - 1
            q = slice(q_entries[0]["start"], q_entries[0]["start"] + 4) if q_entries else None
            raw_flips = count(lambda s: int(((ra[s][P[s] + i, q] * ra[s][P[s] - 1, q]).sum(1) < 0).sum())
                              if q else 0)
            items.append(make_item(
                f"chunk_{j}", f"actor chunk target, step {j} (a(t{f'+{i}' if i else ''}) − a(t−1))",
                f"target of the actor (`action_output[:, {i}]`, Huber, weight 1.0)",
                a_names, {s: ra[s][P[s] + i] - ra[s][P[s] - 1] for s in SPLITS},
                {s: (ca[s][P[s] + i] - ca[s][P[s] - 1]) / scale_dim[i] for s in SPLITS},
                {s: tgt[s]["action_output"][:, i] for s in SPLITS},
                {s: (mag_a(ca[s][P[s] + i]) + mag_a(ca[s][P[s] - 1])) * sa / scale_dim[i] for s in SPLITS},
                [read("action"), canon_a_step, pair_step, norm_step("action", "a"),
                 (f"chunk offset: y_i = (a_n(t+i) − a_n(t−1)) × F_i, F_i = (std + 1e-8) / scale[i, group] "
                  f"= o_i / scale[i, group] in raw units",
                  where(data.chunk_targets, "return ((data") + ", F: " + where(data.chunk_factor, "return (std + 1e-8)"),
                  f"norm_stats.json `action_chunk`: per step and group ({', '.join(group_names)}) "
                  f"max(median, {c['floor_c']} × mean) of the offset norm over train pairs ("
                  + where(data.compute_chunk_scale, "scale = np.maximum(np.median(norms") + f"); step {j}: "
                  + ", ".join(f"{gn} {g(c['scale'][i][gi])}" for gi, gn in enumerate(group_names)))],
                [f"Anchor = a(t−1), the last command before t: step 1 is a(t) − a(t−1), one frame of motion.",
                 "The quaternion offset is per component (q(t+i) − q(t−1)), not a rotation angle; "
                 "the group scale is the norm of the 4-D component offset.",
                 "The mean cancels in the difference: the target does not depend on the action mean.",
                 f"Float32: the pipeline subtracts two normalised float32 values and multiplies by F_{j} = std / scale "
                 f"(per dim: {', '.join(g(v) for v in sa / scale_dim[i])}), so its rounding error is amplified: "
                 f"see the Check line (harmless next to the Huber loss, but not bit-exact).",
                 f"Raw offsets use the file signs: pairs whose raw q(t+{i}) and q(t−1) have opposite signs "
                 f"(q·q < 0, raw component offset up to ~1.4): {raw_flips}. The canonical step removes them.",
                 f"Pairs: {n_pairs} (t from lo + 1 to hi − k + 1)."],
                [f"Produced by the actor: Dense(256, relu) → Dense({c['k']}×{ad}, "
                 f"{model.get_layer('action_chunk').get_config()['activation']}) `action_chunk` → Reshape "
                 f"({where(swm.build_actor_world_model, 'name=\"action_chunk\"')}); loss Huber "
                 f"({where(run_config.build_model, 'keras.losses.Huber()')})."],
                "unbounded (offset / typical offset size)", ex_j,
                units=[u for _, u in a_info], meanings=[m for m, _ in a_info],
                source=f"`action` at rows t + {i} and t − 1"))

    # 7-8: history, the Δs and Δa parts
    if run_history(cfg):
        hs = np.asarray(stats["history_scale"]["scale"])
        ok_s, ok_a, ok_a_win = {}, {}, {}
        for s in SPLITS:
            ok_s[s], ok_a[s], ok_a_win[s] = prev_masks(T[s], keep_ranges)
        h_model = (f"`history_input` ({where(swm.build_actor_world_model, 'name=\"history_input\"')}) is appended last "
                   f"to the actor's and the WM's Concatenate ({where(swm.build_actor_world_model, '+ history)   # NEW (history): + h(t)')}): "
                   "no layer of its own, only more input rows of the first Dense of each branch.")
        t2 = {s: np.maximum(P[s] - 2, 0) for s in SPLITS}
        for key, title, names, raw, formula, pipe, mag, steps, special in [
            ("history_ds", "history Δs(t) = s(t) − s(t−1)", [f"Δ{n}" for n in s_names],
             {s: np.where(ok_s[s][:, None], rs[s][P[s]] - rs[s][P[s] - 1], 0.0) for s in SPLITS},
             {s: np.where(ok_s[s][:, None], cs[s][P[s]] - cs[s][P[s] - 1], 0.0) / (ss + 1e-8) / hs[:sd] for s in SPLITS},
             {s: inp[s]["history_input"][:, :sd] for s in SPLITS},
             {s: (mag_s(cs[s][P[s]]) + mag_s(cs[s][P[s] - 1])) / hs[:sd] for s in SPLITS},
             [read("observation.state"), pair_step, norm_step("observation.state", "s"),
              ("Δs = s_n(t) − s_n(t−1), 0 if s(t−1) is not in the kept window",
               where(data.history_changes, "ds = np.where") + ", rule: " + where(data.prev_in_window, "return t - 1 >= first_kept[t]"), "none"),
              ("divide by the history scale (no mean subtracted: 0 = no change)",
               where(data.load_split, 'out["history"][pair_index] = (h / scale)'),
               "norm_stats.json `history_scale`[:state_dim]: std of the real changes over train pairs ("
               + where(data.compute_history_scale, "scale = np.concatenate") + ")")],
             [f"Fallback (0 = change unknown): {count(lambda s: int((~ok_s[s]).sum()))} pairs."]),
            ("history_da", "history Δa(t−1) = a(t−1) − a(t−2)", [f"Δ{n}" for n in a_names],
             {s: np.where(ok_a[s][:, None], ra[s][P[s] - 1] - ra[s][t2[s]], 0.0) for s in SPLITS},
             {s: np.where(ok_a[s][:, None], ca[s][P[s] - 1] - ca[s][t2[s]], 0.0) / (sa + 1e-8) / hs[sd:] for s in SPLITS},
             {s: inp[s]["history_input"][:, sd:] for s in SPLITS},
             {s: (mag_a(ca[s][P[s] - 1]) + mag_a(ca[s][t2[s]])) / hs[sd:] for s in SPLITS},
             [read("action"), canon_a_step, pair_step, norm_step("action", "a"),
              ("Δa = a_n(t−1) − a_n(t−2), 0 if a(t−2) is not in the episode",
               where(data.history_changes, "da = anchor") + ", rule: " + where(data.prev2_in_episode, "return (t >= 2)"), "none"),
              ("divide by the history scale (no mean subtracted)",
               where(data.load_split, 'out["history"][pair_index] = (h / scale)'),
               "norm_stats.json `history_scale`[state_dim:] (" + where(data.compute_history_scale, "scale = np.concatenate") + ")")],
             [f"Fallback (0 = change unknown): {count(lambda s: int((~ok_a[s]).sum()))} pairs (t = lo + 1 at an episode start).",
              f"Unlike Δs, the rule is only 'same episode': a(t−2) may lie before the kept window "
              f"({count(lambda s: int((ok_a[s] & ~ok_a_win[s]).sum()))} pairs); those rows exist, they are just untrimmed.",
              "The Δa of the quaternion is per component, like the chunk."]),
        ]:
            items.append(make_item(key, title, "input of both branches (part of `history_input`)", names,
                                   raw, formula, pipe, mag, steps, special, [h_model],
                                   "unbounded (change / std of the change)", ex_j,
                                   units=[dim_info(n[1:])[1] for n in names],
                                   meanings=["change of: " + dim_info(n[1:])[0] for n in names],
                                   source="state (Δs) / action (Δa) rows t, t−1 / t−1, t−2"))

    # 9: language (zeros)
    lang = {s: inp[s]["language_input"] for s in SPLITS}
    nonzero = sum(int(np.count_nonzero(lang[s])) for s in SPLITS)
    texts = {s: instruction_counts(root, split_episodes(s, root)) for s in SPLITS}
    all_texts = texts["train"] + texts["val"]
    zero = {s: np.zeros((len(P[s]), 1)) for s in SPLITS}
    item = make_item(
        "language", "language_input", "input of both branches", ["all 512 dims"], None,
        zero, {s: lang[s][:, :1] for s in SPLITS}, zero,
        [("constant: zeros(512) for every pair", where(data.get_pairs, '"language_input"'), "none")],
        [f"The parquet has `instruction.text`, but it is not used: {len(all_texts)} distinct strings over the "
         f"train + val rows, most common: " + "; ".join(f"“{t}” ({n})" for t, n in all_texts.most_common(3)) + ".",
         f"Non-zero values in the arrays given to the model: {nonzero}."],
        [f"`language_input` ({where(swm.build_actor_world_model, 'name=\"language_input\"')}) goes into both "
         "Concatenates. Its Dense kernel rows get a zero gradient (input 0), so with Adam they never change from their initial values."],
        "exactly 0", ex_j, units=["—"], meanings=["no information"], source="none (constant)")
    item["check"] = (float(nonzero), 0.0, sum(lang[s].size for s in SPLITS), nonzero == 0)
    item["shape"] = f"{tuple(lang['val'].shape[1:])} {lang['val'].dtype}"
    for it in items:   # language (constant) gets no tails line
        add_tails(it, T)
    items.append(item)
    return items, ex_j


# --- G: images (a few episodes) ---
def native_pass(sub, root, every, ex_t):
    """Decode the subset episodes at native resolution: uint8 histograms of the frames at pair rows
    t (every `every`-th) and t + 1, and the native example frames t, t+1."""
    hist = {"t": np.zeros((3, 256), np.int64), "t1": np.zeros((3, 256), np.int64)}
    ex, shape, ep_idx = {}, None, sub["episode_index"]
    pairs = sub["pair_index"][::every]
    for ep in np.unique(ep_idx):
        start = int(np.searchsorted(ep_idx, ep))
        local = pairs[ep_idx[pairs] == ep] - start
        want = {"t": set(local.tolist()), "t1": set((local + 1).tolist())}
        ex_local = ex_t - start if ex_t is not None and ep_idx[ex_t] == ep else None
        with av.open(str(data.video_path(int(ep), data.CAMERA, root))) as container:
            for i, frame in enumerate(container.decode(video=0)):
                hit = [k for k in want if i in want[k]]
                if not hit and i not in (ex_local, None if ex_local is None else ex_local + 1):
                    continue
                a = frame.to_ndarray(format="rgb24")
                shape = a.shape
                for k in hit:
                    for c in range(3):
                        hist[k][c] += np.bincount(a[..., c].ravel(), minlength=256)
                if ex_local is not None and i in (ex_local, ex_local + 1):
                    ex["t" if i == ex_local else "t1"] = a
    return hist, ex, shape

def final_hist(frames, rows, batch=256):
    h = np.zeros((3, 256), np.int64)
    for b in range(0, len(rows), batch):
        f = frames[rows[b:b + batch]]
        for c in range(3):
            h[c] += np.bincount(f[..., c].ravel(), minlength=256)
    return h

def cache_vs_direct(split, sub, root):
    """Max |cache - direct decode| on the subset's frames, if the cache exists and starts with
    the same episodes (read only, never built)."""
    npy = data.CACHE_DIR / root.name / f"{split}_{data.CAMERA.split('.', 2)[-1]}_{data.IMG_SIZE}.npy"
    if not npy.exists() or not npy.with_suffix(".json").exists():
        return "no cache"
    eps = [int(e) for e in np.unique(sub["episode_index"])]
    if json.loads(npy.with_suffix(".json").read_text())["episodes"][:len(eps)] != eps:
        return "the cache starts with other episodes"
    cache, n, diff = np.load(npy, mmap_mode="r"), len(sub["frames"]), 0
    for b in range(0, n, 256):
        diff = max(diff, int(np.abs(cache[b:min(b + 256, n)].astype(np.int16) - sub["frames"][b:b + 256]).max()))
    return f"max |cache − direct decode| = {diff} (uint8)"

def image_items(sub, cfg, every, ex_t, model, fig_dir):
    root = dataset_root(cfg)
    info = data.load_info(root)["features"][data.CAMERA]
    nat, ex_nat, shapes, finals, checks, caches = {}, None, None, {}, {}, {}
    for s in SPLITS:
        nat[s], ex, shapes = native_pass(sub[s], root, every, ex_t if s == "val" else None)
        if s == "val":
            ex_nat = ex
        t = sub[s]["pair_index"]
        finals[s] = {"t": final_hist(sub[s]["frames"], t), "t1": final_hist(sub[s]["frames"], t + 1)}
        probe = np.unique(np.concatenate([t[:64], [ex_t] if s == "val" else []]).astype(np.int64))
        inputs, targets = data.get_pairs(sub[s], probe)   # the real path, on a few pairs
        for k, got in (("t", inputs["image_input"]), ("t1", targets["next_image_output"])):
            checks.setdefault(k, ({}, {}, {}))
            checks[k][0][s] = sub[s]["frames"][probe + (k == "t1")] / 255.0   # formula, float64
            checks[k][1][s], checks[k][2][s] = got, 0.0                        # pipeline, mag
        caches[s] = cache_vs_direct(s, sub[s], root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pics = {}
    for k in ("t", "t1"):
        for res, arr in (("native", ex_nat[k]), ("224", sub["val"]["frames"][ex_t + (k == "t1")])):
            name = f"example_{k}_{res}.png"
            Image.fromarray(np.asarray(arr)).save(fig_dir / name)
            pics[f"{k}_{res}"] = name
    H, W = shapes[:2]
    vals = np.arange(256, dtype=np.float64)
    items = []
    for k, key, title, role, rows in [
            ("t", "image_input", "image_input (frame t)", "input of the shared encoder (actor and world model)", "t"),
            ("t1", "next_image", "next_image target (frame t+1)", "target of the world model's image decoder "
             f"(`next_image_output`, MSE, weight {run_config.loss_weights(cfg)['next_image_output']})", "t + 1")]:
        final = {s: hist_stats(finals[s][k], vals / 255.0) for s in SPLITS}
        cy, cx = H // 2, W // 2
        f224 = sub["val"]["frames"][ex_t + (k == "t1")]
        items.append({
            "key": key, "title": title, "role": role, "names": ["R", "G", "B"],
            "units": ["0–255", "0–255", "0–255"], "meanings": ["red", "green", "blue"],
            "source": f"video `{data.CAMERA}` frame at row {rows}: {info['info']['video.codec']} "
                      f"{info['info']['video.pix_fmt']}, {W}×{H}, {info['info']['video.fps']} fps",
            "shape": "(224, 224, 3) float32",
            "raw": {s: hist_stats(nat[s][k], vals) for s in SPLITS},
            "final": final, "range": measured_range(final), "bound": "[0, 1] (uint8 / 255)",
            "short": SHORT[key],
            "steps": [
                (f"decode {info['info']['video.codec']} {info['info']['video.pix_fmt']} → rgb24 and resize "
                 f"{W}×{H} → 224×224 in one PyAV reformat (bilinear, the default; aspect ratio not kept)",
                 where(data.load_frames, 'to_ndarray(format="rgb24"'), "none"),
                ("full splits: stored once as uint8 in the memory-mapped cache (subsets decode directly)",
                 where(data.cached_frames, "out[start:start + T] = ep_frames"), "none"),
                (f"get_pairs: `frames[{rows.replace(' ', '')}].astype(float32) / 255`",
                 where(data.get_pairs, f"images_t{'1' if k == 't1' else ''} = data"), "none")],
            "check": f32_check(*checks[k]),   # on the first 64 pairs of each split (+ the example)
            "special": [
                f"Images: only the first episodes of each split (frames: {', '.join(f'{s} {len(sub[s]['frames'])}' for s in SPLITS)}; "
                f"pairs: {', '.join(f'{s} {len(sub[s]['pair_index'])}' for s in SPLITS)}); raw = every {every}th pair row.",
                f"Resize: x ÷ {W / 224:.2f}, y ÷ {H / 224:.2f}: objects are squeezed horizontally by {W / H:.2f}×.",
                "Only the left endoscope camera (the right video is an exact copy); episodes without that video "
                f"are skipped (in these splits: {', '.join(f'{s} {len(sub[s]['skipped_episodes'])}' for s in SPLITS)}).",
                "Raw = after the decoder's YUV 4:2:0 → RGB conversion (the stored pixels are YUV).",
                f"Frame cache (what training read): " + "; ".join(f"{s}: {caches[s]}" for s in SPLITS) + ".",
                "Keep ranges select the pairs, not the frames: the cache holds every frame."],
            "model": ([f"`image_input` ({where(swm.build_actor_world_model, 'name=\"image_input\"')}) → simple_cnn, first layer "
                       f"Conv2D(32, 3, stride 2, relu) ({where(encoders.simple_cnn, 'Conv2D(32')}), no preprocessing layer: the "
                       "encoder sees [0, 1] directly. Shared by the actor and the world model."] if k == "t" else
                      [f"Compared with `next_image_output`, the decoder's last Conv2DTranspose(3, sigmoid) "
                       f"({where(swm.build_actor_world_model, 'activation=\"sigmoid\"')}); loss MSE."]),
            "example": [(c, f"{int(ex_nat[k][cy, cx, i])} (frame mean {ex_nat[k][..., i].mean():.1f})",
                         f"{f224[112, 112, i] / 255:.4f} (frame mean {f224[..., i].mean() / 255:.4f})")
                        for i, c in enumerate("RGB")],
            "pics": [pics[f"{k}_native"], pics[f"{k}_224"]],
        })
    return items, pics


# --- H: predictions of the run's best weights on the val image episodes ---
def predictions(model, sub_val, ex_t, batch):
    acts, states, t_acts, t_states = [], [], [], []
    bins = np.linspace(0.0, 1.0, PRED_BINS + 1)
    hist = np.zeros((3, PRED_BINS), np.int64)
    lo, hi, s1, s2, n = np.full(3, np.inf), np.full(3, -np.inf), np.zeros(3), np.zeros(3), 0
    ex_j = int(np.searchsorted(sub_val["pair_index"], ex_t))
    ex_img, seen = None, 0
    for inputs, targets in make_dataset(sub_val, batch):
        a, s, img = (np.asarray(o) for o in model(inputs, training=False))
        acts.append(a); states.append(s)
        t_acts.append(targets["action_output"].numpy()); t_states.append(targets["next_state_output"].numpy())
        px = img.reshape(-1, 3).astype(np.float64)
        lo, hi = np.minimum(lo, px.min(0)), np.maximum(hi, px.max(0))
        s1, s2, n = s1 + px.sum(0), s2 + (px ** 2).sum(0), n + len(px)
        for c in range(3):
            hist[c] += np.histogram(px[:, c], bins=bins)[0]
        if seen <= ex_j < seen + len(a):
            ex_img = img[ex_j - seen]
        seen += len(a)
    img_stats = hist_stats(hist, (bins[:-1] + bins[1:]) / 2)
    img_stats.update(min=lo, max=hi, mean=s1 / n, std=np.sqrt(s2 / n - (s1 / n) ** 2))   # exact
    return {"action": np.concatenate(acts), "state": np.concatenate(states),
            "t_action": np.concatenate(t_acts), "t_state": np.concatenate(t_states),
            "image": img_stats, "ex_img": ex_img, "ex_j": ex_j}


# --- I: the markdown ---
def render_item(n, it, ex_label):
    L = [f"## {n}. {it['title']}", "",
         f"- **Role:** {it['role']}",
         f"- **Shape, dtype:** {it['shape']} (per pair; the model gets batches of these)",
         f"- **Raw source:** {it['source']}", ""]
    if it["meanings"]:
        L += ["| dim | physical meaning | unit |", "|---|---|---|"]
        L += [f"| `{nm}` | {m} | {u} |" for nm, m, u in zip(it["names"], it["meanings"], it["units"])] + [""]
    if it["raw"]:
        L += ["**Raw values** (raw units, before any step)", "", stats_table(it["names"], it["raw"], it["units"]), ""]
    else:
        L += ["**Raw values:** none (no data source).", ""]
    L += ["**Transformations, in order**", ""]
    L += [f"{i}. {txt}. Code: {loc}. Statistics: {st}." for i, (txt, loc, st) in enumerate(it["steps"], 1)] + [""]
    diff, ratio, n_vals, ok = it["check"]
    L += [f"**Check** (the formulas above, recomputed here in float64 from the raw values, vs what `get_pairs` "
          f"gives the model): max |difference| {g(diff)} over {n_vals} values, at most {g(ratio)} × the float32 "
          f"rounding bound → {'OK' if ok else '**MISMATCH**'}", ""]
    L += ["**Final values** (what the model receives)", "", stats_table(it["names"], it["final"]), ""]
    lo, hi, verdict = it["range"]
    L += [f"**In [0, 1]? [-1, 1]? Unbounded?** By construction: {it['bound']}. Measured: {g(lo)} … {g(hi)} → {verdict}.", ""]
    L += ["**Special cases**", ""] + [f"- {s}" for s in it["special"]] + [""]
    L += [f"**Example** ({ex_label})", "", "| dim | raw | final |", "|---|---|---|"]
    L += [f"| `{d}` | {r if isinstance(r, str) else ('—' if r is None else g(r))} | {f if isinstance(f, str) else g(f)} |"
          for d, r, f in it["example"]] + [""]
    if it.get("pics"):
        L += [f"Native and model-input frame: ![]({LINK_PICS}/{it['pics'][0]}) ![]({LINK_PICS}/{it['pics'][1]})", ""]
    L += ["**Model side**", ""] + [f"- {m}" for m in it["model"]] + [""]
    return "\n".join(L)

def render_outputs(n0, model, cfg, pred, sub_val, a_names, s_names, ex_label, pics):
    out, k = [], cfg.get("chunk")
    act = lambda name: model.get_layer(name).get_config().get("activation", "—")
    label = "split"
    # actor
    blocks = [f"## {n0}. Output: actor chunk (`action_output`)", "",
              f"- **Layer:** Dense({k}×{len(a_names)}) `action_chunk`, activation **{act('action_chunk')}**, then Reshape "
              f"({where(swm.build_actor_world_model, 'name=\"action_chunk\"')}).",
              "- **Output range:** unbounded (linear).", f"- **Shape:** {pred['action'].shape[1:]} float32.",
              f"- **Measured** on the val image episodes ({len(pred['action'])} pairs), prediction next to its target:", ""]
    for j in CHUNK_STEPS:
        i = j - 1
        both = {"pred": dim_stats(pred["action"][:, i]), "target": dim_stats(pred["t_action"][:, i])}
        blocks += [f"Step {j}:", "", stats_table(a_names, both, label=label), ""]
    blocks += [f"Example ({ex_label}), step 1/8/16 predicted: " + "; ".join(
        f"step {j}: [{', '.join(g(v) for v in pred['action'][pred['ex_j'], j - 1])}]" for j in CHUNK_STEPS), ""]
    out.append("\n".join(blocks))
    # state
    state_layer = "state_change" if cfg.get("state_skip") else ("state_delta" if cfg.get("state_delta") else "next_state_output")
    both = {"pred": dim_stats(pred["state"]), "target": dim_stats(pred["t_state"])}
    out.append("\n".join([
        f"## {n0 + 1}. Output: next state (`next_state_output`)", "",
        f"- **Layer:** {'state_input + ' if cfg.get('state_skip') else ''}Dense({len(s_names)}) `{state_layer}`, "
        f"activation **{act(state_layer)}**"
        + (f", zero-initialised, so the untrained model is copy_state ({where(swm.build_actor_world_model, 'name=\"state_change\"')})." if cfg.get("state_skip") else "."),
        "- **Output range:** unbounded (linear; normalised state units).",
        f"- **Measured** on the val image episodes ({len(pred['state'])} pairs):", "",
        stats_table(s_names, both, label=label), "",
        f"Example ({ex_label}) predicted: [{', '.join(g(v) for v in pred['state'][pred['ex_j']])}]", ""]))
    # image
    Image.fromarray(np.clip(np.round(pred["ex_img"] * 255), 0, 255).astype(np.uint8)).save(FIG_DIR / "example_t1_pred.png")
    s = pred["image"]
    out.append("\n".join([
        f"## {n0 + 2}. Output: next image (`next_image_output`)", "",
        f"- **Layer:** Conv2DTranspose(3, 3, stride 2), activation **{act('next_image_output')}** "
        f"({where(swm.build_actor_world_model, 'activation=\"sigmoid\"')}).",
        "- **Output range:** (0, 1), open (sigmoid); the target's 0 and 1 are reachable only in the limit.",
        f"- **Measured** on the val image episodes ({s['n']} pixels per channel; min/max/mean/std exact, "
        f"p1/p99 from {PRED_BINS} bins):", "",
        stats_table(["R", "G", "B"], {"pred": s}, label=label), "",
        f"Example ({ex_label}): target ![]({LINK_PICS}/{pics['t1_224']}) predicted ![]({LINK_PICS}/example_t1_pred.png)", ""]))
    return out

def render_summary(items, outputs_rows):
    L = ["| item | shape | transformation | final range (train + val) | in [0, 1]? |", "|---|---|---|---|---|"]
    for it in items:
        lo, hi, verdict = it["range"]
        L.append(f"| {it['title']} | {it['shape']} | {it['short']} | {g(lo)} … {g(hi)} | "
                 f"{'yes' if verdict == 'in [0, 1]' else 'no'} |")
    L += outputs_rows
    return "\n".join(L)


# --- J: main ---
def main():
    global LINK_BASE, LINK_PICS, FIG_DIR
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--image-episodes", type=int, default=3, help="first N episodes of train and val for images")
    p.add_argument("--raw-every", type=int, default=10, help="native-resolution raw images: every N-th pair row")
    p.add_argument("--example-t", type=int, default=None, help="flat val row of the worked example (default: pick_example_pair)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", default="docs/data-card.md")
    args = p.parse_args()

    out_md = Path(args.out)
    FIG_DIR = out_md.with_suffix("")                  # docs/data-card/ for the example frames
    LINK_BASE = Path(os.path.relpath(REPO, out_md.parent.resolve())).as_posix()
    LINK_PICS = FIG_DIR.name
    run = OUTPUTS / args.run
    cfg = json.loads((run / "config.json").read_text())
    stats = data.load_norm_stats(run / "norm_stats.json")
    root, quat_ref, keep_ranges = dataset_root(cfg), run_quat_ref(cfg, run), run_keep_ranges(cfg, run)
    if not (cfg.get("chunk") and run_history(cfg)):
        raise SystemExit(f"{args.run}: this card expects a --chunk --history run")

    T = load_tables(cfg, stats, quat_ref, keep_ranges)
    sub = load_image_subsets(cfg, stats, quat_ref, keep_ranges, args.image_episodes)
    image_eps = [int(e) for e in np.unique(sub["val"]["episode_index"])]
    ex_t = args.example_t if args.example_t is not None else pick_example_pair(T["val"], image_eps)
    ex_ep = int(T["val"]["episode_index"][ex_t])
    assert ex_ep in image_eps and ex_t in set(sub["val"]["pair_index"].tolist()), "example pair needs frames"
    assert sub["val"]["episode_index"][ex_t] == ex_ep        # the subset = the first rows of val
    ex_frame = ex_t - int(np.searchsorted(T["val"]["episode_index"], ex_ep))
    ex_label = f"val pair t = row {ex_t}: episode {ex_ep}, frame {ex_frame}"

    model = build_model(cfg, stats)
    model.load_weights(str(run / "best.weights.h5"))
    t_items, _ = table_items(T, cfg, stats, quat_ref, keep_ranges, ex_t, model)
    i_items, pics = image_items(sub, cfg, args.raw_every, ex_t, model, FIG_DIR)
    pred = predictions(model, sub["val"], ex_t, args.batch)
    s_names, a_names = data.feature_names(root)

    items = i_items[:1] + i_items[1:] + t_items              # images first, as in the request
    rows = [f"| output: actor chunk | {pred['action'].shape[1:]} float32 | Dense, linear | "
            f"{g(pred['action'].min())} … {g(pred['action'].max())} (val pred.) | no (unbounded) |",
            f"| output: next state | {pred['state'].shape[1:]} float32 | state + Dense, linear | "
            f"{g(pred['state'].min())} … {g(pred['state'].max())} (val pred.) | no (unbounded) |",
            f"| output: next image | (224, 224, 3) float32 | sigmoid | "
            f"{g(pred['image']['min'].min())} … {g(pred['image']['max'].max())} (val pred.) | yes, (0, 1) |"]
    failed = [it["title"] for it in items if not it["check"][3]]
    head = [
        "# Data card: inputs and targets of the actor–world model", "",
        f"Generated by `scripts/analysis/report_data.py` on {datetime.date.today()} at commit `{git_commit()}`; "
        "do not edit by hand, rerun the script:", "",
        f"```bash\npython -m scripts.analysis.report_data {' '.join(sys.argv[1:])}\n```", "",
        f"- **Run:** `{args.run}` (its norm_stats.json, quat_reference.json, keep_ranges.csv; all train only)",
        f"- **Dataset:** `{root}`; options: canonical quaternion {cfg.get('canonical_quaternion')}, chunk {cfg.get('chunk')}, "
        f"state skip {cfg.get('state_skip')}, history {run_history(cfg)}, keep ranges `{cfg.get('keep_ranges')}`",
        f"- **Tables:** full train ({len(T['train']['pair_index'])} pairs) and val ({len(T['val']['pair_index'])} pairs). "
        f"**Images and predictions:** first {args.image_episodes} episodes of each split. **Test split: never loaded.**",
        f"- **Worked example:** {ex_label}, the same pair in every section.",
        f"- **Checks:** {'all OK' if not failed else 'MISMATCH in ' + ', '.join(failed)} "
        f"(each transformation recomputed in float64 from the raw values; a value passes within "
        f"{F32_C:g} × eps32 × its float32 magnitude).", "",
        "## Summary", "", render_summary(items, rows), ""]
    sections = [render_item(n, it, ex_label) for n, it in enumerate(items, 1)]
    sections += render_outputs(len(items) + 1, model, cfg, pred, sub["val"], a_names, s_names, ex_label, pics)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(head) + "\n" + "\n".join(sections))
    print(f"wrote {out_md} and {FIG_DIR}/; checks: {'all OK' if not failed else failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
