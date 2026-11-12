"""
look_tool_projection.py

Question: if the commanded tool position (action kuka_pose.x/y/z) is projected into the
endoscope image with the calibration files, does the point land on the tool, so that
"outside the image" detects the frames where the tool is not visible?

Hypothesis: kuka_pose is expressed in the same `world` frame as the camera transform, so
the projection lands on or near the tool (maybe with a constant offset if the pose is the
flange, not the tool tip).
Prediction: the dots land near the tool in visible frames and fall outside the image in
the last 1-3 s of episodes where the tool leaves the view.
Surprises: dots nowhere near the tool (different frames / inverted transform / units), or
good on some episodes but not others (camera moved between sessions).

The maths (all sizes and numbers are read from the dataset folder):
  1. the transform is the camera pose in the world: p_world = R p_cam + t,
     so world -> camera is the inverse:            p_cam   = R^T (p_world - t)
  2. Z <= 0 -> behind the camera -> not visible
  3. pinhole at the calibration size: u = fx X/Z + cx,  v = fy Y/Z + cy
  4. rescale per axis to the video size: u * W_video/W_calib,  v * H_video/H_calib
  5. visible <=> Z > 0 and 0 <= u < W_video and 0 <= v < H_video
  6. --distortion applies plumb_bob, but only to the DRAWN dot: visibility is always
     decided on the undistorted point, because the polynomial is only valid inside the
     calibrated field of view and can fold far-away points back into the image.
Video lag (--lag-frames k): frame t is paired with the pose at t - k; the first k frames
have no pose and are "not evaluated" (NaN).

Part A (table only): ranges of the world position and of the camera coordinates on 10
train episodes spread over the split (including the Part B episodes).
Part B (video): 8 frames per episode with the projected dot -> one PNG grid per episode,
plus a CSV with every frame of the episode.

Output: outputs/analysis/tool_projection/episode_<n>[_lag<k>][_dist].{png,csv}

--strips (timing check, skips Parts A and B): does the projected IN/OUT switch happen at
the frame where the tool TIP really enters / leaves the view?
  transition = a change of `visible` between consecutive evaluated frames; its projected
  frame is the FIRST frame of the new state. Labels: entry (first, if the episode starts
  invisible), exit (last, if it ends invisible), middle_k (all others, e.g. flicker at the
  border; kept raw). One strip per transition: offsets -16..+14 step 2 (includes 0).
  Template CSV (append-only, never overwrites your annotations):
  timing_validation[_lag<k>].csv: episode, transition, direction, projected_frame,
  real_frame, error_frames, notes.
  Lag is a pure shift: projected(lag k) = projected(lag 0) + k, so annotate only the lag-0
  CSV and let --summarise evaluate any lag.
--summarise CSV: error = projected - real (+ lag shift); > 0 means the projection is late.

--all (tables only, lag 0, no margin): visibility of every train and val episode.
  lead / trail = invisible frames before the first / after the last visible frame,
  middle = invisible frames between them; fully invisible episodes are their own bucket
  ("whole"), so start + end + middle + whole = all invisible frames.
  Trimming = keep [first_visible, last_visible] (length L). Pairs follow load_split:
  t -> t+1: L - 1;  chunk k = 16 (a(t-1) .. a(t+15) inside): L - 16  (never below 0).
  Output: visibility_all.csv (full run) or visibility_subset.csv (--episodes).
  --export-keep-ranges PATH (with a full --all): the columns episode, first_visible,
  last_visible as the keep-ranges file for training (data.load_keep_ranges).

Usage:
  python -m scripts.look_tool_projection --episodes 0            # quick test, 1 episode
  python -m scripts.look_tool_projection                         # 3 train episodes
  python -m scripts.look_tool_projection --lag-frames 3 --distortion
  python -m scripts.look_tool_projection --strips                # 8 episodes, lag 0
  python -m scripts.look_tool_projection --summarise outputs/analysis/tool_projection/timing_validation.csv --lag-frames 3
  python -m scripts.look_tool_projection --all --episodes 0 409  # quick test (1 train, 1 val)
  python -m scripts.look_tool_projection --all                   # every train + val episode
  python -m scripts.look_tool_projection --all --export-keep-ranges configs/keep_ranges/star_il_tool_visible.csv
"""
import argparse
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")                     # no display on the DGX
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from models import data

STAR_IL = Path("data/open_h/Surgical/jhu/imerse/star_il/star_il")
OUT = Path("outputs/analysis/tool_projection")
POS_NAMES = ("kuka_pose.x", "kuka_pose.y", "kuka_pose.z")   # found by name, not by index
N_PART_A = 10            # episodes for the ranges
N_PART_B = 3             # default episodes for the dot test (start, middle, end of train)
END_SECONDS = 3          # the last 3 s get 4 of the 8 panels
STRIP_OFFSETS = range(-16, 15, 2)   # 16 thumbnails, offset 0 = the projected transition
STRIP_EXTRA = (317, 136)            # STAR-IL: lowest visible fractions in Part A (skipped if not in train)
STRIP_RANDOM = 3                    # plus this many random train episodes ...
STRIP_SEED = 0                      # ... with a fixed seed
CHUNK_K = 16                        # --all: chunk length of E8 (train.py --chunk 16)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=STAR_IL)
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="episodes for Part B (default: start, middle, end of train)")
    p.add_argument("--lag-frames", type=int, default=0,
                   help="pair frame t with the pose at t - lag (README: ~3 frames)")
    p.add_argument("--distortion", action="store_true",
                   help="apply plumb_bob to the drawn dot (visibility stays undistorted)")
    p.add_argument("--strips", action="store_true",
                   help="timing check: one strip per IN/OUT transition + template CSV")
    p.add_argument("--summarise", type=Path, default=None,
                   help="filled timing CSV -> error statistics (also for --lag-frames)")
    p.add_argument("--all", action="store_true",
                   help="visibility of every train + val episode (tables only, lag 0)")
    p.add_argument("--export-keep-ranges", type=Path, default=None,
                   help="with a full --all: write episode, first_visible, last_visible here")
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


# --- A: calibration (read from the dataset folder) ---
def quat_to_matrix(q):
    """Rotation matrix from a unit quaternion in ROS order (x, y, z, w)."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])

def load_intrinsics(root):
    """K (3x3), D (k1 k2 p1 p2 k3) and the calibration size (width, height)."""
    info = yaml.safe_load((root / "camera_info.yaml").read_text())["camera_info"]
    assert info["distortion_model"] == "plumb_bob", info["distortion_model"]
    K = np.array(info["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(info["D"], dtype=np.float64)
    return K, D, (info["width"], info["height"])

def load_extrinsics(root):
    """R, t of the camera in the world: p_world = R p_cam + t."""
    doc = yaml.safe_load((root / "transformation_side_camera_wrt_robot_base.yaml").read_text())
    tf = doc["transform_data"]["transform"]
    t = np.array([tf["translation"][k] for k in "xyz"], dtype=np.float64)
    R = quat_to_matrix([tf["rotation"][k] for k in "xyzw"])
    assert abs(np.linalg.det(R) - 1) < 1e-6, "not a rotation"
    print(f"transform: {doc['parent_frame']} -> {doc['child_frame']}, t = {t.round(4)} m")
    return R, t

def video_size(root, camera=data.CAMERA):
    """(width, height) of the video, from info.json."""
    v = data.load_info(root)["features"][camera]["info"]
    return v["video.width"], v["video.height"]


# --- B: geometry (pure numpy, no I/O) ---
def world_to_camera(p_world, R, t):
    """(N, 3) world points -> (N, 3) camera points: p_cam = R^T (p_world - t).
    Row-vector form: (p - t) @ R  equals  (R^T (p - t)^T)^T."""
    return (p_world - t) @ R

def project(p_cam, K, D=None):
    """Pinhole (+ optional plumb_bob) at the calibration size -> (u, v).
    Points behind the camera (Z <= 0) get NaN."""
    X, Y, Z = p_cam.T
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = np.where(Z > 0, X / Z, np.nan), np.where(Z > 0, Y / Z, np.nan)
    if D is not None:                                   # plumb_bob: radial + tangential
        k1, k2, p1, p2, k3 = D
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x, y = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    return K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]

def to_video(u, v, calib_size, vid_size):
    """Rescale from the calibration size to the video size, separately per axis."""
    return u * vid_size[0] / calib_size[0], v * vid_size[1] / calib_size[1]


# --- C: lag (pair frame t with the pose at t - lag) ---
def shift_poses(pos, lag):
    """(T, 3) poses -> (T, 3) poses aligned to the frames, and the source index per frame.
    Frames without a source pose (t - lag outside the episode) get NaN / -1."""
    T = len(pos)
    src = np.arange(T) - lag
    ok = (src >= 0) & (src < T)
    out = np.full_like(pos, np.nan, dtype=np.float64)
    out[ok] = pos[src[ok]]
    return out, np.where(ok, src, -1)


# --- D: per-episode projection (table only) ---
def position_columns(root):
    _, action_names = data.feature_names(root)
    return [action_names.index(n) for n in POS_NAMES]

def project_episode(ep, root, calib, lag):
    """Every frame of one episode -> DataFrame with world, camera and pixel coordinates."""
    K, D, calib_size, vid_size, R, t = calib
    _, actions = data.load_table(ep, root)              # raw metres, no canonicalisation
    pos, src = shift_poses(actions[:, position_columns(root)].astype(np.float64), lag)
    p_cam = world_to_camera(pos, R, t)
    u, v = to_video(*project(p_cam, K), calib_size, vid_size)
    ud, vd = to_video(*project(p_cam, K, D), calib_size, vid_size)
    evaluated = src >= 0
    inside = (p_cam[:, 2] > 0) & (u >= 0) & (u < vid_size[0]) & (v >= 0) & (v < vid_size[1])
    visible = pd.array(np.where(evaluated, inside, None), dtype="boolean")  # NA = not evaluated
    return pd.DataFrame({
        "episode": ep, "frame": np.arange(len(pos)), "pose_frame": src,
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "X": p_cam[:, 0], "Y": p_cam[:, 1], "Z": p_cam[:, 2],
        "u": u, "v": v, "u_dist": ud, "v_dist": vd, "visible": visible})


# --- E: Part A, ranges ---
def spread(episodes, k):
    """k episodes evenly spread over a list (first and last included)."""
    idx = np.linspace(0, len(episodes) - 1, min(k, len(episodes))).round().astype(int)
    return [episodes[i] for i in idx]

def part_a_episodes(train, part_b):
    """N_PART_A train episodes spread over the split; each Part B episode replaces the
    closest one, so the count stays N_PART_A and the spread is kept."""
    eps = spread(train, N_PART_A)
    for b in part_b:
        if b in train and b not in eps:
            eps[int(np.argmin(np.abs(np.array(eps) - b)))] = b
    return sorted(set(eps))

def part_a(episodes, root, calib, lag):
    rows = []
    for ep in episodes:
        df = project_episode(ep, root, calib, lag)
        ev = df[df["pose_frame"] >= 0]
        row = {"episode": ep, "frames": len(df)}
        for c in ("x", "y", "z", "X", "Y", "Z"):
            row[f"{c}_min"], row[f"{c}_med"], row[f"{c}_max"] = ev[c].min(), ev[c].median(), ev[c].max()
        row["front"] = float((ev["Z"] > 0).mean())
        row["visible"] = float(ev["visible"].astype(bool).mean())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("episode")
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", "{:.3f}".format):
        print("\nPart A: world position (m), min / median / max")
        print(table[[f"{c}_{s}" for c in "xyz" for s in ("min", "med", "max")]])
        print("\nPart A: camera coordinates (m), min / median / max; Z = depth")
        print(table[[f"{c}_{s}" for c in "XYZ" for s in ("min", "med", "max")]])
        print("\nPart A: fraction in front of the camera (Z > 0) and projected inside the image")
        print(table[["frames", "front", "visible"]])


# --- F: Part B, the dot test ---
def pick_frames(T, fps):
    """8 frames: start (0, 0.5 s), two in the middle, and four in the last 3 s."""
    end = [T - END_SECONDS * fps, T - 2 * fps, T - fps, T - 1]
    mid = list(np.linspace(fps // 2, end[0], 4).round().astype(int)[1:3])
    return sorted({int(f) for f in [0, fps // 2] + mid + end if 0 <= f < T})

def load_native_frames(ep, root, wanted):
    """Decode at native resolution, keep only the wanted indices (AV1: sequential decode).
    Returns the frames and the total number of frames in the video."""
    keep, n = {}, 0
    with av.open(str(data.video_path(ep, root=root))) as container:
        for n, f in enumerate(container.decode(video=0)):
            if n in wanted:
                keep[n] = f.to_ndarray(format="rgb24")
    return keep, n + 1

def label(row):
    """Text for one panel: where the point is and its depth Z."""
    if row["pose_frame"] < 0:
        return "no pose (lag)"
    if not row["Z"] > 0:
        return f"BEHIND  Z={row['Z']:.3f} m"
    return f"OUT  u={row['u']:.0f} v={row['v']:.0f}  Z={row['Z']:.3f} m"

def is_in(row):
    """True only for evaluated frames projected inside the image (NA = not evaluated)."""
    return row["visible"] is not pd.NA and bool(row["visible"])

def draw_panel(ax, img, row, vid_size, distortion):
    """One frame with its projection: dot + depth if IN, a corner label otherwise.
    Shared by the Part B grid and the strips."""
    ax.imshow(img)
    if is_in(row):
        u, v = (row["u_dist"], row["v_dist"]) if distortion else (row["u"], row["v"])
        ax.scatter([u], [v], s=60, c="red", edgecolors="white", linewidths=1.5)
        ax.text(u + 12, v, f"Z={row['Z']:.3f} m", color="yellow", fontsize=8,
                va="center", bbox=dict(facecolor="black", alpha=0.5, pad=1))
    else:
        ax.text(8, 24, label(row), color="red", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, pad=2))
    ax.set_xlim(0, vid_size[0])
    ax.set_ylim(vid_size[1], 0)

def draw_episode(ep, df, frames, fps, vid_size, distortion, path):
    picked = sorted(frames)
    cols = 4
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows), squeeze=False)
    T = len(df)
    for ax, f in zip(axes.flat, picked):
        draw_panel(ax, frames[f], df.iloc[f], vid_size, distortion)
        ax.set_title(f"frame {f}  ({(f - T + 1) / fps:+.1f} s from end)", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(picked):]:
        ax.axis("off")
    fig.suptitle(f"episode {ep}  (lag {df.attrs['lag']} frames, "
                 f"{'with' if distortion else 'no'} distortion)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

def run_suffix(args, distortion=True):
    """File suffix so variants don't overwrite each other: _lag<k>, _dist."""
    return ((f"_lag{args.lag_frames}" if args.lag_frames else "")
            + ("_dist" if distortion and args.distortion else ""))

def part_b(ep, root, calib, args, fps):
    df = project_episode(ep, root, calib, args.lag_frames)
    df.attrs["lag"] = args.lag_frames
    wanted = pick_frames(len(df), fps)
    frames, n_video = load_native_frames(ep, root, set(wanted))
    if n_video != len(df):
        print(f"  WARNING episode {ep}: {n_video} video frames vs {len(df)} table rows")
    h, w = next(iter(frames.values())).shape[:2]
    assert (w, h) == calib[3], f"decoded {w}x{h}, info.json says {calib[3]}"
    suffix = run_suffix(args)
    png, csv = args.out / f"episode_{ep}{suffix}.png", args.out / f"episode_{ep}{suffix}.csv"
    draw_episode(ep, df, frames, fps, calib[3], args.distortion, png)
    df[["episode", "frame", "pose_frame", "u", "v", "u_dist", "v_dist", "Z", "visible"]].to_csv(
        csv, index=False, float_format="%.4f")
    ev = df[df["pose_frame"] >= 0]
    print(f"  episode {ep}: {len(df)} frames, visible {ev['visible'].astype(bool).mean():.1%}, "
          f"panels {wanted}\n    -> {png}\n    -> {csv}")


# --- H: transitions (where `visible` changes) ---
def find_transitions(df):
    """List of (label, direction, projected_frame), with projected_frame the FIRST frame of
    the new state. Only evaluated frames count (the first `lag` frames are skipped)."""
    ev = df[df["pose_frame"] >= 0]
    vis = ev["visible"].astype(bool).to_numpy()
    frames = ev["frame"].to_numpy()
    idx = np.nonzero(vis[1:] != vis[:-1])[0] + 1          # first index of each new state
    out, k = [], 0
    for j, i in enumerate(idx):
        direction = "in" if vis[i] else "out"
        if j == 0 and not vis[0] and direction == "in":
            name = "entry"
        elif j == len(idx) - 1 and not vis[-1] and direction == "out":
            name = "exit"
        else:
            k += 1
            name = f"middle_{k}"
        out.append((name, direction, int(frames[i])))
    return out


# --- I: strips (16 thumbnails around each transition) ---
def draw_strip(ep, name, direction, f0, df, frames, vid_size, args, path):
    fig, axes = plt.subplots(4, 4, figsize=(19.2, 11.6))   # ~480x270 px per thumbnail
    for ax, off in zip(axes.flat, STRIP_OFFSETS):
        f = f0 + off
        ax.set_xticks([]); ax.set_yticks([])
        if f not in frames:                                  # before / after the episode
            ax.axis("off")
            ax.set_title(f"frame {f}: outside episode", fontsize=9)
            continue
        row = df.iloc[f]
        draw_panel(ax, frames[f], row, vid_size, args.distortion)
        status = "IN" if is_in(row) else ("n/a" if row["pose_frame"] < 0 else "OUT")
        ax.set_title(f"frame {f}  ({off:+d})  {status}", fontsize=10,
                     fontweight="bold" if off == 0 else "normal")
        for s in ax.spines.values():                         # frame the transition panel
            s.set_visible(off == 0)
            s.set_color("red"); s.set_linewidth(4)
    fig.suptitle(f"episode {ep}  {name} ({direction})  projected frame {f0}  "
                 f"(lag {args.lag_frames})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)

def strip_episodes(train):
    """spread(train, 3) + STRIP_EXTRA + STRIP_RANDOM random train episodes (fixed seed)."""
    base = spread(train, N_PART_B) + [e for e in STRIP_EXTRA if e in train]
    base = list(dict.fromkeys(base))                         # unique, order kept
    rest = [e for e in train if e not in base]
    rng = np.random.default_rng(STRIP_SEED)
    extra = sorted(int(e) for e in rng.choice(rest, size=min(STRIP_RANDOM, len(rest)), replace=False))
    print(f"strip episodes: {base} + random (seed {STRIP_SEED}) {extra}")
    return base + extra


# --- J: template CSV (append-only) ---
TIMING_COLUMNS = ["episode", "transition", "direction", "projected_frame",
                  "real_frame", "error_frames", "notes"]

def append_template(rows, path):
    """Add only (episode, transition) keys that are not in the file yet: your filled-in
    rows are never touched. Warn if a stored projected_frame no longer matches."""
    new = pd.DataFrame(rows, columns=TIMING_COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        stored = {(r.episode, r.transition): r.projected_frame for r in old.itertuples()}
        for r in new.itertuples():
            key = (r.episode, r.transition)
            if key in stored and stored[key] != r.projected_frame:
                print(f"  WARNING {key}: stored projected_frame {stored[key]}, now {r.projected_frame}")
        new = new[[(r.episode, r.transition) not in stored for r in new.itertuples()]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return path

def run_strips(episodes, root, calib, args):
    strip_dir = args.out / "strips"
    strip_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in episodes:
        df = project_episode(ep, root, calib, args.lag_frames)
        trans = find_transitions(df)
        n_mid = sum(name.startswith("middle") for name, _, _ in trans)
        print(f"  episode {ep}: {len(trans)} transitions ({n_mid} middle / flicker)")
        wanted = {t + o for _, _, t in trans for o in STRIP_OFFSETS if 0 <= t + o < len(df)}
        frames, _ = load_native_frames(ep, root, wanted) if wanted else ({}, 0)
        for name, direction, f0 in trans:
            png = strip_dir / f"episode_{ep}_{name}{run_suffix(args)}.png"
            draw_strip(ep, name, direction, f0, df, frames, calib[3], args, png)
            rows.append([ep, name, direction, f0, None, None, None])
            print(f"    {name:9s} {direction:3s} frame {f0:4d} -> {png}")
    csv = append_template(rows, args.out / f"timing_validation{run_suffix(args, distortion=False)}.csv")
    print(f"template CSV -> {csv}")


# --- K: summary of the hand-annotated CSV ---
def summarise_timing(csv_path, lag=0):
    """error = (projected + lag) - real, in frames (> 0: projection late). Writes the lag-0
    error to error_frames (the only column it changes) and prints statistics per direction.
    The CSV is the lag-0 one; lag k only shifts every projected frame by k."""
    df = pd.read_csv(csv_path)
    done = df["real_frame"].notna()
    df.loc[done, "error_frames"] = df.loc[done, "projected_frame"] - df.loc[done, "real_frame"]
    df.to_csv(csv_path, index=False)
    print(f"{csv_path}: {done.sum()} of {len(df)} transitions annotated")
    for shift in sorted({0, lag}):
        print(f"\nlag {shift} frames: error = projected + {shift} - real")
        for direction in ("in", "out"):
            e = df.loc[done & (df["direction"] == direction), "error_frames"].astype(float) + shift
            if len(e) == 0:
                print(f"  {direction:3s}: no annotated rows")
                continue
            print(f"  {direction:3s}: n={len(e):2d}  mean {e.mean():+.1f}  median {e.median():+.1f}  "
                  f"std {e.std(ddof=1) if len(e) > 1 else float('nan'):.1f}  max|e| {e.abs().max():.0f}")


# --- L: per-episode visibility (tables only) ---
def episode_visibility(ep, split, root, calib):
    """One row per episode: how many frames are invisible at the start, end and middle."""
    vis = project_episode(ep, root, calib, lag=0)["visible"].astype(bool).to_numpy()
    T, seen = len(vis), np.nonzero(vis)[0]
    row = {"episode": ep, "split": split, "n_frames": T, "n_invisible": int((~vis).sum())}
    if len(seen) == 0:                                   # fully invisible: "whole" bucket
        row.update(lead_invisible=0, trail_invisible=0, middle_invisible=0,
                   n_middle_transitions=0, first_visible=None, last_visible=None)
        return row
    s, e = int(seen[0]), int(seen[-1])
    inner = vis[s:e + 1]                                 # starts and ends visible
    row.update(lead_invisible=s, trail_invisible=T - 1 - e,
               middle_invisible=int((~inner).sum()),
               n_middle_transitions=int((inner[1:] != inner[:-1]).sum()),   # 2 per absence
               first_visible=s, last_visible=e)
    return row


# --- M: what trimming keeps (same pair rules as data.load_split) ---
def n_pairs(length, k=None):
    """t -> t+1 pairs: length - 1; chunk k (a(t-1) .. a(t+k-1) inside): length - k."""
    return max(0, length - (1 if k is None else k))

def add_pair_counts(table):
    """Kept length L = last - first + 1 (0 if fully invisible) and the pairs before/after."""
    t = table.copy()
    t["kept_len"] = (t["last_visible"] - t["first_visible"] + 1).fillna(0).astype(int)
    for name, k in (("pairs", None), ("chunk", CHUNK_K)):
        t[f"{name}_now"] = [n_pairs(n, k) for n in t["n_frames"]]
        t[f"{name}_kept"] = [n_pairs(n, k) for n in t["kept_len"]]
    return t


# --- N: printed summary, per split ---
def summarise_visibility(table, fps):
    def pct(x, n): return f"{x:6d} ({x / n:5.1%})"
    for split, g in add_pair_counts(table).groupby("split", sort=False):
        whole = g["first_visible"].isna()
        N = int(g["n_frames"].sum())
        print(f"\n=== {split}: {len(g)} episodes, {N} frames ===")
        print(f"invisible frames  {pct(int(g['n_invisible'].sum()), N)} of all frames")
        print(f"  start           {pct(int(g['lead_invisible'].sum()), N)}")
        print(f"  end             {pct(int(g['trail_invisible'].sum()), N)}")
        print(f"  middle          {pct(int(g['middle_invisible'].sum()), N)}")
        print(f"  whole episode   {pct(int(g.loc[whole, 'n_invisible'].sum()), N)}")

        seen = g[~whole]                                  # episodes with a visible frame
        print(f"lead / trail in seconds (over the {len(seen)} episodes with a visible frame):")
        for col, name in (("lead_invisible", "lead "), ("trail_invisible", "trail")):
            s = seen[col] / fps
            print(f"  {name}: mean {s.mean():.2f}  median {s.median():.2f}  max {s.max():.2f}  "
                  f"(> 0 in {int((s > 0).sum())} episodes)")

        print(f"episodes: fully visible {int((g['n_invisible'] == 0).sum())}, "
              f"with a middle absence {int((g['middle_invisible'] > 0).sum())}, "
              f"fully invisible {int(whole.sum())}"
              + (f" {g.loc[whole, 'episode'].tolist()}" if whole.any() else ""))
        top = g[g["middle_invisible"] > 0].nlargest(5, "middle_invisible")
        if len(top):
            print("most middle-invisible frames:")
            print(top[["episode", "n_frames", "middle_invisible", "n_middle_transitions"]]
                  .to_string(index=False))

        print("trimming start + end (middle kept):")
        for name, label in (("pairs", "t -> t+1"), ("chunk", f"chunk {CHUNK_K}")):
            now, kept = int(g[f"{name}_now"].sum()), int(g[f"{name}_kept"].sum())
            print(f"  {label:9s}: {now} -> {kept} pairs ({kept - now:+d}, {kept / now - 1:+.1%})")
        short = g["kept_len"] < CHUNK_K + 1
        print(f"  episodes keeping < {CHUNK_K + 1} frames (no chunk pair left): {int(short.sum())}"
              f" (already < {CHUNK_K + 1} before trimming: {int((g['n_frames'] < CHUNK_K + 1).sum())})")

def run_all(episodes, root, calib, out, subset, export=None):
    rows = [episode_visibility(ep, split, root, calib) for ep, split in episodes]
    table = pd.DataFrame(rows)
    for c in ("first_visible", "last_visible"):
        table[c] = table[c].astype("Int64")             # integers with empty cells
    csv = out / ("visibility_subset.csv" if subset else "visibility_all.csv")
    table.to_csv(csv, index=False)
    print(f"per-episode table -> {csv}")
    if export is not None:                               # the training keep-ranges file
        export.parent.mkdir(parents=True, exist_ok=True)
        table[["episode", "first_visible", "last_visible"]].to_csv(export, index=False)
        print(f"keep ranges ({len(table)} episodes, train + val) -> {export}")
    summarise_visibility(table, data.load_info(root)["fps"])


# --- G: main ---
def allowed_episodes(root):
    """Train and val only: the test split is never touched."""
    return {s: data.get_split_episodes(s, root) for s in data.ALLOWED_SPLITS}

def main():
    args = parse_args()
    if args.summarise:                                       # no data needed, only the CSV
        summarise_timing(args.summarise, args.lag_frames)
        return
    root = args.dataset_root
    args.out.mkdir(parents=True, exist_ok=True)

    splits = allowed_episodes(root)
    train, _ = data.drop_missing_videos("train", splits["train"], root=root)
    if args.episodes:
        allowed = set(splits["train"]) | set(splits["val"])
        bad = [e for e in args.episodes if e not in allowed]
        if bad:
            raise ValueError(f"episodes {bad} are not in train/val")
        part_b_eps = args.episodes
    else:
        part_b_eps = strip_episodes(train) if args.strips else spread(train, N_PART_B)

    K, D, calib_size = load_intrinsics(root)
    R, t = load_extrinsics(root)
    vid_size = video_size(root)
    fps = data.load_info(root)["fps"]
    calib = (K, D, calib_size, vid_size, R, t)
    print(f"calibration {calib_size[0]}x{calib_size[1]} -> video {vid_size[0]}x{vid_size[1]} "
          f"(scale u {vid_size[0] / calib_size[0]:.4f}, v {vid_size[1] / calib_size[1]:.4f}), "
          f"lag {args.lag_frames} frames, distortion {'on' if args.distortion else 'off'}")

    if args.all:                                             # visibility of train + val
        if args.lag_frames:
            raise ValueError("--all uses lag 0 (the validated setting); drop --lag-frames")
        if args.export_keep_ranges and args.episodes:   # a subset file would miss episodes
            raise ValueError("--export-keep-ranges needs the full --all run (no --episodes)")
        episodes = []
        for split in data.ALLOWED_SPLITS:                    # never the test split
            kept, _ = data.drop_missing_videos(split, splits[split], root=root)
            episodes += [(ep, split) for ep in kept if not args.episodes or ep in args.episodes]
        run_all(episodes, root, calib, args.out, subset=bool(args.episodes),
                export=args.export_keep_ranges)
        return
    if args.export_keep_ranges:
        raise ValueError("--export-keep-ranges works only with --all")
    if args.strips:                                         # timing check only
        print("\nStrips: transitions")
        run_strips(part_b_eps, root, calib, args)
        return
    part_a(part_a_episodes(train, part_b_eps), root, calib, args.lag_frames)
    print("\nPart B: dot test")
    for ep in part_b_eps:
        part_b(ep, root, calib, args, fps)


if __name__ == "__main__":
    main()
