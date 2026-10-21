"""Load Open-H (LeRobot v2.1) episodes and build world-model triplets."""
import json
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path("data/open_h/Surgical/jhu/imerse/srt_tissue_lift")
CAMERA = "observation.images.endoscope.left"   # the right view is a copy
IMG_SIZE = 224                                  # the size the model expects
CACHE_DIR = Path("data/cache")                  # decoded, resized uint8 frames per split
ALLOWED_SPLITS = ("train", "val")               # the test split stays untouched for now


# --- A: metadata ---
def load_info(root=ROOT):
    return json.loads((root / "meta" / "info.json").read_text())

def load_stats(root=ROOT):
    return json.loads((root / "meta" / "stats.json").read_text())

def get_split_episodes(split, root=ROOT):
    """Episode indices of a split from info.json, e.g. "train": "0:180" -> 0..179."""
    start, stop = load_info(root)["splits"][split].split(":")
    return list(range(int(start), int(stop)))

def feature_dims(root=ROOT):
    """(state_dim, action_dim), read from info.json instead of hard-coding them."""
    features = load_info(root)["features"]
    return features["observation.state"]["shape"][0], features["action"]["shape"][0]

def feature_names(root=ROOT):
    """(state_names, action_names) from info.json, e.g. "kuka_pose.x".
    Open-H nests them in one list ([[...]]), other LeRobot datasets in a dict
    ({"motors": [...]}); "dim0", "dim1", ... if a dataset has none."""
    features = load_info(root)["features"]
    def names(key):
        n = features[key].get("names")
        n = next(iter(n.values())) if isinstance(n, dict) else n
        n = n[0] if n and isinstance(n[0], list) else n
        return list(n) if n else [f"dim{i}" for i in range(features[key]["shape"][0])]
    return names("observation.state"), names("action")

def load_episode_lengths(root=ROOT):
    """Number of frames per episode, from meta/episodes.jsonl."""
    lengths = {}
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            lengths[ep["episode_index"]] = ep["length"]
    return lengths


# --- B: the table (state and action) ---
def table_path(episode, root=ROOT):
    info = load_info(root)
    chunk = episode // info["chunks_size"]
    return root / info["data_path"].format(episode_chunk=chunk, episode_index=episode)

def load_table(episode, root=ROOT, quat_ref=None):
    """Raw state and action of one episode. With quat_ref (see D1), every quaternion
    gets the sign of its reference rotation: the one place where labels are canonicalised,
    so normalisation stats and training data always see the same values."""
    df = pd.read_parquet(table_path(episode, root))
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    if quat_ref is not None:
        states = canonicalise_quaternions(states, quat_ref, "observation.state")
        actions = canonicalise_quaternions(actions, quat_ref, "action")
    return states, actions


# --- C: the video (frames) ---
def video_path(episode, camera=CAMERA, root=ROOT):
    info = load_info(root)
    chunk = episode // info["chunks_size"]
    return root / info["video_path"].format(
        episode_chunk=chunk, video_key=camera, episode_index=episode)

def drop_missing_videos(split, episodes, camera=CAMERA, root=ROOT, verbose=True):
    """Keep only the episodes that have a video file for `camera`.
    Returns (kept, skipped); prints how many were dropped from the split."""
    kept = [ep for ep in episodes if video_path(ep, camera, root).exists()]
    skipped = sorted(set(episodes) - set(kept))
    if verbose:
        print(f"{split}: dropped {len(skipped)} of {len(episodes)} episodes "
              f"(no {camera} video){': ' + str(skipped) if skipped else ''}")
    return kept, skipped

def load_frames(episode, camera=CAMERA, size=IMG_SIZE, root=ROOT):
    path = video_path(episode, camera, root)
    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24", width=size, height=size)
                  for f in container.decode(video=0)]
    return np.stack(frames)


# --- D: normalisation ---
def normalise(x, stats, key):
    mean = np.asarray(stats[key]["mean"], dtype=np.float32)
    std = np.asarray(stats[key]["std"], dtype=np.float32)
    return (x - mean) / (std + 1e-8)

def unnormalise(x, stats, key):
    """Exact inverse of normalise(): back to the dataset's own units."""
    mean = np.asarray(stats[key]["mean"], dtype=np.float32)
    std = np.asarray(stats[key]["std"], dtype=np.float32)
    return x * (std + 1e-8) + mean

def compute_norm_stats(episodes, root=ROOT, quat_ref=None):
    """Mean/std of state and action over the given (training) episodes only.
    Same layout as stats.json, so normalise() works with it.
    quat_ref: computed on canonicalised quaternions (must match load_split's quat_ref).
    Raises (never falls back) if any episode's table is missing."""
    missing = [ep for ep in episodes if not table_path(ep, root).exists()]
    if not episodes or missing:
        raise FileNotFoundError(
            f"cannot compute normalisation stats for {root}: {len(missing)} of "
            f"{len(episodes)} episodes have no parquet file (first: {missing[:10]})")
    states, actions = zip(*(load_table(ep, root, quat_ref) for ep in episodes))
    stats = {}
    for key, x in [("observation.state", np.concatenate(states)),
                   ("action", np.concatenate(actions))]:
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), 1e-6)   # guard: a constant dimension would divide by ~0
        stats[key] = {"mean": mean.tolist(), "std": std.tolist()}
    return stats

def save_norm_stats(stats, path):
    Path(path).write_text(json.dumps(stats, indent=2))

def load_norm_stats(path):
    return json.loads(Path(path).read_text())


# --- D1: quaternion sign (optional, train.py --canonical-quaternion) ---
# q and -q are the same rotation, but far apart for an MSE loss. Open-H stores either
# sign (STAR-IL: 37% of train episodes on the "other" side). Canonical = every quaternion
# flipped to the side of one reference rotation (the train Markley mean), at load time;
# the dataset files are never changed.
QUAT_KEYS = ("observation.state", "action")
QUAT_SAFE_DEG = 150.0   # beyond this from the reference, the sign is close to ambiguous again

def quaternion_starts(names):
    """Start index of every quaternion [<p>.x, <p>.y, <p>.z, <p>.w] whose prefix <p> ends
    in an orientation word, e.g. "kuka_pose.ori", "psm1_sp.orientation"."""
    starts = []
    for i in range(len(names) - 3):
        prefix = names[i].rsplit(".", 1)[0]
        if (prefix.rsplit(".", 1)[-1].startswith("ori")
                and names[i:i + 4] == [f"{prefix}.{c}" for c in "xyzw"]):
            starts.append(i)
    return starts

def average_quaternion(qs):
    """Markley mean: eigenvector of sum(q q^T) with the largest eigenvalue.
    q q^T = (-q)(-q)^T, so the result does not depend on the stored signs."""
    _, vecs = np.linalg.eigh(qs.T @ qs)
    return vecs[:, -1]

def compute_quat_reference(episodes, root=ROOT):
    """{key: [{"start", "names", "q", "max_angle_deg"}]} from the given (training) episodes.
    Raises if neither state nor action contains a quaternion (the option would do nothing)."""
    names = dict(zip(QUAT_KEYS, feature_names(root)))
    states, actions = zip(*(load_table(ep, root) for ep in episodes))   # raw signs
    tables = {"observation.state": np.concatenate(states), "action": np.concatenate(actions)}
    ref = {}
    for key in QUAT_KEYS:
        ref[key] = []
        for s in quaternion_starts(names[key]):
            q = tables[key][:, s:s + 4].astype(np.float64)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            r = average_quaternion(q)
            max_deg = float(np.degrees(2 * np.arccos(np.clip(np.abs(q @ r).min(), 0, 1))))
            ref[key].append({"start": s, "names": names[key][s:s + 4], "q": r.tolist(),
                             "max_angle_deg": max_deg})
            print(f"quaternion {key}[{s}:{s + 4}]: reference {np.round(r, 3)}, "
                  f"max {max_deg:.1f} deg from it")
            if max_deg > QUAT_SAFE_DEG:
                print(f"  WARNING: some train orientations are > {QUAT_SAFE_DEG:g} deg from the "
                      f"reference; their canonical sign is close to ambiguous")
    if not any(ref.values()):
        raise ValueError(f"canonical quaternions requested, but no quaternion found in the "
                         f"info.json names of {root}")
    return ref

def canonicalise_quaternions(x, quat_ref, key):
    """Copy of x (raw units, (N, D)) with each quaternion's sign flipped where dot(q, ref) < 0."""
    x = np.array(x, copy=True)
    for entry in quat_ref.get(key, []):
        q = x[:, entry["start"]:entry["start"] + 4]
        q *= np.where(q @ np.asarray(entry["q"], q.dtype) < 0, -1, 1)[:, None].astype(q.dtype)
    return x

def save_quat_reference(ref, path):
    Path(path).write_text(json.dumps(ref, indent=2))

def load_quat_reference(path):
    return json.loads(Path(path).read_text())


# --- E: triplets ---
def load_episode_triplets(episode, root=ROOT, stats=None):
    """One episode as triplets. Stats: the given ones, else meta/stats.json, else
    (single-episode scripts only) computed from this episode, with a warning."""
    if not video_path(episode, CAMERA, root).exists():
        raise FileNotFoundError(f"episode {episode} has no {CAMERA} video in {root}")
    states, actions = load_table(episode, root)
    frames = load_frames(episode, root=root)
    assert len(frames) == len(states), (
        f"episode {episode}: {len(frames)} frames vs {len(states)} rows")

    if stats is None and (root / "meta" / "stats.json").exists():
        stats = load_stats(root)
    elif stats is None:
        print(f"WARNING: {root}/meta/stats.json not found: normalising episode {episode} "
              f"with ITS OWN mean/std. Fine for shape checks and overfitting, not for training.")
        stats = compute_norm_stats([episode], root)
    states = normalise(states, stats, "observation.state")
    actions = normalise(actions, stats, "action")
    images = frames.astype(np.float32) / 255.0

    T = len(states)
    inputs = {
        "image_input":    images[:-1],                         # frame t
        "state_input":    states[:-1],                         # state t
        "language_input": np.zeros((T - 1, 512), np.float32),  # one task: no information
        "action_input":   actions[:-1],                        # action t
    }
    targets = {
        "action_output":     actions[:-1],   # actor head: imitate action t
        "next_state_output": states[1:],     # world model: state t+1
        "next_image_output": images[1:],     # world model: frame t+1
    }
    return inputs, targets

def load_instruction(episode, root=ROOT):
    path = table_path(episode, root)
    return str(pd.read_parquet(path, columns=["instruction.text"])["instruction.text"].iloc[0])


# --- F0: frame cache (decode the videos once per split) ---
def check_split(split):
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split {split!r} is not allowed (allowed: {ALLOWED_SPLITS})")

def _cache_manifest(split, episodes, camera, size, root):
    """Everything the cached frames depend on. If any of it changes, the cache is rebuilt."""
    info, lengths = load_info(root), load_episode_lengths(root)
    return {"dataset": root.as_posix(), "codebase_version": info["codebase_version"],
            "split": split, "camera": camera, "size": size,
            "episodes": [int(ep) for ep in episodes],
            "lengths": [lengths[ep] for ep in episodes],
            "video_bytes": [video_path(ep, camera, root).stat().st_size for ep in episodes]}

def cached_frames(split, episodes, camera=CAMERA, size=IMG_SIZE, root=ROOT, cache_dir=CACHE_DIR):
    """All frames of a split as a read-only memory-mapped uint8 array (N, size, size, 3).
    Built once in cache_dir; rebuilt if missing or if it no longer matches the metadata."""
    check_split(split)
    npy = Path(cache_dir) / root.name / f"{split}_{camera.split('.', 2)[-1]}_{size}.npy"
    meta = npy.with_suffix(".json")
    manifest = _cache_manifest(split, episodes, camera, size, root)
    shape = (sum(manifest["lengths"]), size, size, 3)

    if npy.exists() and meta.exists() and json.loads(meta.read_text()) == manifest:
        frames = np.load(npy, mmap_mode="r")
        if frames.shape == shape and frames.dtype == np.uint8:
            return frames
    print(f"building frame cache {npy} ({shape[0]} frames, {np.prod(shape) / 1e9:.2f} GB)")

    # write into a temporary file episode by episode, rename only when complete:
    # an interrupted build can never leave a cache that looks valid
    npy.parent.mkdir(parents=True, exist_ok=True)
    meta.unlink(missing_ok=True)
    tmp = npy.with_name(npy.stem + ".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=shape)
    start = 0
    for i, (ep, T) in enumerate(zip(manifest["episodes"], manifest["lengths"])):
        ep_frames = load_frames(ep, camera=camera, size=size, root=root)
        assert len(ep_frames) == T, f"episode {ep}: {len(ep_frames)} frames vs {T} in episodes.jsonl"
        out[start:start + T] = ep_frames
        start += T
        if (i + 1) % 20 == 0 or i + 1 == len(episodes):
            print(f"  cached {i + 1}/{len(episodes)} episodes")
    out.flush()
    del out
    os.replace(tmp, npy)
    meta.write_text(json.dumps(manifest, indent=1))
    return np.load(npy, mmap_mode="r")


# --- F: many episodes (a whole split) ---
def load_split(split, root=ROOT, camera=CAMERA, size=IMG_SIZE, stats=None, episodes=None,
               use_cache=True, cache_dir=CACHE_DIR, quat_ref=None, with_frames=True):
    """Load every episode of a split ("train" or "val") into flat arrays.

    Each frame is stored once (uint8). `pair_index` lists the positions t whose
    t+1 is in the same episode, so a pair never crosses an episode boundary.
    If `stats` is None, they are computed on the training split.
    `episodes` optionally restricts the split to a subset (e.g. for quick checks).
    Episodes without a video for `camera` are skipped (listed in "skipped_episodes").
    `quat_ref` (D1): canonical quaternion signs; `stats` must have been computed with it.
    Full splits read their frames from the memory-mapped cache (built on first use);
    subsets decode their videos directly and never touch the cache.
    `with_frames=False` (state-only models): tables only, "frames" is None. Episodes are
    still filtered by video, so the pairs are exactly those of an image model.
    """
    check_split(split)
    from_cache = with_frames and use_cache and episodes is None
    if episodes is None:
        episodes = get_split_episodes(split, root)
    episodes, skipped = drop_missing_videos(split, episodes, camera, root)
    if stats is None:
        train_eps, _ = drop_missing_videos("train", get_split_episodes("train", root),
                                           camera, root, verbose=False)
        stats = compute_norm_stats(train_eps, root, quat_ref)

    # allocate once, so all frames never exist twice in memory
    lengths = load_episode_lengths(root)
    N = sum(lengths[ep] for ep in episodes)
    if not with_frames:
        frames = None
    elif from_cache:
        frames = cached_frames(split, episodes, camera, size, root, cache_dir)
    else:
        frames = np.empty((N, size, size, 3), np.uint8)
    state_dim, action_dim = feature_dims(root)
    states = np.empty((N, state_dim), np.float32)
    actions = np.empty((N, action_dim), np.float32)
    episode_index = np.empty(N, np.int64)
    pair_index = []

    start = 0
    for ep in episodes:
        ep_states, ep_actions = load_table(ep, root, quat_ref)
        T = len(ep_states)
        assert T == lengths[ep], f"episode {ep}: {T} rows vs {lengths[ep]} in episodes.jsonl"
        if with_frames and not from_cache:
            ep_frames = load_frames(ep, camera=camera, size=size, root=root)
            assert len(ep_frames) == T, f"episode {ep}: {len(ep_frames)} frames vs {T} rows"
            frames[start:start + T] = ep_frames
        states[start:start + T] = normalise(ep_states, stats, "observation.state")
        actions[start:start + T] = normalise(ep_actions, stats, "action")
        episode_index[start:start + T] = ep
        pair_index.append(np.arange(start, start + T - 1))   # last frame has no t+1
        start += T

    return {
        "frames": frames,                 # (N, size, size, 3) uint8, or None (with_frames=False)
        "states": states,                 # (N, state_dim) normalised
        "actions": actions,               # (N, action_dim) normalised
        "episode_index": episode_index,   # (N,) which episode each frame is from
        "pair_index": np.concatenate(pair_index),  # (P,) valid t, with t+1 = t + 1
        "skipped_episodes": skipped,      # episodes dropped: no video for `camera`
        "stats": stats,
    }


def get_pairs(data, idx):
    """Turn pair positions t (from data["pair_index"]) into the model's inputs/targets.
    Images become float in [0, 1] only here, for this batch."""
    t = np.asarray(idx)
    images_t = data["frames"][t].astype(np.float32) / 255.0
    images_t1 = data["frames"][t + 1].astype(np.float32) / 255.0
    inputs = {
        "image_input":    images_t,                                  # frame t
        "state_input":    data["states"][t],                         # state t
        "language_input": np.zeros((len(t), 512), np.float32),       # one task: no information
        "action_input":   data["actions"][t],                        # action t
    }
    targets = {
        "action_output":     data["actions"][t],       # actor head: imitate action t
        "next_state_output": data["states"][t + 1],    # world model: state t+1
        "next_image_output": images_t1,                # world model: frame t+1
    }
    return inputs, targets