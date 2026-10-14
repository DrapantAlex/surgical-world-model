"""Load Open-H (LeRobot v2.1) episodes and build world-model triplets."""
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path("data/open_h/Surgical/jhu/imerse/srt_tissue_lift")
CAMERA = "observation.images.endoscope.left"   # the right view is a copy
IMG_SIZE = 224                                  # the size the model expects


# --- A: metadata ---
def load_info(root=ROOT):
    return json.loads((root / "meta" / "info.json").read_text())

def load_stats(root=ROOT):
    return json.loads((root / "meta" / "stats.json").read_text())

def get_split_episodes(split, root=ROOT):
    """Episode indices of a split from info.json, e.g. "train": "0:180" -> 0..179."""
    start, stop = load_info(root)["splits"][split].split(":")
    return list(range(int(start), int(stop)))

def load_episode_lengths(root=ROOT):
    """Number of frames per episode, from meta/episodes.jsonl."""
    lengths = {}
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            lengths[ep["episode_index"]] = ep["length"]
    return lengths


# --- B: the table (state and action) ---
def load_table(episode, root=ROOT):
    info = load_info(root)
    chunk = episode // info["chunks_size"]
    path = root / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    df = pd.read_parquet(path)
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    return states, actions


# --- C: the video (frames) ---
def load_frames(episode, camera=CAMERA, size=IMG_SIZE, root=ROOT):
    info = load_info(root)
    chunk = episode // info["chunks_size"]
    path = root / info["video_path"].format(
        episode_chunk=chunk, video_key=camera, episode_index=episode)
    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24", width=size, height=size)
                  for f in container.decode(video=0)]
    return np.stack(frames)


# --- D: normalisation ---
def normalise(x, stats, key):
    mean = np.asarray(stats[key]["mean"], dtype=np.float32)
    std = np.asarray(stats[key]["std"], dtype=np.float32)
    return (x - mean) / (std + 1e-8)

def compute_norm_stats(episodes, root=ROOT):
    """Mean/std of state and action over the given (training) episodes only.
    Same layout as stats.json, so normalise() works with it."""
    states, actions = zip(*(load_table(ep, root) for ep in episodes))
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


# --- E: triplets ---
def load_episode_triplets(episode, root=ROOT):
    states, actions = load_table(episode, root)
    frames = load_frames(episode, root=root)
    assert len(frames) == len(states), (
        f"episode {episode}: {len(frames)} frames vs {len(states)} rows")

    stats = load_stats(root)
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
    info = load_info(root)
    chunk = episode // info["chunks_size"]
    path = root / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    return str(pd.read_parquet(path, columns=["instruction.text"])["instruction.text"].iloc[0])


# --- F: many episodes (a whole split) ---
def load_split(split, root=ROOT, camera=CAMERA, size=IMG_SIZE, stats=None, episodes=None):
    """Load every episode of a split ("train", "val", "test", ...) into flat arrays.

    Each frame is stored once (uint8). `pair_index` lists the positions t whose
    t+1 is in the same episode, so a pair never crosses an episode boundary.
    If `stats` is None, they are computed on the training split.
    `episodes` optionally restricts the split to a subset (e.g. for quick checks).
    """
    if episodes is None:
        episodes = get_split_episodes(split, root)
    if stats is None:
        stats = compute_norm_stats(get_split_episodes("train", root), root)

    # allocate once, so all frames never exist twice in memory
    lengths = load_episode_lengths(root)
    N = sum(lengths[ep] for ep in episodes)
    frames = np.empty((N, size, size, 3), np.uint8)
    states = np.empty((N, 16), np.float32)
    actions = np.empty((N, 16), np.float32)
    episode_index = np.empty(N, np.int64)
    pair_index = []

    start = 0
    for ep in episodes:
        ep_states, ep_actions = load_table(ep, root)
        ep_frames = load_frames(ep, camera=camera, size=size, root=root)
        T = len(ep_states)
        assert len(ep_frames) == T == lengths[ep], (
            f"episode {ep}: {len(ep_frames)} frames vs {T} rows vs {lengths[ep]} in episodes.jsonl")

        frames[start:start + T] = ep_frames
        states[start:start + T] = normalise(ep_states, stats, "observation.state")
        actions[start:start + T] = normalise(ep_actions, stats, "action")
        episode_index[start:start + T] = ep
        pair_index.append(np.arange(start, start + T - 1))   # last frame has no t+1
        start += T

    return {
        "frames": frames,                 # (N, size, size, 3) uint8
        "states": states,                 # (N, 16) normalised
        "actions": actions,               # (N, 16) normalised
        "episode_index": episode_index,   # (N,) which episode each frame is from
        "pair_index": np.concatenate(pair_index),  # (P,) valid t, with t+1 = t + 1
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