"""Load one Open-H (LeRobot v2.1) episode and build world-model triplets."""
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