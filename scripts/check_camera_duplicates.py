"""
check_camera_duplicates.py

Question: for EVERY episode of one Open-H dataset, do the left and right
endoscope videos contain the same images?

Three levels of evidence, from cheapest to most expensive:
  1. Statistics   - per-episode image stats in meta/episodes_stats.jsonl (one small file)
  2. Fingerprints - sha256 of each MP4, read from the Hub listing (no video download)
  3. Pixels       - decode both videos and compare frame by frame
                    (only with --pixels, and only for episodes levels 1-2 cannot settle)

Usage (on the DGX or the laptop):
  python check_camera_duplicates.py --subdir "Surgical/<institution>/<dataset>"
  python check_camera_duplicates.py --subdir "..." --pixels --max-pixel-episodes 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"


# --------------------------------------------------------------------------
# Section 1 - find the camera keys from meta/info.json
# --------------------------------------------------------------------------
def get_meta_file(subdir, name):
    """Download (or reuse from cache) one small file from the meta/ folder."""
    return hf_hub_download(REPO, f"{subdir}/meta/{name}", repo_type="dataset")


def find_camera_pair(subdir):
    info = json.load(open(get_meta_file(subdir, "info.json")))
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    print("Video features in this dataset:", *video_keys, sep="\n  ")

    left = [k for k in video_keys if "left" in k]
    if len(left) != 1:
        raise SystemExit(f"Expected exactly one 'left' camera, found {left}. Pass --left/--right.")
    right = left[0].replace("left", "right")
    if right not in video_keys:
        raise SystemExit(f"No matching right camera for {left[0]}. Pass --left/--right.")
    return left[0], right


# --------------------------------------------------------------------------
# Section 2 - Level 1: compare per-episode image statistics
# --------------------------------------------------------------------------
def compare_stats(subdir, left, right):
    """Return {episode_index: (stats_identical, max_abs_difference)}."""
    result = {}
    with open(get_meta_file(subdir, "episodes_stats.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            sl, sr = row["stats"][left], row["stats"][right]
            diffs = [
                np.abs(np.asarray(sl[s], dtype=float) - np.asarray(sr[s], dtype=float)).max()
                for s in ("min", "max", "mean", "std")
            ]
            result[row["episode_index"]] = (max(diffs) == 0.0, float(max(diffs)))
    return result


# --------------------------------------------------------------------------
# Section 3 - Level 2: compare file fingerprints (sha256) without downloading
# --------------------------------------------------------------------------
def episode_from_path(path):
    # ".../episode_000042.mp4" -> 42
    return int(Path(path).stem.split("_")[-1])


def compare_hashes(subdir, left, right):
    """Return {episode_index: (hash_identical, left_path, right_path)}."""
    api = HfApi()
    hashes = {left: {}, right: {}}
    paths = {left: {}, right: {}}
    for entry in api.list_repo_tree(REPO, path_in_repo=f"{subdir}/videos",
                                    repo_type="dataset", recursive=True):
        lfs = getattr(entry, "lfs", None)          # folders have no .lfs
        if lfs is None or not entry.path.endswith(".mp4"):
            continue
        for cam in (left, right):
            if f"/{cam}/" in entry.path:
                ep = episode_from_path(entry.path)
                hashes[cam][ep] = lfs.sha256
                paths[cam][ep] = entry.path

    result = {}
    for ep in sorted(set(hashes[left]) | set(hashes[right])):
        same = ep in hashes[left] and hashes[left].get(ep) == hashes[right].get(ep)
        result[ep] = (same, paths[left].get(ep), paths[right].get(ep))
    return result


# --------------------------------------------------------------------------
# Section 4 - Level 3: decode both videos and compare pixels
# --------------------------------------------------------------------------
def read_frames(local_path):
    import av  # PyAV, already used in models/data.py
    with av.open(local_path) as container:
        return [fr.to_ndarray(format="rgb24") for fr in container.decode(video=0)]


def compare_pixels(left_path, right_path):
    fl = read_frames(hf_hub_download(REPO, left_path, repo_type="dataset"))
    fr = read_frames(hf_hub_download(REPO, right_path, repo_type="dataset"))
    if len(fl) != len(fr):
        return {"same_length": False, "n_left": len(fl), "n_right": len(fr)}
    diffs = [np.abs(a.astype(np.int16) - b.astype(np.int16)).max() for a, b in zip(fl, fr)]
    return {"same_length": True, "n_frames": len(fl),
            "identical_frames": int(sum(d == 0 for d in diffs)),
            "max_pixel_diff": int(max(diffs))}


# --------------------------------------------------------------------------
# Section 5 - put it together and print a verdict per episode
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", required=True, help='e.g. "Surgical/<institution>/<dataset>"')
    ap.add_argument("--left"), ap.add_argument("--right")
    ap.add_argument("--pixels", action="store_true", help="run level 3 on unsettled episodes")
    ap.add_argument("--max-pixel-episodes", type=int, default=5)
    args = ap.parse_args()

    left, right = (args.left, args.right) if args.left else find_camera_pair(args.subdir)
    print(f"\nComparing\n  L = {left}\n  R = {right}\n")

    stats = compare_stats(args.subdir, left, right)
    hashes = compare_hashes(args.subdir, left, right)

    n = len(hashes)
    n_hash = sum(h[0] for h in hashes.values())
    n_stats = sum(s[0] for s in stats.values())
    print(f"Episodes with videos:            {n}")
    print(f"Level 1 - identical stats:       {n_stats}/{len(stats)}")
    print(f"Level 2 - identical MP4 (sha256):{n_hash:>4}/{n}")

    unsettled = [ep for ep, (same, _, _) in hashes.items() if not same]
    if unsettled:
        print("\nEpisodes whose files differ:")
        for ep in unsettled:
            s_same, s_diff = stats.get(ep, (None, None))
            print(f"  ep {ep:4d}  stats identical={s_same}  max stats diff={s_diff}")

    if args.pixels and unsettled:
        print(f"\nLevel 3 - pixel check on up to {args.max_pixel_episodes} episodes:")
        for ep in unsettled[: args.max_pixel_episodes]:
            _, lp, rp = hashes[ep]
            if lp and rp:
                print(f"  ep {ep:4d}", compare_pixels(lp, rp))


if __name__ == "__main__":
    main()