"""Download STAR-IL from Open-H: one episode (laptop) or the full sub-dataset (DGX)."""
import argparse
from huggingface_hub import snapshot_download

REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
ROOT = "Surgical/jhu/imerse/star_il/star_il"      # note: the folder name appears twice
CAM = "observation.images.endoscope.left"
WRIST = "observation.images.wrist.left"

def build_patterns(full: bool):
    # Metadata, README and calibration files: always small, always useful
    patterns = [f"{ROOT}/meta/*", f"{ROOT}/README*.md", f"{ROOT}/*.yaml"]
    if full:
        # All episodes, left endoscope only (as we did for tissue lift)
        patterns += [f"{ROOT}/data/chunk-*/*.parquet",
                     f"{ROOT}/videos/chunk-*/{CAM}/*.mp4"]
    else:
        # Episode 0 only, plus its wrist video, just to look at it
        ep = "episode_000000"
        patterns += [f"{ROOT}/data/chunk-000/{ep}.parquet",
                     f"{ROOT}/videos/chunk-000/{CAM}/{ep}.mp4",
                     f"{ROOT}/videos/chunk-000/{WRIST}/{ep}.mp4"]
    return patterns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="download all 512 episodes")
    parser.add_argument("--local-dir", default="data/open-h")
    args = parser.parse_args()

    snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                      allow_patterns=build_patterns(args.full),
                      local_dir=args.local_dir)