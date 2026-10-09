from huggingface_hub import snapshot_download

ROOT = "Surgical/jhu/imerse/srt_tissue_lift"

snapshot_download(
    repo_id="nvidia/PhysicalAI-Robotics-Open-H-Embodiment",
    repo_type="dataset",
    local_dir="data/open_h",
    allow_patterns=[
        f"{ROOT}/README.md",
        f"{ROOT}/meta/*",
        f"{ROOT}/data/chunk-000/episode_000000.parquet",
        f"{ROOT}/videos/chunk-000/*/episode_000000.mp4",
    ],
)