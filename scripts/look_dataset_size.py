from huggingface_hub import HfApi

REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
PREFIX  = "Surgical/jhu/imerse/srt_tissue_lift"

files = HfApi().list_repo_tree(REPO_ID, repo_type="dataset",
                               path_in_repo=PREFIX, recursive=True)

total = {"meta": 0, "data": 0, "video_left": 0, "video_other": 0, "other": 0}
for f in files:
    size = getattr(f, "size", None)
    if size is None:                      # folders have no size
        continue
    if "/meta/" in f.path:
        total["meta"] += size
    elif "/data/" in f.path:
        total["data"] += size
    elif "endoscope.left" in f.path:
        total["video_left"] += size
    elif "/videos/" in f.path:
        total["video_other"] += size
    else:
        total["other"] += size            # e.g. README.md

for k, v in total.items():
    print(f"{k:12s} {v / 1e9:7.2f} GB")