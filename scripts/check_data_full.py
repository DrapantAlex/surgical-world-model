# scripts/check_data_full.py
import numpy as np
from models import data

for split in ["train", "val"]:
    eps = list(data.get_split_episodes(split, data.ROOT))
    print(split, "episodes:", len(eps), eps[0], "…", eps[-1])

train = data.load_split("train")                              # stats computed here
val   = data.load_split("val", stats=train["stats"])

for name, d in [("train", train), ("val", val)]:
    n_frames = len(d["frames"])
    n_eps    = len(np.unique(d["episode_index"]))
    n_pairs  = len(d["pair_index"])
    print(f"{name}: frames={n_frames} episodes={n_eps} pairs={n_pairs} "
          f"(expected {n_frames - n_eps})  RAM={d['frames'].nbytes / 1e9:.2f} GB")
    # no pair crosses an episode boundary
    t = d["pair_index"]
    assert (d["episode_index"][t] == d["episode_index"][t + 1]).all()

a = train["actions"]
print("train actions mean≈0:", np.abs(a.mean(0)).max().round(3),
      " std≈1:", a.std(0).min().round(3), a.std(0).max().round(3))