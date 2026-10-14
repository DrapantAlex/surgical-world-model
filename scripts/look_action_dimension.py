import json, sys

path = sys.argv[1]   # e.g. data/srt_tissue_lift/meta/info.json
info = json.load(open(path))

for key in ["action", "observation.state"]:
    feat = info["features"][key]
    names = feat["names"][0] if isinstance(feat["names"][0], list) else feat["names"]
    print(f"\n{key}  shape={feat['shape']}  dtype={feat['dtype']}")
    for i, n in enumerate(names):
        print(f"  [{i:2d}] {n}")