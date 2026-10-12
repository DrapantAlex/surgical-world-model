from pathlib import Path
import numpy as np
import av
from PIL import Image

ROOT = "data/open_h/Surgical/jhu/imerse/srt_tissue_lift"
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# --- A: decode one camera into a (T, H, W, 3) uint8 array ---
def read_frames(camera, width=320, height=256):
    path = f"{ROOT}/videos/chunk-000/{camera}/episode_000000.mp4"
    with av.open(path) as container:
        frames = [f.to_ndarray(format="rgb24", width=width, height=height)
                  for f in container.decode(video=0)]
    return np.stack(frames)

left = read_frames("observation.images.endoscope.left")
print("left:", left.shape, left.dtype)

# --- B: the alignment check ---
print("frames == parquet rows:", len(left) == 122)

# --- C: are left and right really different cameras? ---
right = read_frames("observation.images.endoscope.right")
print("identical:", np.array_equal(left, right))
print("mean abs difference:", np.abs(left.astype(int) - right.astype(int)).mean())

# --- D: save first, middle and last frame to look at ---
for t in [0, 61, 121]:
    Image.fromarray(left[t]).save(OUT / f"endo_left_{t:03d}.png")

path = f"{ROOT}/videos/chunk-000/observation.images.endoscope.left/episode_000000.mp4"
with av.open(path) as container:
    frame = next(container.decode(video=0))           # only frame 0
    for w, h in [(1280, 1024), (320, 256), (224, 224)]:
        img = frame.to_ndarray(format="rgb24", width=w, height=h)
        Image.fromarray(img).save(OUT / f"frame0_{w}x{h}.png")