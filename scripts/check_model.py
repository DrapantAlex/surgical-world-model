"""Shape check: one batch of one episode through the model (no training).

  python -m scripts.check_model [--dataset-root ...] [--episode 0]
"""
import argparse
from pathlib import Path

from models.data import ROOT, feature_dims, load_episode_triplets
from models.simple_world_model.swm import build_actor_world_model

p = argparse.ArgumentParser()
p.add_argument("--dataset-root", default=ROOT.as_posix(), help="default: tissue lift")
p.add_argument("--episode", type=int, default=0)
args = p.parse_args()
root = Path(args.dataset_root)

state_dim, action_dim = feature_dims(root)
print(f"{root.name}: state_dim={state_dim} action_dim={action_dim} (from info.json)")
model = build_actor_world_model(state_dim, action_dim)
model.summary()

inputs, targets = load_episode_triplets(args.episode, root)
batch = {k: v[:2] for k, v in inputs.items()}          # just 2 triplets
for name, x in batch.items():
    print(f"in   {name:20s} {x.shape}")
for name, out in zip(model.output_names, model.predict(batch, verbose=0)):
    print(f"out  {name:20s} {out.shape}  (target {targets[name][:2].shape})")
