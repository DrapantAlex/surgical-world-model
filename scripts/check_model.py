from models.data import load_episode_triplets
from models.simple_world_model.swm import build_actor_world_model

model = build_actor_world_model()
model.summary()

inputs, targets = load_episode_triplets(0)
batch = {k: v[:2] for k, v in inputs.items()}          # just 2 triplets
for name, out in zip(model.output_names, model.predict(batch, verbose=0)):
    print(f"{name:20s} {out.shape}")