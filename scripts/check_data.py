from models.data import load_episode_triplets

inputs, targets = load_episode_triplets(0)
for name, arr in {**inputs, **targets}.items():
    print(f"{name:20s} {str(arr.shape):20s} mean={arr.mean():+.3f} std={arr.std():.3f}")