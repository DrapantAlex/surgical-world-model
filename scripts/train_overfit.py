"""Step 4: overfit the actor-world model on one Open-H episode."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # the DGX has no screen: draw straight to files
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow import keras

from models.data import load_episode_triplets
from models.simple_world_model.swm import build_actor_world_model

EPISODE = 0
EPOCHS = 300
BATCH_SIZE = 16
OUT = Path("outputs/overfit_ep0")
OUT.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(0)     # same initial weights every run

# --- A: data ---
inputs, targets = load_episode_triplets(EPISODE)

# --- B: the copy baseline ---
copy_mse = float(np.mean((inputs["image_input"] - targets["next_image_output"]) ** 2))
print(f"copy baseline image MSE: {copy_mse:.6f}")

# --- C: model and losses (as in the professor's script) ---
model = build_actor_world_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss={"action_output": keras.losses.Huber(),
          "next_state_output": "mse",
          "next_image_output": "mse"},
    loss_weights={"action_output": 1.0,
                  "next_state_output": 1.0,
                  "next_image_output": 0.1},
)

# --- D: training ---
history = model.fit(inputs, targets, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, shuffle=True, verbose=2)

# --- E: final check on the same 121 triplets ---
pred_action, pred_state, pred_image = model.predict(inputs, batch_size=BATCH_SIZE, verbose=0)
model_mse = float(np.mean((pred_image - targets["next_image_output"]) ** 2))
state_mse = float(np.mean((pred_state - targets["next_state_output"]) ** 2))
print(f"image MSE: model {model_mse:.6f} | copy {copy_mse:.6f} | ratio {model_mse / copy_mse:.3f}")
print(f"next-state MSE (normalised units): {state_mse:.6f}")

# --- F: pictures: input t | true t+1 | predicted t+1 ---
for t in [0, 60, 120]:
    row = np.concatenate([inputs["image_input"][t],
                          targets["next_image_output"][t],
                          pred_image[t]], axis=1)
    Image.fromarray((row * 255).astype(np.uint8)).save(OUT / f"t{t:03d}_input_true_pred.png")

# --- G: loss curves ---
plt.figure(figsize=(8, 4))
for key in ["next_image_output_loss", "next_state_output_loss", "action_output_loss"]:
    plt.plot(history.history[key], label=key)
plt.yscale("log")
plt.xlabel("epoch"); plt.ylabel("loss (log scale)"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "loss_curves.png")
print("saved to", OUT)