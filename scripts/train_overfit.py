"""Step 4: overfit the actor-world model on one Open-H episode."""
from pathlib import Path

import numpy as np
import wandb
from PIL import Image
from tensorflow import keras

from models.data import load_episode_triplets
from models.simple_world_model.swm import build_actor_world_model   # check: folder name

# --- 0: configuration, the ONLY place where settings live ---
CONFIG = dict(episode=0, epochs=300, batch_size=16, lr=3e-4, img_size=224,
              latent_dim=512, seed=0,
              w_action=1.0, w_state=1.0, w_image=0.1)
run = wandb.init(project="surgical-world-model", name="overfit-ep0", config=CONFIG)

keras.utils.set_random_seed(CONFIG["seed"])
OUT = Path(f"outputs/overfit_ep{CONFIG['episode']}")
OUT.mkdir(parents=True, exist_ok=True)

# --- A: data ---
inputs, targets = load_episode_triplets(CONFIG["episode"])

# --- B: the copy baseline ---
copy_mse = float(np.mean((inputs["image_input"] - targets["next_image_output"]) ** 2))
run.summary["copy_baseline_image_mse"] = copy_mse
print(f"copy baseline image MSE: {copy_mse:.6f}")

# --- C: model and losses ---
model = build_actor_world_model(latent_dim=CONFIG["latent_dim"])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG["lr"]),
    loss={"action_output": keras.losses.Huber(),
          "next_state_output": "mse",
          "next_image_output": "mse"},
    loss_weights={"action_output": CONFIG["w_action"],
                  "next_state_output": CONFIG["w_state"],
                  "next_image_output": CONFIG["w_image"]},
)

# --- D: training, with the losses sent to W&B after every epoch ---
class WandbEpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        run.log({k: float(v) for k, v in (logs or {}).items()}, step=epoch)

history = model.fit(inputs, targets, epochs=CONFIG["epochs"],
                    batch_size=CONFIG["batch_size"], shuffle=True, verbose=2,
                    callbacks=[WandbEpochLogger()])                      # FIXED

# --- E: final check on the same 121 triplets ---
pred_action, pred_state, pred_image = model.predict(
    inputs, batch_size=CONFIG["batch_size"], verbose=0)
model_mse = float(np.mean((pred_image - targets["next_image_output"]) ** 2))
state_mse = float(np.mean((pred_state - targets["next_state_output"]) ** 2))
print(f"image MSE: model {model_mse:.6f} | copy {copy_mse:.6f} | ratio {model_mse / copy_mse:.3f}")
print(f"next-state MSE (normalised units): {state_mse:.6f}")

run.summary["image_mse_model"] = model_mse
run.summary["image_ratio_vs_copy"] = model_mse / copy_mse
run.summary["next_state_mse"] = state_mse

# --- F: pictures (input t | true t+1 | predicted t+1), local and on W&B ---
images = []
for t in [0, 60, 120]:
    row = np.concatenate([inputs["image_input"][t], targets["next_image_output"][t],
                          pred_image[t]], axis=1)
    row = (np.clip(row, 0, 1) * 255).astype(np.uint8)                    # explicit 0..255
    Image.fromarray(row).save(OUT / f"t{t:03d}_input_true_pred.png")
    images.append(wandb.Image(row, caption=f"t={t}: input | true t+1 | predicted t+1"))
run.log({"predictions": images})

run.finish()