"""Step 4: overfit the actor-world model on one Open-H episode.

  python -m scripts.train_overfit [--dataset-root data/open_h/Surgical/jhu/imerse/star_il/star_il]
"""
import argparse
from pathlib import Path

import numpy as np
import wandb
from PIL import Image
from tensorflow import keras

from models.data import ROOT, feature_dims, load_episode_triplets, load_instruction
from models.simple_world_model.swm import build_actor_world_model
from models.visualization import WorldModelTimelineCallback

# --- 0: configuration, the ONLY place where settings live (+ the dataset, from the CLI) ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", default=ROOT.as_posix(), help="default: tissue lift")
CONFIG = dict(dataset_root=parser.parse_args().dataset_root, episode=0, epochs=300, batch_size=16, lr=3e-4, img_size=224,
              latent_dim=None, seed=0,
              w_action=1.0, w_state=1.0, w_image=0.1)
run = wandb.init(project="surgical-world-model",
                 name=f"overfit-ep0-latent{CONFIG['latent_dim']}", config=CONFIG)

keras.utils.set_random_seed(CONFIG["seed"])
OUT = Path("outputs") / run.name           
OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("logs") / run.name            

# --- A: data ---
ROOT_DIR = Path(CONFIG["dataset_root"])
inputs, targets = load_episode_triplets(CONFIG["episode"], ROOT_DIR)

# --- B: the copy baseline ---
copy_mse = float(np.mean((inputs["image_input"] - targets["next_image_output"]) ** 2))
run.summary["copy_baseline_image_mse"] = copy_mse
print(f"copy baseline image MSE: {copy_mse:.6f}")

# --- C: model and losses ---
model = build_actor_world_model(*feature_dims(ROOT_DIR), latent_dim=CONFIG["latent_dim"])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG["lr"]),
    loss={"action_output": keras.losses.Huber(),
          "next_state_output": "mse",
          "next_image_output": "mse"},
    loss_weights={"action_output": CONFIG["w_action"],
                  "next_state_output": CONFIG["w_state"],
                  "next_image_output": CONFIG["w_image"]},
)

# --- D: training, with losses and timeline sent to W&B / TensorBoard ---
class WandbEpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        run.log({k: float(v) for k, v in (logs or {}).items()}, step=epoch)

timeline_cb = WorldModelTimelineCallback(                                  
    LOG_DIR, inputs, targets, load_instruction(CONFIG["episode"], ROOT_DIR), wandb_run=run,
    num_frames=8, start=0, stride=1)

history = model.fit(inputs, targets, epochs=CONFIG["epochs"],
                    batch_size=CONFIG["batch_size"], shuffle=True, verbose=2,
                    callbacks=[WandbEpochLogger(),
                               timeline_cb,                                  
                               keras.callbacks.TensorBoard(log_dir=str(LOG_DIR))])  

model.save(OUT / "model.keras")

# --- E: final check on the same 121 triplets, each output against a baseline ---
pred_action, pred_state, pred_image = model.predict(
    inputs, batch_size=CONFIG["batch_size"], verbose=0)

model_mse = float(np.mean((pred_image - targets["next_image_output"]) ** 2))
state_mse = float(np.mean((pred_state - targets["next_state_output"]) ** 2))
state_copy = float(np.mean((inputs["state_input"] - targets["next_state_output"]) ** 2))   
action_mse = float(np.mean((pred_action - targets["action_output"]) ** 2))                  
action_zero = float(np.mean(targets["action_output"] ** 2))                                 

print(f"image  MSE: model {model_mse:.6f} | copy {copy_mse:.6f} | ratio {model_mse / copy_mse:.3f}")
print(f"state  MSE: model {state_mse:.6f} | copy {state_copy:.6f} | ratio {state_mse / state_copy:.3f}")
print(f"action MSE: model {action_mse:.6f} | mean-action {action_zero:.6f}")

run.summary.update({
    "image_mse_model": model_mse, "image_ratio_vs_copy": model_mse / copy_mse,
    "next_state_mse": state_mse, "state_ratio_vs_copy": state_mse / state_copy,      
    "action_mse": action_mse, "action_mse_mean_baseline": action_zero,              
})

# --- F: pictures (input t | true t+1 | predicted t+1), local and on W&B ---
images = []
for t in [0, 60, 120]:
    row = np.concatenate([inputs["image_input"][t], targets["next_image_output"][t],
                          pred_image[t]], axis=1)
    row = (np.clip(row, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(row).save(OUT / f"t{t:03d}_input_true_pred.png")
    images.append(wandb.Image(row, caption=f"t={t}: input | true t+1 | predicted t+1"))
run.log({"predictions": images})

run.finish()