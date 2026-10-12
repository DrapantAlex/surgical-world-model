import os
import io
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
# ==========================================
# 1. DATASET UTILITIES & PIPELINE
# ==========================================



def dataset2path(dataset_name):
  """Maps the dataset name to the external Open X-Embodiment GCS bucket."""
  version = '0.1.0'
  return f'gs://gresearch/robotics/{dataset_name}/{version}'



def transition_map_fn(step_t, step_t_next):
  """Maps sequential steps into World Model inputs and targets."""
  # Inputs at time t
  image_t = tf.cast(step_t['observation']['image'], tf.float32) / 255.0
  state_t = step_t['observation']['state']
  language = step_t['language_embedding']
  action_t = step_t['action']

  # Targets at time t+1
  image_t_next = tf.cast(step_t_next['observation']['image'], tf.float32) / 255.0
  state_t_next = step_t_next['observation']['state']

  inputs = {
     'image_input': image_t,
     'state_input': state_t,
     'language_input': language,
     'action_input': action_t  # Action taken at time t
  }

  targets = {
     'action_output': action_t,             # Behavior Cloning target
     'next_state_output': state_t_next,     # Proprioception target
     'next_image_output': image_t_next      # Visual dynamics target
  }

  return inputs, targets



def prepare_transition_dataset(dataset_name, batch_size=32, max_episodes=None):
  """Constructs a temporal tf.data.Dataset pipeline."""
  b = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
  split_str = f'train[:{max_episodes}]' if max_episodes else 'train'
  ds = b.as_dataset(split=split_str)

  # Zip each episode's steps with itself shifted by 1 to yield (t, t+1) pairs
  ds = ds.map(
     lambda ep: tf.data.Dataset.zip((ep['steps'], ep['steps'].skip(1))), 
     num_parallel_calls=tf.data.AUTOTUNE
  )
  # Flatten the dataset of paired transitions into a continuous sequence of transitions
  ds = ds.flat_map(lambda x: x)
  ds = ds.map(transition_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

  return ds.shuffle(2000).batch(batch_size).prefetch(tf.data.AUTOTUNE)





# ==========================================
# 2. JOINT ACTOR-WORLD MODEL ARCHITECTURE
# ==========================================



def build_actor_world_model():
  """Builds a multi-headed network for policy and dynamics prediction."""
  # Inputs
  image_input = keras.Input(shape=(224, 224, 3), name='image_input')
  state_input = keras.Input(shape=(7,), name='state_input')
  language_input = keras.Input(shape=(512,), name='language_input')
  action_input = keras.Input(shape=(4,), name='action_input')

  # Visual Encoder
  x = keras.layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(image_input) # 112
  x = keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)           # 56
  x = keras.layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)          # 28
  x = keras.layers.Conv2D(256, (3, 3), strides=2, padding='same', activation='relu')(x)          # 14
  encoded_img = keras.layers.Flatten()(x)

  # Actor Branch (Behavior Cloning)
  actor_features = keras.layers.Concatenate()([encoded_img, state_input, language_input])
  a_x = keras.layers.Dense(256, activation='relu')(actor_features)
  action_output = keras.layers.Dense(4, activation='linear', name='action_output')(a_x)

  # World Model Branch (Dynamics Prediction)
  wm_features = keras.layers.Concatenate()([encoded_img, state_input, language_input, action_input])

  # Proprioception (Next State) Predictor
  s_x = keras.layers.Dense(128, activation='relu')(wm_features)
  next_state_output = keras.layers.Dense(7, activation='linear', name='next_state_output')(s_x)

  # Visual Decoder (Next Image) Predictor
  dec_x = keras.layers.Dense(14 * 14 * 256, activation='relu')(wm_features)
  dec_x = keras.layers.Reshape((14, 14, 256))(dec_x)
  dec_x = keras.layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(dec_x) # 28
  dec_x = keras.layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(dec_x)  # 56
  dec_x = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(dec_x)  # 112
  next_image_output = keras.layers.Conv2DTranspose(3, (3, 3), strides=2, padding='same', activation='sigmoid', name='next_image_output')(dec_x) # 224

  model = keras.Model(
     inputs=[image_input, state_input, language_input, action_input], 
     outputs=[action_output, next_state_output, next_image_output]
  )
  return model





# ==========================================
# 3. TENSORBOARD VISUALIZATION CALLBACK
# ==========================================



class FastWorldModelVisCallback(keras.callbacks.Callback):
  def __init__(self, log_dir, validation_dataset, instruction_text, num_frames=8):
     """
     Visualizes the world model episode as a horizontal timeline:
     GT:   [t_0]   [t_1]   [t_2] ... [t_n]
     PRED: [empty] [t_1]   [t_2] ... [t_n]
     """
     super().__init__()
     self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, "model_visualizations"))
     self.instruction_text = instruction_text

     # Pull a sequential batch for the timeline
     reference_batch = next(iter(validation_dataset.unbatch().batch(num_frames).take(1)))
     self.reference_inputs, self.reference_targets = reference_batch



  def create_timeline_image(self, initial_imgs, true_next_imgs, pred_next_imgs, true_actions, pred_actions):
     """Stitches frames horizontally using cv2."""
     N = true_next_imgs.shape[0]

     # Convert tensors [0.0, 1.0] to numpy uint8 [0, 255]
     # We only need the very first image of the batch as our t_0 observation
     img_t0 = (initial_imgs[0].numpy() * 255).astype(np.uint8)
     true_next_np = (true_next_imgs.numpy() * 255).astype(np.uint8)
     pred_next_np = (pred_next_imgs * 255).astype(np.uint8)

     # 1. Prepare the starting column (t_0)
     img_t0 = np.ascontiguousarray(img_t0)
     cv2.putText(img_t0, "GT [t_0]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

     # Create a black placeholder for PRED at t_0
     blank_img = np.zeros_like(img_t0)
     cv2.putText(blank_img, "[empty]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

     gt_row_frames = [img_t0]
     pred_row_frames = [blank_img]

     # 2. Iterate through t_1 to t_n and append them horizontally
     for i in range(N):
         gt_i = np.ascontiguousarray(true_next_np[i])
         pred_i = np.ascontiguousarray(pred_next_np[i])

         # Annotate GT Frames
         cv2.putText(gt_i, f"GT [t_{i+1}]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
         act_true_str = f"Act: {true_actions[i][0]:.2f}, {true_actions[i][1]:.2f}"
         cv2.putText(gt_i, act_true_str, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

         # Annotate PRED Frames
         cv2.putText(pred_i, f"PRED [t_{i+1}]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1, cv2.LINE_AA)
         act_pred_str = f"Act: {pred_actions[i][0]:.2f}, {pred_actions[i][1]:.2f}"
         cv2.putText(pred_i, act_pred_str, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

         gt_row_frames.append(gt_i)
         pred_row_frames.append(pred_i)

     # 3. Stitch the rows together
     gt_row_stitched = np.concatenate(gt_row_frames, axis=1)
     pred_row_stitched = np.concatenate(pred_row_frames, axis=1)

     # Stack GT row on top of PRED row
     grid = np.concatenate([gt_row_stitched, pred_row_stitched], axis=0)

     # 4. Attach the language instruction header
     header = np.zeros((60, grid.shape[1], 3), dtype=np.uint8)
     cv2.putText(header, "Task Instruction:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
     cv2.putText(header, self.instruction_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

     final_img = np.concatenate([header, grid], axis=0)

     # Expand dims for TensorBoard (Batch=1, H, W, Channels)
     return np.expand_dims(final_img, axis=0)



  def on_epoch_end(self, epoch, logs=None):
     # Generate predictions for the sequence
     print("predicting")
     predictions = self.model.predict(self.reference_inputs, verbose=0)
     pred_actions = predictions[0]
     pred_next_images = predictions[2]
     print("predicting done")
     # Extract ground truth components
     initial_images = self.reference_inputs['image_input']
     true_actions = self.reference_targets['action_output'].numpy()
     true_next_images = self.reference_targets['next_image_output']
     print("creating vis")
     # Build the horizontal timeline composite
     composite_image = self.create_timeline_image(
         initial_images, true_next_images, pred_next_images, true_actions, pred_actions
     )
     print("creating vis done")
     # Log to TensorBoard
     with self.file_writer.as_default():
         tf.summary.image("Episode_Timeline_Rollout", composite_image, step=epoch)
         self.file_writer.flush()
# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================



if __name__ == "__main__":
  DATASET_NAME = "ucsd_pick_and_place_dataset_converted_externally_to_rlds"
  BATCH_SIZE = 64
  EPOCHS = 15
  LOG_DIR = os.path.join("logs", "ucsd_world_model")

  print(f"Preparing dataset: {DATASET_NAME}")
  # Prepare training stream (limited to 250 episodes for demonstration)
  train_dataset = prepare_transition_dataset(DATASET_NAME, batch_size=BATCH_SIZE, max_episodes=250)

  # Prepare a small visualization stream
  vis_dataset = prepare_transition_dataset(DATASET_NAME, batch_size=16, max_episodes=5)
  b = tfds.builder_from_directory(builder_dir=dataset2path(DATASET_NAME))
  raw_sample = next(iter(b.as_dataset(split='train[:1]')))
  # Extract the string from the first step of the episode
  instruction_text = next(iter(raw_sample['steps']))['language_instruction'].numpy().decode('utf-8')

  print("Building Actor-World Model...")
  model = build_actor_world_model()

  # Compile with distinct losses and weights to balance the gradients
  model.compile(
     optimizer=keras.optimizers.Adam(learning_rate=3e-4),
     loss={
         'action_output': keras.losses.Huber(),
         'next_state_output': 'mse',
         'next_image_output': 'mse'
     },
     loss_weights={
         'action_output': 1.0,        # Prioritize policy learning
         'next_state_output': 1.0,    # Prioritize state dynamics
         'next_image_output': 0.1     # Scale down high-dimensional image reconstruction
     }
  )

  # Setup Callbacks
  tensorboard_cb = keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0, update_freq='epoch')
  #visualizer_cb = WorldModelVisCallback(log_dir=LOG_DIR, validation_dataset=vis_dataset, num_frames=16)
  visualizer_cb = FastWorldModelVisCallback(
     log_dir="logs/world_model_fast", 
     validation_dataset=vis_dataset, 
     instruction_text=instruction_text,
     num_frames=8  # Creates a grid of 8 sequential transition steps
  )

  print("Starting training loop...")
  model.fit(
     train_dataset,
     epochs=EPOCHS,
     steps_per_epoch=200, 
     callbacks=[tensorboard_cb, visualizer_cb]
  )

  print(f"Training complete. View visualizations by running:\n  tensorboard --logdir={LOG_DIR}")