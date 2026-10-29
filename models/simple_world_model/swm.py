"""Actor-world model adapted from Open-X to Open-H."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from models.encoders import build_encoder

IMG_H, IMG_W = 224, 224
LANG_DIM = 512
LATENT_DIM = 512   # NEW: size of the bottleneck

# CHANGED: state_dim / action_dim have no default; pass data.feature_dims(root)
def build_actor_world_model(state_dim, action_dim, img_h=IMG_H, img_w=IMG_W, lang_dim=LANG_DIM,
                            latent_dim=LATENT_DIM, encoder="simple_cnn",
                            freeze_encoder=False, unfreeze_last_n=0, delta_factor=None,
                            action_chunk=None):
    # NEW (E7): delta_factor = data.state_delta_factor(stats) -> the state head predicts the
    #   change f (in Δ/scale units) and next_state = state + delta_factor * f; None = original
    # NEW (E8): action_chunk = k -> the actor predicts k offsets from a(t-1), shape (k, action_dim)
    # --- A: inputs ---
    image_input = keras.Input(shape=(img_h, img_w, 3), name="image_input")
    state_input = keras.Input(shape=(state_dim,), name="state_input")
    language_input = keras.Input(shape=(lang_dim,), name="language_input")
    action_input = keras.Input(shape=(action_dim,), name="action_input")

    # --- B: visual encoder (CHANGED: chosen by name; "simple_cnn" = the original 4 convs) ---
    encoded_img = build_encoder(encoder, image_input, freeze_encoder, unfreeze_last_n)
    feat_h, feat_w = img_h // 16, img_w // 16          # CHANGED: computed, not hard-coded 14
                                                       # (decoder start: 4 x2 upsamplings back to img size)

    # --- C: actor branch (sizes only) ---
    actor_features = layers.Concatenate()([encoded_img, state_input, language_input])
    a_x = layers.Dense(256, activation="relu")(actor_features)
    if action_chunk is None:
        action_output = layers.Dense(action_dim, name="action_output")(a_x)
    else:
        # NEW (E8): one Dense for the whole chunk, then (k, action_dim); same name for the loss
        a_x = layers.Dense(action_chunk * action_dim, name="action_chunk")(a_x)
        action_output = layers.Reshape((action_chunk, action_dim), name="action_output")(a_x)

    # --- D: world-model branch ---
    wm_features = layers.Concatenate()(
        [encoded_img, state_input, language_input, action_input])

    s_x = layers.Dense(128, activation="relu")(wm_features)
    if delta_factor is None:
        next_state_output = layers.Dense(state_dim, name="next_state_output")(s_x)
    else:
        # NEW (E7): skip connection; delta_factor is a fixed constant, not trained
        f = layers.Dense(state_dim, name="state_delta")(s_x)
        k = np.asarray(delta_factor, dtype="float32")
        next_state_output = layers.Add(name="next_state_output")(
            [state_input, layers.Lambda(lambda x: x * k, name="delta_to_state")(f)])

    # latent_dim=None -> original decoder: the big Dense reads wm_features directly
    #   (~2.5 billion weights, ~40 GB GPU memory to train).
    # latent_dim=512  -> bottleneck z first (~52 million weights in this part).
    if latent_dim is None:
        dec_in = wm_features
    else:
        dec_in = layers.Dense(latent_dim, activation="relu", name="latent")(wm_features)

    dec = layers.Dense(feat_h * feat_w * 256, activation="relu")(dec_in)
    dec = layers.Reshape((feat_h, feat_w, 256))(dec)
    dec = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(dec)
    dec = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(dec)
    dec = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(dec)
    next_image_output = layers.Conv2DTranspose(
        3, 3, strides=2, padding="same", activation="sigmoid",
        name="next_image_output")(dec)

    return keras.Model(
        inputs=[image_input, state_input, language_input, action_input],
        outputs=[action_output, next_state_output, next_image_output],
        name="actor_world_model")

