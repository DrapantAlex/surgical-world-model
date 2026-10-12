"""Actor-world model adapted from Open-X to Open-H."""
from tensorflow import keras
from tensorflow.keras import layers

IMG_H, IMG_W = 224, 224
STATE_DIM = 16     # CHANGED: was 7  (2 arms x (xyz + quaternion + jaw))
ACTION_DIM = 16    # CHANGED: was 4
LANG_DIM = 512
LATENT_DIM = 512   # NEW: size of the bottleneck


def build_actor_world_model(img_h=IMG_H, img_w=IMG_W, state_dim=STATE_DIM,
                            action_dim=ACTION_DIM, lang_dim=LANG_DIM,
                            latent_dim=LATENT_DIM):
    # --- A: inputs ---
    image_input = keras.Input(shape=(img_h, img_w, 3), name="image_input")
    state_input = keras.Input(shape=(state_dim,), name="state_input")
    language_input = keras.Input(shape=(lang_dim,), name="language_input")
    action_input = keras.Input(shape=(action_dim,), name="action_input")

    # --- B: visual encoder (unchanged) ---
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(image_input)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    feat_h, feat_w = img_h // 16, img_w // 16          # CHANGED: computed, not hard-coded 14
    encoded_img = layers.Flatten()(x)

    # --- C: actor branch (sizes only) ---
    actor_features = layers.Concatenate()([encoded_img, state_input, language_input])
    a_x = layers.Dense(256, activation="relu")(actor_features)
    action_output = layers.Dense(action_dim, name="action_output")(a_x)

    # --- D: world-model branch ---
    wm_features = layers.Concatenate()(
        [encoded_img, state_input, language_input, action_input])

    s_x = layers.Dense(128, activation="relu")(wm_features)
    next_state_output = layers.Dense(state_dim, name="next_state_output")(s_x)

    # In the original model, the first decoder layer has about 2.5 billion parameters, which needs roughly 40 GB of memory to train. 
    # With the bottleneck, the same branch has about 52 million.
    z = layers.Dense(latent_dim, activation="relu", name="latent")(wm_features)
    
    dec = layers.Dense(feat_h * feat_w * 256, activation="relu")(z)            # CHANGED: input z
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