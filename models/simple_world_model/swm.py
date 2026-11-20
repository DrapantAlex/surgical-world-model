"""Actor-world model adapted from Open-X to Open-H."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from models.encoders import build_encoder

IMG_H, IMG_W = 224, 224
LANG_DIM = 512
LATENT_DIM = 512   # NEW: size of the bottleneck
WM_MIXER_KERNEL = 7   # NEW (1c): depthwise kernel of the WM ConvMixer blocks (7 x 7 of the 14 x 14 map)

# CHANGED: state_dim / action_dim have no default; pass data.feature_dims(root)
def build_actor_world_model(state_dim, action_dim, img_h=IMG_H, img_w=IMG_W, lang_dim=LANG_DIM,
                            latent_dim=LATENT_DIM, encoder="simple_cnn",
                            freeze_encoder=False, unfreeze_last_n=0, delta_factor=None,
                            action_chunk=None, state_skip=False, state_layers=1, init_seed=0,
                            wm_mixer_blocks=0, history_dim=0):
    # NEW (E7): delta_factor = data.state_delta_factor(stats) -> the state head predicts the
    #   change f (in Δ/scale units) and next_state = state + delta_factor * f; None = original
    # NEW (E8): action_chunk = k -> the actor predicts k offsets from a(t-1), shape (k, action_dim)
    # NEW (1a): state_skip=True -> next_state = state + Dense(s_x), the Dense zero-initialised, so
    #   the untrained model is exactly copy_state; same loss (MSE) and targets as the original
    # NEW (1b): state_layers = K hidden Dense(128, relu) layers in the state head; 1 = original;
    #   init_seed (the run's --seed) seeds the K-1 extra layers' initialisers
    # NEW (1c): wm_mixer_blocks = D residual ConvMixer blocks on the encoder map, read by the WM
    #   branch only (both heads); 0 = original. Each block starts as the identity.
    # NEW (history): history_dim = len(h(t)) = state_dim + action_dim -> one more Input,
    #   history_input = [Δs(t), Δa(t-1)] / scale (data.py D3), appended LAST to the actor's and
    #   the WM's Concatenate; 0 = original. No new layer: the first Dense of the actor, the first
    #   Dense of the state head and the latent Dense get history_dim more input rows. Glorot's
    #   limit depends on fan_in, so those three kernels start different from 1a; every other
    #   weight starts identical (each initialiser draw takes one seed, whatever its shape).
    if history_dim < 0:
        raise ValueError(f"history_dim must be >= 0, got {history_dim}")
    if state_skip and delta_factor is not None:
        raise ValueError("state_skip and delta_factor are two different state heads: choose one")
    if state_layers < 1:
        raise ValueError(f"state_layers must be >= 1, got {state_layers}")
    if wm_mixer_blocks < 0:
        raise ValueError(f"wm_mixer_blocks must be >= 0, got {wm_mixer_blocks}")
    if wm_mixer_blocks > 0 and encoder != "simple_cnn":
        # Keras Applications backbones return pooling="avg": no spatial map left (step 6)
        raise ValueError(f"wm_mixer_blocks needs the spatial map of simple_cnn, not {encoder!r}")
    # --- A: inputs ---
    image_input = keras.Input(shape=(img_h, img_w, 3), name="image_input")
    state_input = keras.Input(shape=(state_dim,), name="state_input")
    language_input = keras.Input(shape=(lang_dim,), name="language_input")
    action_input = keras.Input(shape=(action_dim,), name="action_input")
    history = ([keras.Input(shape=(history_dim,), name="history_input")]   # NEW (history)
               if history_dim > 0 else [])

    # --- B: visual encoder (CHANGED: chosen by name; "simple_cnn" = the original 4 convs) ---
    encoded_img = build_encoder(encoder, image_input, freeze_encoder, unfreeze_last_n)
    feat_h, feat_w = img_h // 16, img_w // 16          # CHANGED: computed, not hard-coded 14
                                                       # (decoder start: 4 x2 upsamplings back to img size)

    # --- C: actor branch (sizes only) ---
    actor_features = layers.Concatenate()(
        [encoded_img, state_input, language_input] + history)   # NEW (history): + h(t)
    a_x = layers.Dense(256, activation="relu")(actor_features)
    if action_chunk is None:
        action_output = layers.Dense(action_dim, name="action_output")(a_x)
    else:
        # NEW (E8): one Dense for the whole chunk, then (k, action_dim); same name for the loss
        a_x = layers.Dense(action_chunk * action_dim, name="action_chunk")(a_x)
        action_output = layers.Reshape((action_chunk, action_dim), name="action_output")(a_x)

    # --- D: world-model branch ---
    wm_img = encoded_img
    if wm_mixer_blocks > 0:
        # NEW (1c): the flat encoder output back to its map (feat_h, feat_w, C), D blocks, flat again.
        #   Original ConvMixer: x = BN(gelu(Depthwise(x))) + x ; x = BN(gelu(Pointwise(x))).
        #   Here the residual wraps the WHOLE block, y = x + Pointwise(relu(Depthwise(x))), with
        #   Pointwise zero-initialised (kernel and bias): every block starts as y = x, so the
        #   untrained model equals the D = 0 model exactly (like 1a for the state). No
        #   normalisation and no activation after the addition, so nothing breaks that identity.
        #   Every new layer has an explicit name (Keras auto-names per class: an unnamed Reshape
        #   would rename the decoder's) and no draw from the global initialiser sequence
        #   (Depthwise: own seeded Glorot, as in 1b; biases and Pointwise: zeros).
        n_pos = feat_h * feat_w
        if encoded_img.shape[-1] % n_pos != 0:
            raise ValueError(f"encoder output {encoded_img.shape[-1]} is not a "
                             f"{feat_h} x {feat_w} x C map")
        C = encoded_img.shape[-1] // n_pos
        x = layers.Reshape((feat_h, feat_w, C), name="wm_map")(encoded_img)
        
        for i in range(wm_mixer_blocks):
            h = layers.DepthwiseConv2D(
                WM_MIXER_KERNEL, padding="same", activation="relu",
                depthwise_initializer=keras.initializers.GlorotUniform(seed=init_seed + 2000 + i),
                name=f"wm_mixer_dw_{i}")(x)                                   # 1. move: each channel, 7×7 neighbourhood
            h = layers.Conv2D(
                C, 1, kernel_initializer="zeros", bias_initializer="zeros",
                name=f"wm_mixer_pw_{i}")(h)                                   # 2. combine channels, starts at 0
            x = layers.Add(name=f"wm_mixer_add_{i}")([x, h])                  # 3. residual: x + correction
        wm_img = layers.Flatten(name="wm_flat")(x)
    
    wm_features = layers.Concatenate()(
        [wm_img, state_input, language_input, action_input]   # 1c: wm_img (actor: encoded_img)
        + history)                                             # NEW (history): + h(t)

    s_x = layers.Dense(128, activation="relu")(wm_features)
    # NEW (1b): deeper state head. Keras draws initial weights in layer-creation order from one
    #   global sequence, and these layers are created before the image decoder: with default
    #   initialisers they would shift the decoder's initial weights. Their own seeded Glorot
    #   (bias zeros, no randomness) keeps every other layer's init identical to K = 1.
    for i in range(1, state_layers):
        s_x = layers.Dense(128, activation="relu", name=f"state_hidden_{i}",
                           kernel_initializer=keras.initializers.GlorotUniform(
                               seed=init_seed + 1000 + i))(s_x)
    if state_skip:
        # NEW (1a): skip connection in normalised units; only this output layer starts at zero
        #   (bias is zero by default), so at initialisation next_state = state_input exactly
        delta = layers.Dense(state_dim, kernel_initializer="zeros", name="state_change")(s_x)
        next_state_output = layers.Add(name="next_state_output")([state_input, delta])
    elif delta_factor is None:
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
        inputs=[image_input, state_input, language_input, action_input] + history,
        outputs=[action_output, next_state_output, next_image_output],
        name="actor_world_model")

