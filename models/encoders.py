"""Image encoders for the actor-world model, chosen by name (train.py --encoder).

Every encoder receives images in [0, 1] (what the data pipeline gives) and
returns one flat feature vector per image. Anything encoder-specific, such as a
Keras Applications preprocess_input, is a layer INSIDE the encoder, so the data
pipeline is the same for every encoder.

To add a backbone: add one line to APPLICATIONS (constructor, preprocess_input).
"""
from tensorflow import keras
from tensorflow.keras import layers

# --- A: pretrained Keras Applications backbones (ImageNet weights) ---
APPLICATIONS = {
    "efficientnetb0": (keras.applications.EfficientNetB0,
                       keras.applications.efficientnet.preprocess_input),
}
ENCODERS = ["simple_cnn", *APPLICATIONS]


# --- B: the original encoder (from the professor's script, unchanged) ---
def simple_cnn(image_input):
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(image_input)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    return layers.Flatten()(x)


# --- C: a pretrained backbone, frozen, partly fine-tuned, or fully trained ---
def keras_application(name, image_input, freeze=False, unfreeze_last_n=0):
    constructor, preprocess = APPLICATIONS[name]
    backbone = constructor(include_top=False, weights="imagenet",
                           input_shape=tuple(image_input.shape[1:]), pooling="avg")

    if freeze:
        backbone.trainable = False
        if unfreeze_last_n > 0:
            backbone.trainable = True            # sets every layer trainable ...
            for layer in backbone.layers[:-unfreeze_last_n]:
                layer.trainable = False          # ... then refreeze all but the last N
    for layer in backbone.layers:                # BatchNorm statistics are never updated
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    x = layers.Rescaling(255.0, name="to_0_255")(image_input)   # Applications expect [0, 255]
    x = layers.Lambda(preprocess, name="preprocess_input")(x)
    return backbone(x, training=False)   # BatchNorm always in inference mode


# --- D: one entry point, used by the model builder ---
def build_encoder(name, image_input, freeze=False, unfreeze_last_n=0):
    if unfreeze_last_n and not freeze:
        raise ValueError("unfreeze_last_n needs freeze=True (freeze everything, then unfreeze the top)")
    if name == "simple_cnn":
        if freeze:
            raise ValueError("simple_cnn starts from random weights: freezing it makes no sense")
        return simple_cnn(image_input)
    if name in APPLICATIONS:
        return keras_application(name, image_input, freeze, unfreeze_last_n)
    raise ValueError(f"unknown encoder {name!r}; choose from {ENCODERS}")
