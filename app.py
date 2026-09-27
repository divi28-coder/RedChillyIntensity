import os
import json
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from PIL import Image

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = "processed_dataset"
TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

MODEL_PATH = "chilli_powder_model.keras"
CLASS_FILE = "classes.txt"
PROFILE_FILE = "colour_profiles.json"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

# Only one short training stage.
EPOCHS = 5

SEED = 42

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.isdir(TRAIN_DIR):

    raise FileNotFoundError(
        f"Dataset not found:\n{TRAIN_DIR}"
    )


# ============================================================
# CLASSES
# ============================================================

class_names = sorted(

    [
        folder
        for folder in os.listdir(
            TRAIN_DIR
        )

        if os.path.isdir(
            os.path.join(
                TRAIN_DIR,
                folder
            )
        )
    ]
)

if len(class_names) < 2:

    raise ValueError(
        "At least two classes are required."
    )


print()
print("=" * 60)
print("🌶️ FAST CHILLI MODEL TRAINING")
print("=" * 60)

print(
    "Classes:",
    len(class_names)
)


# ============================================================
# SAVE CLASSES
# ============================================================

with open(
    CLASS_FILE,
    "w",
    encoding="utf-8"
) as file:

    for name in class_names:

        file.write(
            name + "\n"
        )


# ============================================================
# DATASET
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="int",

    class_names=class_names,

    validation_split=0.20,

    subset="training",

    seed=SEED,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True
)


validation_ds = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="int",

    class_names=class_names,

    validation_split=0.20,

    subset="validation",

    seed=SEED,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# LIGHT AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.03
        ),

        layers.RandomZoom(
            0.05
        )
    ]
)


# ============================================================
# MOBILENETV2
# ============================================================

print(
    "Loading MobileNetV2..."
)

base_model = MobileNetV2(

    include_top=False,

    weights="imagenet",

    input_shape=(
        224,
        224,
        3
    )
)

# IMPORTANT:
# Do not fine-tune.
# This makes training much faster.

base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        224,
        224,
        3
    )
)

x = augmentation(
    inputs
)

x = preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(
    x
)

x = layers.Dropout(
    0.2
)(
    x
)

outputs = layers.Dense(

    len(class_names),

    activation="softmax"
)(
    x
)

model = models.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# TRAIN
# ============================================================

print()
print(
    "Training model..."
)

model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS
)


# ============================================================
# SAVE
# ============================================================

model.save(
    MODEL_PATH
)

print()
print(
    "Model saved:",
    MODEL_PATH
)


# ============================================================
# COLOUR FEATURES
# ============================================================

def extract_colour_features(image):

    image = image.convert(
        "RGB"
    )

    rgb = np.asarray(
        image,
        dtype=np.float64
    )

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    mean_r = float(
        np.mean(r)
    )

    mean_g = float(
        np.mean(g)
    )

    mean_b = float(
        np.mean(b)
    )

    brightness = (
        mean_r +
        mean_g +
        mean_b
    ) / 3.0

    channel_max = np.max(
        rgb,
        axis=2
    )

    channel_min = np.min(
        rgb,
        axis=2
    )

    intensity = (

        np.mean(
            channel_max -
            channel_min
        )

        / 255.0

    ) * 100.0

    total = (
        mean_r +
        mean_g +
        mean_b
    )

    if total > 0:

        red_dominance = (
            mean_r /
            total
        ) * 100.0

    else:

        red_dominance = 0.0

    red_excess = max(

        0.0,

        mean_r -
        (
            mean_g +
            mean_b
        ) / 2.0
    )

    return {

        "red":
            mean_r,

        "green":
            mean_g,

        "blue":
            mean_b,

        "brightness":
            brightness,

        "intensity":
            intensity,

        "red_dominance":
            red_dominance,

        "red_excess":
            red_excess
    }


# ============================================================
# BUILD COLOUR PROFILES
# ============================================================

print()
print(
    "Building dataset colour profiles..."
)

colour_profiles = {}


for class_name in class_names:

    folder = os.path.join(
        TRAIN_DIR,
        class_name
    )

    image_files = []

    for filename in os.listdir(
        folder
    ):

        if filename.lower().endswith(
            VALID_EXTENSIONS
        ):

            image_files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    feature_list = []

    for image_path in image_files:

        try:

            with Image.open(
                image_path
            ) as image:

                feature_list.append(

                    extract_colour_features(
                        image
                    )
                )

        except Exception:

            continue


    if not feature_list:

        continue


    keys = [

        "red",
        "green",
        "blue",
        "brightness",
        "intensity",
        "red_dominance",
        "red_excess"
    ]


    profile = {}


    for key in keys:

        profile[key] = float(

            np.mean(

                [
                    item[key]
                    for item in feature_list
                ]
            )
        )


    profile["image_count"] = len(
        feature_list
    )


    colour_profiles[
        class_name
    ] = profile


# ============================================================
# SAVE COLOUR PROFILES
# ============================================================

with open(

    PROFILE_FILE,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        colour_profiles,

        file,

        indent=2
    )


print()
print(
    "Colour profiles saved:",
    PROFILE_FILE
)


# ============================================================
# EVALUATION
# ============================================================

loss, accuracy = model.evaluate(

    validation_ds,

    verbose=0
)

print()
print("=" * 60)
print(
    f"Validation accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Validation loss: "
    f"{loss:.4f}"
)

print("=" * 60)

print(
    "Training completed."
)