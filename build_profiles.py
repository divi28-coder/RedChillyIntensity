import os
import glob
import json

import numpy as np
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = os.path.join(
    "processed_dataset",
    "train"
)

COLOUR_PROFILE_FILE = (
    "colour_profiles.json"
)

VARIETY_DATASET_DIR = (
    "variety_dataset"
)

VARIETY_PROFILE_FILE = (
    "variety_profiles.json"
)


# ============================================================
# IMAGE FEATURES
# ============================================================

def extract_colour_features(image):

    image = image.convert("RGB")

    image = image.resize(
        (128, 128)
    )

    rgb = np.asarray(
        image,
        dtype=np.float32
    )

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    brightness = (
        mean_r +
        mean_g +
        mean_b
    ) / 3

    maximum = np.max(
        rgb,
        axis=2
    )

    minimum = np.min(
        rgb,
        axis=2
    )

    saturation = np.mean(
        (
            maximum -
            minimum
        ) / 255
    ) * 100

    total = (
        mean_r +
        mean_g +
        mean_b +
        0.000001
    )

    red_ratio = (
        mean_r /
        total *
        100
    )

    red_dominance = max(
        0,
        mean_r -
        (
            mean_g +
            mean_b
        ) / 2
    )

    quantized = (
        rgb.astype(np.uint8) // 64
    )

    colour_index = (
        quantized[:, :, 0] * 16
        +
        quantized[:, :, 1] * 4
        +
        quantized[:, :, 2]
    )

    histogram = np.bincount(
        colour_index.flatten(),
        minlength=64
    ).astype(
        np.float32
    )

    histogram /= (
        np.sum(histogram) +
        1e-8
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

        "saturation":
            saturation,

        "red_ratio":
            red_ratio,

        "red_dominance":
            red_dominance,

        "histogram":
            histogram.tolist()
    }


# ============================================================
# AVERAGE FEATURES
# ============================================================

def average_features(items):

    keys = [
        "red",
        "green",
        "blue",
        "brightness",
        "saturation",
        "red_ratio",
        "red_dominance"
    ]

    result = {}

    for key in keys:

        result[key] = float(
            np.mean(
                [
                    item[key]
                    for item in items
                ]
            )
        )

    histograms = np.asarray(
        [
            item["histogram"]
            for item in items
        ],
        dtype=np.float32
    )

    average_histogram = np.mean(
        histograms,
        axis=0
    )

    average_histogram /= (
        np.sum(
            average_histogram
        )
        +
        1e-8
    )

    result["histogram"] = (
        average_histogram.tolist()
    )

    result["image_count"] = len(
        items
    )

    return result


# ============================================================
# BUILD DATASET COLOUR PROFILES
# ============================================================

def build_colour_profiles():

    print()
    print(
        "🎨 Building dataset colour profiles..."
    )

    if not os.path.exists(
        TRAIN_DIR
    ):

        print(
            "ERROR: Training directory not found:"
        )

        print(
            TRAIN_DIR
        )

        return

    profiles = {}

    class_directories = [
        path
        for path in glob.glob(
            os.path.join(
                TRAIN_DIR,
                "*"
            )
        )
        if os.path.isdir(path)
    ]

    for class_dir in class_directories:

        class_name = os.path.basename(
            class_dir
        )

        print(
            "Processing:",
            class_name
        )

        image_paths = []

        for extension in [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.webp"
        ]:

            image_paths.extend(
                glob.glob(
                    os.path.join(
                        class_dir,
                        extension
                    )
                )
            )

        # Limit to 300 images/class.
        # Increase later if required.
        image_paths = image_paths[:300]

        features = []

        for image_path in image_paths:

            try:

                with Image.open(
                    image_path
                ) as image:

                    features.append(
                        extract_colour_features(
                            image
                        )
                    )

            except Exception as error:

                print(
                    "Skipping:",
                    image_path,
                    error
                )

        if features:

            profiles[class_name] = (
                average_features(
                    features
                )
            )

    with open(
        COLOUR_PROFILE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            profiles,
            f,
            indent=2
        )

    print()
    print(
        "Saved:",
        COLOUR_PROFILE_FILE
    )

    print(
        "Profiles:",
        len(profiles)
    )


# ============================================================
# VARIETY DATASET
# ============================================================

def build_variety_profiles():

    if not os.path.exists(
        VARIETY_DATASET_DIR
    ):

        print()
        print(
            "No variety_dataset folder found."
        )

        return

    profiles = {}

    for variety in [
        "Kashmiri",
        "Byadgi",
        "Guntur",
        "Teja"
    ]:

        folder = os.path.join(
            VARIETY_DATASET_DIR,
            variety
        )

        if not os.path.isdir(folder):
            continue

        print(
            "Processing variety:",
            variety
        )

        image_paths = []

        for extension in [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.webp"
        ]:

            image_paths.extend(
                glob.glob(
                    os.path.join(
                        folder,
                        extension
                    )
                )
            )

        image_paths = image_paths[:300]

        features = []

        for image_path in image_paths:

            try:

                with Image.open(
                    image_path
                ) as image:

                    features.append(
                        extract_colour_features(
                            image
                        )
                    )

            except Exception:
                pass

        if features:

            profiles[variety] = (
                average_features(
                    features
                )
            )

    if profiles:

        with open(
            VARIETY_PROFILE_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                profiles,
                f,
                indent=2
            )

        print()
        print(
            "Saved:",
            VARIETY_PROFILE_FILE
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_colour_profiles()

    build_variety_profiles()

    print()
    print(
        "=========================================="
    )

    print(
        "Profile building completed."
    )

    print(
        "Now run: python app.py"
    )

    print(
        "=========================================="
    )