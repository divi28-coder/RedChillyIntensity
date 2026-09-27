import os

# Disable TensorFlow informational/warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
import numpy as np
from PIL import Image

import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "chilli_powder_model.keras"
IMG_SIZE = (224, 224)


# ============================================================
# CLASS NAMES
# MUST MATCH THE TRAINING CLASS ORDER
# ============================================================

CLASS_NAMES = [
    "Brick powder normal sample",
    "Red chilli powder normal Sample",
    "Red chilli powder with 10_ brick powder",
    "Red chilli powder with 15_ brick powder",
    "Red chilli powder with 20_ brick powder",
    "Red chilli powder with 25_ brick powder",
    "Red chilli powder with 30_ brick powder",
    "Red chilli powder with 35_ brick powder",
    "Red chilli powder with 40_ brick powder",
    "Red chilli powder with 45_ brick powder",
    "Red chilli powder with 50_ brick powder",
    "Red chilli powder with 5_ brick powder"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("ERROR: Could not load model.")
    print(e)
    exit()


# ============================================================
# ASK FOR IMAGE PATH
# ============================================================

image_path = input("\nEnter image path: ").strip().strip('"')


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.isfile(image_path):
    print("\nERROR: Image not found.")
    print("Path checked:")
    print(image_path)
    exit()

print("Image found successfully!")


# ============================================================
# LOAD IMAGE
# ============================================================

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print("\nERROR: Could not open image.")
    print(e)
    exit()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

image = image.resize(IMG_SIZE)

image_array = np.array(image, dtype=np.float32)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning prediction...")

try:
    predictions = model.predict(image_array, verbose=0)

    print("Prediction completed.")
    print("Prediction shape:", predictions.shape)

except Exception as e:
    print("\nERROR during prediction:")
    print(e)
    exit()


# Remove batch dimension
predictions = predictions[0]

print("\nNumber of predictions:", len(predictions))


# ============================================================
# FIND BEST CLASS
# ============================================================

predicted_index = int(np.argmax(predictions))

predicted_class = CLASS_NAMES[predicted_index]

confidence = float(predictions[predicted_index]) * 100


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n========================================")
print("       CHILLI POWDER PREDICTION")
print("========================================")

print("Predicted class :", predicted_class)
print(f"Confidence      : {confidence:.2f}%")

print("========================================")


# ============================================================
# ALL CLASS PREDICTIONS
# ============================================================

print("\nAll predictions:")
# ============================================================
# DEBUG: SHOW RAW MODEL OUTPUT
# ============================================================

print("\nRAW MODEL OUTPUT:")
print(predictions)

print("\nPrediction values:")
for i in range(len(predictions)):
    print(i, predictions[i])


# ============================================================
# FIND BEST CLASS
# ============================================================

predicted_index = int(np.argmax(predictions))

predicted_class = CLASS_NAMES[predicted_index]

confidence = float(predictions[predicted_index]) * 100


# ============================================================
# FINAL RESULT
# ============================================================

print("\n========================================")
print("       CHILLI POWDER PREDICTION")
print("========================================")

print("Predicted class :", predicted_class)
print(f"Confidence      : {confidence:.2f}%")

print("========================================")


# ============================================================
# ALL CLASS PREDICTIONS
# ============================================================

print("\nAll class predictions:")
print("----------------------------------------")

for i in range(len(predictions)):
    probability = float(predictions[i]) * 100

    print(
        f"{CLASS_NAMES[i]:55s} "
        f"{probability:.2f}%"
    )

print("----------------------------------------")