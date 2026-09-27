import tensorflow as tf
import numpy as np
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATASET = r"D:\nandha internship\Redchillypowderprediction\processed_dataset"
TEST_DIR = os.path.join(DATASET, "test")

MODEL_PATH = "chilli_powder_model.keras"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, ":", name)

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

print("\nRunning predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

# --------------------------------------------------
# ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

print("\n======================================")
print("TEST ACCURACY")
print("======================================")

print(f"{accuracy * 100:.2f}%")

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

cm = confusion_matrix(y_true, y_pred)

print(cm)

# --------------------------------------------------
# SAVE CONFUSION MATRIX
# --------------------------------------------------

np.savetxt(
    "confusion_matrix.txt",
    cm,
    fmt="%d"
)

print("\nConfusion matrix saved to:")
print("confusion_matrix.txt")