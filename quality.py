import os
from PIL import Image
from collections import Counter

DATASET_DIR = "processed_dataset"

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}

def check_dataset(folder):
    print("\n" + "=" * 70)
    print("DATASET QUALITY CHECK")
    print("=" * 70)

    if not os.path.exists(folder):
        print("Dataset folder not found:", folder)
        return

    total = 0
    valid = 0
    corrupted = 0
    low_resolution = 0

    class_counts = Counter()

    for root, dirs, files in os.walk(folder):

        image_files = [
            f for f in files
            if os.path.splitext(f)[1].lower()
            in VALID_EXTENSIONS
        ]

        if not image_files:
            continue

        class_name = os.path.basename(root)

        for filename in image_files:

            total += 1
            class_counts[class_name] += 1

            path = os.path.join(root, filename)

            try:
                with Image.open(path) as img:

                    img.verify()

                # Reopen after verify()
                with Image.open(path) as img:

                    width, height = img.size

                    if width < 224 or height < 224:
                        low_resolution += 1

                    valid += 1

            except Exception:
                corrupted += 1

    print("\nDATASET SUMMARY")
    print("-" * 70)

    print("Total images       :", total)
    print("Valid images       :", valid)
    print("Corrupted images   :", corrupted)
    print("Low-resolution     :", low_resolution)

    print("\nIMAGES PER CLASS")
    print("-" * 70)

    for class_name, count in sorted(
        class_counts.items()
    ):
        print(
            f"{class_name:<35} {count:>5}"
        )

    print("\nQUALITY ASSESSMENT")
    print("-" * 70)

    if total == 0:
        print("❌ No images found.")

    elif corrupted > 0:
        print(
            "⚠️ Remove or replace corrupted images."
        )

    elif low_resolution > total * 0.30:
        print(
            "⚠️ Many images have low resolution."
        )

    elif total < 100:
        print(
            "⚠️ Dataset is quite small."
        )

    else:
        print(
            "✅ Basic dataset quality looks acceptable."
        )

    print("\nRECOMMENDATION")
    print("-" * 70)

    for class_name, count in sorted(
        class_counts.items()
    ):

        if count < 30:
            print(
                f"⚠️ {class_name}: only {count} images"
            )

        elif count < 100:
            print(
                f"🟡 {class_name}: {count} images"
            )

        else:
            print(
                f"🟢 {class_name}: {count} images"
            )


check_dataset(DATASET_DIR)