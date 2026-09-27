import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE = r"D:\nandha internship\Redchillypowderprediction\Dataset for Adulterated Red Chilli Powder with Brick Powder"

OUTPUT = r"D:\nandha internship\Redchillypowderprediction\processed_dataset"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Create output folders
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT, split), exist_ok=True)

# Find the 12 class folders
classes = [
    x for x in os.listdir(SOURCE)
    if os.path.isdir(os.path.join(SOURCE, x))
]

print("Classes found:", len(classes))
print()

total = 0

for class_name in sorted(classes):

    class_path = os.path.join(SOURCE, class_name)

    # Find every image inside the class, including subfolders
    images = []

    for root, dirs, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, file))

    print(f"{class_name}: {len(images)} images")

    if len(images) < 3:
        print("Skipping class because it has fewer than 3 images.")
        continue

    # 70% train, 15% validation, 15% test
    train_images, temp_images = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    validation_images, test_images = train_test_split(
        temp_images,
        test_size=0.50,
        random_state=42
    )

    for split, split_images in [
        ("train", train_images),
        ("validation", validation_images),
        ("test", test_images)
    ]:

        destination = os.path.join(
            OUTPUT,
            split,
            class_name
        )

        os.makedirs(destination, exist_ok=True)

        for index, image_path in enumerate(split_images):

            extension = os.path.splitext(image_path)[1]

            new_name = f"{index:05d}{extension}"

            destination_path = os.path.join(
                destination,
                new_name
            )

            shutil.copy2(image_path, destination_path)

    total += len(images)

print()
print("=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)
print("Total images:", total)
print("Output:", OUTPUT)