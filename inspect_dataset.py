import os

DATASET_PATH = r"D:\nandha internship\Redchillypowderprediction\Dataset for Adulterated Red Chilli Powder with Brick Powder"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("\nIMAGE COUNT BY CLASS")
print("=" * 80)

total_images = 0
classes = 0

for class_name in sorted(os.listdir(DATASET_PATH)):

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    count = 0

    for root, dirs, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                count += 1

    print(f"{class_name}: {count}")

    total_images += count
    classes += 1

print("=" * 80)
print("TOTAL IMAGES:", total_images)
print("TOTAL CLASSES:", classes)