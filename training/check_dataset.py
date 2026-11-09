from pathlib import Path
from PIL import Image

dataset_dir = Path("dataset")

for class_dir in sorted(dataset_dir.iterdir()):
    if not class_dir.is_dir():
        continue

    dimensions = {}
    corrupted = []

    for image_path in class_dir.glob("*"):
        try:
            with Image.open(image_path) as image:
                dimensions[image.size] = dimensions.get(image.size, 0) + 1
                image.verify()

        except Exception:
            corrupted.append(image_path.name)

    print(f"\nClass: {class_dir.name}")
    print(f"Images: {sum(dimensions.values()) + len(corrupted)}")
    print("Dimensions:")

    for size, count in sorted(dimensions.items()):
        print(f"  {size}: {count} images")

    print(f"Corrupted images: {len(corrupted)}")

    if corrupted:
        print("Files:", corrupted)
