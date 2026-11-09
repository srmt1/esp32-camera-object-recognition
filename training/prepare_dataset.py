from pathlib import Path
from PIL import Image
import random
import shutil

SOURCE_DIR = Path("dataset")
OUTPUT_DIR = Path("processed_dataset")

IMAGE_SIZE = (32, 32)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

splits = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}

# Remove previous processed dataset
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for class_dir in sorted(SOURCE_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.jpg"))
    random.shuffle(images)

    total = len(images)

    train_end = int(total * splits["train"])
    val_end = train_end + int(total * splits["validation"])

    split_images = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    print(f"\nClass: {class_dir.name}")

    for split_name, split_files in split_images.items():

        output_class_dir = OUTPUT_DIR / split_name / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_files:

            with Image.open(image_path) as image:
                image = image.convert("L")
                image = image.resize(IMAGE_SIZE)

                output_path = output_class_dir / (
                    image_path.stem + ".png"
                )

                image.save(output_path)

        print(f"  {split_name}: {len(split_files)} images")

print("\nDataset preparation complete.")
