import requests
from PIL import Image
from io import BytesIO
from pathlib import Path

CAMERA_URL = "http://192.168.4.1/capture"

DATASET_DIR = Path("dataset")


def capture_image():
    response = requests.get(CAMERA_URL, timeout=10)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))
    image.load()

    return image


def next_filename(class_name):
    class_dir = DATASET_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Find the highest existing image number
    numbers = []

    for path in class_dir.glob(f"{class_name}_*.jpg"):
        try:
            number = int(path.stem.rsplit("_", 1)[1])
            numbers.append(number)
        except (ValueError, IndexError):
            continue

    next_number = max(numbers, default=0) + 1

    return class_dir / f"{class_name}_{next_number:03d}.jpg"


def main():
    print("ESP32 Object Recognition - Image Capture")
    print("----------------------------------------")

    while True:
        class_name = input("\nEnter object class (or 'quit'): ").strip().lower()

        if class_name == "quit":
            break

        if not class_name.isalnum():
            print("Use letters and numbers only.")
            continue

        print(f"\nPreparing to capture: {class_name}")
        print("Position your object in front of the camera.")

        while True:
            try:
                command = input(
                    "\nPress ENTER to capture, "
                    "type 'next' to change class, or 'quit' to exit: "
                ).strip().lower()

                if command == "quit":
                    return

                if command == "next":
                    break

                if command:
                    print("Press ENTER, type 'next', or type 'quit'.")
                    continue

                image = capture_image()

                filename = next_filename(class_name)
                image.save(filename, "JPEG", quality=95)

                print(f"Saved: {filename}")
                print(f"Image dimensions: {image.size}")

                image.show()

            except requests.RequestException as error:
                print(f"Camera communication error: {error}")

            except Exception as error:
                print(f"Capture error: {error}")


if __name__ == "__main__":
    main()
