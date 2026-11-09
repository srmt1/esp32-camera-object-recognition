import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_DIR = Path("processed_dataset")
IMAGE_SIZE = (32, 32)

model = tf.keras.models.load_model("object_classifier.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=1,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names

errors = []

for images, labels in test_ds:
    actual = int(labels.numpy()[0])

    images = images / 255.0

    prediction = model.predict(images, verbose=0)
    predicted = int(tf.argmax(prediction[0]))

    if actual != predicted:
        errors.append((
            images[0].numpy().squeeze(),
            class_names[actual],
            class_names[predicted]
        ))

print(f"\nFound {len(errors)} incorrectly classified images.")

if errors:
    fig, axes = plt.subplots(
        1, len(errors),
        figsize=(3 * len(errors), 4)
    )

    if len(errors) == 1:
        axes = [axes]

    for ax, (image, actual, predicted) in zip(axes, errors):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Actual: {actual}\nPredicted: {predicted}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
