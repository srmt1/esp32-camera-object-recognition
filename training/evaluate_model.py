import tensorflow as tf
import numpy as np
from pathlib import Path

DATASET_DIR = Path("processed_dataset")
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 16

# Load the saved model
model = tf.keras.models.load_model("object_classifier_32.keras")

# Load test images
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = test_ds.class_names

# Normalise pixels in exactly the same way as training
normalisation = tf.keras.layers.Rescaling(1.0 / 255)
test_ds = test_ds.map(
    lambda images, labels: (normalisation(images), labels)
)

# Run predictions
all_labels = []
all_predictions = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    all_labels.extend(np.argmax(labels.numpy(), axis=1))
    all_predictions.extend(np.argmax(predictions, axis=1))

# Calculate accuracy
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

accuracy = np.mean(all_labels == all_predictions)

print("\nTest accuracy: {:.1f}%".format(accuracy * 100))

# Confusion matrix
matrix = tf.math.confusion_matrix(
    all_labels,
    all_predictions,
    num_classes=len(class_names)
).numpy()

print("\nConfusion matrix:")
print("Rows = actual class, columns = predicted class")
print()

print("          ", end="")
for name in class_names:
    print("{:>8}".format(name), end="")
print()

for i, name in enumerate(class_names):
    print("{:>10}".format(name), end="")
    for value in matrix[i]:
        print("{:>8}".format(value), end="")
    print()

# Individual predictions
print("\nIndividual results:")
for actual, predicted in zip(all_labels, all_predictions):
    result = "CORRECT" if actual == predicted else "INCORRECT"
    print(
        "Actual: {:<8} Predicted: {:<8} {}".format(
            class_names[actual],
            class_names[predicted],
            result
        )
    )
