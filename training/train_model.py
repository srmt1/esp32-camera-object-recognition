import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Dataset directories
DATASET_DIR = Path("processed_dataset")

# Image settings
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 16

# Load training images
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

# Load validation images
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names

print("\nClass names:", class_names)

# Normalise pixel values from 0-255 to 0-1
normalisation = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda images, labels: (normalisation(images), labels)
)

val_ds = val_ds.map(
    lambda images, labels: (normalisation(images), labels)
)

# Build CNN
model = models.Sequential([
    layers.Input(shape=(32, 32, 1)),

    layers.Conv2D(8, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(16, activation="relu"),

    layers.Dense(3, activation="softmax")
])

# Configure training
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Display network structure
model.summary()

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

# Save trained model
model.save("object_classifier_32.keras")

print("\nModel saved as object_classifier_32.keras")
