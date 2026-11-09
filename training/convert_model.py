import tensorflow as tf
from pathlib import Path

MODEL_FILE = "object_classifier_32.keras"
DATASET_DIR = "processed_dataset/train"
OUTPUT_FILE = "object_classifier_int8.tflite"

IMAGE_SIZE = (32, 32)


def representative_dataset():
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=1,
        shuffle=False
    )

    print("Representative dataset classes:", dataset.class_names)

    for images, _ in dataset.take(100):
        # Match the normalisation used when training the model.
        images = tf.cast(images, tf.float32) / 255.0
        yield [images.numpy()]


def main():
    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_FILE)

    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Require integer-only operations for microcontroller deployment.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("\nConverting model...")
    tflite_model = converter.convert()

    Path(OUTPUT_FILE).write_bytes(tflite_model)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Model size: {len(tflite_model):,} bytes")

    # Inspect the resulting model's input and output quantisation.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    print("\nInput details:")
    print(interpreter.get_input_details())

    print("\nOutput details:")
    print(interpreter.get_output_details())


if __name__ == "__main__":
    main()
