# ESP32 Camera Object Recognition

An embedded machine-learning project using an ESP32, an OV2640 camera and TensorFlow Lite Micro to recognise objects directly on a microcontroller.

The project covers the complete embedded AI workflow: image collection, dataset preparation, neural-network training, model quantisation and deployment to an ESP32.

## Project Overview

The system recognises three object categories:

- Cup
- Phone
- Spoon

The neural network is trained on a PC using TensorFlow/Keras, converted to an INT8 TensorFlow Lite model, and deployed to the ESP32 using TensorFlow Lite Micro.

Image classification runs locally on the ESP32. No cloud inference service is required.

### Processing Pipeline

1. Capture images using the ESP32 camera.
2. Transfer images to a PC and organise the dataset.
3. Prepare training, validation and test datasets.
4. Train a convolutional neural network (CNN).
5. Evaluate the trained model.
6. Convert the model to fully quantised INT8 TensorFlow Lite.
7. Deploy the model to the ESP32.
8. Capture images and perform inference directly on the microcontroller.

## Repository Structure

```text
esp32-camera-object-recognition/
│
├── README.md
├── .gitignore
│
├── firmware/
│   ├── CMakeLists.txt
│   ├── dependencies.lock
│   └── main/
│       ├── hello_world_main.c
│       ├── classifier.cpp
│       ├── classifier.h
│       ├── model_data.h
│       ├── object_classifier_int8.tflite
│       └── ...
│
└── training/
    ├── capture_images.py
    ├── check_dataset.py
    ├── prepare_dataset.py
    ├── train_model.py
    ├── evaluate_model.py
    ├── inspect_errors.py
    ├── convert_model.py
    ├── requirements.txt
    ├── dataset/
    └── object_classifier_32.keras
```

## Hardware

- ESP32 development board
- OV2640 camera module
- Ubuntu 20.04 development PC

The camera captures JPEG images at 320 × 240 resolution.

## Software

- Ubuntu 20.04
- ESP-IDF 5.2.3
- TensorFlow 2.13.1
- TensorFlow Lite
- TensorFlow Lite Micro
- Python 3
- C and C++

## Dataset

The initial dataset contains 832 images:

| Category | Images |
|---|---:|
| Cup | 304 |
| Phone | 233 |
| Spoon | 295 |
| **Total** | **832** |

Dataset split:

| Subset | Images |
|---|---:|
| Training | 581 |
| Validation | 123 |
| Testing | 128 |

Images are converted to grayscale and resized to 32 × 32 pixels before being passed to the network.

## Neural Network

The current classifier uses a small convolutional neural network:

| Layer | Configuration |
|---|---|
| Input | 32 × 32 × 1 |
| Conv2D | 8 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 16 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | Flatten feature maps |
| Dense | 16 neurons, ReLU |
| Output | 3 neurons, Softmax |

The model contains approximately 10,531 trainable parameters.

### Model Performance

The 32 × 32 model achieved approximately 93% accuracy on the held-out test dataset (119 correct predictions out of 128).

### Model Quantisation

The trained model is converted to a fully quantised INT8 TensorFlow Lite model.

- Model input: 32 × 32 × 1
- Input type: INT8
- Output type: INT8
- Model size: approximately 15 KB

The quantised model is embedded into the ESP32 firmware.

## Training

The `training/` directory contains the Python scripts for image capture, dataset preparation, training, evaluation and model conversion.

Create a Python virtual environment:

```bash
cd training

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The main scripts are:

| Script | Purpose |
|---|---|
| `capture_images.py` | Capture images from the ESP32 |
| `check_dataset.py` | Check dataset contents |
| `prepare_dataset.py` | Prepare training data |
| `train_model.py` | Train the CNN |
| `evaluate_model.py` | Evaluate model performance |
| `inspect_errors.py` | Inspect misclassified images |
| `convert_model.py` | Convert and quantise the model |

## ESP32 Firmware

The `firmware/` directory contains the ESP-IDF application.

Main components:

- `hello_world_main.c`: Camera initialisation, Wi-Fi access point, HTTP server and main application loop.
- `classifier.cpp`: Image preprocessing and TensorFlow Lite Micro inference.
- `classifier.h`: C-compatible classifier interface.
- `model_data.h`: Embedded INT8 model data.

### Camera Configuration

The camera uses an OV2640 sensor with the following configuration:

- Resolution: QVGA (320 × 240)
- Pixel format: JPEG
- XCLK: 20 MHz
- Frame buffer: PSRAM

### Wi-Fi and HTTP

The ESP32 starts a Wi-Fi access point:

- SSID: `ESP32-Camera`
- IP address: `192.168.4.1`

An HTTP endpoint provides camera images:

```text
http://192.168.4.1/capture
```

The classifier independently processes captured camera frames on the ESP32.

### Building and Flashing

Install ESP-IDF and activate the environment:

```bash
. ~/esp/esp-idf/export.sh
```

### ESP-IDF Configuration

The firmware requires an ESP32 board with external PSRAM and at least 4 MB of flash.

The project includes `firmware/sdkconfig.defaults`, which configures:

- 4 MB flash
- Large single-application partition (no OTA)
- Quad SPI PSRAM
- PSRAM operating at 40 MHz
- PSRAM initialisation at startup
- PSRAM allocation through standard malloc()

When building for the first time, ESP-IDF uses these defaults to generate
the project's `sdkconfig` configuration.

Build the firmware:

```bash
cd firmware
idf.py set-target esp32
idf.py build
```

Create a local Wi-Fi configuration file from the example:

```bash
cp main/wifi_config.example.h main/wifi_config.h
vim main/wifi_config.h
```

Set your own access-point password in `wifi_config.h`.

Flash and monitor:

```bash
idf.py -p /dev/ttyUSB0 -b 115200 flash monitor
```

## Current Limitations

- Classification only: the system predicts one class for the entire image.
- No object localisation or bounding boxes.
- Currently supports three object categories.
- Uses 32 × 32 grayscale input images.
- Classification performance depends on lighting, camera positioning and training data.

## Future Development

- Investigate object detection and localisation.
- Expand the number of recognised object categories.
- Experiment with different CNN architectures and input resolutions.
- Optimise inference speed and memory usage.
- Display classification results on an external display.
- Investigate more powerful embedded AI hardware.

## Learning Objectives

This project provides practical experience with:

- Embedded machine learning and TinyML.
- CNN architecture and image classification.
- Dataset collection and preparation.
- Model training, evaluation and quantisation.
- TensorFlow Lite Micro.
- ESP-IDF and ESP32 development.
- Camera interfaces and image processing.
- Memory allocation and optimisation on embedded hardware.
- C/C++ integration with embedded AI libraries.
