# ESP32 Camera Object Recognition

An embedded machine-learning project using an ESP32, an OV2640 camera and TensorFlow Lite Micro to recognise objects directly on the microcontroller.

The current implementation classifies camera images into three categories:

- Cup
- Phone
- Spoon

The project demonstrates how to train a small convolutional neural network (CNN) on a PC, quantise it, deploy it to an ESP32 and perform image inference locally without requiring a cloud service.

## 1. Project overview

The ESP32 captures images using the OV2640 camera and runs a trained INT8 neural network using TensorFlow Lite Micro.

The processing pipeline is:

```text
             OV2640 Camera
                   |
                   v
             JPEG Capture
                   |
                   v
             JPEG Decoding
                   |
                   v
             RGB888 Image
                   |
                   v
           RGB to Greyscale
                   |
                   v
            Resize to 32x32
                   |
                   v
             INT8 Quantisation
                   |
                   v
          TensorFlow Lite Micro
                   |
                   v
              CNN Inference
                   |
                   v
          cup / phone / spoon
```

Inference takes place entirely on the ESP32. The PC is only required for model training and firmware development.

## 2. Hardware

| Component | Description |
|---|---|
| Microcontroller | ESP32, dual-core, up to 240 MHz |
| Camera | OV2640 |
| External memory | PSRAM |
| USB interface | CP210x USB-to-UART |
| Development host | Ubuntu 20.04 |

The camera is configured to capture JPEG images at QVGA resolution (320 × 240).

### Camera GPIO configuration

| Camera signal | ESP32 GPIO |
|---|---:|
| Y2 / D0 | 5 |
| Y3 / D1 | 18 |
| Y4 / D2 | 19 |
| Y5 / D3 | 21 |
| Y6 / D4 | 36 |
| Y7 / D5 | 39 |
| Y8 / D6 | 34 |
| Y9 / D7 | 35 |
| XCLK | 0 |
| PCLK | 22 |
| VSYNC | 25 |
| HREF | 23 |
| SIOD / SDA | 26 |
| SIOC / SCL | 27 |
| PWDN | 32 |
| RESET | 33 |

## 3. Software

| Software | Version / description |
|---|---|
| Operating system | Ubuntu 20.04 |
| ESP-IDF | 5.2.3 |
| ESP32 camera component | Espressif `esp32-camera` |
| ML framework | TensorFlow 2.13.1 (PC training environment) |
| Embedded inference | TensorFlow Lite Micro |
| Model format | TensorFlow Lite, INT8 quantised |
| Programming languages | C and C++ |

ESP-IDF managed components are used for the camera and TensorFlow Lite Micro dependencies.

## 4. Machine-learning model

The model is a small convolutional neural network trained to distinguish between cups, phones and spoons.

### Dataset

The initial dataset contains 832 images:

| Class | Number of images |
|---|---:|
| Cup | 304 |
| Phone | 233 |
| Spoon | 295 |
| **Total** | **832** |

The dataset was divided into training, validation and test sets:

| Dataset | Images |
|---|---:|
| Training | 581 |
| Validation | 123 |
| Testing | 128 |

### CNN architecture

The network accepts a 32 × 32 greyscale image.

| Layer | Configuration |
|---|---|
| Input | 32 × 32 × 1 |
| Conv2D | 8 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 16 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | 576 features |
| Dense | 16 neurons, ReLU |
| Output | 3 neurons, Softmax |

The model has approximately 10,531 trainable parameters.

### Model performance

The model achieved approximately 93% accuracy on the held-out test set (119 correct predictions out of 128).

This is an initial experimental result. Performance on images captured by the ESP32 can differ from the test-set result depending on lighting, object position, background and image preprocessing.

### Quantisation

The trained TensorFlow model was converted to an INT8 TensorFlow Lite model.

The deployed model has:

- Input shape: `[1, 32, 32, 1]`
- Input data type: `int8`
- Output shape: `[1, 3]`
- Output data type: `int8`
- Model size: approximately 15 KB

INT8 quantisation reduces model storage and computational requirements compared with a floating-point model.

## 5. Firmware architecture

The firmware separates camera/application management from machine-learning inference.

### `hello_world_main.c`

Main application responsibilities:

- Initialise the camera.
- Configure the ESP32 Wi-Fi access point.
- Start the HTTP server.
- Capture images periodically.
- Pass captured JPEG images to the classifier.
- Return camera framebuffers to the camera driver.

### `classifier.h`

Declares the classifier interface:

```c
bool classifier_init(void);

void classifier_predict_jpeg(
    const uint8_t *jpeg_data,
    size_t jpeg_length
);
```

The header uses `extern "C"` guards so the C application can call functions implemented in C++.

### `classifier.cpp`

Implements the embedded inference pipeline:

1. Initialise TensorFlow Lite Micro.
2. Allocate RGB and tensor-arena memory in PSRAM.
3. Register the model's required operators.
4. Decode camera JPEG images into RGB888.
5. Convert RGB pixels to greyscale.
6. Resize images to 32 × 32 using bilinear interpolation.
7. Quantise image pixels to INT8.
8. Execute the neural network.
9. Select the highest-scoring class and report its confidence.

### `model_data.h`

Contains the trained INT8 TensorFlow Lite model as a C/C++ byte array.

This allows the model to be compiled directly into the ESP32 firmware.

## 6. Wi-Fi and image capture

The ESP32 starts a Wi-Fi access point.

Default network address:

```text
192.168.4.1
```

The HTTP server exposes a capture endpoint:

```text
http://192.168.4.1/capture
```

This returns a JPEG image from the camera.

The classifier also processes captured images locally, independently of the HTTP capture endpoint.

Do not commit Wi-Fi passwords or other credentials to the repository. Configure them locally where required.

## 7. Building the firmware

### Requirements

- Ubuntu 20.04
- ESP-IDF 5.2.3
- ESP32 development board with OV2640 camera
- PSRAM enabled and available

### Navigate to the project

```bash
cd ~/esp/esp32-camera-test
```

### Select the ESP32 target

```bash
idf.py set-target esp32
```

### Configure the project

```bash
idf.py menuconfig
```

The partition table must provide sufficient space for the application. This project uses the **Single factory app (large), no OTA** partition layout.

Check that the camera and PSRAM configuration is appropriate for the board.


### Activate ESP-IDF

```bash
. ~/esp/esp-idf/export.sh
```

### Build

```bash
idf.py build
```

### Flash and monitor

The USB serial device used during development is `/dev/ttyUSB0`.

```bash
idf.py -p /dev/ttyUSB0 -b 115200 flash monitor
```

## 8. Expected output

After successful initialisation, the serial monitor should report that the classifier has been initialised.

During operation, predictions appear approximately every two seconds:

```text
Image captured: 320 x 240, 4879 bytes
CLASSIFIER: Prediction: spoon (98.8%)
```

The confidence value is the model's Softmax output converted from INT8 to a floating-point value. It should not be interpreted as a guarantee that the prediction is correct.

## 9. Current limitations

This is an image-classification project, not yet an object-detection system.

The current model:

- Predicts one class for the entire image.
- Does not identify the object's position.
- Does not produce bounding boxes.
- Is limited to the three trained classes.
- Uses a low-resolution 32 × 32 greyscale input.

If an image contains several objects, the classifier still returns only one class.

## 10. Planned development

Potential future improvements:

- Expand and improve the training dataset.
- Evaluate performance under different lighting and backgrounds.
- Compare PC and ESP32 inference results.
- Optimise image preprocessing and inference time.
- Experiment with object detection and bounding-box prediction.
- Evaluate the memory and performance constraints of more complex models on the ESP32.

## 11. Learning objectives

This project is also intended as a practical embedded-systems learning exercise, covering:

- ESP32 development using ESP-IDF.
- Camera interfacing and image formats.
- C and C++ interoperability.
- Pointer handling and dynamic memory allocation.
- PSRAM and memory management.
- CNN architecture and inference.
- TensorFlow Lite Micro.
- INT8 quantisation.
- Embedded machine-learning performance constraints.
