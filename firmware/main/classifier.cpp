#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include "esp_log.h"
#include "esp_heap_caps.h"

#include "esp_camera.h"
#include "img_converters.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#include "model_data.h"
#include "classifier.h"

static const char *TAG = "CLASSIFIER";

static constexpr int IMAGE_WIDTH = 320;
static constexpr int IMAGE_HEIGHT = 240;
static constexpr int MODEL_WIDTH = 32;
static constexpr int MODEL_HEIGHT = 32;

static constexpr size_t RGB_BUFFER_SIZE =
    IMAGE_WIDTH * IMAGE_HEIGHT * 3;

static constexpr size_t TENSOR_ARENA_SIZE = 150 * 1024;

static uint8_t *rgb_buffer = nullptr;
static uint8_t *tensor_arena = nullptr;

static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *model_input = nullptr;
static TfLiteTensor *model_output = nullptr;

static constexpr const char *class_names[] = {
    "cup",
    "phone",
    "spoon"
};

/* Convert one RGB pixel to grayscale, matching Pillow's L conversion. */
static uint8_t rgb_to_gray(int x, int y)
{
    const size_t offset = (y * IMAGE_WIDTH + x) * 3;

    const uint8_t r = rgb_buffer[offset];
    const uint8_t g = rgb_buffer[offset + 1];
    const uint8_t b = rgb_buffer[offset + 2];

    return static_cast<uint8_t>(
        (299 * r + 587 * g + 114 * b + 500) / 1000
    );
}

/* Bilinear resize from 320x240 to 32x32 grayscale. */
static void prepare_input()
{
    for (int dy = 0; dy < MODEL_HEIGHT; ++dy) {

        float sy = (dy + 0.5f) * IMAGE_HEIGHT / MODEL_HEIGHT - 0.5f;

        if (sy < 0) sy = 0;

        int y0 = static_cast<int>(sy);
        int y1 = y0 + 1;

        if (y1 >= IMAGE_HEIGHT) y1 = IMAGE_HEIGHT - 1;

        float fy = sy - y0;

        for (int dx = 0; dx < MODEL_WIDTH; ++dx) {

            float sx = (dx + 0.5f) * IMAGE_WIDTH / MODEL_WIDTH - 0.5f;

            if (sx < 0) sx = 0;

            int x0 = static_cast<int>(sx);
            int x1 = x0 + 1;

            if (x1 >= IMAGE_WIDTH) x1 = IMAGE_WIDTH - 1;

            float fx = sx - x0;

            const float p00 = rgb_to_gray(x0, y0);
            const float p01 = rgb_to_gray(x1, y0);
            const float p10 = rgb_to_gray(x0, y1);
            const float p11 = rgb_to_gray(x1, y1);

            const float top = p00 + fx * (p01 - p00);
            const float bottom = p10 + fx * (p11 - p10);

            const uint8_t gray =
                static_cast<uint8_t>(top + fy * (bottom - top) + 0.5f);

            const int index = dy * MODEL_WIDTH + dx;

            // Input quantisation: scale = 1/255, zero point = -128.
            model_input->data.int8[index] =
                static_cast<int8_t>(static_cast<int>(gray) - 128);
        }
    }
}

extern "C" bool classifier_init(void)
{
    ESP_LOGI(TAG, "Initialising TensorFlow Lite Micro");

    const tflite::Model *model = tflite::GetModel(g_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch");
        return false;
    }

    // Allocate working memory in PSRAM.
    rgb_buffer = static_cast<uint8_t *>(
        heap_caps_malloc(RGB_BUFFER_SIZE,
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    );

    tensor_arena = static_cast<uint8_t *>(
        heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE,
                                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    );

    if (rgb_buffer == nullptr || tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Could not allocate classifier memory");
        return false;
    }

    static tflite::MicroMutableOpResolver<5> resolver;

    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) {
        ESP_LOGE(TAG, "Could not register model operators");
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return false;
    }

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    ESP_LOGI(TAG, "Input tensor: %d x %d x %d",
             model_input->dims->data[1],
             model_input->dims->data[2],
             model_input->dims->data[3]);

    ESP_LOGI(TAG, "Classifier initialised successfully");

    return true;
}

extern "C" void classifier_predict_jpeg(
    const uint8_t *jpeg_data,
    size_t jpeg_length)
{
    if (interpreter == nullptr || jpeg_data == nullptr) {
        return;
    }

    // Decode the camera JPEG into RGB888.
    if (!fmt2rgb888(jpeg_data, jpeg_length, PIXFORMAT_JPEG, rgb_buffer)) {
        ESP_LOGE(TAG, "JPEG to RGB conversion failed");
        return;
    }

    prepare_input();

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Model inference failed");
        return;
    }

    int best_index = 0;
    int8_t best_value = model_output->data.int8[0];

    for (int i = 1; i < 3; ++i) {
        if (model_output->data.int8[i] > best_value) {
            best_value = model_output->data.int8[i];
            best_index = i;
        }
    }

    const float confidence =
        (best_value - model_output->params.zero_point) *
        model_output->params.scale;

    ESP_LOGI(TAG, "Prediction: %s (%.1f%%)",
             class_names[best_index],
             confidence * 100.0f);
}
