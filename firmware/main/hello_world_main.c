#include <stdio.h>
#include "classifier.h"
#include "esp_camera.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "esp_http_server.h"
#include "wifi_config.h"

static esp_err_t capture_handler(httpd_req_t *req)
{
    camera_fb_t *fb = esp_camera_fb_get();

    if (fb == NULL)
    {
        ESP_LOGE("HTTP", "Camera capture failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    ESP_LOGI("HTTP", "Captured JPEG: %u bytes", (unsigned int)fb->len);

    httpd_resp_set_type(req, "image/jpeg");

    esp_err_t result = httpd_resp_send(
        req,
        (const char *)fb->buf,
        fb->len
    );

    esp_camera_fb_return(fb);

    return result;
}

static void start_webserver(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    httpd_handle_t server = NULL;

    if (httpd_start(&server, &config) == ESP_OK)
    {
        httpd_uri_t capture_uri = {
            .uri = "/capture",
            .method = HTTP_GET,
            .handler = capture_handler,
            .user_ctx = NULL
        };

        httpd_register_uri_handler(server, &capture_uri);

        ESP_LOGI("HTTP", "Web server started");
        ESP_LOGI("HTTP", "Capture URL: http://192.168.4.1/capture");
    }
    else
    {
        ESP_LOGE("HTTP", "Failed to start web server");
    }
}

static void wifi_init_softap(void)
{
    // Initialise non-volatile storage
    esp_err_t ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }

    ESP_ERROR_CHECK(ret);

    // Initialise network interface and event loop
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // Create default Wi-Fi access point interface
    esp_netif_create_default_wifi_ap();

    // Initialise Wi-Fi
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // Configure access point
    wifi_config_t wifi_config = {
        .ap = {
            .ssid = "ESP32-Camera",
            .ssid_len = 0,
            .channel = 1,
            .password = WIFI_AP_PASSWORD,
            .max_connection = 1,
            .authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {
                .required = false
            }
        }
    };

    // Set access-point mode and configuration
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));

    ESP_ERROR_CHECK(
        esp_wifi_set_config(WIFI_IF_AP, &wifi_config)
    );

    // Start Wi-Fi
    ESP_ERROR_CHECK(esp_wifi_start());

    printf("\nWi-Fi access point started\n");
    printf("SSID: ESP32-Camera\n");
    printf("IP address: 192.168.4.1\n");
}

void app_main(void)
{
    printf("\nESP32 OV2640 camera test\n");
    printf("========================\n");

    camera_config_t config = {
        .pin_pwdn       = 32,
        .pin_reset     = 33,
        .pin_xclk      = 0,
        .pin_sccb_sda  = 26,
        .pin_sccb_scl  = 27,

        .pin_d7        = 35,
        .pin_d6        = 34,
        .pin_d5        = 39,
        .pin_d4        = 36,
        .pin_d3        = 21,
        .pin_d2        = 19,
        .pin_d1        = 18,
        .pin_d0        = 5,

        .pin_vsync     = 25,
        .pin_href      = 23,
        .pin_pclk      = 22,

        .xclk_freq_hz  = 20000000,
        .ledc_timer    = LEDC_TIMER_0,
        .ledc_channel  = LEDC_CHANNEL_0,

        .pixel_format  = PIXFORMAT_JPEG,
        .frame_size    = FRAMESIZE_QVGA,
        .jpeg_quality  = 12,

        .fb_count      = 1,
        .fb_location   = CAMERA_FB_IN_PSRAM,
        .grab_mode     = CAMERA_GRAB_WHEN_EMPTY,

        .sccb_i2c_port  = 0
    };

    printf("\nStarting Wi-Fi...\n");
    wifi_init_softap();

    printf("Initialising camera...\n");

    esp_err_t err = esp_camera_init(&config);

    if (err != ESP_OK) {
        printf("Camera initialisation failed: 0x%x\n", err);
        return;
    }

    printf("Camera initialised successfully.\n");
    printf("\nInitialising object classifier...\n");

    bool classifier_ready = classifier_init();

    if (classifier_ready) {
        printf("Object classifier ready.\n");
    } else {
        printf("Object classifier initialisation failed.\n");
        printf("Camera will continue without classification.\n");
    }
    printf("\nStarting HTTP server...\n");
    start_webserver();

    sensor_t *sensor = esp_camera_sensor_get();

    if (sensor != NULL) {
        printf("Sensor PID: 0x%04x\n", sensor->id.PID);
    }

    while (1)
    {
        printf("Capturing image...\n");

        camera_fb_t *fb = esp_camera_fb_get();

        if (fb == NULL)
        {
            printf("Camera capture failed.\n");
        }
        else
        {
            printf("Image captured: %u x %u, %u bytes\n",
           (unsigned int)fb->width,
           (unsigned int)fb->height,
           (unsigned int)fb->len);

            if (classifier_ready) {
                classifier_predict_jpeg(fb->buf, fb->len);
            }

            esp_camera_fb_return(fb);
	}

        vTaskDelay(pdMS_TO_TICKS(2000));
    }

}
