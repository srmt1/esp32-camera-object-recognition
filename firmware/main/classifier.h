#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool classifier_init(void);

void classifier_predict_jpeg(const uint8_t *jpeg_data, size_t jpeg_length);

#ifdef __cplusplus
}
#endif
