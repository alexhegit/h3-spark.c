#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static uint16_t f32_to_bf16(float value) {
    union {
        uint32_t word;
        float as_float;
    } converted;
    converted.as_float = value;
    converted.word += 0x7fffu + ((converted.word >> 16u) & 1u);
    return (uint16_t)(converted.word >> 16u);
}

static float bf16_to_f32(uint16_t bits) {
    union {
        uint32_t word;
        float as_float;
    } converted;
    converted.word = (uint32_t)bits << 16u;
    return converted.as_float;
}

static void check(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message);
    failures++;
}

static float silu_ref(float value) {
    return value / (1.0f + expf(-value));
}

static void rms_norm_ref(const float *input, const float *weight, float *output,
                         uint32_t rows, uint32_t width, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *row_input = input + (size_t)row * width;
        float *row_output = output + (size_t)row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float value = row_input[column];
            sum = fmaf(value, value, sum);
        }
        float inverse = 1.0f / sqrtf(sum / (float)width + epsilon);
        for (uint32_t column = 0; column < width; column++) {
            row_output[column] =
                row_input[column] * inverse * weight[column];
        }
    }
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    check(gpu != NULL, "h3_gpu_create");
    if (!gpu) {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }

    check(h3_gpu_begin(gpu), "h3_gpu_begin");

    const uint32_t silu_count = 512;
    float silu_host[512];
    uint16_t silu_in_bf16[512];
    uint16_t silu_out_bf16[512];
    for (uint32_t i = 0; i < silu_count; i++) {
        silu_host[i] = -3.0f + (float)i * (6.0f / (float)(silu_count - 1));
        silu_in_bf16[i] = f32_to_bf16(silu_host[i]);
    }

    h3_gpu_tensor *silu_in =
        h3_gpu_tensor_from_bf16(gpu, silu_in_bf16, silu_count);
    h3_gpu_tensor *silu_out = h3_gpu_tensor_new_bf16(gpu, silu_count);
    check(silu_in && silu_out, "silu tensor alloc");
    if (silu_in && silu_out) {
        check(h3_gpu_silu_bf16(gpu, silu_out, silu_in, silu_count), "silu");
        check(h3_gpu_submit(gpu), "submit silu");
        check(h3_gpu_tensor_read_bf16(silu_out, silu_out_bf16, silu_count),
              "read silu");
        for (uint32_t i = 0; i < silu_count; i++) {
            float got = bf16_to_f32(silu_out_bf16[i]);
            float expected = silu_ref(silu_host[i]);
            if (fabsf(got - expected) >= 2e-2f) {
                fprintf(stderr,
                        "FAIL: silu mismatch at %u got=%f expected=%f\n", i, got,
                        expected);
                failures++;
                break;
            }
        }
    }

    const uint32_t rows = 4;
    const uint32_t width = 128;
    const float epsilon = 1e-5f;
    const size_t count = (size_t)rows * width;
    float norm_host[count];
    float weight_host[width];
    uint16_t norm_in_bf16[count];
    uint16_t weight_bf16[width];
    uint16_t norm_out_bf16[count];
    float norm_ref[count];

    for (uint32_t column = 0; column < width; column++) {
        weight_host[column] = 0.5f + 0.01f * (float)column;
        weight_bf16[column] = f32_to_bf16(weight_host[column]);
    }
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < width; column++) {
            size_t index = (size_t)row * width + column;
            norm_host[index] = sinf((float)index * 0.07f);
            norm_in_bf16[index] = f32_to_bf16(norm_host[index]);
        }
    }
    rms_norm_ref(norm_host, weight_host, norm_ref, rows, width, epsilon);

    h3_gpu_tensor *norm_in = h3_gpu_tensor_from_bf16(gpu, norm_in_bf16, count);
    h3_gpu_tensor *norm_weight =
        h3_gpu_tensor_from_bf16(gpu, weight_bf16, width);
    h3_gpu_tensor *norm_out = h3_gpu_tensor_new_bf16(gpu, count);
    check(norm_in && norm_weight && norm_out, "rms_norm tensor alloc");
    if (norm_in && norm_weight && norm_out) {
        check(h3_gpu_rms_norm_bf16(gpu, norm_out, norm_in, norm_weight, rows,
                                   width, epsilon),
              "rms_norm");
        check(h3_gpu_submit(gpu), "submit rms_norm");
        check(h3_gpu_tensor_read_bf16(norm_out, norm_out_bf16, count),
              "read rms_norm");
        for (size_t i = 0; i < count; i++) {
            float got = bf16_to_f32(norm_out_bf16[i]);
            if (fabsf(got - norm_ref[i]) >= 2e-2f) {
                fprintf(stderr,
                        "FAIL: rms_norm mismatch at %zu got=%f expected=%f\n",
                        i, got, norm_ref[i]);
                failures++;
                break;
            }
        }
    }

    h3_gpu_tensor_free(silu_in);
    h3_gpu_tensor_free(silu_out);
    h3_gpu_tensor_free(norm_in);
    h3_gpu_tensor_free(norm_weight);
    h3_gpu_tensor_free(norm_out);
    h3_gpu_free(gpu);

    if (failures) {
        fprintf(stderr, "h3_cuda_ops: %d failure(s)\n", failures);
        return 1;
    }
    puts("ok: CUDA op tests passed");
    return 0;
}
