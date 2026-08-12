#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message);
    failures++;
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    check(gpu != NULL, "h3_gpu_create");
    if (!gpu) {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }

    check(h3_gpu_is_m5(gpu), "GB10 fast-path probe");
    check(h3_gpu_begin(gpu), "h3_gpu_begin");

    const size_t count = 1024;
    float host_f32[1024];
    uint16_t host_bf16[1024];
    for (size_t i = 0; i < count; i++) host_f32[i] = sinf((float)i * 0.01f);

    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, host_f32, count);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *roundtrip = h3_gpu_tensor_new_f32(gpu, count);
    h3_gpu_tensor *left = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *right = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *sum = h3_gpu_tensor_new_bf16(gpu, count);
    check(input && bf16 && roundtrip && left && right && sum, "tensor alloc");

    if (input && bf16 && roundtrip && left && right && sum) {
        check(h3_gpu_cast_f32_to_bf16(gpu, bf16, input, (uint32_t)count),
              "cast f32->bf16");
        check(h3_gpu_submit(gpu), "submit after cast");
        check(h3_gpu_tensor_read_bf16(bf16, host_bf16, count), "read bf16");
        check(h3_gpu_copy_bf16(gpu, left, 0, bf16, 0, count), "copy bf16");
        check(h3_gpu_copy_bf16(gpu, right, 0, bf16, 0, count), "copy bf16");
        check(h3_gpu_add_bf16(gpu, sum, left, right, (uint32_t)count),
              "add bf16");
        check(h3_gpu_cast_bf16_to_f32(gpu, roundtrip, sum, (uint32_t)count),
              "cast bf16->f32");
        check(h3_gpu_submit(gpu), "submit after add");

        float output[1024];
        check(h3_gpu_tensor_read_f32(roundtrip, output, count), "read f32");
        if (!failures) {
            float expected = host_f32[10] + host_f32[10];
            check(fabsf(output[10] - expected) < 1e-2f, "add numerical check");
        }
    }

    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(bf16);
    h3_gpu_tensor_free(roundtrip);
    h3_gpu_tensor_free(left);
    h3_gpu_tensor_free(right);
    h3_gpu_tensor_free(sum);
    h3_gpu_free(gpu);

    if (failures) {
        fprintf(stderr, "h3_cuda_smoke: %d failure(s)\n", failures);
        return 1;
    }
    puts("ok: CUDA smoke tests passed");
    return 0;
}
