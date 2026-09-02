/* Times the DiT's attention on its own shape, so the kernel can be worked on
 * without a 20-second pipeline run between attempts.
 *
 * fox-fast attends over 1874 tokens with 56 heads of 128, which is 100.8 GFLOP
 * per call. The MMA kernel does that at ~33 TFLOP/s against a BF16 GEMM ceiling
 * near 98, and this exists to find out where the rest goes: H3_SDPA_HALF=qk or
 * pv drops one of the two matmuls, which localises the cost without needing the
 * hardware counters ncu cannot read on this machine.
 *
 * Build: make -f Makefile.linux h3_sdpa_bench
 */

#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec * 1e-9;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

static float next_normal(unsigned *state) {
    *state = *state * 1664525u + 1013904223u;
    float a = (float)((*state >> 8) & 0xffffffu) / (float)0x1000000u;
    *state = *state * 1664525u + 1013904223u;
    float b = (float)((*state >> 8) & 0xffffffu) / (float)0x1000000u;
    if (a < 1e-7f) a = 1e-7f;
    /* Box-Muller without the libm cosf, which is not the point here. */
    float radius = -2.0f * logf(a);
    if (radius < 0.0f) radius = 0.0f;
    return sqrtf(radius) * (b * 2.0f - 1.0f);
}

int main(int argc, char **argv) {
    uint32_t sequence = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 1874u;
    uint32_t heads = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 56u;
    uint32_t head_dim = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 128u;
    int iterations = argc > 4 ? atoi(argv[4]) : 20;

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "h3_gpu_create failed: %s\n", error);
        return 1;
    }

    size_t count = (size_t)sequence * heads * head_dim;
    uint16_t *host = (uint16_t *)malloc(count * sizeof(*host));
    if (!host) {
        fprintf(stderr, "host alloc failed\n");
        return 1;
    }
    unsigned state = 7u;
    for (size_t index = 0; index < count; index++)
        host[index] = f32_to_bf16(next_normal(&state));

    h3_gpu_tensor *query = h3_gpu_tensor_from_bf16(gpu, host, count);
    for (size_t index = 0; index < count; index++)
        host[index] = f32_to_bf16(next_normal(&state));
    h3_gpu_tensor *key = h3_gpu_tensor_from_bf16(gpu, host, count);
    for (size_t index = 0; index < count; index++)
        host[index] = f32_to_bf16(next_normal(&state));
    h3_gpu_tensor *value = h3_gpu_tensor_from_bf16(gpu, host, count);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, count);
    free(host);
    if (!query || !key || !value || !output) {
        fprintf(stderr, "device alloc failed: %s\n", h3_gpu_error(gpu));
        return 1;
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    for (int warm = 0; warm < 3; warm++) {
        if (!h3_gpu_sdpa_bf16(gpu, output, query, key, value, sequence, heads,
                              head_dim, scale) ||
            !h3_gpu_submit(gpu)) {
            fprintf(stderr, "sdpa failed: %s\n", h3_gpu_error(gpu));
            return 1;
        }
    }

    double start = now_seconds();
    for (int iteration = 0; iteration < iterations; iteration++)
        h3_gpu_sdpa_bf16(gpu, output, query, key, value, sequence, heads,
                         head_dim, scale);
    if (!h3_gpu_submit(gpu)) {
        fprintf(stderr, "sdpa failed: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    double seconds = (now_seconds() - start) / iterations;

    /* Q·Kᵀ and P·V, two flops per multiply-accumulate. */
    double flops = 4.0 * (double)sequence * sequence * head_dim * heads;
    printf("seq %u heads %u dim %u: %.3f ms  %.1f TFLOP/s", sequence, heads,
           head_dim, seconds * 1e3, flops / seconds * 1e-12);
    const char *half = getenv("H3_SDPA_HALF");
    if (half && *half) printf("  HALF=%s", half);
    printf("\n");

    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(output);
    h3_gpu_free(gpu);
    return 0;
}
