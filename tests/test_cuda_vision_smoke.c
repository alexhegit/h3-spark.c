#include "h3_vision_encoder.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    IMAGE_HEIGHT = 64,
    IMAGE_WIDTH = 64,
    FRAMES = 1,
    CHANNELS = 3
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static int weights_available(const char *model_root) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/FL2VA/text_encoder/config.json", model_root);
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    fclose(file);
    return 1;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void require_finite_output(const h3_vision_output *output) {
    size_t count = output->tokens * H3_VISION_OUTPUT_WIDTH;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(bf16_to_f32(output->merged[index]))) {
            fprintf(stderr, "FAIL: non-finite merged at %zu\n", index);
            exit(1);
        }
    }
    for (unsigned stack = 0; stack < H3_VISION_DEEPSTACKS; stack++) {
        for (size_t index = 0; index < count; index++) {
            if (!isfinite(bf16_to_f32(output->deepstack[stack][index]))) {
                fprintf(stderr, "FAIL: non-finite deepstack %u at %zu\n",
                        stack, index);
                exit(1);
            }
        }
    }
}

static uint64_t hash_bf16(const uint16_t *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count; index++) {
        hash ^= values[index] & 255u;
        hash *= UINT64_C(1099511628211);
        hash ^= values[index] >> 8;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t hash_output(const h3_vision_output *output) {
    size_t count = output->tokens * H3_VISION_OUTPUT_WIDTH;
    uint64_t hash = hash_bf16(output->merged, count);
    for (unsigned stack = 0; stack < H3_VISION_DEEPSTACKS; stack++) {
        uint64_t part = hash_bf16(output->deepstack[stack], count);
        hash ^= part;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void fill_pixels(float *pixels, size_t count) {
    for (size_t index = 0; index < count; index++) {
        float value = 0.5f + 0.25f * sinf((float)index * 0.07f);
        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
        pixels[index] = value;
    }
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : getenv("H3_MODEL_ROOT");
    if (!model_root) model_root = "MiniMax-H3";
    if (!weights_available(model_root)) {
        const char *fallback = "/home/alex/HF-MODELS/MiniMax-H3";
        if (model_root != fallback && weights_available(fallback))
            model_root = fallback;
    }
    if (!weights_available(model_root)) {
        puts("skip: MiniMax-H3 text encoder / vision weights not available");
        return 0;
    }

    char weights_path[1024];
    snprintf(weights_path, sizeof(weights_path), "%s/FL2VA/text_encoder",
             model_root);

    const size_t pixel_count =
        (size_t)FRAMES * CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    if (!pixels) fail("host allocation failed");
    fill_pixels(pixels, pixel_count);

    char error[512];
    h3_vision_output first;
    h3_vision_output second;
    if (!h3_vision_encode_bf16(weights_path, "h3_shaders.metal", pixels, FRAMES,
                               IMAGE_HEIGHT, IMAGE_WIDTH, NULL, NULL, &first,
                               error, sizeof(error))) {
        free(pixels);
        fail(error);
    }
    require_finite_output(&first);
    uint64_t first_hash = hash_output(&first);

    if (!h3_vision_encode_bf16(weights_path, "h3_shaders.metal", pixels, FRAMES,
                               IMAGE_HEIGHT, IMAGE_WIDTH, NULL, NULL, &second,
                               error, sizeof(error))) {
        h3_vision_output_free(&first);
        free(pixels);
        fail(error);
    }
    require_finite_output(&second);

    size_t elements = first.tokens * H3_VISION_OUTPUT_WIDTH;
    if (first.tokens != second.tokens || first.grid_h != second.grid_h ||
        first.grid_w != second.grid_w ||
        memcmp(first.merged, second.merged,
               elements * sizeof(*first.merged)) != 0) {
        h3_vision_output_free(&first);
        h3_vision_output_free(&second);
        free(pixels);
        fail("vision encoder merged output is not deterministic");
    }
    for (unsigned stack = 0; stack < H3_VISION_DEEPSTACKS; stack++) {
        if (memcmp(first.deepstack[stack], second.deepstack[stack],
                   elements * sizeof(*first.merged)) != 0) {
            h3_vision_output_free(&first);
            h3_vision_output_free(&second);
            free(pixels);
            fail("vision encoder deepstack output is not deterministic");
        }
    }

    printf("cuda vision smoke: %zux%zu grid, %zu tokens, hash %016llx, "
           "%.2f MiB peak, %llu direct, %llu SDPA, %llu submissions\n",
           (size_t)first.grid_h, (size_t)first.grid_w, first.tokens,
           (unsigned long long)first_hash,
           (double)first.gpu_stats.peak_live_bytes / (1024.0 * 1024.0),
           (unsigned long long)first.gpu_stats.direct_dispatches,
           (unsigned long long)first.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)first.gpu_stats.submissions);

    h3_vision_output_free(&first);
    h3_vision_output_free(&second);
    free(pixels);
    puts("ok: CUDA Qwen vision smoke passed (real weights, no MLX fixture)");
    return 0;
}
