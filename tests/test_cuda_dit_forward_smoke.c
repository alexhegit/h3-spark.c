#include "h3_dit.h"
#include "h3_host.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_ROWS = 6,
    TEXT_WIDTH = 5120,
    LATENT_T = 2,
    LATENT_H = 2,
    LATENT_W = 2,
    AUDIO_T = 8,
    FRAME_COUNT = 5,
    VIDEO_ELEMENTS = 24 * LATENT_T * LATENT_H * LATENT_W,
    AUDIO_ELEMENTS = 32 * 2 * AUDIO_T
};

static uint32_t rng_state = 0xC0FFEEu;

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static float rng_f32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) * (1.0f / 16777216.0f) - 0.5f;
}

static uint16_t f32_to_bf16(float value) {
    union {
        uint32_t word;
        float as_float;
    } converted;
    converted.as_float = value;
    converted.word += 0x7fffu + ((converted.word >> 16u) & 1u);
    return (uint16_t)(converted.word >> 16u);
}

static int weights_available(const char *model_root) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/FL2VA/transformer/config.json", model_root);
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    fclose(file);
    return 1;
}

static void require_finite(const float *values, size_t count, const char *label) {
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            fprintf(stderr, "FAIL: %s non-finite at %zu\n", label, index);
            exit(1);
        }
    }
}

static uint64_t hash_f32(const float *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count; index++) {
        union {
            float as_float;
            uint32_t word;
        } converted;
        converted.as_float = values[index];
        hash ^= converted.word;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void configure_bf16_path(void) {
    setenv("H3_DISABLE_INT8_MLP", "1", 1);
    setenv("H3_DISABLE_INT8_QKV", "1", 1);
    setenv("H3_DISABLE_INT8_ATTENTION_OUT", "1", 1);
    setenv("H3_DISABLE_TOKEN_REDUCTION", "1", 1);
}

static h3_dit *load_smoke_dit(const char *weights_path, h3_text_embedding *text,
                              h3_layout *layout, h3_sigma_schedule *sigmas,
                              unsigned active_blocks, char *error,
                              size_t error_size) {
    return h3_dit_load_t2va(
        weights_path, "h3_shaders.metal", text, layout, sigmas, active_blocks,
        1, 0, 0, 1.0f,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        NULL, NULL, error, error_size);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : getenv("H3_MODEL_ROOT");
    int full_blocks = argc > 2 && strcmp(argv[2], "full") == 0;
    if (!model_root) model_root = "MiniMax-H3";
    if (!weights_available(model_root)) {
        const char *fallback = "/home/alex/HF-MODELS/MiniMax-H3";
        if (model_root != fallback && weights_available(fallback))
            model_root = fallback;
    }
    if (!weights_available(model_root)) {
        puts("skip: MiniMax-H3 transformer weights not available");
        return 0;
    }

    configure_bf16_path();

    char weights_path[1024];
    snprintf(weights_path, sizeof(weights_path), "%s/FL2VA/transformer",
             model_root);

    uint16_t *text_bf16 = malloc((size_t)TEXT_ROWS * TEXT_WIDTH * sizeof(*text_bf16));
    float *video = malloc((size_t)VIDEO_ELEMENTS * sizeof(*video));
    float *audio = malloc((size_t)AUDIO_ELEMENTS * sizeof(*audio));
    float *video_out_a = malloc((size_t)VIDEO_ELEMENTS * sizeof(*video_out_a));
    float *audio_out_a = malloc((size_t)AUDIO_ELEMENTS * sizeof(*audio_out_a));
    float *video_out_b = malloc((size_t)VIDEO_ELEMENTS * sizeof(*video_out_b));
    float *audio_out_b = malloc((size_t)AUDIO_ELEMENTS * sizeof(*audio_out_b));
    if (!text_bf16 || !video || !audio || !video_out_a || !audio_out_a ||
        !video_out_b || !audio_out_b)
        fail("host allocation failed");

    rng_state = 0xC0FFEEu;
    for (size_t index = 0; index < (size_t)TEXT_ROWS * TEXT_WIDTH; index++)
        text_bf16[index] = f32_to_bf16(rng_f32() * 0.1f);
    for (size_t index = 0; index < VIDEO_ELEMENTS; index++)
        video[index] = rng_f32() * 0.05f;
    for (size_t index = 0; index < AUDIO_ELEMENTS; index++)
        audio[index] = rng_f32() * 0.05f;

    h3_text_embedding text = {
        .tokens = TEXT_ROWS,
        .width = TEXT_WIDTH,
        .values = text_bf16,
    };
    h3_layout_spec spec = {TEXT_ROWS, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                           FRAME_COUNT, NULL, 0, NULL, 0};
    h3_layout layout;
    char error[512];
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) fail(error);

    h3_sigma_schedule sigmas;
    if (!h3_schedule_build(20, &sigmas)) fail("cannot build sigma schedule");

    unsigned active_blocks = full_blocks ? 50u : 25u;
    h3_dit *dit = load_smoke_dit(weights_path, &text, &layout, &sigmas,
                                 active_blocks, error, sizeof(error));
    if (!dit) {
        fprintf(stderr, "FAIL: h3_dit_load_t2va: %s\n", error);
        free(text_bf16);
        free(video);
        free(audio);
        free(video_out_a);
        free(audio_out_a);
        free(video_out_b);
        free(audio_out_b);
        h3_layout_free(&layout);
        return 1;
    }
    if (h3_dit_video_elements(dit) != VIDEO_ELEMENTS ||
        h3_dit_audio_elements(dit) != AUDIO_ELEMENTS)
        fail("DiT latent geometry mismatch");

    if (!h3_dit_forward(dit, 0, video, audio, video_out_a, audio_out_a, error,
                        sizeof(error))) {
        fprintf(stderr, "FAIL: h3_dit_forward: %s\n", error);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        return 1;
    }
    require_finite(video_out_a, VIDEO_ELEMENTS, "video velocity");
    require_finite(audio_out_a, AUDIO_ELEMENTS, "audio velocity");
    uint64_t video_hash = hash_f32(video_out_a, VIDEO_ELEMENTS);
    uint64_t audio_hash = hash_f32(audio_out_a, AUDIO_ELEMENTS);

    if (!h3_dit_forward(dit, 0, video, audio, video_out_b, audio_out_b, error,
                        sizeof(error)))
        fail(error);
    if (memcmp(video_out_a, video_out_b,
               (size_t)VIDEO_ELEMENTS * sizeof(*video_out_a)) != 0 ||
        memcmp(audio_out_a, audio_out_b,
               (size_t)AUDIO_ELEMENTS * sizeof(*audio_out_a)) != 0)
        fail("DiT forward output is not deterministic");

    h3_gpu_stats stats;
    if (!h3_dit_get_gpu_stats(dit, &stats)) fail("cannot read GPU stats");
    printf("cuda dit forward smoke: %u active blocks, video hash %016llx, "
           "audio hash %016llx, %.2f MiB peak, %llu submissions\n",
           active_blocks, (unsigned long long)video_hash,
           (unsigned long long)audio_hash,
           (double)stats.peak_live_bytes / (1024.0 * 1024.0),
           (unsigned long long)stats.submissions);

    h3_dit_free(dit);
    h3_layout_free(&layout);
    free(text_bf16);
    free(video);
    free(audio);
    free(video_out_a);
    free(audio_out_a);
    free(video_out_b);
    free(audio_out_b);

    if (full_blocks) {
        puts("ok: CUDA production DiT forward smoke passed (50 active blocks)");
    } else {
        puts("ok: CUDA production DiT forward smoke passed (25 active blocks)");
    }
    return 0;
}
