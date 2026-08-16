#include "h3_video_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int weights_available(const char *model_root) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/FL2VA/video_vae/source/config.json",
             model_root);
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    fclose(file);
    return 1;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "cuda video encoder tile: %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : getenv("H3_MODEL_ROOT");
    if (!model_root) model_root = "MiniMax-H3";
    if (!weights_available(model_root)) {
        const char *fallback = "/home/alex/HF-MODELS/MiniMax-H3";
        if (weights_available(fallback)) {
            model_root = fallback;
        } else {
            puts("skip: video VAE weights not available");
            return 0;
        }
    }

    enum { HEIGHT = 64, WIDTH = 64, FRAMES = 1, CHANNELS = 24 };
    size_t pixel_count = 3u * (size_t)FRAMES * HEIGHT * WIDTH;
    float *pixels = (float *)malloc(pixel_count * sizeof(float));
    if (!pixels) {
        fprintf(stderr, "FAIL: OOM\n");
        return 1;
    }
    for (size_t i = 0; i < pixel_count; i++)
        pixels[i] = sinf((float)i * 0.01f) * 0.5f + 0.5f;

    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source", model_root);
    char error[512];
    h3_video_latent got;
    if (!h3_video_vae_encode(weights, "h3_shaders.metal", pixels, FRAMES,
                             HEIGHT, WIDTH, progress, NULL, &got, error,
                             sizeof(error))) {
        fprintf(stderr, "FAIL: h3_video_vae_encode: %s\n", error);
        free(pixels);
        return 1;
    }

    size_t latent_count =
        (size_t)CHANNELS * (size_t)got.time * (size_t)got.height *
        (size_t)got.width;
    double abs_max = 0.0;
    int finite = 1;
    for (size_t i = 0; i < latent_count; i++) {
        if (!isfinite(got.values[i])) {
            finite = 0;
            break;
        }
        double value = fabs((double)got.values[i]);
        if (value > abs_max) abs_max = value;
    }
    printf("cuda video encoder smoke: latent 24x%dx%dx%d, abs-max %.6g, "
           "%.2f MiB peak, %llu conv, %llu submissions\n",
           got.time, got.height, got.width, abs_max,
           (double)got.gpu_stats.peak_live_bytes / (1024.0 * 1024.0),
           (unsigned long long)got.gpu_stats.mps_conv_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (!finite) {
        fprintf(stderr, "FAIL: non-finite encoder latent\n");
        h3_video_latent_free(&got);
        free(pixels);
        return 1;
    }
    if (got.gpu_stats.mps_conv_dispatches == 0) {
        fprintf(stderr, "FAIL: expected conv dispatches > 0\n");
        h3_video_latent_free(&got);
        free(pixels);
        return 1;
    }
    h3_video_latent_free(&got);
    free(pixels);
    puts("ok: CUDA video VAE encode smoke passed");
    return 0;
}
