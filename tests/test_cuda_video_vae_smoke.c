#include "h3_video_vae.h"

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
    if (completed == 1 || completed == total || completed % 6 == 0)
        fprintf(stderr, "cuda video VAE load: %d/%d blocks\n", completed,
                total);
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

    /* Tiny latent: channels×T×H×W = 24×2×2×2 → 5 frames @ 32×32. */
    enum { LATENT_COUNT = 24 * 2 * 2 * 2 };
    float latent[LATENT_COUNT];
    for (int i = 0; i < LATENT_COUNT; i++)
        latent[i] = sinf((float)i * 0.017f) * 0.05f;

    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source", model_root);
    char error[512];
    h3_video_frames got;
    if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent, 2, 2, 2,
                             progress, NULL, &got, error, sizeof(error))) {
        fprintf(stderr, "FAIL: h3_video_vae_decode: %s\n", error);
        return 1;
    }
    if (got.frames != 5 || got.height != 32 || got.width != 32) {
        fprintf(stderr, "FAIL: unexpected shape %d×%d×%d\n", got.frames,
                got.height, got.width);
        h3_video_frames_free(&got);
        return 1;
    }

    size_t rgb_count =
        (size_t)got.frames * (size_t)got.height * (size_t)got.width * 3u;
    double abs_max = 0.0;
    int finite = 1;
    for (size_t i = 0; i < rgb_count; i++) {
        if (!isfinite(got.rgb[i])) {
            finite = 0;
            break;
        }
        double value = fabs((double)got.rgb[i]);
        if (value > abs_max) abs_max = value;
    }
    printf("cuda video VAE smoke: %d frames %dx%d, abs-max %.6g, "
           "%.2f MiB peak, %llu SDPA, %llu submissions\n",
           got.frames, got.width, got.height, abs_max,
           (double)got.gpu_stats.peak_live_bytes / (1024.0 * 1024.0),
           (unsigned long long)got.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (!finite) {
        fprintf(stderr, "FAIL: non-finite RGB from video VAE decode\n");
        h3_video_frames_free(&got);
        return 1;
    }
    if (got.gpu_stats.mps_sdpa_dispatches != 36) {
        fprintf(stderr, "FAIL: expected 36 SDPA, got %llu\n",
                (unsigned long long)got.gpu_stats.mps_sdpa_dispatches);
        h3_video_frames_free(&got);
        return 1;
    }
    h3_video_frames_free(&got);
    puts("ok: CUDA video VAE decode smoke passed");
    return 0;
}
