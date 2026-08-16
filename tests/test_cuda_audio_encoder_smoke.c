#include "h3_audio_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SAMPLES = 1600 }; /* two latent frames at 800 samples each */

static int weights_available(const char *model_root) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/FL2VA/audio_vae/model.safetensors",
             model_root);
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    fclose(file);
    return 1;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "cuda audio encoder stage: %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : getenv("H3_MODEL_ROOT");
    if (!model_root) model_root = "MiniMax-H3";
    if (!weights_available(model_root)) {
        const char *fallback = "/home/alex/HF-MODELS/MiniMax-H3";
        if (weights_available(fallback)) {
            model_root = fallback;
        } else {
            puts("skip: audio VAE weights not available");
            return 0;
        }
    }

    float *pcm = (float *)malloc((size_t)2 * SAMPLES * sizeof(float));
    if (!pcm) {
        fprintf(stderr, "FAIL: OOM\n");
        return 1;
    }
    for (int i = 0; i < 2 * SAMPLES; i++)
        pcm[i] = sinf((float)i * 0.01f) * 0.1f;

    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/audio_vae", model_root);
    char error[512];
    h3_audio_latent got;
    if (!h3_audio_vae_encode(weights, "h3_shaders.metal", pcm, SAMPLES,
                             progress, NULL, &got, error, sizeof(error))) {
        fprintf(stderr, "FAIL: h3_audio_vae_encode: %s\n", error);
        free(pcm);
        return 1;
    }

    size_t count =
        (size_t)32 * (size_t)got.stereo * (size_t)got.length;
    double abs_max = 0.0;
    int finite = 1;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(got.values[i])) {
            finite = 0;
            break;
        }
        double value = fabs((double)got.values[i]);
        if (value > abs_max) abs_max = value;
    }
    printf("cuda audio encoder smoke: latent 32x%dx%d, abs-max %.6g, "
           "%.2f MiB peak, %llu conv, %llu SDPA, %llu submissions\n",
           got.stereo, got.length, abs_max,
           (double)got.gpu_stats.peak_live_bytes / (1024.0 * 1024.0),
           (unsigned long long)got.gpu_stats.mps_conv_dispatches,
           (unsigned long long)got.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (!finite) {
        fprintf(stderr, "FAIL: non-finite audio latent\n");
        h3_audio_latent_free(&got);
        free(pcm);
        return 1;
    }
    h3_audio_latent_free(&got);
    free(pcm);
    puts("ok: CUDA audio VAE encode smoke passed");
    return 0;
}
