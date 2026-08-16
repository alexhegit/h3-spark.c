#include "h3_audio_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LATENT_LENGTH = 4, LATENT_COUNT = 32 * 2 * LATENT_LENGTH };

static int weights_available(const char *model_root) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/FL2VA/audio_vae/config.json", model_root);
    FILE *file = fopen(path, "rb");
    if (!file) {
        snprintf(path, sizeof(path), "%s/FL2VA/audio_vae/model.safetensors",
                 model_root);
        file = fopen(path, "rb");
    }
    if (!file) return 0;
    fclose(file);
    return 1;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "cuda AudioVAE stage: %d/%d\n", completed, total);
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

    float latent[LATENT_COUNT];
    for (int i = 0; i < LATENT_COUNT; i++)
        latent[i] = sinf((float)i * 0.013f) * 0.02f;

    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/audio_vae", model_root);
    char error[512];
    h3_audio_waveform got;
    if (!h3_audio_vae_decode(weights, "h3_shaders.metal", latent, LATENT_LENGTH,
                             progress, NULL, &got, error, sizeof(error))) {
        fprintf(stderr, "FAIL: h3_audio_vae_decode: %s\n", error);
        return 1;
    }
    if (got.channels != 2 || got.sample_rate != 32000 ||
        got.samples != LATENT_LENGTH * 800) {
        fprintf(stderr, "FAIL: unexpected waveform shape ch=%d samples=%d sr=%d\n",
                got.channels, got.samples, got.sample_rate);
        h3_audio_waveform_free(&got);
        return 1;
    }

    size_t count = (size_t)got.channels * (size_t)got.samples;
    double abs_max = 0.0;
    int finite = 1;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(got.pcm[i])) {
            finite = 0;
            break;
        }
        double value = fabs((double)got.pcm[i]);
        if (value > abs_max) abs_max = value;
    }
    printf("cuda audio VAE smoke: %d ch × %d samples @ %d Hz, abs-max %.6g, "
           "%.2f MiB peak, %llu conv, %llu submissions\n",
           got.channels, got.samples, got.sample_rate, abs_max,
           (double)got.gpu_stats.peak_live_bytes / (1024.0 * 1024.0),
           (unsigned long long)got.gpu_stats.mps_conv_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (!finite) {
        fprintf(stderr, "FAIL: non-finite PCM from audio VAE decode\n");
        h3_audio_waveform_free(&got);
        return 1;
    }
    if (got.gpu_stats.mps_conv_dispatches == 0) {
        fprintf(stderr, "FAIL: expected conv dispatches > 0\n");
        h3_audio_waveform_free(&got);
        return 1;
    }
    h3_audio_waveform_free(&got);
    puts("ok: CUDA audio VAE decode smoke passed");
    return 0;
}
