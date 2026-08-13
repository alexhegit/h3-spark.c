#include "h3_text_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void require_finite(const h3_text_embedding *embedding) {
    size_t count = embedding->tokens * embedding->width;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(bf16_to_f32(embedding->values[index]))) {
            fprintf(stderr, "FAIL: non-finite text output at %zu\n", index);
            exit(1);
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

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : getenv("H3_MODEL_ROOT");
    int full_layers = argc > 2 && strcmp(argv[2], "full") == 0;
    if (!model_root) model_root = "MiniMax-H3";
    if (!weights_available(model_root)) {
        const char *fallback = "/home/alex/HF-MODELS/MiniMax-H3";
        if (model_root != fallback && weights_available(fallback))
            model_root = fallback;
    }
    if (!weights_available(model_root)) {
        puts("skip: MiniMax-H3 text encoder weights not available");
        return 0;
    }

    char weights_path[1024];
    snprintf(weights_path, sizeof(weights_path), "%s/FL2VA/text_encoder",
             model_root);

    static const uint32_t token_ids[] = {
        151643u, 872u, 525u, 279u, 1196u, 374u, 264u, 279u,
    };
    const size_t token_count = sizeof(token_ids) / sizeof(token_ids[0]);
    const int layer_count = full_layers ? 50 : 1;

    char error[512];
    h3_text_embedding first;
    h3_text_embedding second;
    if (!h3_text_encode_layers_bf16(weights_path, "h3_shaders.metal", token_ids,
                                    token_count, layer_count, NULL, NULL,
                                    &first, error, sizeof(error))) {
        fail(error);
    }
    require_finite(&first);
    size_t elements = first.tokens * first.width;
    uint64_t first_hash = hash_bf16(first.values, elements);

    if (!h3_text_encode_layers_bf16(weights_path, "h3_shaders.metal", token_ids,
                                    token_count, layer_count, NULL, NULL,
                                    &second, error, sizeof(error))) {
        h3_text_embedding_free(&first);
        fail(error);
    }
    require_finite(&second);
    if (first.tokens != second.tokens || first.width != second.width ||
        memcmp(first.values, second.values,
               elements * sizeof(*first.values)) != 0) {
        h3_text_embedding_free(&first);
        h3_text_embedding_free(&second);
        fail("text encoder output is not deterministic");
    }

    printf("cuda text smoke: %zu tokens, %d layer(s), hash %016llx, "
           "%.2f MiB peak, %llu direct, %llu submissions\n",
           first.tokens, layer_count, (unsigned long long)first_hash,
           (double)first.gpu_stats.peak_live_bytes / (1024.0 * 1024.0),
           (unsigned long long)first.gpu_stats.direct_dispatches,
           (unsigned long long)first.gpu_stats.submissions);

    h3_text_embedding_free(&first);
    h3_text_embedding_free(&second);
    if (full_layers) {
        puts("ok: CUDA Qwen 50-layer text smoke passed (real weights, no MLX)");
    } else {
        puts("ok: CUDA Qwen layer-0 text smoke passed (real weights, no MLX)");
    }
    return 0;
}
