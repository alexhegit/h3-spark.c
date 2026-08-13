#include "h3_gpu.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_ROWS = 6,
    AUDIO_ROWS = 16,
    VIDEO_ROWS = 2,
    SEQUENCE = 24,
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    TIME_INPUT = 256,
    TIME_HIDDEN = 5376,
    TIME_DIM = 2688,
    ROPE_HALF = 48,
    MODALITIES = 3,
    SLOTS = 6,
    MAX_TENSORS = 256
};

typedef struct {
    h3_weight_store *weights;
    h3_gpu *gpu;
    h3_gpu_tensor *owned[MAX_TENSORS];
    size_t owned_count;
} test_context;

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
} block_weights;

typedef struct {
    h3_gpu_tensor *text_qwen;
    h3_gpu_tensor *video_rows;
    h3_gpu_tensor *audio_rows;
    h3_gpu_tensor *time_features;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *row_map;
    h3_gpu_tensor *refined;
    h3_gpu_tensor *refiner_norm;
    h3_gpu_tensor *refiner_qkv;
    h3_gpu_tensor *refiner_q;
    h3_gpu_tensor *refiner_k;
    h3_gpu_tensor *refiner_v;
    h3_gpu_tensor *refiner_heads;
    h3_gpu_tensor *refiner_branch;
    h3_gpu_tensor *refiner_fc1;
    h3_gpu_tensor *refiner_activated;
    h3_gpu_tensor *video_f32;
    h3_gpu_tensor *audio_f32;
    h3_gpu_tensor *video_bf16;
    h3_gpu_tensor *audio_bf16;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *time_hidden;
    h3_gpu_tensor *time_silu;
    h3_gpu_tensor *time_output;
    h3_gpu_tensor *time_bf16;
    h3_gpu_tensor *time_bf16_silu;
    h3_gpu_tensor *modulation;
    h3_gpu_tensor *mod_attention;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention_heads;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *after_attention;
    h3_gpu_tensor *mod_mlp;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *mlp_output;
    h3_gpu_tensor *block_output;
    h3_gpu_tensor *final_modulation;
    h3_gpu_tensor *final_audio_input;
    h3_gpu_tensor *final_video_input;
    h3_gpu_tensor *final_audio_norm;
    h3_gpu_tensor *final_video_norm;
    h3_gpu_tensor *final_audio_cast;
    h3_gpu_tensor *final_video_cast;
    h3_gpu_tensor *audio_rows_output;
    h3_gpu_tensor *video_rows_output;
    h3_gpu_tensor *final_rows;
} run_tensors;

static uint32_t rng_state = 0xA5A5A5A5u;

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

static float bf16_to_f32(uint16_t bits) {
    union {
        uint32_t word;
        float as_float;
    } converted;
    converted.word = (uint32_t)bits << 16u;
    return converted.as_float;
}

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_cuda_dit_block_smoke.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static h3_gpu_tensor *own(test_context *test, h3_gpu_tensor *tensor) {
    require(tensor != NULL, "GPU tensor allocation failed");
    require(test->owned_count < MAX_TENSORS, "tensor registry overflow");
    test->owned[test->owned_count++] = tensor;
    return tensor;
}

static void gpu_call(test_context *test, int ok, const char *operation) {
    if (ok) return;
    fprintf(stderr, "FAIL tests/test_cuda_dit_block_smoke.c: %s: %s\n",
            operation, h3_gpu_error(test->gpu));
    exit(1);
}

static h3_gpu_tensor *new_bf16(test_context *test, size_t elements) {
    return own(test, h3_gpu_tensor_new_bf16(test->gpu, elements));
}

static h3_gpu_tensor *new_f32(test_context *test, size_t elements) {
    return own(test, h3_gpu_tensor_new_f32(test->gpu, elements));
}

static h3_gpu_tensor *weight_bf16_1d(test_context *test, const char *name,
                                     uint64_t width) {
    uint64_t shape[] = {width};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(
        test->weights, test->gpu, name, 1, shape, error, sizeof(error));
    if (!tensor) die(error);
    return own(test, tensor);
}

static h3_gpu_tensor *weight_bf16_2d(test_context *test, const char *name,
                                     uint64_t rows, uint64_t columns) {
    uint64_t shape[] = {rows, columns};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(
        test->weights, test->gpu, name, 2, shape, error, sizeof(error));
    if (!tensor) die(error);
    return own(test, tensor);
}

static h3_gpu_tensor *weight_f32_1d(test_context *test, const char *name,
                                    uint64_t width) {
    uint64_t shape[] = {width};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_f32(
        test->weights, test->gpu, name, 1, shape, error, sizeof(error));
    if (!tensor) die(error);
    return own(test, tensor);
}

static h3_gpu_tensor *weight_f32_2d(test_context *test, const char *name,
                                    uint64_t rows, uint64_t columns) {
    uint64_t shape[] = {rows, columns};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_f32(
        test->weights, test->gpu, name, 2, shape, error, sizeof(error));
    if (!tensor) die(error);
    return own(test, tensor);
}

static h3_gpu_tensor *weight_bf16_1d_ephemeral(test_context *test,
                                               const char *name,
                                               uint64_t width) {
    uint64_t shape[] = {width};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(
        test->weights, test->gpu, name, 1, shape, error, sizeof(error));
    if (!tensor) die(error);
    return tensor;
}

static h3_gpu_tensor *weight_bf16_2d_ephemeral(test_context *test,
                                               const char *name, uint64_t rows,
                                               uint64_t columns) {
    uint64_t shape[] = {rows, columns};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(
        test->weights, test->gpu, name, 2, shape, error, sizeof(error));
    if (!tensor) die(error);
    return tensor;
}

static block_weights load_block_ephemeral(test_context *test,
                                          const char *prefix) {
    block_weights result;
    char name[192];
#define BF1E(field, suffix, width)                                            \
    do {                                                                      \
        snprintf(name, sizeof(name), "%s%s", prefix, suffix);                \
        result.field = weight_bf16_1d_ephemeral(test, name, width);           \
    } while (0)
#define BF2E(field, suffix, rows, columns)                                    \
    do {                                                                      \
        snprintf(name, sizeof(name), "%s%s", prefix, suffix);                \
        result.field = weight_bf16_2d_ephemeral(test, name, rows, columns);   \
    } while (0)
    BF1E(norm1, "norm1.weight", HIDDEN);
    BF1E(norm2, "norm2.weight", HIDDEN);
    BF2E(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    BF1E(q_norm, "attn.q_norm.weight", HEAD_DIM);
    BF1E(k_norm, "attn.k_norm.weight", HEAD_DIM);
    BF2E(out, "attn.out_proj.weight", HIDDEN, INNER);
    BF2E(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    BF2E(fc2, "mlp.fc2.weight", HIDDEN, FFN);
#undef BF1E
#undef BF2E
    return result;
}

static void free_block_weights(block_weights *weight) {
    if (!weight) return;
    h3_gpu_tensor_free(weight->norm1);
    h3_gpu_tensor_free(weight->norm2);
    h3_gpu_tensor_free(weight->qkv);
    h3_gpu_tensor_free(weight->q_norm);
    h3_gpu_tensor_free(weight->k_norm);
    h3_gpu_tensor_free(weight->out);
    h3_gpu_tensor_free(weight->fc1);
    h3_gpu_tensor_free(weight->fc2);
    memset(weight, 0, sizeof(*weight));
}

static block_weights load_block(test_context *test, const char *prefix) {
    block_weights result;
    char name[192];
#define BF1(field, suffix, width)                                             \
    do {                                                                      \
        snprintf(name, sizeof(name), "%s%s", prefix, suffix);                \
        result.field = weight_bf16_1d(test, name, width);                   \
    } while (0)
#define BF2(field, suffix, rows, columns)                                     \
    do {                                                                      \
        snprintf(name, sizeof(name), "%s%s", prefix, suffix);                \
        result.field = weight_bf16_2d(test, name, rows, columns);            \
    } while (0)
    BF1(norm1, "norm1.weight", HIDDEN);
    BF1(norm2, "norm2.weight", HIDDEN);
    BF2(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    BF1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    BF1(k_norm, "attn.k_norm.weight", HEAD_DIM);
    BF2(out, "attn.out_proj.weight", HIDDEN, INNER);
    BF2(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    BF2(fc2, "mlp.fc2.weight", HIDDEN, FFN);
#undef BF1
#undef BF2
    return result;
}

static int weights_available(const char *model_root) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/FL2VA/transformer/config.json", model_root);
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    fclose(file);
    return 1;
}

static void reset_rng(void) {
    rng_state = 0xA5A5A5A5u;
}

static void fill_host_inputs(float *text_f32, size_t text_count,
                             float *video_f32, size_t video_count,
                             float *audio_f32, size_t audio_count,
                             float *time_f32, size_t time_count,
                             float *rope_cos_f32, float *rope_sin_f32,
                             size_t rope_count, uint32_t *row_map) {
    reset_rng();
    for (size_t index = 0; index < text_count; index++)
        text_f32[index] = rng_f32() * 0.05f;
    for (size_t index = 0; index < video_count; index++)
        video_f32[index] = rng_f32() * 0.1f;
    for (size_t index = 0; index < audio_count; index++)
        audio_f32[index] = rng_f32() * 0.1f;
    for (size_t index = 0; index < time_count; index++)
        time_f32[index] = rng_f32() * 0.02f;
    for (size_t index = 0; index < rope_count; index++) {
        rope_cos_f32[index] = cosf((float)index * 0.017f);
        rope_sin_f32[index] = sinf((float)index * 0.017f);
    }
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        if (row < TEXT_ROWS)
            row_map[row] = 0;
        else if (row < TEXT_ROWS + AUDIO_ROWS)
            row_map[row] = 1;
        else
            row_map[row] = 2;
    }
}

static void upload_inputs(test_context *test, run_tensors *t,
                          const float *text_f32, const float *video_f32,
                          const float *audio_f32, const float *time_f32,
                          const float *rope_cos_f32, const float *rope_sin_f32) {
    size_t text_count = (size_t)TEXT_ROWS * TEXT_DIM;
    size_t video_count = (size_t)VIDEO_ROWS * 96;
    size_t audio_count = (size_t)AUDIO_ROWS * 32;
    uint16_t *text_bf16 = malloc(text_count * sizeof(*text_bf16));
    uint16_t *rope_cos_bf16 =
        malloc((size_t)SEQUENCE * ROPE_HALF * sizeof(*rope_cos_bf16));
    uint16_t *rope_sin_bf16 =
        malloc((size_t)SEQUENCE * ROPE_HALF * sizeof(*rope_sin_bf16));
    require(text_bf16 && rope_cos_bf16 && rope_sin_bf16,
            "host input allocation failed");
    for (size_t index = 0; index < text_count; index++)
        text_bf16[index] = f32_to_bf16(text_f32[index]);
    for (size_t index = 0; index < (size_t)SEQUENCE * ROPE_HALF; index++) {
        rope_cos_bf16[index] = f32_to_bf16(rope_cos_f32[index]);
        rope_sin_bf16[index] = f32_to_bf16(rope_sin_f32[index]);
    }
    gpu_call(test, h3_gpu_tensor_write_bf16(t->text_qwen, text_bf16, text_count),
             "upload text_qwen");
    gpu_call(test, h3_gpu_tensor_write_f32(t->video_rows, video_f32, video_count),
             "upload video_rows");
    gpu_call(test, h3_gpu_tensor_write_f32(t->audio_rows, audio_f32, audio_count),
             "upload audio_rows");
    gpu_call(test, h3_gpu_tensor_write_f32(t->time_features, time_f32, TIME_INPUT),
             "upload time_features");
    gpu_call(test, h3_gpu_tensor_write_bf16(
                     t->rope_cos, rope_cos_bf16, (size_t)SEQUENCE * ROPE_HALF),
             "upload rope_cos");
    gpu_call(test, h3_gpu_tensor_write_bf16(
                     t->rope_sin, rope_sin_bf16, (size_t)SEQUENCE * ROPE_HALF),
             "upload rope_sin");
    free(text_bf16);
    free(rope_cos_bf16);
    free(rope_sin_bf16);
}

static void run_refiner_block(test_context *test, const block_weights *weight,
                                h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                                h3_gpu_tensor *qkv, h3_gpu_tensor *query,
                                h3_gpu_tensor *key, h3_gpu_tensor *value,
                                h3_gpu_tensor *attention_heads,
                                h3_gpu_tensor *branch, h3_gpu_tensor *fc1,
                                h3_gpu_tensor *activated,
                                const h3_gpu_tensor *dummy_rope,
                                const char *label) {
    gpu_call(test, h3_gpu_rms_norm_bf16(test->gpu, norm, hidden, weight->norm1,
                                        TEXT_ROWS, HIDDEN, 1e-5f), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, qkv, norm, weight->qkv, NULL,
                                      TEXT_ROWS, HIDDEN, INNER * 3), label);
    gpu_call(test, h3_gpu_grouped_qkv_rope_bf16(
                     test->gpu, query, key, value, qkv, weight->q_norm,
                     weight->k_norm, dummy_rope, dummy_rope, TEXT_ROWS, HEADS,
                     HEAD_DIM, 0, 1e-5f),
             label);
    gpu_call(test, h3_gpu_sdpa_bf16(test->gpu, attention_heads, query, key,
                                    value, TEXT_ROWS, HEADS, HEAD_DIM,
                                    1.0f / sqrtf((float)HEAD_DIM)), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, branch, attention_heads,
                                      weight->out, NULL, TEXT_ROWS, INNER,
                                      HIDDEN), label);
    gpu_call(test, h3_gpu_add_bf16(test->gpu, hidden, hidden, branch,
                                   TEXT_ROWS * HIDDEN), label);
    gpu_call(test, h3_gpu_rms_norm_bf16(test->gpu, norm, hidden, weight->norm2,
                                        TEXT_ROWS, HIDDEN, 1e-5f), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, fc1, norm, weight->fc1, NULL,
                                      TEXT_ROWS, HIDDEN, FFN * 2), label);
    gpu_call(test, h3_gpu_swiglu_bf16(test->gpu, activated, fc1, TEXT_ROWS,
                                      FFN), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, branch, activated, weight->fc2,
                                      NULL, TEXT_ROWS, FFN, HIDDEN), label);
    gpu_call(test, h3_gpu_add_bf16(test->gpu, hidden, hidden, branch,
                                   TEXT_ROWS * HIDDEN), label);
}

static void run_dit_block_inplace(test_context *test,
                                  const block_weights *weight,
                                  h3_gpu_tensor *adaln_w, h3_gpu_tensor *adaln_b,
                                  h3_gpu_tensor *time_silu, h3_gpu_tensor *hidden,
                                  run_tensors *t, const char *label) {
    gpu_call(test, h3_gpu_linear_bf16(
                     test->gpu, t->modulation, time_silu, adaln_w, adaln_b, 1,
                     TIME_DIM, MODALITIES * SLOTS * HIDDEN),
             label);
    gpu_call(test, h3_gpu_adaln_bf16(
                     test->gpu, t->mod_attention, hidden, weight->norm1,
                     t->modulation, t->row_map, SEQUENCE, HIDDEN, SLOTS, 0, 1,
                     1e-5f),
             label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, t->qkv, t->mod_attention,
                                      weight->qkv, NULL, SEQUENCE, HIDDEN,
                                      INNER * 3), label);
    gpu_call(test, h3_gpu_grouped_qkv_rope_bf16(
                     test->gpu, t->query, t->key, t->value, t->qkv,
                     weight->q_norm, weight->k_norm, t->rope_cos, t->rope_sin,
                     SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
             label);
    gpu_call(test, h3_gpu_sdpa_bf16(
                     test->gpu, t->attention_heads, t->query, t->key, t->value,
                     SEQUENCE, HEADS, HEAD_DIM,
                     1.0f / sqrtf((float)HEAD_DIM)), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, t->attention_output,
                                      t->attention_heads, weight->out, NULL,
                                      SEQUENCE, INNER, HIDDEN), label);
    gpu_call(test, h3_gpu_gate_bf16(
                     test->gpu, t->after_attention, hidden, t->attention_output,
                     t->modulation, t->row_map, SEQUENCE, HIDDEN, SLOTS, 2),
             label);
    gpu_call(test, h3_gpu_adaln_bf16(test->gpu, t->mod_mlp, t->after_attention,
                                     weight->norm2, t->modulation, t->row_map,
                                     SEQUENCE, HIDDEN, SLOTS, 3, 4, 1e-5f),
             label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, t->fc1, t->mod_mlp, weight->fc1,
                                      NULL, SEQUENCE, HIDDEN, FFN * 2), label);
    gpu_call(test, h3_gpu_swiglu_bf16(test->gpu, t->activated, t->fc1, SEQUENCE,
                                      FFN), label);
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, t->mlp_output, t->activated,
                                      weight->fc2, NULL, SEQUENCE, FFN,
                                      HIDDEN), label);
    gpu_call(test, h3_gpu_gate_bf16(test->gpu, hidden, t->after_attention,
                                    t->mlp_output, t->modulation, t->row_map,
                                    SEQUENCE, HIDDEN, SLOTS, 5), label);
}

static void run_block0_forward(test_context *test, run_tensors *t,
                               const block_weights *refiner0,
                               const block_weights *refiner1,
                               const block_weights *block0,
                               h3_gpu_tensor *condition_w,
                               h3_gpu_tensor *condition_b,
                               h3_gpu_tensor *refiner_final,
                               h3_gpu_tensor *video_w, h3_gpu_tensor *video_b,
                               h3_gpu_tensor *audio_w, h3_gpu_tensor *audio_b,
                               h3_gpu_tensor *time_in_w, h3_gpu_tensor *time_in_b,
                               h3_gpu_tensor *time_out_w,
                               h3_gpu_tensor *time_out_b,
                               h3_gpu_tensor *adaln_w, h3_gpu_tensor *adaln_b,
                               int submit) {
    gpu_call(test, h3_gpu_begin(test->gpu), "begin forward");
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, t->refined, t->text_qwen,
                                      condition_w, condition_b, TEXT_ROWS,
                                      TEXT_DIM, HIDDEN),
             "condition projection");
    run_refiner_block(test, refiner0, t->refined, t->refiner_norm, t->refiner_qkv,
                      t->refiner_q, t->refiner_k, t->refiner_v, t->refiner_heads,
                      t->refiner_branch, t->refiner_fc1, t->refiner_activated,
                      refiner0->q_norm, "refiner block 0");
    run_refiner_block(test, refiner1, t->refined, t->refiner_norm, t->refiner_qkv,
                      t->refiner_q, t->refiner_k, t->refiner_v, t->refiner_heads,
                      t->refiner_branch, t->refiner_fc1, t->refiner_activated,
                      refiner1->q_norm, "refiner block 1");
    gpu_call(test, h3_gpu_rms_norm_bf16(test->gpu, t->refined, t->refined,
                                          refiner_final, TEXT_ROWS, HIDDEN,
                                          1e-5f), "refiner final norm");
    gpu_call(test, h3_gpu_linear_f32(test->gpu, t->video_f32, t->video_rows,
                                     video_w, video_b, VIDEO_ROWS, 96, HIDDEN),
             "video patch projection");
    gpu_call(test, h3_gpu_linear_f32(test->gpu, t->audio_f32, t->audio_rows,
                                     audio_w, audio_b, AUDIO_ROWS, 32, HIDDEN),
             "audio patch projection");
    gpu_call(test, h3_gpu_cast_f32_to_bf16(test->gpu, t->video_bf16, t->video_f32,
                                           VIDEO_ROWS * HIDDEN), "video cast");
    gpu_call(test, h3_gpu_cast_f32_to_bf16(test->gpu, t->audio_bf16, t->audio_f32,
                                           AUDIO_ROWS * HIDDEN), "audio cast");
    gpu_call(test, h3_gpu_copy_bf16(test->gpu, t->hidden, 0, t->refined, 0,
                                    TEXT_ROWS * HIDDEN), "pack text");
    gpu_call(test, h3_gpu_copy_bf16(test->gpu, t->hidden, TEXT_ROWS * HIDDEN,
                                    t->audio_bf16, 0, AUDIO_ROWS * HIDDEN),
             "pack audio");
    gpu_call(test, h3_gpu_copy_bf16(
                     test->gpu, t->hidden, (TEXT_ROWS + AUDIO_ROWS) * HIDDEN,
                     t->video_bf16, 0, VIDEO_ROWS * HIDDEN),
             "pack video");
    gpu_call(test, h3_gpu_linear_f32(test->gpu, t->time_hidden, t->time_features,
                                     time_in_w, time_in_b, 1, TIME_INPUT,
                                     TIME_HIDDEN), "time projection in");
    gpu_call(test, h3_gpu_silu_f32(test->gpu, t->time_silu, t->time_hidden,
                                   TIME_HIDDEN), "time SiLU");
    gpu_call(test, h3_gpu_linear_f32(test->gpu, t->time_output, t->time_silu,
                                     time_out_w, time_out_b, 1, TIME_HIDDEN,
                                     TIME_DIM), "time projection out");
    gpu_call(test, h3_gpu_cast_f32_to_bf16(test->gpu, t->time_bf16, t->time_output,
                                           TIME_DIM), "time cast");
    gpu_call(test, h3_gpu_silu_bf16(test->gpu, t->time_bf16_silu, t->time_bf16,
                                    TIME_DIM), "AdaLN SiLU");
    gpu_call(test, h3_gpu_linear_bf16(
                     test->gpu, t->modulation, t->time_bf16_silu, adaln_w,
                     adaln_b, 1, TIME_DIM, MODALITIES * SLOTS * HIDDEN),
             "block-0 AdaLN projection");
    gpu_call(test, h3_gpu_adaln_bf16(
                     test->gpu, t->mod_attention, t->hidden, block0->norm1,
                     t->modulation, t->row_map, SEQUENCE, HIDDEN, SLOTS, 0, 1,
                     1e-5f),
             "block-0 attention AdaLN");
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, t->qkv, t->mod_attention,
                                      block0->qkv, NULL, SEQUENCE, HIDDEN,
                                      INNER * 3), "block-0 QKV");
    gpu_call(test, h3_gpu_grouped_qkv_rope_bf16(
                     test->gpu, t->query, t->key, t->value, t->qkv,
                     block0->q_norm, block0->k_norm, t->rope_cos, t->rope_sin,
                     SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
             "block-0 QK norm/RoPE");
    gpu_call(test, h3_gpu_sdpa_bf16(
                     test->gpu, t->attention_heads, t->query, t->key, t->value,
                     SEQUENCE, HEADS, HEAD_DIM,
                     1.0f / sqrtf((float)HEAD_DIM)), "block-0 attention");
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, t->attention_output,
                                      t->attention_heads, block0->out, NULL,
                                      SEQUENCE, INNER, HIDDEN),
             "block-0 attention output");
    gpu_call(test, h3_gpu_gate_bf16(
                     test->gpu, t->after_attention, t->hidden, t->attention_output,
                     t->modulation, t->row_map, SEQUENCE, HIDDEN, SLOTS, 2),
             "block-0 attention gate");
    gpu_call(test, h3_gpu_adaln_bf16(test->gpu, t->mod_mlp, t->after_attention,
                                     block0->norm2, t->modulation, t->row_map,
                                     SEQUENCE, HIDDEN, SLOTS, 3, 4, 1e-5f),
             "block-0 MLP AdaLN");
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, t->fc1, t->mod_mlp, block0->fc1,
                                      NULL, SEQUENCE, HIDDEN, FFN * 2),
             "block-0 MLP input");
    gpu_call(test, h3_gpu_swiglu_bf16(test->gpu, t->activated, t->fc1, SEQUENCE,
                                      FFN), "block-0 SwiGLU");
    gpu_call(test, h3_gpu_linear_bf16(test->gpu, t->mlp_output, t->activated,
                                      block0->fc2, NULL, SEQUENCE, FFN,
                                      HIDDEN), "block-0 MLP output");
    gpu_call(test, h3_gpu_gate_bf16(test->gpu, t->block_output, t->after_attention,
                                    t->mlp_output, t->modulation, t->row_map,
                                    SEQUENCE, HIDDEN, SLOTS, 5),
             "block-0 MLP gate");
    if (submit)
        gpu_call(test, h3_gpu_submit(test->gpu), "submit forward");
}

static void run_full_step_forward(
    test_context *test, run_tensors *t, const block_weights *refiner0,
    const block_weights *refiner1, const block_weights *block0,
    h3_gpu_tensor *condition_w, h3_gpu_tensor *condition_b,
    h3_gpu_tensor *refiner_final, h3_gpu_tensor *video_w, h3_gpu_tensor *video_b,
    h3_gpu_tensor *audio_w, h3_gpu_tensor *audio_b, h3_gpu_tensor *time_in_w,
    h3_gpu_tensor *time_in_b, h3_gpu_tensor *time_out_w,
    h3_gpu_tensor *time_out_b, h3_gpu_tensor *block0_adaln_w,
    h3_gpu_tensor *block0_adaln_b, h3_gpu_tensor *final_norm,
    h3_gpu_tensor *final_adaln_w, h3_gpu_tensor *final_adaln_b,
    h3_gpu_tensor *final_video_w, h3_gpu_tensor *final_video_b,
    h3_gpu_tensor *final_audio_w, h3_gpu_tensor *final_audio_b) {
    run_block0_forward(test, t, refiner0, refiner1, block0, condition_w,
                       condition_b, refiner_final, video_w, video_b, audio_w,
                       audio_b, time_in_w, time_in_b, time_out_w, time_out_b,
                       block0_adaln_w, block0_adaln_b, 0);
    for (int layer = 1; layer < 50; layer++) {
        char prefix[64];
        char name[128];
        snprintf(prefix, sizeof(prefix), "blocks.%d.", layer);
        block_weights weight = load_block_ephemeral(test, prefix);
        snprintf(name, sizeof(name), "%sadaln_proj.linear.weight", prefix);
        h3_gpu_tensor *layer_adaln_w = weight_bf16_2d_ephemeral(
            test, name, MODALITIES * SLOTS * HIDDEN, TIME_DIM);
        snprintf(name, sizeof(name), "%sadaln_proj.linear.bias", prefix);
        h3_gpu_tensor *layer_adaln_b = weight_bf16_1d_ephemeral(
            test, name, MODALITIES * SLOTS * HIDDEN);
        run_dit_block_inplace(test, &weight, layer_adaln_w, layer_adaln_b,
                              t->time_bf16_silu, t->block_output, t, prefix);
        h3_gpu_tensor_free(layer_adaln_w);
        h3_gpu_tensor_free(layer_adaln_b);
        free_block_weights(&weight);
        if (layer == 1 || (layer + 1) % 10 == 0) {
            fprintf(stderr, "cuda DiT step smoke: %d/50 blocks\n", layer + 1);
        }
    }
    gpu_call(test, h3_gpu_linear_bf16(
                     test->gpu, t->final_modulation, t->time_bf16_silu,
                     final_adaln_w, final_adaln_b, 1, TIME_DIM, 2 * HIDDEN),
             "final AdaLN projection");
    gpu_call(test, h3_gpu_copy_bf16(
                     test->gpu, t->final_audio_input, 0, t->block_output,
                     TEXT_ROWS * HIDDEN, AUDIO_ROWS * HIDDEN),
             "slice final audio");
    gpu_call(test, h3_gpu_copy_bf16(
                     test->gpu, t->final_video_input, 0, t->block_output,
                     (TEXT_ROWS + AUDIO_ROWS) * HIDDEN, VIDEO_ROWS * HIDDEN),
             "slice final video");
    gpu_call(test, h3_gpu_adaln_bf16(
                     test->gpu, t->final_audio_norm, t->final_audio_input,
                     final_norm, t->final_modulation, t->final_rows, AUDIO_ROWS,
                     HIDDEN, 2, 0, 1, 1e-5f),
             "final audio AdaLN");
    gpu_call(test, h3_gpu_adaln_bf16(
                     test->gpu, t->final_video_norm, t->final_video_input,
                     final_norm, t->final_modulation, t->final_rows, VIDEO_ROWS,
                     HIDDEN, 2, 0, 1, 1e-5f),
             "final video AdaLN");
    gpu_call(test, h3_gpu_cast_bf16_to_f32(
                     test->gpu, t->final_audio_cast, t->final_audio_norm,
                     AUDIO_ROWS * HIDDEN),
             "final audio cast");
    gpu_call(test, h3_gpu_cast_bf16_to_f32(
                     test->gpu, t->final_video_cast, t->final_video_norm,
                     VIDEO_ROWS * HIDDEN),
             "final video cast");
    gpu_call(test, h3_gpu_linear_f32(
                     test->gpu, t->audio_rows_output, t->final_audio_cast,
                     final_audio_w, final_audio_b, AUDIO_ROWS, HIDDEN, 32),
             "final audio head");
    gpu_call(test, h3_gpu_linear_f32(
                     test->gpu, t->video_rows_output, t->final_video_cast,
                     final_video_w, final_video_b, VIDEO_ROWS, HIDDEN, 96),
             "final video head");
    gpu_call(test, h3_gpu_submit(test->gpu), "submit full DiT step");
}

static run_tensors alloc_run_tensors(test_context *test) {
    run_tensors t;
    memset(&t, 0, sizeof(t));
    t.text_qwen = new_bf16(test, (size_t)TEXT_ROWS * TEXT_DIM);
    t.video_rows = new_f32(test, (size_t)VIDEO_ROWS * 96);
    t.audio_rows = new_f32(test, (size_t)AUDIO_ROWS * 32);
    t.time_features = new_f32(test, TIME_INPUT);
    t.rope_cos = new_bf16(test, (size_t)SEQUENCE * ROPE_HALF);
    t.rope_sin = new_bf16(test, (size_t)SEQUENCE * ROPE_HALF);
    t.refined = new_bf16(test, (size_t)TEXT_ROWS * HIDDEN);
    t.refiner_norm = new_bf16(test, (size_t)TEXT_ROWS * HIDDEN);
    t.refiner_qkv = new_bf16(test, (size_t)TEXT_ROWS * INNER * 3);
    t.refiner_q = new_bf16(test, (size_t)TEXT_ROWS * INNER);
    t.refiner_k = new_bf16(test, (size_t)TEXT_ROWS * INNER);
    t.refiner_v = new_bf16(test, (size_t)TEXT_ROWS * INNER);
    t.refiner_heads = new_bf16(test, (size_t)TEXT_ROWS * INNER);
    t.refiner_branch = new_bf16(test, (size_t)TEXT_ROWS * HIDDEN);
    t.refiner_fc1 = new_bf16(test, (size_t)TEXT_ROWS * FFN * 2);
    t.refiner_activated = new_bf16(test, (size_t)TEXT_ROWS * FFN);
    t.video_f32 = new_f32(test, (size_t)VIDEO_ROWS * HIDDEN);
    t.audio_f32 = new_f32(test, (size_t)AUDIO_ROWS * HIDDEN);
    t.video_bf16 = new_bf16(test, (size_t)VIDEO_ROWS * HIDDEN);
    t.audio_bf16 = new_bf16(test, (size_t)AUDIO_ROWS * HIDDEN);
    t.hidden = new_bf16(test, (size_t)SEQUENCE * HIDDEN);
    t.time_hidden = new_f32(test, TIME_HIDDEN);
    t.time_silu = new_f32(test, TIME_HIDDEN);
    t.time_output = new_f32(test, TIME_DIM);
    t.time_bf16 = new_bf16(test, TIME_DIM);
    t.time_bf16_silu = new_bf16(test, TIME_DIM);
    t.modulation = new_bf16(test, (size_t)MODALITIES * SLOTS * HIDDEN);
    t.mod_attention = new_bf16(test, (size_t)SEQUENCE * HIDDEN);
    t.qkv = new_bf16(test, (size_t)SEQUENCE * INNER * 3);
    t.query = new_bf16(test, (size_t)SEQUENCE * INNER);
    t.key = new_bf16(test, (size_t)SEQUENCE * INNER);
    t.value = new_bf16(test, (size_t)SEQUENCE * INNER);
    t.attention_heads = new_bf16(test, (size_t)SEQUENCE * INNER);
    t.attention_output = new_bf16(test, (size_t)SEQUENCE * HIDDEN);
    t.after_attention = new_bf16(test, (size_t)SEQUENCE * HIDDEN);
    t.mod_mlp = new_bf16(test, (size_t)SEQUENCE * HIDDEN);
    t.fc1 = new_bf16(test, (size_t)SEQUENCE * FFN * 2);
    t.activated = new_bf16(test, (size_t)SEQUENCE * FFN);
    t.mlp_output = new_bf16(test, (size_t)SEQUENCE * HIDDEN);
    t.block_output = new_bf16(test, (size_t)SEQUENCE * HIDDEN);
    t.final_modulation = new_bf16(test, 2u * HIDDEN);
    t.final_audio_input = new_bf16(test, (size_t)AUDIO_ROWS * HIDDEN);
    t.final_video_input = new_bf16(test, (size_t)VIDEO_ROWS * HIDDEN);
    t.final_audio_norm = new_bf16(test, (size_t)AUDIO_ROWS * HIDDEN);
    t.final_video_norm = new_bf16(test, (size_t)VIDEO_ROWS * HIDDEN);
    t.final_audio_cast = new_f32(test, (size_t)AUDIO_ROWS * HIDDEN);
    t.final_video_cast = new_f32(test, (size_t)VIDEO_ROWS * HIDDEN);
    t.audio_rows_output = new_f32(test, (size_t)AUDIO_ROWS * 32);
    t.video_rows_output = new_f32(test, (size_t)VIDEO_ROWS * 96);
    uint32_t zero_rows[AUDIO_ROWS];
    memset(zero_rows, 0, sizeof(zero_rows));
    t.final_rows =
        own(test, h3_gpu_tensor_from_u32(test->gpu, zero_rows, AUDIO_ROWS));
    return t;
}

static void require_finite_bf16(const uint16_t *values, size_t count,
                                const char *label) {
    size_t bad = 0;
    for (size_t index = 0; index < count; index++) {
        float value = bf16_to_f32(values[index]);
        if (!isfinite(value)) bad++;
    }
    if (bad) {
        fprintf(stderr,
                "FAIL: %s has %zu non-finite values out of %zu\n", label, bad,
                count);
        exit(1);
    }
}

static void require_finite_f32(const float *values, size_t count,
                               const char *label) {
    size_t bad = 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) bad++;
    }
    if (bad) {
        fprintf(stderr,
                "FAIL: %s has %zu non-finite values out of %zu\n", label, bad,
                count);
        exit(1);
    }
}

static void cleanup(test_context *test) {
    for (size_t index = 0; index < test->owned_count; index++)
        h3_gpu_tensor_free(test->owned[index]);
    h3_gpu_free(test->gpu);
    h3_weight_store_free(test->weights);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : getenv("H3_MODEL_ROOT");
    int full_step = argc > 2 && strcmp(argv[2], "full") == 0;
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

    char weights_path[1024];
    snprintf(weights_path, sizeof(weights_path), "%s/FL2VA/transformer",
             model_root);

    test_context test;
    memset(&test, 0, sizeof(test));
    char error[512];
    test.weights = h3_weight_store_open(weights_path, error, sizeof(error));
    if (!test.weights) {
        fprintf(stderr, "skip: cannot open transformer weights: %s\n", error);
        return 0;
    }
    test.gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!test.gpu) die(error);

    block_weights refiner0 = load_block(&test, "token_refiner.blocks.0.");
    block_weights refiner1 = load_block(&test, "token_refiner.blocks.1.");
    block_weights block0 = load_block(&test, "blocks.0.");
    h3_gpu_tensor *condition_w =
        weight_bf16_2d(&test, "condition_proj.weight", HIDDEN, TEXT_DIM);
    h3_gpu_tensor *condition_b =
        weight_bf16_1d(&test, "condition_proj.bias", HIDDEN);
    h3_gpu_tensor *refiner_final =
        weight_bf16_1d(&test, "token_refiner.final_norm.weight", HIDDEN);
    h3_gpu_tensor *video_w =
        weight_f32_2d(&test, "video_patch_proj.weight", HIDDEN, 96);
    h3_gpu_tensor *video_b =
        weight_f32_1d(&test, "video_patch_proj.bias", HIDDEN);
    h3_gpu_tensor *audio_w =
        weight_f32_2d(&test, "audio_patch_proj.weight", HIDDEN, 32);
    h3_gpu_tensor *audio_b =
        weight_f32_1d(&test, "audio_patch_proj.bias", HIDDEN);
    h3_gpu_tensor *time_in_w =
        weight_f32_2d(&test, "time_embedder.proj_in.weight", TIME_HIDDEN,
                        TIME_INPUT);
    h3_gpu_tensor *time_in_b =
        weight_f32_1d(&test, "time_embedder.proj_in.bias", TIME_HIDDEN);
    h3_gpu_tensor *time_out_w =
        weight_f32_2d(&test, "time_embedder.proj_out.weight", TIME_DIM,
                        TIME_HIDDEN);
    h3_gpu_tensor *time_out_b =
        weight_f32_1d(&test, "time_embedder.proj_out.bias", TIME_DIM);
    h3_gpu_tensor *adaln_w = weight_bf16_2d(
        &test, "blocks.0.adaln_proj.linear.weight",
        MODALITIES * SLOTS * HIDDEN, TIME_DIM);
    h3_gpu_tensor *adaln_b = weight_bf16_1d(
        &test, "blocks.0.adaln_proj.linear.bias", MODALITIES * SLOTS * HIDDEN);
    h3_gpu_tensor *final_norm =
        weight_bf16_1d(&test, "final_layer.norm.weight", HIDDEN);
    h3_gpu_tensor *final_adaln_w = weight_bf16_2d(
        &test, "final_layer.adaln_proj.linear.weight", 2 * HIDDEN, TIME_DIM);
    h3_gpu_tensor *final_adaln_b = weight_bf16_1d(
        &test, "final_layer.adaln_proj.linear.bias", 2 * HIDDEN);
    h3_gpu_tensor *final_video_w =
        weight_f32_2d(&test, "final_layer.video_out.weight", 96, HIDDEN);
    h3_gpu_tensor *final_video_b =
        weight_f32_1d(&test, "final_layer.video_out.bias", 96);
    h3_gpu_tensor *final_audio_w =
        weight_f32_2d(&test, "final_layer.audio_out.weight", 32, HIDDEN);
    h3_gpu_tensor *final_audio_b =
        weight_f32_1d(&test, "final_layer.audio_out.bias", 32);

    run_tensors tensors = alloc_run_tensors(&test);

    float *text_f32 = malloc((size_t)TEXT_ROWS * TEXT_DIM * sizeof(*text_f32));
    float *video_f32 =
        malloc((size_t)VIDEO_ROWS * 96 * sizeof(*video_f32));
    float *audio_f32 =
        malloc((size_t)AUDIO_ROWS * 32 * sizeof(*audio_f32));
    float *time_f32 = malloc(TIME_INPUT * sizeof(*time_f32));
    float *rope_cos_f32 =
        malloc((size_t)SEQUENCE * ROPE_HALF * sizeof(*rope_cos_f32));
    float *rope_sin_f32 =
        malloc((size_t)SEQUENCE * ROPE_HALF * sizeof(*rope_sin_f32));
    uint32_t row_map[SEQUENCE];
    size_t output_count = (size_t)SEQUENCE * HIDDEN;
    uint16_t *first = malloc(output_count * sizeof(*first));
    uint16_t *second = malloc(output_count * sizeof(*second));
    require(text_f32 && video_f32 && audio_f32 && time_f32 && rope_cos_f32 &&
                rope_sin_f32 && first && second,
            "host buffer allocation failed");

    fill_host_inputs(text_f32, (size_t)TEXT_ROWS * TEXT_DIM, video_f32,
                     (size_t)VIDEO_ROWS * 96, audio_f32,
                     (size_t)AUDIO_ROWS * 32, time_f32, TIME_INPUT,
                     rope_cos_f32, rope_sin_f32,
                     (size_t)SEQUENCE * ROPE_HALF, row_map);
    tensors.row_map =
        own(&test, h3_gpu_tensor_from_u32(test.gpu, row_map, SEQUENCE));

    if (full_step) {
        const size_t audio_out_count = (size_t)AUDIO_ROWS * 32;
        const size_t video_out_count = (size_t)VIDEO_ROWS * 96;
        const size_t block_out_count = (size_t)SEQUENCE * HIDDEN;
        float *audio_first = malloc(audio_out_count * sizeof(*audio_first));
        float *audio_second = malloc(audio_out_count * sizeof(*audio_second));
        float *video_first = malloc(video_out_count * sizeof(*video_first));
        float *video_second = malloc(video_out_count * sizeof(*video_second));
        uint16_t *block_first = malloc(block_out_count * sizeof(*block_first));
        uint16_t *block_second = malloc(block_out_count * sizeof(*block_second));
        require(audio_first && audio_second && video_first && video_second &&
                    block_first && block_second,
                "full-step host buffer allocation failed");

        upload_inputs(&test, &tensors, text_f32, video_f32, audio_f32, time_f32,
                      rope_cos_f32, rope_sin_f32);
        run_full_step_forward(
            &test, &tensors, &refiner0, &refiner1, &block0, condition_w,
            condition_b, refiner_final, video_w, video_b, audio_w, audio_b,
            time_in_w, time_in_b, time_out_w, time_out_b, adaln_w, adaln_b,
            final_norm, final_adaln_w, final_adaln_b, final_video_w,
            final_video_b, final_audio_w, final_audio_b);
        require(h3_gpu_tensor_read_f32(tensors.audio_rows_output, audio_first,
                                       audio_out_count),
                "read first audio output");
        require(h3_gpu_tensor_read_f32(tensors.video_rows_output, video_first,
                                       video_out_count),
                "read first video output");
        require(h3_gpu_tensor_read_bf16(tensors.block_output, block_first,
                                        block_out_count),
                "read first block output");
        require_finite_f32(audio_first, audio_out_count, "audio_rows_output");
        require_finite_f32(video_first, video_out_count, "video_rows_output");
        require_finite_bf16(block_first, block_out_count, "block_output");

        upload_inputs(&test, &tensors, text_f32, video_f32, audio_f32, time_f32,
                      rope_cos_f32, rope_sin_f32);
        run_full_step_forward(
            &test, &tensors, &refiner0, &refiner1, &block0, condition_w,
            condition_b, refiner_final, video_w, video_b, audio_w, audio_b,
            time_in_w, time_in_b, time_out_w, time_out_b, adaln_w, adaln_b,
            final_norm, final_adaln_w, final_adaln_b, final_video_w,
            final_video_b, final_audio_w, final_audio_b);
        require(h3_gpu_tensor_read_f32(tensors.audio_rows_output, audio_second,
                                       audio_out_count),
                "read second audio output");
        require(h3_gpu_tensor_read_f32(tensors.video_rows_output, video_second,
                                       video_out_count),
                "read second video output");
        require(h3_gpu_tensor_read_bf16(tensors.block_output, block_second,
                                        block_out_count),
                "read second block output");
        require(memcmp(audio_first, audio_second,
                       audio_out_count * sizeof(*audio_first)) == 0,
                "audio output is not deterministic");
        require(memcmp(video_first, video_second,
                       video_out_count * sizeof(*video_first)) == 0,
                "video output is not deterministic");
        require(memcmp(block_first, block_second,
                       block_out_count * sizeof(*block_first)) == 0,
                "block output is not deterministic");

        h3_gpu_stats stats;
        require(h3_gpu_get_stats(test.gpu, &stats), "read GPU stats");
        printf("cuda dit step smoke: %.2f MiB peak live, %llu direct, "
               "%llu SDPA, %llu submissions\n",
               (double)stats.peak_live_bytes / (1024.0 * 1024.0),
               (unsigned long long)stats.direct_dispatches,
               (unsigned long long)stats.mps_sdpa_dispatches,
               (unsigned long long)stats.submissions);

        free(audio_first);
        free(audio_second);
        free(video_first);
        free(video_second);
        free(block_first);
        free(block_second);
        cleanup(&test);
        puts("ok: CUDA DiT 50-block step smoke passed (real weights, no MLX)");
        return 0;
    }

    upload_inputs(&test, &tensors, text_f32, video_f32, audio_f32, time_f32,
                  rope_cos_f32, rope_sin_f32);
    run_block0_forward(&test, &tensors, &refiner0, &refiner1, &block0,
                       condition_w, condition_b, refiner_final, video_w, video_b,
                       audio_w, audio_b, time_in_w, time_in_b, time_out_w,
                       time_out_b, adaln_w, adaln_b, 1);
    require(h3_gpu_tensor_read_bf16(tensors.block_output, first, output_count),
            "read first block output");
    require_finite_bf16(first, output_count, "block_output");

    upload_inputs(&test, &tensors, text_f32, video_f32, audio_f32, time_f32,
                  rope_cos_f32, rope_sin_f32);
    run_block0_forward(&test, &tensors, &refiner0, &refiner1, &block0,
                       condition_w, condition_b, refiner_final, video_w, video_b,
                       audio_w, audio_b, time_in_w, time_in_b, time_out_w,
                       time_out_b, adaln_w, adaln_b, 1);
    require(h3_gpu_tensor_read_bf16(tensors.block_output, second, output_count),
            "read second block output");
    require(memcmp(first, second, output_count * sizeof(*first)) == 0,
            "block output is not deterministic across repeated runs");

    h3_gpu_stats stats;
    require(h3_gpu_get_stats(test.gpu, &stats), "read GPU stats");
    printf("cuda dit block smoke: %.2f MiB peak live, %llu direct, "
           "%llu SDPA, %llu submissions\n",
           (double)stats.peak_live_bytes / (1024.0 * 1024.0),
           (unsigned long long)stats.direct_dispatches,
           (unsigned long long)stats.mps_sdpa_dispatches,
           (unsigned long long)stats.submissions);

    free(text_f32);
    free(video_f32);
    free(audio_f32);
    free(time_f32);
    free(rope_cos_f32);
    free(rope_sin_f32);
    free(first);
    free(second);
    cleanup(&test);
    puts("ok: CUDA DiT block-0 smoke passed (real weights, no MLX fixture)");
    return 0;
}
