#include "h3_gpu.h"

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

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

static void check(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message);
    failures++;
}

static float silu_ref(float value) {
    return value / (1.0f + expf(-value));
}

static void linear_ref(const float *input, const float *weight,
                       const float *bias, float *output, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < output_dim; column++) {
            float sum = bias ? bias[column] : 0.0f;
            for (uint32_t k = 0; k < input_dim; k++) {
                sum = fmaf(input[(size_t)row * input_dim + k],
                           weight[(size_t)column * input_dim + k], sum);
            }
            output[(size_t)row * output_dim + column] = sum;
        }
    }
}

static void adaln_ref(const float *input, const float *weight,
                      const float *modulation, const uint32_t *row_map,
                      float *output, uint32_t rows, uint32_t width,
                      uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *row_input = input + (size_t)row * width;
        float *row_output = output + (size_t)row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float value = row_input[column];
            sum = fmaf(value, value, sum);
        }
        float inverse = 1.0f / sqrtf(sum / (float)width + epsilon);
        size_t base = (size_t)row_map[row] * slots * width;
        for (uint32_t column = 0; column < width; column++) {
            float normalized =
                row_input[column] * inverse * weight[column];
            float shift =
                modulation[base + shift_slot * width + column];
            float scale =
                modulation[base + scale_slot * width + column];
            row_output[column] = normalized * (1.0f + scale) + shift;
        }
    }
}

static void gate_ref(const float *residual, const float *branch,
                     const float *modulation, const uint32_t *row_map,
                     float *output, uint32_t rows, uint32_t width,
                     uint32_t slots, uint32_t gate_slot) {
    for (uint32_t row = 0; row < rows; row++) {
        size_t base = (size_t)row_map[row] * slots * width;
        for (uint32_t column = 0; column < width; column++) {
            size_t index = (size_t)row * width + column;
            float gate = modulation[base + gate_slot * width + column];
            output[index] = residual[index] + branch[index] * gate;
        }
    }
}

static void swiglu_ref(const float *fused, float *output, uint32_t rows,
                       uint32_t width) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < width; column++) {
            size_t base = (size_t)row * width * 2u;
            float gate = fused[base + column];
            float up = fused[base + width + column];
            output[(size_t)row * width + column] =
                gate / (1.0f + expf(-gate)) * up;
        }
    }
}

static float gelu_ref(float value, int approximate) {
    if (approximate) {
        float inner = 0.7978845608028654f *
                      (value + 0.044715f * value * value * value);
        if (inner <= -10.0f) return 0.0f;
        if (inner >= 10.0f) return value;
        return 0.5f * value * (1.0f + tanhf(inner));
    }
    if (value <= -10.0f) return 0.0f;
    if (value >= 10.0f) return value;
    return 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
}

static void grouped_qkv_rope_ref(const float *qkv, const float *q_weight,
                                 const float *k_weight, const float *rope_cos,
                                 const float *rope_sin, float *query,
                                 float *key, float *value, uint32_t sequence,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon) {
    uint32_t inner = heads * head_dim;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t row_base = row * inner * 3u;
            uint32_t q_base = row_base + head * head_dim * 3u;
            uint32_t k_base = q_base + head_dim;
            uint32_t v_base = k_base + head_dim;
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float q = qkv[q_base + d];
                float k = qkv[k_base + d];
                q_sum = fmaf(q, q, q_sum);
                k_sum = fmaf(k, k, k_sum);
            }
            float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
            float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                float q0 = qkv[q_base + dimension] * q_inverse *
                           q_weight[dimension];
                float k0 = qkv[k_base + dimension] * k_inverse *
                           k_weight[dimension];
                if (dimension < rope_half) {
                    uint32_t pair = dimension + rope_half;
                    float q1 = qkv[q_base + pair] * q_inverse *
                               q_weight[pair];
                    float k1 = qkv[k_base + pair] * k_inverse *
                               k_weight[pair];
                    float c = rope_cos[row * rope_half + dimension];
                    float s = rope_sin[row * rope_half + dimension];
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else if (dimension < rope_half * 2u) {
                    uint32_t pair = dimension - rope_half;
                    float q1 = qkv[q_base + pair] * q_inverse *
                               q_weight[pair];
                    float k1 = qkv[k_base + pair] * k_inverse *
                               k_weight[pair];
                    float c = rope_cos[row * rope_half + pair];
                    float s = rope_sin[row * rope_half + pair];
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                uint32_t output_index =
                    (row * heads + head) * head_dim + dimension;
                query[output_index] = q0;
                key[output_index] = k0;
                value[output_index] = qkv[v_base + dimension];
            }
        }
    }
}

static void vision_qkv_rope_ref(const float *qkv, const float *rope_cos,
                                const float *rope_sin, float *query,
                                float *key, float *value, uint32_t sequence,
                                uint32_t heads, uint32_t head_dim,
                                uint32_t rope_half) {
    uint32_t inner = heads * head_dim;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            size_t row_base = (size_t)row * inner * 3u;
            size_t q_base = row_base + (size_t)head * head_dim;
            size_t k_base = row_base + inner + (size_t)head * head_dim;
            size_t v_base = row_base + inner * 2u + (size_t)head * head_dim;
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                uint32_t pair = dimension < rope_half ? dimension + rope_half
                                                     : dimension - rope_half;
                float c = rope_cos[row * rope_half + (dimension % rope_half)];
                float s = rope_sin[row * rope_half + (dimension % rope_half)];
                float q0 = qkv[q_base + dimension];
                float k0 = qkv[k_base + dimension];
                float q1 = qkv[q_base + pair];
                float k1 = qkv[k_base + pair];
                float qr =
                    dimension < rope_half ? q0 * c - q1 * s : q0 * c + q1 * s;
                float kr =
                    dimension < rope_half ? k0 * c - k1 * s : k0 * c + k1 * s;
                size_t output_index =
                    ((size_t)row * heads + head) * head_dim + dimension;
                query[output_index] = qr;
                key[output_index] = kr;
                value[output_index] = qkv[v_base + dimension];
            }
        }
    }
}

/* Metal h3_video_qkv_rope_f32: interleaved QKV + per-head RMS + RoPE. */
static void video_qkv_rope_ref(const float *qkv, const float *rope_cos,
                               const float *rope_sin, float *query, float *key,
                               float *value, uint32_t sequence, uint32_t heads,
                               uint32_t head_dim, uint32_t rope_half,
                               float epsilon) {
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            size_t base =
                ((size_t)row * heads + head) * head_dim * 3u;
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float q = qkv[base + d];
                float k = qkv[base + head_dim + d];
                q_sum = fmaf(q, q, q_sum);
                k_sum = fmaf(k, k, k_sum);
            }
            float qi = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
            float ki = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                float q0 = qkv[base + dimension] * qi;
                float k0 = qkv[base + head_dim + dimension] * ki;
                if (dimension < rope_half) {
                    uint32_t pair = dimension + rope_half;
                    float q1 = qkv[base + pair] * qi;
                    float k1 = qkv[base + head_dim + pair] * ki;
                    float c = rope_cos[row * rope_half + dimension];
                    float s = rope_sin[row * rope_half + dimension];
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else if (dimension < rope_half * 2u) {
                    uint32_t pair = dimension - rope_half;
                    float q1 = qkv[base + pair] * qi;
                    float k1 = qkv[base + head_dim + pair] * ki;
                    float c = rope_cos[row * rope_half + pair];
                    float s = rope_sin[row * rope_half + pair];
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                size_t output_index =
                    ((size_t)row * heads + head) * head_dim + dimension;
                query[output_index] = q0;
                key[output_index] = k0;
                value[output_index] = qkv[base + head_dim * 2u + dimension];
            }
        }
    }
}

static void sdpa_ref(const float *query, const float *key, const float *value,
                     float *output, uint32_t sequence, uint32_t heads,
                     uint32_t head_dim, float scale) {
    for (uint32_t head = 0; head < heads; head++) {
        for (uint32_t q_row = 0; q_row < sequence; q_row++) {
            float scores[256];
            float max_score = -INFINITY;
            for (uint32_t k_row = 0; k_row < sequence; k_row++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    size_t q_index =
                        ((size_t)q_row * heads + head) * head_dim + d;
                    size_t k_index =
                        ((size_t)k_row * heads + head) * head_dim + d;
                    dot = fmaf(query[q_index], key[k_index], dot);
                }
                scores[k_row] = dot * scale;
                if (scores[k_row] > max_score) max_score = scores[k_row];
            }
            float sum = 0.0f;
            for (uint32_t k_row = 0; k_row < sequence; k_row++) {
                scores[k_row] = expf(scores[k_row] - max_score);
                sum += scores[k_row];
            }
            float inverse = 1.0f / sum;
            for (uint32_t d = 0; d < head_dim; d++) {
                float accumulated = 0.0f;
                for (uint32_t k_row = 0; k_row < sequence; k_row++) {
                    size_t v_index =
                        ((size_t)k_row * heads + head) * head_dim + d;
                    accumulated = fmaf(scores[k_row] * inverse, value[v_index],
                                       accumulated);
                }
                size_t o_index =
                    ((size_t)q_row * heads + head) * head_dim + d;
                output[o_index] = accumulated;
            }
        }
    }
}

static float bf16_roundtrip(float value) {
    return bf16_to_f32(f32_to_bf16(value));
}

static void head_rms_norm_ref(const float *input, const float *weight,
                              float *output, uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) {
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            size_t base = ((size_t)row * heads + head) * head_dim;
            float sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float value = input[base + d];
                sum = fmaf(value, value, sum);
            }
            float inverse = 1.0f / sqrtf(sum / (float)head_dim + epsilon);
            for (uint32_t d = 0; d < head_dim; d++) {
                output[base + d] = input[base + d] * inverse * weight[d];
            }
        }
    }
}

static void rope_text_ref(float *query, float *key, const float *rope_cos,
                          const float *rope_sin, uint32_t sequence,
                          uint32_t query_heads, uint32_t kv_heads,
                          uint32_t head_dim) {
    uint32_t half_dim = head_dim / 2u;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < query_heads; head++) {
            size_t base = ((size_t)row * query_heads + head) * head_dim;
            for (uint32_t d = 0; d < half_dim; d++) {
                float first = query[base + d];
                float second = query[base + half_dim + d];
                float c = rope_cos[row * half_dim + d];
                float s = rope_sin[row * half_dim + d];
                query[base + d] = first * c - second * s;
                query[base + half_dim + d] = second * c + first * s;
            }
        }
        for (uint32_t head = 0; head < kv_heads; head++) {
            size_t base = ((size_t)row * kv_heads + head) * head_dim;
            for (uint32_t d = 0; d < half_dim; d++) {
                float first = key[base + d];
                float second = key[base + half_dim + d];
                float c = rope_cos[row * half_dim + d];
                float s = rope_sin[row * half_dim + d];
                key[base + d] = first * c - second * s;
                key[base + half_dim + d] = second * c + first * s;
            }
        }
    }
}

static void gqa_causal_ref(const float *query, const float *key,
                           const float *value, float *output, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale) {
    uint32_t q_per_kv = query_heads / kv_heads;
    for (uint32_t q_row = 0; q_row < sequence; q_row++) {
        uint32_t key_count = q_row + 1u;
        for (uint32_t q_head = 0; q_head < query_heads; q_head++) {
            uint32_t kv_head = q_head / q_per_kv;
            size_t q_base = ((size_t)q_row * query_heads + q_head) * head_dim;
            float scaled_query[128];
            for (uint32_t d = 0; d < head_dim; d++) {
                scaled_query[d] =
                    bf16_roundtrip(query[q_base + d] * scale);
            }
            float scores[256];
            float max_score = -INFINITY;
            for (uint32_t k_row = 0; k_row < key_count; k_row++) {
                size_t k_base =
                    ((size_t)k_row * kv_heads + kv_head) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot = fmaf(scaled_query[d], key[k_base + d], dot);
                }
                scores[k_row] = dot;
                if (dot > max_score) max_score = dot;
            }
            float sum = 0.0f;
            for (uint32_t k_row = 0; k_row < key_count; k_row++) {
                scores[k_row] = expf(scores[k_row] - max_score);
                sum += scores[k_row];
            }
            float inverse = 1.0f / sum;
            for (uint32_t d = 0; d < head_dim; d++) {
                float accumulated = 0.0f;
                for (uint32_t k_row = 0; k_row < key_count; k_row++) {
                    size_t v_base =
                        ((size_t)k_row * kv_heads + kv_head) * head_dim;
                    accumulated = fmaf(scores[k_row] * inverse, value[v_base + d],
                                       accumulated);
                }
                output[q_base + d] = accumulated;
            }
        }
    }
}

static void silu_mul_ref(const float *gate, const float *up, float *output,
                         size_t count) {
    for (size_t i = 0; i < count; i++) {
        output[i] = silu_ref(gate[i]) * up[i];
    }
}

static void rms_norm_ref(const float *input, const float *weight, float *output,
                         uint32_t rows, uint32_t width, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *row_input = input + (size_t)row * width;
        float *row_output = output + (size_t)row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float value = row_input[column];
            sum = fmaf(value, value, sum);
        }
        float inverse = 1.0f / sqrtf(sum / (float)width + epsilon);
        for (uint32_t column = 0; column < width; column++) {
            row_output[column] =
                row_input[column] * inverse * weight[column];
        }
    }
}

static void quantize_bf16_int8_rows_ref(const float *input, int8_t *output,
                                       float *scales, uint32_t rows,
                                       uint32_t columns, float clip) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *row_input = input + (size_t)row * columns;
        float max_abs = 0.0f;
        for (uint32_t column = 0; column < columns; column++) {
            float value = fabsf(row_input[column]);
            if (value > max_abs) max_abs = value;
        }
        float clipped_max = max_abs * clip;
        float scale =
            clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
        float inverse =
            clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
        scales[row] = scale;
        int8_t *row_output = output + (size_t)row * columns;
        for (uint32_t column = 0; column < columns; column++) {
            int quantized = (int)rintf(row_input[column] * inverse);
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            row_output[column] = (int8_t)quantized;
        }
    }
}

static void linear_int8_ref(const int8_t *input, const float *input_scales,
                            const int8_t *weight, const float *weight_scales,
                            float *output, uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < output_dim; column++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < input_dim; k++) {
                sum += (int32_t)input[(size_t)row * input_dim + k] *
                       (int32_t)weight[(size_t)column * input_dim + k];
            }
            output[(size_t)row * output_dim + column] =
                (float)sum * input_scales[row] * weight_scales[column];
        }
    }
}

static void layer_norm_ref(const float *input, const float *weight,
                           const float *bias, float *output, uint32_t rows,
                           uint32_t width, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *row_input = input + (size_t)row * width;
        float *row_output = output + (size_t)row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++)
            sum += row_input[column];
        float mean = sum / (float)width;
        float sq_sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float centered = row_input[column] - mean;
            sq_sum = fmaf(centered, centered, sq_sum);
        }
        float inverse = 1.0f / sqrtf(sq_sum / (float)width + epsilon);
        for (uint32_t column = 0; column < width; column++) {
            float normalized = (row_input[column] - mean) * inverse;
            row_output[column] =
                fmaf(normalized, weight[column], bias[column]);
        }
    }
}

static void text_qk_rope_ref(const float *query_input, const float *key_input,
                             const float *q_weight, const float *k_weight,
                             const float *rope_cos, const float *rope_sin,
                             float *query_output, float *key_output,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) {
    uint32_t half_dim = head_dim / 2u;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < query_heads; head++) {
            size_t q_base = ((size_t)row * query_heads + head) * head_dim;
            float q_sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float value = query_input[q_base + d];
                q_sum = fmaf(value, value, q_sum);
            }
            float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                uint32_t pair = dimension < half_dim ? dimension + half_dim
                                                     : dimension - half_dim;
                float c = rope_cos[row * half_dim + (dimension % half_dim)];
                float s = rope_sin[row * half_dim + (dimension % half_dim)];
                float q0 = query_input[q_base + dimension] * q_inverse *
                           q_weight[dimension];
                float q1 = query_input[q_base + pair] * q_inverse *
                           q_weight[pair];
                query_output[q_base + dimension] =
                    dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
            }
        }
        for (uint32_t head = 0; head < kv_heads; head++) {
            size_t k_base = ((size_t)row * kv_heads + head) * head_dim;
            float k_sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float value = key_input[k_base + d];
                k_sum = fmaf(value, value, k_sum);
            }
            float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                uint32_t pair = dimension < half_dim ? dimension + half_dim
                                                     : dimension - half_dim;
                float c = rope_cos[row * half_dim + (dimension % half_dim)];
                float s = rope_sin[row * half_dim + (dimension % half_dim)];
                float k0 = key_input[k_base + dimension] * k_inverse *
                           k_weight[dimension];
                float k1 = key_input[k_base + pair] * k_inverse *
                           k_weight[pair];
                key_output[k_base + dimension] =
                    dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
            }
        }
    }
}

#define STAGE_LANES 6
#define STAGE_LANE_ELEMENTS ((size_t)3 << 20)

typedef struct {
    const char *path;
    h3_gpu_tensor *tensor;
    uint64_t offset;
    int ok;
} stage_lane;

static uint16_t stage_pattern(size_t index) {
    return (uint16_t)(((uint32_t)index * 2654435761u) >> 16u);
}

static void *stage_lane_main(void *raw) {
    stage_lane *lane = (stage_lane *)raw;
    char error[256];
    error[0] = '\0';
    lane->ok = h3_gpu_tensor_read_file_bf16(lane->tensor, lane->path,
                                            lane->offset, STAGE_LANE_ELEMENTS,
                                            error, sizeof(error));
    if (!lane->ok) fprintf(stderr, "lane read: %s\n", error);
    return NULL;
}

/* The Qwen loader reads one layer's tensors from several lanes at once, so the
 * staging buffers behind h3_gpu_tensor_read_file_bf16 must not be shared. When
 * they were, lanes overwrote each other between the pread and the copy and the
 * text encoder ran on scrambled weights (docs/PERF_BASELINE.md, 2026-08-25). */
static void check_concurrent_staged_reads(h3_gpu *gpu) {
    /* Small chunks so every lane's read spans more than one staging chunk. */
    setenv("H3_LOAD_STAGE_MIB", "8", 1);
    char path[] = "/tmp/h3_stage_lanes_XXXXXX";
    int fd = mkstemp(path);
    check(fd >= 0, "staging race temp file");
    if (fd < 0) return;
    uint16_t *expected =
        (uint16_t *)malloc(STAGE_LANE_ELEMENTS * STAGE_LANES * sizeof(uint16_t));
    uint16_t *actual =
        (uint16_t *)malloc(STAGE_LANE_ELEMENTS * sizeof(uint16_t));
    check(expected && actual, "staging race host buffers");
    if (!expected || !actual) {
        free(expected);
        free(actual);
        close(fd);
        unlink(path);
        return;
    }
    for (size_t i = 0; i < STAGE_LANE_ELEMENTS * STAGE_LANES; i++)
        expected[i] = stage_pattern(i);
    size_t file_bytes = STAGE_LANE_ELEMENTS * STAGE_LANES * sizeof(uint16_t);
    check(write(fd, expected, file_bytes) == (ssize_t)file_bytes,
          "staging race file write");
    close(fd);

    stage_lane lanes[STAGE_LANES];
    pthread_t workers[STAGE_LANES];
    h3_gpu_tensor *tensors[STAGE_LANES];
    for (int lane = 0; lane < STAGE_LANES; lane++) {
        tensors[lane] = h3_gpu_tensor_new_bf16(gpu, STAGE_LANE_ELEMENTS);
        check(tensors[lane] != NULL, "staging race tensor alloc");
        lanes[lane].path = path;
        lanes[lane].tensor = tensors[lane];
        lanes[lane].offset =
            (uint64_t)lane * STAGE_LANE_ELEMENTS * sizeof(uint16_t);
    }
    /* A race only sometimes loses: the shared-buffer build failed two rounds in
     * three, so one round is not enough of a gate. */
    int mismatched_lanes = 0;
    int failed_reads = 0;
    for (int round = 0; round < 4; round++) {
        for (int lane = 0; lane < STAGE_LANES; lane++) {
            lanes[lane].ok = 0;
            check(pthread_create(&workers[lane], NULL, stage_lane_main,
                                 &lanes[lane]) == 0,
                  "staging race lane spawn");
        }
        for (int lane = 0; lane < STAGE_LANES; lane++)
            pthread_join(workers[lane], NULL);
        check(h3_gpu_submit(gpu), "staging race submit");
        for (int lane = 0; lane < STAGE_LANES; lane++) {
            if (!lanes[lane].ok) {
                failed_reads++;
                continue;
            }
            if (!tensors[lane]) continue;
            check(h3_gpu_tensor_read_bf16(tensors[lane], actual,
                                          STAGE_LANE_ELEMENTS),
                  "staging race readback");
            if (memcmp(actual, expected + (size_t)lane * STAGE_LANE_ELEMENTS,
                       STAGE_LANE_ELEMENTS * sizeof(uint16_t)) != 0)
                mismatched_lanes++;
        }
    }
    check(failed_reads == 0, "concurrent staged reads all succeed");
    check(mismatched_lanes == 0, "concurrent staged reads keep their own bytes");
    if (mismatched_lanes || failed_reads) {
        fprintf(stderr,
                "staging race: %d lane read(s) failed, %d read foreign bytes\n",
                failed_reads, mismatched_lanes);
    }
    for (int lane = 0; lane < STAGE_LANES; lane++)
        h3_gpu_tensor_free(tensors[lane]);
    free(expected);
    free(actual);
    unlink(path);
    unsetenv("H3_LOAD_STAGE_MIB");
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    check(gpu != NULL, "h3_gpu_create");
    if (!gpu) {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }

    check(h3_gpu_begin(gpu), "h3_gpu_begin");

    const uint32_t silu_count = 512;
    float silu_host[512];
    uint16_t silu_in_bf16[512];
    uint16_t silu_out_bf16[512];
    for (uint32_t i = 0; i < silu_count; i++) {
        silu_host[i] = -3.0f + (float)i * (6.0f / (float)(silu_count - 1));
        silu_in_bf16[i] = f32_to_bf16(silu_host[i]);
    }

    h3_gpu_tensor *silu_in =
        h3_gpu_tensor_from_bf16(gpu, silu_in_bf16, silu_count);
    h3_gpu_tensor *silu_out = h3_gpu_tensor_new_bf16(gpu, silu_count);
    check(silu_in && silu_out, "silu tensor alloc");
    if (silu_in && silu_out) {
        check(h3_gpu_silu_bf16(gpu, silu_out, silu_in, silu_count), "silu");
        check(h3_gpu_submit(gpu), "submit silu");
        check(h3_gpu_tensor_read_bf16(silu_out, silu_out_bf16, silu_count),
              "read silu");
        for (uint32_t i = 0; i < silu_count; i++) {
            float got = bf16_to_f32(silu_out_bf16[i]);
            float expected = silu_ref(silu_host[i]);
            if (fabsf(got - expected) >= 2e-2f) {
                fprintf(stderr,
                        "FAIL: silu mismatch at %u got=%f expected=%f\n", i, got,
                        expected);
                failures++;
                break;
            }
        }
    }

    const uint32_t rows = 4;
    const uint32_t width = 128;
    const float epsilon = 1e-5f;
    const size_t count = (size_t)rows * width;
    float norm_host[count];
    float weight_host[width];
    uint16_t norm_in_bf16[count];
    uint16_t weight_bf16[width];
    uint16_t norm_out_bf16[count];
    float norm_ref[count];

    for (uint32_t column = 0; column < width; column++) {
        weight_host[column] = 0.5f + 0.01f * (float)column;
        weight_bf16[column] = f32_to_bf16(weight_host[column]);
    }
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < width; column++) {
            size_t index = (size_t)row * width + column;
            norm_host[index] = sinf((float)index * 0.07f);
            norm_in_bf16[index] = f32_to_bf16(norm_host[index]);
        }
    }
    rms_norm_ref(norm_host, weight_host, norm_ref, rows, width, epsilon);

    h3_gpu_tensor *norm_in = h3_gpu_tensor_from_bf16(gpu, norm_in_bf16, count);
    h3_gpu_tensor *norm_weight =
        h3_gpu_tensor_from_bf16(gpu, weight_bf16, width);
    h3_gpu_tensor *norm_out = h3_gpu_tensor_new_bf16(gpu, count);
    check(norm_in && norm_weight && norm_out, "rms_norm tensor alloc");
    if (norm_in && norm_weight && norm_out) {
        check(h3_gpu_rms_norm_bf16(gpu, norm_out, norm_in, norm_weight, rows,
                                   width, epsilon),
              "rms_norm");
        check(h3_gpu_submit(gpu), "submit rms_norm");
        check(h3_gpu_tensor_read_bf16(norm_out, norm_out_bf16, count),
              "read rms_norm");
        for (size_t i = 0; i < count; i++) {
            float got = bf16_to_f32(norm_out_bf16[i]);
            if (fabsf(got - norm_ref[i]) >= 2e-2f) {
                fprintf(stderr,
                        "FAIL: rms_norm mismatch at %zu got=%f expected=%f\n",
                        i, got, norm_ref[i]);
                failures++;
                break;
            }
        }
    }

    float layer_norm_bias_host[width];
    uint16_t layer_norm_bias_bf16[width];
    float layer_norm_ref_out[count];
    for (uint32_t column = 0; column < width; column++) {
        layer_norm_bias_host[column] = -0.01f + 0.005f * (float)column;
        layer_norm_bias_bf16[column] = f32_to_bf16(layer_norm_bias_host[column]);
    }
    layer_norm_ref(norm_host, weight_host, layer_norm_bias_host,
                   layer_norm_ref_out, rows, width, epsilon);
    h3_gpu_tensor *layer_norm_out = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *layer_norm_bias =
        h3_gpu_tensor_from_bf16(gpu, layer_norm_bias_bf16, width);
    check(layer_norm_out && layer_norm_bias, "layer_norm tensor alloc");
    if (layer_norm_out && layer_norm_bias) {
        check(h3_gpu_layer_norm_bf16(gpu, layer_norm_out, norm_in, norm_weight,
                                     layer_norm_bias, rows, width, epsilon),
              "layer_norm");
        check(h3_gpu_submit(gpu), "submit layer_norm");
        uint16_t layer_norm_out_bf16[count];
        check(h3_gpu_tensor_read_bf16(layer_norm_out, layer_norm_out_bf16, count),
              "read layer_norm");
        for (size_t i = 0; i < count; i++) {
            float got = bf16_to_f32(layer_norm_out_bf16[i]);
            if (fabsf(got - layer_norm_ref_out[i]) >= 2e-2f) {
                fprintf(stderr,
                        "FAIL: layer_norm mismatch at %zu got=%f expected=%f\n",
                        i, got, layer_norm_ref_out[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t linear_rows = 8;
    const uint32_t linear_input_dim = 64;
    const uint32_t linear_output_dim = 32;
    const size_t linear_input_count = (size_t)linear_rows * linear_input_dim;
    const size_t linear_weight_count = (size_t)linear_output_dim * linear_input_dim;
    const size_t linear_output_count = (size_t)linear_rows * linear_output_dim;
    float linear_input_host[linear_input_count];
    float linear_weight_host[linear_weight_count];
    float linear_bias_host[linear_output_dim];
    uint16_t linear_input_bf16[linear_input_count];
    uint16_t linear_weight_bf16[linear_weight_count];
    uint16_t linear_bias_bf16[linear_output_dim];
    uint16_t linear_out_bf16[linear_output_count];
    float linear_ref_out[linear_output_count];
    float linear_ref_bias_out[linear_output_count];

    for (size_t i = 0; i < linear_input_count; i++) {
        linear_input_host[i] = sinf((float)i * 0.11f);
        linear_input_bf16[i] = f32_to_bf16(linear_input_host[i]);
    }
    for (size_t i = 0; i < linear_weight_count; i++) {
        linear_weight_host[i] = cosf((float)i * 0.05f) * 0.25f;
        linear_weight_bf16[i] = f32_to_bf16(linear_weight_host[i]);
    }
    for (uint32_t i = 0; i < linear_output_dim; i++) {
        linear_bias_host[i] = 0.01f * (float)i;
        linear_bias_bf16[i] = f32_to_bf16(linear_bias_host[i]);
    }
    linear_ref(linear_input_host, linear_weight_host, NULL, linear_ref_out,
               linear_rows, linear_input_dim, linear_output_dim);
    linear_ref(linear_input_host, linear_weight_host, linear_bias_host,
               linear_ref_bias_out, linear_rows, linear_input_dim,
               linear_output_dim);

    h3_gpu_tensor *linear_in =
        h3_gpu_tensor_from_bf16(gpu, linear_input_bf16, linear_input_count);
    h3_gpu_tensor *linear_weight =
        h3_gpu_tensor_from_bf16(gpu, linear_weight_bf16, linear_weight_count);
    h3_gpu_tensor *linear_bias =
        h3_gpu_tensor_from_bf16(gpu, linear_bias_bf16, linear_output_dim);
    h3_gpu_tensor *linear_out =
        h3_gpu_tensor_new_bf16(gpu, linear_output_count);
    h3_gpu_tensor *linear_out_bias =
        h3_gpu_tensor_new_bf16(gpu, linear_output_count);
    check(linear_in && linear_weight && linear_bias && linear_out &&
              linear_out_bias,
          "linear tensor alloc");
    if (linear_in && linear_weight && linear_bias && linear_out &&
        linear_out_bias) {
        check(h3_gpu_linear_bf16(gpu, linear_out, linear_in, linear_weight,
                                 NULL, linear_rows, linear_input_dim,
                                 linear_output_dim),
              "linear");
        check(h3_gpu_submit(gpu), "submit linear");
        check(h3_gpu_tensor_read_bf16(linear_out, linear_out_bf16,
                                      linear_output_count),
              "read linear");
        for (size_t i = 0; i < linear_output_count; i++) {
            float got = bf16_to_f32(linear_out_bf16[i]);
            if (fabsf(got - linear_ref_out[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: linear mismatch at %zu got=%f expected=%f\n", i,
                        got, linear_ref_out[i]);
                failures++;
                break;
            }
        }

        check(h3_gpu_linear_bf16(gpu, linear_out_bias, linear_in, linear_weight,
                                 linear_bias, linear_rows, linear_input_dim,
                                 linear_output_dim),
              "linear with bias");
        check(h3_gpu_submit(gpu), "submit linear bias");
        check(h3_gpu_tensor_read_bf16(linear_out_bias, linear_out_bf16,
                                      linear_output_count),
              "read linear bias");
        for (size_t i = 0; i < linear_output_count; i++) {
            float got = bf16_to_f32(linear_out_bf16[i]);
            if (fabsf(got - linear_ref_bias_out[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: linear bias mismatch at %zu got=%f expected=%f\n",
                        i, got, linear_ref_bias_out[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t adaln_rows = 4;
    const uint32_t adaln_width = 64;
    const uint32_t adaln_slots = 6;
    const uint32_t adaln_shift = 0;
    const uint32_t adaln_scale = 1;
    const uint32_t adaln_gate = 2;
    const size_t adaln_count = (size_t)adaln_rows * adaln_width;
    const size_t adaln_mod_count = (size_t)adaln_slots * adaln_width;
    float adaln_input_host[adaln_count];
    float adaln_residual_host[adaln_count];
    float adaln_branch_host[adaln_count];
    float adaln_norm_host[adaln_width];
    float adaln_mod_host[adaln_mod_count];
    uint16_t adaln_input_bf16[adaln_count];
    uint16_t adaln_residual_bf16[adaln_count];
    uint16_t adaln_branch_bf16[adaln_count];
    uint16_t adaln_norm_bf16[adaln_width];
    uint16_t adaln_mod_bf16[adaln_mod_count];
    uint16_t adaln_out_bf16[adaln_count];
    uint16_t gate_out_bf16[adaln_count];
    uint32_t row_map_host[adaln_rows];
    float adaln_ref_out[adaln_count];
    float gate_ref_out[adaln_count];

    for (uint32_t row = 0; row < adaln_rows; row++) row_map_host[row] = 0;
    for (uint32_t column = 0; column < adaln_width; column++) {
        adaln_norm_host[column] = 0.75f + 0.01f * (float)column;
        adaln_norm_bf16[column] = f32_to_bf16(adaln_norm_host[column]);
    }
    for (size_t slot = 0; slot < adaln_slots; slot++) {
        for (uint32_t column = 0; column < adaln_width; column++) {
            size_t index = slot * adaln_width + column;
            adaln_mod_host[index] = sinf((float)index * 0.03f) * 0.1f;
            adaln_mod_bf16[index] = f32_to_bf16(adaln_mod_host[index]);
        }
    }
    for (size_t i = 0; i < adaln_count; i++) {
        adaln_input_host[i] = cosf((float)i * 0.09f);
        adaln_residual_host[i] = adaln_input_host[i];
        adaln_branch_host[i] = sinf((float)i * 0.07f) * 0.5f;
        adaln_input_bf16[i] = f32_to_bf16(adaln_input_host[i]);
        adaln_residual_bf16[i] = f32_to_bf16(adaln_residual_host[i]);
        adaln_branch_bf16[i] = f32_to_bf16(adaln_branch_host[i]);
    }
    adaln_ref(adaln_input_host, adaln_norm_host, adaln_mod_host, row_map_host,
              adaln_ref_out, adaln_rows, adaln_width, adaln_slots, adaln_shift,
              adaln_scale, 1e-5f);
    gate_ref(adaln_residual_host, adaln_branch_host, adaln_mod_host,
             row_map_host, gate_ref_out, adaln_rows, adaln_width, adaln_slots,
             adaln_gate);

    h3_gpu_tensor *adaln_in =
        h3_gpu_tensor_from_bf16(gpu, adaln_input_bf16, adaln_count);
    h3_gpu_tensor *adaln_norm =
        h3_gpu_tensor_from_bf16(gpu, adaln_norm_bf16, adaln_width);
    h3_gpu_tensor *adaln_mod =
        h3_gpu_tensor_from_bf16(gpu, adaln_mod_bf16, adaln_mod_count);
    h3_gpu_tensor *adaln_row_map =
        h3_gpu_tensor_from_u32(gpu, row_map_host, adaln_rows);
    h3_gpu_tensor *adaln_out = h3_gpu_tensor_new_bf16(gpu, adaln_count);
    h3_gpu_tensor *gate_residual =
        h3_gpu_tensor_from_bf16(gpu, adaln_residual_bf16, adaln_count);
    h3_gpu_tensor *gate_branch =
        h3_gpu_tensor_from_bf16(gpu, adaln_branch_bf16, adaln_count);
    h3_gpu_tensor *gate_out = h3_gpu_tensor_new_bf16(gpu, adaln_count);
    check(adaln_in && adaln_norm && adaln_mod && adaln_row_map && adaln_out &&
              gate_residual && gate_branch && gate_out,
          "adaln/gate tensor alloc");
    if (adaln_in && adaln_norm && adaln_mod && adaln_row_map && adaln_out &&
        gate_residual && gate_branch && gate_out) {
        check(h3_gpu_adaln_bf16(gpu, adaln_out, adaln_in, adaln_norm, adaln_mod,
                                adaln_row_map, adaln_rows, adaln_width,
                                adaln_slots, adaln_shift, adaln_scale, 1e-5f),
              "adaln");
        check(h3_gpu_submit(gpu), "submit adaln");
        check(h3_gpu_tensor_read_bf16(adaln_out, adaln_out_bf16, adaln_count),
              "read adaln");
        for (size_t i = 0; i < adaln_count; i++) {
            float got = bf16_to_f32(adaln_out_bf16[i]);
            if (fabsf(got - adaln_ref_out[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: adaln mismatch at %zu got=%f expected=%f\n", i,
                        got, adaln_ref_out[i]);
                failures++;
                break;
            }
        }

        check(h3_gpu_gate_bf16(gpu, gate_out, gate_residual, gate_branch,
                               adaln_mod, adaln_row_map, adaln_rows,
                               adaln_width, adaln_slots, adaln_gate),
              "gate");
        check(h3_gpu_submit(gpu), "submit gate");
        check(h3_gpu_tensor_read_bf16(gate_out, gate_out_bf16, adaln_count),
              "read gate");
        for (size_t i = 0; i < adaln_count; i++) {
            float got = bf16_to_f32(gate_out_bf16[i]);
            if (fabsf(got - gate_ref_out[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: gate mismatch at %zu got=%f expected=%f\n", i,
                        got, gate_ref_out[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t swiglu_rows = 3;
    const uint32_t swiglu_width = 32;
    const size_t swiglu_fused_count = (size_t)swiglu_rows * swiglu_width * 2u;
    const size_t swiglu_out_count = (size_t)swiglu_rows * swiglu_width;
    float swiglu_fused_host[swiglu_fused_count];
    uint16_t swiglu_fused_bf16[swiglu_fused_count];
    uint16_t swiglu_out_bf16[swiglu_out_count];
    float swiglu_ref_out[swiglu_out_count];
    for (size_t i = 0; i < swiglu_fused_count; i++) {
        swiglu_fused_host[i] = sinf((float)i * 0.13f);
        swiglu_fused_bf16[i] = f32_to_bf16(swiglu_fused_host[i]);
    }
    swiglu_ref(swiglu_fused_host, swiglu_ref_out, swiglu_rows, swiglu_width);

    h3_gpu_tensor *swiglu_fused =
        h3_gpu_tensor_from_bf16(gpu, swiglu_fused_bf16, swiglu_fused_count);
    h3_gpu_tensor *swiglu_out =
        h3_gpu_tensor_new_bf16(gpu, swiglu_out_count);
    check(swiglu_fused && swiglu_out, "swiglu tensor alloc");
    if (swiglu_fused && swiglu_out) {
        check(h3_gpu_swiglu_bf16(gpu, swiglu_out, swiglu_fused, swiglu_rows,
                                 swiglu_width),
              "swiglu");
        check(h3_gpu_submit(gpu), "submit swiglu");
        check(h3_gpu_tensor_read_bf16(swiglu_out, swiglu_out_bf16,
                                      swiglu_out_count),
              "read swiglu");
        for (size_t i = 0; i < swiglu_out_count; i++) {
            float got = bf16_to_f32(swiglu_out_bf16[i]);
            if (fabsf(got - swiglu_ref_out[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: swiglu mismatch at %zu got=%f expected=%f\n", i,
                        got, swiglu_ref_out[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t mlp_rows = 2;
    const uint32_t mlp_input_dim = 16;
    const uint32_t mlp_hidden = 8;
    const uint32_t mlp_out = 12;
    const size_t mlp_input_count = (size_t)mlp_rows * mlp_input_dim;
    const size_t mlp_fc1_count = (size_t)mlp_hidden * 2u * mlp_input_dim;
    const size_t mlp_fc2_count = (size_t)mlp_out * mlp_hidden;
    const size_t mlp_output_count = (size_t)mlp_rows * mlp_out;
    float mlp_input_host[mlp_input_count];
    float mlp_fc1_host[mlp_fc1_count];
    float mlp_fc2_host[mlp_fc2_count];
    uint16_t mlp_input_bf16[mlp_input_count];
    uint16_t mlp_fc1_bf16[mlp_fc1_count];
    uint16_t mlp_fc2_bf16[mlp_fc2_count];
    uint16_t mlp_out_bf16[mlp_output_count];
    float mlp_ref_out[mlp_output_count];
    float mlp_fc1_fused[(size_t)mlp_rows * mlp_hidden * 2u];
    float mlp_swiglu[(size_t)mlp_rows * mlp_hidden];

    for (size_t i = 0; i < mlp_input_count; i++) {
        mlp_input_host[i] = cosf((float)i * 0.2f);
        mlp_input_bf16[i] = f32_to_bf16(mlp_input_host[i]);
    }
    for (size_t i = 0; i < mlp_fc1_count; i++) {
        mlp_fc1_host[i] = sinf((float)i * 0.11f) * 0.2f;
        mlp_fc1_bf16[i] = f32_to_bf16(mlp_fc1_host[i]);
    }
    for (size_t i = 0; i < mlp_fc2_count; i++) {
        mlp_fc2_host[i] = cosf((float)i * 0.07f) * 0.3f;
        mlp_fc2_bf16[i] = f32_to_bf16(mlp_fc2_host[i]);
    }
    linear_ref(mlp_input_host, mlp_fc1_host, NULL, mlp_fc1_fused, mlp_rows,
               mlp_input_dim, mlp_hidden * 2u);
    swiglu_ref(mlp_fc1_fused, mlp_swiglu, mlp_rows, mlp_hidden);
    linear_ref(mlp_swiglu, mlp_fc2_host, NULL, mlp_ref_out, mlp_rows,
               mlp_hidden, mlp_out);

    h3_gpu_tensor *mlp_in =
        h3_gpu_tensor_from_bf16(gpu, mlp_input_bf16, mlp_input_count);
    h3_gpu_tensor *mlp_fc1 =
        h3_gpu_tensor_from_bf16(gpu, mlp_fc1_bf16, mlp_fc1_count);
    h3_gpu_tensor *mlp_fc2 =
        h3_gpu_tensor_from_bf16(gpu, mlp_fc2_bf16, mlp_fc2_count);
    h3_gpu_tensor *mlp_output =
        h3_gpu_tensor_new_bf16(gpu, mlp_output_count);
    check(mlp_in && mlp_fc1 && mlp_fc2 && mlp_output, "mlp tensor alloc");
    if (mlp_in && mlp_fc1 && mlp_fc2 && mlp_output) {
        check(h3_gpu_mlp_bf16(gpu, mlp_output, mlp_in, mlp_fc1, mlp_fc2,
                              mlp_rows, mlp_input_dim, mlp_hidden, mlp_out),
              "mlp");
        check(h3_gpu_submit(gpu), "submit mlp");
        check(h3_gpu_tensor_read_bf16(mlp_output, mlp_out_bf16,
                                      mlp_output_count),
              "read mlp");
        for (size_t i = 0; i < mlp_output_count; i++) {
            float got = bf16_to_f32(mlp_out_bf16[i]);
            if (fabsf(got - mlp_ref_out[i]) >= 8e-2f) {
                fprintf(stderr,
                        "FAIL: mlp mismatch at %zu got=%f expected=%f\n", i,
                        got, mlp_ref_out[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t gelu_count = 256;
    float gelu_host[256];
    uint16_t gelu_in_bf16[256];
    uint16_t gelu_out_bf16[256];
    for (uint32_t i = 0; i < gelu_count; i++) {
        gelu_host[i] = -4.0f + (float)i * (8.0f / (float)(gelu_count - 1));
        gelu_in_bf16[i] = f32_to_bf16(gelu_host[i]);
    }
    h3_gpu_tensor *gelu_in =
        h3_gpu_tensor_from_bf16(gpu, gelu_in_bf16, gelu_count);
    h3_gpu_tensor *gelu_out = h3_gpu_tensor_new_bf16(gpu, gelu_count);
    check(gelu_in && gelu_out, "gelu tensor alloc");
    if (gelu_in && gelu_out) {
        check(h3_gpu_gelu_bf16(gpu, gelu_out, gelu_in, gelu_count, 1), "gelu");
        check(h3_gpu_submit(gpu), "submit gelu");
        check(h3_gpu_tensor_read_bf16(gelu_out, gelu_out_bf16, gelu_count),
              "read gelu");
        for (uint32_t i = 0; i < gelu_count; i++) {
            float got = bf16_to_f32(gelu_out_bf16[i]);
            float expected = gelu_ref(gelu_host[i], 1);
            if (fabsf(got - expected) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: gelu mismatch at %u got=%f expected=%f\n", i,
                        got, expected);
                failures++;
                break;
            }
        }
    }

    const uint32_t embed_tokens = 5;
    const uint32_t embed_vocab = 8;
    const uint32_t embed_width = 16;
    const size_t embed_weight_count = (size_t)embed_vocab * embed_width;
    const size_t embed_output_count = (size_t)embed_tokens * embed_width;
    uint32_t token_ids_host[] = {0, 3, 7, 99, 2};
    float embed_weight_host[embed_weight_count];
    uint16_t embed_weight_bf16[embed_weight_count];
    uint16_t embed_out_bf16[embed_output_count];
    for (size_t i = 0; i < embed_weight_count; i++) {
        embed_weight_host[i] = (float)(i + 1) * 0.01f;
        embed_weight_bf16[i] = f32_to_bf16(embed_weight_host[i]);
    }
    h3_gpu_tensor *embed_weight =
        h3_gpu_tensor_from_bf16(gpu, embed_weight_bf16, embed_weight_count);
    h3_gpu_tensor *embed_ids =
        h3_gpu_tensor_from_u32(gpu, token_ids_host, embed_tokens);
    h3_gpu_tensor *embed_out =
        h3_gpu_tensor_new_bf16(gpu, embed_output_count);
    check(embed_weight && embed_ids && embed_out, "embedding tensor alloc");
    if (embed_weight && embed_ids && embed_out) {
        check(h3_gpu_embedding_bf16(gpu, embed_out, embed_weight, embed_ids,
                                    embed_tokens, embed_vocab, embed_width),
              "embedding");
        check(h3_gpu_submit(gpu), "submit embedding");
        check(h3_gpu_tensor_read_bf16(embed_out, embed_out_bf16,
                                      embed_output_count),
              "read embedding");
        for (uint32_t token = 0; token < embed_tokens; token++) {
            uint32_t identifier = token_ids_host[token];
            for (uint32_t column = 0; column < embed_width; column++) {
                size_t index = (size_t)token * embed_width + column;
                float expected = identifier < embed_vocab
                                     ? embed_weight_host[(size_t)identifier *
                                                               embed_width +
                                                           column]
                                     : 0.0f;
                float got = bf16_to_f32(embed_out_bf16[index]);
                if (fabsf(got - expected) >= 1e-2f) {
                    fprintf(stderr,
                            "FAIL: embedding mismatch token=%u col=%u got=%f "
                            "expected=%f\n",
                            token, column, got, expected);
                    failures++;
                    token = embed_tokens;
                    break;
                }
            }
        }
    }

    const uint32_t f32_silu_count = 256;
    float f32_silu_host[f32_silu_count];
    float f32_silu_ref[f32_silu_count];
    float f32_silu_out[f32_silu_count];
    for (uint32_t i = 0; i < f32_silu_count; i++) {
        f32_silu_host[i] = -4.0f + (float)i * (8.0f / (float)(f32_silu_count - 1));
        f32_silu_ref[i] = silu_ref(f32_silu_host[i]);
    }
    h3_gpu_tensor *f32_silu_in =
        h3_gpu_tensor_from_f32(gpu, f32_silu_host, f32_silu_count);
    h3_gpu_tensor *f32_silu_out_t =
        h3_gpu_tensor_new_f32(gpu, f32_silu_count);
    check(f32_silu_in && f32_silu_out_t, "silu_f32 tensor alloc");
    if (f32_silu_in && f32_silu_out_t) {
        check(h3_gpu_silu_f32(gpu, f32_silu_out_t, f32_silu_in, f32_silu_count),
              "silu_f32");
        check(h3_gpu_submit(gpu), "submit silu_f32");
        check(h3_gpu_tensor_read_f32(f32_silu_out_t, f32_silu_out,
                                       f32_silu_count),
              "read silu_f32");
        for (uint32_t i = 0; i < f32_silu_count; i++) {
            if (fabsf(f32_silu_out[i] - f32_silu_ref[i]) >= 1e-5f) {
                fprintf(stderr,
                        "FAIL: silu_f32 mismatch at %u got=%f expected=%f\n",
                        i, f32_silu_out[i], f32_silu_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t f32_linear_rows = 6;
    const uint32_t f32_linear_in = 32;
    const uint32_t f32_linear_output_dim = 96;
    const size_t f32_linear_input_count =
        (size_t)f32_linear_rows * f32_linear_in;
    const size_t f32_linear_weight_count =
        (size_t)f32_linear_output_dim * f32_linear_in;
    const size_t f32_linear_output_count =
        (size_t)f32_linear_rows * f32_linear_output_dim;
    float f32_linear_input_host[f32_linear_input_count];
    float f32_linear_weight_host[f32_linear_weight_count];
    float f32_linear_bias_host[f32_linear_output_dim];
    float f32_linear_ref_out[f32_linear_output_count];
    float f32_linear_ref_bias_out[f32_linear_output_count];
    float f32_linear_out_host[f32_linear_output_count];
    for (size_t i = 0; i < f32_linear_input_count; i++)
        f32_linear_input_host[i] = sinf((float)i * 0.13f);
    for (size_t i = 0; i < f32_linear_weight_count; i++)
        f32_linear_weight_host[i] = cosf((float)i * 0.07f) * 0.2f;
    for (uint32_t i = 0; i < f32_linear_output_dim; i++)
        f32_linear_bias_host[i] = 0.01f * (float)i;
    linear_ref(f32_linear_input_host, f32_linear_weight_host, NULL,
               f32_linear_ref_out, f32_linear_rows, f32_linear_in,
               f32_linear_output_dim);
    linear_ref(f32_linear_input_host, f32_linear_weight_host,
               f32_linear_bias_host, f32_linear_ref_bias_out, f32_linear_rows,
               f32_linear_in, f32_linear_output_dim);

    h3_gpu_tensor *f32_linear_in_t = h3_gpu_tensor_from_f32(
        gpu, f32_linear_input_host, f32_linear_input_count);
    h3_gpu_tensor *f32_linear_weight_t = h3_gpu_tensor_from_f32(
        gpu, f32_linear_weight_host, f32_linear_weight_count);
    h3_gpu_tensor *f32_linear_bias_t = h3_gpu_tensor_from_f32(
        gpu, f32_linear_bias_host, f32_linear_output_dim);
    h3_gpu_tensor *f32_linear_out_t =
        h3_gpu_tensor_new_f32(gpu, f32_linear_output_count);
    h3_gpu_tensor *f32_linear_out_bias_t =
        h3_gpu_tensor_new_f32(gpu, f32_linear_output_count);
    check(f32_linear_in_t && f32_linear_weight_t && f32_linear_bias_t &&
              f32_linear_out_t && f32_linear_out_bias_t,
          "linear_f32 tensor alloc");
    if (f32_linear_in_t && f32_linear_weight_t && f32_linear_bias_t &&
        f32_linear_out_t && f32_linear_out_bias_t) {
        check(h3_gpu_linear_f32(gpu, f32_linear_out_t, f32_linear_in_t,
                                f32_linear_weight_t, NULL, f32_linear_rows,
                                f32_linear_in, f32_linear_output_dim),
              "linear_f32");
        check(h3_gpu_submit(gpu), "submit linear_f32");
        check(h3_gpu_tensor_read_f32(f32_linear_out_t, f32_linear_out_host,
                                       f32_linear_output_count),
              "read linear_f32");
        for (size_t i = 0; i < f32_linear_output_count; i++) {
            if (fabsf(f32_linear_out_host[i] - f32_linear_ref_out[i]) >= 1e-4f) {
                fprintf(stderr,
                        "FAIL: linear_f32 mismatch at %zu got=%f expected=%f\n",
                        i, f32_linear_out_host[i], f32_linear_ref_out[i]);
                failures++;
                break;
            }
        }

        check(h3_gpu_linear_f32(gpu, f32_linear_out_bias_t, f32_linear_in_t,
                                f32_linear_weight_t, f32_linear_bias_t,
                                f32_linear_rows, f32_linear_in,
                                f32_linear_output_dim),
              "linear_f32 bias");
        check(h3_gpu_submit(gpu), "submit linear_f32 bias");
        check(h3_gpu_tensor_read_f32(f32_linear_out_bias_t, f32_linear_out_host,
                                     f32_linear_output_count),
              "read linear_f32 bias");
        for (size_t i = 0; i < f32_linear_output_count; i++) {
            if (fabsf(f32_linear_out_host[i] - f32_linear_ref_bias_out[i]) >=
                1e-4f) {
                fprintf(stderr,
                        "FAIL: linear_f32 bias mismatch at %zu got=%f "
                        "expected=%f\n",
                        i, f32_linear_out_host[i],
                        f32_linear_ref_bias_out[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t f32_elem_count = 64;
    float f32_left_host[f32_elem_count];
    float f32_right_host[f32_elem_count];
    float f32_scale_host[8];
    float f32_clip_host[f32_elem_count];
    float f32_add_scaled_ref[f32_elem_count];
    float f32_scale_add_ref[f32_elem_count];
    float f32_clip_ref[f32_elem_count];
    for (uint32_t i = 0; i < f32_elem_count; i++) {
        f32_left_host[i] = sinf((float)i * 0.13f);
        f32_right_host[i] = cosf((float)i * 0.09f);
        f32_clip_host[i] = sinf((float)i * 0.41f) * 3.0f;
        f32_add_scaled_ref[i] = f32_left_host[i] * 0.25f + f32_right_host[i] * 0.75f;
        f32_clip_ref[i] = f32_clip_host[i];
        if (f32_clip_ref[i] < -1.0f) f32_clip_ref[i] = -1.0f;
        if (f32_clip_ref[i] > 1.0f) f32_clip_ref[i] = 1.0f;
    }
    for (uint32_t i = 0; i < 8u; i++)
        f32_scale_host[i] = 0.5f + 0.05f * (float)i;
    const uint32_t scale_add_rows = 8;
    const uint32_t scale_add_width = 8;
    for (uint32_t row = 0; row < scale_add_rows; row++) {
        for (uint32_t column = 0; column < scale_add_width; column++) {
            size_t index = (size_t)row * scale_add_width + column;
            f32_scale_add_ref[index] =
                f32_left_host[index] + f32_right_host[index] * f32_scale_host[column];
        }
    }
    h3_gpu_tensor *f32_left_t =
        h3_gpu_tensor_from_f32(gpu, f32_left_host, f32_elem_count);
    h3_gpu_tensor *f32_right_t =
        h3_gpu_tensor_from_f32(gpu, f32_right_host, f32_elem_count);
    h3_gpu_tensor *f32_out_t = h3_gpu_tensor_new_f32(gpu, f32_elem_count);
    check(f32_left_t && f32_right_t && f32_out_t, "f32 elementwise alloc");
    if (f32_left_t && f32_right_t && f32_out_t) {
        check(h3_gpu_add_scaled_f32(gpu, f32_out_t, f32_left_t, f32_right_t,
                                    0.25f, 0.75f, f32_elem_count),
              "add_scaled_f32");
        check(h3_gpu_submit(gpu), "submit add_scaled_f32");
        float f32_add_scaled_out[f32_elem_count];
        check(h3_gpu_tensor_read_f32(f32_out_t, f32_add_scaled_out,
                                     f32_elem_count),
              "read add_scaled_f32");
        for (uint32_t i = 0; i < f32_elem_count; i++) {
            if (fabsf(f32_add_scaled_out[i] - f32_add_scaled_ref[i]) >= 1e-5f) {
                fprintf(stderr,
                        "FAIL: add_scaled_f32 mismatch at %u got=%f expected=%f\n",
                        i, f32_add_scaled_out[i], f32_add_scaled_ref[i]);
                failures++;
                break;
            }
        }
        h3_gpu_tensor *f32_clip_in_t =
            h3_gpu_tensor_from_f32(gpu, f32_clip_host, f32_elem_count);
        check(f32_clip_in_t, "clip_f32 input alloc");
        if (f32_clip_in_t) {
            check(h3_gpu_clip_f32(gpu, f32_out_t, f32_clip_in_t, f32_elem_count,
                                  -1.0f, 1.0f),
                  "clip_f32");
            check(h3_gpu_submit(gpu), "submit clip_f32");
            float f32_clip_out[f32_elem_count];
            check(h3_gpu_tensor_read_f32(f32_out_t, f32_clip_out, f32_elem_count),
                  "read clip_f32");
            for (uint32_t i = 0; i < f32_elem_count; i++) {
                if (fabsf(f32_clip_out[i] - f32_clip_ref[i]) >= 1e-6f) {
                    fprintf(stderr,
                            "FAIL: clip_f32 mismatch at %u got=%f expected=%f\n",
                            i, f32_clip_out[i], f32_clip_ref[i]);
                    failures++;
                    break;
                }
            }
            h3_gpu_tensor_free(f32_clip_in_t);
        }
        h3_gpu_tensor *f32_residual_t =
            h3_gpu_tensor_from_f32(gpu, f32_left_host, scale_add_rows * scale_add_width);
        h3_gpu_tensor *f32_branch_t =
            h3_gpu_tensor_from_f32(gpu, f32_right_host, scale_add_rows * scale_add_width);
        h3_gpu_tensor *f32_scale_t =
            h3_gpu_tensor_from_f32(gpu, f32_scale_host, scale_add_width);
        h3_gpu_tensor *f32_scale_add_out =
            h3_gpu_tensor_new_f32(gpu, (size_t)scale_add_rows * scale_add_width);
        check(f32_residual_t && f32_branch_t && f32_scale_t && f32_scale_add_out,
              "scale_add_f32 alloc");
        if (f32_residual_t && f32_branch_t && f32_scale_t && f32_scale_add_out) {
            check(h3_gpu_scale_add_f32(gpu, f32_scale_add_out, f32_residual_t,
                                       f32_branch_t, f32_scale_t,
                                       scale_add_rows, scale_add_width),
                  "scale_add_f32");
            check(h3_gpu_submit(gpu), "submit scale_add_f32");
            float f32_scale_add_out_host[scale_add_rows * scale_add_width];
            check(h3_gpu_tensor_read_f32(
                      f32_scale_add_out, f32_scale_add_out_host,
                      (size_t)scale_add_rows * scale_add_width),
                  "read scale_add_f32");
            for (size_t i = 0; i < (size_t)scale_add_rows * scale_add_width; i++) {
                if (fabsf(f32_scale_add_out_host[i] - f32_scale_add_ref[i]) >= 1e-5f) {
                    fprintf(stderr,
                            "FAIL: scale_add_f32 mismatch at %zu got=%f expected=%f\n",
                            i, f32_scale_add_out_host[i], f32_scale_add_ref[i]);
                    failures++;
                    break;
                }
            }
            h3_gpu_tensor_free(f32_scale_add_out);
        }
        h3_gpu_tensor_free(f32_residual_t);
        h3_gpu_tensor_free(f32_branch_t);
        h3_gpu_tensor_free(f32_scale_t);
    }

    const uint32_t f32_ln_rows = 4;
    const uint32_t f32_ln_width = 64;
    const size_t f32_ln_count = (size_t)f32_ln_rows * f32_ln_width;
    float f32_ln_in_host[f32_ln_count];
    float f32_ln_weight_host[f32_ln_width];
    float f32_ln_bias_host[f32_ln_width];
    float f32_ln_ref[f32_ln_count];
    for (uint32_t column = 0; column < f32_ln_width; column++) {
        f32_ln_weight_host[column] = 0.5f + 0.01f * (float)column;
        f32_ln_bias_host[column] = -0.01f + 0.005f * (float)column;
    }
    for (uint32_t row = 0; row < f32_ln_rows; row++) {
        for (uint32_t column = 0; column < f32_ln_width; column++) {
            size_t index = (size_t)row * f32_ln_width + column;
            f32_ln_in_host[index] = sinf((float)index * 0.07f);
        }
    }
    layer_norm_ref(f32_ln_in_host, f32_ln_weight_host, f32_ln_bias_host,
                   f32_ln_ref, f32_ln_rows, f32_ln_width, 1e-5f);
    h3_gpu_tensor *f32_ln_in =
        h3_gpu_tensor_from_f32(gpu, f32_ln_in_host, f32_ln_count);
    h3_gpu_tensor *f32_ln_weight =
        h3_gpu_tensor_from_f32(gpu, f32_ln_weight_host, f32_ln_width);
    h3_gpu_tensor *f32_ln_bias =
        h3_gpu_tensor_from_f32(gpu, f32_ln_bias_host, f32_ln_width);
    h3_gpu_tensor *f32_ln_out = h3_gpu_tensor_new_f32(gpu, f32_ln_count);
    check(f32_ln_in && f32_ln_weight && f32_ln_bias && f32_ln_out,
          "layer_norm_f32 alloc");
    if (f32_ln_in && f32_ln_weight && f32_ln_bias && f32_ln_out) {
        check(h3_gpu_layer_norm_f32(gpu, f32_ln_out, f32_ln_in, f32_ln_weight,
                                    f32_ln_bias, f32_ln_rows, f32_ln_width,
                                    1e-5f),
              "layer_norm_f32");
        check(h3_gpu_submit(gpu), "submit layer_norm_f32");
        float f32_ln_out_host[f32_ln_count];
        check(h3_gpu_tensor_read_f32(f32_ln_out, f32_ln_out_host, f32_ln_count),
              "read layer_norm_f32");
        for (size_t i = 0; i < f32_ln_count; i++) {
            if (fabsf(f32_ln_out_host[i] - f32_ln_ref[i]) >= 1e-4f) {
                fprintf(stderr,
                        "FAIL: layer_norm_f32 mismatch at %zu got=%f expected=%f\n",
                        i, f32_ln_out_host[i], f32_ln_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t f32_swiglu_rows = 4;
    const uint32_t f32_swiglu_width = 32;
    const size_t f32_swiglu_fused_count =
        (size_t)f32_swiglu_rows * f32_swiglu_width * 2u;
    const size_t f32_swiglu_out_count =
        (size_t)f32_swiglu_rows * f32_swiglu_width;
    float f32_swiglu_fused_host[f32_swiglu_fused_count];
    float f32_swiglu_ref[f32_swiglu_out_count];
    for (size_t i = 0; i < f32_swiglu_fused_count; i++)
        f32_swiglu_fused_host[i] = sinf((float)i * 0.11f);
    swiglu_ref(f32_swiglu_fused_host, f32_swiglu_ref, f32_swiglu_rows,
               f32_swiglu_width);
    h3_gpu_tensor *f32_swiglu_fused =
        h3_gpu_tensor_from_f32(gpu, f32_swiglu_fused_host, f32_swiglu_fused_count);
    h3_gpu_tensor *f32_swiglu_out =
        h3_gpu_tensor_new_f32(gpu, f32_swiglu_out_count);
    check(f32_swiglu_fused && f32_swiglu_out, "swiglu_f32 alloc");
    if (f32_swiglu_fused && f32_swiglu_out) {
        check(h3_gpu_swiglu_f32(gpu, f32_swiglu_out, f32_swiglu_fused,
                                f32_swiglu_rows, f32_swiglu_width),
              "swiglu_f32");
        check(h3_gpu_submit(gpu), "submit swiglu_f32");
        float f32_swiglu_out_host[f32_swiglu_out_count];
        check(h3_gpu_tensor_read_f32(f32_swiglu_out, f32_swiglu_out_host,
                                     f32_swiglu_out_count),
              "read swiglu_f32");
        for (size_t i = 0; i < f32_swiglu_out_count; i++) {
            if (fabsf(f32_swiglu_out_host[i] - f32_swiglu_ref[i]) >= 1e-4f) {
                fprintf(stderr,
                        "FAIL: swiglu_f32 mismatch at %zu got=%f expected=%f\n",
                        i, f32_swiglu_out_host[i], f32_swiglu_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t f32_geglu_count = 64;
    float f32_geglu_gate_host[f32_geglu_count];
    float f32_geglu_linear_host[f32_geglu_count];
    float f32_geglu_ref[f32_geglu_count];
    for (uint32_t i = 0; i < f32_geglu_count; i++) {
        f32_geglu_gate_host[i] = sinf((float)i * 0.17f);
        f32_geglu_linear_host[i] = cosf((float)i * 0.13f);
        f32_geglu_ref[i] =
            gelu_ref(f32_geglu_gate_host[i], 1) * f32_geglu_linear_host[i];
    }
    h3_gpu_tensor *f32_geglu_gate =
        h3_gpu_tensor_from_f32(gpu, f32_geglu_gate_host, f32_geglu_count);
    h3_gpu_tensor *f32_geglu_linear =
        h3_gpu_tensor_from_f32(gpu, f32_geglu_linear_host, f32_geglu_count);
    h3_gpu_tensor *f32_geglu_out =
        h3_gpu_tensor_new_f32(gpu, f32_geglu_count);
    check(f32_geglu_gate && f32_geglu_linear && f32_geglu_out, "geglu_f32 alloc");
    if (f32_geglu_gate && f32_geglu_linear && f32_geglu_out) {
        check(h3_gpu_geglu_f32(gpu, f32_geglu_out, f32_geglu_gate,
                               f32_geglu_linear, f32_geglu_count),
              "geglu_f32");
        check(h3_gpu_submit(gpu), "submit geglu_f32");
        float f32_geglu_out_host[f32_geglu_count];
        check(h3_gpu_tensor_read_f32(f32_geglu_out, f32_geglu_out_host,
                                     f32_geglu_count),
              "read geglu_f32");
        for (uint32_t i = 0; i < f32_geglu_count; i++) {
            if (fabsf(f32_geglu_out_host[i] - f32_geglu_ref[i]) >= 1e-4f) {
                fprintf(stderr,
                        "FAIL: geglu_f32 mismatch at %u got=%f expected=%f\n",
                        i, f32_geglu_out_host[i], f32_geglu_ref[i]);
                failures++;
                break;
            }
        }
    }

    {
        const uint32_t rms_rows = 4;
        const uint32_t rms_width = 64;
        const size_t rms_count = (size_t)rms_rows * rms_width;
        float rms_in_host[rms_count];
        float rms_weight_host[rms_width];
        float rms_ref[rms_count];
        for (uint32_t column = 0; column < rms_width; column++)
            rms_weight_host[column] = 0.75f + 0.01f * (float)column;
        for (size_t i = 0; i < rms_count; i++)
            rms_in_host[i] = sinf((float)i * 0.09f);
        rms_norm_ref(rms_in_host, rms_weight_host, rms_ref, rms_rows, rms_width,
                     1e-5f);
        h3_gpu_tensor *rms_in = h3_gpu_tensor_from_f32(gpu, rms_in_host, rms_count);
        h3_gpu_tensor *rms_weight =
            h3_gpu_tensor_from_f32(gpu, rms_weight_host, rms_width);
        h3_gpu_tensor *rms_out = h3_gpu_tensor_new_f32(gpu, rms_count);
        check(rms_in && rms_weight && rms_out, "rms_norm_f32 alloc");
        if (rms_in && rms_weight && rms_out) {
            check(h3_gpu_rms_norm_f32(gpu, rms_out, rms_in, rms_weight, rms_rows,
                                      rms_width, 1e-5f),
                  "rms_norm_f32");
            check(h3_gpu_submit(gpu), "submit rms_norm_f32");
            float rms_got[rms_count];
            check(h3_gpu_tensor_read_f32(rms_out, rms_got, rms_count),
                  "read rms_norm_f32");
            for (size_t i = 0; i < rms_count; i++) {
                if (fabsf(rms_got[i] - rms_ref[i]) >= 1e-4f) {
                    fprintf(stderr,
                            "FAIL: rms_norm_f32 mismatch at %zu got=%f "
                            "expected=%f\n",
                            i, rms_got[i], rms_ref[i]);
                    failures++;
                    break;
                }
            }
        }
        h3_gpu_tensor_free(rms_in);
        h3_gpu_tensor_free(rms_weight);
        h3_gpu_tensor_free(rms_out);
    }

    {
        const uint32_t vq_seq = 3;
        const uint32_t vq_heads = 2;
        const uint32_t vq_dim = 8;
        const uint32_t vq_rope = 4;
        const size_t vq_out = (size_t)vq_seq * vq_heads * vq_dim;
        const size_t vq_qkv = vq_out * 3u;
        const size_t vq_rope_n = (size_t)vq_seq * vq_rope;
        float vq_qkv_host[vq_qkv];
        float vq_cos_host[vq_rope_n];
        float vq_sin_host[vq_rope_n];
        float vq_q_ref[vq_out];
        float vq_k_ref[vq_out];
        float vq_v_ref[vq_out];
        for (size_t i = 0; i < vq_qkv; i++)
            vq_qkv_host[i] = sinf((float)i * 0.07f);
        for (size_t i = 0; i < vq_rope_n; i++) {
            vq_cos_host[i] = cosf((float)i * 0.11f);
            vq_sin_host[i] = sinf((float)i * 0.13f);
        }
        video_qkv_rope_ref(vq_qkv_host, vq_cos_host, vq_sin_host, vq_q_ref,
                           vq_k_ref, vq_v_ref, vq_seq, vq_heads, vq_dim, vq_rope,
                           1e-5f);
        h3_gpu_tensor *vq_qkv_t =
            h3_gpu_tensor_from_f32(gpu, vq_qkv_host, vq_qkv);
        h3_gpu_tensor *vq_cos_t =
            h3_gpu_tensor_from_f32(gpu, vq_cos_host, vq_rope_n);
        h3_gpu_tensor *vq_sin_t =
            h3_gpu_tensor_from_f32(gpu, vq_sin_host, vq_rope_n);
        h3_gpu_tensor *vq_q = h3_gpu_tensor_new_f32(gpu, vq_out);
        h3_gpu_tensor *vq_k = h3_gpu_tensor_new_f32(gpu, vq_out);
        h3_gpu_tensor *vq_v = h3_gpu_tensor_new_f32(gpu, vq_out);
        check(vq_qkv_t && vq_cos_t && vq_sin_t && vq_q && vq_k && vq_v,
              "video_qkv_rope_f32 alloc");
        if (vq_qkv_t && vq_cos_t && vq_sin_t && vq_q && vq_k && vq_v) {
            check(h3_gpu_video_qkv_rope_f32(gpu, vq_q, vq_k, vq_v, vq_qkv_t,
                                            vq_cos_t, vq_sin_t, vq_seq, vq_heads,
                                            vq_dim, vq_rope, 1e-5f),
                  "video_qkv_rope_f32");
            check(h3_gpu_submit(gpu), "submit video_qkv_rope_f32");
            float vq_q_got[vq_out];
            float vq_k_got[vq_out];
            float vq_v_got[vq_out];
            check(h3_gpu_tensor_read_f32(vq_q, vq_q_got, vq_out),
                  "read video q");
            check(h3_gpu_tensor_read_f32(vq_k, vq_k_got, vq_out),
                  "read video k");
            check(h3_gpu_tensor_read_f32(vq_v, vq_v_got, vq_out),
                  "read video v");
            for (size_t i = 0; i < vq_out; i++) {
                if (fabsf(vq_q_got[i] - vq_q_ref[i]) >= 1e-4f ||
                    fabsf(vq_k_got[i] - vq_k_ref[i]) >= 1e-4f ||
                    fabsf(vq_v_got[i] - vq_v_ref[i]) >= 1e-4f) {
                    fprintf(stderr,
                            "FAIL: video_qkv_rope_f32 mismatch at %zu\n", i);
                    failures++;
                    break;
                }
            }
        }
        h3_gpu_tensor_free(vq_qkv_t);
        h3_gpu_tensor_free(vq_cos_t);
        h3_gpu_tensor_free(vq_sin_t);
        h3_gpu_tensor_free(vq_q);
        h3_gpu_tensor_free(vq_k);
        h3_gpu_tensor_free(vq_v);
    }

    {
        const uint32_t sd_seq = 4;
        const uint32_t sd_heads = 2;
        const uint32_t sd_dim = 8;
        const size_t sd_count = (size_t)sd_seq * sd_heads * sd_dim;
        float sd_q_host[sd_count];
        float sd_k_host[sd_count];
        float sd_v_host[sd_count];
        float sd_ref[sd_count];
        for (size_t i = 0; i < sd_count; i++) {
            sd_q_host[i] = sinf((float)i * 0.05f);
            sd_k_host[i] = cosf((float)i * 0.07f);
            sd_v_host[i] = sinf((float)i * 0.11f + 0.3f);
        }
        float scale = 1.0f / sqrtf((float)sd_dim);
        sdpa_ref(sd_q_host, sd_k_host, sd_v_host, sd_ref, sd_seq, sd_heads,
                 sd_dim, scale);
        h3_gpu_tensor *sd_q = h3_gpu_tensor_from_f32(gpu, sd_q_host, sd_count);
        h3_gpu_tensor *sd_k = h3_gpu_tensor_from_f32(gpu, sd_k_host, sd_count);
        h3_gpu_tensor *sd_v = h3_gpu_tensor_from_f32(gpu, sd_v_host, sd_count);
        h3_gpu_tensor *sd_out = h3_gpu_tensor_new_f32(gpu, sd_count);
        check(sd_q && sd_k && sd_v && sd_out, "sdpa_f32 alloc");
        if (sd_q && sd_k && sd_v && sd_out) {
            check(h3_gpu_sdpa_f32(gpu, sd_out, sd_q, sd_k, sd_v, sd_seq,
                                  sd_heads, sd_dim, scale),
                  "sdpa_f32");
            check(h3_gpu_submit(gpu), "submit sdpa_f32");
            float sd_got[sd_count];
            check(h3_gpu_tensor_read_f32(sd_out, sd_got, sd_count),
                  "read sdpa_f32");
            for (size_t i = 0; i < sd_count; i++) {
                if (fabsf(sd_got[i] - sd_ref[i]) >= 1e-4f) {
                    fprintf(stderr,
                            "FAIL: sdpa_f32 mismatch at %zu got=%f expected=%f\n",
                            i, sd_got[i], sd_ref[i]);
                    failures++;
                    break;
                }
            }
        }
        h3_gpu_tensor_free(sd_q);
        h3_gpu_tensor_free(sd_k);
        h3_gpu_tensor_free(sd_v);
        h3_gpu_tensor_free(sd_out);
    }

    {
        /* weight_norm + conv1d + transpose + snake smoke oracles */
        const uint32_t wn_outer = 4;
        const uint32_t wn_inner = 8;
        float wn_vec[wn_outer * wn_inner];
        float wn_mag[wn_outer];
        float wn_ref[wn_outer * wn_inner];
        for (uint32_t row = 0; row < wn_outer; row++) {
            wn_mag[row] = 0.5f + 0.1f * (float)row;
            float square = 0.0f;
            for (uint32_t col = 0; col < wn_inner; col++) {
                float v = sinf((float)(row * wn_inner + col) * 0.2f);
                wn_vec[row * wn_inner + col] = v;
                square += v * v;
            }
            float scale = wn_mag[row] / sqrtf(square);
            for (uint32_t col = 0; col < wn_inner; col++)
                wn_ref[row * wn_inner + col] =
                    wn_vec[row * wn_inner + col] * scale;
        }
        h3_gpu_tensor *wn_v =
            h3_gpu_tensor_from_f32(gpu, wn_vec, wn_outer * wn_inner);
        h3_gpu_tensor *wn_m = h3_gpu_tensor_from_f32(gpu, wn_mag, wn_outer);
        h3_gpu_tensor *wn_o =
            h3_gpu_tensor_new_f32(gpu, wn_outer * wn_inner);
        check(wn_v && wn_m && wn_o, "weight_norm alloc");
        if (wn_v && wn_m && wn_o) {
            check(h3_gpu_weight_norm_f32(gpu, wn_o, wn_v, wn_m, wn_outer,
                                         wn_inner),
                  "weight_norm_f32");
            check(h3_gpu_submit(gpu), "submit weight_norm");
            float wn_got[wn_outer * wn_inner];
            check(h3_gpu_tensor_read_f32(wn_o, wn_got, wn_outer * wn_inner),
                  "read weight_norm");
            for (uint32_t i = 0; i < wn_outer * wn_inner; i++) {
                if (fabsf(wn_got[i] - wn_ref[i]) >= 1e-4f) {
                    fprintf(stderr, "FAIL: weight_norm_f32 at %u\n", i);
                    failures++;
                    break;
                }
            }
        }
        h3_gpu_tensor_free(wn_v);
        h3_gpu_tensor_free(wn_m);
        h3_gpu_tensor_free(wn_o);

        const uint32_t c_batch = 1, c_len = 5, c_in = 2, c_out = 3, c_k = 3;
        const uint32_t c_pad = 1, c_dil = 1, c_stride = 1;
        uint32_t c_out_len =
            (c_len + 2 * c_pad - (c_dil * (c_k - 1) + 1)) / c_stride + 1;
        float c_in_h[c_batch * c_len * c_in];
        float c_w_h[c_out * c_in * c_k];
        float c_b_h[c_out];
        float c_ref[c_batch * c_out_len * c_out];
        for (uint32_t i = 0; i < c_batch * c_len * c_in; i++)
            c_in_h[i] = sinf((float)i * 0.15f);
        for (uint32_t i = 0; i < c_out * c_in * c_k; i++)
            c_w_h[i] = cosf((float)i * 0.09f);
        for (uint32_t i = 0; i < c_out; i++) c_b_h[i] = 0.01f * (float)i;
        for (uint32_t b = 0; b < c_batch; b++) {
            for (uint32_t t = 0; t < c_out_len; t++) {
                for (uint32_t oc = 0; oc < c_out; oc++) {
                    float acc = c_b_h[oc];
                    for (uint32_t ic = 0; ic < c_in; ic++) {
                        for (uint32_t k = 0; k < c_k; k++) {
                            int tin = (int)t * (int)c_stride - (int)c_pad +
                                      (int)k * (int)c_dil;
                            if (tin < 0 || tin >= (int)c_len) continue;
                            acc += c_in_h[(b * c_len + (uint32_t)tin) * c_in +
                                          ic] *
                                   c_w_h[(oc * c_in + ic) * c_k + k];
                        }
                    }
                    c_ref[(b * c_out_len + t) * c_out + oc] = acc;
                }
            }
        }
        h3_gpu_tensor *c_in_t =
            h3_gpu_tensor_from_f32(gpu, c_in_h, c_batch * c_len * c_in);
        h3_gpu_tensor *c_w_t =
            h3_gpu_tensor_from_f32(gpu, c_w_h, c_out * c_in * c_k);
        h3_gpu_tensor *c_b_t = h3_gpu_tensor_from_f32(gpu, c_b_h, c_out);
        h3_gpu_tensor *c_o_t =
            h3_gpu_tensor_new_f32(gpu, c_batch * c_out_len * c_out);
        check(c_in_t && c_w_t && c_b_t && c_o_t, "conv1d alloc");
        if (c_in_t && c_w_t && c_b_t && c_o_t) {
            check(h3_gpu_conv1d_f32(gpu, c_o_t, c_in_t, c_w_t, c_b_t, c_batch,
                                    c_len, c_in, c_out, c_k, c_pad, c_dil),
                  "conv1d_f32");
            check(h3_gpu_submit(gpu), "submit conv1d");
            float c_got[c_batch * c_out_len * c_out];
            check(h3_gpu_tensor_read_f32(c_o_t, c_got,
                                         c_batch * c_out_len * c_out),
                  "read conv1d");
            for (uint32_t i = 0; i < c_batch * c_out_len * c_out; i++) {
                if (fabsf(c_got[i] - c_ref[i]) >= 1e-4f) {
                    fprintf(stderr, "FAIL: conv1d_f32 at %u got=%f want=%f\n",
                            i, c_got[i], c_ref[i]);
                    failures++;
                    break;
                }
            }
        }
        h3_gpu_tensor_free(c_in_t);
        h3_gpu_tensor_free(c_w_t);
        h3_gpu_tensor_free(c_b_t);
        h3_gpu_tensor_free(c_o_t);

        const uint32_t sn_b = 1, sn_l = 4, sn_c = 3;
        float sn_in[sn_b * sn_l * sn_c];
        float sn_alpha[sn_c];
        float sn_ref[sn_b * sn_l * sn_c];
        for (uint32_t i = 0; i < sn_c; i++) sn_alpha[i] = 0.5f + 0.1f * (float)i;
        for (uint32_t i = 0; i < sn_b * sn_l * sn_c; i++) {
            sn_in[i] = sinf((float)i * 0.2f);
            float a = sn_alpha[i % sn_c];
            float x = sn_in[i];
            float wave = sinf(a * x);
            sn_ref[i] = x + wave * wave / (a + 1e-9f);
        }
        h3_gpu_tensor *sn_i =
            h3_gpu_tensor_from_f32(gpu, sn_in, sn_b * sn_l * sn_c);
        h3_gpu_tensor *sn_a = h3_gpu_tensor_from_f32(gpu, sn_alpha, sn_c);
        h3_gpu_tensor *sn_o =
            h3_gpu_tensor_new_f32(gpu, sn_b * sn_l * sn_c);
        check(sn_i && sn_a && sn_o, "snake1d alloc");
        if (sn_i && sn_a && sn_o) {
            check(h3_gpu_snake1d_f32(gpu, sn_o, sn_i, sn_a, sn_b, sn_l, sn_c),
                  "snake1d_f32");
            check(h3_gpu_submit(gpu), "submit snake1d");
            float sn_got[sn_b * sn_l * sn_c];
            check(h3_gpu_tensor_read_f32(sn_o, sn_got, sn_b * sn_l * sn_c),
                  "read snake1d");
            for (uint32_t i = 0; i < sn_b * sn_l * sn_c; i++) {
                if (fabsf(sn_got[i] - sn_ref[i]) >= 1e-4f) {
                    fprintf(stderr, "FAIL: snake1d_f32 at %u\n", i);
                    failures++;
                    break;
                }
            }
        }
        h3_gpu_tensor_free(sn_i);
        h3_gpu_tensor_free(sn_a);
        h3_gpu_tensor_free(sn_o);
    }

    const uint32_t qkv_sequence = 3;
    const uint32_t qkv_heads = 2;
    const uint32_t qkv_head_dim = 8;
    const uint32_t qkv_rope_half = 4;
    const uint32_t qkv_inner = qkv_heads * qkv_head_dim;
    const size_t qkv_count = (size_t)qkv_sequence * qkv_inner * 3u;
    const size_t qkv_out_count = (size_t)qkv_sequence * qkv_inner;
    const size_t rope_count = (size_t)qkv_sequence * qkv_rope_half;
    float qkv_host[qkv_count];
    float q_norm_host[qkv_head_dim];
    float k_norm_host[qkv_head_dim];
    float rope_cos_host[rope_count];
    float rope_sin_host[rope_count];
    uint16_t qkv_bf16[qkv_count];
    uint16_t q_norm_bf16[qkv_head_dim];
    uint16_t k_norm_bf16[qkv_head_dim];
    uint16_t rope_cos_bf16[rope_count];
    uint16_t rope_sin_bf16[rope_count];
    uint16_t qkv_query_bf16[qkv_out_count];
    uint16_t qkv_key_bf16[qkv_out_count];
    uint16_t qkv_value_bf16[qkv_out_count];
    float qkv_query_ref[qkv_out_count];
    float qkv_key_ref[qkv_out_count];
    float qkv_value_ref[qkv_out_count];

    for (size_t i = 0; i < qkv_count; i++) {
        qkv_host[i] = sinf((float)i * 0.17f) * 0.4f;
        qkv_bf16[i] = f32_to_bf16(qkv_host[i]);
    }
    for (uint32_t i = 0; i < qkv_head_dim; i++) {
        q_norm_host[i] = 0.8f + 0.01f * (float)i;
        k_norm_host[i] = 0.7f + 0.02f * (float)i;
        q_norm_bf16[i] = f32_to_bf16(q_norm_host[i]);
        k_norm_bf16[i] = f32_to_bf16(k_norm_host[i]);
    }
    for (size_t i = 0; i < rope_count; i++) {
        rope_cos_host[i] = cosf((float)i * 0.31f);
        rope_sin_host[i] = sinf((float)i * 0.31f);
        rope_cos_bf16[i] = f32_to_bf16(rope_cos_host[i]);
        rope_sin_bf16[i] = f32_to_bf16(rope_sin_host[i]);
    }
    grouped_qkv_rope_ref(qkv_host, q_norm_host, k_norm_host, rope_cos_host,
                         rope_sin_host, qkv_query_ref, qkv_key_ref,
                         qkv_value_ref, qkv_sequence, qkv_heads, qkv_head_dim,
                         qkv_rope_half, 1e-5f);

    h3_gpu_tensor *qkv_in = h3_gpu_tensor_from_bf16(gpu, qkv_bf16, qkv_count);
    h3_gpu_tensor *qkv_q_norm =
        h3_gpu_tensor_from_bf16(gpu, q_norm_bf16, qkv_head_dim);
    h3_gpu_tensor *qkv_k_norm =
        h3_gpu_tensor_from_bf16(gpu, k_norm_bf16, qkv_head_dim);
    h3_gpu_tensor *qkv_rope_cos =
        h3_gpu_tensor_from_bf16(gpu, rope_cos_bf16, rope_count);
    h3_gpu_tensor *qkv_rope_sin =
        h3_gpu_tensor_from_bf16(gpu, rope_sin_bf16, rope_count);
    h3_gpu_tensor *qkv_query =
        h3_gpu_tensor_new_bf16(gpu, qkv_out_count);
    h3_gpu_tensor *qkv_key = h3_gpu_tensor_new_bf16(gpu, qkv_out_count);
    h3_gpu_tensor *qkv_value = h3_gpu_tensor_new_bf16(gpu, qkv_out_count);
    check(qkv_in && qkv_q_norm && qkv_k_norm && qkv_rope_cos && qkv_rope_sin &&
              qkv_query && qkv_key && qkv_value,
          "qkv_rope tensor alloc");
    if (qkv_in && qkv_q_norm && qkv_k_norm && qkv_rope_cos && qkv_rope_sin &&
        qkv_query && qkv_key && qkv_value) {
        check(h3_gpu_grouped_qkv_rope_bf16(
                  gpu, qkv_query, qkv_key, qkv_value, qkv_in, qkv_q_norm,
                  qkv_k_norm, qkv_rope_cos, qkv_rope_sin, qkv_sequence,
                  qkv_heads, qkv_head_dim, qkv_rope_half, 1e-5f),
              "grouped_qkv_rope");
        check(h3_gpu_submit(gpu), "submit grouped_qkv_rope");
        check(h3_gpu_tensor_read_bf16(qkv_query, qkv_query_bf16, qkv_out_count),
              "read qkv query");
        check(h3_gpu_tensor_read_bf16(qkv_key, qkv_key_bf16, qkv_out_count),
              "read qkv key");
        check(h3_gpu_tensor_read_bf16(qkv_value, qkv_value_bf16, qkv_out_count),
              "read qkv value");
        for (size_t i = 0; i < qkv_out_count; i++) {
            float got_q = bf16_to_f32(qkv_query_bf16[i]);
            float got_k = bf16_to_f32(qkv_key_bf16[i]);
            float got_v = bf16_to_f32(qkv_value_bf16[i]);
            if (fabsf(got_q - qkv_query_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: qkv query mismatch at %zu got=%f expected=%f\n",
                        i, got_q, qkv_query_ref[i]);
                failures++;
                break;
            }
            if (fabsf(got_k - qkv_key_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: qkv key mismatch at %zu got=%f expected=%f\n",
                        i, got_k, qkv_key_ref[i]);
                failures++;
                break;
            }
            if (fabsf(got_v - qkv_value_ref[i]) >= 1e-2f) {
                fprintf(stderr,
                        "FAIL: qkv value mismatch at %zu got=%f expected=%f\n",
                        i, got_v, qkv_value_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t vision_sequence = 3;
    const uint32_t vision_heads = 2;
    const uint32_t vision_head_dim = 8;
    const uint32_t vision_rope_half = vision_head_dim / 2u;
    const size_t vision_inner = (size_t)vision_heads * vision_head_dim;
    const size_t vision_qkv_count = (size_t)vision_sequence * vision_inner * 3u;
    const size_t vision_out_count = (size_t)vision_sequence * vision_inner;
    const size_t vision_rope_count = (size_t)vision_sequence * vision_rope_half;
    float vision_qkv_host[vision_qkv_count];
    float vision_rope_cos_host[vision_rope_count];
    float vision_rope_sin_host[vision_rope_count];
    uint16_t vision_qkv_bf16[vision_qkv_count];
    uint16_t vision_rope_cos_bf16[vision_rope_count];
    uint16_t vision_rope_sin_bf16[vision_rope_count];
    uint16_t vision_query_bf16[vision_out_count];
    uint16_t vision_key_bf16[vision_out_count];
    uint16_t vision_value_bf16[vision_out_count];
    float vision_query_ref[vision_out_count];
    float vision_key_ref[vision_out_count];
    float vision_value_ref[vision_out_count];
    for (size_t i = 0; i < vision_qkv_count; i++) {
        vision_qkv_host[i] = sinf((float)i * 0.29f);
        vision_qkv_bf16[i] = f32_to_bf16(vision_qkv_host[i]);
    }
    for (size_t i = 0; i < vision_rope_count; i++) {
        vision_rope_cos_host[i] = cosf((float)i * 0.37f);
        vision_rope_sin_host[i] = sinf((float)i * 0.37f);
        vision_rope_cos_bf16[i] = f32_to_bf16(vision_rope_cos_host[i]);
        vision_rope_sin_bf16[i] = f32_to_bf16(vision_rope_sin_host[i]);
    }
    vision_qkv_rope_ref(vision_qkv_host, vision_rope_cos_host,
                        vision_rope_sin_host, vision_query_ref, vision_key_ref,
                        vision_value_ref, vision_sequence, vision_heads,
                        vision_head_dim, vision_rope_half);
    h3_gpu_tensor *vision_qkv =
        h3_gpu_tensor_from_bf16(gpu, vision_qkv_bf16, vision_qkv_count);
    h3_gpu_tensor *vision_rope_cos =
        h3_gpu_tensor_from_bf16(gpu, vision_rope_cos_bf16, vision_rope_count);
    h3_gpu_tensor *vision_rope_sin =
        h3_gpu_tensor_from_bf16(gpu, vision_rope_sin_bf16, vision_rope_count);
    h3_gpu_tensor *vision_query =
        h3_gpu_tensor_new_bf16(gpu, vision_out_count);
    h3_gpu_tensor *vision_key =
        h3_gpu_tensor_new_bf16(gpu, vision_out_count);
    h3_gpu_tensor *vision_value =
        h3_gpu_tensor_new_bf16(gpu, vision_out_count);
    check(vision_qkv && vision_rope_cos && vision_rope_sin && vision_query &&
              vision_key && vision_value,
          "vision_qkv_rope tensor alloc");
    if (vision_qkv && vision_rope_cos && vision_rope_sin && vision_query &&
        vision_key && vision_value) {
        check(h3_gpu_vision_qkv_rope_bf16(
                  gpu, vision_query, vision_key, vision_value, vision_qkv,
                  vision_rope_cos, vision_rope_sin, vision_sequence,
                  vision_heads, vision_head_dim, vision_rope_half),
              "vision_qkv_rope");
        check(h3_gpu_submit(gpu), "submit vision_qkv_rope");
        check(h3_gpu_tensor_read_bf16(vision_query, vision_query_bf16,
                                      vision_out_count),
              "read vision query");
        check(h3_gpu_tensor_read_bf16(vision_key, vision_key_bf16,
                                      vision_out_count),
              "read vision key");
        check(h3_gpu_tensor_read_bf16(vision_value, vision_value_bf16,
                                      vision_out_count),
              "read vision value");
        for (size_t i = 0; i < vision_out_count; i++) {
            float got_q = bf16_to_f32(vision_query_bf16[i]);
            float got_k = bf16_to_f32(vision_key_bf16[i]);
            float got_v = bf16_to_f32(vision_value_bf16[i]);
            if (fabsf(got_q - vision_query_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: vision query mismatch at %zu got=%f expected=%f\n",
                        i, got_q, vision_query_ref[i]);
                failures++;
                break;
            }
            if (fabsf(got_k - vision_key_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: vision key mismatch at %zu got=%f expected=%f\n",
                        i, got_k, vision_key_ref[i]);
                failures++;
                break;
            }
            if (fabsf(got_v - vision_value_ref[i]) >= 1e-2f) {
                fprintf(stderr,
                        "FAIL: vision value mismatch at %zu got=%f expected=%f\n",
                        i, got_v, vision_value_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t sdpa_sequence = 4;
    const uint32_t sdpa_heads = 2;
    const uint32_t sdpa_head_dim = 8;
    const size_t sdpa_count =
        (size_t)sdpa_sequence * sdpa_heads * sdpa_head_dim;
    const float sdpa_scale = 1.0f / sqrtf((float)sdpa_head_dim);
    float sdpa_query_host[sdpa_count];
    float sdpa_key_host[sdpa_count];
    float sdpa_value_host[sdpa_count];
    uint16_t sdpa_query_bf16[sdpa_count];
    uint16_t sdpa_key_bf16[sdpa_count];
    uint16_t sdpa_value_bf16[sdpa_count];
    uint16_t sdpa_out_bf16[sdpa_count];
    float sdpa_out_ref[sdpa_count];
    for (size_t i = 0; i < sdpa_count; i++) {
        sdpa_query_host[i] = sinf((float)i * 0.23f);
        sdpa_key_host[i] = cosf((float)i * 0.19f);
        sdpa_value_host[i] = sinf((float)i * 0.11f) * 0.5f;
        sdpa_query_bf16[i] = f32_to_bf16(sdpa_query_host[i]);
        sdpa_key_bf16[i] = f32_to_bf16(sdpa_key_host[i]);
        sdpa_value_bf16[i] = f32_to_bf16(sdpa_value_host[i]);
    }
    sdpa_ref(sdpa_query_host, sdpa_key_host, sdpa_value_host, sdpa_out_ref,
             sdpa_sequence, sdpa_heads, sdpa_head_dim, sdpa_scale);

    h3_gpu_tensor *sdpa_query =
        h3_gpu_tensor_from_bf16(gpu, sdpa_query_bf16, sdpa_count);
    h3_gpu_tensor *sdpa_key =
        h3_gpu_tensor_from_bf16(gpu, sdpa_key_bf16, sdpa_count);
    h3_gpu_tensor *sdpa_value =
        h3_gpu_tensor_from_bf16(gpu, sdpa_value_bf16, sdpa_count);
    h3_gpu_tensor *sdpa_out = h3_gpu_tensor_new_bf16(gpu, sdpa_count);
    check(sdpa_query && sdpa_key && sdpa_value && sdpa_out, "sdpa tensor alloc");
    if (sdpa_query && sdpa_key && sdpa_value && sdpa_out) {
        check(h3_gpu_sdpa_bf16(gpu, sdpa_out, sdpa_query, sdpa_key, sdpa_value,
                               sdpa_sequence, sdpa_heads, sdpa_head_dim,
                               sdpa_scale),
              "sdpa");
        check(h3_gpu_submit(gpu), "submit sdpa");
        check(h3_gpu_tensor_read_bf16(sdpa_out, sdpa_out_bf16, sdpa_count),
              "read sdpa");
        for (size_t i = 0; i < sdpa_count; i++) {
            float got = bf16_to_f32(sdpa_out_bf16[i]);
            if (fabsf(got - sdpa_out_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: sdpa mismatch at %zu got=%f expected=%f\n", i,
                        got, sdpa_out_ref[i]);
                failures++;
                break;
            }
        }
    }

    {
        const uint32_t d128_seq = 5;
        const uint32_t d128_heads = 2;
        const uint32_t d128_dim = 128;
        const size_t d128_count = (size_t)d128_seq * d128_heads * d128_dim;
        const float d128_scale = 1.0f / sqrtf((float)d128_dim);
        float *d128_q = (float *)malloc(d128_count * sizeof(float));
        float *d128_k = (float *)malloc(d128_count * sizeof(float));
        float *d128_v = (float *)malloc(d128_count * sizeof(float));
        float *d128_ref = (float *)malloc(d128_count * sizeof(float));
        uint16_t *d128_q_b = (uint16_t *)malloc(d128_count * sizeof(uint16_t));
        uint16_t *d128_k_b = (uint16_t *)malloc(d128_count * sizeof(uint16_t));
        uint16_t *d128_v_b = (uint16_t *)malloc(d128_count * sizeof(uint16_t));
        uint16_t *d128_out_b = (uint16_t *)malloc(d128_count * sizeof(uint16_t));
        check(d128_q && d128_k && d128_v && d128_ref && d128_q_b && d128_k_b &&
                  d128_v_b && d128_out_b,
              "d128 sdpa host alloc");
        if (d128_q && d128_k && d128_v && d128_ref) {
            for (size_t i = 0; i < d128_count; i++) {
                d128_q[i] = sinf((float)i * 0.013f);
                d128_k[i] = cosf((float)i * 0.017f);
                d128_v[i] = sinf((float)i * 0.011f) * 0.5f;
                d128_q_b[i] = f32_to_bf16(d128_q[i]);
                d128_k_b[i] = f32_to_bf16(d128_k[i]);
                d128_v_b[i] = f32_to_bf16(d128_v[i]);
            }
            sdpa_ref(d128_q, d128_k, d128_v, d128_ref, d128_seq, d128_heads,
                     d128_dim, d128_scale);
            h3_gpu_tensor *tq =
                h3_gpu_tensor_from_bf16(gpu, d128_q_b, d128_count);
            h3_gpu_tensor *tk =
                h3_gpu_tensor_from_bf16(gpu, d128_k_b, d128_count);
            h3_gpu_tensor *tv =
                h3_gpu_tensor_from_bf16(gpu, d128_v_b, d128_count);
            h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, d128_count);
            check(tq && tk && tv && to, "d128 sdpa tensor alloc");
            if (tq && tk && tv && to) {
                check(h3_gpu_sdpa_bf16(gpu, to, tq, tk, tv, d128_seq,
                                       d128_heads, d128_dim, d128_scale),
                      "sdpa d128 q2");
                check(h3_gpu_submit(gpu), "submit sdpa d128");
                check(h3_gpu_tensor_read_bf16(to, d128_out_b, d128_count),
                      "read sdpa d128");
                for (size_t i = 0; i < d128_count; i++) {
                    float got = bf16_to_f32(d128_out_b[i]);
                    if (fabsf(got - d128_ref[i]) >= 8e-2f) {
                        fprintf(stderr,
                                "FAIL: sdpa d128 mismatch at %zu got=%f "
                                "expected=%f\n",
                                i, got, d128_ref[i]);
                        failures++;
                        break;
                    }
                }
            }
            h3_gpu_tensor_free(tq);
            h3_gpu_tensor_free(tk);
            h3_gpu_tensor_free(tv);
            h3_gpu_tensor_free(to);
        }
        free(d128_q);
        free(d128_k);
        free(d128_v);
        free(d128_ref);
        free(d128_q_b);
        free(d128_k_b);
        free(d128_v_b);
        free(d128_out_b);
    }

    /* Tensor-core SDPA. The sequence is deliberately not a multiple of the
     * 64-wide MMA tile so the query and key masks both get exercised, and
     * both output layouts are checked because the DiT uses the head-major
     * one. */
    {
        const uint32_t mma_seq = 200;
        const uint32_t mma_heads = 3;
        const uint32_t mma_dim = 128;
        const size_t mma_count = (size_t)mma_seq * mma_heads * mma_dim;
        const float mma_scale = 1.0f / sqrtf((float)mma_dim);
        float *mq = (float *)malloc(mma_count * sizeof(float));
        float *mk = (float *)malloc(mma_count * sizeof(float));
        float *mv = (float *)malloc(mma_count * sizeof(float));
        float *mref = (float *)malloc(mma_count * sizeof(float));
        uint16_t *mqb = (uint16_t *)malloc(mma_count * sizeof(uint16_t));
        uint16_t *mkb = (uint16_t *)malloc(mma_count * sizeof(uint16_t));
        uint16_t *mvb = (uint16_t *)malloc(mma_count * sizeof(uint16_t));
        uint16_t *mout = (uint16_t *)malloc(mma_count * sizeof(uint16_t));
        check(mq && mk && mv && mref && mqb && mkb && mvb && mout,
              "mma sdpa host alloc");
        if (mq && mk && mv && mref && mqb && mkb && mvb && mout) {
            for (size_t i = 0; i < mma_count; i++) {
                mq[i] = sinf((float)i * 0.017f) * 1.3f;
                mk[i] = cosf((float)i * 0.023f);
                mv[i] = sinf((float)i * 0.009f + 0.7f) * 0.5f;
                mqb[i] = f32_to_bf16(mq[i]);
                mkb[i] = f32_to_bf16(mk[i]);
                mvb[i] = f32_to_bf16(mv[i]);
            }
            sdpa_ref(mq, mk, mv, mref, mma_seq, mma_heads, mma_dim, mma_scale);
            h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, mqb, mma_count);
            h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, mkb, mma_count);
            h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, mvb, mma_count);
            h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, mma_count);
            check(tq && tk && tv && to, "mma sdpa tensor alloc");
            if (tq && tk && tv && to) {
                setenv("H3_SDPA_MMA", "1", 1);
                for (int head_major = 0; head_major < 2; head_major++) {
                    int ok = head_major
                                 ? h3_gpu_sdpa_bf16_head_major_output(
                                       gpu, to, tq, tk, tv, mma_seq, mma_heads,
                                       mma_dim, mma_scale)
                                 : h3_gpu_sdpa_bf16(gpu, to, tq, tk, tv,
                                                    mma_seq, mma_heads,
                                                    mma_dim, mma_scale);
                    check(ok, "mma sdpa dispatch");
                    check(h3_gpu_submit(gpu), "submit mma sdpa");
                    check(h3_gpu_tensor_read_bf16(to, mout, mma_count),
                          "read mma sdpa");
                    double worst = 0.0;
                    size_t worst_at = 0;
                    for (uint32_t row = 0; row < mma_seq; row++) {
                        for (uint32_t head = 0; head < mma_heads; head++) {
                            for (uint32_t d = 0; d < mma_dim; d++) {
                                /* sdpa_ref lays its output out token-major. */
                                size_t ref_index =
                                    ((size_t)row * mma_heads + head) * mma_dim +
                                    d;
                                size_t got_index =
                                    head_major
                                        ? ((size_t)head * mma_seq + row) *
                                              mma_dim + d
                                        : ref_index;
                                double delta = fabs(bf16_to_f32(mout[got_index]) -
                                                    mref[ref_index]);
                                if (delta > worst) {
                                    worst = delta;
                                    worst_at = ref_index;
                                }
                            }
                        }
                    }
                    if (worst >= 6e-2) {
                        fprintf(stderr,
                                "FAIL: mma sdpa (head_major=%d) worst abs "
                                "error %g at %zu\n",
                                head_major, worst, worst_at);
                        failures++;
                    } else {
                        fprintf(stderr,
                                "mma sdpa head_major=%d worst abs error %.5f\n",
                                head_major, worst);
                    }
                }
                unsetenv("H3_SDPA_MMA");
            }
            h3_gpu_tensor_free(tq);
            h3_gpu_tensor_free(tk);
            h3_gpu_tensor_free(tv);
            h3_gpu_tensor_free(to);
        }
        free(mq);
        free(mk);
        free(mv);
        free(mref);
        free(mqb);
        free(mkb);
        free(mvb);
        free(mout);
    }

    /* The CPU reference above is limited to short sequences, so also compare
     * the two GPU kernels against each other at the DiT's own sequence length,
     * where the softmax range is much wider. */
    {
        const uint32_t big_seq = 1874;
        const uint32_t big_heads = 1;
        const uint32_t big_dim = 128;
        const size_t big_count = (size_t)big_seq * big_heads * big_dim;
        const float big_scale = 1.0f / sqrtf((float)big_dim);
        uint16_t *bq = (uint16_t *)malloc(big_count * sizeof(uint16_t));
        uint16_t *bk = (uint16_t *)malloc(big_count * sizeof(uint16_t));
        uint16_t *bv = (uint16_t *)malloc(big_count * sizeof(uint16_t));
        uint16_t *b_wave = (uint16_t *)malloc(big_count * sizeof(uint16_t));
        uint16_t *b_mma = (uint16_t *)malloc(big_count * sizeof(uint16_t));
        check(bq && bk && bv && b_wave && b_mma, "big sdpa host alloc");
        if (bq && bk && bv && b_wave && b_mma) {
            for (size_t i = 0; i < big_count; i++) {
                bq[i] = f32_to_bf16(sinf((float)i * 0.0031f) * 2.0f);
                bk[i] = f32_to_bf16(cosf((float)i * 0.0047f) * 1.5f);
                bv[i] = f32_to_bf16(sinf((float)i * 0.0019f + 0.4f));
            }
            h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, bq, big_count);
            h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, bk, big_count);
            h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, bv, big_count);
            h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, big_count);
            check(tq && tk && tv && to, "big sdpa tensor alloc");
            if (tq && tk && tv && to) {
                setenv("H3_SDPA_WAVE", "1", 1);
                check(h3_gpu_sdpa_bf16_head_major_output(
                          gpu, to, tq, tk, tv, big_seq, big_heads, big_dim,
                          big_scale),
                      "big sdpa wave");
                check(h3_gpu_submit(gpu), "submit big sdpa wave");
                check(h3_gpu_tensor_read_bf16(to, b_wave, big_count),
                      "read big sdpa wave");
                unsetenv("H3_SDPA_WAVE");
                setenv("H3_SDPA_MMA", "1", 1);
                check(h3_gpu_sdpa_bf16_head_major_output(
                          gpu, to, tq, tk, tv, big_seq, big_heads, big_dim,
                          big_scale),
                      "big sdpa mma");
                check(h3_gpu_submit(gpu), "submit big sdpa mma");
                check(h3_gpu_tensor_read_bf16(to, b_mma, big_count),
                      "read big sdpa mma");
                unsetenv("H3_SDPA_MMA");
                /* Both kernels are compared against an F32 CPU reference at
                 * this length: at 1874 keys the output is a heavily cancelling
                 * average, so a relative error against each other says
                 * nothing about which one is closer to the true value. */
                double *scores = (double *)malloc(big_seq * sizeof(double));
                double wave_error = 0.0;
                double mma_error = 0.0;
                double norm = 0.0;
                if (scores) {
                    for (uint32_t row = 0; row < big_seq; row++) {
                        double maximum = -INFINITY;
                        for (uint32_t key = 0; key < big_seq; key++) {
                            double dot = 0.0;
                            for (uint32_t d = 0; d < big_dim; d++)
                                dot += (double)bf16_to_f32(
                                           bq[(size_t)row * big_dim + d]) *
                                       (double)bf16_to_f32(
                                           bk[(size_t)key * big_dim + d]);
                            scores[key] = dot * (double)big_scale;
                            if (scores[key] > maximum) maximum = scores[key];
                        }
                        double sum = 0.0;
                        for (uint32_t key = 0; key < big_seq; key++) {
                            scores[key] = exp(scores[key] - maximum);
                            sum += scores[key];
                        }
                        for (uint32_t d = 0; d < big_dim; d++) {
                            double accumulated = 0.0;
                            for (uint32_t key = 0; key < big_seq; key++)
                                accumulated +=
                                    scores[key] *
                                    (double)bf16_to_f32(
                                        bv[(size_t)key * big_dim + d]);
                            accumulated /= sum;
                            size_t index = (size_t)row * big_dim + d;
                            double dw =
                                bf16_to_f32(b_wave[index]) - accumulated;
                            double dm =
                                bf16_to_f32(b_mma[index]) - accumulated;
                            wave_error += dw * dw;
                            mma_error += dm * dm;
                            norm += accumulated * accumulated;
                        }
                    }
                    free(scores);
                }
                double wave_rel = norm > 0.0 ? sqrt(wave_error / norm) : 1.0;
                double mma_rel = norm > 0.0 ? sqrt(mma_error / norm) : 1.0;
                fprintf(stderr,
                        "sdpa relL2 vs f32 reference at seq %u: wave %.5f, "
                        "mma %.5f\n",
                        big_seq, wave_rel, mma_rel);
                if (mma_rel >= 3.0 * wave_rel + 1e-3) {
                    fprintf(stderr,
                            "FAIL: mma sdpa is much less accurate than wave "
                            "sdpa\n");
                    failures++;
                }
            }
            h3_gpu_tensor_free(tq);
            h3_gpu_tensor_free(tk);
            h3_gpu_tensor_free(tv);
            h3_gpu_tensor_free(to);
        }
        free(bq);
        free(bk);
        free(bv);
        free(b_wave);
        free(b_mma);
    }

    /* Same comparison for the video VAE's F32 attention, whose head_dim is 64.
     * The sequence is deliberately not a multiple of the kernel's 64-wide
     * tiles so both the query and the key mask are exercised. */
    {
        const uint32_t vseq = 900;
        const uint32_t vheads = 2;
        const uint32_t vdim = 64;
        const size_t vcount = (size_t)vseq * vheads * vdim;
        const float vscale = 1.0f / sqrtf((float)vdim);
        float *vq = (float *)malloc(vcount * sizeof(float));
        float *vk = (float *)malloc(vcount * sizeof(float));
        float *vv = (float *)malloc(vcount * sizeof(float));
        float *v_wave = (float *)malloc(vcount * sizeof(float));
        float *v_mma = (float *)malloc(vcount * sizeof(float));
        check(vq && vk && vv && v_wave && v_mma, "f32 sdpa host alloc");
        if (vq && vk && vv && v_wave && v_mma) {
            for (size_t i = 0; i < vcount; i++) {
                vq[i] = sinf((float)i * 0.0037f) * 2.0f;
                vk[i] = cosf((float)i * 0.0053f) * 1.5f;
                vv[i] = sinf((float)i * 0.0023f + 0.7f);
            }
            h3_gpu_tensor *tq = h3_gpu_tensor_from_f32(gpu, vq, vcount);
            h3_gpu_tensor *tk = h3_gpu_tensor_from_f32(gpu, vk, vcount);
            h3_gpu_tensor *tv = h3_gpu_tensor_from_f32(gpu, vv, vcount);
            h3_gpu_tensor *to = h3_gpu_tensor_new_f32(gpu, vcount);
            check(tq && tk && tv && to, "f32 sdpa tensor alloc");
            if (tq && tk && tv && to) {
                setenv("H3_SDPA_F32_WAVE", "1", 1);
                check(h3_gpu_sdpa_f32(gpu, to, tq, tk, tv, vseq, vheads, vdim,
                                      vscale),
                      "f32 sdpa wave");
                check(h3_gpu_submit(gpu), "submit f32 sdpa wave");
                check(h3_gpu_tensor_read_f32(to, v_wave, vcount),
                      "read f32 sdpa wave");
                unsetenv("H3_SDPA_F32_WAVE");
                check(h3_gpu_sdpa_f32(gpu, to, tq, tk, tv, vseq, vheads, vdim,
                                      vscale),
                      "f32 sdpa mma");
                check(h3_gpu_submit(gpu), "submit f32 sdpa mma");
                check(h3_gpu_tensor_read_f32(to, v_mma, vcount),
                      "read f32 sdpa mma");
                double *scores = (double *)malloc(vseq * sizeof(double));
                double wave_error = 0.0;
                double mma_error = 0.0;
                double norm = 0.0;
                if (scores) {
                    for (uint32_t head = 0; head < vheads; head++) {
                        for (uint32_t row = 0; row < vseq; row++) {
                            double maximum = -INFINITY;
                            for (uint32_t key = 0; key < vseq; key++) {
                                double dot = 0.0;
                                for (uint32_t d = 0; d < vdim; d++)
                                    dot += (double)vq[((size_t)row * vheads +
                                                       head) *
                                                          vdim +
                                                      d] *
                                           (double)vk[((size_t)key * vheads +
                                                       head) *
                                                          vdim +
                                                      d];
                                scores[key] = dot * (double)vscale;
                                if (scores[key] > maximum)
                                    maximum = scores[key];
                            }
                            double sum = 0.0;
                            for (uint32_t key = 0; key < vseq; key++) {
                                scores[key] = exp(scores[key] - maximum);
                                sum += scores[key];
                            }
                            for (uint32_t d = 0; d < vdim; d++) {
                                double accumulated = 0.0;
                                for (uint32_t key = 0; key < vseq; key++)
                                    accumulated +=
                                        scores[key] *
                                        (double)vv[((size_t)key * vheads +
                                                    head) *
                                                       vdim +
                                                   d];
                                accumulated /= sum;
                                size_t index =
                                    ((size_t)row * vheads + head) * vdim + d;
                                double dw = v_wave[index] - accumulated;
                                double dm = v_mma[index] - accumulated;
                                wave_error += dw * dw;
                                mma_error += dm * dm;
                                norm += accumulated * accumulated;
                            }
                        }
                    }
                    free(scores);
                }
                double wave_rel = norm > 0.0 ? sqrt(wave_error / norm) : 1.0;
                double mma_rel = norm > 0.0 ? sqrt(mma_error / norm) : 1.0;
                fprintf(stderr,
                        "f32 sdpa relL2 vs f64 reference at seq %u dim 64: "
                        "wave %.6f, mma %.6f\n",
                        vseq, wave_rel, mma_rel);
                if (!(mma_rel < 5e-3)) {
                    fprintf(stderr,
                            "FAIL: f32 mma sdpa relL2 %.6f exceeds 5e-3\n",
                            mma_rel);
                    failures++;
                }
            }
            h3_gpu_tensor_free(tq);
            h3_gpu_tensor_free(tk);
            h3_gpu_tensor_free(tv);
            h3_gpu_tensor_free(to);
        }
        free(vq);
        free(vk);
        free(vv);
        free(v_wave);
        free(v_mma);
    }

    const uint32_t head_norm_sequence = 3;
    const uint32_t head_norm_heads = 2;
    const uint32_t head_norm_dim = 8;
    const size_t head_norm_count =
        (size_t)head_norm_sequence * head_norm_heads * head_norm_dim;
    float head_norm_host[head_norm_count];
    float head_norm_weight_host[head_norm_dim];
    uint16_t head_norm_bf16[head_norm_count];
    uint16_t head_norm_weight_bf16[head_norm_dim];
    uint16_t head_norm_out_bf16[head_norm_count];
    float head_norm_ref[head_norm_count];
    for (size_t i = 0; i < head_norm_count; i++) {
        head_norm_host[i] = sinf((float)i * 0.29f) * 0.6f;
        head_norm_bf16[i] = f32_to_bf16(head_norm_host[i]);
    }
    for (uint32_t i = 0; i < head_norm_dim; i++) {
        head_norm_weight_host[i] = 0.9f + 0.02f * (float)i;
        head_norm_weight_bf16[i] = f32_to_bf16(head_norm_weight_host[i]);
    }
    head_rms_norm_ref(head_norm_host, head_norm_weight_host, head_norm_ref,
                      head_norm_sequence, head_norm_heads, head_norm_dim, 1e-6f);
    h3_gpu_tensor *head_norm_tensor =
        h3_gpu_tensor_from_bf16(gpu, head_norm_bf16, head_norm_count);
    h3_gpu_tensor *head_norm_weight =
        h3_gpu_tensor_from_bf16(gpu, head_norm_weight_bf16, head_norm_dim);
    check(head_norm_tensor && head_norm_weight, "head_rms_norm tensor alloc");
    if (head_norm_tensor && head_norm_weight) {
        check(h3_gpu_head_rms_norm_bf16(
                  gpu, head_norm_tensor, head_norm_weight, head_norm_sequence,
                  head_norm_heads, head_norm_dim, 1e-6f),
              "head_rms_norm");
        check(h3_gpu_submit(gpu), "submit head_rms_norm");
        check(h3_gpu_tensor_read_bf16(head_norm_tensor, head_norm_out_bf16,
                                      head_norm_count),
              "read head_rms_norm");
        for (size_t i = 0; i < head_norm_count; i++) {
            float got = bf16_to_f32(head_norm_out_bf16[i]);
            if (fabsf(got - head_norm_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: head_rms_norm mismatch at %zu got=%f expected=%f\n",
                        i, got, head_norm_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t text_rope_sequence = 4;
    const uint32_t text_rope_q_heads = 4;
    const uint32_t text_rope_kv_heads = 2;
    const uint32_t text_rope_head_dim = 8;
    const uint32_t text_rope_half = text_rope_head_dim / 2u;
    const size_t text_rope_q_count =
        (size_t)text_rope_sequence * text_rope_q_heads * text_rope_head_dim;
    const size_t text_rope_kv_count =
        (size_t)text_rope_sequence * text_rope_kv_heads * text_rope_head_dim;
    const size_t text_rope_table_count =
        (size_t)text_rope_sequence * text_rope_half;
    float text_rope_q_host[text_rope_q_count];
    float text_rope_k_host[text_rope_kv_count];
    float text_rope_cos_host[text_rope_table_count];
    float text_rope_sin_host[text_rope_table_count];
    uint16_t text_rope_q_bf16[text_rope_q_count];
    uint16_t text_rope_k_bf16[text_rope_kv_count];
    uint16_t text_rope_q_out_bf16[text_rope_q_count];
    uint16_t text_rope_k_out_bf16[text_rope_kv_count];
    float text_rope_q_ref[text_rope_q_count];
    float text_rope_k_ref[text_rope_kv_count];
    for (size_t i = 0; i < text_rope_q_count; i++) {
        text_rope_q_host[i] = sinf((float)i * 0.21f);
        text_rope_q_bf16[i] = f32_to_bf16(text_rope_q_host[i]);
        text_rope_q_ref[i] = text_rope_q_host[i];
    }
    for (size_t i = 0; i < text_rope_kv_count; i++) {
        text_rope_k_host[i] = cosf((float)i * 0.17f);
        text_rope_k_bf16[i] = f32_to_bf16(text_rope_k_host[i]);
        text_rope_k_ref[i] = text_rope_k_host[i];
    }
    for (size_t i = 0; i < text_rope_table_count; i++) {
        text_rope_cos_host[i] = cosf((float)i * 0.33f);
        text_rope_sin_host[i] = sinf((float)i * 0.33f);
    }
    rope_text_ref(text_rope_q_ref, text_rope_k_ref, text_rope_cos_host,
                  text_rope_sin_host, text_rope_sequence, text_rope_q_heads,
                  text_rope_kv_heads, text_rope_head_dim);
    h3_gpu_tensor *text_rope_q =
        h3_gpu_tensor_from_bf16(gpu, text_rope_q_bf16, text_rope_q_count);
    h3_gpu_tensor *text_rope_k =
        h3_gpu_tensor_from_bf16(gpu, text_rope_k_bf16, text_rope_kv_count);
    h3_gpu_tensor *text_rope_cos =
        h3_gpu_tensor_from_f32(gpu, text_rope_cos_host, text_rope_table_count);
    h3_gpu_tensor *text_rope_sin =
        h3_gpu_tensor_from_f32(gpu, text_rope_sin_host, text_rope_table_count);
    check(text_rope_q && text_rope_k && text_rope_cos && text_rope_sin,
          "rope_text tensor alloc");
    if (text_rope_q && text_rope_k && text_rope_cos && text_rope_sin) {
        check(h3_gpu_rope_text_bf16(
                  gpu, text_rope_q, text_rope_k, text_rope_cos, text_rope_sin,
                  text_rope_sequence, text_rope_q_heads, text_rope_kv_heads,
                  text_rope_head_dim),
              "rope_text");
        check(h3_gpu_submit(gpu), "submit rope_text");
        check(h3_gpu_tensor_read_bf16(text_rope_q, text_rope_q_out_bf16,
                                      text_rope_q_count),
              "read rope_text query");
        check(h3_gpu_tensor_read_bf16(text_rope_k, text_rope_k_out_bf16,
                                      text_rope_kv_count),
              "read rope_text key");
        for (size_t i = 0; i < text_rope_q_count; i++) {
            float got = bf16_to_f32(text_rope_q_out_bf16[i]);
            if (fabsf(got - text_rope_q_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: rope_text query mismatch at %zu got=%f expected=%f\n",
                        i, got, text_rope_q_ref[i]);
                failures++;
                break;
            }
        }
        for (size_t i = 0; i < text_rope_kv_count; i++) {
            float got = bf16_to_f32(text_rope_k_out_bf16[i]);
            if (fabsf(got - text_rope_k_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: rope_text key mismatch at %zu got=%f expected=%f\n",
                        i, got, text_rope_k_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t qk_rope_sequence = 3;
    const uint32_t qk_rope_q_heads = 4;
    const uint32_t qk_rope_kv_heads = 2;
    const uint32_t qk_rope_head_dim = 8;
    const float qk_rope_epsilon = 1e-5f;
    const size_t qk_rope_q_count =
        (size_t)qk_rope_sequence * qk_rope_q_heads * qk_rope_head_dim;
    const size_t qk_rope_kv_count =
        (size_t)qk_rope_sequence * qk_rope_kv_heads * qk_rope_head_dim;
    const size_t qk_rope_table_count =
        (size_t)qk_rope_sequence * (qk_rope_head_dim / 2u);
    float qk_rope_q_host[qk_rope_q_count];
    float qk_rope_k_host[qk_rope_kv_count];
    float qk_rope_q_weight_host[qk_rope_head_dim];
    float qk_rope_k_weight_host[qk_rope_head_dim];
    float qk_rope_cos_host[qk_rope_table_count];
    float qk_rope_sin_host[qk_rope_table_count];
    uint16_t qk_rope_q_bf16[qk_rope_q_count];
    uint16_t qk_rope_k_bf16[qk_rope_kv_count];
    uint16_t qk_rope_q_weight_bf16[qk_rope_head_dim];
    uint16_t qk_rope_k_weight_bf16[qk_rope_head_dim];
    uint16_t qk_rope_cos_bf16[qk_rope_table_count];
    uint16_t qk_rope_sin_bf16[qk_rope_table_count];
    uint16_t qk_rope_q_out_bf16[qk_rope_q_count];
    uint16_t qk_rope_k_out_bf16[qk_rope_kv_count];
    float qk_rope_q_ref[qk_rope_q_count];
    float qk_rope_k_ref[qk_rope_kv_count];
    float qk_rope_q_out_ref[qk_rope_q_count];
    float qk_rope_k_out_ref[qk_rope_kv_count];
    for (size_t i = 0; i < qk_rope_q_count; i++) {
        qk_rope_q_host[i] = sinf((float)i * 0.19f);
        qk_rope_q_bf16[i] = f32_to_bf16(qk_rope_q_host[i]);
        qk_rope_q_ref[i] = qk_rope_q_host[i];
    }
    for (size_t i = 0; i < qk_rope_kv_count; i++) {
        qk_rope_k_host[i] = cosf((float)i * 0.23f);
        qk_rope_k_bf16[i] = f32_to_bf16(qk_rope_k_host[i]);
        qk_rope_k_ref[i] = qk_rope_k_host[i];
    }
    for (uint32_t d = 0; d < qk_rope_head_dim; d++) {
        qk_rope_q_weight_host[d] = 0.8f + 0.02f * (float)d;
        qk_rope_k_weight_host[d] = 0.7f + 0.03f * (float)d;
        qk_rope_q_weight_bf16[d] = f32_to_bf16(qk_rope_q_weight_host[d]);
        qk_rope_k_weight_bf16[d] = f32_to_bf16(qk_rope_k_weight_host[d]);
    }
    for (size_t i = 0; i < qk_rope_table_count; i++) {
        qk_rope_cos_host[i] = cosf((float)i * 0.41f);
        qk_rope_sin_host[i] = sinf((float)i * 0.41f);
        qk_rope_cos_bf16[i] = f32_to_bf16(qk_rope_cos_host[i]);
        qk_rope_sin_bf16[i] = f32_to_bf16(qk_rope_sin_host[i]);
    }
    text_qk_rope_ref(qk_rope_q_ref, qk_rope_k_ref, qk_rope_q_weight_host,
                     qk_rope_k_weight_host, qk_rope_cos_host, qk_rope_sin_host,
                     qk_rope_q_out_ref, qk_rope_k_out_ref, qk_rope_sequence,
                     qk_rope_q_heads, qk_rope_kv_heads, qk_rope_head_dim,
                     qk_rope_epsilon);
    h3_gpu_tensor *qk_rope_q_in =
        h3_gpu_tensor_from_bf16(gpu, qk_rope_q_bf16, qk_rope_q_count);
    h3_gpu_tensor *qk_rope_k_in =
        h3_gpu_tensor_from_bf16(gpu, qk_rope_k_bf16, qk_rope_kv_count);
    h3_gpu_tensor *qk_rope_q_weight =
        h3_gpu_tensor_from_bf16(gpu, qk_rope_q_weight_bf16, qk_rope_head_dim);
    h3_gpu_tensor *qk_rope_k_weight =
        h3_gpu_tensor_from_bf16(gpu, qk_rope_k_weight_bf16, qk_rope_head_dim);
    h3_gpu_tensor *qk_rope_cos =
        h3_gpu_tensor_from_bf16(gpu, qk_rope_cos_bf16, qk_rope_table_count);
    h3_gpu_tensor *qk_rope_sin =
        h3_gpu_tensor_from_bf16(gpu, qk_rope_sin_bf16, qk_rope_table_count);
    h3_gpu_tensor *qk_rope_q_out =
        h3_gpu_tensor_new_bf16(gpu, qk_rope_q_count);
    h3_gpu_tensor *qk_rope_k_out =
        h3_gpu_tensor_new_bf16(gpu, qk_rope_kv_count);
    check(qk_rope_q_in && qk_rope_k_in && qk_rope_q_weight && qk_rope_k_weight &&
              qk_rope_cos && qk_rope_sin && qk_rope_q_out && qk_rope_k_out,
          "text_qk_rope tensor alloc");
    if (qk_rope_q_in && qk_rope_k_in && qk_rope_q_weight && qk_rope_k_weight &&
        qk_rope_cos && qk_rope_sin && qk_rope_q_out && qk_rope_k_out) {
        check(h3_gpu_text_qk_rope_bf16(
                  gpu, qk_rope_q_out, qk_rope_k_out, qk_rope_q_in, qk_rope_k_in,
                  qk_rope_q_weight, qk_rope_k_weight, qk_rope_cos, qk_rope_sin,
                  qk_rope_sequence, qk_rope_q_heads, qk_rope_kv_heads,
                  qk_rope_head_dim, qk_rope_epsilon),
              "text_qk_rope");
        check(h3_gpu_submit(gpu), "submit text_qk_rope");
        check(h3_gpu_tensor_read_bf16(qk_rope_q_out, qk_rope_q_out_bf16,
                                      qk_rope_q_count),
              "read text_qk_rope query");
        check(h3_gpu_tensor_read_bf16(qk_rope_k_out, qk_rope_k_out_bf16,
                                      qk_rope_kv_count),
              "read text_qk_rope key");
        for (size_t i = 0; i < qk_rope_q_count; i++) {
            float got = bf16_to_f32(qk_rope_q_out_bf16[i]);
            if (fabsf(got - qk_rope_q_out_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: text_qk_rope query mismatch at %zu got=%f expected=%f\n",
                        i, got, qk_rope_q_out_ref[i]);
                failures++;
                break;
            }
        }
        for (size_t i = 0; i < qk_rope_kv_count; i++) {
            float got = bf16_to_f32(qk_rope_k_out_bf16[i]);
            if (fabsf(got - qk_rope_k_out_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: text_qk_rope key mismatch at %zu got=%f expected=%f\n",
                        i, got, qk_rope_k_out_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t gqa_sequence = 4;
    const uint32_t gqa_q_heads = 4;
    const uint32_t gqa_kv_heads = 2;
    const uint32_t gqa_head_dim = 8;
    const size_t gqa_q_count =
        (size_t)gqa_sequence * gqa_q_heads * gqa_head_dim;
    const size_t gqa_kv_count =
        (size_t)gqa_sequence * gqa_kv_heads * gqa_head_dim;
    const float gqa_scale = 1.0f / sqrtf((float)gqa_head_dim);
    float gqa_q_host[gqa_q_count];
    float gqa_k_host[gqa_kv_count];
    float gqa_v_host[gqa_kv_count];
    uint16_t gqa_q_bf16[gqa_q_count];
    uint16_t gqa_k_bf16[gqa_kv_count];
    uint16_t gqa_v_bf16[gqa_kv_count];
    uint16_t gqa_out_bf16[gqa_q_count];
    float gqa_out_ref[gqa_q_count];
    for (size_t i = 0; i < gqa_q_count; i++) {
        gqa_q_host[i] = sinf((float)i * 0.27f);
        gqa_q_bf16[i] = f32_to_bf16(gqa_q_host[i]);
    }
    for (size_t i = 0; i < gqa_kv_count; i++) {
        gqa_k_host[i] = cosf((float)i * 0.13f);
        gqa_v_host[i] = sinf((float)i * 0.09f) * 0.5f;
        gqa_k_bf16[i] = f32_to_bf16(gqa_k_host[i]);
        gqa_v_bf16[i] = f32_to_bf16(gqa_v_host[i]);
    }
    gqa_causal_ref(gqa_q_host, gqa_k_host, gqa_v_host, gqa_out_ref, gqa_sequence,
                   gqa_q_heads, gqa_kv_heads, gqa_head_dim, gqa_scale);
    h3_gpu_tensor *gqa_q =
        h3_gpu_tensor_from_bf16(gpu, gqa_q_bf16, gqa_q_count);
    h3_gpu_tensor *gqa_k =
        h3_gpu_tensor_from_bf16(gpu, gqa_k_bf16, gqa_kv_count);
    h3_gpu_tensor *gqa_v =
        h3_gpu_tensor_from_bf16(gpu, gqa_v_bf16, gqa_kv_count);
    h3_gpu_tensor *gqa_out = h3_gpu_tensor_new_bf16(gpu, gqa_q_count);
    check(gqa_q && gqa_k && gqa_v && gqa_out, "gqa_causal tensor alloc");
    if (gqa_q && gqa_k && gqa_v && gqa_out) {
        check(h3_gpu_gqa_causal_bf16(gpu, gqa_out, gqa_q, gqa_k, gqa_v,
                                     gqa_sequence, gqa_q_heads, gqa_kv_heads,
                                     gqa_head_dim, gqa_scale),
              "gqa_causal");
        check(h3_gpu_submit(gpu), "submit gqa_causal");
        check(h3_gpu_tensor_read_bf16(gqa_out, gqa_out_bf16, gqa_q_count),
              "read gqa_causal");
        for (size_t i = 0; i < gqa_q_count; i++) {
            float got = bf16_to_f32(gqa_out_bf16[i]);
            if (fabsf(got - gqa_out_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: gqa_causal mismatch at %zu got=%f expected=%f\n",
                        i, got, gqa_out_ref[i]);
                failures++;
                break;
            }
        }
    }

    const size_t silu_mul_count = 32;
    float silu_mul_gate_host[silu_mul_count];
    float silu_mul_up_host[silu_mul_count];
    uint16_t silu_mul_gate_bf16[silu_mul_count];
    uint16_t silu_mul_up_bf16[silu_mul_count];
    uint16_t silu_mul_out_bf16[silu_mul_count];
    float silu_mul_out_ref[silu_mul_count];
    for (size_t i = 0; i < silu_mul_count; i++) {
        silu_mul_gate_host[i] = sinf((float)i * 0.41f);
        silu_mul_up_host[i] = cosf((float)i * 0.37f);
        silu_mul_gate_bf16[i] = f32_to_bf16(silu_mul_gate_host[i]);
        silu_mul_up_bf16[i] = f32_to_bf16(silu_mul_up_host[i]);
    }
    silu_mul_ref(silu_mul_gate_host, silu_mul_up_host, silu_mul_out_ref,
                 silu_mul_count);
    h3_gpu_tensor *silu_mul_gate =
        h3_gpu_tensor_from_bf16(gpu, silu_mul_gate_bf16, silu_mul_count);
    h3_gpu_tensor *silu_mul_up =
        h3_gpu_tensor_from_bf16(gpu, silu_mul_up_bf16, silu_mul_count);
    h3_gpu_tensor *silu_mul_out =
        h3_gpu_tensor_new_bf16(gpu, silu_mul_count);
    check(silu_mul_gate && silu_mul_up && silu_mul_out, "silu_mul tensor alloc");
    if (silu_mul_gate && silu_mul_up && silu_mul_out) {
        check(h3_gpu_silu_mul_bf16(gpu, silu_mul_out, silu_mul_gate, silu_mul_up,
                                   (uint32_t)silu_mul_count),
              "silu_mul");
        check(h3_gpu_submit(gpu), "submit silu_mul");
        check(h3_gpu_tensor_read_bf16(silu_mul_out, silu_mul_out_bf16,
                                      silu_mul_count),
              "read silu_mul");
        for (size_t i = 0; i < silu_mul_count; i++) {
            float got = bf16_to_f32(silu_mul_out_bf16[i]);
            if (fabsf(got - silu_mul_out_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: silu_mul mismatch at %zu got=%f expected=%f\n",
                        i, got, silu_mul_out_ref[i]);
                failures++;
                break;
            }
        }
    }

    enum {
        TP_FULL_ROWS = 6,
        TP_REDUCED_ROWS = 4,
        TP_BASELINE_ROWS = 2,
        TP_WIDTH = 4,
        TP_PADDING = 4
    };
    const float tp_input_values[TP_FULL_ROWS * TP_WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f,       10.0f, 12.0f, 14.0f, 16.0f,
        14.0f, 16.0f, 18.0f, 20.0f,   -8.0f, -4.0f, 0.0f, 4.0f,
        -4.0f, 0.0f, 4.0f, 8.0f,      20.0f, 21.0f, 22.0f, 23.0f,
    };
    const float tp_pooled_values[TP_REDUCED_ROWS * TP_WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f,  12.0f, 14.0f, 16.0f, 18.0f,
        -6.0f, -2.0f, 2.0f, 6.0f, 20.0f, 21.0f, 22.0f, 23.0f,
    };
    const float tp_processed_values[TP_REDUCED_ROWS * TP_WIDTH] = {
        2.0f, 3.0f, 4.0f, 5.0f,  14.0f, 12.0f, 20.0f, 14.0f,
        -5.0f, 0.0f, 5.0f, 10.0f, 30.0f, 31.0f, 32.0f, 33.0f,
    };
    const float tp_expanded_values[TP_FULL_ROWS * TP_WIDTH] = {
        2.0f, 3.0f, 4.0f, 5.0f,  12.0f, 10.0f, 18.0f, 12.0f,
        16.0f, 14.0f, 22.0f, 16.0f, -7.0f, -2.0f, 3.0f, 8.0f,
        -3.0f, 2.0f, 7.0f, 12.0f, 30.0f, 31.0f, 32.0f, 33.0f,
    };
    const uint32_t tp_pairs[TP_REDUCED_ROWS * 2] = {0, 0, 1, 2, 3, 4, 5, 5};
    const uint32_t tp_baseline_indices[TP_REDUCED_ROWS] = {
        UINT32_MAX, 0, 1, UINT32_MAX};
    const uint32_t tp_parents[TP_FULL_ROWS] = {0, 1, 1, 2, 2, 3};
    uint16_t tp_input_bf16[TP_PADDING + TP_FULL_ROWS * TP_WIDTH];
    uint16_t tp_processed_bf16[TP_REDUCED_ROWS * TP_WIDTH];
    uint16_t tp_expected_pooled[TP_REDUCED_ROWS * TP_WIDTH];
    uint16_t tp_expected_expanded[TP_FULL_ROWS * TP_WIDTH];
    for (size_t i = 0; i < TP_PADDING; i++)
        tp_input_bf16[i] = f32_to_bf16(-99.0f);
    for (size_t i = 0; i < (size_t)TP_FULL_ROWS * TP_WIDTH; i++) {
        tp_input_bf16[TP_PADDING + i] = f32_to_bf16(tp_input_values[i]);
        tp_expected_expanded[i] = f32_to_bf16(tp_expanded_values[i]);
    }
    for (size_t i = 0; i < (size_t)TP_REDUCED_ROWS * TP_WIDTH; i++) {
        tp_processed_bf16[i] = f32_to_bf16(tp_processed_values[i]);
        tp_expected_pooled[i] = f32_to_bf16(tp_pooled_values[i]);
    }
    h3_gpu_tensor *tp_input = h3_gpu_tensor_from_bf16(
        gpu, tp_input_bf16, TP_PADDING + TP_FULL_ROWS * TP_WIDTH);
    h3_gpu_tensor *tp_pairs_t =
        h3_gpu_tensor_from_u32(gpu, tp_pairs, TP_REDUCED_ROWS * 2);
    h3_gpu_tensor *tp_baseline_idx = h3_gpu_tensor_from_u32(
        gpu, tp_baseline_indices, TP_REDUCED_ROWS);
    h3_gpu_tensor *tp_pooled = h3_gpu_tensor_new_bf16(
        gpu, (TP_REDUCED_ROWS + TP_BASELINE_ROWS) * TP_WIDTH);
    h3_gpu_tensor *tp_original = h3_gpu_tensor_new_bf16(
        gpu, TP_PADDING + TP_FULL_ROWS * TP_WIDTH);
    h3_gpu_tensor *tp_processed = h3_gpu_tensor_from_bf16(
        gpu, tp_processed_bf16, TP_REDUCED_ROWS * TP_WIDTH);
    h3_gpu_tensor *tp_parents_t =
        h3_gpu_tensor_from_u32(gpu, tp_parents, TP_FULL_ROWS);
    h3_gpu_tensor *tp_expanded = h3_gpu_tensor_new_bf16(
        gpu, TP_FULL_ROWS * TP_WIDTH);
    check(tp_input && tp_pairs_t && tp_baseline_idx && tp_pooled &&
              tp_original && tp_processed && tp_parents_t && tp_expanded,
          "token pool tensor alloc");
    if (tp_input && tp_pairs_t && tp_baseline_idx && tp_pooled &&
        tp_original && tp_processed && tp_parents_t && tp_expanded) {
        check(h3_gpu_token_pool_bf16(
                  gpu, tp_pooled, tp_input, TP_PADDING, tp_original,
                  TP_PADDING, tp_pooled, TP_REDUCED_ROWS * TP_WIDTH,
                  tp_baseline_idx, tp_pairs_t, TP_FULL_ROWS, TP_REDUCED_ROWS,
                  TP_BASELINE_ROWS, TP_WIDTH),
              "token_pool");
        check(h3_gpu_submit(gpu), "submit token_pool");
        uint16_t tp_got_pooled[TP_REDUCED_ROWS * TP_WIDTH];
        check(h3_gpu_tensor_read_bf16(tp_pooled, tp_got_pooled,
                                      TP_REDUCED_ROWS * TP_WIDTH),
              "read token_pool");
        if (memcmp(tp_got_pooled, tp_expected_pooled, sizeof(tp_got_pooled)) !=
            0) {
            fprintf(stderr, "FAIL: token_pool mismatch\n");
            failures++;
        }
        check(h3_gpu_token_expand_delta_bf16(
                  gpu, tp_expanded, tp_original, TP_PADDING, tp_processed,
                  tp_pooled, TP_REDUCED_ROWS * TP_WIDTH, tp_baseline_idx,
                  tp_parents_t, TP_FULL_ROWS, TP_REDUCED_ROWS,
                  TP_BASELINE_ROWS, TP_WIDTH, 1, 1.0f),
              "token_expand_delta");
        check(h3_gpu_submit(gpu), "submit token_expand_delta");
        uint16_t tp_got_expanded[TP_FULL_ROWS * TP_WIDTH];
        check(h3_gpu_tensor_read_bf16(tp_expanded, tp_got_expanded,
                                      TP_FULL_ROWS * TP_WIDTH),
              "read token_expand_delta");
        if (memcmp(tp_got_expanded, tp_expected_expanded,
                   sizeof(tp_got_expanded)) != 0) {
            fprintf(stderr, "FAIL: token_expand_delta mismatch\n");
            failures++;
        }
    }

    /* GA_MAPS: the row map selects a modulation row, so the modulation has to
     * cover every value in it. It used to be one row short, which put both the
     * reference and the kernel one row past the end of their buffers; they
     * agreed only as long as the memory behind each happened to be zero. */
    enum { GA_ROWS = 6, GA_WIDTH = 4, GA_SLOTS = 2, GA_HEAD = 3, GA_MAPS = 2 };
    float ga_input_host[GA_ROWS * GA_WIDTH];
    float ga_branch_host[GA_ROWS * GA_WIDTH];
    float ga_norm_host[GA_WIDTH];
    float ga_mod_host[GA_MAPS * GA_SLOTS * GA_WIDTH];
    uint32_t ga_row_map[GA_ROWS] = {0, 1, 0, 1, 0, 1};
    uint16_t ga_input_bf16[GA_ROWS * GA_WIDTH];
    uint16_t ga_branch_bf16[GA_ROWS * GA_WIDTH];
    uint16_t ga_norm_bf16[GA_WIDTH];
    uint16_t ga_mod_bf16[GA_MAPS * GA_SLOTS * GA_WIDTH];
    uint16_t ga_out_bf16[GA_ROWS * GA_WIDTH];
    float ga_gate_ref_f32[GA_ROWS * GA_WIDTH];
    float ga_out_ref_f32[GA_ROWS * GA_WIDTH];
    float ga_head_weight_host[GA_HEAD * GA_WIDTH];
    float ga_head_bias_host[GA_HEAD];
    uint16_t ga_head_weight_bf16[GA_HEAD * GA_WIDTH];
    uint16_t ga_head_bias_bf16[GA_HEAD];
    uint16_t ga_head_ref[GA_ROWS * GA_HEAD];
    uint16_t ga_head_out_bf16[GA_ROWS * GA_HEAD];
    float ga_adaln_host[GA_ROWS * GA_WIDTH];
    float ga_head_ref_f32[GA_ROWS * GA_HEAD];
    for (size_t i = 0; i < (size_t)GA_ROWS * GA_WIDTH; i++) {
        ga_input_host[i] = sinf((float)i * 0.19f);
        ga_branch_host[i] = cosf((float)i * 0.23f) * 0.5f;
        ga_input_bf16[i] = f32_to_bf16(ga_input_host[i]);
        ga_branch_bf16[i] = f32_to_bf16(ga_branch_host[i]);
    }
    for (uint32_t i = 0; i < GA_WIDTH; i++) {
        ga_norm_host[i] = 0.8f + 0.05f * (float)i;
        ga_norm_bf16[i] = f32_to_bf16(ga_norm_host[i]);
    }
    for (size_t i = 0; i < (size_t)GA_MAPS * GA_SLOTS * GA_WIDTH; i++) {
        ga_mod_host[i] = sinf((float)i * 0.31f) * 0.25f;
        ga_mod_bf16[i] = f32_to_bf16(ga_mod_host[i]);
        ga_mod_host[i] = bf16_to_f32(ga_mod_bf16[i]);
    }
    gate_ref(ga_input_host, ga_branch_host, ga_mod_host, ga_row_map,
             ga_gate_ref_f32, GA_ROWS, GA_WIDTH, GA_SLOTS, 0);
    adaln_ref(ga_gate_ref_f32, ga_norm_host, ga_mod_host, ga_row_map,
              ga_out_ref_f32, GA_ROWS, GA_WIDTH, GA_SLOTS, 0, 1, 1e-5f);
    uint16_t ga_out_ref[GA_ROWS * GA_WIDTH];
    for (size_t i = 0; i < (size_t)GA_ROWS * GA_WIDTH; i++)
        ga_out_ref[i] = f32_to_bf16(ga_out_ref_f32[i]);
    for (size_t i = 0; i < (size_t)GA_HEAD * GA_WIDTH; i++) {
        ga_head_weight_host[i] = ((float)((int)(i % 7) - 3)) * 0.125f;
        ga_head_weight_bf16[i] = f32_to_bf16(ga_head_weight_host[i]);
    }
    for (uint32_t i = 0; i < GA_HEAD; i++) {
        ga_head_bias_host[i] = (float)i * 0.0625f;
        ga_head_bias_bf16[i] = f32_to_bf16(ga_head_bias_host[i]);
    }
    adaln_ref(ga_input_host, ga_norm_host, ga_mod_host, ga_row_map,
              ga_adaln_host, GA_ROWS, GA_WIDTH, GA_SLOTS, 0, 1, 1e-5f);
    linear_ref(ga_adaln_host, ga_head_weight_host, ga_head_bias_host,
               ga_head_ref_f32, GA_ROWS, GA_WIDTH, GA_HEAD);
    for (size_t i = 0; i < (size_t)GA_ROWS * GA_HEAD; i++)
        ga_head_ref[i] = f32_to_bf16(ga_head_ref_f32[i]);
    h3_gpu_tensor *ga_residual =
        h3_gpu_tensor_from_bf16(gpu, ga_input_bf16, GA_ROWS * GA_WIDTH);
    h3_gpu_tensor *ga_branch =
        h3_gpu_tensor_from_bf16(gpu, ga_branch_bf16, GA_ROWS * GA_WIDTH);
    h3_gpu_tensor *ga_norm =
        h3_gpu_tensor_from_bf16(gpu, ga_norm_bf16, GA_WIDTH);
    h3_gpu_tensor *ga_mod =
        h3_gpu_tensor_from_bf16(gpu, ga_mod_bf16,
                                GA_MAPS * GA_SLOTS * GA_WIDTH);
    h3_gpu_tensor *ga_map =
        h3_gpu_tensor_from_u32(gpu, ga_row_map, GA_ROWS);
    h3_gpu_tensor *ga_gate_res =
        h3_gpu_tensor_new_bf16(gpu, GA_ROWS * GA_WIDTH);
    h3_gpu_tensor *ga_out = h3_gpu_tensor_new_bf16(gpu, GA_ROWS * GA_WIDTH);
    h3_gpu_tensor *ga_head_weight =
        h3_gpu_tensor_from_bf16(gpu, ga_head_weight_bf16, GA_HEAD * GA_WIDTH);
    h3_gpu_tensor *ga_head_bias =
        h3_gpu_tensor_from_bf16(gpu, ga_head_bias_bf16, GA_HEAD);
    h3_gpu_tensor *ga_inverse = h3_gpu_tensor_new_f32(gpu, GA_ROWS);
    h3_gpu_tensor *ga_head_out =
        h3_gpu_tensor_new_bf16(gpu, GA_ROWS * GA_HEAD);
    if (ga_residual && ga_branch && ga_norm && ga_mod && ga_map &&
        ga_gate_res && ga_out && ga_head_weight && ga_head_bias &&
        ga_inverse && ga_head_out) {
        check(h3_gpu_gate_adaln_bf16(
                  gpu, ga_gate_res, ga_out, ga_residual, ga_branch, ga_norm,
                  ga_mod, ga_mod, ga_map, GA_ROWS, GA_WIDTH, GA_SLOTS, 0, 0, 1,
                  1e-5f),
              "gate_adaln");
        check(h3_gpu_adaln_linear_bf16(
                  gpu, ga_head_out, ga_inverse, ga_residual, 0, ga_norm,
                  ga_mod, ga_map, ga_head_weight, ga_head_bias, GA_ROWS,
                  GA_WIDTH, GA_HEAD, GA_SLOTS, 0, 1, 1e-5f),
              "adaln_linear");
        check(h3_gpu_submit(gpu), "submit gate_adaln/adaln_linear");
        check(h3_gpu_tensor_read_bf16(ga_out, ga_out_bf16, GA_ROWS * GA_WIDTH),
              "read gate_adaln");
        check(h3_gpu_tensor_read_bf16(ga_head_out, ga_head_out_bf16,
                                      GA_ROWS * GA_HEAD),
              "read adaln_linear");
        for (size_t i = 0; i < (size_t)GA_ROWS * GA_WIDTH; i++) {
            float got = bf16_to_f32(ga_out_bf16[i]);
            float want = bf16_to_f32(ga_out_ref[i]);
            if (fabsf(got - want) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: gate_adaln mismatch at %zu got=%f expected=%f\n",
                        i, got, want);
                failures++;
                break;
            }
        }
        for (size_t i = 0; i < (size_t)GA_ROWS * GA_HEAD; i++) {
            float got = bf16_to_f32(ga_head_out_bf16[i]);
            float want = bf16_to_f32(ga_head_ref[i]);
            if (fabsf(got - want) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: adaln_linear mismatch at %zu got=%f expected=%f\n",
                        i, got, want);
                failures++;
                break;
            }
        }
    }

    const uint32_t iq_rows = 4;
    const uint32_t iq_cols = 32;
    const uint32_t iq_out = 16;
    const uint32_t iq_padded = (iq_rows + 127u) & ~127u;
    float iq_input_host[iq_rows * iq_cols];
    float iq_weight_host[iq_out * iq_cols];
    uint16_t iq_input_bf16[iq_rows * iq_cols];
    uint16_t iq_weight_bf16[iq_out * iq_cols];
    int8_t iq_weight_qi8[iq_out * iq_cols];
    float iq_weight_scales[iq_out];
    int8_t iq_input_qi8[iq_rows * iq_cols];
    float iq_input_scales[iq_rows];
    float iq_linear_ref[iq_rows * iq_out];
    uint16_t iq_linear_out_bf16[iq_rows * iq_out];
    for (size_t i = 0; i < (size_t)iq_rows * iq_cols; i++) {
        iq_input_host[i] = sinf((float)i * 0.13f) * 0.75f;
        iq_input_bf16[i] = f32_to_bf16(iq_input_host[i]);
    }
    for (size_t i = 0; i < (size_t)iq_out * iq_cols; i++) {
        iq_weight_host[i] = cosf((float)i * 0.07f) * 0.35f;
        iq_weight_bf16[i] = f32_to_bf16(iq_weight_host[i]);
    }
    for (size_t i = 0; i < (size_t)iq_rows * iq_cols; i++)
        iq_input_host[i] = bf16_to_f32(iq_input_bf16[i]);
    for (size_t i = 0; i < (size_t)iq_out * iq_cols; i++)
        iq_weight_host[i] = bf16_to_f32(iq_weight_bf16[i]);
    quantize_bf16_int8_rows_ref(iq_weight_host, iq_weight_qi8, iq_weight_scales,
                                iq_out, iq_cols, 1.0f);
    quantize_bf16_int8_rows_ref(iq_input_host, iq_input_qi8, iq_input_scales,
                                iq_rows, iq_cols, 1.0f);
    linear_int8_ref(iq_input_qi8, iq_input_scales, iq_weight_qi8,
                    iq_weight_scales, iq_linear_ref, iq_rows, iq_cols, iq_out);

    h3_gpu_tensor *iq_weight_bf16_t =
        h3_gpu_tensor_from_bf16(gpu, iq_weight_bf16, (size_t)iq_out * iq_cols);
    h3_gpu_tensor *iq_weight_i8 =
        h3_gpu_tensor_new_i8(gpu, (size_t)iq_out * iq_cols);
    h3_gpu_tensor *iq_weight_scale_t =
        h3_gpu_tensor_new_f32(gpu, iq_out);
    h3_gpu_tensor *iq_input_t =
        h3_gpu_tensor_from_bf16(gpu, iq_input_bf16, (size_t)iq_rows * iq_cols);
    h3_gpu_tensor *iq_quant_t =
        h3_gpu_tensor_new_i8(gpu, (size_t)iq_padded * iq_cols);
    h3_gpu_tensor *iq_input_scale_t =
        h3_gpu_tensor_new_f32(gpu, iq_padded);
    h3_gpu_tensor *iq_out_t =
        h3_gpu_tensor_new_bf16(gpu, (size_t)iq_rows * iq_out);
    check(iq_weight_bf16_t && iq_weight_i8 && iq_weight_scale_t && iq_input_t &&
              iq_quant_t && iq_input_scale_t && iq_out_t,
          "int8 linear tensor alloc");
    if (iq_weight_bf16_t && iq_weight_i8 && iq_weight_scale_t && iq_input_t &&
        iq_quant_t && iq_input_scale_t && iq_out_t) {
        check(h3_gpu_quantize_weight_int8(gpu, iq_weight_i8, iq_weight_scale_t,
                                          iq_weight_bf16_t, iq_out, iq_cols),
              "quantize_weight_int8");
        check(h3_gpu_linear_int8_bf16(gpu, iq_out_t, iq_quant_t,
                                      iq_input_scale_t, iq_input_t, iq_weight_i8,
                                      iq_weight_scale_t, iq_rows, iq_cols,
                                      iq_out, 0),
              "linear_int8_bf16");
        check(h3_gpu_submit(gpu), "submit int8 linear");
        float iq_got_scales[iq_out];
        check(h3_gpu_tensor_read_f32(iq_weight_scale_t, iq_got_scales, iq_out),
              "read weight scales");
        for (uint32_t i = 0; i < iq_out; i++) {
            if (fabsf(iq_got_scales[i] - iq_weight_scales[i]) >= 1e-5f) {
                fprintf(stderr,
                        "FAIL: weight scale mismatch at %u got=%f expected=%f\n",
                        i, iq_got_scales[i], iq_weight_scales[i]);
                failures++;
                break;
            }
        }
        check(h3_gpu_tensor_read_bf16(iq_out_t, iq_linear_out_bf16,
                                      (size_t)iq_rows * iq_out),
              "read linear_int8");
        for (size_t i = 0; i < (size_t)iq_rows * iq_out; i++) {
            float got = bf16_to_f32(iq_linear_out_bf16[i]);
            if (fabsf(got - iq_linear_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: linear_int8 mismatch at %zu got=%f expected=%f\n",
                        i, got, iq_linear_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t hm_rows = 3;
    const uint32_t hm_heads = 2;
    const uint32_t hm_dim = 8;
    const uint32_t hm_out = 16;
    const uint32_t hm_in = hm_heads * hm_dim;
    const uint32_t hm_padded = (hm_rows + 127u) & ~127u;
    float hm_input_host[hm_rows * hm_in];
    float hm_row_major[hm_rows * hm_in];
    float hm_weight_host[hm_out * hm_in];
    uint16_t hm_input_bf16[hm_rows * hm_in];
    uint16_t hm_weight_bf16[hm_out * hm_in];
    int8_t hm_weight_qi8[hm_out * hm_in];
    float hm_weight_scales[hm_out];
    int8_t hm_input_qi8[hm_rows * hm_in];
    float hm_input_scales[hm_rows];
    float hm_ref[hm_rows * hm_out];
    uint16_t hm_out_bf16[hm_rows * hm_out];
    for (uint32_t head = 0; head < hm_heads; head++) {
        for (uint32_t row = 0; row < hm_rows; row++) {
            for (uint32_t d = 0; d < hm_dim; d++) {
                size_t hm_index = ((size_t)head * hm_rows + row) * hm_dim + d;
                size_t rm_index = ((size_t)row * hm_heads + head) * hm_dim + d;
                hm_input_host[hm_index] =
                    sinf((float)hm_index * 0.19f) * 0.6f;
                hm_input_bf16[hm_index] = f32_to_bf16(hm_input_host[hm_index]);
                hm_row_major[rm_index] = bf16_to_f32(hm_input_bf16[hm_index]);
            }
        }
    }
    for (size_t i = 0; i < (size_t)hm_out * hm_in; i++) {
        hm_weight_host[i] = cosf((float)i * 0.11f) * 0.4f;
        hm_weight_bf16[i] = f32_to_bf16(hm_weight_host[i]);
        hm_weight_host[i] = bf16_to_f32(hm_weight_bf16[i]);
    }
    quantize_bf16_int8_rows_ref(hm_weight_host, hm_weight_qi8, hm_weight_scales,
                                hm_out, hm_in, 1.0f);
    quantize_bf16_int8_rows_ref(hm_row_major, hm_input_qi8, hm_input_scales,
                                hm_rows, hm_in, 1.0f);
    linear_int8_ref(hm_input_qi8, hm_input_scales, hm_weight_qi8,
                    hm_weight_scales, hm_ref, hm_rows, hm_in, hm_out);

    h3_gpu_tensor *hm_input_t =
        h3_gpu_tensor_from_bf16(gpu, hm_input_bf16, (size_t)hm_rows * hm_in);
    h3_gpu_tensor *hm_weight_bf16_t =
        h3_gpu_tensor_from_bf16(gpu, hm_weight_bf16, (size_t)hm_out * hm_in);
    h3_gpu_tensor *hm_weight_i8 =
        h3_gpu_tensor_new_i8(gpu, (size_t)hm_out * hm_in);
    h3_gpu_tensor *hm_weight_scale_t = h3_gpu_tensor_new_f32(gpu, hm_out);
    h3_gpu_tensor *hm_quant_t =
        h3_gpu_tensor_new_i8(gpu, (size_t)hm_padded * hm_in);
    h3_gpu_tensor *hm_input_scale_t = h3_gpu_tensor_new_f32(gpu, hm_padded);
    h3_gpu_tensor *hm_out_t =
        h3_gpu_tensor_new_bf16(gpu, (size_t)hm_rows * hm_out);
    check(hm_input_t && hm_weight_bf16_t && hm_weight_i8 && hm_weight_scale_t &&
              hm_quant_t && hm_input_scale_t && hm_out_t,
          "head_major int8 alloc");
    if (hm_input_t && hm_weight_bf16_t && hm_weight_i8 && hm_weight_scale_t &&
        hm_quant_t && hm_input_scale_t && hm_out_t) {
        check(h3_gpu_quantize_weight_int8(gpu, hm_weight_i8, hm_weight_scale_t,
                                          hm_weight_bf16_t, hm_out, hm_in),
              "quantize head_major weight");
        check(h3_gpu_linear_int8_head_major_bf16(
                  gpu, hm_out_t, hm_quant_t, hm_input_scale_t, hm_input_t,
                  hm_weight_i8, hm_weight_scale_t, hm_rows, hm_heads, hm_dim,
                  hm_out),
              "linear_int8_head_major");
        check(h3_gpu_submit(gpu), "submit head_major int8");
        check(h3_gpu_tensor_read_bf16(hm_out_t, hm_out_bf16,
                                      (size_t)hm_rows * hm_out),
              "read head_major int8");
        for (size_t i = 0; i < (size_t)hm_rows * hm_out; i++) {
            float got = bf16_to_f32(hm_out_bf16[i]);
            if (fabsf(got - hm_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: head_major int8 mismatch at %zu got=%f expected=%f\n",
                        i, got, hm_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t mlp8_rows = 4;
    const uint32_t mlp8_in = 32;
    const uint32_t mlp8_hidden = 64;
    const uint32_t mlp8_out = 32;
    const uint32_t mlp8_padded = (mlp8_rows + 127u) & ~127u;
    float mlp8_input_host[mlp8_rows * mlp8_in];
    float mlp8_fc1_host[mlp8_hidden * 2 * mlp8_in];
    float mlp8_fc2_host[mlp8_out * mlp8_hidden];
    uint16_t mlp8_input_bf16[mlp8_rows * mlp8_in];
    uint16_t mlp8_fc1_bf16[mlp8_hidden * 2 * mlp8_in];
    uint16_t mlp8_fc2_bf16[mlp8_out * mlp8_hidden];
    int8_t mlp8_fc1_qi8[mlp8_hidden * 2 * mlp8_in];
    int8_t mlp8_fc2_qi8[mlp8_out * mlp8_hidden];
    float mlp8_fc1_scales[mlp8_hidden * 2];
    float mlp8_fc2_scales[mlp8_out];
    int8_t mlp8_qin[mlp8_rows * mlp8_in];
    float mlp8_qin_scales[mlp8_rows];
    float mlp8_fc1_out[mlp8_rows * mlp8_hidden * 2];
    float mlp8_activated[mlp8_rows * mlp8_hidden];
    int8_t mlp8_qact[mlp8_rows * mlp8_hidden];
    float mlp8_qact_scales[mlp8_rows];
    float mlp8_ref[mlp8_rows * mlp8_out];
    uint16_t mlp8_out_bf16[mlp8_rows * mlp8_out];
    for (size_t i = 0; i < (size_t)mlp8_rows * mlp8_in; i++) {
        mlp8_input_host[i] = sinf((float)i * 0.09f) * 0.5f;
        mlp8_input_bf16[i] = f32_to_bf16(mlp8_input_host[i]);
        mlp8_input_host[i] = bf16_to_f32(mlp8_input_bf16[i]);
    }
    for (size_t i = 0; i < (size_t)mlp8_hidden * 2 * mlp8_in; i++) {
        mlp8_fc1_host[i] = cosf((float)i * 0.05f) * 0.2f;
        mlp8_fc1_bf16[i] = f32_to_bf16(mlp8_fc1_host[i]);
        mlp8_fc1_host[i] = bf16_to_f32(mlp8_fc1_bf16[i]);
    }
    for (size_t i = 0; i < (size_t)mlp8_out * mlp8_hidden; i++) {
        mlp8_fc2_host[i] = sinf((float)i * 0.04f) * 0.25f;
        mlp8_fc2_bf16[i] = f32_to_bf16(mlp8_fc2_host[i]);
        mlp8_fc2_host[i] = bf16_to_f32(mlp8_fc2_bf16[i]);
    }
    quantize_bf16_int8_rows_ref(mlp8_fc1_host, mlp8_fc1_qi8, mlp8_fc1_scales,
                                mlp8_hidden * 2, mlp8_in, 1.0f);
    quantize_bf16_int8_rows_ref(mlp8_fc2_host, mlp8_fc2_qi8, mlp8_fc2_scales,
                                mlp8_out, mlp8_hidden, 1.0f);
    quantize_bf16_int8_rows_ref(mlp8_input_host, mlp8_qin, mlp8_qin_scales,
                                mlp8_rows, mlp8_in, 1.0f);
    linear_int8_ref(mlp8_qin, mlp8_qin_scales, mlp8_fc1_qi8, mlp8_fc1_scales,
                    mlp8_fc1_out, mlp8_rows, mlp8_in, mlp8_hidden * 2);
    swiglu_ref(mlp8_fc1_out, mlp8_activated, mlp8_rows, mlp8_hidden);
    quantize_bf16_int8_rows_ref(mlp8_activated, mlp8_qact, mlp8_qact_scales,
                                mlp8_rows, mlp8_hidden, 1.0f);
    linear_int8_ref(mlp8_qact, mlp8_qact_scales, mlp8_fc2_qi8, mlp8_fc2_scales,
                    mlp8_ref, mlp8_rows, mlp8_hidden, mlp8_out);

    h3_gpu_tensor *mlp8_in_t = h3_gpu_tensor_from_bf16(
        gpu, mlp8_input_bf16, (size_t)mlp8_rows * mlp8_in);
    h3_gpu_tensor *mlp8_fc1_bf16_t = h3_gpu_tensor_from_bf16(
        gpu, mlp8_fc1_bf16, (size_t)mlp8_hidden * 2 * mlp8_in);
    h3_gpu_tensor *mlp8_fc2_bf16_t = h3_gpu_tensor_from_bf16(
        gpu, mlp8_fc2_bf16, (size_t)mlp8_out * mlp8_hidden);
    h3_gpu_tensor *mlp8_fc1_i8 =
        h3_gpu_tensor_new_i8(gpu, (size_t)mlp8_hidden * 2 * mlp8_in);
    h3_gpu_tensor *mlp8_fc1_scale_t =
        h3_gpu_tensor_new_f32(gpu, mlp8_hidden * 2);
    h3_gpu_tensor *mlp8_fc2_i8 =
        h3_gpu_tensor_new_i8(gpu, (size_t)mlp8_out * mlp8_hidden);
    h3_gpu_tensor *mlp8_fc2_scale_t = h3_gpu_tensor_new_f32(gpu, mlp8_out);
    h3_gpu_tensor *mlp8_act =
        h3_gpu_tensor_new_bf16(gpu, (size_t)mlp8_rows * mlp8_hidden);
    h3_gpu_tensor *mlp8_quant =
        h3_gpu_tensor_new_i8(gpu, (size_t)mlp8_padded * mlp8_hidden);
    h3_gpu_tensor *mlp8_act_scales = h3_gpu_tensor_new_f32(gpu, mlp8_padded);
    h3_gpu_tensor *mlp8_out_t =
        h3_gpu_tensor_new_bf16(gpu, (size_t)mlp8_rows * mlp8_out);
    check(mlp8_in_t && mlp8_fc1_bf16_t && mlp8_fc2_bf16_t && mlp8_fc1_i8 &&
              mlp8_fc1_scale_t && mlp8_fc2_i8 && mlp8_fc2_scale_t && mlp8_act &&
              mlp8_quant && mlp8_act_scales && mlp8_out_t,
          "mlp_int8 alloc");
    if (mlp8_in_t && mlp8_fc1_bf16_t && mlp8_fc2_bf16_t && mlp8_fc1_i8 &&
        mlp8_fc1_scale_t && mlp8_fc2_i8 && mlp8_fc2_scale_t && mlp8_act &&
        mlp8_quant && mlp8_act_scales && mlp8_out_t) {
        check(h3_gpu_quantize_weight_int8(gpu, mlp8_fc1_i8, mlp8_fc1_scale_t,
                                          mlp8_fc1_bf16_t, mlp8_hidden * 2,
                                          mlp8_in),
              "quantize mlp fc1");
        check(h3_gpu_quantize_weight_int8(gpu, mlp8_fc2_i8, mlp8_fc2_scale_t,
                                          mlp8_fc2_bf16_t, mlp8_out,
                                          mlp8_hidden),
              "quantize mlp fc2");
        check(h3_gpu_mlp_int8_bf16(
                  gpu, mlp8_out_t, mlp8_act, mlp8_quant, mlp8_act_scales,
                  mlp8_in_t, mlp8_fc1_i8, mlp8_fc1_scale_t, mlp8_fc2_i8,
                  mlp8_fc2_scale_t, mlp8_fc1_bf16_t, mlp8_fc2_bf16_t, mlp8_rows,
                  mlp8_in, mlp8_hidden, mlp8_out, 0, 0, 1, 0),
              "mlp_int8_bf16");
        check(h3_gpu_submit(gpu), "submit mlp_int8");
        check(h3_gpu_tensor_read_bf16(mlp8_out_t, mlp8_out_bf16,
                                      (size_t)mlp8_rows * mlp8_out),
              "read mlp_int8");
        for (size_t i = 0; i < (size_t)mlp8_rows * mlp8_out; i++) {
            float got = bf16_to_f32(mlp8_out_bf16[i]);
            if (fabsf(got - mlp8_ref[i]) >= 8e-2f) {
                fprintf(stderr,
                        "FAIL: mlp_int8 mismatch at %zu got=%f expected=%f\n",
                        i, got, mlp8_ref[i]);
                failures++;
                break;
            }
        }
    }

    /* The check above pins the INT8 MLP to an INT8 reference. This one asks a
     * different question at DiT widths: how far the INT8 MLP lands from the
     * BF16 MLP it replaces, which is what picking a default trades away. */
    {
        const uint32_t q_rows = 64;
        const uint32_t q_in = 5376;
        const uint32_t q_hidden = 14336;
        const uint32_t q_out = 5376;
        const uint32_t q_padded = (q_rows + 127u) & ~127u;
        size_t in_count = (size_t)q_rows * q_in;
        size_t fc1_count = (size_t)q_hidden * 2u * q_in;
        size_t fc2_count = (size_t)q_out * q_hidden;
        size_t out_count = (size_t)q_rows * q_out;
        uint16_t *q_input = malloc(in_count * sizeof(*q_input));
        uint16_t *q_fc1 = malloc(fc1_count * sizeof(*q_fc1));
        uint16_t *q_fc2 = malloc(fc2_count * sizeof(*q_fc2));
        uint16_t *q_got_int8 = malloc(out_count * sizeof(*q_got_int8));
        uint16_t *q_got_bf16 = malloc(out_count * sizeof(*q_got_bf16));
        check(q_input && q_fc1 && q_fc2 && q_got_int8 && q_got_bf16,
              "int8-vs-bf16 MLP host alloc");
        if (q_input && q_fc1 && q_fc2 && q_got_int8 && q_got_bf16) {
            /* AdaLN hands the MLP a normalized row, so unit-scale inputs and
             * fan-in scaled weights match the shape the DiT actually sees. */
            for (size_t i = 0; i < in_count; i++)
                q_input[i] = f32_to_bf16(sinf((float)i * 0.017f) +
                                         0.3f * cosf((float)i * 0.101f));
            for (size_t i = 0; i < fc1_count; i++)
                q_fc1[i] = f32_to_bf16(sinf((float)i * 0.0031f) * 0.02f);
            for (size_t i = 0; i < fc2_count; i++)
                q_fc2[i] = f32_to_bf16(cosf((float)i * 0.0027f) * 0.01f);

            h3_gpu_tensor *t_in = h3_gpu_tensor_from_bf16(gpu, q_input, in_count);
            h3_gpu_tensor *t_fc1 = h3_gpu_tensor_from_bf16(gpu, q_fc1, fc1_count);
            h3_gpu_tensor *t_fc2 = h3_gpu_tensor_from_bf16(gpu, q_fc2, fc2_count);
            h3_gpu_tensor *t_fc1_i8 = h3_gpu_tensor_new_i8(gpu, fc1_count);
            h3_gpu_tensor *t_fc1_scale =
                h3_gpu_tensor_new_f32(gpu, q_hidden * 2u);
            h3_gpu_tensor *t_fc2_i8 = h3_gpu_tensor_new_i8(gpu, fc2_count);
            h3_gpu_tensor *t_fc2_scale = h3_gpu_tensor_new_f32(gpu, q_out);
            h3_gpu_tensor *t_act =
                h3_gpu_tensor_new_bf16(gpu, (size_t)q_rows * q_hidden);
            h3_gpu_tensor *t_quant =
                h3_gpu_tensor_new_i8(gpu, (size_t)q_padded * q_hidden);
            h3_gpu_tensor *t_scales = h3_gpu_tensor_new_f32(gpu, q_padded);
            h3_gpu_tensor *t_out_int8 = h3_gpu_tensor_new_bf16(gpu, out_count);
            h3_gpu_tensor *t_out_bf16 = h3_gpu_tensor_new_bf16(gpu, out_count);
            check(t_in && t_fc1 && t_fc2 && t_fc1_i8 && t_fc1_scale &&
                      t_fc2_i8 && t_fc2_scale && t_act && t_quant && t_scales &&
                      t_out_int8 && t_out_bf16,
                  "int8-vs-bf16 MLP device alloc");
            if (t_in && t_fc1 && t_fc2 && t_fc1_i8 && t_fc1_scale && t_fc2_i8 &&
                t_fc2_scale && t_act && t_quant && t_scales && t_out_int8 &&
                t_out_bf16) {
                check(h3_gpu_quantize_weight_int8(gpu, t_fc1_i8, t_fc1_scale,
                                                  t_fc1, q_hidden * 2u, q_in),
                      "quantize wide fc1");
                check(h3_gpu_quantize_weight_int8(gpu, t_fc2_i8, t_fc2_scale,
                                                  t_fc2, q_out, q_hidden),
                      "quantize wide fc2");
                check(h3_gpu_mlp_int8_bf16(
                          gpu, t_out_int8, t_act, t_quant, t_scales, t_in,
                          t_fc1_i8, t_fc1_scale, t_fc2_i8, t_fc2_scale, t_fc1,
                          t_fc2, q_rows, q_in, q_hidden, q_out, 0, 0, 1, 0),
                      "wide mlp_int8_bf16");
                check(h3_gpu_mlp_bf16(gpu, t_out_bf16, t_in, t_fc1, t_fc2,
                                      q_rows, q_in, q_hidden, q_out),
                      "wide mlp_bf16");
                check(h3_gpu_submit(gpu), "submit wide MLP pair");
                check(h3_gpu_tensor_read_bf16(t_out_int8, q_got_int8,
                                              out_count),
                      "read wide mlp_int8");
                check(h3_gpu_tensor_read_bf16(t_out_bf16, q_got_bf16,
                                              out_count),
                      "read wide mlp_bf16");
                double err = 0.0;
                double norm = 0.0;
                for (size_t i = 0; i < out_count; i++) {
                    double a = bf16_to_f32(q_got_int8[i]);
                    double b = bf16_to_f32(q_got_bf16[i]);
                    err += (a - b) * (a - b);
                    norm += b * b;
                }
                double rel = norm > 0.0 ? sqrt(err / norm) : 1.0;
                printf("int8 MLP vs BF16 MLP relL2 at DiT width: %.5f\n", rel);
                if (!(rel < 3e-2)) {
                    fprintf(stderr,
                            "FAIL: int8 MLP relL2 %.5f is too far from BF16\n",
                            rel);
                    failures++;
                }
            }
            h3_gpu_tensor_free(t_out_bf16);
            h3_gpu_tensor_free(t_out_int8);
            h3_gpu_tensor_free(t_scales);
            h3_gpu_tensor_free(t_quant);
            h3_gpu_tensor_free(t_act);
            h3_gpu_tensor_free(t_fc2_scale);
            h3_gpu_tensor_free(t_fc2_i8);
            h3_gpu_tensor_free(t_fc1_scale);
            h3_gpu_tensor_free(t_fc1_i8);
            h3_gpu_tensor_free(t_fc2);
            h3_gpu_tensor_free(t_fc1);
            h3_gpu_tensor_free(t_in);
        }
        free(q_got_bf16);
        free(q_got_int8);
        free(q_fc2);
        free(q_fc1);
        free(q_input);
    }

    const uint32_t gaq_rows = 4;
    const uint32_t gaq_width = 32;
    const uint32_t gaq_slots = 6;
    const uint32_t gaq_padded = (gaq_rows + 127u) & ~127u;
    float gaq_res_host[gaq_rows * gaq_width];
    float gaq_branch_host[gaq_rows * gaq_width];
    float gaq_norm_host[gaq_width];
    float gaq_mod_host[gaq_slots * gaq_width];
    uint32_t gaq_map[gaq_rows];
    uint16_t gaq_res_bf16[gaq_rows * gaq_width];
    uint16_t gaq_branch_bf16[gaq_rows * gaq_width];
    uint16_t gaq_norm_bf16[gaq_width];
    uint16_t gaq_mod_bf16[gaq_slots * gaq_width];
    float gaq_gate_ref[gaq_rows * gaq_width];
    float gaq_adaln_ref[gaq_rows * gaq_width];
    int8_t gaq_qi8[gaq_rows * gaq_width];
    float gaq_scales_ref[gaq_rows];
    uint16_t gaq_gate_out_bf16[gaq_rows * gaq_width];
    for (uint32_t i = 0; i < gaq_rows; i++) gaq_map[i] = 0;
    for (size_t i = 0; i < (size_t)gaq_rows * gaq_width; i++) {
        gaq_res_host[i] = sinf((float)i * 0.08f);
        gaq_branch_host[i] = cosf((float)i * 0.06f) * 0.5f;
        gaq_res_bf16[i] = f32_to_bf16(gaq_res_host[i]);
        gaq_branch_bf16[i] = f32_to_bf16(gaq_branch_host[i]);
        gaq_res_host[i] = bf16_to_f32(gaq_res_bf16[i]);
        gaq_branch_host[i] = bf16_to_f32(gaq_branch_bf16[i]);
    }
    for (uint32_t i = 0; i < gaq_width; i++) {
        gaq_norm_host[i] = 0.75f + 0.01f * (float)i;
        gaq_norm_bf16[i] = f32_to_bf16(gaq_norm_host[i]);
        gaq_norm_host[i] = bf16_to_f32(gaq_norm_bf16[i]);
    }
    for (size_t i = 0; i < (size_t)gaq_slots * gaq_width; i++) {
        gaq_mod_host[i] = sinf((float)i * 0.03f) * 0.2f;
        gaq_mod_bf16[i] = f32_to_bf16(gaq_mod_host[i]);
        gaq_mod_host[i] = bf16_to_f32(gaq_mod_bf16[i]);
    }
    gate_ref(gaq_res_host, gaq_branch_host, gaq_mod_host, gaq_map, gaq_gate_ref,
             gaq_rows, gaq_width, gaq_slots, 0);
    adaln_ref(gaq_gate_ref, gaq_norm_host, gaq_mod_host, gaq_map, gaq_adaln_ref,
              gaq_rows, gaq_width, gaq_slots, 1, 2, 1e-5f);
    quantize_bf16_int8_rows_ref(gaq_adaln_ref, gaq_qi8, gaq_scales_ref, gaq_rows,
                                gaq_width, 1.0f);

    h3_gpu_tensor *gaq_res =
        h3_gpu_tensor_from_bf16(gpu, gaq_res_bf16, (size_t)gaq_rows * gaq_width);
    h3_gpu_tensor *gaq_branch = h3_gpu_tensor_from_bf16(
        gpu, gaq_branch_bf16, (size_t)gaq_rows * gaq_width);
    h3_gpu_tensor *gaq_norm =
        h3_gpu_tensor_from_bf16(gpu, gaq_norm_bf16, gaq_width);
    h3_gpu_tensor *gaq_mod = h3_gpu_tensor_from_bf16(
        gpu, gaq_mod_bf16, (size_t)gaq_slots * gaq_width);
    h3_gpu_tensor *gaq_map_t =
        h3_gpu_tensor_from_u32(gpu, gaq_map, gaq_rows);
    h3_gpu_tensor *gaq_gate =
        h3_gpu_tensor_new_bf16(gpu, (size_t)gaq_rows * gaq_width);
    h3_gpu_tensor *gaq_quant =
        h3_gpu_tensor_new_i8(gpu, (size_t)gaq_padded * gaq_width);
    h3_gpu_tensor *gaq_scales = h3_gpu_tensor_new_f32(gpu, gaq_padded);
    check(gaq_res && gaq_branch && gaq_norm && gaq_mod && gaq_map_t && gaq_gate &&
              gaq_quant && gaq_scales,
          "gate_adaln_quantize alloc");
    if (gaq_res && gaq_branch && gaq_norm && gaq_mod && gaq_map_t && gaq_gate &&
        gaq_quant && gaq_scales) {
        check(h3_gpu_gate_adaln_quantize_int8(
                  gpu, gaq_gate, gaq_quant, gaq_scales, gaq_res, gaq_branch,
                  gaq_norm, gaq_mod, gaq_mod, gaq_map_t, gaq_rows, gaq_padded,
                  gaq_width, gaq_slots, 0, 1, 2, 1e-5f),
              "gate_adaln_quantize_int8");
        check(h3_gpu_submit(gpu), "submit gate_adaln_quantize");
        check(h3_gpu_tensor_read_bf16(gaq_gate, gaq_gate_out_bf16,
                                      (size_t)gaq_rows * gaq_width),
              "read gated residual");
        float gaq_scales_got[gaq_rows];
        check(h3_gpu_tensor_read_f32(gaq_scales, gaq_scales_got, gaq_rows),
              "read quant scales");
        for (size_t i = 0; i < (size_t)gaq_rows * gaq_width; i++) {
            float got = bf16_to_f32(gaq_gate_out_bf16[i]);
            if (fabsf(got - gaq_gate_ref[i]) >= 5e-2f) {
                fprintf(stderr,
                        "FAIL: gate_adaln_quantize gate mismatch at %zu\n", i);
                failures++;
                break;
            }
        }
        for (uint32_t i = 0; i < gaq_rows; i++) {
            if (fabsf(gaq_scales_got[i] - gaq_scales_ref[i]) >= 1e-4f) {
                fprintf(stderr,
                        "FAIL: gate_adaln_quantize scale mismatch at %u got=%f expected=%f\n",
                        i, gaq_scales_got[i], gaq_scales_ref[i]);
                failures++;
                break;
            }
        }
    }

    const uint32_t patch_rows = 1;
    const uint32_t patch_in_dim = 32;
    const uint32_t patch_out_dim = 5376;
    const size_t patch_in_count = (size_t)patch_rows * patch_in_dim;
    const size_t patch_w_count = (size_t)patch_out_dim * patch_in_dim;
    const size_t patch_out_count = (size_t)patch_rows * patch_out_dim;
    float *patch_in_host = (float *)malloc(patch_in_count * sizeof(float));
    float *patch_w_host = (float *)malloc(patch_w_count * sizeof(float));
    float *patch_b_host = (float *)malloc(patch_out_dim * sizeof(float));
    float *patch_f32_host = (float *)malloc(patch_out_count * sizeof(float));
    uint16_t *patch_ref_bf16 = (uint16_t *)malloc(patch_out_count * sizeof(uint16_t));
    uint16_t *patch_out_bf16 = (uint16_t *)malloc(patch_out_count * sizeof(uint16_t));
    if (patch_in_host && patch_w_host && patch_b_host && patch_f32_host &&
        patch_ref_bf16 && patch_out_bf16) {
        for (size_t i = 0; i < patch_in_count; i++)
            patch_in_host[i] = sinf((float)i * 0.11f);
        for (size_t i = 0; i < patch_w_count; i++)
            patch_w_host[i] = cosf((float)i * 0.07f) * 0.01f;
        for (size_t i = 0; i < patch_out_dim; i++)
            patch_b_host[i] = sinf((float)i * 0.03f) * 0.001f;
        linear_ref(patch_in_host, patch_w_host, patch_b_host, patch_f32_host,
                   patch_rows, patch_in_dim, patch_out_dim);
        for (size_t i = 0; i < patch_out_count; i++)
            patch_ref_bf16[i] = f32_to_bf16(patch_f32_host[i]);
        h3_gpu_tensor *patch_in_t =
            h3_gpu_tensor_from_f32(gpu, patch_in_host, patch_in_count);
        h3_gpu_tensor *patch_w =
            h3_gpu_tensor_from_f32(gpu, patch_w_host, patch_w_count);
        h3_gpu_tensor *patch_b =
            h3_gpu_tensor_from_f32(gpu, patch_b_host, patch_out_dim);
        h3_gpu_tensor *patch_out_t =
            h3_gpu_tensor_new_bf16(gpu, patch_out_count);
        if (patch_in_t && patch_w && patch_b && patch_out_t) {
            check(h3_gpu_patch_linear_bf16(
                      gpu, patch_out_t, patch_in_t, patch_w, patch_b,
                      patch_rows, patch_in_dim, patch_out_dim),
                  "patch_linear");
            check(h3_gpu_submit(gpu), "submit patch_linear");
            check(h3_gpu_tensor_read_bf16(patch_out_t, patch_out_bf16,
                                          patch_out_count),
                  "read patch_linear");
            for (size_t i = 0; i < patch_out_count; i++) {
                float got = bf16_to_f32(patch_out_bf16[i]);
                float want = bf16_to_f32(patch_ref_bf16[i]);
                if (fabsf(got - want) >= 5e-2f) {
                    fprintf(stderr,
                            "FAIL: patch_linear mismatch at %zu got=%f expected=%f\n",
                            i, got, want);
                    failures++;
                    break;
                }
            }
        }
        h3_gpu_tensor_free(patch_in_t);
        h3_gpu_tensor_free(patch_w);
        h3_gpu_tensor_free(patch_b);
        h3_gpu_tensor_free(patch_out_t);
    }
    free(patch_in_host);
    free(patch_w_host);
    free(patch_b_host);
    free(patch_f32_host);
    free(patch_ref_bf16);
    free(patch_out_bf16);

    h3_gpu_tensor_free(silu_in);
    h3_gpu_tensor_free(silu_out);
    h3_gpu_tensor_free(norm_in);
    h3_gpu_tensor_free(norm_weight);
    h3_gpu_tensor_free(norm_out);
    h3_gpu_tensor_free(linear_in);
    h3_gpu_tensor_free(linear_weight);
    h3_gpu_tensor_free(linear_bias);
    h3_gpu_tensor_free(linear_out);
    h3_gpu_tensor_free(linear_out_bias);
    h3_gpu_tensor_free(adaln_in);
    h3_gpu_tensor_free(adaln_norm);
    h3_gpu_tensor_free(adaln_mod);
    h3_gpu_tensor_free(adaln_row_map);
    h3_gpu_tensor_free(adaln_out);
    h3_gpu_tensor_free(gate_residual);
    h3_gpu_tensor_free(gate_branch);
    h3_gpu_tensor_free(gate_out);
    h3_gpu_tensor_free(swiglu_fused);
    h3_gpu_tensor_free(swiglu_out);
    h3_gpu_tensor_free(mlp_in);
    h3_gpu_tensor_free(mlp_fc1);
    h3_gpu_tensor_free(mlp_fc2);
    h3_gpu_tensor_free(mlp_output);
    h3_gpu_tensor_free(gelu_in);
    h3_gpu_tensor_free(gelu_out);
    h3_gpu_tensor_free(embed_weight);
    h3_gpu_tensor_free(embed_ids);
    h3_gpu_tensor_free(embed_out);
    h3_gpu_tensor_free(f32_silu_in);
    h3_gpu_tensor_free(f32_silu_out_t);
    h3_gpu_tensor_free(f32_linear_in_t);
    h3_gpu_tensor_free(f32_linear_weight_t);
    h3_gpu_tensor_free(f32_linear_bias_t);
    h3_gpu_tensor_free(f32_linear_out_t);
    h3_gpu_tensor_free(f32_linear_out_bias_t);
    h3_gpu_tensor_free(qkv_in);
    h3_gpu_tensor_free(qkv_q_norm);
    h3_gpu_tensor_free(qkv_k_norm);
    h3_gpu_tensor_free(qkv_rope_cos);
    h3_gpu_tensor_free(qkv_rope_sin);
    h3_gpu_tensor_free(qkv_query);
    h3_gpu_tensor_free(qkv_key);
    h3_gpu_tensor_free(qkv_value);
    h3_gpu_tensor_free(sdpa_query);
    h3_gpu_tensor_free(sdpa_key);
    h3_gpu_tensor_free(sdpa_value);
    h3_gpu_tensor_free(sdpa_out);
    h3_gpu_tensor_free(head_norm_tensor);
    h3_gpu_tensor_free(head_norm_weight);
    h3_gpu_tensor_free(text_rope_q);
    h3_gpu_tensor_free(text_rope_k);
    h3_gpu_tensor_free(text_rope_cos);
    h3_gpu_tensor_free(text_rope_sin);
    h3_gpu_tensor_free(gqa_q);
    h3_gpu_tensor_free(gqa_k);
    h3_gpu_tensor_free(gqa_v);
    h3_gpu_tensor_free(gqa_out);
    h3_gpu_tensor_free(silu_mul_gate);
    h3_gpu_tensor_free(silu_mul_up);
    h3_gpu_tensor_free(silu_mul_out);
    h3_gpu_tensor_free(tp_input);
    h3_gpu_tensor_free(tp_pairs_t);
    h3_gpu_tensor_free(tp_baseline_idx);
    h3_gpu_tensor_free(tp_pooled);
    h3_gpu_tensor_free(tp_original);
    h3_gpu_tensor_free(tp_processed);
    h3_gpu_tensor_free(tp_parents_t);
    h3_gpu_tensor_free(tp_expanded);
    h3_gpu_tensor_free(ga_residual);
    h3_gpu_tensor_free(ga_branch);
    h3_gpu_tensor_free(ga_norm);
    h3_gpu_tensor_free(ga_mod);
    h3_gpu_tensor_free(ga_map);
    h3_gpu_tensor_free(ga_gate_res);
    h3_gpu_tensor_free(ga_out);
    h3_gpu_tensor_free(ga_head_weight);
    h3_gpu_tensor_free(ga_head_bias);
    h3_gpu_tensor_free(ga_inverse);
    h3_gpu_tensor_free(ga_head_out);
    h3_gpu_tensor_free(iq_weight_bf16_t);
    h3_gpu_tensor_free(iq_weight_i8);
    h3_gpu_tensor_free(iq_weight_scale_t);
    h3_gpu_tensor_free(iq_input_t);
    h3_gpu_tensor_free(iq_quant_t);
    h3_gpu_tensor_free(iq_input_scale_t);
    h3_gpu_tensor_free(iq_out_t);
    h3_gpu_tensor_free(hm_input_t);
    h3_gpu_tensor_free(hm_weight_bf16_t);
    h3_gpu_tensor_free(hm_weight_i8);
    h3_gpu_tensor_free(hm_weight_scale_t);
    h3_gpu_tensor_free(hm_quant_t);
    h3_gpu_tensor_free(hm_input_scale_t);
    h3_gpu_tensor_free(hm_out_t);
    h3_gpu_tensor_free(mlp8_in_t);
    h3_gpu_tensor_free(mlp8_fc1_bf16_t);
    h3_gpu_tensor_free(mlp8_fc2_bf16_t);
    h3_gpu_tensor_free(mlp8_fc1_i8);
    h3_gpu_tensor_free(mlp8_fc1_scale_t);
    h3_gpu_tensor_free(mlp8_fc2_i8);
    h3_gpu_tensor_free(mlp8_fc2_scale_t);
    h3_gpu_tensor_free(mlp8_act);
    h3_gpu_tensor_free(mlp8_quant);
    h3_gpu_tensor_free(mlp8_act_scales);
    h3_gpu_tensor_free(mlp8_out_t);
    h3_gpu_tensor_free(gaq_res);
    h3_gpu_tensor_free(gaq_branch);
    h3_gpu_tensor_free(gaq_norm);
    h3_gpu_tensor_free(gaq_mod);
    h3_gpu_tensor_free(gaq_map_t);
    h3_gpu_tensor_free(gaq_gate);
    h3_gpu_tensor_free(gaq_quant);
    h3_gpu_tensor_free(gaq_scales);

    check_concurrent_staged_reads(gpu);

    h3_gpu_free(gpu);

    if (failures) {
        fprintf(stderr, "h3_cuda_ops: %d failure(s)\n", failures);
        return 1;
    }
    puts("ok: CUDA op tests passed");
    return 0;
}
