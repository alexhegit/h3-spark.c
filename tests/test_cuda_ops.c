#include "h3_gpu.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    enum { GA_ROWS = 6, GA_WIDTH = 4, GA_SLOTS = 2, GA_HEAD = 3 };
    float ga_input_host[GA_ROWS * GA_WIDTH];
    float ga_branch_host[GA_ROWS * GA_WIDTH];
    float ga_norm_host[GA_WIDTH];
    float ga_mod_host[GA_SLOTS * GA_WIDTH];
    uint32_t ga_row_map[GA_ROWS] = {0, 1, 0, 1, 0, 1};
    uint16_t ga_input_bf16[GA_ROWS * GA_WIDTH];
    uint16_t ga_branch_bf16[GA_ROWS * GA_WIDTH];
    uint16_t ga_norm_bf16[GA_WIDTH];
    uint16_t ga_mod_bf16[GA_SLOTS * GA_WIDTH];
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
    for (size_t i = 0; i < (size_t)GA_SLOTS * GA_WIDTH; i++) {
        ga_mod_host[i] = sinf((float)i * 0.31f) * 0.25f;
        ga_mod_bf16[i] = f32_to_bf16(ga_mod_host[i]);
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
        h3_gpu_tensor_from_bf16(gpu, ga_mod_bf16, GA_SLOTS * GA_WIDTH);
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
    h3_gpu_free(gpu);

    if (failures) {
        fprintf(stderr, "h3_cuda_ops: %d failure(s)\n", failures);
        return 1;
    }
    puts("ok: CUDA op tests passed");
    return 0;
}
