#include "h3_gpu.h"

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
    h3_gpu_free(gpu);

    if (failures) {
        fprintf(stderr, "h3_cuda_ops: %d failure(s)\n", failures);
        return 1;
    }
    puts("ok: CUDA op tests passed");
    return 0;
}
