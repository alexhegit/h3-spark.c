/* What the tensor cores actually deliver on this part, at the DiT's own GEMM
 * shapes, so precision decisions rest on measurements instead of spec sheets.
 *
 * D = A^T B with A column-major K x M and B column-major K x N, which is the
 * only layout the narrow-precision paths accept. */

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                  \
                    cudaGetErrorString(status));                               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

typedef struct {
    const char *name;
    int m;
    int n;
    int k;
} gemm_shape;

typedef struct {
    const char *name;
    cudaDataType input;
    cudaDataType output;
    cublasComputeType_t compute;
    cudaDataType scale;
    int block_scaled;
    /* Ask cuBLASLt to fold the per-row weight and activation scales into the
     * epilogue, which is what would let the DiT drop its separate accumulator
     * pass. */
    int outer_vector_scaled;
    /* Charge the GEMM for the separate scale-application pass the INT8 path
     * runs today, so the comparison is against real elapsed time. */
    int apply_pass;
    /* MX-style scaling: one UE8M0 per 32 elements of K. For E4M3 this is the
     * only granularity finer than per-token, and activations are where the FP8
     * error actually comes from. */
    int vec32_scaled;
} gemm_precision;

/* Stand-in for h3_int8_apply_scales_bf16_kernel: same traffic, same shape. */
__global__ static void apply_scales(const int32_t *accum, const float *rows,
                                    const float *columns, __nv_bfloat16 *out,
                                    long long count, int width) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = (float)accum[index] * rows[index / width] *
                  columns[index % width];
    out[index] = __float2bfloat16(value);
}

static size_t element_bytes(cudaDataType type) {
    switch (type) {
        case CUDA_R_32F: return 4;
        case CUDA_R_16F:
        case CUDA_R_16BF: return 2;
        case CUDA_R_8I:
        case CUDA_R_8F_E4M3: return 1;
        case CUDA_R_32I: return 4;
        default: return 0; /* FP4 is packed two per byte and handled apart. */
    }
}

static void *device_fill(size_t bytes) {
    void *memory = NULL;
    CHECK(cudaMalloc(&memory, bytes ? bytes : 1));
    CHECK(cudaMemset(memory, 0x11, bytes));
    return memory;
}

/* Returns TFLOP/s, or 0 when cuBLASLt has no algorithm for the combination. */
static double bench(cublasLtHandle_t lt, const gemm_shape *shape,
                    const gemm_precision *precision, void *workspace,
                    size_t workspace_bytes, const char **note,
                    double *elapsed_ms) {
    int m = shape->m, n = shape->n, k = shape->k;
    size_t a_bytes, b_bytes;
    if (precision->input == CUDA_R_4F_E2M1) {
        a_bytes = (size_t)k * m / 2;
        b_bytes = (size_t)k * n / 2;
    } else {
        a_bytes = (size_t)k * m * element_bytes(precision->input);
        b_bytes = (size_t)k * n * element_bytes(precision->input);
    }
    size_t d_bytes = (size_t)m * n * element_bytes(precision->output);
    void *a = device_fill(a_bytes);
    void *b = device_fill(b_bytes);
    void *d = device_fill(d_bytes);
    /* Block scaling wants one UE4M3 per 16 elements of K; outer-vector scaling
     * wants one FP32 per output row and per token. */
    size_t scale_group = precision->vec32_scaled ? 32 : 16;
    size_t a_scale_bytes =
        precision->outer_vector_scaled
            ? (size_t)m * sizeof(float)
            : (size_t)(((size_t)k + scale_group - 1) / scale_group) * m;
    size_t b_scale_bytes =
        precision->outer_vector_scaled
            ? (size_t)n * sizeof(float)
            : (size_t)(((size_t)k + scale_group - 1) / scale_group) * n;
    int scaled = precision->block_scaled || precision->outer_vector_scaled ||
                 precision->vec32_scaled;
    void *a_scale = scaled ? device_fill(a_scale_bytes) : NULL;
    void *b_scale = scaled ? device_fill(b_scale_bytes) : NULL;
    void *epilogue_out =
        precision->apply_pass ? device_fill((size_t)m * n * 2) : NULL;
    void *channel_scale =
        precision->apply_pass ? device_fill((size_t)m * sizeof(float)) : NULL;
    void *token_scale =
        precision->apply_pass ? device_fill((size_t)n * sizeof(float)) : NULL;

    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t layout_a = NULL, layout_b = NULL, layout_d = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasOperation_t transpose = CUBLAS_OP_T, plain = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    double tflops = 0.0;
    int found = 0;
    float alpha = 1.0f, beta = 0.0f;
    int32_t alpha_i = 1, beta_i = 0;
    const void *alpha_pointer = precision->scale == CUDA_R_32I
                                    ? (const void *)&alpha_i
                                    : (const void *)&alpha;
    const void *beta_pointer = precision->scale == CUDA_R_32I
                                   ? (const void *)&beta_i
                                   : (const void *)&beta;
    const int iterations = 30;
    cudaEvent_t start = NULL, stop = NULL;
    float milliseconds = 0.0f;
    double seconds = 0.0;
    cublasStatus_t status =
        cublasLtMatmulDescCreate(&operation, precision->compute,
                                 precision->scale);
    if (status != CUBLAS_STATUS_SUCCESS) {
        *note = "desc";
        goto done;
    }
    cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &transpose, sizeof(transpose));
    cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &plain, sizeof(plain));
    if (precision->outer_vector_scaled)
        scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    if (precision->vec32_scaled)
        scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    if (scaled) {
        if (cublasLtMatmulDescSetAttribute(
                operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode,
                sizeof(scale_mode)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(
                operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode,
                sizeof(scale_mode)) != CUBLAS_STATUS_SUCCESS) {
            *note = "scale mode";
            goto done;
        }
        cublasLtMatmulDescSetAttribute(operation,
                                       CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                       &a_scale, sizeof(a_scale));
        cublasLtMatmulDescSetAttribute(operation,
                                       CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                       &b_scale, sizeof(b_scale));
    }
    if (cublasLtMatrixLayoutCreate(&layout_a, precision->input, k, m, k) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&layout_b, precision->input, k, n, k) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&layout_d, precision->output, m, n, m) !=
            CUBLAS_STATUS_SUCCESS) {
        *note = "layout";
        goto done;
    }
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
        sizeof(workspace_bytes));
    status = cublasLtMatmulAlgoGetHeuristic(lt, operation, layout_a, layout_b,
                                            layout_d, layout_d, preference, 1,
                                            &heuristic, &found);
    if (status != CUBLAS_STATUS_SUCCESS || !found) {
        *note = "no algorithm";
        goto done;
    }
    for (int warm = 0; warm < 3; warm++) {
        status = cublasLtMatmul(lt, operation, alpha_pointer, a, layout_a, b,
                                layout_b, beta_pointer, d, layout_d, d,
                                layout_d, &heuristic.algo, workspace,
                                workspace_bytes, 0);
        if (status != CUBLAS_STATUS_SUCCESS) {
            *note = "matmul";
            goto done;
        }
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; iteration++) {
        cublasLtMatmul(lt, operation, alpha_pointer, a, layout_a, b, layout_b,
                       beta_pointer, d, layout_d, d, layout_d, &heuristic.algo,
                       workspace, workspace_bytes, 0);
        if (precision->apply_pass) {
            long long count = (long long)m * n;
            apply_scales<<<(unsigned)((count + 255) / 256), 256>>>(
                (const int32_t *)d, (const float *)token_scale,
                (const float *)channel_scale, (__nv_bfloat16 *)epilogue_out,
                count, m);
        }
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    seconds = (double)milliseconds / 1000.0 / iterations;
    tflops = 2.0 * m * n * k / seconds / 1e12;
    *elapsed_ms = seconds * 1000.0;
    *note = NULL;

done:
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (layout_a) cublasLtMatrixLayoutDestroy(layout_a);
    if (layout_b) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_d) cublasLtMatrixLayoutDestroy(layout_d);
    if (operation) cublasLtMatmulDescDestroy(operation);
    cudaFree(a);
    cudaFree(b);
    cudaFree(d);
    cudaFree(a_scale);
    cudaFree(b_scale);
    cudaFree(epilogue_out);
    cudaFree(channel_scale);
    cudaFree(token_scale);
    return tflops;
}

int main(void) {
    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties, 0));
    printf("device %s sm_%d%d\n", properties.name, properties.major,
           properties.minor);

    cublasLtHandle_t lt = NULL;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasLtCreate failed\n");
        return 1;
    }
    size_t workspace_bytes = (size_t)128 << 20;
    void *workspace = NULL;
    CHECK(cudaMalloc(&workspace, workspace_bytes));

    /* The DiT's four per-block GEMMs, in the orientation h3_gpu_linear uses:
     * m is the output width and n is the token count, so the token count never
     * lands on a leading dimension. */
    const gemm_shape shapes[] = {
        {"qkv      out=21504 tokens=1870 k=5376 ", 21504, 1870, 5376},
        {"attn-out out=5376  tokens=1870 k=7168 ", 5376, 1870, 7168},
        {"fc1      out=28672 tokens=1870 k=5376 ", 28672, 1870, 5376},
        {"fc2      out=5376  tokens=1870 k=14336", 5376, 1870, 14336},
    };
    const gemm_precision precisions[] = {
        {"bf16", CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, 0, 0,
         0},
        {"int8", CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I, 0, 0,
         0},
        /* What the DiT actually pays today. */
        {"int8+apply", CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I, 0,
         0, 1},
        {"fp8e4m3", CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         0, 0, 0},
        /* Narrow output for the same INT8 math: isolates what the int32
         * accumulator costs in write bandwidth alone. */
        {"int8 D=i8", CUDA_R_8I, CUDA_R_8I, CUBLAS_COMPUTE_32I, CUDA_R_32F, 0,
         0, 0},
        {"nvfp4", CUDA_R_4F_E2M1, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         1, 0, 0},
        /* Same 8-bit operands, but a scale every 32 values of K instead of one
         * per token. */
        {"mxfp8", CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         0, 0, 0, 1},
        {"mxfp4", CUDA_R_4F_E2M1, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         0, 0, 0, 1},
    };

    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        printf("\n%s\n", shapes[s].name);
        for (size_t p = 0; p < sizeof(precisions) / sizeof(*precisions); p++) {
            const char *note = NULL;
            double elapsed_ms = 0.0;
            double tflops =
                bench(lt, &shapes[s], &precisions[p], workspace,
                      workspace_bytes, &note, &elapsed_ms);
            if (tflops > 0.0)
                printf("  %-10s %8.1f TFLOP/s   %6.3f ms\n", precisions[p].name,
                       tflops, elapsed_ms);
            else
                printf("  %-10s unsupported (%s)\n", precisions[p].name,
                       note ? note : "unknown");
        }
    }
    cudaFree(workspace);
    cublasLtDestroy(lt);
    return 0;
}
