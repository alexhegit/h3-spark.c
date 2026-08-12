extern "C" {
#include "h3_gpu.h"
#include "h3_gpu_cuda_internal.h"
}

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct h3_gpu_tensor {
    h3_gpu *owner;
    void *device;
    size_t elements;
    size_t bytes;
    h3_gpu_dtype dtype;
};

struct h3_gpu {
    cudaStream_t stream;
    cublasHandle_t cublas;
    int device;
    int fast_path;
    int tensor_fast_path;
    char error[512];
    char profile_label[128];
    h3_gpu_stats stats;
};

static inline __device__ __host__ float h3_bf16_bits_to_f32(uint16_t bits) {
    union {
        uint32_t word;
        float as_float;
    } converted;
    converted.word = (uint32_t)bits << 16u;
    return converted.as_float;
}

static inline __device__ __host__ uint16_t h3_f32_to_bf16_bits(float input) {
    union {
        uint32_t word;
        float as_float;
    } converted;
    converted.as_float = input;
    converted.word += 0x7fffu + ((converted.word >> 16u) & 1u);
    return (uint16_t)(converted.word >> 16u);
}

extern "C" {

void h3_gpu_cuda_set_error(h3_gpu *gpu, const char *operation) {
    if (!gpu || !operation) return;
    snprintf(gpu->error, sizeof(gpu->error), "%s is not implemented on CUDA yet",
             operation);
}

const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu ? gpu->error : "invalid GPU context";
}

static int h3_gpu_fail(h3_gpu *gpu, const char *format, ...) {
    if (!gpu) return 0;
    va_list args;
    va_start(args, format);
    vsnprintf(gpu->error, sizeof(gpu->error), format, args);
    va_end(args);
    return 0;
}

static size_t h3_dtype_bytes(h3_gpu_dtype dtype) {
    switch (dtype) {
    case H3_GPU_F32: return sizeof(float);
    case H3_GPU_BF16: return sizeof(uint16_t);
    case H3_GPU_I8: return sizeof(int8_t);
    case H3_GPU_U32: return sizeof(uint32_t);
    default: return 0;
    }
}

static int h3_cuda_check(h3_gpu *gpu, cudaError_t status, const char *where) {
    if (status == cudaSuccess) return 1;
    return h3_gpu_fail(gpu, "%s: %s", where, cudaGetErrorString(status));
}

static int h3_cublas_check(h3_gpu *gpu, cublasStatus_t status,
                           const char *where) {
    if (status == CUBLAS_STATUS_SUCCESS) return 1;
    return h3_gpu_fail(gpu, "%s: cuBLAS error %d", where, (int)status);
}

static h3_gpu_tensor *h3_tensor_alloc(h3_gpu *gpu, size_t elements,
                                      h3_gpu_dtype dtype) {
    if (!gpu || !elements) {
        if (gpu) h3_gpu_fail(gpu, "invalid tensor allocation request");
        return NULL;
    }
    size_t bytes = elements * h3_dtype_bytes(dtype);
    if (bytes / h3_dtype_bytes(dtype) != elements) {
        h3_gpu_fail(gpu, "tensor allocation overflow");
        return NULL;
    }
    h3_gpu_tensor *tensor = (h3_gpu_tensor *)calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3_gpu_fail(gpu, "out of memory allocating tensor handle");
        return NULL;
    }
    cudaError_t status = cudaMalloc(&tensor->device, bytes);
    if (status != cudaSuccess) {
        free(tensor);
        h3_gpu_fail(gpu, "cudaMalloc failed: %s", cudaGetErrorString(status));
        return NULL;
    }
    tensor->owner = gpu;
    tensor->elements = elements;
    tensor->bytes = bytes;
    tensor->dtype = dtype;
    gpu->stats.tensor_allocations++;
    gpu->stats.allocated_bytes += bytes;
    gpu->stats.live_bytes += bytes;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    gpu->error[0] = '\0';
    return tensor;
}

h3_gpu *h3_gpu_create(const char *shader_source_path, char *error,
                      size_t error_size) {
    (void)shader_source_path;
    h3_gpu *gpu = (h3_gpu *)calloc(1, sizeof(*gpu));
    if (!gpu) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return NULL;
    }
    cudaError_t status = cudaSetDevice(0);
    if (status != cudaSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cudaSetDevice failed: %s",
                     cudaGetErrorString(status));
        }
        free(gpu);
        return NULL;
    }
    gpu->device = 0;
    status = cudaStreamCreate(&gpu->stream);
    if (status != cudaSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cudaStreamCreate failed: %s",
                     cudaGetErrorString(status));
        }
        free(gpu);
        return NULL;
    }
    if (cublasCreate(&gpu->cublas) != CUBLAS_STATUS_SUCCESS) {
        if (error && error_size) snprintf(error, error_size, "cublasCreate failed");
        cudaStreamDestroy(gpu->stream);
        free(gpu);
        return NULL;
    }
    cublasSetStream(gpu->cublas, gpu->stream);
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess) {
        gpu->fast_path = props.major >= 12 ? 1 : 0;
        gpu->tensor_fast_path = gpu->fast_path;
    }
    gpu->error[0] = '\0';
    return gpu;
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    if (gpu->cublas) cublasDestroy(gpu->cublas);
    if (gpu->stream) cudaStreamDestroy(gpu->stream);
    free(gpu);
}

int h3_gpu_is_m5(const h3_gpu *gpu) {
    return gpu && gpu->fast_path;
}

int h3_gpu_has_nax_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
    return gpu && gpu->tensor_fast_path;
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_tensor_alloc(gpu, elements, H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_tensor_alloc(gpu, elements, H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_tensor_alloc(gpu, elements, H3_GPU_I8);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner) {
        tensor->owner->stats.live_bytes -= tensor->bytes;
    }
    if (tensor->device) cudaFree(tensor->device);
    free(tensor);
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->elements : 0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->dtype : H3_GPU_F32;
}

static int h3_tensor_check(const h3_gpu_tensor *tensor, h3_gpu_dtype dtype,
                           size_t elements) {
    return tensor && tensor->device && tensor->dtype == dtype &&
           tensor->elements >= elements;
}

static int h3_read_file_at(const char *path, uint64_t offset, void *buffer,
                           size_t bytes, char *error, size_t error_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        }
        return 0;
    }
    ssize_t got = pread(fd, buffer, bytes, (off_t)offset);
    close(fd);
    if (got < 0 || (size_t)got != bytes) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot read %zu bytes from %s", bytes,
                     path);
        }
        return 0;
    }
    return 1;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!tensor) return NULL;
    void *host = malloc(tensor->bytes);
    if (!host) {
        h3_gpu_tensor_free(tensor);
        h3_gpu_fail(gpu, "out of memory staging BF16 weight read");
        return NULL;
    }
    if (!h3_read_file_at(path, file_offset, host, tensor->bytes, gpu->error,
                         sizeof(gpu->error))) {
        free(host);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    if (!h3_cuda_check(gpu, cudaMemcpy(tensor->device, host, tensor->bytes,
                                       cudaMemcpyHostToDevice),
                       "cudaMemcpy load_bf16")) {
        free(host);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    free(host);
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(gpu, elements);
    if (!tensor) return NULL;
    void *host = malloc(tensor->bytes);
    if (!host) {
        h3_gpu_tensor_free(tensor);
        h3_gpu_fail(gpu, "out of memory staging F32 weight read");
        return NULL;
    }
    if (!h3_read_file_at(path, file_offset, host, tensor->bytes, gpu->error,
                         sizeof(gpu->error))) {
        free(host);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    if (!h3_cuda_check(gpu, cudaMemcpy(tensor->device, host, tensor->bytes,
                                       cudaMemcpyHostToDevice),
                       "cudaMemcpy load_f32")) {
        free(host);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    free(host);
    return tensor;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    if (!tensor || tensor->dtype != H3_GPU_BF16 || tensor->elements < elements)
        return 0;
    size_t bytes = elements * sizeof(uint16_t);
    void *host = malloc(bytes);
    if (!host) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }
    if (!h3_read_file_at(path, file_offset, host, bytes, error, error_size)) {
        free(host);
        return 0;
    }
    cudaError_t status = cudaMemcpy(tensor->device, host, bytes,
                                    cudaMemcpyHostToDevice);
    free(host);
    if (status != cudaSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cudaMemcpy failed: %s",
                     cudaGetErrorString(status));
        }
        return 0;
    }
    return 1;
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    int ok = h3_gpu_tensor_read_file_bf16(tensor, path, file_offset, elements,
                                          error, error_size);
#ifdef __linux__
    if (ok) {
        int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd);
        }
    }
#endif
    return ok;
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(gpu, elements);
    if (!tensor || !values) return NULL;
    if (!h3_cuda_check(gpu, cudaMemcpy(tensor->device, values,
                                       tensor->bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy from_f32")) {
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!tensor || !values) return NULL;
    if (!h3_cuda_check(gpu, cudaMemcpy(tensor->device, values,
                                       tensor->bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy from_bf16")) {
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    h3_gpu_tensor *tensor = h3_tensor_alloc(gpu, elements, H3_GPU_U32);
    if (!tensor || !values) return NULL;
    if (!h3_cuda_check(gpu, cudaMemcpy(tensor->device, values,
                                       tensor->bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy from_u32")) {
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_gpu_tensor_read_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    if (!h3_tensor_check(tensor, H3_GPU_F32, source_offset + elements) || !values)
        return 0;
    size_t bytes = elements * sizeof(float);
    return cudaMemcpy(values, (char *)tensor->device + source_offset * sizeof(float),
                      bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    if (!h3_tensor_check(tensor, H3_GPU_BF16, elements) || !values) return 0;
    return cudaMemcpy(values, tensor->device, elements * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_gpu_tensor_write_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    if (!h3_tensor_check(tensor, H3_GPU_F32, destination_offset + elements) ||
        !values)
        return 0;
    return cudaMemcpy((char *)tensor->device + destination_offset * sizeof(float),
                      values, elements * sizeof(float),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_gpu_tensor_write_bf16_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    if (!h3_tensor_check(tensor, H3_GPU_BF16, destination_offset + elements) ||
        !values)
        return 0;
    return cudaMemcpy((char *)tensor->device +
                          destination_offset * sizeof(uint16_t),
                      values, elements * sizeof(uint16_t),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

__global__ static void h3_copy_bf16_kernel(const uint16_t *source,
                                           uint16_t *destination,
                                           size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) destination[index] = source[index];
}

__global__ static void h3_copy_f32_kernel(const float *source, float *destination,
                                          size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) destination[index] = source[index];
}

__global__ static void h3_cast_f32_to_bf16_kernel(const float *input,
                                                  uint16_t *output,
                                                  size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = h3_f32_to_bf16_bits(input[index]);
}

__global__ static void h3_cast_bf16_to_f32_kernel(const uint16_t *input,
                                                  float *output, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = h3_bf16_bits_to_f32(input[index]);
}

__global__ static void h3_add_bf16_kernel(const uint16_t *left,
                                          const uint16_t *right,
                                          uint16_t *output, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float sum = h3_bf16_bits_to_f32(left[index]) +
                    h3_bf16_bits_to_f32(right[index]);
        output[index] = h3_f32_to_bf16_bits(sum);
    }
}

int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset, const h3_gpu_tensor *source,
                     size_t source_offset, size_t elements) {
    if (!gpu || !destination || !source ||
        destination->dtype != H3_GPU_BF16 || source->dtype != H3_GPU_BF16 ||
        destination->elements < destination_offset + elements ||
        source->elements < source_offset + elements)
        return h3_gpu_fail(gpu, "invalid BF16 copy request");
    const uint16_t *src = (const uint16_t *)source->device + source_offset;
    uint16_t *dst = (uint16_t *)destination->device + destination_offset;
    unsigned threads = 256;
    unsigned blocks = (unsigned)((elements + threads - 1) / threads);
    h3_copy_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(src, dst, elements);
    gpu->stats.blit_copies++;
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_copy_bf16");
}

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset, const h3_gpu_tensor *source,
                    size_t source_offset, size_t elements) {
    if (!gpu || !destination || !source ||
        destination->dtype != H3_GPU_F32 || source->dtype != H3_GPU_F32 ||
        destination->elements < destination_offset + elements ||
        source->elements < source_offset + elements)
        return h3_gpu_fail(gpu, "invalid F32 copy request");
    const float *src = (const float *)source->device + source_offset;
    float *dst = (float *)destination->device + destination_offset;
    unsigned threads = 256;
    unsigned blocks = (unsigned)((elements + threads - 1) / threads);
    h3_copy_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(src, dst, elements);
    gpu->stats.blit_copies++;
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_copy_f32");
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!gpu || !output || !input || output->dtype != H3_GPU_BF16 ||
        input->dtype != H3_GPU_F32 || output->elements < elements ||
        input->elements < elements)
        return h3_gpu_fail(gpu, "invalid F32 to BF16 cast request");
    unsigned threads = 256;
    unsigned blocks = (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_cast_f32_to_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (uint16_t *)output->device, elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_cast_f32_to_bf16");
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!gpu || !output || !input || output->dtype != H3_GPU_F32 ||
        input->dtype != H3_GPU_BF16 || output->elements < elements ||
        input->elements < elements)
        return h3_gpu_fail(gpu, "invalid BF16 to F32 cast request");
    unsigned threads = 256;
    unsigned blocks = (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_cast_bf16_to_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)input->device, (float *)output->device, elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_cast_bf16_to_f32");
}

int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!gpu || !output || !left || !right || output->dtype != H3_GPU_BF16 ||
        left->dtype != H3_GPU_BF16 || right->dtype != H3_GPU_BF16 ||
        output->elements < elements || left->elements < elements ||
        right->elements < elements)
        return h3_gpu_fail(gpu, "invalid BF16 add request");
    unsigned threads = 256;
    unsigned blocks = (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_add_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)left->device, (const uint16_t *)right->device,
        (uint16_t *)output->device, elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_add_bf16");
}

__global__ static void h3_silu_bf16_kernel(const uint16_t *input,
                                           uint16_t *output, uint32_t count) {
    uint32_t index = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = h3_bf16_bits_to_f32(input[index]);
    output[index] = h3_f32_to_bf16_bits(value / (1.0f + expf(-value)));
}

struct h3_rms_norm_args {
    uint32_t rows;
    uint32_t width;
    float epsilon;
};

__global__ static void h3_rms_norm_bf16_kernel(const uint16_t *input,
                                               const uint16_t *weight,
                                               uint16_t *output,
                                               h3_rms_norm_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;

    extern __shared__ float reductions[];
    const uint16_t *row_input = input + (size_t)row * args.width;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float value = h3_bf16_bits_to_f32(row_input[column]);
        local_sum = fmaf(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse =
        rsqrtf(reductions[0] / (float)args.width + args.epsilon);
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_bits_to_f32(row_input[column]) * inverse;
        output[(size_t)row * args.width + column] = h3_f32_to_bf16_bits(
            normalized * h3_bf16_bits_to_f32(weight[column]));
    }
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    if (!gpu || !output || !input || output->dtype != H3_GPU_BF16 ||
        input->dtype != H3_GPU_BF16 || output->elements < elements ||
        input->elements < elements)
        return h3_gpu_fail(gpu, "invalid SiLU request");
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_silu_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)input->device, (uint16_t *)output->device, elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_silu_bf16");
}

int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    size_t count = (size_t)rows * width;
    if (!gpu || !output || !input || !weight ||
        output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 ||
        weight->dtype != H3_GPU_BF16 || output->elements < count ||
        input->elements < count || weight->elements < width || !rows || !width)
        return h3_gpu_fail(gpu, "invalid RMSNorm request");
    h3_rms_norm_args args = {rows, width, epsilon};
    unsigned threads = 256;
    h3_rms_norm_bf16_kernel<<<rows, threads, threads * sizeof(float),
                              gpu->stream>>>(
        (const uint16_t *)input->device, (const uint16_t *)weight->device,
        (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_rms_norm_bf16");
}

__global__ static void h3_linear_add_bias_bf16_kernel(uint16_t *output,
                                                      const uint16_t *bias,
                                                      uint32_t rows,
                                                      uint32_t output_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)rows * output_dim;
    if (index >= count) return;
    uint32_t column = (uint32_t)(index % output_dim);
    float sum = h3_bf16_bits_to_f32(output[index]) +
                h3_bf16_bits_to_f32(bias[column]);
    output[index] = h3_f32_to_bf16_bits(sum);
}

int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!gpu || !output || !input || !weight ||
        output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 ||
        weight->dtype != H3_GPU_BF16 ||
        (bias && bias->dtype != H3_GPU_BF16) ||
        output->elements < output_count || input->elements < input_count ||
        weight->elements < weight_count ||
        (bias && bias->elements < output_dim) || !rows || !input_dim ||
        !output_dim)
        return h3_gpu_fail(gpu, "invalid linear request");

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim, (int)rows,
        (int)input_dim, &alpha, weight->device, CUDA_R_16BF, (int)input_dim,
        input->device, CUDA_R_16BF, (int)input_dim, &beta, output->device,
        CUDA_R_16BF, (int)output_dim, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (!h3_cublas_check(gpu, status, "cublasGemmEx linear")) return 0;
    gpu->stats.direct_dispatches++;

    if (bias) {
        unsigned threads = 256;
        unsigned blocks =
            (unsigned)((output_count + threads - 1) / threads);
        h3_linear_add_bias_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (uint16_t *)output->device, (const uint16_t *)bias->device, rows,
            output_dim);
        gpu->stats.direct_dispatches++;
        return h3_cuda_check(gpu, cudaGetLastError(),
                             "h3_linear_add_bias_bf16");
    }
    return 1;
}

int h3_gpu_begin(h3_gpu *gpu) {
    if (!gpu) return 0;
    gpu->error[0] = '\0';
    return 1;
}

int h3_gpu_continue(h3_gpu *gpu) {
    if (!gpu) return 0;
    cudaError_t status = cudaStreamSynchronize(gpu->stream);
    gpu->stats.submissions++;
    return h3_cuda_check(gpu, status, "cudaStreamSynchronize continue");
}

int h3_gpu_submit(h3_gpu *gpu) {
    if (!gpu) return 0;
    cudaError_t status = cudaStreamSynchronize(gpu->stream);
    gpu->stats.submissions++;
    return h3_cuda_check(gpu, status, "cudaStreamSynchronize submit");
}

int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats) return 0;
    *stats = gpu->stats;
    return 1;
}

void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    if (!gpu || !label) return;
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%s", label);
}

void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    (void)gpu;
    (void)phase;
}

} /* extern "C" */
