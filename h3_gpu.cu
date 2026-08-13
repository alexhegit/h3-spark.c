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

__global__ static void h3_silu_f32_kernel(const float *input, float *output,
                                          uint32_t count) {
    uint32_t index = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = input[index];
    output[index] = value / (1.0f + expf(-value));
}

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    if (!gpu || !output || !input || output->dtype != H3_GPU_F32 ||
        input->dtype != H3_GPU_F32 || output->elements < elements ||
        input->elements < elements)
        return h3_gpu_fail(gpu, "invalid F32 SiLU request");
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_silu_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (float *)output->device, elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_silu_f32");
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

__global__ static void h3_linear_add_bias_f32_kernel(float *output,
                                                     const float *bias,
                                                     uint32_t rows,
                                                     uint32_t output_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)rows * output_dim;
    if (index >= count) return;
    uint32_t column = (uint32_t)(index % output_dim);
    output[index] += bias[column];
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!gpu || !output || !input || !weight ||
        output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
        weight->dtype != H3_GPU_F32 ||
        (bias && bias->dtype != H3_GPU_F32) ||
        output->elements < output_count || input->elements < input_count ||
        weight->elements < weight_count ||
        (bias && bias->elements < output_dim) || !rows || !input_dim ||
        !output_dim)
        return h3_gpu_fail(gpu, "invalid F32 linear request");

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim, (int)rows,
        (int)input_dim, &alpha, weight->device, CUDA_R_32F, (int)input_dim,
        input->device, CUDA_R_32F, (int)input_dim, &beta, output->device,
        CUDA_R_32F, (int)output_dim, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (!h3_cublas_check(gpu, status, "cublasGemmEx linear_f32")) return 0;
    gpu->stats.direct_dispatches++;

    if (bias) {
        unsigned threads = 256;
        unsigned blocks =
            (unsigned)((output_count + threads - 1) / threads);
        h3_linear_add_bias_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (float *)output->device, (const float *)bias->device, rows,
            output_dim);
        gpu->stats.direct_dispatches++;
        return h3_cuda_check(gpu, cudaGetLastError(),
                             "h3_linear_add_bias_f32");
    }
    return 1;
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

struct h3_adaln_args {
    uint32_t rows;
    uint32_t width;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    float epsilon;
    size_t input_offset;
};

__global__ static void h3_adaln_bf16_kernel(
    const uint16_t *input, const uint16_t *weight, const uint16_t *modulation,
    const uint32_t *row_map, uint16_t *output, h3_adaln_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;

    extern __shared__ float reductions[];
    const uint16_t *row_input =
        input + args.input_offset + (size_t)row * args.width;
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
    uint32_t map_row = row_map[row];
    size_t base = (size_t)map_row * args.slots * args.width;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_bits_to_f32(row_input[column]) * inverse *
                           h3_bf16_bits_to_f32(weight[column]);
        float shift = h3_bf16_bits_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_bits_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[(size_t)row * args.width + column] = h3_f32_to_bf16_bits(
            normalized * (1.0f + scale) + shift);
    }
}

static int h3_gpu_adaln_bf16_impl(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *input,
                                  size_t input_offset,
                                  const h3_gpu_tensor *norm_weight,
                                  const h3_gpu_tensor *modulation,
                                  const h3_gpu_tensor *row_map, uint32_t rows,
                                  uint32_t width, uint32_t slots,
                                  uint32_t shift_slot, uint32_t scale_slot,
                                  float epsilon) {
    size_t count = (size_t)rows * width;
    if (!gpu || !output || !input || !norm_weight || !modulation ||
        !row_map || output->dtype != H3_GPU_BF16 ||
        input->dtype != H3_GPU_BF16 || norm_weight->dtype != H3_GPU_BF16 ||
        modulation->dtype != H3_GPU_BF16 ||
        row_map->dtype != H3_GPU_U32 ||
        output->elements < count ||
        input->elements < input_offset + count ||
        norm_weight->elements < width || row_map->elements < rows ||
        shift_slot >= slots || scale_slot >= slots || !rows || !width)
        return h3_gpu_fail(gpu, "invalid AdaLN request");
    h3_adaln_args args = {rows,       width,      slots,
                          shift_slot, scale_slot, epsilon,
                          input_offset};
    unsigned threads = 256;
    h3_adaln_bf16_kernel<<<rows, threads, threads * sizeof(float),
                           gpu->stream>>>(
        (const uint16_t *)input->device, (const uint16_t *)norm_weight->device,
        (const uint16_t *)modulation->device,
        (const uint32_t *)row_map->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_adaln_bf16");
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    return h3_gpu_adaln_bf16_impl(gpu, output, input, 0, norm_weight,
                                  modulation, row_map, rows, width, slots,
                                  shift_slot, scale_slot, epsilon);
}

int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *norm_weight,
                             const h3_gpu_tensor *modulation,
                             const h3_gpu_tensor *row_map, uint32_t rows,
                             uint32_t width, uint32_t slots,
                             uint32_t shift_slot, uint32_t scale_slot,
                             float epsilon) {
    return h3_gpu_adaln_bf16_impl(gpu, output, input, input_offset,
                                  norm_weight, modulation, row_map, rows,
                                  width, slots, shift_slot, scale_slot,
                                  epsilon);
}

struct h3_gate_args {
    uint32_t rows;
    uint32_t width;
    uint32_t slots;
    uint32_t gate_slot;
};

__global__ static void h3_gate_bf16_kernel(
    const uint16_t *residual, const uint16_t *branch,
    const uint16_t *modulation, const uint32_t *row_map, uint16_t *output,
    h3_gate_args args) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.y;
    if (row >= args.rows || column >= args.width) return;
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    float gate = h3_bf16_bits_to_f32(
        modulation[base + args.gate_slot * args.width + column]);
    size_t index = (size_t)row * args.width + column;
    float value = h3_bf16_bits_to_f32(residual[index]) +
                  h3_bf16_bits_to_f32(branch[index]) * gate;
    output[index] = h3_f32_to_bf16_bits(value);
}

int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    size_t count = (size_t)rows * width;
    if (!gpu || !output || !residual || !branch || !modulation || !row_map ||
        output->dtype != H3_GPU_BF16 || residual->dtype != H3_GPU_BF16 ||
        branch->dtype != H3_GPU_BF16 || modulation->dtype != H3_GPU_BF16 ||
        row_map->dtype != H3_GPU_U32 || output->elements < count ||
        residual->elements < count || branch->elements < count ||
        row_map->elements < rows || gate_slot >= slots || !rows || !width)
        return h3_gpu_fail(gpu, "invalid gate request");
    h3_gate_args args = {rows, width, slots, gate_slot};
    dim3 threads(256, 1, 1);
    dim3 blocks((width + threads.x - 1) / threads.x, rows, 1);
    h3_gate_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)residual->device, (const uint16_t *)branch->device,
        (const uint16_t *)modulation->device,
        (const uint32_t *)row_map->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_gate_bf16");
}

struct h3_swiglu_args {
    uint32_t rows;
    uint32_t width;
};

__global__ static void h3_swiglu_bf16_kernel(const uint16_t *fused,
                                             uint16_t *output,
                                             h3_swiglu_args args) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.y;
    if (row >= args.rows || column >= args.width) return;
    size_t base = ((size_t)row * args.width + column);
    size_t fused_base = (size_t)row * args.width * 2u;
    float gate = h3_bf16_bits_to_f32(fused[fused_base + column]);
    float up = h3_bf16_bits_to_f32(fused[fused_base + args.width + column]);
    output[base] = h3_f32_to_bf16_bits(gate / (1.0f + expf(-gate)) * up);
}

int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    size_t fused_count = (size_t)rows * width * 2u;
    size_t output_count = (size_t)rows * width;
    if (!gpu || !output || !fused || output->dtype != H3_GPU_BF16 ||
        fused->dtype != H3_GPU_BF16 || output->elements < output_count ||
        fused->elements < fused_count || !rows || !width)
        return h3_gpu_fail(gpu, "invalid SwiGLU request");
    h3_swiglu_args args = {rows, width};
    dim3 threads(256, 1, 1);
    dim3 blocks((width + threads.x - 1) / threads.x, rows, 1);
    h3_swiglu_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)fused->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_swiglu_bf16");
}

int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    size_t fc1_count = (size_t)rows * hidden_dim * 2u;
    size_t hidden_count = (size_t)rows * hidden_dim;
    if (!gpu || !output || !input || !fc1_weight || !fc2_weight ||
        output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 ||
        fc1_weight->dtype != H3_GPU_BF16 || fc2_weight->dtype != H3_GPU_BF16 ||
        output->elements < (size_t)rows * output_dim ||
        input->elements < (size_t)rows * input_dim ||
        fc1_weight->elements < (size_t)hidden_dim * 2u * input_dim ||
        fc2_weight->elements < (size_t)output_dim * hidden_dim || !rows ||
        !input_dim || !hidden_dim || !output_dim)
        return h3_gpu_fail(gpu, "invalid MLP request");

    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(gpu, fc1_count);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    if (!fc1 || !hidden) {
        h3_gpu_tensor_free(fc1);
        h3_gpu_tensor_free(hidden);
        return h3_gpu_fail(gpu, "MLP temp tensor allocation failed");
    }

    int ok = h3_gpu_linear_bf16(gpu, fc1, input, fc1_weight, NULL, rows,
                                input_dim, hidden_dim * 2u) &&
             h3_gpu_swiglu_bf16(gpu, hidden, fc1, rows, hidden_dim) &&
             h3_gpu_linear_bf16(gpu, output, hidden, fc2_weight, NULL, rows,
                                hidden_dim, output_dim);
    h3_gpu_tensor_free(fc1);
    h3_gpu_tensor_free(hidden);
    return ok;
}

static inline __device__ __host__ float h3_erf_approx(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = fabsf(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t) +
                          1.421413741f) *
                             t -
                         0.284496736f) *
                            t +
                        0.254829592f) *
                       t;
    return sign * (1.0f - polynomial * expf(-x * x));
}

struct h3_gelu_bf16_args {
    uint32_t elements;
    uint32_t approximate;
};

__global__ static void h3_gelu_bf16_kernel(const uint16_t *input,
                                           uint16_t *output,
                                           h3_gelu_bf16_args args) {
    uint32_t index = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= args.elements) return;
    float value = h3_bf16_bits_to_f32(input[index]);
    float activated;
    if (args.approximate) {
        float inner = 0.7978845608028654f *
                      (value + 0.044715f * value * value * value);
        activated = inner <= -10.0f ? 0.0f
                                    : inner >= 10.0f
                                          ? value
                                          : 0.5f * value * (1.0f + tanhf(inner));
    } else {
        activated = value <= -10.0f ? 0.0f
                                    : value >= 10.0f
                                          ? value
                                          : 0.5f * value *
                                                (1.0f + h3_erf_approx(value *
                                                                      0.7071067811865475f));
    }
    output[index] = h3_f32_to_bf16_bits(activated);
}

int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    if (!gpu || !output || !input || output->dtype != H3_GPU_BF16 ||
        input->dtype != H3_GPU_BF16 || output->elements < elements ||
        input->elements < elements)
        return h3_gpu_fail(gpu, "invalid GELU request");
    h3_gelu_bf16_args args = {elements, approximate ? 1u : 0u};
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_gelu_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)input->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_gelu_bf16");
}

struct h3_embedding_args {
    uint32_t tokens;
    uint32_t vocab_size;
    uint32_t width;
};

__global__ static void h3_embedding_bf16_kernel(
    const uint16_t *weight, const uint32_t *token_ids, uint16_t *output,
    h3_embedding_args args) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t token = (uint32_t)blockIdx.y;
    if (token >= args.tokens || column >= args.width) return;
    uint32_t identifier = token_ids[token];
    output[(size_t)token * args.width + column] =
        identifier < args.vocab_size
            ? weight[(size_t)identifier * args.width + column]
            : (uint16_t)0;
}

struct h3_qkv_rope_args {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t rope_half;
    uint32_t grouped;
    float epsilon;
};

__global__ static void h3_qkv_rope_bf16_kernel(
    const uint16_t *qkv, const uint16_t *q_weight, const uint16_t *k_weight,
    const uint16_t *rope_cos, const uint16_t *rope_sin, uint16_t *query,
    uint16_t *key, uint16_t *value, h3_qkv_rope_args args) {
    uint32_t dimension = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t row = (uint32_t)blockIdx.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence)
        return;
    uint32_t inner = args.heads * args.head_dim;
    uint32_t row_base = row * inner * 3u;
    uint32_t q_base = row_base + head * args.head_dim;
    uint32_t k_base = q_base + inner;
    uint32_t v_base = q_base + inner * 2u;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3u;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float q = h3_bf16_bits_to_f32(qkv[q_base + d]);
        float k = h3_bf16_bits_to_f32(qkv[k_base + d]);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = rsqrtf(q_sum / (float)args.head_dim + args.epsilon);
    float k_inverse = rsqrtf(k_sum / (float)args.head_dim + args.epsilon);
    float q0 = h3_bf16_bits_to_f32(qkv[q_base + dimension]) * q_inverse *
               h3_bf16_bits_to_f32(q_weight[dimension]);
    float k0 = h3_bf16_bits_to_f32(qkv[k_base + dimension]) * k_inverse *
               h3_bf16_bits_to_f32(k_weight[dimension]);
    if (dimension < args.rope_half) {
        uint32_t pair = dimension + args.rope_half;
        float q1 = h3_bf16_bits_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_bits_to_f32(q_weight[pair]);
        float k1 = h3_bf16_bits_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_bits_to_f32(k_weight[pair]);
        float c =
            h3_bf16_bits_to_f32(rope_cos[row * args.rope_half + dimension]);
        float s =
            h3_bf16_bits_to_f32(rope_sin[row * args.rope_half + dimension]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2u) {
        uint32_t pair = dimension - args.rope_half;
        float q1 = h3_bf16_bits_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_bits_to_f32(q_weight[pair]);
        float k1 = h3_bf16_bits_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_bits_to_f32(k_weight[pair]);
        float c = h3_bf16_bits_to_f32(rope_cos[row * args.rope_half + pair]);
        float s = h3_bf16_bits_to_f32(rope_sin[row * args.rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint32_t output_index =
        (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16_bits(q0);
    key[output_index] = h3_f32_to_bf16_bits(k0);
    value[output_index] = qkv[v_base + dimension];
}

static int h3_gpu_qkv_rope_bf16_layout(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
    const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half,
    uint32_t grouped, float epsilon) {
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!gpu || !query || !key || !value || !qkv || !q_norm || !k_norm ||
        !rope_cos || !rope_sin || query->dtype != H3_GPU_BF16 ||
        key->dtype != H3_GPU_BF16 || value->dtype != H3_GPU_BF16 ||
        qkv->dtype != H3_GPU_BF16 || q_norm->dtype != H3_GPU_BF16 ||
        k_norm->dtype != H3_GPU_BF16 || rope_cos->dtype != H3_GPU_BF16 ||
        rope_sin->dtype != H3_GPU_BF16 || qkv->elements < count * 3u ||
        q_norm->elements < head_dim || k_norm->elements < head_dim ||
        rope_cos->elements < rope_count || rope_sin->elements < rope_count ||
        query->elements < count || key->elements < count ||
        value->elements < count || !sequence || !heads || !head_dim ||
        rope_half * 2u > head_dim)
        return h3_gpu_fail(gpu, "invalid QKV/RoPE request");
    h3_qkv_rope_args args = {sequence, heads, head_dim, rope_half, grouped,
                             epsilon};
    dim3 threads(head_dim, 1, 1);
    dim3 blocks(1, heads, sequence);
    h3_qkv_rope_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)qkv->device, (const uint16_t *)q_norm->device,
        (const uint16_t *)k_norm->device, (const uint16_t *)rope_cos->device,
        (const uint16_t *)rope_sin->device, (uint16_t *)query->device,
        (uint16_t *)key->device, (uint16_t *)value->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_qkv_rope_bf16");
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
                         h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim, uint32_t rope_half,
                         float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(
        gpu, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
        sequence, heads, head_dim, rope_half, 0u, epsilon);
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(
        gpu, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
        sequence, heads, head_dim, rope_half, 1u, epsilon);
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, h3_gpu_tensor *qkv, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *q_norm,
    const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
    const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim,
    uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t inner = heads * head_dim;
    return h3_gpu_linear_bf16(gpu, qkv, input, weight, NULL, rows, input_dim,
                              inner * 3u) &&
           h3_gpu_qkv_rope_bf16_layout(
               gpu, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
               rows, heads, head_dim, rope_half, 1u, epsilon);
}

struct h3_sdpa_args {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    float scale;
    uint32_t head_major_output;
};

__global__ static void h3_sdpa_bf16_kernel(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    uint16_t *output, h3_sdpa_args args) {
    extern __shared__ float scores[];
    uint32_t head = (uint32_t)blockIdx.x;
    uint32_t q_row = (uint32_t)blockIdx.y;
    uint32_t dim = (uint32_t)threadIdx.x;
    if (head >= args.heads || q_row >= args.sequence || dim >= args.head_dim)
        return;
    size_t q_base = ((size_t)q_row * args.heads + head) * args.head_dim;
    if (dim == 0) {
        float max_score = -INFINITY;
        for (uint32_t k_row = 0; k_row < args.sequence; k_row++) {
            size_t k_base = ((size_t)k_row * args.heads + head) * args.head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < args.head_dim; d++) {
                float q = h3_bf16_bits_to_f32(query[q_base + d]);
                float k = h3_bf16_bits_to_f32(key[k_base + d]);
                dot = fmaf(q, k, dot);
            }
            scores[k_row] = dot * args.scale;
            if (scores[k_row] > max_score) max_score = scores[k_row];
        }
        float sum = 0.0f;
        for (uint32_t k_row = 0; k_row < args.sequence; k_row++) {
            scores[k_row] = expf(scores[k_row] - max_score);
            sum += scores[k_row];
        }
        float inverse = 1.0f / sum;
        for (uint32_t k_row = 0; k_row < args.sequence; k_row++)
            scores[k_row] *= inverse;
    }
    __syncthreads();
    float accumulated = 0.0f;
    for (uint32_t k_row = 0; k_row < args.sequence; k_row++) {
        size_t v_base = ((size_t)k_row * args.heads + head) * args.head_dim;
        accumulated = fmaf(scores[k_row],
                           h3_bf16_bits_to_f32(value[v_base + dim]),
                           accumulated);
    }
    size_t output_index = args.head_major_output
                              ? ((size_t)head * args.sequence + q_row) *
                                        args.head_dim +
                                    dim
                              : q_base + dim;
    output[output_index] = h3_f32_to_bf16_bits(accumulated);
}

static int h3_gpu_sdpa_bf16_impl(h3_gpu *gpu, h3_gpu_tensor *output,
                                   const h3_gpu_tensor *query,
                                   const h3_gpu_tensor *key,
                                   const h3_gpu_tensor *value,
                                   uint32_t sequence, uint32_t heads,
                                   uint32_t head_dim, float scale,
                                   int head_major_output) {
    size_t count = (size_t)sequence * heads * head_dim;
    if (!gpu || !output || !query || !key || !value ||
        output->dtype != H3_GPU_BF16 || query->dtype != H3_GPU_BF16 ||
        key->dtype != H3_GPU_BF16 || value->dtype != H3_GPU_BF16 ||
        output->elements < count || query->elements < count ||
        key->elements < count || value->elements < count || !sequence ||
        !heads || !head_dim)
        return h3_gpu_fail(gpu, "invalid SDPA request");
    h3_sdpa_args args = {sequence, heads, head_dim, scale,
                         head_major_output ? 1u : 0u};
    dim3 threads(head_dim, 1, 1);
    dim3 blocks(heads, sequence, 1);
    size_t shared_bytes = (size_t)sequence * sizeof(float);
    h3_sdpa_bf16_kernel<<<blocks, threads, shared_bytes, gpu->stream>>>(
        (const uint16_t *)query->device, (const uint16_t *)key->device,
        (const uint16_t *)value->device, (uint16_t *)output->device, args);
    gpu->stats.mps_sdpa_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_bf16");
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa_bf16_impl(gpu, output, query, key, value, sequence,
                                 heads, head_dim, scale, 0);
}

int h3_gpu_sdpa_bf16_head_major_output(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query,
    const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa_bf16_impl(gpu, output, query, key, value, sequence,
                                 heads, head_dim, scale, 1);
}

int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    size_t output_count = (size_t)tokens * width;
    if (!gpu || !output || !weight || !token_ids ||
        output->dtype != H3_GPU_BF16 || weight->dtype != H3_GPU_BF16 ||
        token_ids->dtype != H3_GPU_U32 ||
        output->elements < output_count ||
        weight->elements < (size_t)vocab_size * width ||
        token_ids->elements < tokens || !tokens || !vocab_size || !width)
        return h3_gpu_fail(gpu, "invalid embedding request");
    h3_embedding_args args = {tokens, vocab_size, width};
    dim3 threads(256, 1, 1);
    dim3 blocks((width + threads.x - 1) / threads.x, tokens, 1);
    h3_embedding_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)weight->device, (const uint32_t *)token_ids->device,
        (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_embedding_bf16");
}

struct h3_head_norm_args {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    float epsilon;
};

__global__ static void h3_head_rms_norm_bf16_kernel(
    uint16_t *tensor, const uint16_t *weight, h3_head_norm_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    if (row >= args.sequence || head >= args.heads) return;
    size_t base = ((size_t)row * args.heads + head) * args.head_dim;
    float sum = 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_bits_to_f32(tensor[base + d]);
        sum = fmaf(value, value, sum);
    }
    float inverse = rsqrtf(sum / (float)args.head_dim + args.epsilon);
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_bits_to_f32(tensor[base + d]);
        tensor[base + d] = h3_f32_to_bf16_bits(
            value * inverse * h3_bf16_bits_to_f32(weight[d]));
    }
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim, float epsilon) {
    size_t count = (size_t)sequence * heads * head_dim;
    if (!gpu || !tensor || !weight || tensor->dtype != H3_GPU_BF16 ||
        weight->dtype != H3_GPU_BF16 || tensor->elements < count ||
        weight->elements < head_dim || !sequence || !heads || !head_dim)
        return h3_gpu_fail(gpu, "invalid head RMSNorm request");
    h3_head_norm_args args = {sequence, heads, head_dim, epsilon};
    dim3 blocks(sequence, heads, 1);
    h3_head_rms_norm_bf16_kernel<<<blocks, 1, 0, gpu->stream>>>(
        (uint16_t *)tensor->device, (const uint16_t *)weight->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_head_rms_norm_bf16");
}

struct h3_text_rope_args {
    uint32_t sequence;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
};

__global__ static void h3_rope_text_bf16_kernel(
    uint16_t *query, uint16_t *key, const float *rope_cos,
    const float *rope_sin, h3_text_rope_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    if (row >= args.sequence) return;
    uint32_t half_dim = args.head_dim / 2u;
    if (head < args.query_heads) {
        size_t base = ((size_t)row * args.query_heads + head) * args.head_dim;
        for (uint32_t d = 0; d < half_dim; d++) {
            float first = h3_bf16_bits_to_f32(query[base + d]);
            float second = h3_bf16_bits_to_f32(query[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            query[base + d] = h3_f32_to_bf16_bits(first * c - second * s);
            query[base + half_dim + d] =
                h3_f32_to_bf16_bits(second * c + first * s);
        }
    }
    if (head < args.kv_heads) {
        size_t base = ((size_t)row * args.kv_heads + head) * args.head_dim;
        for (uint32_t d = 0; d < half_dim; d++) {
            float first = h3_bf16_bits_to_f32(key[base + d]);
            float second = h3_bf16_bits_to_f32(key[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            key[base + d] = h3_f32_to_bf16_bits(first * c - second * s);
            key[base + half_dim + d] =
                h3_f32_to_bf16_bits(second * c + first * s);
        }
    }
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32, uint32_t sequence,
                          uint32_t query_heads, uint32_t kv_heads,
                          uint32_t head_dim) {
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t kv_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2u);
    if (!gpu || !query || !key || !rope_cos_f32 || !rope_sin_f32 ||
        query->dtype != H3_GPU_BF16 || key->dtype != H3_GPU_BF16 ||
        rope_cos_f32->dtype != H3_GPU_F32 ||
        rope_sin_f32->dtype != H3_GPU_F32 || query->elements < query_count ||
        key->elements < kv_count || rope_cos_f32->elements < rope_count ||
        rope_sin_f32->elements < rope_count || !sequence || !query_heads ||
        !kv_heads || !head_dim || head_dim % 2u || query_heads % kv_heads)
        return h3_gpu_fail(gpu, "invalid text RoPE request");
    h3_text_rope_args args = {sequence, query_heads, kv_heads, head_dim};
    uint32_t head_blocks = query_heads > kv_heads ? query_heads : kv_heads;
    dim3 blocks(sequence, head_blocks, 1);
    h3_rope_text_bf16_kernel<<<blocks, 1, 0, gpu->stream>>>(
        (uint16_t *)query->device, (uint16_t *)key->device,
        (const float *)rope_cos_f32->device,
        (const float *)rope_sin_f32->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_rope_text_bf16");
}

struct h3_gqa_args {
    uint32_t sequence;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float scale;
};

__global__ static void h3_gqa_causal_bf16_kernel(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    uint16_t *output, h3_gqa_args args) {
    extern __shared__ float scores[];
    uint32_t query_row = (uint32_t)blockIdx.x;
    uint32_t query_head = (uint32_t)blockIdx.y;
    uint32_t dim = (uint32_t)threadIdx.x;
    if (query_row >= args.sequence || query_head >= args.query_heads ||
        dim >= args.head_dim)
        return;
    uint32_t kv_head = query_head / (args.query_heads / args.kv_heads);
    uint32_t key_count = query_row + 1u;
    size_t q_base = ((size_t)query_row * args.query_heads + query_head) *
                    args.head_dim;
    if (dim == 0) {
        float scaled_query[128];
        for (uint32_t d = 0; d < args.head_dim; d++) {
            float q = h3_bf16_bits_to_f32(query[q_base + d]) * args.scale;
            scaled_query[d] = h3_bf16_bits_to_f32(h3_f32_to_bf16_bits(q));
        }
        float max_score = -INFINITY;
        for (uint32_t key_row = 0; key_row < key_count; key_row++) {
            size_t k_base =
                ((size_t)key_row * args.kv_heads + kv_head) * args.head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < args.head_dim; d++) {
                dot = fmaf(scaled_query[d],
                           h3_bf16_bits_to_f32(key[k_base + d]), dot);
            }
            scores[key_row] = dot;
            if (dot > max_score) max_score = dot;
        }
        float sum = 0.0f;
        for (uint32_t key_row = 0; key_row < key_count; key_row++) {
            scores[key_row] = expf(scores[key_row] - max_score);
            sum += scores[key_row];
        }
        float inverse = 1.0f / sum;
        for (uint32_t key_row = 0; key_row < key_count; key_row++)
            scores[key_row] *= inverse;
    }
    __syncthreads();
    float accumulated = 0.0f;
    for (uint32_t key_row = 0; key_row < key_count; key_row++) {
        size_t v_base =
            ((size_t)key_row * args.kv_heads + kv_head) * args.head_dim;
        accumulated = fmaf(scores[key_row],
                           h3_bf16_bits_to_f32(value[v_base + dim]),
                           accumulated);
    }
    output[q_base + dim] = h3_f32_to_bf16_bits(accumulated);
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale) {
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t kv_count = (size_t)sequence * kv_heads * head_dim;
    if (!gpu || !output || !query || !key || !value ||
        output->dtype != H3_GPU_BF16 || query->dtype != H3_GPU_BF16 ||
        key->dtype != H3_GPU_BF16 || value->dtype != H3_GPU_BF16 ||
        output->elements < query_count || query->elements < query_count ||
        key->elements < kv_count || value->elements < kv_count || !sequence ||
        !query_heads || !kv_heads || !head_dim || head_dim > 128 ||
        query_heads % kv_heads)
        return h3_gpu_fail(gpu, "invalid GQA causal request");
    h3_gqa_args args = {sequence, query_heads, kv_heads, head_dim, scale};
    dim3 threads(head_dim, 1, 1);
    dim3 blocks(sequence, query_heads, 1);
    size_t shared_bytes = (size_t)sequence * sizeof(float);
    h3_gqa_causal_bf16_kernel<<<blocks, threads, shared_bytes, gpu->stream>>>(
        (const uint16_t *)query->device, (const uint16_t *)key->device,
        (const uint16_t *)value->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_gqa_causal_bf16");
}

__global__ static void h3_silu_mul_bf16_kernel(const uint16_t *gate,
                                               const uint16_t *up,
                                               uint16_t *output,
                                               uint32_t count) {
    uint32_t index = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = h3_bf16_bits_to_f32(gate[index]);
    float other = h3_bf16_bits_to_f32(up[index]);
    output[index] = h3_f32_to_bf16_bits(value / (1.0f + expf(-value)) * other);
}

int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate, const h3_gpu_tensor *up,
                         uint32_t elements) {
    if (!gpu || !output || !gate || !up || output->dtype != H3_GPU_BF16 ||
        gate->dtype != H3_GPU_BF16 || up->dtype != H3_GPU_BF16 ||
        output->elements < elements || gate->elements < elements ||
        up->elements < elements)
        return h3_gpu_fail(gpu, "invalid SiLU mul request");
    unsigned threads = 256;
    unsigned blocks = (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_silu_mul_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)gate->device, (const uint16_t *)up->device,
        (uint16_t *)output->device, elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_silu_mul_bf16");
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
