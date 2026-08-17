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
#include <time.h>
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
    h3_gpu_stats profile_start_stats;
    h3_gpu_stats profile_mark_stats;
    double profile_start_wall;
    double profile_mark_wall;
};

static double h3_gpu_now(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) return 0.0;
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static int h3_gpu_profile_enabled(void) {
    const char *value = getenv("H3_PROFILE");
    return value && *value && strcmp(value, "0") != 0;
}

static uint64_t h3_gpu_counter_delta(uint64_t value, uint64_t start) {
    return value >= start ? value - start : 0;
}

static void h3_gpu_profile_emit(h3_gpu *gpu, const char *phase,
                                const h3_gpu_stats *start, double wall_start) {
    if (!gpu || !phase || !h3_gpu_profile_enabled()) return;
    if (gpu->stream) cudaStreamSynchronize(gpu->stream);
    h3_gpu_stats value = gpu->stats;
    double wall = h3_gpu_now() - wall_start;
    const char *label =
        gpu->profile_label[0] ? gpu->profile_label : "CUDA context";
    fprintf(stderr,
            "h3 profile: %-24s %-14s wall=%8.3fs "
            "peak=%7.3fGiB alloc=%7.3fGiB submissions=%llu "
            "direct=%llu linear=%llu conv=%llu attention=%llu\n",
            label, phase, wall,
            (double)value.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)h3_gpu_counter_delta(value.allocated_bytes,
                                         start->allocated_bytes) /
                (1024.0 * 1024.0 * 1024.0),
            (unsigned long long)h3_gpu_counter_delta(value.submissions,
                                                     start->submissions),
            (unsigned long long)h3_gpu_counter_delta(value.direct_dispatches,
                                                     start->direct_dispatches),
            (unsigned long long)h3_gpu_counter_delta(
                value.mps_linear_dispatches, start->mps_linear_dispatches),
            (unsigned long long)h3_gpu_counter_delta(value.mps_conv_dispatches,
                                                     start->mps_conv_dispatches),
            (unsigned long long)h3_gpu_counter_delta(
                value.mps_sdpa_dispatches, start->mps_sdpa_dispatches));
    fflush(stderr);
}

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
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%s",
             "CUDA context");
    gpu->profile_start_stats = gpu->stats;
    gpu->profile_mark_stats = gpu->stats;
    gpu->profile_start_wall = h3_gpu_now();
    gpu->profile_mark_wall = gpu->profile_start_wall;
    return gpu;
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    h3_gpu_profile_emit(gpu, "total", &gpu->profile_start_stats,
                        gpu->profile_start_wall);
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

__global__ static void h3_sub_bf16_kernel(const uint16_t *left,
                                          const uint16_t *right,
                                          uint16_t *output, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float difference = h3_bf16_bits_to_f32(left[index]) -
                           h3_bf16_bits_to_f32(right[index]);
        output[index] = h3_f32_to_bf16_bits(difference);
    }
}

struct h3_euler_args {
    uint32_t sample_offset;
    uint32_t elements;
    float delta;
    float ratio;
};

__global__ static void h3_euler_bf16_kernel(float *sample, const uint16_t *last,
                                          const uint16_t *previous,
                                          h3_euler_args args) {
    uint32_t index = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= args.elements) return;
    float last_value = h3_bf16_bits_to_f32(last[index]);
    float velocity = fmaf(args.ratio,
                          last_value - h3_bf16_bits_to_f32(previous[index]),
                          last_value);
    uint32_t sample_index = args.sample_offset + index;
    sample[sample_index] =
        fmaf(args.delta, velocity, sample[sample_index]);
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

int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!gpu || !output || !left || !right || output->dtype != H3_GPU_BF16 ||
        left->dtype != H3_GPU_BF16 || right->dtype != H3_GPU_BF16 ||
        output->elements < elements || left->elements < elements ||
        right->elements < elements)
        return h3_gpu_fail(gpu, "invalid BF16 sub request");
    unsigned threads = 256;
    unsigned blocks = (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_sub_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)left->device, (const uint16_t *)right->device,
        (uint16_t *)output->device, elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_sub_bf16");
}

int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    if (!gpu || !sample || !last || !previous ||
        sample->dtype != H3_GPU_F32 || last->dtype != H3_GPU_BF16 ||
        previous->dtype != H3_GPU_BF16 ||
        sample->elements < sample_offset + elements ||
        last->elements < elements || previous->elements < elements)
        return h3_gpu_fail(gpu, "invalid Euler request");
    h3_euler_args args = {
        (uint32_t)sample_offset, elements, delta, ratio};
    unsigned threads = 256;
    unsigned blocks = (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_euler_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (float *)sample->device, (const uint16_t *)last->device,
        (const uint16_t *)previous->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_euler_bf16");
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

__global__ static void h3_layer_norm_bf16_kernel(const uint16_t *input,
                                                 const uint16_t *weight,
                                                 const uint16_t *bias,
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
        local_sum += h3_bf16_bits_to_f32(row_input[column]);
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float mean = reductions[0] / (float)args.width;
    __syncthreads();
    float local_square = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float centered = h3_bf16_bits_to_f32(row_input[column]) - mean;
        local_square = fmaf(centered, centered, local_square);
    }
    reductions[tid] = local_square;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse =
        rsqrtf(reductions[0] / (float)args.width + args.epsilon);
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized =
            (h3_bf16_bits_to_f32(row_input[column]) - mean) * inverse;
        float value = fmaf(normalized, h3_bf16_bits_to_f32(weight[column]),
                           h3_bf16_bits_to_f32(bias[column]));
        output[(size_t)row * args.width + column] = h3_f32_to_bf16_bits(value);
    }
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    size_t count = (size_t)rows * width;
    if (!gpu || !output || !input || !weight || !bias ||
        output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 ||
        weight->dtype != H3_GPU_BF16 || bias->dtype != H3_GPU_BF16 ||
        output->elements < count || input->elements < count ||
        weight->elements < width || bias->elements < width || !rows || !width)
        return h3_gpu_fail(gpu, "invalid LayerNorm request");
    h3_rms_norm_args args = {rows, width, epsilon};
    unsigned threads = 256;
    h3_layer_norm_bf16_kernel<<<rows, threads, threads * sizeof(float),
                                  gpu->stream>>>(
        (const uint16_t *)input->device, (const uint16_t *)weight->device,
        (const uint16_t *)bias->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_layer_norm_bf16");
}

struct h3_vision_qkv_rope_args {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t rope_half;
};

__global__ static void h3_vision_qkv_rope_bf16_kernel(
    const uint16_t *qkv, const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query, uint16_t *key, uint16_t *value,
    h3_vision_qkv_rope_args args) {
    uint32_t dimension = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t row = (uint32_t)blockIdx.z;
    if (dimension >= args.head_dim || head >= args.heads ||
        row >= args.sequence)
        return;
    uint32_t inner = args.heads * args.head_dim;
    size_t row_base = (size_t)row * inner * 3u;
    size_t q_base = row_base + (size_t)head * args.head_dim;
    size_t k_base = row_base + inner + (size_t)head * args.head_dim;
    size_t v_base = row_base + inner * 2u + (size_t)head * args.head_dim;
    uint32_t half_dim = args.rope_half;
    uint32_t pair =
        dimension < half_dim ? dimension + half_dim : dimension - half_dim;
    float c = h3_bf16_bits_to_f32(
        rope_cos[row * half_dim + (dimension % half_dim)]);
    float s = h3_bf16_bits_to_f32(
        rope_sin[row * half_dim + (dimension % half_dim)]);
    float q0 = h3_bf16_bits_to_f32(qkv[q_base + dimension]);
    float k0 = h3_bf16_bits_to_f32(qkv[k_base + dimension]);
    float q1 = h3_bf16_bits_to_f32(qkv[q_base + pair]);
    float k1 = h3_bf16_bits_to_f32(qkv[k_base + pair]);
    float qr =
        dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
    float kr =
        dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
    size_t output_index =
        ((size_t)row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16_bits(qr);
    key[output_index] = h3_f32_to_bf16_bits(kr);
    value[output_index] = qkv[v_base + dimension];
}

int h3_gpu_vision_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                h3_gpu_tensor *key, h3_gpu_tensor *value,
                                const h3_gpu_tensor *qkv,
                                const h3_gpu_tensor *rope_cos,
                                const h3_gpu_tensor *rope_sin,
                                uint32_t sequence, uint32_t heads,
                                uint32_t head_dim, uint32_t rope_half) {
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!gpu || !query || !key || !value || !qkv || !rope_cos || !rope_sin ||
        query->dtype != H3_GPU_BF16 || key->dtype != H3_GPU_BF16 ||
        value->dtype != H3_GPU_BF16 || qkv->dtype != H3_GPU_BF16 ||
        rope_cos->dtype != H3_GPU_BF16 || rope_sin->dtype != H3_GPU_BF16 ||
        qkv->elements < count * 3u || rope_cos->elements < rope_count ||
        rope_sin->elements < rope_count || query->elements < count ||
        key->elements < count || value->elements < count || !sequence ||
        !heads || !head_dim || rope_half * 2u != head_dim)
        return h3_gpu_fail(gpu, "invalid vision QKV/RoPE request");
    h3_vision_qkv_rope_args args = {sequence, heads, head_dim, rope_half};
    dim3 threads(head_dim, 1, 1);
    dim3 blocks(1, heads, sequence);
    h3_vision_qkv_rope_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)qkv->device, (const uint16_t *)rope_cos->device,
        (const uint16_t *)rope_sin->device, (uint16_t *)query->device,
        (uint16_t *)key->device, (uint16_t *)value->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_vision_qkv_rope_bf16");
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

struct h3_text_qk_rope_args {
    uint32_t sequence;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float epsilon;
};

__global__ static void h3_text_qk_rope_bf16_kernel(
    const uint16_t *query_input, const uint16_t *key_input,
    const uint16_t *q_weight, const uint16_t *k_weight,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query_output, uint16_t *key_output,
    h3_text_qk_rope_args args) {
    uint32_t dimension = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t row = (uint32_t)blockIdx.z;
    if (dimension >= args.head_dim || head >= args.query_heads ||
        row >= args.sequence)
        return;
    uint32_t half_dim = args.head_dim / 2u;
    uint32_t pair =
        dimension < half_dim ? dimension + half_dim : dimension - half_dim;
    float c = h3_bf16_bits_to_f32(
        rope_cos[row * half_dim + (dimension % half_dim)]);
    float s = h3_bf16_bits_to_f32(
        rope_sin[row * half_dim + (dimension % half_dim)]);

    size_t q_base = ((size_t)row * args.query_heads + head) * args.head_dim;
    float q_sum = 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_bits_to_f32(query_input[q_base + d]);
        q_sum = fmaf(value, value, q_sum);
    }
    float q_inverse = rsqrtf(q_sum / (float)args.head_dim + args.epsilon);
    float q0 = h3_bf16_bits_to_f32(query_input[q_base + dimension]) *
               q_inverse * h3_bf16_bits_to_f32(q_weight[dimension]);
    float q1 = h3_bf16_bits_to_f32(query_input[q_base + pair]) * q_inverse *
               h3_bf16_bits_to_f32(q_weight[pair]);
    float q_rotated =
        dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
    query_output[q_base + dimension] = h3_f32_to_bf16_bits(q_rotated);

    if (head < args.kv_heads) {
        size_t k_base = ((size_t)row * args.kv_heads + head) * args.head_dim;
        float k_sum = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) {
            float value = h3_bf16_bits_to_f32(key_input[k_base + d]);
            k_sum = fmaf(value, value, k_sum);
        }
        float k_inverse = rsqrtf(k_sum / (float)args.head_dim + args.epsilon);
        float k0 = h3_bf16_bits_to_f32(key_input[k_base + dimension]) *
                   k_inverse * h3_bf16_bits_to_f32(k_weight[dimension]);
        float k1 = h3_bf16_bits_to_f32(key_input[k_base + pair]) * k_inverse *
                   h3_bf16_bits_to_f32(k_weight[pair]);
        float k_rotated =
            dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
        key_output[k_base + dimension] = h3_f32_to_bf16_bits(k_rotated);
    }
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) {
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2u);
    if (!gpu || !query_output || !key_output || !query_input || !key_input ||
        !q_norm || !k_norm || !rope_cos || !rope_sin ||
        query_output->dtype != H3_GPU_BF16 ||
        key_output->dtype != H3_GPU_BF16 ||
        query_input->dtype != H3_GPU_BF16 ||
        key_input->dtype != H3_GPU_BF16 || q_norm->dtype != H3_GPU_BF16 ||
        k_norm->dtype != H3_GPU_BF16 || rope_cos->dtype != H3_GPU_BF16 ||
        rope_sin->dtype != H3_GPU_BF16 ||
        query_output->elements < query_count ||
        key_output->elements < key_count ||
        query_input->elements < query_count ||
        key_input->elements < key_count || q_norm->elements < head_dim ||
        k_norm->elements < head_dim || rope_cos->elements < rope_count ||
        rope_sin->elements < rope_count || !sequence || !query_heads ||
        !kv_heads || !head_dim || head_dim % 2u || query_heads % kv_heads)
        return h3_gpu_fail(gpu, "invalid text QK/RoPE request");
    h3_text_qk_rope_args args = {sequence, query_heads, kv_heads, head_dim,
                                 epsilon};
    dim3 threads(head_dim, 1, 1);
    dim3 blocks(1, query_heads, sequence);
    h3_text_qk_rope_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)query_input->device,
        (const uint16_t *)key_input->device,
        (const uint16_t *)q_norm->device, (const uint16_t *)k_norm->device,
        (const uint16_t *)rope_cos->device, (const uint16_t *)rope_sin->device,
        (uint16_t *)query_output->device, (uint16_t *)key_output->device,
        args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_gpu_text_qk_rope_bf16");
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

struct h3_token_pool_args {
    uint32_t input_offset;
    uint32_t original_offset;
    uint32_t baseline_offset;
    uint32_t rows;
    uint32_t width;
};

__global__ static void h3_token_pool_bf16_kernel(
    const uint16_t *input, const uint32_t *pairs, uint16_t *output,
    uint16_t *baseline, const uint32_t *baseline_indices, uint16_t *original,
    h3_token_pool_args args) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t first_row = pairs[row * 2u];
    uint32_t second_row = pairs[row * 2u + 1u];
    uint16_t first =
        input[args.input_offset + first_row * args.width + column];
    original[args.original_offset + first_row * args.width + column] = first;
    uint16_t pooled = first;
    if (first_row != second_row) {
        uint16_t second =
            input[args.input_offset + second_row * args.width + column];
        original[args.original_offset + second_row * args.width + column] =
            second;
        pooled = h3_f32_to_bf16_bits(
            (h3_bf16_bits_to_f32(first) + h3_bf16_bits_to_f32(second)) *
            0.5f);
    }
    output[row * args.width + column] = pooled;
    uint32_t baseline_index = baseline_indices[row];
    if (baseline_index != 0xffffffffu) {
        baseline[args.baseline_offset + baseline_index * args.width + column] =
            pooled;
    }
}

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!gpu || !output || !input || !original || !baseline ||
        !baseline_indices || !pairs || output->dtype != H3_GPU_BF16 ||
        input->dtype != H3_GPU_BF16 || original->dtype != H3_GPU_BF16 ||
        baseline->dtype != H3_GPU_BF16 ||
        baseline_indices->dtype != H3_GPU_U32 ||
        pairs->dtype != H3_GPU_U32 || !input_rows || !rows ||
        rows > input_rows || baseline_rows > rows || !width ||
        output->elements < elements ||
        input->elements < input_offset + input_elements ||
        original->elements < original_offset + input_elements ||
        baseline->elements < baseline_offset + baseline_elements ||
        baseline_indices->elements < rows || pairs->elements < (size_t)rows * 2)
        return h3_gpu_fail(gpu, "invalid token pool request");
    h3_token_pool_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width};
    dim3 threads(256, 1, 1);
    dim3 blocks((width + threads.x - 1) / threads.x, rows, 1);
    h3_token_pool_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)input->device, (const uint32_t *)pairs->device,
        (uint16_t *)output->device, (uint16_t *)baseline->device,
        (const uint32_t *)baseline_indices->device,
        (uint16_t *)original->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_token_pool_bf16");
}

struct h3_token_pool_adaln_args {
    uint32_t input_offset;
    uint32_t original_offset;
    uint32_t baseline_offset;
    uint32_t rows;
    uint32_t width;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    float epsilon;
};

__global__ static void h3_token_pool_adaln_bf16_kernel(
    const uint16_t *input, const uint32_t *pairs, uint16_t *residual,
    uint16_t *baseline, const uint32_t *baseline_indices, uint16_t *original,
    const uint16_t *weight, const uint16_t *modulation, const uint32_t *row_map,
    uint16_t *output, h3_token_pool_adaln_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    extern __shared__ unsigned char shared_raw[];
    float *reductions = (float *)shared_raw;
    uint16_t *pooled_values =
        (uint16_t *)(shared_raw + threads * sizeof(float));
    uint32_t first_row = pairs[row * 2u];
    uint32_t second_row = pairs[row * 2u + 1u];
    uint32_t baseline_index = baseline_indices[row];
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        uint16_t first =
            input[args.input_offset + first_row * args.width + column];
        original[args.original_offset + first_row * args.width + column] =
            first;
        uint16_t pooled = first;
        if (first_row != second_row) {
            uint16_t second =
                input[args.input_offset + second_row * args.width + column];
            original[args.original_offset + second_row * args.width + column] =
                second;
            pooled = h3_f32_to_bf16_bits(
                (h3_bf16_bits_to_f32(first) + h3_bf16_bits_to_f32(second)) *
                0.5f);
        }
        uint32_t destination = row * args.width + column;
        residual[destination] = pooled;
        pooled_values[column] = pooled;
        if (baseline_index != 0xffffffffu) {
            baseline[args.baseline_offset + baseline_index * args.width +
                     column] = pooled;
        }
        float value = h3_bf16_bits_to_f32(pooled);
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
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_bits_to_f32(pooled_values[column]) *
                           inverse * h3_bf16_bits_to_f32(weight[column]);
        float shift = h3_bf16_bits_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_bits_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] = h3_f32_to_bf16_bits(
            normalized * (1.0f + scale) + shift);
    }
}

int h3_gpu_token_pool_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *input, size_t input_offset, h3_gpu_tensor *original,
    size_t original_offset, h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
    const h3_gpu_tensor *row_map, uint32_t input_rows, uint32_t rows,
    uint32_t baseline_rows, uint32_t width, uint32_t slots,
    uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!gpu || !residual || !output || !input || !original || !baseline ||
        !baseline_indices || !pairs || !norm_weight || !modulation ||
        !row_map || width > 5376u || !input_rows || !rows || rows > input_rows ||
        baseline_rows > rows || shift_slot >= slots || scale_slot >= slots ||
        residual->dtype != H3_GPU_BF16 || output->dtype != H3_GPU_BF16 ||
        input->dtype != H3_GPU_BF16 || original->dtype != H3_GPU_BF16 ||
        baseline->dtype != H3_GPU_BF16 ||
        norm_weight->dtype != H3_GPU_BF16 ||
        modulation->dtype != H3_GPU_BF16 ||
        baseline_indices->dtype != H3_GPU_U32 ||
        pairs->dtype != H3_GPU_U32 || row_map->dtype != H3_GPU_U32 ||
        residual->elements < elements || output->elements < elements ||
        input->elements < input_offset + input_elements ||
        original->elements < original_offset + input_elements ||
        baseline->elements < baseline_offset + baseline_elements ||
        norm_weight->elements < width || row_map->elements < rows ||
        pairs->elements < (size_t)rows * 2 ||
        baseline_indices->elements < rows)
        return h3_gpu_fail(gpu, "invalid fused token pool AdaLN request");
    h3_token_pool_adaln_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width, slots,
        shift_slot, scale_slot, epsilon};
    unsigned threads = 256;
    size_t shared_bytes =
        threads * sizeof(float) + (size_t)width * sizeof(uint16_t);
    h3_token_pool_adaln_bf16_kernel<<<rows, threads, shared_bytes,
                                      gpu->stream>>>(
        (const uint16_t *)input->device, (const uint32_t *)pairs->device,
        (uint16_t *)residual->device, (uint16_t *)baseline->device,
        (const uint32_t *)baseline_indices->device,
        (uint16_t *)original->device, (const uint16_t *)norm_weight->device,
        (const uint16_t *)modulation->device,
        (const uint32_t *)row_map->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_token_pool_adaln_bf16");
}

struct h3_token_expand_args {
    uint32_t original_offset;
    uint32_t baseline_offset;
    uint32_t rows;
    uint32_t width;
    uint32_t exact_prefix_rows;
    float update_scale;
};

__global__ static void h3_token_expand_delta_bf16_kernel(
    const uint16_t *original, const uint16_t *reduced,
    const uint16_t *baseline, const uint32_t *baseline_indices,
    const uint32_t *parents, uint16_t *output, h3_token_expand_args args) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t parent = parents[row];
    uint32_t destination = row * args.width + column;
    uint32_t reduced_index = parent * args.width + column;
    if (row < args.exact_prefix_rows) {
        output[destination] = reduced[reduced_index];
        return;
    }
    uint32_t baseline_row = baseline_indices[parent];
    if (baseline_row == 0xffffffffu) {
        output[destination] = reduced[reduced_index];
        return;
    }
    uint32_t baseline_index =
        args.baseline_offset + baseline_row * args.width + column;
    float update = h3_bf16_bits_to_f32(reduced[reduced_index]) -
                   h3_bf16_bits_to_f32(baseline[baseline_index]);
    output[destination] = h3_f32_to_bf16_bits(
        h3_bf16_bits_to_f32(original[args.original_offset + destination]) +
        args.update_scale * update);
}

int h3_gpu_token_expand_delta_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *original,
    size_t original_offset, const h3_gpu_tensor *reduced,
    const h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
    uint32_t width, uint32_t exact_prefix_rows, float update_scale) {
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!gpu || !output || !original || !reduced || !baseline ||
        !baseline_indices || !parents || output->dtype != H3_GPU_BF16 ||
        original->dtype != H3_GPU_BF16 || reduced->dtype != H3_GPU_BF16 ||
        baseline->dtype != H3_GPU_BF16 ||
        baseline_indices->dtype != H3_GPU_U32 ||
        parents->dtype != H3_GPU_U32 || !rows || !reduced_rows || !width ||
        exact_prefix_rows > rows || output->elements < elements ||
        original->elements < original_offset + elements ||
        reduced->elements < reduced_elements ||
        baseline->elements < baseline_offset + baseline_elements ||
        baseline_indices->elements < reduced_rows || parents->elements < rows)
        return h3_gpu_fail(gpu, "invalid token expand delta request");
    h3_token_expand_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, update_scale};
    dim3 threads(256, 1, 1);
    dim3 blocks((width + threads.x - 1) / threads.x, rows, 1);
    h3_token_expand_delta_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)original->device, (const uint16_t *)reduced->device,
        (const uint16_t *)baseline->device,
        (const uint32_t *)baseline_indices->device,
        (const uint32_t *)parents->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_token_expand_delta_bf16");
}

struct h3_token_expand_adaln_args {
    uint32_t original_offset;
    uint32_t baseline_offset;
    uint32_t rows;
    uint32_t width;
    uint32_t exact_prefix_rows;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    float update_scale;
    float epsilon;
};

__global__ static void h3_token_expand_adaln_bf16_kernel(
    const uint16_t *original, const uint16_t *reduced,
    const uint16_t *baseline, const uint32_t *baseline_indices,
    const uint32_t *parents, uint16_t *residual, const uint16_t *weight,
    const uint16_t *modulation, const uint32_t *row_map, uint16_t *output,
    h3_token_expand_adaln_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    extern __shared__ unsigned char shared_raw[];
    float *reductions = (float *)shared_raw;
    uint16_t *restored_values =
        (uint16_t *)(shared_raw + threads * sizeof(float));
    uint32_t parent = parents[row];
    uint32_t baseline_row = baseline_indices[parent];
    bool direct = row < args.exact_prefix_rows ||
                    baseline_row == 0xffffffffu;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        uint32_t destination = row * args.width + column;
        uint32_t reduced_index = parent * args.width + column;
        uint16_t restored = reduced[reduced_index];
        if (!direct) {
            uint32_t baseline_index =
                args.baseline_offset + baseline_row * args.width + column;
            float update = h3_bf16_bits_to_f32(restored) -
                           h3_bf16_bits_to_f32(baseline[baseline_index]);
            restored = h3_f32_to_bf16_bits(
                h3_bf16_bits_to_f32(
                    original[args.original_offset + destination]) +
                args.update_scale * update);
        }
        restored_values[column] = restored;
        residual[destination] = restored;
        float value = h3_bf16_bits_to_f32(restored);
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
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_bits_to_f32(restored_values[column]) *
                           inverse * h3_bf16_bits_to_f32(weight[column]);
        float shift = h3_bf16_bits_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_bits_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] = h3_f32_to_bf16_bits(
            normalized * (1.0f + scale) + shift);
    }
}

int h3_gpu_token_expand_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *original, size_t original_offset,
    const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline,
    size_t baseline_offset, const h3_gpu_tensor *baseline_indices,
    const h3_gpu_tensor *parents, const h3_gpu_tensor *norm_weight,
    const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
    uint32_t width, uint32_t exact_prefix_rows, float update_scale,
    uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
    float epsilon) {
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!gpu || !residual || !output || !original || !reduced || !baseline ||
        !baseline_indices || !parents || !norm_weight || !modulation ||
        !row_map || width > 5376u || !rows || !reduced_rows || !width ||
        exact_prefix_rows > rows || shift_slot >= slots ||
        scale_slot >= slots || residual->dtype != H3_GPU_BF16 ||
        output->dtype != H3_GPU_BF16 || original->dtype != H3_GPU_BF16 ||
        reduced->dtype != H3_GPU_BF16 || baseline->dtype != H3_GPU_BF16 ||
        norm_weight->dtype != H3_GPU_BF16 ||
        modulation->dtype != H3_GPU_BF16 ||
        baseline_indices->dtype != H3_GPU_U32 ||
        parents->dtype != H3_GPU_U32 || row_map->dtype != H3_GPU_U32 ||
        residual->elements < elements || output->elements < elements ||
        original->elements < original_offset + elements ||
        reduced->elements < reduced_elements ||
        baseline->elements < baseline_offset + baseline_elements ||
        norm_weight->elements < width || row_map->elements < rows ||
        baseline_indices->elements < reduced_rows || parents->elements < rows)
        return h3_gpu_fail(gpu, "invalid fused token expand AdaLN request");
    h3_token_expand_adaln_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, slots, shift_slot, scale_slot,
        update_scale, epsilon};
    unsigned threads = 256;
    size_t shared_bytes =
        threads * sizeof(float) + (size_t)width * sizeof(uint16_t);
    h3_token_expand_adaln_bf16_kernel<<<rows, threads, shared_bytes,
                                        gpu->stream>>>(
        (const uint16_t *)original->device, (const uint16_t *)reduced->device,
        (const uint16_t *)baseline->device,
        (const uint32_t *)baseline_indices->device,
        (const uint32_t *)parents->device, (uint16_t *)residual->device,
        (const uint16_t *)norm_weight->device,
        (const uint16_t *)modulation->device,
        (const uint32_t *)row_map->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_token_expand_adaln_bf16");
}

struct h3_linear_tiled_args {
    uint32_t rows;
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t has_bias;
    uint32_t input_offset;
    uint32_t output_offset;
};

__global__ static void h3_linear_f32_tiled_bf16_kernel(
    const float *input, const float *weight, const float *bias,
    uint16_t *output, h3_linear_tiled_args args) {
    __shared__ float input_tile[16][16];
    __shared__ float weight_tile[16][16];
    uint32_t row = blockIdx.y * 16u + threadIdx.y;
    uint32_t column = blockIdx.x * 16u + threadIdx.x;
    float sum = args.has_bias && column < args.output_dim ?
                    bias[column] : 0.0f;
    uint32_t tile_count = (args.input_dim + 15u) / 16u;
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t input_k = tile * 16u + threadIdx.x;
        input_tile[threadIdx.y][threadIdx.x] =
            row < args.rows && input_k < args.input_dim ?
                input[args.input_offset + row * args.input_dim + input_k] :
                0.0f;
        uint32_t weight_k = tile * 16u + threadIdx.y;
        weight_tile[threadIdx.y][threadIdx.x] =
            column < args.output_dim && weight_k < args.input_dim ?
                weight[(size_t)column * args.input_dim + weight_k] : 0.0f;
        __syncthreads();
        for (uint32_t k = 0; k < 16u; k++) {
            sum = fmaf(input_tile[threadIdx.y][k], weight_tile[k][threadIdx.x],
                       sum);
        }
        __syncthreads();
    }
    if (row < args.rows && column < args.output_dim) {
        output[args.output_offset + row * args.output_dim + column] =
            h3_f32_to_bf16_bits(sum);
    }
}

__global__ static void h3_linear_f32_tiled_bf16_map_kernel(
    const float *input, const float *weight, const float *bias,
    uint16_t *output, const uint32_t *row_map, h3_linear_tiled_args args) {
    __shared__ float input_tile[16][16];
    __shared__ float weight_tile[16][16];
    __shared__ uint32_t output_rows[16];
    uint32_t row = blockIdx.y * 16u + threadIdx.y;
    uint32_t column = blockIdx.x * 16u + threadIdx.x;
    if (threadIdx.x == 0) {
        output_rows[threadIdx.y] = row < args.rows ? row_map[row] : 0u;
    }
    __syncthreads();
    float sum = args.has_bias && column < args.output_dim ?
                    bias[column] : 0.0f;
    uint32_t tile_count = (args.input_dim + 15u) / 16u;
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t input_k = tile * 16u + threadIdx.x;
        input_tile[threadIdx.y][threadIdx.x] =
            row < args.rows && input_k < args.input_dim ?
                input[row * args.input_dim + input_k] : 0.0f;
        uint32_t weight_k = tile * 16u + threadIdx.y;
        weight_tile[threadIdx.y][threadIdx.x] =
            column < args.output_dim && weight_k < args.input_dim ?
                weight[(size_t)column * args.input_dim + weight_k] : 0.0f;
        __syncthreads();
        for (uint32_t k = 0; k < 16u; k++) {
            sum = fmaf(input_tile[threadIdx.y][k], weight_tile[k][threadIdx.x],
                       sum);
        }
        __syncthreads();
    }
    if (row < args.rows && column < args.output_dim) {
        output[output_rows[threadIdx.y] * args.output_dim + column] =
            h3_f32_to_bf16_bits(sum);
    }
}

static int h3_patch_shape_ok(uint32_t input_dim, uint32_t output_dim) {
    return output_dim == 5376u && (input_dim == 32u || input_dim == 96u);
}

static int h3_gpu_patch_linear_bf16_offset_impl(
    h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset,
    const h3_gpu_tensor *input, size_t input_offset,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows,
    uint32_t input_dim, uint32_t output_dim) {
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!gpu || !output || !input || !weight ||
        output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_F32 ||
        weight->dtype != H3_GPU_F32 ||
        (bias && bias->dtype != H3_GPU_F32) ||
        !h3_patch_shape_ok(input_dim, output_dim) || !rows ||
        output->elements < output_offset + output_count ||
        input->elements < input_offset + input_count ||
        weight->elements < weight_count ||
        (bias && bias->elements < output_dim))
        return h3_gpu_fail(gpu, "invalid patch linear request");
    h3_linear_tiled_args args = {rows, input_dim, output_dim,
                                 bias ? 1u : 0u, (uint32_t)input_offset,
                                 (uint32_t)output_offset};
    dim3 threads(16, 16, 1);
    dim3 blocks((output_dim + 15u) / 16u, (rows + 15u) / 16u, 1);
    h3_linear_f32_tiled_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (const float *)weight->device,
        bias ? (const float *)bias->device : (const float *)input->device,
        (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_patch_linear_bf16");
}

int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset_impl(
        gpu, output, 0, input, 0, weight, bias, rows, input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16_offset(
    h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset,
    const h3_gpu_tensor *input, size_t input_offset,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows,
    uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset_impl(
        gpu, output, output_offset, input, input_offset, weight, bias, rows,
        input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16_map(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
    const h3_gpu_tensor *row_map, uint32_t output_rows, uint32_t rows,
    uint32_t input_dim, uint32_t output_dim) {
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)output_rows * output_dim;
    if (!gpu || !output || !input || !weight || !row_map ||
        output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_F32 ||
        weight->dtype != H3_GPU_F32 ||
        (bias && bias->dtype != H3_GPU_F32) ||
        row_map->dtype != H3_GPU_U32 ||
        !h3_patch_shape_ok(input_dim, output_dim) || !rows ||
        output->elements < output_count || input->elements < input_count ||
        weight->elements < weight_count ||
        row_map->elements < rows || (bias && bias->elements < output_dim))
        return h3_gpu_fail(gpu, "invalid mapped patch linear request");
    h3_linear_tiled_args args = {rows, input_dim, output_dim,
                                 bias ? 1u : 0u, 0u, 0u};
    dim3 threads(16, 16, 1);
    dim3 blocks((output_dim + 15u) / 16u, (rows + 15u) / 16u, 1);
    h3_linear_f32_tiled_bf16_map_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (const float *)weight->device,
        bias ? (const float *)bias->device : (const float *)input->device,
        (uint16_t *)output->device, (const uint32_t *)row_map->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_patch_linear_bf16_map");
}

struct h3_gate_adaln_args {
    uint32_t rows;
    uint32_t width;
    uint32_t slots;
    uint32_t gate_slot;
    uint32_t shift_slot;
    uint32_t scale_slot;
    float epsilon;
};

__global__ static void h3_gate_adaln_bf16_kernel(
    const uint16_t *residual, const uint16_t *branch,
    const uint16_t *gate_modulation, const uint32_t *row_map,
    const uint16_t *weight, const uint16_t *norm_modulation,
    uint16_t *gated_residual, uint16_t *output, h3_gate_adaln_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    extern __shared__ unsigned char shared_raw[];
    float *reductions = (float *)shared_raw;
    uint16_t *gated_values =
        (uint16_t *)(shared_raw + threads * sizeof(float));
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        size_t index = (size_t)row * args.width + column;
        float gate = h3_bf16_bits_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        uint16_t gated = h3_f32_to_bf16_bits(
            h3_bf16_bits_to_f32(residual[index]) +
            h3_bf16_bits_to_f32(branch[index]) * gate);
        gated_residual[index] = gated;
        gated_values[column] = gated;
        float value = h3_bf16_bits_to_f32(gated);
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
        float normalized = h3_bf16_bits_to_f32(gated_values[column]) *
                           inverse * h3_bf16_bits_to_f32(weight[column]);
        float shift = h3_bf16_bits_to_f32(
            norm_modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_bits_to_f32(
            norm_modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] = h3_f32_to_bf16_bits(
            normalized * (1.0f + scale) + shift);
    }
}

int h3_gpu_gate_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation,
    const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot,
    uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    size_t elements = (size_t)rows * width;
    if (!gpu || !gated_residual || !output || !residual || !branch ||
        !norm_weight || !gate_modulation || !norm_modulation || !row_map ||
        width > 5376u || gate_slot >= slots || shift_slot >= slots ||
        scale_slot >= slots || !rows || !width ||
        gated_residual->dtype != H3_GPU_BF16 || output->dtype != H3_GPU_BF16 ||
        residual->dtype != H3_GPU_BF16 || branch->dtype != H3_GPU_BF16 ||
        norm_weight->dtype != H3_GPU_BF16 ||
        gate_modulation->dtype != H3_GPU_BF16 ||
        norm_modulation->dtype != H3_GPU_BF16 ||
        row_map->dtype != H3_GPU_U32 || gated_residual->elements < elements ||
        output->elements < elements || residual->elements < elements ||
        branch->elements < elements || norm_weight->elements < width ||
        row_map->elements < rows)
        return h3_gpu_fail(gpu, "invalid fused gate AdaLN request");
    h3_gate_adaln_args args = {rows, width, slots, gate_slot, shift_slot,
                               scale_slot, epsilon};
    unsigned threads = 256;
    size_t shared_bytes =
        threads * sizeof(float) + (size_t)width * sizeof(uint16_t);
    h3_gate_adaln_bf16_kernel<<<rows, threads, shared_bytes, gpu->stream>>>(
        (const uint16_t *)residual->device, (const uint16_t *)branch->device,
        (const uint16_t *)gate_modulation->device,
        (const uint32_t *)row_map->device, (const uint16_t *)norm_weight->device,
        (const uint16_t *)norm_modulation->device,
        (uint16_t *)gated_residual->device, (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_gate_adaln_bf16");
}

struct h3_rms_inverse_args {
    uint32_t rows;
    uint32_t width;
    float epsilon;
    size_t input_offset;
};

__global__ static void h3_rms_inverse_bf16_kernel(
    const uint16_t *input, float *inverse, h3_rms_inverse_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    extern __shared__ float reductions[];
    const uint16_t *row_input =
        input + args.input_offset + (size_t)row * args.width;
    float local_sum = 0.0f;
    for (uint32_t k = tid; k < args.width; k += threads) {
        float value = h3_bf16_bits_to_f32(row_input[k]);
        local_sum = fmaf(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        inverse[row] =
            rsqrtf(reductions[0] / (float)args.width + args.epsilon);
    }
}

struct h3_adaln_linear_args {
    uint32_t rows;
    uint32_t width;
    uint32_t output_dim;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    uint32_t has_bias;
    size_t input_offset;
};

__global__ static void h3_adaln_linear_bf16_kernel(
    const uint16_t *input, const float *inverse, const uint16_t *norm_weight,
    const uint16_t *modulation, const uint32_t *row_map,
    const uint16_t *weight, const uint16_t *bias, uint16_t *output,
    h3_adaln_linear_args args) {
    __shared__ uint16_t input_tile[16][16];
    __shared__ float weight_tile[16][16];
    uint32_t row = blockIdx.y * 16u + threadIdx.y;
    uint32_t column = blockIdx.x * 16u + threadIdx.x;
    float sum = args.has_bias && column < args.output_dim ?
                    h3_bf16_bits_to_f32(bias[column]) : 0.0f;
    size_t base = row < args.rows ?
                      (size_t)row_map[row] * args.slots * args.width : 0;
    uint32_t tile_count = (args.width + 15u) / 16u;
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t input_k = tile * 16u + threadIdx.x;
        uint16_t normalized = 0;
        if (row < args.rows && input_k < args.width) {
            float value = h3_bf16_bits_to_f32(
                input[args.input_offset + row * args.width + input_k]);
            float shift = h3_bf16_bits_to_f32(
                modulation[base + args.shift_slot * args.width + input_k]);
            float scale = h3_bf16_bits_to_f32(
                modulation[base + args.scale_slot * args.width + input_k]);
            float normed = value * inverse[row] *
                             h3_bf16_bits_to_f32(norm_weight[input_k]);
            normalized = h3_f32_to_bf16_bits(normed * (1.0f + scale) + shift);
        }
        input_tile[threadIdx.y][threadIdx.x] = normalized;
        uint32_t weight_k = tile * 16u + threadIdx.y;
        weight_tile[threadIdx.y][threadIdx.x] =
            column < args.output_dim && weight_k < args.width ?
                h3_bf16_bits_to_f32(
                    weight[(size_t)column * args.width + weight_k]) :
                0.0f;
        __syncthreads();
        for (uint32_t k = 0; k < 16u; k++) {
            sum = fmaf(h3_bf16_bits_to_f32(input_tile[threadIdx.y][k]),
                       weight_tile[k][threadIdx.x], sum);
        }
        __syncthreads();
    }
    if (row < args.rows && column < args.output_dim) {
        output[row * args.output_dim + column] = h3_f32_to_bf16_bits(sum);
    }
}

int h3_gpu_adaln_linear_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *inverse,
    const h3_gpu_tensor *input, size_t input_offset,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
    const h3_gpu_tensor *row_map, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *bias, uint32_t rows, uint32_t width,
    uint32_t output_dim, uint32_t slots, uint32_t shift_slot,
    uint32_t scale_slot, float epsilon) {
    size_t input_count = (size_t)rows * width;
    size_t weight_count = (size_t)output_dim * width;
    size_t output_count = (size_t)rows * output_dim;
    if (!gpu || !output || !inverse || !input || !norm_weight ||
        !modulation || !row_map || !weight ||
        shift_slot >= slots || scale_slot >= slots || !rows || !width ||
        output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 ||
        inverse->dtype != H3_GPU_F32 || norm_weight->dtype != H3_GPU_BF16 ||
        modulation->dtype != H3_GPU_BF16 || weight->dtype != H3_GPU_BF16 ||
        row_map->dtype != H3_GPU_U32 ||
        (bias && bias->dtype != H3_GPU_BF16) ||
        output->elements < output_count ||
        input->elements < input_offset + input_count ||
        inverse->elements < rows || norm_weight->elements < width ||
        row_map->elements < rows || weight->elements < weight_count ||
        (bias && bias->elements < output_dim))
        return h3_gpu_fail(gpu, "invalid fused AdaLN linear request");
    h3_rms_inverse_args inv_args = {rows, width, epsilon, input_offset};
    unsigned inv_threads = 256;
    h3_rms_inverse_bf16_kernel<<<rows, inv_threads,
                                 inv_threads * sizeof(float),
                                 gpu->stream>>>(
        (const uint16_t *)input->device, (float *)inverse->device, inv_args);
    gpu->stats.direct_dispatches++;
    if (!h3_cuda_check(gpu, cudaGetLastError(), "h3_rms_inverse_bf16"))
        return 0;
    h3_adaln_linear_args args = {rows, width, output_dim, slots, shift_slot,
                                 scale_slot, bias ? 1u : 0u, input_offset};
    dim3 threads(16, 16, 1);
    dim3 blocks((output_dim + 15u) / 16u, (rows + 15u) / 16u, 1);
    h3_adaln_linear_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)input->device, (const float *)inverse->device,
        (const uint16_t *)norm_weight->device,
        (const uint16_t *)modulation->device,
        (const uint32_t *)row_map->device, (const uint16_t *)weight->device,
        bias ? (const uint16_t *)bias->device :
               (const uint16_t *)input->device,
        (uint16_t *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_adaln_linear_bf16");
}

struct h3_scale_add_args {
    uint32_t rows;
    uint32_t width;
};

__global__ static void h3_scale_add_f32_kernel(const float *residual,
                                               const float *branch,
                                               const float *scale, float *output,
                                               h3_scale_add_args args) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.y;
    if (row >= args.rows || column >= args.width) return;
    size_t index = (size_t)row * args.width + column;
    output[index] = residual[index] + branch[index] * scale[column];
}

int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    size_t count = (size_t)rows * width;
    if (!gpu || !output || !residual || !branch || !scale ||
        output->dtype != H3_GPU_F32 || residual->dtype != H3_GPU_F32 ||
        branch->dtype != H3_GPU_F32 || scale->dtype != H3_GPU_F32 ||
        output->elements < count || residual->elements < count ||
        branch->elements < count || scale->elements < width || !rows || !width)
        return h3_gpu_fail(gpu, "invalid scale-add request");
    h3_scale_add_args args = {rows, width};
    dim3 threads(256, 1, 1);
    dim3 blocks((width + 255u) / 256u, rows, 1);
    h3_scale_add_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)residual->device, (const float *)branch->device,
        (const float *)scale->device, (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_scale_add_f32");
}

__global__ static void h3_layer_norm_f32_kernel(const float *input,
                                                const float *weight,
                                                const float *bias, float *output,
                                                h3_rms_norm_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;

    extern __shared__ float reductions[];
    const float *row_input = input + (size_t)row * args.width;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads)
        local_sum += row_input[column];
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float mean = reductions[0] / (float)args.width;
    __syncthreads();
    float local_square = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float centered = row_input[column] - mean;
        local_square = fmaf(centered, centered, local_square);
    }
    reductions[tid] = local_square;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse =
        rsqrtf(reductions[0] / (float)args.width + args.epsilon);
    for (uint32_t column = tid; column < args.width; column += threads) {
        output[(size_t)row * args.width + column] =
            (row_input[column] - mean) * inverse * weight[column] +
            bias[column];
    }
}

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    size_t count = (size_t)rows * width;
    if (!gpu || !output || !input || !weight || !bias ||
        output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
        weight->dtype != H3_GPU_F32 || bias->dtype != H3_GPU_F32 ||
        output->elements < count || input->elements < count ||
        weight->elements < width || bias->elements < width || !rows || !width)
        return h3_gpu_fail(gpu, "invalid F32 LayerNorm request");
    h3_rms_norm_args args = {rows, width, epsilon};
    unsigned threads = 256;
    h3_layer_norm_f32_kernel<<<rows, threads, threads * sizeof(float),
                                 gpu->stream>>>(
        (const float *)input->device, (const float *)weight->device,
        (const float *)bias->device, (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_layer_norm_f32");
}

__global__ static void h3_swiglu_f32_kernel(const float *fused, float *output,
                                            h3_swiglu_args args) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.y;
    if (row >= args.rows || column >= args.width) return;
    size_t base = (size_t)row * args.width + column;
    size_t fused_base = (size_t)row * args.width * 2u;
    float gate = fused[fused_base + column];
    float up = fused[fused_base + args.width + column];
    output[base] = gate / (1.0f + expf(-gate)) * up;
}

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    size_t fused_count = (size_t)rows * width * 2u;
    size_t output_count = (size_t)rows * width;
    if (!gpu || !output || !fused || output->dtype != H3_GPU_F32 ||
        fused->dtype != H3_GPU_F32 || output->elements < output_count ||
        fused->elements < fused_count || !rows || !width)
        return h3_gpu_fail(gpu, "invalid F32 SwiGLU request");
    h3_swiglu_args args = {rows, width};
    dim3 threads(256, 1, 1);
    dim3 blocks((width + threads.x - 1) / threads.x, rows, 1);
    h3_swiglu_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)fused->device, (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_swiglu_f32");
}

__global__ static void h3_geglu_f32_kernel(const float *gate, const float *linear,
                                         float *output, uint32_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float x = gate[index];
    float cube = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * cube);
    float gelu = 0.5f * x * (1.0f + tanhf(inner));
    output[index] = gelu * linear[index];
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate, const h3_gpu_tensor *linear,
                     uint32_t elements) {
    if (!gpu || !output || !gate || !linear ||
        output->dtype != H3_GPU_F32 || gate->dtype != H3_GPU_F32 ||
        linear->dtype != H3_GPU_F32 || output->elements < elements ||
        gate->elements < elements || linear->elements < elements || !elements)
        return h3_gpu_fail(gpu, "invalid GeGLU request");
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_geglu_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)gate->device, (const float *)linear->device,
        (float *)output->device, elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_geglu_f32");
}

struct h3_add_scaled_args {
    uint32_t elements;
    float left_scale;
    float right_scale;
};

__global__ static void h3_add_scaled_f32_kernel(const float *left,
                                                const float *right,
                                                float *output,
                                                h3_add_scaled_args args) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= args.elements) return;
    output[index] =
        left[index] * args.left_scale + right[index] * args.right_scale;
}

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    if (!gpu || !output || !left || !right ||
        output->dtype != H3_GPU_F32 || left->dtype != H3_GPU_F32 ||
        right->dtype != H3_GPU_F32 || output->elements < elements ||
        left->elements < elements || right->elements < elements || !elements)
        return h3_gpu_fail(gpu, "invalid add-scaled request");
    h3_add_scaled_args args = {elements, left_scale, right_scale};
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_add_scaled_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)left->device, (const float *)right->device,
        (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_add_scaled_f32");
}

struct h3_clip_args {
    uint32_t elements;
    float minimum;
    float maximum;
};

__global__ static void h3_clip_f32_kernel(const float *input, float *output,
                                          h3_clip_args args) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= args.elements) return;
    float value = input[index];
    if (value < args.minimum) value = args.minimum;
    if (value > args.maximum) value = args.maximum;
    output[index] = value;
}

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    if (!gpu || !output || !input || output->dtype != H3_GPU_F32 ||
        input->dtype != H3_GPU_F32 || output->elements < elements ||
        input->elements < elements || !elements || minimum > maximum)
        return h3_gpu_fail(gpu, "invalid clip request");
    h3_clip_args args = {elements, minimum, maximum};
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)(((size_t)elements + threads - 1) / threads);
    h3_clip_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_clip_f32");
}

__global__ static void h3_rms_norm_f32_kernel(const float *input,
                                              const float *weight, float *output,
                                              h3_rms_norm_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;

    extern __shared__ float reductions[];
    const float *row_input = input + (size_t)row * args.width;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float value = row_input[column];
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
    for (uint32_t column = tid; column < args.width; column += threads)
        output[(size_t)row * args.width + column] =
            row_input[column] * inverse * weight[column];
}

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    size_t count = (size_t)rows * width;
    if (!gpu || !output || !input || !weight ||
        output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
        weight->dtype != H3_GPU_F32 || output->elements < count ||
        input->elements < count || weight->elements < width || !rows || !width)
        return h3_gpu_fail(gpu, "invalid F32 RMSNorm request");
    h3_rms_norm_args args = {rows, width, epsilon};
    unsigned threads = 256;
    h3_rms_norm_f32_kernel<<<rows, threads, threads * sizeof(float),
                             gpu->stream>>>(
        (const float *)input->device, (const float *)weight->device,
        (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_rms_norm_f32");
}

struct h3_video_qkv_rope_args {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t rope_half;
    float epsilon;
};

__global__ static void h3_video_qkv_rope_f32_kernel(
    const float *qkv, const float *rope_cos, const float *rope_sin,
    float *query, float *key, float *value, h3_video_qkv_rope_args args) {
    uint32_t dimension = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t row = (uint32_t)blockIdx.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence)
        return;
    size_t base =
        ((size_t)row * args.heads + head) * args.head_dim * 3u;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float q = qkv[base + d];
        float k = qkv[base + args.head_dim + d];
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float qi = rsqrtf(q_sum / (float)args.head_dim + args.epsilon);
    float ki = rsqrtf(k_sum / (float)args.head_dim + args.epsilon);
    float q0 = qkv[base + dimension] * qi;
    float k0 = qkv[base + args.head_dim + dimension] * ki;
    if (dimension < args.rope_half) {
        uint32_t pair = dimension + args.rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2u) {
        uint32_t pair = dimension - args.rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    size_t output_index =
        ((size_t)row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[base + args.head_dim * 2u + dimension];
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    size_t out_count = (size_t)sequence * heads * head_dim;
    size_t qkv_count = out_count * 3u;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!gpu || !query || !key || !value || !qkv || !rope_cos || !rope_sin ||
        query->dtype != H3_GPU_F32 || key->dtype != H3_GPU_F32 ||
        value->dtype != H3_GPU_F32 || qkv->dtype != H3_GPU_F32 ||
        rope_cos->dtype != H3_GPU_F32 || rope_sin->dtype != H3_GPU_F32 ||
        query->elements < out_count || key->elements < out_count ||
        value->elements < out_count || qkv->elements < qkv_count ||
        rope_cos->elements < rope_count || rope_sin->elements < rope_count ||
        !sequence || !heads || !head_dim || !rope_half ||
        rope_half * 2u > head_dim)
        return h3_gpu_fail(gpu, "invalid video QKV/RoPE request");
    h3_video_qkv_rope_args args = {sequence, heads, head_dim, rope_half,
                                   epsilon};
    dim3 threads(32, 1, 1);
    dim3 blocks((head_dim + threads.x - 1) / threads.x, heads, sequence);
    h3_video_qkv_rope_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)qkv->device, (const float *)rope_cos->device,
        (const float *)rope_sin->device, (float *)query->device,
        (float *)key->device, (float *)value->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_video_qkv_rope_f32");
}

__global__ static void h3_sdpa_f32_kernel(const float *query, const float *key,
                                          const float *value, float *output,
                                          h3_sdpa_args args) {
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
            size_t k_base =
                ((size_t)k_row * args.heads + head) * args.head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < args.head_dim; d++)
                dot = fmaf(query[q_base + d], key[k_base + d], dot);
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
        accumulated = fmaf(scores[k_row], value[v_base + dim], accumulated);
    }
    output[q_base + dim] = accumulated;
}

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    size_t count = (size_t)sequence * heads * head_dim;
    if (!gpu || !output || !query || !key || !value ||
        output->dtype != H3_GPU_F32 || query->dtype != H3_GPU_F32 ||
        key->dtype != H3_GPU_F32 || value->dtype != H3_GPU_F32 ||
        output->elements < count || query->elements < count ||
        key->elements < count || value->elements < count || !sequence ||
        !heads || !head_dim)
        return h3_gpu_fail(gpu, "invalid F32 SDPA request");
    h3_sdpa_args args = {sequence, heads, head_dim, scale, 0u};
    dim3 threads(head_dim, 1, 1);
    dim3 blocks(heads, sequence, 1);
    size_t shared_bytes = (size_t)sequence * sizeof(float);
    h3_sdpa_f32_kernel<<<blocks, threads, shared_bytes, gpu->stream>>>(
        (const float *)query->device, (const float *)key->device,
        (const float *)value->device, (float *)output->device, args);
    gpu->stats.mps_sdpa_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_f32");
}

__global__ static void h3_weight_norm_f32_kernel(const float *vector,
                                                 const float *magnitude,
                                                 float *output, uint32_t outer,
                                                 uint32_t inner) {
    uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    size_t base = (size_t)row * inner;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < inner; index++) {
        float value = vector[base + index];
        square_sum = fmaf(value, value, square_sum);
    }
    float scale = magnitude[row] * rsqrtf(square_sum);
    for (uint32_t index = 0; index < inner; index++)
        output[base + index] = vector[base + index] * scale;
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude, uint32_t outer,
                           uint32_t inner) {
    size_t count = (size_t)outer * inner;
    if (!gpu || !output || !vector || !magnitude ||
        output->dtype != H3_GPU_F32 || vector->dtype != H3_GPU_F32 ||
        magnitude->dtype != H3_GPU_F32 || output->elements < count ||
        vector->elements < count || magnitude->elements < outer || !outer ||
        !inner)
        return h3_gpu_fail(gpu, "invalid weight-norm request");
    unsigned threads = 256;
    unsigned blocks = (outer + threads - 1u) / threads;
    h3_weight_norm_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)vector->device, (const float *)magnitude->device,
        (float *)output->device, outer, inner);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_weight_norm_f32");
}

struct h3_conv1d_args {
    uint32_t batch;
    uint32_t length;
    uint32_t output_length;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t has_bias;
};

__global__ static void h3_conv1d_f32_kernel(const float *input,
                                            const float *weight,
                                            const float *bias, float *output,
                                            h3_conv1d_args args) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)args.batch * args.output_length * args.output_channels;
    if (index >= total) return;
    uint32_t oc = (uint32_t)(index % args.output_channels);
    size_t rem = index / args.output_channels;
    uint32_t t_out = (uint32_t)(rem % args.output_length);
    uint32_t batch = (uint32_t)(rem / args.output_length);
    float acc = args.has_bias ? bias[oc] : 0.0f;
    for (uint32_t ic = 0; ic < args.input_channels; ic++) {
        for (uint32_t k = 0; k < args.kernel; k++) {
            int32_t t_in = (int32_t)t_out * (int32_t)args.stride -
                           (int32_t)args.padding +
                           (int32_t)k * (int32_t)args.dilation;
            if (t_in < 0 || t_in >= (int32_t)args.length) continue;
            size_t in_index =
                ((size_t)batch * args.length + (size_t)t_in) *
                    args.input_channels +
                ic;
            size_t w_index =
                ((size_t)oc * args.input_channels + ic) * args.kernel + k;
            acc = fmaf(input[in_index], weight[w_index], acc);
        }
    }
    output[index] = acc;
}

static int h3_gpu_conv1d_impl(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input,
                              const h3_gpu_tensor *weight,
                              const h3_gpu_tensor *bias, uint32_t batch,
                              uint32_t length, uint32_t input_channels,
                              uint32_t output_channels, uint32_t kernel,
                              uint32_t stride, uint32_t padding,
                              uint32_t dilation) {
    uint64_t effective = (uint64_t)dilation * (kernel - 1u) + 1u;
    if (!gpu || !output || !input || !weight || !batch || !length ||
        !input_channels || !output_channels || !kernel || !stride ||
        !dilation || (uint64_t)length + 2ull * padding < effective)
        return h3_gpu_fail(gpu, "invalid Conv1d request");
    uint32_t output_length = (uint32_t)(((uint64_t)length + 2ull * padding -
                                         effective) /
                                            stride +
                                        1u);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
        weight->dtype != H3_GPU_F32 || (bias && bias->dtype != H3_GPU_F32) ||
        output->elements < output_count || input->elements < input_count ||
        weight->elements < weight_count ||
        (bias && bias->elements < output_channels))
        return h3_gpu_fail(gpu, "invalid Conv1d tensor shapes");
    h3_conv1d_args args = {batch,         length,         output_length,
                           input_channels, output_channels, kernel,
                           stride,        padding,        dilation,
                           bias ? 1u : 0u};
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)((output_count + threads - 1) / threads);
    h3_conv1d_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (const float *)weight->device,
        bias ? (const float *)bias->device : NULL, (float *)output->device,
        args);
    gpu->stats.mps_conv_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_conv1d_f32");
}

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t batch,
                             uint32_t length, uint32_t input_channels,
                             uint32_t output_channels, uint32_t kernel,
                             uint32_t stride, uint32_t padding,
                             uint32_t dilation) {
    return h3_gpu_conv1d_impl(gpu, output, input, weight, bias, batch, length,
                              input_channels, output_channels, kernel, stride,
                              padding, dilation);
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_impl(gpu, output, input, weight, bias, batch, length,
                              input_channels, output_channels, kernel, 1u,
                              padding, dilation);
}

struct h3_conv_transpose1d_args {
    uint32_t batch;
    uint32_t length;
    uint32_t output_length;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel;
    uint32_t stride;
    uint32_t padding;
    uint32_t has_bias;
};

__global__ static void h3_conv_transpose1d_f32_kernel(
    const float *input, const float *weight, const float *bias, float *output,
    h3_conv_transpose1d_args args) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total =
        (size_t)args.batch * args.output_length * args.output_channels;
    if (index >= total) return;
    uint32_t oc = (uint32_t)(index % args.output_channels);
    size_t rem = index / args.output_channels;
    uint32_t t_out = (uint32_t)(rem % args.output_length);
    uint32_t batch = (uint32_t)(rem / args.output_length);
    float acc = args.has_bias ? bias[oc] : 0.0f;
    for (uint32_t ic = 0; ic < args.input_channels; ic++) {
        for (uint32_t k = 0; k < args.kernel; k++) {
            int32_t numerator =
                (int32_t)t_out + (int32_t)args.padding - (int32_t)k;
            if (numerator < 0 || (numerator % (int32_t)args.stride) != 0)
                continue;
            int32_t t_in = numerator / (int32_t)args.stride;
            if (t_in < 0 || t_in >= (int32_t)args.length) continue;
            size_t in_index =
                ((size_t)batch * args.length + (size_t)t_in) *
                    args.input_channels +
                ic;
            /* Transpose weight layout IOK: [ic, oc, k]. */
            size_t w_index =
                ((size_t)ic * args.output_channels + oc) * args.kernel + k;
            acc = fmaf(input[in_index], weight[w_index], acc);
        }
    }
    output[index] = acc;
}

int h3_gpu_conv_transpose1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input,
                                const h3_gpu_tensor *weight,
                                const h3_gpu_tensor *bias, uint32_t batch,
                                uint32_t length, uint32_t input_channels,
                                uint32_t output_channels, uint32_t kernel,
                                uint32_t stride, uint32_t padding) {
    if (!gpu || !output || !input || !weight || !batch || !length ||
        !input_channels || !output_channels || !kernel || !stride ||
        (uint64_t)(length - 1u) * stride + kernel < 2ull * padding)
        return h3_gpu_fail(gpu, "invalid ConvTranspose1d request");
    uint32_t output_length = (uint32_t)((uint64_t)(length - 1u) * stride +
                                        kernel - 2ull * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
        weight->dtype != H3_GPU_F32 || (bias && bias->dtype != H3_GPU_F32) ||
        output->elements < output_count || input->elements < input_count ||
        weight->elements < weight_count ||
        (bias && bias->elements < output_channels))
        return h3_gpu_fail(gpu, "invalid ConvTranspose1d tensor shapes");
    h3_conv_transpose1d_args args = {
        batch,         length,         output_length, input_channels,
        output_channels, kernel,       stride,        padding,
        bias ? 1u : 0u};
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)((output_count + threads - 1) / threads);
    h3_conv_transpose1d_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (const float *)weight->device,
        bias ? (const float *)bias->device : NULL, (float *)output->device,
        args);
    gpu->stats.mps_conv_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_conv_transpose1d_f32");
}

struct h3_audio_activation_args {
    uint32_t batch;
    uint32_t length;
    uint32_t channels;
};

__global__ static void h3_alias_free_snake_f32_kernel(
    const float *input, const float *alpha_log, const float *beta_log,
    const float *upsample_filter, const float *downsample_filter, float *output,
    h3_audio_activation_args args) {
    size_t count = (size_t)args.batch * args.length * args.channels;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    uint32_t channel = (uint32_t)(gid % args.channels);
    size_t rem = gid / args.channels;
    uint32_t time = (uint32_t)(rem % args.length);
    uint32_t batch = (uint32_t)(rem / args.length);
    float alpha = expf(alpha_log[channel]);
    float beta = expf(beta_log[channel]);
    float result = 0.0f;
    for (int down_k = 0; down_k < 12; down_k++) {
        int up_time = (int)time * 2 + down_k - 5;
        if (up_time < 0) up_time = 0;
        int up_max = (int)args.length * 2 - 1;
        if (up_time > up_max) up_time = up_max;
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_k = 0; up_k < 12; up_k++) {
            int numerator = raw_time - up_k;
            if (numerator < 0 || (numerator & 1)) continue;
            int padded_time = numerator / 2;
            int source_time = padded_time - 5;
            if (source_time < 0) source_time = 0;
            if (source_time > (int)args.length - 1)
                source_time = (int)args.length - 1;
            size_t source =
                ((size_t)batch * args.length + (size_t)source_time) *
                    args.channels +
                channel;
            upsampled =
                fmaf(input[source], 2.0f * upsample_filter[up_k], upsampled);
        }
        float sine = sinf(alpha * upsampled);
        float activated = upsampled + sine * sine / (beta + 1e-9f);
        result = fmaf(activated, downsample_filter[down_k], result);
    }
    output[gid] = result;
}

int h3_gpu_alias_free_snake_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *alpha_log, const h3_gpu_tensor *beta_log,
    const h3_gpu_tensor *upsample_filter,
    const h3_gpu_tensor *downsample_filter, uint32_t batch, uint32_t length,
    uint32_t channels) {
    size_t count = (size_t)batch * length * channels;
    if (!gpu || !output || !input || !alpha_log || !beta_log ||
        !upsample_filter || !downsample_filter ||
        output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
        alpha_log->dtype != H3_GPU_F32 || beta_log->dtype != H3_GPU_F32 ||
        upsample_filter->dtype != H3_GPU_F32 ||
        downsample_filter->dtype != H3_GPU_F32 || output->elements < count ||
        input->elements < count || alpha_log->elements < channels ||
        beta_log->elements < channels || upsample_filter->elements < 12 ||
        downsample_filter->elements < 12 || !batch || !length || !channels)
        return h3_gpu_fail(gpu, "invalid alias-free Snake request");
    h3_audio_activation_args args = {batch, length, channels};
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)((count + threads - 1) / threads);
    h3_alias_free_snake_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (const float *)alpha_log->device,
        (const float *)beta_log->device, (const float *)upsample_filter->device,
        (const float *)downsample_filter->device, (float *)output->device,
        args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_alias_free_snake_f32");
}

__global__ static void h3_snake1d_f32_kernel(const float *input,
                                             const float *alpha, float *output,
                                             h3_audio_activation_args args) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)args.batch * args.length * args.channels;
    if (index >= count) return;
    float a = alpha[index % args.channels];
    float x = input[index];
    float wave = sinf(a * x);
    output[index] = x + wave * wave / (a + 1e-9f);
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input, const h3_gpu_tensor *alpha,
                       uint32_t batch, uint32_t length, uint32_t channels) {
    size_t count = (size_t)batch * length * channels;
    if (!gpu || !output || !input || !alpha || output->dtype != H3_GPU_F32 ||
        input->dtype != H3_GPU_F32 || alpha->dtype != H3_GPU_F32 ||
        output->elements < count || input->elements < count ||
        alpha->elements < channels || !batch || !length || !channels)
        return h3_gpu_fail(gpu, "invalid Snake1d request");
    h3_audio_activation_args args = {batch, length, channels};
    unsigned threads = 256;
    unsigned blocks = (unsigned)((count + threads - 1) / threads);
    h3_snake1d_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (const float *)alpha->device,
        (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_snake1d_f32");
}

static __device__ __host__ int h3_reflect_coordinate(int coordinate,
                                                     int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

struct h3_vae_encoder_pad_args {
    uint32_t batch;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t depth_front;
    uint32_t height_before;
    uint32_t height_after;
    uint32_t width_before;
    uint32_t width_after;
};

__global__ static void h3_vae_encoder_pad_f32_kernel(
    const float *input, float *output, h3_vae_encoder_pad_args args) {
    uint32_t channel = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t out_x = (uint32_t)blockIdx.y;
    uint32_t out_height =
        args.height + args.height_before + args.height_after;
    uint32_t out_width = args.width + args.width_before + args.width_after;
    uint32_t out_depth = args.depth + args.depth_front;
    uint32_t plane = (uint32_t)blockIdx.z;
    if (channel >= args.channels || out_x >= out_width ||
        plane >= args.batch * out_depth * out_height)
        return;
    uint32_t out_y = plane % out_height;
    uint32_t temporal_plane = plane / out_height;
    uint32_t out_t = temporal_plane % out_depth;
    uint32_t batch = temporal_plane / out_depth;
    size_t destination =
        ((((size_t)batch * out_depth + out_t) * out_height + out_y) * out_width +
         out_x) *
            args.channels +
        channel;
    if (out_t < args.depth_front) {
        output[destination] = 0.0f;
        return;
    }
    int source_y = h3_reflect_coordinate((int)out_y - (int)args.height_before,
                                         (int)args.height);
    int source_x = h3_reflect_coordinate((int)out_x - (int)args.width_before,
                                         (int)args.width);
    uint32_t source_t = out_t - args.depth_front;
    size_t source =
        ((((size_t)batch * args.depth + source_t) * args.height +
          (uint32_t)source_y) *
             args.width +
         (uint32_t)source_x) *
            args.channels +
        channel;
    output[destination] = input[source];
}

int h3_gpu_vae_encoder_pad_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
    uint32_t channels, uint32_t depth_front, uint32_t height_before,
    uint32_t height_after, uint32_t width_before, uint32_t width_after) {
    uint32_t out_depth = depth + depth_front;
    uint32_t out_height = height + height_before + height_after;
    uint32_t out_width = width + width_before + width_after;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count =
        (size_t)batch * out_depth * out_height * out_width * channels;
    if (!gpu || !output || !input || output->dtype != H3_GPU_F32 ||
        input->dtype != H3_GPU_F32 || output->elements < output_count ||
        input->elements < input_count || !batch || !depth || !height ||
        !width || !channels)
        return h3_gpu_fail(gpu, "invalid VAE encoder pad request");
    h3_vae_encoder_pad_args args = {
        batch,         depth,         height,       width,       channels,
        depth_front,   height_before, height_after, width_before, width_after};
    dim3 threads(32, 1, 1);
    dim3 blocks((channels + threads.x - 1) / threads.x, out_width,
                batch * out_depth * out_height);
    h3_vae_encoder_pad_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_vae_encoder_pad_f32");
}

struct h3_vae_encoder_norm_args {
    uint32_t batch;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t groups;
    float epsilon;
};

__global__ static void h3_vae_encoder_group_norm_silu_f32_kernel(
    const float *input, const float *weight, const float *bias, float *output,
    h3_vae_encoder_norm_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t rows = args.batch * args.depth * args.groups;
    if (row >= rows) return;
    uint32_t channels_per_group = args.channels / args.groups;
    uint32_t group_index = row % args.groups;
    uint32_t temporal_plane = row / args.groups;
    uint32_t elements = args.height * args.width * channels_per_group;
    extern __shared__ float reductions[];
    float local = 0.0f;
    for (uint32_t index = tid; index < elements; index += threads) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel =
            group_index * channels_per_group + index % channels_per_group;
        size_t source =
            ((size_t)temporal_plane * args.height * args.width + spatial) *
                args.channels +
            channel;
        local += input[source];
    }
    reductions[tid] = local;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float mean = reductions[0] / (float)elements;
    local = 0.0f;
    for (uint32_t index = tid; index < elements; index += threads) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel =
            group_index * channels_per_group + index % channels_per_group;
        size_t source =
            ((size_t)temporal_plane * args.height * args.width + spatial) *
                args.channels +
            channel;
        float centered = input[source] - mean;
        local = fmaf(centered, centered, local);
    }
    reductions[tid] = local;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / (float)elements + args.epsilon);
    for (uint32_t index = tid; index < elements; index += threads) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel =
            group_index * channels_per_group + index % channels_per_group;
        size_t destination =
            ((size_t)temporal_plane * args.height * args.width + spatial) *
                args.channels +
            channel;
        float value =
            (input[destination] - mean) * inverse * weight[channel] +
            bias[channel];
        output[destination] = value / (1.0f + expf(-value));
    }
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch,
    uint32_t depth, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t groups, float epsilon) {
    size_t count = (size_t)batch * depth * height * width * channels;
    if (!gpu || !output || !input || !weight || !bias ||
        output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
        weight->dtype != H3_GPU_F32 || bias->dtype != H3_GPU_F32 ||
        output->elements < count || input->elements < count ||
        weight->elements < channels || bias->elements < channels || !batch ||
        !depth || !height || !width || !channels || !groups ||
        channels % groups != 0)
        return h3_gpu_fail(gpu, "invalid VAE group-norm request");
    h3_vae_encoder_norm_args args = {batch, depth, height, width,
                                     channels, groups, epsilon};
    uint32_t rows = batch * depth * groups;
    unsigned threads = 256;
    h3_vae_encoder_group_norm_silu_f32_kernel<<<
        rows, threads, threads * sizeof(float), gpu->stream>>>(
        (const float *)input->device, (const float *)weight->device,
        (const float *)bias->device, (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(),
                         "h3_vae_encoder_group_norm_silu_f32");
}

struct h3_conv3d_args {
    uint32_t batch;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t output_depth;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel_depth;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t stride_depth;
    uint32_t stride_height;
    uint32_t stride_width;
    uint32_t has_bias;
};

__global__ static void h3_conv3d_f32_kernel(const float *input,
                                            const float *weight,
                                            const float *bias, float *output,
                                            h3_conv3d_args args) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)args.batch * args.output_depth * args.output_height *
                   args.output_width * args.output_channels;
    if (index >= total) return;
    uint32_t oc = (uint32_t)(index % args.output_channels);
    size_t rem = index / args.output_channels;
    uint32_t ox = (uint32_t)(rem % args.output_width);
    rem /= args.output_width;
    uint32_t oy = (uint32_t)(rem % args.output_height);
    rem /= args.output_height;
    uint32_t od = (uint32_t)(rem % args.output_depth);
    uint32_t batch = (uint32_t)(rem / args.output_depth);
    float acc = args.has_bias ? bias[oc] : 0.0f;
    for (uint32_t ic = 0; ic < args.input_channels; ic++) {
        for (uint32_t kd = 0; kd < args.kernel_depth; kd++) {
            for (uint32_t kh = 0; kh < args.kernel_height; kh++) {
                for (uint32_t kw = 0; kw < args.kernel_width; kw++) {
                    uint32_t id = od * args.stride_depth + kd;
                    uint32_t ih = oy * args.stride_height + kh;
                    uint32_t iw = ox * args.stride_width + kw;
                    size_t in_index =
                        ((((size_t)batch * args.depth + id) * args.height +
                          ih) *
                             args.width +
                         iw) *
                            args.input_channels +
                        ic;
                    size_t w_index =
                        (((((size_t)oc * args.input_channels + ic) *
                               args.kernel_depth +
                           kd) *
                              args.kernel_height +
                          kh) *
                             args.kernel_width +
                         kw);
                    acc = fmaf(input[in_index], weight[w_index], acc);
                }
            }
        }
    }
    output[index] = acc;
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) {
    if (!gpu || !output || !input || !weight || !batch || !depth || !height ||
        !width || !input_channels || !output_channels || !kernel_depth ||
        !kernel_height || !kernel_width || !stride_depth || !stride_height ||
        !stride_width || depth < kernel_depth || height < kernel_height ||
        width < kernel_width)
        return h3_gpu_fail(gpu, "invalid Conv3d request");
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1u;
    uint32_t output_height = (height - kernel_height) / stride_height + 1u;
    uint32_t output_width = (width - kernel_width) / stride_width + 1u;
    size_t input_count =
        (size_t)batch * depth * height * width * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels *
                          kernel_depth * kernel_height * kernel_width;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * output_channels;
    if (output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
        weight->dtype != H3_GPU_F32 || (bias && bias->dtype != H3_GPU_F32) ||
        output->elements < output_count || input->elements < input_count ||
        weight->elements < weight_count ||
        (bias && bias->elements < output_channels))
        return h3_gpu_fail(gpu, "invalid Conv3d tensor shapes");
    h3_conv3d_args args = {
        batch,         depth,          height,         width,
        output_depth,  output_height,  output_width,   input_channels,
        output_channels, kernel_depth, kernel_height,  kernel_width,
        stride_depth,  stride_height,  stride_width,   bias ? 1u : 0u};
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)((output_count + threads - 1) / threads);
    h3_conv3d_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)input->device, (const float *)weight->device,
        bias ? (const float *)bias->device : NULL, (float *)output->device,
        args);
    gpu->stats.mps_conv_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_conv3d_f32");
}

struct h3_audio_qkv_args {
    uint32_t batch;
    uint32_t length;
    uint32_t heads;
    uint32_t head_dim;
};

__global__ static void h3_audio_qkv_split_f32_kernel(
    const float *qkv, const float *q_bias, const float *k_bias,
    const float *v_bias, float *query, float *key, float *value,
    h3_audio_qkv_args args) {
    size_t width = (size_t)args.heads * args.head_dim;
    size_t count = (size_t)args.batch * args.length * width;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    uint32_t column = (uint32_t)(gid % width);
    size_t row = gid / width;
    size_t base = row * width * 3u;
    query[gid] = qkv[base + column] + q_bias[column];
    key[gid] = qkv[base + width + column] + k_bias[column];
    value[gid] = qkv[base + width * 2u + column] + v_bias[column];
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                               h3_gpu_tensor *key, h3_gpu_tensor *value,
                               const h3_gpu_tensor *qkv,
                               const h3_gpu_tensor *q_bias,
                               const h3_gpu_tensor *k_bias,
                               const h3_gpu_tensor *v_bias, uint32_t batch,
                               uint32_t length, uint32_t heads,
                               uint32_t head_dim) {
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (!gpu || !query || !key || !value || !qkv || !q_bias || !k_bias ||
        !v_bias || query->dtype != H3_GPU_F32 || key->dtype != H3_GPU_F32 ||
        value->dtype != H3_GPU_F32 || qkv->dtype != H3_GPU_F32 ||
        q_bias->dtype != H3_GPU_F32 || k_bias->dtype != H3_GPU_F32 ||
        v_bias->dtype != H3_GPU_F32 || query->elements < count ||
        key->elements < count || value->elements < count ||
        qkv->elements < count * 3u || q_bias->elements < width ||
        k_bias->elements < width || v_bias->elements < width || !batch ||
        !length || !heads || !head_dim)
        return h3_gpu_fail(gpu, "invalid audio QKV split request");
    h3_audio_qkv_args args = {batch, length, heads, head_dim};
    unsigned threads = 256;
    unsigned blocks = (unsigned)((count + threads - 1) / threads);
    h3_audio_qkv_split_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)qkv->device, (const float *)q_bias->device,
        (const float *)k_bias->device, (const float *)v_bias->device,
        (float *)query->device, (float *)key->device, (float *)value->device,
        args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_audio_qkv_split_f32");
}

struct h3_sdpa_causal_args {
    uint32_t batch;
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    float scale;
};

__global__ static void h3_sdpa_causal_f32_kernel(
    const float *query, const float *key, const float *value, float *output,
    h3_sdpa_causal_args args) {
    extern __shared__ float scores[];
    uint32_t head = (uint32_t)blockIdx.x;
    uint32_t q_row = (uint32_t)blockIdx.y;
    uint32_t batch = (uint32_t)blockIdx.z;
    uint32_t dim = (uint32_t)threadIdx.x;
    if (batch >= args.batch || head >= args.heads || q_row >= args.sequence ||
        dim >= args.head_dim)
        return;
    size_t q_base =
        (((size_t)batch * args.sequence + q_row) * args.heads + head) *
        args.head_dim;
    if (dim == 0) {
        float max_score = -INFINITY;
        for (uint32_t k_row = 0; k_row < args.sequence; k_row++) {
            if (k_row > q_row) {
                scores[k_row] = -INFINITY;
                continue;
            }
            size_t k_base =
                (((size_t)batch * args.sequence + k_row) * args.heads +
                 head) *
                args.head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < args.head_dim; d++)
                dot = fmaf(query[q_base + d], key[k_base + d], dot);
            scores[k_row] = dot * args.scale;
            if (scores[k_row] > max_score) max_score = scores[k_row];
        }
        float sum = 0.0f;
        for (uint32_t k_row = 0; k_row < args.sequence; k_row++) {
            if (k_row > q_row) {
                scores[k_row] = 0.0f;
                continue;
            }
            scores[k_row] = expf(scores[k_row] - max_score);
            sum += scores[k_row];
        }
        float inverse = 1.0f / sum;
        for (uint32_t k_row = 0; k_row <= q_row; k_row++)
            scores[k_row] *= inverse;
    }
    __syncthreads();
    float accumulated = 0.0f;
    for (uint32_t k_row = 0; k_row <= q_row; k_row++) {
        size_t v_base =
            (((size_t)batch * args.sequence + k_row) * args.heads + head) *
            args.head_dim;
        accumulated = fmaf(scores[k_row], value[v_base + dim], accumulated);
    }
    output[q_base + dim] = accumulated;
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t batch,
                           uint32_t sequence, uint32_t heads,
                           uint32_t head_dim, float scale) {
    size_t count = (size_t)batch * sequence * heads * head_dim;
    if (!gpu || !output || !query || !key || !value ||
        output->dtype != H3_GPU_F32 || query->dtype != H3_GPU_F32 ||
        key->dtype != H3_GPU_F32 || value->dtype != H3_GPU_F32 ||
        output->elements < count || query->elements < count ||
        key->elements < count || value->elements < count || !batch ||
        !sequence || !heads || !head_dim)
        return h3_gpu_fail(gpu, "invalid causal SDPA request");
    h3_sdpa_causal_args args = {batch, sequence, heads, head_dim, scale};
    dim3 threads(head_dim, 1, 1);
    dim3 blocks(heads, sequence, batch);
    size_t shared_bytes = (size_t)sequence * sizeof(float);
    h3_sdpa_causal_f32_kernel<<<blocks, threads, shared_bytes, gpu->stream>>>(
        (const float *)query->device, (const float *)key->device,
        (const float *)value->device, (float *)output->device, args);
    gpu->stats.mps_sdpa_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_causal_f32");
}

struct h3_audio_pool_args {
    uint32_t batch;
    uint32_t length;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t output_dim;
};

__global__ static void h3_audio_attention_pool_f32_kernel(
    const float *attended, float *output, h3_audio_pool_args args) {
    size_t count = (size_t)args.batch * args.length * args.output_dim;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    uint32_t column = (uint32_t)(gid % args.output_dim);
    size_t row = gid / args.output_dim;
    uint32_t pool = args.head_dim / args.output_dim;
    float sum = 0.0f;
    for (uint32_t head = 0; head < args.heads; head++) {
        size_t base =
            (row * args.heads + head) * args.head_dim + column * pool;
        for (uint32_t item = 0; item < pool; item++)
            sum += attended[base + item];
    }
    output[gid] = sum / (float)(args.heads * pool);
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                    const h3_gpu_tensor *attended,
                                    uint32_t batch, uint32_t length,
                                    uint32_t heads, uint32_t head_dim,
                                    uint32_t output_dim) {
    size_t input_count = (size_t)batch * length * heads * head_dim;
    size_t output_count = (size_t)batch * length * output_dim;
    if (!gpu || !output || !attended || output->dtype != H3_GPU_F32 ||
        attended->dtype != H3_GPU_F32 || output->elements < output_count ||
        attended->elements < input_count || !batch || !length || !heads ||
        !head_dim || !output_dim || head_dim % output_dim != 0)
        return h3_gpu_fail(gpu, "invalid audio attention pool request");
    h3_audio_pool_args args = {batch, length, heads, head_dim, output_dim};
    unsigned threads = 256;
    unsigned blocks =
        (unsigned)((output_count + threads - 1) / threads);
    h3_audio_attention_pool_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)attended->device, (float *)output->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(),
                         "h3_audio_attention_pool_f32");
}

struct h3_int8_quant_args {
    uint32_t rows;
    uint32_t dispatch_rows;
    uint32_t columns;
    float clip;
};

__global__ static void h3_quantize_bf16_int8_rows_kernel(
    const uint16_t *input, int8_t *output, float *scales,
    h3_int8_quant_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.dispatch_rows) return;

    extern __shared__ float reductions[];
    size_t base = (size_t)row * args.columns;
    if (row >= args.rows) {
        for (uint32_t column = tid; column < args.columns; column += threads)
            output[base + column] = 0;
        if (tid == 0) scales[row] = 1.0f;
        return;
    }

    const uint16_t *row_input = input + base;
    float local_max = 0.0f;
    for (uint32_t column = tid; column < args.columns; column += threads) {
        float value = fabsf(h3_bf16_bits_to_f32(row_input[column]));
        if (value > local_max) local_max = value;
    }
    reductions[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) {
            float other = reductions[tid + stride];
            if (other > reductions[tid]) reductions[tid] = other;
        }
        __syncthreads();
    }
    float clipped_max = reductions[0] * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    __syncthreads();
    for (uint32_t column = tid; column < args.columns; column += threads) {
        float value = h3_bf16_bits_to_f32(row_input[column]) * inverse;
        int quantized = (int)rintf(value);
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;
        output[base + column] = (int8_t)quantized;
    }
}

static int h3_gpu_quantize_bf16_int8_rows(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *scales,
    const h3_gpu_tensor *input, uint32_t rows, uint32_t dispatch_rows,
    uint32_t columns, float clip) {
    if (!gpu || !output || !scales || !input || !rows ||
        dispatch_rows < rows || !columns || clip <= 0.0f ||
        input->dtype != H3_GPU_BF16 || output->dtype != H3_GPU_I8 ||
        scales->dtype != H3_GPU_F32 ||
        input->elements < (size_t)rows * columns ||
        output->elements < (size_t)dispatch_rows * columns ||
        scales->elements < dispatch_rows)
        return h3_gpu_fail(gpu, "invalid BF16→INT8 row quantize request");
    h3_int8_quant_args args = {rows, dispatch_rows, columns, clip};
    unsigned threads = 256;
    h3_quantize_bf16_int8_rows_kernel<<<dispatch_rows, threads,
                                          threads * sizeof(float),
                                          gpu->stream>>>(
        (const uint16_t *)input->device, (int8_t *)output->device,
        (float *)scales->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(),
                         "h3_quantize_bf16_int8_rows");
}

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
    return h3_gpu_quantize_bf16_int8_rows(gpu, output, scales, input, rows,
                                          rows, columns, 1.0f);
}

__global__ static void h3_int8_apply_scales_bf16_kernel(
    const int32_t *accum, const float *input_scales,
    const float *weight_scales, uint16_t *output, uint32_t rows,
    uint32_t output_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)rows * output_dim;
    if (index >= count) return;
    uint32_t row = (uint32_t)(index / output_dim);
    uint32_t column = (uint32_t)(index % output_dim);
    float value = (float)accum[index] * input_scales[row] * weight_scales[column];
    output[index] = h3_f32_to_bf16_bits(value);
}

__global__ static void h3_linear_int8_naive_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, uint32_t rows,
    uint32_t input_dim, uint32_t output_dim) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.y;
    if (row >= rows || column >= output_dim) return;
    int32_t sum = 0;
    size_t input_base = (size_t)row * input_dim;
    size_t weight_base = (size_t)column * input_dim;
    for (uint32_t k = 0; k < input_dim; k++) {
        sum += (int32_t)input[input_base + k] * (int32_t)weight[weight_base + k];
    }
    float value =
        (float)sum * input_scales[row] * weight_scales[column];
    output[(size_t)row * output_dim + column] = h3_f32_to_bf16_bits(value);
}

static int h3_gpu_linear_int8_bf16_impl(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
    int use_slower_uncached_int8_scales, int input_is_quantized) {
    (void)use_slower_uncached_int8_scales;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (padded_rows < rows) padded_rows = rows;
    size_t output_count = (size_t)rows * output_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t quant_capacity = (size_t)padded_rows * input_dim;
    if (!gpu || !output || !quantized_input || !input_scales || !weight ||
        !weight_scales || (!input_is_quantized && !input) ||
        output->dtype != H3_GPU_BF16 ||
        quantized_input->dtype != H3_GPU_I8 ||
        input_scales->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_I8 ||
        weight_scales->dtype != H3_GPU_F32 ||
        (!input_is_quantized && input->dtype != H3_GPU_BF16) ||
        output->elements < output_count ||
        quantized_input->elements < quant_capacity ||
        input_scales->elements < padded_rows ||
        weight->elements < weight_count ||
        weight_scales->elements < output_dim ||
        (!input_is_quantized &&
         input->elements < (size_t)rows * input_dim) ||
        !rows || !input_dim || !output_dim)
        return h3_gpu_fail(gpu, "invalid INT8 linear request");

    if (!input_is_quantized) {
        if (!h3_gpu_quantize_bf16_int8_rows(
                gpu, quantized_input, input_scales, input, rows, padded_rows,
                input_dim, 1.0f))
            return 0;
    }

    /* Prefer cuBLAS INT8 GEMM when available; fall back to a naive kernel. */
    int32_t *accum = NULL;
    cudaError_t alloc_status =
        cudaMalloc((void **)&accum, output_count * sizeof(*accum));
    if (alloc_status == cudaSuccess) {
        int32_t alpha = 1;
        int32_t beta = 0;
        cublasStatus_t status = cublasGemmEx(
            gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim, (int)rows,
            (int)input_dim, &alpha, weight->device, CUDA_R_8I, (int)input_dim,
            quantized_input->device, CUDA_R_8I, (int)input_dim, &beta, accum,
            CUDA_R_32I, (int)output_dim, CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT);
        if (status == CUBLAS_STATUS_SUCCESS) {
            gpu->stats.direct_dispatches++;
            unsigned threads = 256;
            unsigned blocks =
                (unsigned)((output_count + threads - 1) / threads);
            h3_int8_apply_scales_bf16_kernel<<<blocks, threads, 0,
                                                 gpu->stream>>>(
                accum, (const float *)input_scales->device,
                (const float *)weight_scales->device, (uint16_t *)output->device,
                rows, output_dim);
            cudaFree(accum);
            gpu->stats.direct_dispatches++;
            return h3_cuda_check(gpu, cudaGetLastError(),
                                 "h3_int8_apply_scales_bf16");
        }
        cudaFree(accum);
    }

    dim3 threads(256, 1, 1);
    dim3 blocks((output_dim + threads.x - 1) / threads.x, rows, 1);
    h3_linear_int8_naive_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const int8_t *)quantized_input->device, (const int8_t *)weight->device,
        (const float *)input_scales->device,
        (const float *)weight_scales->device, (uint16_t *)output->device, rows,
        input_dim, output_dim);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_linear_int8_naive");
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
    return h3_gpu_linear_int8_bf16_impl(
        gpu, output, quantized_input, input_scales, input, weight,
        weight_scales, rows, input_dim, output_dim,
        use_slower_uncached_int8_scales, 0);
}

__global__ static void h3_quantize_bf16_int8_head_major_kernel(
    const uint16_t *input, int8_t *output, float *scales, uint32_t rows,
    uint32_t padded_rows, uint32_t heads, uint32_t head_dim) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t columns = heads * head_dim;
    if (row >= padded_rows) return;

    extern __shared__ float reductions[];
    size_t out_base = (size_t)row * columns;
    if (row >= rows) {
        for (uint32_t column = tid; column < columns; column += threads)
            output[out_base + column] = 0;
        if (tid == 0) scales[row] = 1.0f;
        return;
    }

    float local_max = 0.0f;
    for (uint32_t column = tid; column < columns; column += threads) {
        uint32_t head = column / head_dim;
        uint32_t dim = column % head_dim;
        size_t in_index =
            ((size_t)head * rows + row) * head_dim + dim;
        float value = fabsf(h3_bf16_bits_to_f32(input[in_index]));
        if (value > local_max) local_max = value;
    }
    reductions[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) {
            float other = reductions[tid + stride];
            if (other > reductions[tid]) reductions[tid] = other;
        }
        __syncthreads();
    }
    float clipped_max = reductions[0];
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    __syncthreads();
    for (uint32_t column = tid; column < columns; column += threads) {
        uint32_t head = column / head_dim;
        uint32_t dim = column % head_dim;
        size_t in_index =
            ((size_t)head * rows + row) * head_dim + dim;
        int quantized =
            (int)rintf(h3_bf16_bits_to_f32(input[in_index]) * inverse);
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;
        output[out_base + column] = (int8_t)quantized;
    }
}

int h3_gpu_linear_int8_head_major_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t output_dim) {
    uint32_t input_dim = heads * head_dim;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (padded_rows < rows) padded_rows = rows;
    if (!gpu || !output || !quantized_input || !input_scales || !input ||
        !weight || !weight_scales || !rows || !heads || !head_dim ||
        !output_dim || input->dtype != H3_GPU_BF16 ||
        quantized_input->dtype != H3_GPU_I8 ||
        input_scales->dtype != H3_GPU_F32 ||
        input->elements < (size_t)rows * input_dim ||
        quantized_input->elements < (size_t)padded_rows * input_dim ||
        input_scales->elements < padded_rows)
        return h3_gpu_fail(gpu, "invalid head-major INT8 linear request");

    unsigned threads = 256;
    h3_quantize_bf16_int8_head_major_kernel<<<padded_rows, threads,
                                                threads * sizeof(float),
                                                gpu->stream>>>(
        (const uint16_t *)input->device, (int8_t *)quantized_input->device,
        (float *)input_scales->device, rows, padded_rows, heads, head_dim);
    gpu->stats.direct_dispatches++;
    if (!h3_cuda_check(gpu, cudaGetLastError(),
                       "h3_quantize_bf16_int8_head_major"))
        return 0;
    return h3_gpu_linear_int8_bf16_impl(
        gpu, output, quantized_input, input_scales, NULL, weight,
        weight_scales, rows, input_dim, output_dim, 0, 1);
}

struct h3_int8_group_quant_args {
    uint32_t rows;
    uint32_t dispatch_rows;
    uint32_t columns;
    uint32_t group_size;
    uint32_t groups;
};

__global__ static void h3_quantize_bf16_int8_groups_kernel(
    const uint16_t *input, int8_t *output, float *scales,
    h3_int8_group_quant_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t group = (uint32_t)blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.dispatch_rows || group >= args.groups) return;

    extern __shared__ float reductions[];
    size_t row_base = (size_t)row * args.columns;
    uint32_t group_start = group * args.group_size;
    if (row >= args.rows) {
        for (uint32_t column = tid; column < args.group_size; column += threads)
            output[row_base + group_start + column] = 0;
        if (tid == 0) scales[(size_t)row * args.groups + group] = 1.0f;
        return;
    }

    float local_max = 0.0f;
    for (uint32_t column = tid; column < args.group_size; column += threads) {
        float value =
            fabsf(h3_bf16_bits_to_f32(input[row_base + group_start + column]));
        if (value > local_max) local_max = value;
    }
    reductions[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads / 2u; stride; stride >>= 1u) {
        if (tid < stride) {
            float other = reductions[tid + stride];
            if (other > reductions[tid]) reductions[tid] = other;
        }
        __syncthreads();
    }
    float clipped_max = reductions[0];
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[(size_t)row * args.groups + group] = scale;
    __syncthreads();
    for (uint32_t column = tid; column < args.group_size; column += threads) {
        int quantized = (int)rintf(
            h3_bf16_bits_to_f32(input[row_base + group_start + column]) *
            inverse);
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;
        output[row_base + group_start + column] = (int8_t)quantized;
    }
}

static int h3_gpu_quantize_bf16_int8_groups(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *scales,
    const h3_gpu_tensor *input, uint32_t rows, uint32_t dispatch_rows,
    uint32_t columns, uint32_t group_size) {
    if (!group_size || columns % group_size)
        return h3_gpu_fail(gpu, "invalid grouped INT8 quantize geometry");
    uint32_t groups = columns / group_size;
    if (!gpu || !output || !scales || !input || !rows ||
        dispatch_rows < rows || !columns || input->dtype != H3_GPU_BF16 ||
        output->dtype != H3_GPU_I8 || scales->dtype != H3_GPU_F32 ||
        input->elements < (size_t)rows * columns ||
        output->elements < (size_t)dispatch_rows * columns ||
        scales->elements < (size_t)dispatch_rows * groups)
        return h3_gpu_fail(gpu, "invalid grouped INT8 quantize request");
    h3_int8_group_quant_args args = {rows, dispatch_rows, columns, group_size,
                                     groups};
    unsigned threads = 256;
    dim3 blocks(dispatch_rows, groups, 1);
    h3_quantize_bf16_int8_groups_kernel<<<blocks, threads,
                                            threads * sizeof(float),
                                            gpu->stream>>>(
        (const uint16_t *)input->device, (int8_t *)output->device,
        (float *)scales->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(),
                         "h3_quantize_bf16_int8_groups");
}

__global__ static void h3_linear_int8_grouped_naive_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, uint32_t rows,
    uint32_t input_dim, uint32_t output_dim, uint32_t group_size,
    uint32_t groups) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.y;
    if (row >= rows || column >= output_dim) return;
    float total = 0.0f;
    size_t input_base = (size_t)row * input_dim;
    size_t weight_base = (size_t)column * input_dim;
    for (uint32_t group = 0; group < groups; group++) {
        int32_t sum = 0;
        uint32_t start = group * group_size;
        for (uint32_t k = 0; k < group_size; k++) {
            sum += (int32_t)input[input_base + start + k] *
                   (int32_t)weight[weight_base + start + k];
        }
        total += (float)sum * input_scales[(size_t)row * groups + group] *
                 weight_scales[column];
    }
    output[(size_t)row * output_dim + column] = h3_f32_to_bf16_bits(total);
}

static int h3_gpu_linear_int8_grouped_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *quantized_input,
    const h3_gpu_tensor *input_scales, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim, uint32_t group_size) {
    if (!group_size || input_dim % group_size)
        return h3_gpu_fail(gpu, "invalid grouped INT8 linear geometry");
    uint32_t groups = input_dim / group_size;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (padded_rows < rows) padded_rows = rows;
    size_t output_count = (size_t)rows * output_dim;
    if (!gpu || !output || !quantized_input || !input_scales || !weight ||
        !weight_scales || output->dtype != H3_GPU_BF16 ||
        quantized_input->dtype != H3_GPU_I8 ||
        input_scales->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_I8 ||
        weight_scales->dtype != H3_GPU_F32 ||
        output->elements < output_count ||
        quantized_input->elements < (size_t)padded_rows * input_dim ||
        input_scales->elements < (size_t)padded_rows * groups ||
        weight->elements < (size_t)output_dim * input_dim ||
        weight_scales->elements < output_dim || !rows || !input_dim ||
        !output_dim)
        return h3_gpu_fail(gpu, "invalid grouped INT8 linear request");

    dim3 threads(256, 1, 1);
    dim3 blocks((output_dim + threads.x - 1) / threads.x, rows, 1);
    h3_linear_int8_grouped_naive_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const int8_t *)quantized_input->device, (const int8_t *)weight->device,
        (const float *)input_scales->device,
        (const float *)weight_scales->device, (uint16_t *)output->device, rows,
        input_dim, output_dim, group_size, groups);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(),
                         "h3_linear_int8_grouped_naive");
}

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k, int use_int8_row_fc2,
                         int input_is_quantized) {
    (void)use_slower_grouped_quantizer;
    (void)use_slower_dynamic_fc1_k;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (padded_rows < rows) padded_rows = rows;
    uint32_t fc2_groups =
        hidden_dim >= 1024u && (hidden_dim % 1024u) == 0u ? hidden_dim / 1024u
                                                         : 1u;
    size_t activation_capacity =
        (size_t)padded_rows *
        (input_dim > hidden_dim ? input_dim : hidden_dim);
    const char *stage = getenv("H3_INT8_MLP_STAGE");
    int int8_fc1 = !stage || (strcmp(stage, "fc2") && strcmp(stage, "bf16"));
    int int8_fc2 = !stage || (strcmp(stage, "fc1") && strcmp(stage, "bf16"));
    int grouped_fc2 = int8_fc2 && !use_int8_row_fc2 && fc2_groups > 1u;

    if (!gpu || !output || !activated || !quantized_activation ||
        !activation_scales || !fc1_weight || !fc1_scales || !fc2_weight ||
        !fc2_scales || (!input_is_quantized && !input) || !rows ||
        !input_dim || !hidden_dim || !output_dim ||
        quantized_activation->dtype != H3_GPU_I8 ||
        activation_scales->dtype != H3_GPU_F32 ||
        fc1_weight->dtype != H3_GPU_I8 || fc1_scales->dtype != H3_GPU_F32 ||
        fc2_weight->dtype != H3_GPU_I8 || fc2_scales->dtype != H3_GPU_F32 ||
        activated->dtype != H3_GPU_BF16 || output->dtype != H3_GPU_BF16 ||
        quantized_activation->elements < activation_capacity ||
        activation_scales->elements <
            (size_t)padded_rows * (grouped_fc2 ? fc2_groups : 1u) ||
        fc1_weight->elements < (size_t)hidden_dim * 2u * input_dim ||
        fc1_scales->elements < (size_t)hidden_dim * 2u ||
        fc2_weight->elements < (size_t)output_dim * hidden_dim ||
        fc2_scales->elements < output_dim ||
        activated->elements < (size_t)rows * hidden_dim ||
        output->elements < (size_t)rows * output_dim ||
        (!input_is_quantized &&
         (input->dtype != H3_GPU_BF16 ||
          input->elements < (size_t)rows * input_dim)) ||
        (!int8_fc1 &&
         (!fc1_bf16 || fc1_bf16->dtype != H3_GPU_BF16 ||
          fc1_bf16->elements < (size_t)hidden_dim * 2u * input_dim)) ||
        (!int8_fc2 &&
         (!fc2_bf16 || fc2_bf16->dtype != H3_GPU_BF16 ||
          fc2_bf16->elements < (size_t)output_dim * hidden_dim)))
        return h3_gpu_fail(gpu, "invalid INT8 MLP request");
    if (input_is_quantized && !int8_fc1)
        return h3_gpu_fail(gpu, "prequantized MLP input requires int8 FC1");

    h3_gpu_tensor *fc1_fused =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim * 2u);
    if (!fc1_fused) return h3_gpu_fail(gpu, "INT8 MLP FC1 temp alloc failed");

    int ok = 1;
    if (int8_fc1) {
        if (!input_is_quantized &&
            !h3_gpu_quantize_bf16_int8_rows(
                gpu, quantized_activation, activation_scales, input, rows,
                padded_rows, input_dim, 1.0f))
            ok = 0;
        if (ok &&
            !h3_gpu_linear_int8_bf16_impl(
                gpu, fc1_fused, quantized_activation, activation_scales, NULL,
                fc1_weight, fc1_scales, rows, input_dim, hidden_dim * 2u, 0,
                1))
            ok = 0;
        if (ok && !h3_gpu_swiglu_bf16(gpu, activated, fc1_fused, rows, hidden_dim))
            ok = 0;
    } else if (!h3_gpu_linear_bf16(gpu, fc1_fused, input, fc1_bf16, NULL, rows,
                                   input_dim, hidden_dim * 2u) ||
               !h3_gpu_swiglu_bf16(gpu, activated, fc1_fused, rows, hidden_dim)) {
        ok = 0;
    }

    if (ok && int8_fc2) {
        if (grouped_fc2) {
            if (!h3_gpu_quantize_bf16_int8_groups(
                    gpu, quantized_activation, activation_scales, activated,
                    rows, padded_rows, hidden_dim, 1024u))
                ok = 0;
            else if (!h3_gpu_linear_int8_grouped_bf16(
                         gpu, output, quantized_activation, activation_scales,
                         fc2_weight, fc2_scales, rows, hidden_dim, output_dim,
                         1024u))
                ok = 0;
        } else {
            if (!h3_gpu_quantize_bf16_int8_rows(
                    gpu, quantized_activation, activation_scales, activated,
                    rows, padded_rows, hidden_dim, 1.0f))
                ok = 0;
            else if (!h3_gpu_linear_int8_bf16_impl(
                         gpu, output, quantized_activation, activation_scales,
                         NULL, fc2_weight, fc2_scales, rows, hidden_dim,
                         output_dim, 0, 1))
                ok = 0;
        }
    } else if (ok &&
               !h3_gpu_linear_bf16(gpu, output, activated, fc2_bf16, NULL, rows,
                                   hidden_dim, output_dim)) {
        ok = 0;
    }

    h3_gpu_tensor_free(fc1_fused);
    return ok;
}

int h3_gpu_gate_adaln_quantize_int8(
    h3_gpu *gpu, h3_gpu_tensor *gated_residual,
    h3_gpu_tensor *quantized_output, h3_gpu_tensor *quantized_scales,
    const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation,
    const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t padded_rows, uint32_t width, uint32_t slots,
    uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot,
    float epsilon) {
    size_t elements = (size_t)rows * width;
    if (!gpu || !gated_residual || !quantized_output || !quantized_scales ||
        !residual || !branch || !norm_weight || !gate_modulation ||
        !norm_modulation || !row_map || !rows || padded_rows < rows || !width ||
        gate_slot >= slots || shift_slot >= slots || scale_slot >= slots)
        return h3_gpu_fail(gpu, "invalid gate AdaLN quantize request");

    h3_gpu_tensor *adaln = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!adaln) return h3_gpu_fail(gpu, "gate AdaLN quantize temp alloc failed");
    int ok =
        h3_gpu_gate_adaln_bf16(gpu, gated_residual, adaln, residual, branch,
                               norm_weight, gate_modulation, norm_modulation,
                               row_map, rows, width, slots, gate_slot,
                               shift_slot, scale_slot, epsilon) &&
        h3_gpu_quantize_bf16_int8_rows(gpu, quantized_output, quantized_scales,
                                       adaln, rows, padded_rows, width, 1.0f);
    h3_gpu_tensor_free(adaln);
    return ok;
}

int h3_gpu_grouped_qkv_linear_rope_int8(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t rows, uint32_t input_dim, uint32_t heads, uint32_t head_dim,
    uint32_t rope_half, float epsilon, int input_is_quantized,
    int use_slower_unfused_qkv_rope, int use_slower_scalar_qkv_rms,
    int use_slower_uncached_int8_scales) {
    (void)use_slower_unfused_qkv_rope;
    (void)use_slower_scalar_qkv_rms;
    (void)use_slower_uncached_int8_scales;
    uint32_t inner = heads * head_dim;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (padded_rows < rows) padded_rows = rows;
    size_t projected = (size_t)rows * inner;
    size_t rope_count = (size_t)rows * rope_half;
    if (!gpu || !query || !key || !value || !quantized_input || !input_scales ||
        !weight || !weight_scales || !q_norm || !k_norm || !rope_cos ||
        !rope_sin || (!input_is_quantized && !input) || !rows || !input_dim ||
        !heads || !head_dim ||
        weight->dtype != H3_GPU_I8 || weight_scales->dtype != H3_GPU_F32 ||
        quantized_input->dtype != H3_GPU_I8 ||
        input_scales->dtype != H3_GPU_F32 || query->dtype != H3_GPU_BF16 ||
        key->dtype != H3_GPU_BF16 || value->dtype != H3_GPU_BF16 ||
        q_norm->dtype != H3_GPU_BF16 || k_norm->dtype != H3_GPU_BF16 ||
        rope_cos->dtype != H3_GPU_BF16 || rope_sin->dtype != H3_GPU_BF16 ||
        weight->elements < (size_t)inner * 3u * input_dim ||
        weight_scales->elements < inner * 3u ||
        quantized_input->elements < (size_t)padded_rows * input_dim ||
        input_scales->elements < padded_rows || query->elements < projected ||
        key->elements < projected || value->elements < projected ||
        q_norm->elements < head_dim || k_norm->elements < head_dim ||
        rope_cos->elements < rope_count || rope_sin->elements < rope_count ||
        (!input_is_quantized &&
         (input->dtype != H3_GPU_BF16 ||
          input->elements < (size_t)rows * input_dim)))
        return h3_gpu_fail(gpu, "invalid INT8 QKV/RoPE request");

    if (!input_is_quantized &&
        !h3_gpu_quantize_bf16_int8_rows(gpu, quantized_input, input_scales,
                                        input, rows, padded_rows, input_dim,
                                        1.0f))
        return 0;

    h3_gpu_tensor *qkv =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * inner * 3u);
    if (!qkv) return h3_gpu_fail(gpu, "INT8 QKV temp alloc failed");
    int ok =
        h3_gpu_linear_int8_bf16_impl(
            gpu, qkv, quantized_input, input_scales, NULL, weight,
            weight_scales, rows, input_dim, inner * 3u, 0, 1) &&
        h3_gpu_qkv_rope_bf16_layout(gpu, query, key, value, qkv, q_norm,
                                    k_norm, rope_cos, rope_sin, rows, heads,
                                    head_dim, rope_half, 1u, epsilon);
    h3_gpu_tensor_free(qkv);
    return ok;
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
    if (!gpu || !phase || !*phase || !h3_gpu_profile_enabled()) return;
    h3_gpu_profile_emit(gpu, phase, &gpu->profile_mark_stats,
                        gpu->profile_mark_wall);
    gpu->profile_mark_stats = gpu->stats;
    gpu->profile_mark_wall = h3_gpu_now();
}

} /* extern "C" */
