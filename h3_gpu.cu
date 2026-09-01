extern "C" {
#include "h3_gpu.h"
#include "h3_gpu_cuda_internal.h"
}

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
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
    int pooled;
};

/* The text encoder reads one layer's weights from several lanes at once, so a
 * staging buffer cannot be shared: each reader claims a slot for the duration
 * of a chunk. Four slots keep the readers overlapped without pinning more host
 * memory than the loader needs. */
#define H3_STAGE_SLOTS 4

struct h3_gpu {
    cudaStream_t stream;
    cublasHandle_t cublas;
    void *stage_host[H3_STAGE_SLOTS];
    size_t stage_host_bytes;
    int stage_host_pinned;
    int stage_busy[H3_STAGE_SLOTS];
    int stage_next;
    int stage_event_recorded[H3_STAGE_SLOTS];
    cudaEvent_t stage_copied[H3_STAGE_SLOTS];
    pthread_mutex_t stage_lock;
    pthread_cond_t stage_free;
    int device;
    int fast_path;
    int tensor_fast_path;
    int pool_alloc;
    cublasLtHandle_t lt;
    int lt_ready;
    void *lt_workspace;
    size_t lt_workspace_bytes;
    char error[512];
    char profile_label[128];
    h3_gpu_stats stats;
    h3_gpu_stats profile_start_stats;
    h3_gpu_stats profile_mark_stats;
    double profile_start_wall;
    double profile_mark_wall;
    int op_events_ready;
    int op_event_count;
    unsigned char *op_class;
    cudaEvent_t *op_events;
    int32_t *int8_accum;
    size_t int8_accum_bytes;
    /* An INT8 GEMM whose caller asked to defer the rescale leaves its int32
     * accumulator and the two scale vectors here for the next consumer to fold
     * into its own first read. Valid only until the following INT8 GEMM
     * overwrites the shared accumulator, which is why the only callers are ops
     * that run immediately after their producer. */
    const float *int8_defer_input_scales;
    const float *int8_defer_weight_scales;
    uint32_t int8_defer_rows;
    uint32_t int8_defer_columns;
    /* Scratch for the Conv1d weight transposed to [ic][k][oc], the layout that
     * makes the weight read coalesce. Rebuilt per call rather than cached by
     * source pointer: the VAE streams its weights, so the allocator hands the
     * same address to a later layer with a different shape and a pointer-keyed
     * cache silently returns the wrong filter. Transposing costs about two
     * passes over 46 MB across the whole decode, which the layout wins back
     * many times over. */
    void *conv_weight_scratch;
    size_t conv_weight_scratch_bytes;
    uint64_t int8_cublas_ok;
    uint64_t int8_naive_fallback;
    h3_gpu_tensor *ws_mlp_fc1;
    h3_gpu_tensor *ws_mlp_hidden;
    h3_gpu_tensor *ws_qkv;
    h3_gpu_tensor *ws_int8_fc1;
    h3_gpu_tensor *ws_adaln;
};

enum {
    H3_GPU_OP_LINEAR = 0,
    H3_GPU_OP_SDPA = 1,
    H3_GPU_OP_CONV = 2,
    H3_GPU_OP_EVENT_MAX = 4096
};

void h3_gpu_tensor_free(h3_gpu_tensor *tensor);

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

static double h3_gpu_seconds_delta(double value, double start) {
    return value >= start ? value - start : 0.0;
}

static int h3_env_on(const char *name) {
    const char *value = getenv(name);
    return value && *value && strcmp(value, "0") != 0;
}

/* Quantization levels per side of zero, normally INT8's 127. Lowering it
 * coarsens every quantized tensor by a known factor, which is how a narrower
 * format's error can be put through the full pipeline before anyone writes the
 * kernels for it: 48 levels lands on FP8-E4M3's measured weight error and 13
 * lands on NVFP4's. */
static float h3_int8_levels(void) {
    static float levels = 0.0f;
    if (levels == 0.0f) {
        const char *value = getenv("H3_INT8_LEVELS");
        long parsed = value && *value ? strtol(value, NULL, 10) : 127;
        if (parsed < 1 || parsed > 127) parsed = 127;
        levels = (float)parsed;
    }
    return levels;
}

static void h3_gpu_op_events_destroy(h3_gpu *gpu) {
    if (!gpu || !gpu->op_events_ready) return;
    int count = H3_GPU_OP_EVENT_MAX * 2;
    for (int i = 0; i < count; i++) cudaEventDestroy(gpu->op_events[i]);
    free(gpu->op_events);
    free(gpu->op_class);
    gpu->op_events = NULL;
    gpu->op_class = NULL;
    gpu->op_events_ready = 0;
    gpu->op_event_count = 0;
}

static int h3_gpu_op_events_init(h3_gpu *gpu) {
    if (!gpu || gpu->op_events_ready) return gpu && gpu->op_events_ready;
    gpu->op_events = (cudaEvent_t *)calloc((size_t)H3_GPU_OP_EVENT_MAX * 2u,
                                           sizeof(cudaEvent_t));
    gpu->op_class = (unsigned char *)calloc(H3_GPU_OP_EVENT_MAX, 1);
    if (!gpu->op_events || !gpu->op_class) {
        free(gpu->op_events);
        free(gpu->op_class);
        gpu->op_events = NULL;
        gpu->op_class = NULL;
        return 0;
    }
    for (int i = 0; i < H3_GPU_OP_EVENT_MAX * 2; i++) {
        if (cudaEventCreate(&gpu->op_events[i]) != cudaSuccess) {
            for (int j = 0; j < i; j++) cudaEventDestroy(gpu->op_events[j]);
            free(gpu->op_events);
            free(gpu->op_class);
            gpu->op_events = NULL;
            gpu->op_class = NULL;
            return 0;
        }
    }
    gpu->op_events_ready = 1;
    gpu->op_event_count = 0;
    return 1;
}

static void h3_gpu_op_events_flush(h3_gpu *gpu) {
    if (!gpu || !gpu->op_events_ready || gpu->op_event_count <= 0) return;
    if (gpu->stream) cudaStreamSynchronize(gpu->stream);
    for (int i = 0; i < gpu->op_event_count; i++) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, gpu->op_events[i * 2],
                                 gpu->op_events[i * 2 + 1]) != cudaSuccess)
            continue;
        double seconds = (double)ms * 1e-3;
        unsigned cls = gpu->op_class[i];
        if (cls == H3_GPU_OP_LINEAR) gpu->stats.gpu_linear_seconds += seconds;
        else if (cls == H3_GPU_OP_SDPA) gpu->stats.gpu_sdpa_seconds += seconds;
        else if (cls == H3_GPU_OP_CONV) gpu->stats.gpu_conv_seconds += seconds;
    }
    gpu->op_event_count = 0;
}

static void h3_gpu_op_begin(h3_gpu *gpu, int cls) {
    if (!gpu || !h3_gpu_profile_enabled()) return;
    if (!h3_gpu_op_events_init(gpu)) return;
    if (gpu->op_event_count >= H3_GPU_OP_EVENT_MAX) h3_gpu_op_events_flush(gpu);
    int i = gpu->op_event_count;
    gpu->op_class[i] = (unsigned char)cls;
    cudaEventRecord(gpu->op_events[i * 2], gpu->stream);
}

static void h3_gpu_op_end(h3_gpu *gpu) {
    if (!gpu || !gpu->op_events_ready || !h3_gpu_profile_enabled()) return;
    if (gpu->op_event_count >= H3_GPU_OP_EVENT_MAX) return;
    int i = gpu->op_event_count;
    cudaEventRecord(gpu->op_events[i * 2 + 1], gpu->stream);
    gpu->op_event_count = i + 1;
}

static void h3_gpu_profile_emit(h3_gpu *gpu, const char *phase,
                                const h3_gpu_stats *start, double wall_start) {
    if (!gpu || !phase || !h3_gpu_profile_enabled()) return;
    h3_gpu_op_events_flush(gpu);
    if (gpu->stream) cudaStreamSynchronize(gpu->stream);
    h3_gpu_stats value = gpu->stats;
    double wall = h3_gpu_now() - wall_start;
    const char *label =
        gpu->profile_label[0] ? gpu->profile_label : "CUDA context";
    fprintf(stderr,
            "h3 profile: %-24s %-14s wall=%8.3fs "
            "peak=%7.3fGiB alloc=%7.3fGiB submissions=%llu "
            "direct=%llu linear=%llu conv=%llu attention=%llu "
            "gpu-op linear=%.3fs sdpa=%.3fs conv=%.3fs "
            "int8-cublas=%llu naive=%llu "
            "stage %.3fGiB read=%.3fs copy=%.3fs pin=%.3fs\n",
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
                value.mps_sdpa_dispatches, start->mps_sdpa_dispatches),
            h3_gpu_seconds_delta(value.gpu_linear_seconds,
                                 start->gpu_linear_seconds),
            h3_gpu_seconds_delta(value.gpu_sdpa_seconds,
                                 start->gpu_sdpa_seconds),
            h3_gpu_seconds_delta(value.gpu_conv_seconds,
                                 start->gpu_conv_seconds),
            (unsigned long long)gpu->int8_cublas_ok,
            (unsigned long long)gpu->int8_naive_fallback,
            (double)h3_gpu_counter_delta(value.stage_bytes, start->stage_bytes) /
                (1024.0 * 1024.0 * 1024.0),
            h3_gpu_seconds_delta(value.stage_read_seconds,
                                 start->stage_read_seconds),
            h3_gpu_seconds_delta(value.stage_copy_seconds,
                                 start->stage_copy_seconds),
            h3_gpu_seconds_delta(value.stage_pin_seconds,
                                 start->stage_pin_seconds));
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
    case H3_GPU_F8E4M3: return 1;
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
    /* The stream-ordered allocator recycles blocks out of a pool instead of
     * asking the driver to map fresh pages, which is what the weight loader's
     * thousands of same-shaped staging buffers want: cudaMalloc was averaging
     * 2.4 ms a call there. */
    cudaError_t status;
    if (gpu->pool_alloc && gpu->stream) {
        status = cudaMallocAsync(&tensor->device, bytes, gpu->stream);
        if (status == cudaSuccess) tensor->pooled = 1;
    } else {
        status = cudaMalloc(&tensor->device, bytes);
    }
    if (status != cudaSuccess) {
        free(tensor);
        h3_gpu_fail(gpu, "cudaMalloc failed: %s", cudaGetErrorString(status));
        return NULL;
    }
    /* Diagnostic: if any op reads a tensor before writing it, zeroing every
     * allocation changes the result. Kept because it is the cheapest way to
     * separate an uninitialized read from a genuine race. */
    if (h3_env_on("H3_GPU_ZERO_ALLOC") && gpu->stream)
        cudaMemsetAsync(tensor->device, 0, bytes, gpu->stream);
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
    gpu->stage_host_bytes = 0;
    gpu->stage_host_pinned = 0;
    gpu->stage_next = 0;
    pthread_mutex_init(&gpu->stage_lock, NULL);
    pthread_cond_init(&gpu->stage_free, NULL);
    for (int i = 0; i < H3_STAGE_SLOTS; i++) {
        gpu->stage_host[i] = NULL;
        gpu->stage_busy[i] = 0;
        gpu->stage_event_recorded[i] = 0;
        gpu->stage_copied[i] = NULL;
        cudaEventCreateWithFlags(&gpu->stage_copied[i], cudaEventDisableTiming);
    }
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess) {
        gpu->fast_path = props.major >= 12 ? 1 : 0;
        gpu->tensor_fast_path = gpu->fast_path;
    }
    /* Keep freed pool blocks resident so the loader's repeated shapes are
     * satisfied from the pool. Opt out with H3_GPU_SYNC_ALLOC=1. */
    if (!h3_env_on("H3_GPU_SYNC_ALLOC")) {
        cudaMemPool_t pool = NULL;
        if (cudaDeviceGetDefaultMemPool(&pool, gpu->device) == cudaSuccess &&
            pool) {
            uint64_t threshold = (uint64_t)24 << 30;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                    &threshold);
            gpu->pool_alloc = 1;
        }
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

/* Page-locking runs at roughly 1.5 GB/s, so one h3_gpu's four 128 MiB slots
 * cost about a third of a second, and a generation builds four contexts in
 * sequence — text encoder, DiT, audio VAE, video VAE — paying it every time.
 * cudaFreeHost is not free either, at ~28 ms a slot. The slots are all the same
 * size and no two contexts hold them at once, so a retired one goes back to a
 * process-wide pool and the next context takes it as-is. Only the first context
 * of a run pays the pinning cost. */
static struct {
    void *slots[H3_STAGE_SLOTS];
    size_t bytes;
    unsigned count;
    pthread_mutex_t lock;
} h3_stage_pool = {{NULL}, 0, 0, PTHREAD_MUTEX_INITIALIZER};

static void *h3_stage_pool_take(size_t bytes) {
    void *slot = NULL;
    pthread_mutex_lock(&h3_stage_pool.lock);
    if (h3_stage_pool.bytes == bytes && h3_stage_pool.count)
        slot = h3_stage_pool.slots[--h3_stage_pool.count];
    pthread_mutex_unlock(&h3_stage_pool.lock);
    return slot;
}

/* Returns whether the pool took it; if not, the caller still owns the slot. */
static int h3_stage_pool_give(void *slot, size_t bytes) {
    int kept = 0;
    pthread_mutex_lock(&h3_stage_pool.lock);
    if (!h3_stage_pool.count) h3_stage_pool.bytes = bytes;
    if (h3_stage_pool.bytes == bytes && h3_stage_pool.count < H3_STAGE_SLOTS) {
        h3_stage_pool.slots[h3_stage_pool.count++] = slot;
        kept = 1;
    }
    pthread_mutex_unlock(&h3_stage_pool.lock);
    return kept;
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    h3_gpu_profile_emit(gpu, "total", &gpu->profile_start_stats,
                        gpu->profile_start_wall);
    h3_gpu_op_events_destroy(gpu);
    h3_gpu_tensor_free(gpu->ws_mlp_fc1);
    h3_gpu_tensor_free(gpu->ws_mlp_hidden);
    h3_gpu_tensor_free(gpu->ws_qkv);
    h3_gpu_tensor_free(gpu->ws_int8_fc1);
    h3_gpu_tensor_free(gpu->ws_adaln);
    gpu->ws_mlp_fc1 = gpu->ws_mlp_hidden = gpu->ws_qkv = NULL;
    gpu->ws_int8_fc1 = gpu->ws_adaln = NULL;
    if (gpu->conv_weight_scratch) cudaFree(gpu->conv_weight_scratch);
    gpu->conv_weight_scratch = NULL;
    gpu->conv_weight_scratch_bytes = 0;
    if (gpu->int8_accum) cudaFree(gpu->int8_accum);
    for (int i = 0; i < H3_STAGE_SLOTS; i++) {
        if (gpu->stage_event_recorded[i] && gpu->stage_copied[i])
            cudaEventSynchronize(gpu->stage_copied[i]);
        if (gpu->stage_host[i]) {
            if (!gpu->stage_host_pinned)
                free(gpu->stage_host[i]);
            else if (!h3_stage_pool_give(gpu->stage_host[i],
                                         gpu->stage_host_bytes))
                cudaFreeHost(gpu->stage_host[i]);
        }
        if (gpu->stage_copied[i]) cudaEventDestroy(gpu->stage_copied[i]);
    }
    pthread_cond_destroy(&gpu->stage_free);
    pthread_mutex_destroy(&gpu->stage_lock);
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

h3_gpu_tensor *h3_gpu_tensor_new_f8(h3_gpu *gpu, size_t elements) {
    return h3_tensor_alloc(gpu, elements, H3_GPU_F8E4M3);
}

static int h3_gpu_workspace_disabled(void) {
    return h3_env_on("H3_DISABLE_GPU_WORKSPACE");
}

static h3_gpu_tensor *h3_gpu_workspace_bf16(h3_gpu *gpu, h3_gpu_tensor **slot,
                                            size_t elements) {
    if (!gpu || !slot || !elements) return NULL;
    if (h3_gpu_workspace_disabled())
        return h3_gpu_tensor_new_bf16(gpu, elements);
    if (*slot && (*slot)->elements >= elements) return *slot;
    h3_gpu_tensor_free(*slot);
    *slot = h3_gpu_tensor_new_bf16(gpu, elements);
    return *slot;
}

static void h3_gpu_workspace_release(h3_gpu_tensor *tensor) {
    if (h3_gpu_workspace_disabled()) h3_gpu_tensor_free(tensor);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner) {
        tensor->owner->stats.live_bytes -= tensor->bytes;
    }
    if (tensor->device) {
        if (tensor->pooled && tensor->owner && tensor->owner->stream)
            cudaFreeAsync(tensor->device, tensor->owner->stream);
        else
            cudaFree(tensor->device);
    }
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

/* Kept per thread: several loader lanes read at once, and a cached descriptor
 * one lane closes must not be handed to another. */
static __thread int h3_stage_fd = -1;
static __thread char h3_stage_fd_path[4096];

static int h3_gpu_open_stage_fd(h3_gpu *gpu, const char *path, char *error,
                                size_t error_size) {
    (void)gpu;
    if (!path) return -1;
    if (!h3_env_on("H3_LOAD_FD_CACHE")) {
        int fd = open(path, O_RDONLY);
        if (fd < 0 && error && error_size) {
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        }
        return fd;
    }
    if (h3_stage_fd >= 0 && h3_stage_fd_path[0] &&
        strcmp(h3_stage_fd_path, path) == 0)
        return h3_stage_fd;
    if (h3_stage_fd >= 0) {
        close(h3_stage_fd);
        h3_stage_fd = -1;
        h3_stage_fd_path[0] = '\0';
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        }
        return -1;
    }
    h3_stage_fd = fd;
    snprintf(h3_stage_fd_path, sizeof(h3_stage_fd_path), "%s", path);
    return fd;
}

/* Weight loading is disk-read bound: a single pread stream reaches about
 * 0.94 GB/s on the Spark NVMe while sixteen concurrent readers reach 4.5 GB/s,
 * because the device needs queue depth to stay busy. Large tensors are read by
 * a fan-out of pread slices so the loader tracks the parallel rate instead of
 * the single-stream rate. Set H3_LOAD_READ_THREADS=1 to force the serial path.
 */
#define H3_LOAD_READ_THREADS_MAX 32
#define H3_LOAD_READ_SLICE_MIN ((size_t)4 << 20)

typedef struct {
    const char *path;
    unsigned char *buffer;
    uint64_t offset;
    size_t bytes;
    int ok;
} h3_read_slice;

static int h3_read_fd_range(int fd, unsigned char *buffer, uint64_t offset,
                            size_t bytes) {
    size_t done = 0;
    while (done < bytes) {
        size_t chunk = bytes - done;
        if (chunk > (size_t)1 << 30) chunk = (size_t)1 << 30;
        ssize_t got = pread(fd, buffer + done, chunk, (off_t)(offset + done));
        if (got <= 0) return 0;
        done += (size_t)got;
    }
    return 1;
}

static void *h3_read_slice_worker(void *raw) {
    h3_read_slice *slice = (h3_read_slice *)raw;
    slice->ok = 0;
    int fd = open(slice->path, O_RDONLY);
    if (fd < 0) return NULL;
    slice->ok =
        h3_read_fd_range(fd, slice->buffer, slice->offset, slice->bytes);
    close(fd);
    return NULL;
}

static int h3_read_threads(void) {
    const char *value = getenv("H3_LOAD_READ_THREADS");
    /* Reads keep scaling to about 16 on this 20-core part; 24 and beyond only
     * add contention (DiT load 2.5 s at 16, 3.1 s at 20, 10.3 s at 24). */
    int threads = 16;
    if (value && *value) {
        long parsed = strtol(value, NULL, 10);
        if (parsed >= 1 && parsed <= H3_LOAD_READ_THREADS_MAX)
            threads = (int)parsed;
    }
    return threads;
}

static int h3_read_file_parallel(const char *path, uint64_t offset,
                                 void *buffer, size_t bytes, int threads) {
    /* Slice on 1 MiB boundaries so each reader issues aligned requests. */
    size_t slice_bytes = (bytes + (size_t)threads - 1) / (size_t)threads;
    slice_bytes = (slice_bytes + ((size_t)1 << 20) - 1) & ~(((size_t)1 << 20) - 1);
    h3_read_slice slices[H3_LOAD_READ_THREADS_MAX];
    pthread_t workers[H3_LOAD_READ_THREADS_MAX];
    int count = 0;
    for (size_t start = 0; start < bytes && count < threads;
         start += slice_bytes) {
        size_t span = bytes - start;
        if (span > slice_bytes) span = slice_bytes;
        slices[count].path = path;
        slices[count].buffer = (unsigned char *)buffer + start;
        slices[count].offset = offset + start;
        slices[count].bytes = span;
        slices[count].ok = 0;
        count++;
    }
    if (!count) return 0;
    int spawned = 0;
    for (int i = 1; i < count; i++) {
        if (pthread_create(&workers[i], NULL, h3_read_slice_worker,
                           &slices[i]) != 0)
            break;
        spawned = i;
    }
    h3_read_slice_worker(&slices[0]);
    int ok = slices[0].ok;
    for (int i = 1; i <= spawned; i++) {
        pthread_join(workers[i], NULL);
        if (!slices[i].ok) ok = 0;
    }
    /* Any slice we failed to spawn still has to be read. */
    for (int i = spawned + 1; i < count; i++) {
        h3_read_slice_worker(&slices[i]);
        if (!slices[i].ok) ok = 0;
    }
    return ok;
}

static int h3_read_file_at(h3_gpu *gpu, const char *path, uint64_t offset,
                           void *buffer, size_t bytes, char *error,
                           size_t error_size) {
    int threads = h3_read_threads();
    if (threads > 1 && bytes >= H3_LOAD_READ_SLICE_MIN * 2) {
        size_t want = bytes / H3_LOAD_READ_SLICE_MIN;
        if (want < (size_t)threads) threads = (int)want;
        if (h3_read_file_parallel(path, offset, buffer, bytes, threads))
            return 1;
        if (error && error_size) {
            snprintf(error, error_size, "cannot read %zu bytes from %s", bytes,
                     path);
        }
        return 0;
    }
    int fd = h3_gpu_open_stage_fd(gpu, path, error, error_size);
    if (fd < 0) return 0;
    size_t done = 0;
    while (done < bytes) {
        size_t chunk = bytes - done;
        if (chunk > (size_t)1 << 30) chunk = (size_t)1 << 30;
        ssize_t got = pread(fd, (unsigned char *)buffer + done, chunk,
                            (off_t)(offset + done));
        if (got <= 0) {
            if (error && error_size) {
                snprintf(error, error_size, "cannot read %zu bytes from %s",
                         bytes, path);
            }
            if (!h3_env_on("H3_LOAD_FD_CACHE") && fd >= 0) close(fd);
            return 0;
        }
        done += (size_t)got;
    }
    if (!h3_env_on("H3_LOAD_FD_CACHE")) close(fd);
    return 1;
}

static size_t h3_stage_chunk_bytes(void) {
    const char *value = getenv("H3_LOAD_STAGE_MIB");
    size_t mib = 128;
    if (value && *value) {
        long parsed = strtol(value, NULL, 10);
        if (parsed >= 8 && parsed <= 4096) mib = (size_t)parsed;
    }
    return mib << 20;
}

/* Call with stage_lock held. */
static int h3_gpu_ensure_stage(h3_gpu *gpu) {
    if (!gpu) return 0;
    /* Pinning is slow — around 1.5 GB/s — so the slots are allocated once at a
     * fixed size and larger tensors are copied in chunks instead of growing
     * them. */
    size_t bytes = h3_stage_chunk_bytes();
    int ready = gpu->stage_host_bytes >= bytes;
    for (int i = 0; ready && i < H3_STAGE_SLOTS; i++)
        if (!gpu->stage_host[i]) ready = 0;
    if (ready) return 1;
    for (int i = 0; i < H3_STAGE_SLOTS; i++) {
        if (gpu->stage_event_recorded[i] && gpu->stage_copied[i])
            cudaEventSynchronize(gpu->stage_copied[i]);
        gpu->stage_event_recorded[i] = 0;
        if (gpu->stage_host[i]) {
            if (gpu->stage_host_pinned) cudaFreeHost(gpu->stage_host[i]);
            else free(gpu->stage_host[i]);
            gpu->stage_host[i] = NULL;
        }
    }
    gpu->stage_host_bytes = 0;
    gpu->stage_host_pinned = 0;
    double pin_start = h3_gpu_now();
    int pinned = 1;
    for (int i = 0; i < H3_STAGE_SLOTS; i++) {
        gpu->stage_host[i] = h3_stage_pool_take(bytes);
        if (gpu->stage_host[i]) continue;
        if (cudaMallocHost(&gpu->stage_host[i], bytes) != cudaSuccess) {
            gpu->stage_host[i] = NULL;
            pinned = 0;
            break;
        }
    }
    if (pinned) {
        gpu->stage_host_bytes = bytes;
        gpu->stage_host_pinned = 1;
        gpu->stats.stage_pin_seconds += h3_gpu_now() - pin_start;
        return 1;
    }
    for (int i = 0; i < H3_STAGE_SLOTS; i++) {
        if (gpu->stage_host[i]) cudaFreeHost(gpu->stage_host[i]);
        gpu->stage_host[i] = malloc(bytes);
        if (!gpu->stage_host[i]) {
            for (int j = 0; j < i; j++) {
                free(gpu->stage_host[j]);
                gpu->stage_host[j] = NULL;
            }
            return 0;
        }
    }
    gpu->stage_host_bytes = bytes;
    gpu->stage_host_pinned = 0;
    return 1;
}

/* A slot is released as soon as its copy is enqueued; whoever claims it next
 * waits on the recorded event before overwriting the buffer. */
static int h3_stage_claim(h3_gpu *gpu, char *error, size_t error_size) {
    pthread_mutex_lock(&gpu->stage_lock);
    if (!h3_gpu_ensure_stage(gpu)) {
        pthread_mutex_unlock(&gpu->stage_lock);
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return -1;
    }
    int slot = -1;
    while (slot < 0) {
        /* Hand out slots round-robin: coming back to the slot just released
         * would mean waiting on its own copy instead of overlapping with it. */
        for (int i = 0; i < H3_STAGE_SLOTS; i++) {
            int candidate = (gpu->stage_next + i) % H3_STAGE_SLOTS;
            if (!gpu->stage_busy[candidate]) {
                slot = candidate;
                break;
            }
        }
        if (slot < 0) pthread_cond_wait(&gpu->stage_free, &gpu->stage_lock);
    }
    gpu->stage_next = (slot + 1) % H3_STAGE_SLOTS;
    gpu->stage_busy[slot] = 1;
    pthread_mutex_unlock(&gpu->stage_lock);
    return slot;
}

static void h3_stage_release(h3_gpu *gpu, int slot) {
    pthread_mutex_lock(&gpu->stage_lock);
    gpu->stage_busy[slot] = 0;
    pthread_cond_signal(&gpu->stage_free);
    pthread_mutex_unlock(&gpu->stage_lock);
}

static int h3_copy_file_to_device_chunk(h3_gpu *gpu, void *device,
                                        const char *path, uint64_t offset,
                                        size_t bytes, int slot, char *error,
                                        size_t error_size) {
    if (gpu->stage_event_recorded[slot] && gpu->stage_copied[slot] &&
        cudaEventSynchronize(gpu->stage_copied[slot]) != cudaSuccess) {
        if (error && error_size)
            snprintf(error, error_size, "cudaEventSynchronize staging failed");
        return 0;
    }
    gpu->stage_event_recorded[slot] = 0;
    double read_start = h3_gpu_now();
    if (!h3_read_file_at(gpu, path, offset, gpu->stage_host[slot], bytes, error,
                         error_size))
        return 0;
    double copy_start = h3_gpu_now();
    cudaError_t status;
    int overlap = gpu->stage_host_pinned && gpu->stream &&
                  !h3_env_on("H3_LOAD_SYNC_COPY");
    if (overlap) {
        status = cudaMemcpyAsync(device, gpu->stage_host[slot], bytes,
                                 cudaMemcpyHostToDevice, gpu->stream);
        if (status == cudaSuccess && gpu->stage_copied[slot]) {
            status = cudaEventRecord(gpu->stage_copied[slot], gpu->stream);
            if (status == cudaSuccess) gpu->stage_event_recorded[slot] = 1;
        }
    } else {
        status = cudaMemcpy(device, gpu->stage_host[slot], bytes,
                            cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cudaMemcpy failed: %s",
                     cudaGetErrorString(status));
        }
        return 0;
    }
    double copy_end = h3_gpu_now();
    pthread_mutex_lock(&gpu->stage_lock);
    gpu->stats.stage_bytes += bytes;
    gpu->stats.stage_read_seconds += copy_start - read_start;
    gpu->stats.stage_copy_seconds += copy_end - copy_start;
    pthread_mutex_unlock(&gpu->stage_lock);
    return 1;
}

/* Copying in staging-sized pieces means the pinned buffers stay small and the
 * read of one piece overlaps the copy of the previous one even within a single
 * large tensor. */
static int h3_copy_file_to_device(h3_gpu *gpu, void *device, const char *path,
                                  uint64_t offset, size_t bytes, char *error,
                                  size_t error_size) {
    if (!gpu || !device || !path || !bytes) return 0;
    int ok = 1;
    for (size_t done = 0; done < bytes;) {
        int slot = h3_stage_claim(gpu, error, error_size);
        if (slot < 0) return 0;
        size_t span = bytes - done;
        if (span > gpu->stage_host_bytes) span = gpu->stage_host_bytes;
        ok = h3_copy_file_to_device_chunk(gpu, (unsigned char *)device + done,
                                         path, offset + done, span, slot, error,
                                         error_size);
        h3_stage_release(gpu, slot);
        if (!ok) return 0;
        done += span;
    }
    return ok;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!tensor) return NULL;
    if (!h3_copy_file_to_device(gpu, tensor->device, path, file_offset,
                                tensor->bytes, gpu->error,
                                sizeof(gpu->error))) {
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    gpu->error[0] = '\0';
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(gpu, elements);
    if (!tensor) return NULL;
    if (!h3_copy_file_to_device(gpu, tensor->device, path, file_offset,
                                tensor->bytes, gpu->error,
                                sizeof(gpu->error))) {
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    gpu->error[0] = '\0';
    return tensor;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    if (!tensor || !tensor->owner || tensor->dtype != H3_GPU_BF16 ||
        tensor->elements < elements)
        return 0;
    size_t bytes = elements * sizeof(uint16_t);
    return h3_copy_file_to_device(tensor->owner, tensor->device, path,
                                  file_offset, bytes, error, error_size);
}

int h3_gpu_tensor_read_file_i8(h3_gpu_tensor *tensor, const char *path,
                               uint64_t file_offset, size_t elements,
                               char *error, size_t error_size) {
    if (!tensor || !tensor->owner || tensor->dtype != H3_GPU_I8 ||
        tensor->elements < elements)
        return 0;
    return h3_copy_file_to_device(tensor->owner, tensor->device, path,
                                  file_offset, elements, error, error_size);
}

int h3_gpu_tensor_read_file_f32(h3_gpu_tensor *tensor, const char *path,
                                uint64_t file_offset, size_t elements,
                                char *error, size_t error_size) {
    if (!tensor || !tensor->owner || tensor->dtype != H3_GPU_F32 ||
        tensor->elements < elements)
        return 0;
    return h3_copy_file_to_device(tensor->owner, tensor->device, path,
                                  file_offset, elements * sizeof(float), error,
                                  error_size);
}

int h3_gpu_tensor_read_i8(const h3_gpu_tensor *tensor, int8_t *values,
                          size_t elements) {
    if (!h3_tensor_check(tensor, H3_GPU_I8, elements) || !values) return 0;
    return cudaMemcpy(values, tensor->device, elements,
                      cudaMemcpyDeviceToHost) == cudaSuccess;
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

/* Adding the bias in a separate pass costs a full read and write of the GEMM's
 * output, which on the video VAE's shapes was 0.27 s a run. cuBLASLt can apply
 * it in the GEMM's epilogue instead. The library's plan depends only on the
 * shape, and the VAE issues five of them, so plans are cached. */
typedef struct {
    int rows;
    int input_dim;
    int output_dim;
    cublasLtMatmulAlgo_t algo;
    int valid;
} h3_lt_bias_plan;

static int h3_lt_ensure(h3_gpu *gpu) {
    if (gpu->lt_ready) return gpu->lt != NULL;
    gpu->lt_ready = 1;
    if (cublasLtCreate(&gpu->lt) != CUBLAS_STATUS_SUCCESS) {
        gpu->lt = NULL;
        return 0;
    }
    gpu->lt_workspace_bytes = (size_t)32 << 20;
    if (cudaMalloc(&gpu->lt_workspace, gpu->lt_workspace_bytes) !=
        cudaSuccess) {
        gpu->lt_workspace = NULL;
        gpu->lt_workspace_bytes = 0;
    }
    return 1;
}

/* How many of cuBLASLt's ranked candidates to actually time. Six covers the
 * spread seen on the video VAE's four shapes without making the one-time cost
 * visible. */
#define H3_LT_CANDIDATES 6

/* Counts elements where two products differ, so a candidate can be rejected for
 * disagreeing with the untuned one rather than trusted to agree. */
__global__ static void h3_f32_mismatch_kernel(const float *left,
                                              const float *right, size_t count,
                                              uint32_t *mismatches) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t a, b;
    memcpy(&a, left + index, sizeof(a));
    memcpy(&b, right + index, sizeof(b));
    if (a != b) atomicAdd(mismatches, 1u);
}

/* Times each candidate on the real operands and returns the fastest one that
 * reproduces candidate 0's output exactly. The output is left holding a garbage
 * product, so the caller must run the winner afterwards — which it does anyway.
 *
 * The bit check is the whole point. cuBLASLt ranks split-K algorithms alongside
 * single-pass ones when a workspace is offered, and those sum K in pieces, so
 * candidates do not all produce the same bits. Timing alone then makes the
 * output depend on which candidate happened to measure fastest: 768x768 flipped
 * between two md5s across repeats of one binary before this check existed.
 * Candidate 0 is what the untuned path would have used, so matching it keeps
 * tuning a pure speed decision. */
static int h3_lt_time_candidates(h3_gpu *gpu, cublasLtMatmulDesc_t desc,
                                 cublasLtMatrixLayout_t la,
                                 cublasLtMatrixLayout_t lb,
                                 cublasLtMatrixLayout_t lc, const void *a,
                                 const void *b, void *d, size_t output_bytes,
                                 cublasLtMatmulHeuristicResult_t *results,
                                 int count) {
    if (count <= 1 || h3_env_on("H3_DISABLE_LT_AUTOTUNE")) return 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    /* Reference copy of candidate 0's product, plus a mismatch counter. */
    void *reference = NULL;
    uint32_t *mismatches = NULL;
    if (cudaMalloc(&reference, output_bytes) != cudaSuccess) return 0;
    if (cudaMalloc((void **)&mismatches, sizeof(uint32_t)) != cudaSuccess) {
        cudaFree(reference);
        return 0;
    }
    cudaEvent_t start, stop;
    if (cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess) {
        cudaFree(reference);
        cudaFree(mismatches);
        return 0;
    }
    size_t elements = output_bytes / sizeof(float);
    unsigned threads = 256;
    unsigned blocks = (unsigned)((elements + threads - 1) / threads);
    int best = 0;
    float best_ms = 0.0f;
    for (int index = 0; index < count; index++) {
        /* One warm run so the first candidate is not charged for whatever the
         * stream was doing, then one timed run. */
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 1) cudaEventRecord(start, gpu->stream);
            if (cublasLtMatmul(gpu->lt, desc, &alpha, a, la, b, lb, &beta, d,
                               lc, d, lc, &results[index].algo,
                               gpu->lt_workspace, gpu->lt_workspace_bytes,
                               gpu->stream) != CUBLAS_STATUS_SUCCESS)
                goto next;
        }
        cudaEventRecord(stop, gpu->stream);
        if (cudaEventSynchronize(stop) != cudaSuccess) goto next;
        if (index == 0) {
            if (cudaMemcpyAsync(reference, d, output_bytes,
                                cudaMemcpyDeviceToDevice,
                                gpu->stream) != cudaSuccess ||
                cudaStreamSynchronize(gpu->stream) != cudaSuccess)
                goto done;
        } else {
            uint32_t differing = 1;
            if (cudaMemsetAsync(mismatches, 0, sizeof(uint32_t),
                                gpu->stream) != cudaSuccess)
                goto next;
            h3_f32_mismatch_kernel<<<blocks, threads, 0, gpu->stream>>>(
                (const float *)d, (const float *)reference, elements,
                mismatches);
            if (cudaMemcpyAsync(&differing, mismatches, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost,
                                gpu->stream) != cudaSuccess ||
                cudaStreamSynchronize(gpu->stream) != cudaSuccess || differing)
                goto next;
        }
        float elapsed;
        elapsed = 0.0f;
        if (cudaEventElapsedTime(&elapsed, start, stop) == cudaSuccess &&
            elapsed > 0.0f && (best_ms == 0.0f || elapsed < best_ms)) {
            best_ms = elapsed;
            best = index;
        }
    next:;
    }
done:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(reference);
    cudaFree(mismatches);
    return best;
}

static int h3_linear_f32_bias_fused(h3_gpu *gpu, h3_gpu_tensor *output,
                                    const h3_gpu_tensor *input,
                                    const h3_gpu_tensor *weight,
                                    const h3_gpu_tensor *bias, uint32_t rows,
                                    uint32_t input_dim, uint32_t output_dim,
                                    cublasComputeType_t compute) {
    static h3_lt_bias_plan plans[16];
    if (h3_env_on("H3_F32_SPLIT_BIAS") || !h3_lt_ensure(gpu)) return 0;
    cublasLtMatmulDesc_t desc = NULL;
    cublasLtMatrixLayout_t la = NULL, lb = NULL, lc = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasOperation_t straight = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    const void *bias_pointer = bias->device;
    int ok = 0;
    if (cublasLtMatmulDescCreate(&desc, compute, CUDA_R_32F) !=
        CUBLAS_STATUS_SUCCESS)
        return 0;
    if (cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transpose, sizeof(transpose)) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &straight, sizeof(straight)) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                       &epilogue, sizeof(epilogue)) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bias_pointer, sizeof(bias_pointer)) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, input_dim, output_dim,
                                   input_dim) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, input_dim, rows,
                                   input_dim) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, output_dim, rows,
                                   output_dim) == CUBLAS_STATUS_SUCCESS) {
        h3_lt_bias_plan *plan = NULL;
        for (unsigned i = 0; i < 16u; i++) {
            if (plans[i].valid && plans[i].rows == (int)rows &&
                plans[i].input_dim == (int)input_dim &&
                plans[i].output_dim == (int)output_dim) {
                plan = &plans[i];
                break;
            }
            if (!plans[i].valid && !plan) plan = &plans[i];
        }
        if (plan && !plan->valid) {
            cublasLtMatmulPreference_t preference = NULL;
            cublasLtMatmulHeuristicResult_t results[H3_LT_CANDIDATES];
            int found = 0;
            if (cublasLtMatmulPreferenceCreate(&preference) ==
                CUBLAS_STATUS_SUCCESS) {
                cublasLtMatmulPreferenceSetAttribute(
                    preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &gpu->lt_workspace_bytes,
                    sizeof(gpu->lt_workspace_bytes));
                if (cublasLtMatmulAlgoGetHeuristic(
                        gpu->lt, desc, la, lb, lc, lc, preference,
                        H3_LT_CANDIDATES, results, &found) ==
                        CUBLAS_STATUS_SUCCESS &&
                    found > 0) {
                    /* The heuristic's own ranking is not reliable here: asking
                     * for a different compute type on the video VAE's shapes
                     * made it pick a tiling that ran 16% faster for the same
                     * TF32 math. Every shape that reaches this cache is used
                     * ~144 times in a decode, so timing the candidates once is
                     * cheap against getting the order wrong 144 times. */
                    int best = h3_lt_time_candidates(
                        gpu, desc, la, lb, lc, weight->device, input->device,
                        output->device,
                        (size_t)rows * output_dim * sizeof(float), results,
                        found);
                    plan->algo = results[best].algo;
                    plan->rows = (int)rows;
                    plan->input_dim = (int)input_dim;
                    plan->output_dim = (int)output_dim;
                    plan->valid = 1;
                }
                cublasLtMatmulPreferenceDestroy(preference);
            }
        }
        if (plan && plan->valid) {
            float alpha = 1.0f;
            float beta = 0.0f;
            ok = cublasLtMatmul(gpu->lt, desc, &alpha, weight->device, la,
                                input->device, lb, &beta, output->device, lc,
                                output->device, lc, &plan->algo,
                                gpu->lt_workspace, gpu->lt_workspace_bytes,
                                gpu->stream) == CUBLAS_STATUS_SUCCESS;
        }
    }
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(desc);
    if (ok) gpu->stats.direct_dispatches++;
    return ok;
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

    h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
    float alpha = 1.0f;
    float beta = 0.0f;
    /* TF32 keeps FP32 range with a 10-bit mantissa and runs on the tensor
     * cores. It is worth its precision only where the GEMM is big enough to be
     * MMA-bound, so the small shapes stay exact FP32. The video VAE decodes a
     * latent the DiT has already fixed, so this cannot move the sample, only
     * add numerical noise to the decode: 45.3 dB against exact FP32, and
     * bit-reproducible run to run. H3_DISABLE_F32_TF32=1 forces exact FP32.
     * BF16x9 emulation (CUBLAS_COMPUTE_32F_EMULATED_16BFX9) was measured here
     * and is a wash: nine BF16 products at 98 TFLOP/s is no faster than FP32's
     * own 31, so cuBLAS declines the emulation and the wall does not move. */
    cublasComputeType_t compute =
        (input_dim >= 512u && output_dim >= 512u && rows >= 512u &&
         !h3_env_on("H3_DISABLE_F32_TF32"))
            ? CUBLAS_COMPUTE_32F_FAST_TF32
            : CUBLAS_COMPUTE_32F;
    if (bias && h3_linear_f32_bias_fused(gpu, output, input, weight, bias, rows,
                                         input_dim, output_dim, compute)) {
        h3_gpu_op_end(gpu);
        return 1;
    }
    cublasStatus_t status = cublasGemmEx(
        gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim, (int)rows,
        (int)input_dim, &alpha, weight->device, CUDA_R_32F, (int)input_dim,
        input->device, CUDA_R_32F, (int)input_dim, &beta, output->device,
        CUDA_R_32F, (int)output_dim, compute, CUBLAS_GEMM_DEFAULT);
    if (!h3_cublas_check(gpu, status, "cublasGemmEx linear_f32")) {
        h3_gpu_op_end(gpu);
        return 0;
    }
    gpu->stats.direct_dispatches++;

    if (bias) {
        unsigned threads = 256;
        unsigned blocks =
            (unsigned)((output_count + threads - 1) / threads);
        h3_linear_add_bias_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (float *)output->device, (const float *)bias->device, rows,
            output_dim);
        gpu->stats.direct_dispatches++;
        int ok = h3_cuda_check(gpu, cudaGetLastError(),
                               "h3_linear_add_bias_f32");
        h3_gpu_op_end(gpu);
        return ok;
    }
    h3_gpu_op_end(gpu);
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

    h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim, (int)rows,
        (int)input_dim, &alpha, weight->device, CUDA_R_16BF, (int)input_dim,
        input->device, CUDA_R_16BF, (int)input_dim, &beta, output->device,
        CUDA_R_16BF, (int)output_dim, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (!h3_cublas_check(gpu, status, "cublasGemmEx linear")) {
        h3_gpu_op_end(gpu);
        return 0;
    }
    gpu->stats.direct_dispatches++;

    if (bias) {
        unsigned threads = 256;
        unsigned blocks =
            (unsigned)((output_count + threads - 1) / threads);
        h3_linear_add_bias_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (uint16_t *)output->device, (const uint16_t *)bias->device, rows,
            output_dim);
        gpu->stats.direct_dispatches++;
        int ok = h3_cuda_check(gpu, cudaGetLastError(),
                               "h3_linear_add_bias_bf16");
        h3_gpu_op_end(gpu);
        return ok;
    }
    h3_gpu_op_end(gpu);
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

__device__ __forceinline__ static float h3_branch_value_bf16(
    const uint16_t *branch, const int32_t *branch_accum,
    float branch_row_scale, const float *branch_weight_scales, size_t index,
    uint32_t column) {
    if (branch_accum)
        return h3_bf16_bits_to_f32(h3_f32_to_bf16_bits(
            (float)branch_accum[index] * branch_row_scale *
            branch_weight_scales[column]));
    return h3_bf16_bits_to_f32(branch[index]);
}

__global__ static void h3_gate_bf16_kernel(
    const uint16_t *residual, const uint16_t *branch,
    const uint16_t *modulation, const uint32_t *row_map, uint16_t *output,
    h3_gate_args args, const int32_t *branch_accum,
    const float *branch_input_scales, const float *branch_weight_scales) {
    uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.y;
    if (row >= args.rows || column >= args.width) return;
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    float gate = h3_bf16_bits_to_f32(
        modulation[base + args.gate_slot * args.width + column]);
    size_t index = (size_t)row * args.width + column;
    float branch_row_scale = branch_accum ? branch_input_scales[row] : 0.0f;
    float value = h3_bf16_bits_to_f32(residual[index]) +
                  h3_branch_value_bf16(branch, branch_accum, branch_row_scale,
                                       branch_weight_scales, index, column) *
                      gate;
    output[index] = h3_f32_to_bf16_bits(value);
}

static int h3_take_int8_defer(h3_gpu *gpu, uint32_t rows, uint32_t width,
                              const int32_t **accum, const float **input_scales,
                              const float **weight_scales) {
    if (gpu->int8_defer_rows == rows && gpu->int8_defer_columns == width &&
        gpu->int8_accum) {
        *accum = gpu->int8_accum;
        *input_scales = gpu->int8_defer_input_scales;
        *weight_scales = gpu->int8_defer_weight_scales;
        gpu->int8_defer_rows = 0;
        gpu->int8_defer_columns = 0;
        return 1;
    }
    *accum = NULL;
    *input_scales = NULL;
    *weight_scales = NULL;
    return 0;
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
    const int32_t *branch_accum = NULL;
    const float *branch_input_scales = NULL;
    const float *branch_weight_scales = NULL;
    h3_take_int8_defer(gpu, rows, width, &branch_accum, &branch_input_scales,
                       &branch_weight_scales);
    h3_gate_args args = {rows, width, slots, gate_slot};
    dim3 threads(256, 1, 1);
    dim3 blocks((width + threads.x - 1) / threads.x, rows, 1);
    h3_gate_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)residual->device, (const uint16_t *)branch->device,
        (const uint16_t *)modulation->device,
        (const uint32_t *)row_map->device, (uint16_t *)output->device, args,
        branch_accum, branch_input_scales, branch_weight_scales);
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

    h3_gpu_tensor *fc1 =
        h3_gpu_workspace_bf16(gpu, &gpu->ws_mlp_fc1, fc1_count);
    h3_gpu_tensor *hidden =
        h3_gpu_workspace_bf16(gpu, &gpu->ws_mlp_hidden, hidden_count);
    if (!fc1 || !hidden) {
        h3_gpu_workspace_release(fc1);
        h3_gpu_workspace_release(hidden);
        return h3_gpu_fail(gpu, "MLP temp tensor allocation failed");
    }

    int ok = h3_gpu_linear_bf16(gpu, fc1, input, fc1_weight, NULL, rows,
                                input_dim, hidden_dim * 2u) &&
             h3_gpu_swiglu_bf16(gpu, hidden, fc1, rows, hidden_dim) &&
             h3_gpu_linear_bf16(gpu, output, hidden, fc2_weight, NULL, rows,
                                hidden_dim, output_dim);
    h3_gpu_workspace_release(fc1);
    h3_gpu_workspace_release(hidden);
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

} /* extern "C" — the QKV/RoPE sources below are templated */

/* A materialized BF16 projection, and the INT8 GEMM's accumulator read in its
 * place. The accumulator source rounds to BF16 on the way out so that fusing
 * the rescale into this kernel is bit-identical to writing the BF16 projection
 * and reading it back. */
struct h3_qkv_bf16_source {
    const uint16_t *qkv;
    uint32_t output_dim;
    __device__ uint16_t operator()(uint32_t row, uint32_t column) const {
        return qkv[(size_t)row * output_dim + column];
    }
};

struct h3_qkv_int8_source {
    const int32_t *accum;
    const float *input_scales;
    const float *weight_scales;
    uint32_t output_dim;
    __device__ uint16_t operator()(uint32_t row, uint32_t column) const {
        float value = (float)accum[(size_t)row * output_dim + column] *
                      input_scales[row] * weight_scales[column];
        return h3_f32_to_bf16_bits(value);
    }
};

/* Cooperative RMS: one load per thread, warp+block reduce, then RoPE.
 * Default. Opt out with H3_QKV_ROPE_SERIAL_RMS=1. */
template <typename Source>
__global__ static void h3_qkv_rope_coop_kernel(
    Source source, const uint16_t *q_weight, const uint16_t *k_weight,
    const uint16_t *rope_cos, const uint16_t *rope_sin, uint16_t *query,
    uint16_t *key, uint16_t *value, h3_qkv_rope_args args) {
    uint32_t dimension = (uint32_t)threadIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t row = (uint32_t)blockIdx.z;
    if (head >= args.heads || row >= args.sequence) return;
    uint32_t inner = args.heads * args.head_dim;
    uint32_t q_column = head * args.head_dim;
    uint32_t k_column = q_column + inner;
    uint32_t v_column = q_column + inner * 2u;
    if (args.grouped) {
        q_column = head * args.head_dim * 3u;
        k_column = q_column + args.head_dim;
        v_column = k_column + args.head_dim;
    }
    float q = 0.0f;
    float k = 0.0f;
    uint16_t v_bits = 0;
    if (dimension < args.head_dim) {
        q = h3_bf16_bits_to_f32(source(row, q_column + dimension));
        k = h3_bf16_bits_to_f32(source(row, k_column + dimension));
        v_bits = source(row, v_column + dimension);
    }
    float q_sum = q * q;
    float k_sum = k * k;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_down_sync(0xffffffff, q_sum, offset);
        k_sum += __shfl_down_sync(0xffffffff, k_sum, offset);
    }
    __shared__ float q_warp[32];
    __shared__ float k_warp[32];
    uint32_t warp = (uint32_t)threadIdx.x >> 5;
    uint32_t lane = (uint32_t)threadIdx.x & 31u;
    if (lane == 0u) {
        q_warp[warp] = q_sum;
        k_warp[warp] = k_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t nwarps = ((uint32_t)blockDim.x + 31u) >> 5;
        float qs = 0.0f;
        float ks = 0.0f;
        for (uint32_t w = 0; w < nwarps; w++) {
            qs += q_warp[w];
            ks += k_warp[w];
        }
        q_warp[0] = rsqrtf(qs / (float)args.head_dim + args.epsilon);
        k_warp[0] = rsqrtf(ks / (float)args.head_dim + args.epsilon);
    }
    __syncthreads();
    float q_inverse = q_warp[0];
    float k_inverse = k_warp[0];
    extern __shared__ float pair_smem[];
    float *q_sh = pair_smem;
    float *k_sh = pair_smem + args.head_dim;
    float q0 = 0.0f;
    float k0 = 0.0f;
    if (dimension < args.head_dim) {
        q0 = q * q_inverse * h3_bf16_bits_to_f32(q_weight[dimension]);
        k0 = k * k_inverse * h3_bf16_bits_to_f32(k_weight[dimension]);
        q_sh[dimension] = q0;
        k_sh[dimension] = k0;
    }
    __syncthreads();
    if (dimension >= args.head_dim) return;
    if (dimension < args.rope_half) {
        float q1 = q_sh[dimension + args.rope_half];
        float k1 = k_sh[dimension + args.rope_half];
        float c =
            h3_bf16_bits_to_f32(rope_cos[row * args.rope_half + dimension]);
        float s =
            h3_bf16_bits_to_f32(rope_sin[row * args.rope_half + dimension]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2u) {
        uint32_t pair = dimension - args.rope_half;
        float q1 = q_sh[pair];
        float k1 = k_sh[pair];
        float c = h3_bf16_bits_to_f32(rope_cos[row * args.rope_half + pair]);
        float s = h3_bf16_bits_to_f32(rope_sin[row * args.rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint32_t output_index =
        (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16_bits(q0);
    key[output_index] = h3_f32_to_bf16_bits(k0);
    value[output_index] = v_bits;
}

/* Shared launch geometry for both sources: one block per (head, row), one
 * thread per head dimension, shared memory for the RoPE pair exchange. */
template <typename Source>
static int h3_qkv_rope_coop_launch(h3_gpu *gpu, Source source,
                                   h3_gpu_tensor *query, h3_gpu_tensor *key,
                                   h3_gpu_tensor *value,
                                   const h3_gpu_tensor *q_norm,
                                   const h3_gpu_tensor *k_norm,
                                   const h3_gpu_tensor *rope_cos,
                                   const h3_gpu_tensor *rope_sin,
                                   h3_qkv_rope_args args) {
    uint32_t threads_x = (args.head_dim + 31u) & ~31u;
    if (threads_x < 32u) threads_x = 32u;
    dim3 blocks(1, args.heads, args.sequence);
    dim3 threads(threads_x, 1, 1);
    size_t smem = (size_t)args.head_dim * 2u * sizeof(float);
    h3_qkv_rope_coop_kernel<<<blocks, threads, smem, gpu->stream>>>(
        source, (const uint16_t *)q_norm->device,
        (const uint16_t *)k_norm->device, (const uint16_t *)rope_cos->device,
        (const uint16_t *)rope_sin->device, (uint16_t *)query->device,
        (uint16_t *)key->device, (uint16_t *)value->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_qkv_rope_coop");
}

extern "C" {

static int h3_gpu_qkv_rope_bf16_layout(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
    const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half,
    uint32_t grouped, float epsilon, int force_serial_rms) {
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
    dim3 blocks(1, heads, sequence);
    int serial_rms = force_serial_rms || h3_env_on("H3_QKV_ROPE_SERIAL_RMS");
    if (!serial_rms) {
        h3_qkv_bf16_source source = {(const uint16_t *)qkv->device,
                                     heads * head_dim * 3u};
        return h3_qkv_rope_coop_launch(gpu, source, query, key, value, q_norm,
                                       k_norm, rope_cos, rope_sin, args);
    }
    dim3 threads(head_dim, 1, 1);
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
        sequence, heads, head_dim, rope_half, 0u, epsilon, 0);
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
        sequence, heads, head_dim, rope_half, 1u, epsilon, 0);
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
               rows, heads, head_dim, rope_half, 1u, epsilon, 0);
}

struct h3_sdpa_args {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    float scale;
    uint32_t head_major_output;
    uint32_t kv_head_major;
};

__device__ static inline float h3_warp_reduce_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, mask);
    return value;
}

__device__ static inline float h3_ldg_bf16(const uint16_t *pointer) {
    return h3_bf16_bits_to_f32(__ldg(pointer));
}

__device__ static inline size_t h3_sdpa_output_index(h3_sdpa_args args,
                                                     uint32_t q_pos,
                                                     uint32_t head,
                                                     uint32_t dimension) {
    return args.head_major_output
               ? ((size_t)head * args.sequence + q_pos) * args.head_dim +
                     dimension
               : ((size_t)q_pos * args.heads + head) * args.head_dim +
                     dimension;
}

/* One warp per query row. Online softmax, no S-row in shared memory.
 * Ported from h3-hip.c wave SDPA (gfx1151 KEEP). */

/* Naive reference kernel (serial score loop on thread 0). Kept for
 * H3_SDPA_NAIVE=1 A/B and tiny correctness fallbacks. */
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

__global__ static void h3_sdpa_softmax_rows_f32_kernel(float *scores,
                                                       uint32_t sequence,
                                                       uint32_t heads) {
    uint32_t head = (uint32_t)blockIdx.x;
    uint32_t q_row = (uint32_t)blockIdx.y;
    if (head >= heads || q_row >= sequence) return;
    float *row = scores + ((size_t)head * sequence + q_row) * sequence;
    float max_score = -INFINITY;
    for (uint32_t k = 0; k < sequence; k++)
        if (row[k] > max_score) max_score = row[k];
    float sum = 0.0f;
    for (uint32_t k = 0; k < sequence; k++) {
        row[k] = expf(row[k] - max_score);
        sum += row[k];
    }
    float inverse = 1.0f / sum;
    for (uint32_t k = 0; k < sequence; k++) row[k] *= inverse;
}

__global__ static void h3_sdpa_pack_head_major_bf16_kernel(
    const uint16_t *token_major, uint16_t *head_major, uint32_t sequence,
    uint32_t heads, uint32_t head_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)sequence * heads * head_dim;
    if (index >= count) return;
    uint32_t dim = (uint32_t)(index % head_dim);
    size_t tmp = index / head_dim;
    uint32_t head = (uint32_t)(tmp % heads);
    uint32_t row = (uint32_t)(tmp / heads);
    size_t src = ((size_t)row * heads + head) * head_dim + dim;
    size_t dst = ((size_t)head * sequence + row) * head_dim + dim;
    head_major[dst] = token_major[src];
}

static int h3_gpu_sdpa_bf16_naive(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *query,
                                  const h3_gpu_tensor *key,
                                  const h3_gpu_tensor *value,
                                  uint32_t sequence, uint32_t heads,
                                  uint32_t head_dim, float scale,
                                  int head_major_output) {
    h3_sdpa_args args = {sequence, heads, head_dim, scale,
                         head_major_output ? 1u : 0u, 0u};
    dim3 threads(head_dim, 1, 1);
    dim3 blocks(heads, sequence, 1);
    size_t shared_bytes = (size_t)sequence * sizeof(float);
    h3_sdpa_bf16_kernel<<<blocks, threads, shared_bytes, gpu->stream>>>(
        (const uint16_t *)query->device, (const uint16_t *)key->device,
        (const uint16_t *)value->device, (uint16_t *)output->device, args);
    gpu->stats.mps_sdpa_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_bf16_naive");
}

static int h3_gpu_sdpa_bf16_gemm(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *query,
                                 const h3_gpu_tensor *key,
                                 const h3_gpu_tensor *value,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, float scale,
                                 int head_major_output) {
    size_t score_count = (size_t)heads * sequence * sequence;
    if (score_count / sequence / heads != sequence)
        return h3_gpu_fail(gpu, "SDPA score buffer size overflow");
    h3_gpu_tensor *scores = h3_gpu_tensor_new_f32(gpu, score_count);
    if (!scores) return h3_gpu_fail(gpu, "SDPA score alloc failed");

    int lda = (int)(heads * head_dim);
    float beta = 0.0f;
    int ok = 1;
    for (uint32_t head = 0; head < heads && ok; head++) {
        const uint16_t *query_h =
            (const uint16_t *)query->device + (size_t)head * head_dim;
        const uint16_t *key_h =
            (const uint16_t *)key->device + (size_t)head * head_dim;
        float *scores_h =
            (float *)scores->device + (size_t)head * sequence * sequence;
        /* Row-major scores = scale * Q @ K^T, same OP trick as linear_bf16. */
        cublasStatus_t status = cublasGemmEx(
            gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)sequence,
            (int)sequence, (int)head_dim, &scale, key_h, CUDA_R_16BF, lda,
            query_h, CUDA_R_16BF, lda, &beta, scores_h, CUDA_R_32F,
            (int)sequence, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (!h3_cublas_check(gpu, status, "cublasGemmEx sdpa QK")) ok = 0;
        gpu->stats.mps_linear_dispatches++;
    }
    if (ok) {
        dim3 blocks(heads, sequence, 1);
        h3_sdpa_softmax_rows_f32_kernel<<<blocks, 1, 0, gpu->stream>>>(
            (float *)scores->device, sequence, heads);
        ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_softmax");
        gpu->stats.direct_dispatches++;
    }
    h3_gpu_tensor *token_major = output;
    h3_gpu_tensor *packed = NULL;
    if (ok && head_major_output) {
        packed = h3_gpu_tensor_new_bf16(
            gpu, (size_t)sequence * heads * head_dim);
        if (!packed) {
            ok = h3_gpu_fail(gpu, "SDPA head-major temp alloc failed");
        } else {
            token_major = packed;
        }
    }
    float alpha = 1.0f;
    for (uint32_t head = 0; head < heads && ok; head++) {
        const uint16_t *value_h =
            (const uint16_t *)value->device + (size_t)head * head_dim;
        uint16_t *out_h =
            (uint16_t *)token_major->device + (size_t)head * head_dim;
        float *scores_h =
            (float *)scores->device + (size_t)head * sequence * sequence;
        /* Row-major out = scores @ V. */
        cublasStatus_t status = cublasGemmEx(
            gpu->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)head_dim,
            (int)sequence, (int)sequence, &alpha, value_h, CUDA_R_16BF, lda,
            scores_h, CUDA_R_32F, (int)sequence, &beta, out_h, CUDA_R_16BF,
            lda, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (!h3_cublas_check(gpu, status, "cublasGemmEx sdpa AV")) ok = 0;
        gpu->stats.mps_linear_dispatches++;
    }
    if (ok && head_major_output) {
        unsigned threads = 256;
        size_t count = (size_t)sequence * heads * head_dim;
        unsigned blocks =
            (unsigned)((count + threads - 1) / threads);
        h3_sdpa_pack_head_major_bf16_kernel<<<blocks, threads, 0,
                                              gpu->stream>>>(
            (const uint16_t *)token_major->device, (uint16_t *)output->device,
            sequence, heads, head_dim);
        ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_pack_head_major");
        gpu->stats.direct_dispatches++;
    }
    h3_gpu_tensor_free(packed);
    h3_gpu_tensor_free(scores);
    if (ok) gpu->stats.mps_sdpa_dispatches++;
    return ok;
}

/* Parallel SDPA: one block per (head, query row). Much faster than the
 * thread-0 serial score loop, and matches the reference math. */
__global__ static void h3_sdpa_bf16_parallel_kernel(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    uint16_t *output, h3_sdpa_args args) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *reduce = shared + args.sequence;
    uint32_t head = (uint32_t)blockIdx.x;
    uint32_t q_row = (uint32_t)blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (head >= args.heads || q_row >= args.sequence) return;

    size_t q_base = ((size_t)q_row * args.heads + head) * args.head_dim;
    /* Each thread owns a strided set of key rows for the score pass. */
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads) {
        size_t k_base = ((size_t)k_row * args.heads + head) * args.head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) {
            float q = h3_bf16_bits_to_f32(query[q_base + d]);
            float k = h3_bf16_bits_to_f32(key[k_base + d]);
            dot = fmaf(q, k, dot);
        }
        scores[k_row] = dot * args.scale;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads)
        if (scores[k_row] > local_max) local_max = scores[k_row];
    reduce[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && reduce[tid + stride] > reduce[tid])
            reduce[tid] = reduce[tid + stride];
        __syncthreads();
    }
    float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads) {
        float value = expf(scores[k_row] - max_score);
        scores[k_row] = value;
        local_sum += value;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    float inverse = 1.0f / reduce[0];
    __syncthreads();
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads)
        scores[k_row] *= inverse;
    __syncthreads();

    for (uint32_t d = tid; d < args.head_dim; d += threads) {
        float accumulated = 0.0f;
        for (uint32_t k_row = 0; k_row < args.sequence; k_row++) {
            size_t v_base =
                ((size_t)k_row * args.heads + head) * args.head_dim;
            accumulated = fmaf(scores[k_row],
                               h3_bf16_bits_to_f32(value[v_base + d]),
                               accumulated);
        }
        size_t output_index =
            args.head_major_output
                ? ((size_t)head * args.sequence + q_row) * args.head_dim + d
                : q_base + d;
        output[output_index] = h3_f32_to_bf16_bits(accumulated);
    }
}

static int h3_gpu_sdpa_bf16_parallel(h3_gpu *gpu, h3_gpu_tensor *output,
                                     const h3_gpu_tensor *query,
                                     const h3_gpu_tensor *key,
                                     const h3_gpu_tensor *value,
                                     uint32_t sequence, uint32_t heads,
                                     uint32_t head_dim, float scale,
                                     int head_major_output) {
    h3_sdpa_args args = {sequence, heads, head_dim, scale,
                         head_major_output ? 1u : 0u, 0u};
    unsigned threads = 128;
    while (threads > head_dim && threads > 32) threads >>= 1;
    if (threads < 32) threads = 32;
    size_t shared_bytes =
        (size_t)sequence * sizeof(float) + (size_t)threads * sizeof(float);
    if (shared_bytes > 48u * 1024u)
        return h3_gpu_fail(gpu, "SDPA shared memory exceeds 48KiB");
    dim3 blocks(heads, sequence, 1);
    h3_sdpa_bf16_parallel_kernel<<<blocks, threads, shared_bytes,
                                   gpu->stream>>>(
        (const uint16_t *)query->device, (const uint16_t *)key->device,
        (const uint16_t *)value->device, (uint16_t *)output->device, args);
    gpu->stats.mps_sdpa_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_bf16_parallel");
}

__global__ static void __launch_bounds__(32)
h3_sdpa_bf16_wave_kernel(const uint16_t *query, const uint16_t *key,
                         const uint16_t *value, uint16_t *output,
                         h3_sdpa_args args) {
    uint32_t q_pos = (uint32_t)blockIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t lane = (uint32_t)threadIdx.x;
    if (q_pos >= args.sequence || head >= args.heads) return;
    uint32_t q_base = (q_pos * args.heads + head) * args.head_dim;
    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            q[item] = h3_bf16_bits_to_f32(query[q_base + dimension]) *
                      args.scale;
    }
    float maximum = -1e30f;
    float sum = 0.0f;
    uint32_t stride = args.kv_head_major ? args.head_dim
                                         : args.heads * args.head_dim;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * args.head_dim
                                         : head * args.head_dim;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float partial = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            if (dimension < args.head_dim)
                partial = fmaf(q[item],
                               h3_bf16_bits_to_f32(key[k_base + dimension]),
                               partial);
        }
        float score = h3_warp_reduce_sum(partial);
        float new_max = fmaxf(maximum, score);
        float alpha = expf(maximum - new_max);
        float probability = expf(score - new_max);
#pragma unroll
        for (int item = 0; item < 4; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            float v = 0.0f;
            if (dimension < args.head_dim)
                v = h3_bf16_bits_to_f32(value[k_base + dimension]);
            acc[item] = fmaf(probability, v, acc[item] * alpha);
        }
        sum = fmaf(sum, alpha, probability);
        maximum = new_max;
        k_base += stride;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            output[h3_sdpa_output_index(args, q_pos, head, dimension)] =
                h3_f32_to_bf16_bits(acc[item] * inv);
    }
}

__global__ static void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_q2_kernel(const uint16_t *query, const uint16_t *key,
                                 const uint16_t *value, uint16_t *output,
                                 h3_sdpa_args args) {
    uint32_t q_pos = (uint32_t)blockIdx.x * 2u;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t lane = (uint32_t)threadIdx.x;
    if (q_pos >= args.sequence || head >= args.heads) return;
    int q1_live = (q_pos + 1u) < args.sequence;
    uint32_t q_base0 = (q_pos * args.heads + head) * 128u;
    uint32_t q_base1 = q_base0 + args.heads * 128u;
    float a0 = h3_bf16_bits_to_f32(query[q_base0 + lane]) * args.scale;
    float a1 = h3_bf16_bits_to_f32(query[q_base0 + 32u + lane]) * args.scale;
    float a2 = h3_bf16_bits_to_f32(query[q_base0 + 64u + lane]) * args.scale;
    float a3 = h3_bf16_bits_to_f32(query[q_base0 + 96u + lane]) * args.scale;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    if (q1_live) {
        b0 = h3_bf16_bits_to_f32(query[q_base1 + lane]) * args.scale;
        b1 = h3_bf16_bits_to_f32(query[q_base1 + 32u + lane]) * args.scale;
        b2 = h3_bf16_bits_to_f32(query[q_base1 + 64u + lane]) * args.scale;
        b3 = h3_bf16_bits_to_f32(query[q_base1 + 96u + lane]) * args.scale;
    }
    float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_a2 = 0.0f, acc_a3 = 0.0f;
    float acc_b0 = 0.0f, acc_b1 = 0.0f, acc_b2 = 0.0f, acc_b3 = 0.0f;
    float max_a = -1e30f, max_b = -1e30f, sum_a = 0.0f, sum_b = 0.0f;
    uint32_t stride = args.kv_head_major ? 128u : args.heads * 128u;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 128u
                                         : head * 128u;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float k0 = h3_bf16_bits_to_f32(key[k_base + lane]);
        float k1v = h3_bf16_bits_to_f32(key[k_base + 32u + lane]);
        float k2 = h3_bf16_bits_to_f32(key[k_base + 64u + lane]);
        float k3 = h3_bf16_bits_to_f32(key[k_base + 96u + lane]);
        float v0 = h3_bf16_bits_to_f32(value[k_base + lane]);
        float v1 = h3_bf16_bits_to_f32(value[k_base + 32u + lane]);
        float v2 = h3_bf16_bits_to_f32(value[k_base + 64u + lane]);
        float v3 = h3_bf16_bits_to_f32(value[k_base + 96u + lane]);
        float score_a = h3_warp_reduce_sum(a0 * k0 + a1 * k1v + a2 * k2 + a3 * k3);
        float new_a = fmaxf(max_a, score_a);
        float alpha_a = expf(max_a - new_a);
        float pa = expf(score_a - new_a);
        acc_a0 = fmaf(pa, v0, acc_a0 * alpha_a);
        acc_a1 = fmaf(pa, v1, acc_a1 * alpha_a);
        acc_a2 = fmaf(pa, v2, acc_a2 * alpha_a);
        acc_a3 = fmaf(pa, v3, acc_a3 * alpha_a);
        sum_a = fmaf(sum_a, alpha_a, pa);
        max_a = new_a;
        if (q1_live) {
            float score_b =
                h3_warp_reduce_sum(b0 * k0 + b1 * k1v + b2 * k2 + b3 * k3);
            float new_b = fmaxf(max_b, score_b);
            float alpha_b = expf(max_b - new_b);
            float pb = expf(score_b - new_b);
            acc_b0 = fmaf(pb, v0, acc_b0 * alpha_b);
            acc_b1 = fmaf(pb, v1, acc_b1 * alpha_b);
            acc_b2 = fmaf(pb, v2, acc_b2 * alpha_b);
            acc_b3 = fmaf(pb, v3, acc_b3 * alpha_b);
            sum_b = fmaf(sum_b, alpha_b, pb);
            max_b = new_b;
        }
        k_base += stride;
    }
    float inv_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    output[h3_sdpa_output_index(args, q_pos, head, lane)] =
        h3_f32_to_bf16_bits(acc_a0 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 32u + lane)] =
        h3_f32_to_bf16_bits(acc_a1 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 64u + lane)] =
        h3_f32_to_bf16_bits(acc_a2 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 96u + lane)] =
        h3_f32_to_bf16_bits(acc_a3 * inv_a);
    if (q1_live) {
        float inv_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 1u, head, lane)] =
            h3_f32_to_bf16_bits(acc_b0 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 32u + lane)] =
            h3_f32_to_bf16_bits(acc_b1 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 64u + lane)] =
            h3_f32_to_bf16_bits(acc_b2 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 96u + lane)] =
            h3_f32_to_bf16_bits(acc_b3 * inv_b);
    }
}

__global__ static void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_q4_kernel(const uint16_t *query, const uint16_t *key,
                                 const uint16_t *value, uint16_t *output,
                                 h3_sdpa_args args) {
    uint32_t q_pos = (uint32_t)blockIdx.x * 4u;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t lane = (uint32_t)threadIdx.x;
    if (q_pos >= args.sequence || head >= args.heads) return;
    uint32_t q_live = args.sequence - q_pos;
    if (q_live > 4u) q_live = 4u;
    uint32_t q_base0 = (q_pos * args.heads + head) * 128u;
    uint32_t q_step = args.heads * 128u;
    float q[4][4];
    float acc[4][4];
    float maximum[4];
    float sum[4];
#pragma unroll
    for (int qi = 0; qi < 4; qi++) {
        maximum[qi] = -1e30f;
        sum[qi] = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            acc[qi][d] = 0.0f;
            q[qi][d] = 0.0f;
        }
        if ((uint32_t)qi < q_live) {
            uint32_t qb = q_base0 + (uint32_t)qi * q_step;
#pragma unroll
            for (int d = 0; d < 4; d++)
                q[qi][d] = h3_bf16_bits_to_f32(
                               query[qb + (uint32_t)d * 32u + lane]) *
                           args.scale;
        }
    }
    uint32_t stride = args.kv_head_major ? 128u : args.heads * 128u;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 128u
                                         : head * 128u;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float k0 = h3_bf16_bits_to_f32(key[k_base + lane]);
        float k1 = h3_bf16_bits_to_f32(key[k_base + 32u + lane]);
        float k2 = h3_bf16_bits_to_f32(key[k_base + 64u + lane]);
        float k3 = h3_bf16_bits_to_f32(key[k_base + 96u + lane]);
        float v0 = h3_bf16_bits_to_f32(value[k_base + lane]);
        float v1 = h3_bf16_bits_to_f32(value[k_base + 32u + lane]);
        float v2 = h3_bf16_bits_to_f32(value[k_base + 64u + lane]);
        float v3 = h3_bf16_bits_to_f32(value[k_base + 96u + lane]);
#pragma unroll
        for (int qi = 0; qi < 4; qi++) {
            if ((uint32_t)qi < q_live) {
            float score = h3_warp_reduce_sum(q[qi][0] * k0 + q[qi][1] * k1 +
                                             q[qi][2] * k2 + q[qi][3] * k3);
            float new_max = fmaxf(maximum[qi], score);
            float alpha = expf(maximum[qi] - new_max);
            float p = expf(score - new_max);
            acc[qi][0] = fmaf(p, v0, acc[qi][0] * alpha);
            acc[qi][1] = fmaf(p, v1, acc[qi][1] * alpha);
            acc[qi][2] = fmaf(p, v2, acc[qi][2] * alpha);
            acc[qi][3] = fmaf(p, v3, acc[qi][3] * alpha);
            sum[qi] = fmaf(sum[qi], alpha, p);
            maximum[qi] = new_max;
            }
        }
        k_base += stride;
    }
#pragma unroll
    for (int qi = 0; qi < 4; qi++) {
        if ((uint32_t)qi < q_live) {
            float inv = sum[qi] > 0.0f ? 1.0f / sum[qi] : 0.0f;
            uint32_t qp = q_pos + (uint32_t)qi;
#pragma unroll
            for (int d = 0; d < 4; d++)
                output[h3_sdpa_output_index(args, qp, head,
                                            (uint32_t)d * 32u + lane)] =
                    h3_f32_to_bf16_bits(acc[qi][d] * inv);
        }
    }
}

__global__ static void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_q8_kernel(const uint16_t *query, const uint16_t *key,
                                 const uint16_t *value, uint16_t *output,
                                 h3_sdpa_args args) {
    uint32_t q_pos = (uint32_t)blockIdx.x * 8u;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t lane = (uint32_t)threadIdx.x;
    if (q_pos >= args.sequence || head >= args.heads) return;
    uint32_t q_live = args.sequence - q_pos;
    if (q_live > 8u) q_live = 8u;
    uint32_t q_base0 = (q_pos * args.heads + head) * 128u;
    uint32_t q_step = args.heads * 128u;
    float q[8][4];
    float acc[8][4];
    float maximum[8];
    float sum[8];
#pragma unroll
    for (int qi = 0; qi < 8; qi++) {
        maximum[qi] = -1e30f;
        sum[qi] = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            acc[qi][d] = 0.0f;
            q[qi][d] = 0.0f;
        }
        if ((uint32_t)qi < q_live) {
            uint32_t qb = q_base0 + (uint32_t)qi * q_step;
#pragma unroll
            for (int d = 0; d < 4; d++)
                q[qi][d] = h3_ldg_bf16(query + qb + (uint32_t)d * 32u + lane) *
                           args.scale;
        }
    }
    uint32_t stride = args.kv_head_major ? 128u : args.heads * 128u;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 128u
                                         : head * 128u;
    if (q_live == 8u) {
        auto consume = [&](float k0, float k1, float k2, float k3, float v0,
                           float v1, float v2, float v3) {
#pragma unroll
            for (int qi = 0; qi < 8; qi++) {
                float score = h3_warp_reduce_sum(
                    q[qi][0] * k0 + q[qi][1] * k1 + q[qi][2] * k2 +
                    q[qi][3] * k3);
                float new_max = fmaxf(maximum[qi], score);
                float alpha = __expf(maximum[qi] - new_max);
                float p = __expf(score - new_max);
                acc[qi][0] = fmaf(p, v0, acc[qi][0] * alpha);
                acc[qi][1] = fmaf(p, v1, acc[qi][1] * alpha);
                acc[qi][2] = fmaf(p, v2, acc[qi][2] * alpha);
                acc[qi][3] = fmaf(p, v3, acc[qi][3] * alpha);
                sum[qi] = fmaf(sum[qi], alpha, p);
                maximum[qi] = new_max;
            }
        };
        /* Software-pipeline: load next K/V while consuming the current pair. */
        auto load_kv = [&](uint32_t kb, float *k, float *v) {
            k[0] = h3_ldg_bf16(key + kb + lane);
            k[1] = h3_ldg_bf16(key + kb + 32u + lane);
            k[2] = h3_ldg_bf16(key + kb + 64u + lane);
            k[3] = h3_ldg_bf16(key + kb + 96u + lane);
            v[0] = h3_ldg_bf16(value + kb + lane);
            v[1] = h3_ldg_bf16(value + kb + 32u + lane);
            v[2] = h3_ldg_bf16(value + kb + 64u + lane);
            v[3] = h3_ldg_bf16(value + kb + 96u + lane);
        };
        float cur_k[4], cur_v[4], nxt_k[4], nxt_v[4];
        load_kv(k_base, cur_k, cur_v);
        for (uint32_t k_pos = 0; k_pos + 1u < args.sequence; k_pos++) {
            load_kv(k_base + stride, nxt_k, nxt_v);
            consume(cur_k[0], cur_k[1], cur_k[2], cur_k[3], cur_v[0], cur_v[1],
                    cur_v[2], cur_v[3]);
#pragma unroll
            for (int d = 0; d < 4; d++) {
                cur_k[d] = nxt_k[d];
                cur_v[d] = nxt_v[d];
            }
            k_base += stride;
        }
        consume(cur_k[0], cur_k[1], cur_k[2], cur_k[3], cur_v[0], cur_v[1],
                cur_v[2], cur_v[3]);
    } else {
        for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
            float k0 = h3_ldg_bf16(key + k_base + lane);
            float k1 = h3_ldg_bf16(key + k_base + 32u + lane);
            float k2 = h3_ldg_bf16(key + k_base + 64u + lane);
            float k3 = h3_ldg_bf16(key + k_base + 96u + lane);
            float v0 = h3_ldg_bf16(value + k_base + lane);
            float v1 = h3_ldg_bf16(value + k_base + 32u + lane);
            float v2 = h3_ldg_bf16(value + k_base + 64u + lane);
            float v3 = h3_ldg_bf16(value + k_base + 96u + lane);
#pragma unroll
            for (int qi = 0; qi < 8; qi++) {
                if ((uint32_t)qi < q_live) {
                    float score = h3_warp_reduce_sum(
                        q[qi][0] * k0 + q[qi][1] * k1 + q[qi][2] * k2 +
                        q[qi][3] * k3);
                    float new_max = fmaxf(maximum[qi], score);
                    float alpha = __expf(maximum[qi] - new_max);
                    float p = __expf(score - new_max);
                    acc[qi][0] = fmaf(p, v0, acc[qi][0] * alpha);
                    acc[qi][1] = fmaf(p, v1, acc[qi][1] * alpha);
                    acc[qi][2] = fmaf(p, v2, acc[qi][2] * alpha);
                    acc[qi][3] = fmaf(p, v3, acc[qi][3] * alpha);
                    sum[qi] = fmaf(sum[qi], alpha, p);
                    maximum[qi] = new_max;
                }
            }
            k_base += stride;
        }
    }
#pragma unroll
    for (int qi = 0; qi < 8; qi++) {
        if ((uint32_t)qi < q_live) {
            float inv = sum[qi] > 0.0f ? 1.0f / sum[qi] : 0.0f;
            uint32_t qp = q_pos + (uint32_t)qi;
#pragma unroll
            for (int d = 0; d < 4; d++)
                output[h3_sdpa_output_index(args, qp, head,
                                            (uint32_t)d * 32u + lane)] =
                    h3_f32_to_bf16_bits(acc[qi][d] * inv);
        }
    }
}

/* BF16 tensor-core SDPA for head_dim 128, flash-attention shaped.
 *
 * The wave kernels above run the score and the P·V product on FP32 CUDA cores
 * at ~4 TFLOP/s, which is a small fraction of what the BF16 MMA pipes can do
 * on this part. This kernel runs both products through
 * `mma.sync.m16n8k16.f32.bf16.bf16.f32`: one block per (64-query tile, head),
 * four warps, each owning 16 query rows for the whole head_dim.
 *
 * The PTX fragment maps are what make the online softmax cheap. For the
 * 16x8 accumulator, lane `l` holds rows `l/4` and `l/4 + 8` at columns
 * `2*(l%4)` and `+1`, so every row's max and sum reduce inside a group of
 * four lanes, and rescaling an accumulator by the row's `alpha` needs no
 * cross-lane traffic at all. The A fragment wants exactly the same
 * (row, column) pairs the accumulator already holds, so P feeds the second
 * MMA straight out of the score registers with no round trip through memory.
 */

#define H3_MMA_M 64u
#define H3_MMA_N 64u
/* 128 + 8 halves: makes the shared-memory rows land on distinct banks for
 * both the A-fragment and the B-fragment access patterns. Override with
 * -DH3_MMA_LD=128u to price the pad. */
#ifndef H3_MMA_LD
#define H3_MMA_LD 136u
#endif

__device__ __forceinline__ static void h3_mma_m16n8k16_bf16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* The probabilities and V go through the second MMA in FP16, not BF16: P lives
 * in [0, 1] and V is O(1), so FP16's 11-bit mantissa is a strict gain over
 * BF16's 8, and it is what keeps this kernel as accurate as the FP32 wave
 * kernel it replaces (relL2 vs an F32 reference 0.0017 either way, against
 * 0.013 when P is BF16). Scores stay BF16: Q and K are already BF16 in memory
 * and the products accumulate in F32 exactly. */
__device__ __forceinline__ static void h3_mma_m16n8k16_f16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ static uint32_t h3_pack_bf16_pair(float low,
                                                             float high) {
    return (uint32_t)h3_f32_to_bf16_bits(low) |
           ((uint32_t)h3_f32_to_bf16_bits(high) << 16);
}

__device__ __forceinline__ static uint32_t h3_f16_bits(float value) {
    /* Saturate rather than let an out-of-range activation become an inf that
     * would poison the whole row. */
    value = fminf(fmaxf(value, -65504.0f), 65504.0f);
    return (uint32_t)__half_as_ushort(__float2half_rn(value));
}

__device__ __forceinline__ static uint32_t h3_pack_f16_pair(float low,
                                                            float high) {
    return h3_f16_bits(low) | (h3_f16_bits(high) << 16);
}

/* Two BF16→FP16 converts as one float2→half2. Same RN rounding as
 * h3_pack_f16_pair on finite V (the MMA path saturates elsewhere). */
__device__ __forceinline__ static uint32_t h3_bf16_pair_to_f16(uint32_t raw) {
    nv_bfloat162 in;
    memcpy(&in, &raw, sizeof(in));
    half2 out = __float22half2_rn(__bfloat1622float2(in));
    uint32_t packed;
    memcpy(&packed, &out, sizeof(packed));
    return packed;
}

__global__ __launch_bounds__(128) static void h3_sdpa_bf16_mma_d128_kernel(
    const uint16_t *__restrict__ query, const uint16_t *__restrict__ key,
    const uint16_t *__restrict__ value, uint16_t *__restrict__ output,
    h3_sdpa_args args) {
    __shared__ uint16_t k_tile[H3_MMA_N * H3_MMA_LD];
    __shared__ uint16_t v_tile[H3_MMA_N * H3_MMA_LD];
    /* Q only lives in shared long enough to be read into fragments, so it
     * borrows the V tile and the loop's leading barrier hands it back. */
    uint16_t *q_tile = v_tile;

    const uint32_t sequence = args.sequence;
    const uint32_t heads = args.heads;
    const uint32_t head = (uint32_t)blockIdx.y;
    const uint32_t m0 = (uint32_t)blockIdx.x * H3_MMA_M;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t group = lane >> 2u;
    const uint32_t tig = lane & 3u;
    const uint32_t row_a = warp * 16u + group;
    const uint32_t row_b = row_a + 8u;

    /* Q tile: 64 rows x 128 halves as 64 x 64 uint32 copies. */
    for (uint32_t i = tid; i < H3_MMA_M * 64u; i += 128u) {
        uint32_t row = i >> 6u;
        uint32_t column = (i & 63u) * 2u;
        uint32_t packed = 0;
        uint32_t source = m0 + row;
        if (source < sequence)
            packed = *(const uint32_t *)(query +
                                         ((size_t)source * heads + head) *
                                             128u + column);
        *(uint32_t *)&q_tile[row * H3_MMA_LD + column] = packed;
    }
    __syncthreads();

    uint32_t q_frag[8][4];
#pragma unroll
    for (uint32_t kk = 0; kk < 8u; kk++) {
        uint32_t k0 = kk * 16u + tig * 2u;
        q_frag[kk][0] = *(const uint32_t *)&q_tile[row_a * H3_MMA_LD + k0];
        q_frag[kk][1] = *(const uint32_t *)&q_tile[row_b * H3_MMA_LD + k0];
        q_frag[kk][2] = *(const uint32_t *)&q_tile[row_a * H3_MMA_LD + k0 + 8u];
        q_frag[kk][3] = *(const uint32_t *)&q_tile[row_b * H3_MMA_LD + k0 + 8u];
    }

    float out_acc[16][4];
#pragma unroll
    for (uint32_t dt = 0; dt < 16u; dt++)
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) out_acc[dt][e] = 0.0f;
    float max_a = -INFINITY;
    float max_b = -INFINITY;
    float sum_a = 0.0f;
    float sum_b = 0.0f;

    for (uint32_t n0 = 0; n0 < sequence; n0 += H3_MMA_N) {
        __syncthreads();
        for (uint32_t i = tid; i < H3_MMA_N * 64u; i += 128u) {
            uint32_t row = i >> 6u;
            uint32_t column = (i & 63u) * 2u;
            uint32_t packed_k = 0;
            uint32_t packed_v = 0;
            uint32_t source = n0 + row;
            if (source < sequence) {
                size_t base = ((size_t)source * heads + head) * 128u + column;
                packed_k = *(const uint32_t *)(key + base);
                uint32_t raw_v = *(const uint32_t *)(value + base);
                packed_v = h3_bf16_pair_to_f16(raw_v);
            }
            *(uint32_t *)&k_tile[row * H3_MMA_LD + column] = packed_k;
            *(uint32_t *)&v_tile[row * H3_MMA_LD + column] = packed_v;
        }
        __syncthreads();

        /* S = Q Kᵀ for this 64-key tile, as eight 16x8 accumulators. */
        float score[8][4];
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) score[j][e] = 0.0f;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
            uint32_t k0 = kk * 16u + tig * 2u;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                uint32_t row = (j * 8u + group) * H3_MMA_LD;
                uint32_t b_frag[2] = {
                    *(const uint32_t *)&k_tile[row + k0],
                    *(const uint32_t *)&k_tile[row + k0 + 8u]};
                h3_mma_m16n8k16_bf16(score[j], q_frag[kk], b_frag);
            }
        }

        /* Online softmax. score[j][0..1] is row_a, [2..3] is row_b. */
        uint32_t live = sequence - n0;
        float tile_max_a = -INFINITY;
        float tile_max_b = -INFINITY;
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            uint32_t column = j * 8u + tig * 2u;
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                if (column + e < live) {
                    score[j][e] *= args.scale;
                    score[j][e + 2u] *= args.scale;
                    tile_max_a = fmaxf(tile_max_a, score[j][e]);
                    tile_max_b = fmaxf(tile_max_b, score[j][e + 2u]);
                } else {
                    score[j][e] = -INFINITY;
                    score[j][e + 2u] = -INFINITY;
                }
            }
        }
#pragma unroll
        for (uint32_t mask = 1u; mask < 4u; mask <<= 1u) {
            tile_max_a =
                fmaxf(tile_max_a, __shfl_xor_sync(0xffffffffu, tile_max_a,
                                                  (int)mask));
            tile_max_b =
                fmaxf(tile_max_b, __shfl_xor_sync(0xffffffffu, tile_max_b,
                                                  (int)mask));
        }
        float new_max_a = fmaxf(max_a, tile_max_a);
        float new_max_b = fmaxf(max_b, tile_max_b);
        float alpha_a = isfinite(new_max_a) ? __expf(max_a - new_max_a) : 1.0f;
        float alpha_b = isfinite(new_max_b) ? __expf(max_b - new_max_b) : 1.0f;
        float tile_sum_a = 0.0f;
        float tile_sum_b = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                /* Masked columns hold -inf, and expf(-inf - max) is already 0,
                 * so the comparison the obvious version needs is free. */
                float p_a = __expf(score[j][e] - new_max_a);
                float p_b = __expf(score[j][e + 2u] - new_max_b);
                score[j][e] = p_a;
                score[j][e + 2u] = p_b;
                tile_sum_a += p_a;
                tile_sum_b += p_b;
            }
        }
#pragma unroll
        for (uint32_t mask = 1u; mask < 4u; mask <<= 1u) {
            tile_sum_a += __shfl_xor_sync(0xffffffffu, tile_sum_a, (int)mask);
            tile_sum_b += __shfl_xor_sync(0xffffffffu, tile_sum_b, (int)mask);
        }
        sum_a = sum_a * alpha_a + tile_sum_a;
        sum_b = sum_b * alpha_b + tile_sum_b;
        max_a = new_max_a;
        max_b = new_max_b;
        /* After the first few tiles the running max rarely moves, so the 64
         * accumulator rescales per tile are usually a no-op. Skipping them
         * needs a warp-uniform decision, hence the vote. */
        if (__any_sync(0xffffffffu, alpha_a != 1.0f || alpha_b != 1.0f)) {
#pragma unroll
            for (uint32_t dt = 0; dt < 16u; dt++) {
                out_acc[dt][0] *= alpha_a;
                out_acc[dt][1] *= alpha_a;
                out_acc[dt][2] *= alpha_b;
                out_acc[dt][3] *= alpha_b;
            }
        }

        /* O += P V. P is already in A-fragment order. */
#pragma unroll
        for (uint32_t kk = 0; kk < 4u; kk++) {
            uint32_t p_frag[4] = {
                h3_pack_f16_pair(score[kk * 2u][0], score[kk * 2u][1]),
                h3_pack_f16_pair(score[kk * 2u][2], score[kk * 2u][3]),
                h3_pack_f16_pair(score[kk * 2u + 1u][0],
                                 score[kk * 2u + 1u][1]),
                h3_pack_f16_pair(score[kk * 2u + 1u][2],
                                 score[kk * 2u + 1u][3])};
            uint32_t n_low = (kk * 16u + tig * 2u) * H3_MMA_LD;
            uint32_t n_high = n_low + 8u * H3_MMA_LD;
#pragma unroll
            for (uint32_t dt = 0; dt < 16u; dt++) {
                uint32_t column = dt * 8u + group;
                uint32_t b_frag[2] = {
                    (uint32_t)v_tile[n_low + column] |
                        ((uint32_t)v_tile[n_low + H3_MMA_LD + column] << 16u),
                    (uint32_t)v_tile[n_high + column] |
                        ((uint32_t)v_tile[n_high + H3_MMA_LD + column] << 16u)};
                h3_mma_m16n8k16_f16(out_acc[dt], p_frag, b_frag);
            }
        }
    }

    float inverse_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    float inverse_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
    uint32_t global_a = m0 + row_a;
    uint32_t global_b = m0 + row_b;
#pragma unroll
    for (uint32_t dt = 0; dt < 16u; dt++) {
        uint32_t column = dt * 8u + tig * 2u;
        if (global_a < sequence)
            *(uint32_t *)&output[h3_sdpa_output_index(args, global_a, head,
                                                      column)] =
                h3_pack_bf16_pair(out_acc[dt][0] * inverse_a,
                                  out_acc[dt][1] * inverse_a);
        if (global_b < sequence)
            *(uint32_t *)&output[h3_sdpa_output_index(args, global_b, head,
                                                      column)] =
                h3_pack_bf16_pair(out_acc[dt][2] * inverse_b,
                                  out_acc[dt][3] * inverse_b);
    }
}

static int h3_gpu_sdpa_bf16_mma(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *query,
                                const h3_gpu_tensor *key,
                                const h3_gpu_tensor *value, uint32_t sequence,
                                uint32_t heads, float scale,
                                int head_major_output) {
    h3_sdpa_args args = {sequence, heads, 128u, scale,
                         head_major_output ? 1u : 0u, 0u};
    dim3 blocks((sequence + H3_MMA_M - 1u) / H3_MMA_M, heads, 1);
    dim3 threads(128, 1, 1);
    h3_sdpa_bf16_mma_d128_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)query->device, (const uint16_t *)key->device,
        (const uint16_t *)value->device, (uint16_t *)output->device, args);
    gpu->stats.mps_sdpa_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_bf16_mma_d128");
}

static int h3_gpu_sdpa_bf16_wave(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *query,
                                 const h3_gpu_tensor *key,
                                 const h3_gpu_tensor *value,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, float scale,
                                 int head_major_output) {
    h3_sdpa_args args = {sequence, heads, head_dim, scale,
                         head_major_output ? 1u : 0u, 0u};
    dim3 threads(32, 1, 1);
    if (head_dim == 128u && !h3_env_on("H3_SDPA_D128_Q1")) {
        /* Q8 shares K/V across eight queries. Opt back to Q4 with
         * H3_SDPA_D128_Q4=1, Q2 with H3_SDPA_D128_Q2=1. */
        if (!h3_env_on("H3_SDPA_D128_Q2")) {
            if (!h3_env_on("H3_SDPA_D128_Q4")) {
                dim3 blocks((sequence + 7u) / 8u, heads, 1);
                h3_sdpa_bf16_wave_d128_q8_kernel<<<blocks, threads, 0,
                                                   gpu->stream>>>(
                    (const uint16_t *)query->device,
                    (const uint16_t *)key->device,
                    (const uint16_t *)value->device,
                    (uint16_t *)output->device, args);
                gpu->stats.mps_sdpa_dispatches++;
                return h3_cuda_check(gpu, cudaGetLastError(),
                                     "h3_sdpa_bf16_wave_q8");
            }
            dim3 blocks((sequence + 3u) / 4u, heads, 1);
            h3_sdpa_bf16_wave_d128_q4_kernel<<<blocks, threads, 0,
                                               gpu->stream>>>(
                (const uint16_t *)query->device, (const uint16_t *)key->device,
                (const uint16_t *)value->device, (uint16_t *)output->device,
                args);
            gpu->stats.mps_sdpa_dispatches++;
            return h3_cuda_check(gpu, cudaGetLastError(),
                                 "h3_sdpa_bf16_wave_q4");
        }
        dim3 blocks((sequence + 1u) / 2u, heads, 1);
        h3_sdpa_bf16_wave_d128_q2_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (const uint16_t *)query->device, (const uint16_t *)key->device,
            (const uint16_t *)value->device, (uint16_t *)output->device, args);
        gpu->stats.mps_sdpa_dispatches++;
        return h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_bf16_wave_q2");
    }
    dim3 blocks(sequence, heads, 1);
    h3_sdpa_bf16_wave_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (const uint16_t *)query->device, (const uint16_t *)key->device,
        (const uint16_t *)value->device, (uint16_t *)output->device, args);
    gpu->stats.mps_sdpa_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_bf16_wave");
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
    if (h3_env_on("H3_SDPA_NAIVE"))
        return h3_gpu_sdpa_bf16_naive(gpu, output, query, key, value, sequence,
                                      heads, head_dim, scale,
                                      head_major_output);
    if (h3_env_on("H3_SDPA_GEMM"))
        return h3_gpu_sdpa_bf16_gemm(gpu, output, query, key, value, sequence,
                                     heads, head_dim, scale,
                                     head_major_output);
    if (h3_env_on("H3_SDPA_PARALLEL"))
        return h3_gpu_sdpa_bf16_parallel(gpu, output, query, key, value,
                                         sequence, heads, head_dim, scale,
                                         head_major_output);
    /* Tensor-core attention is the default at head_dim 128 (the DiT's shape):
     * it is ~8x the FP32 wave kernel and no less accurate. Opt back to the
     * wave kernel with H3_SDPA_WAVE=1. */
    if (head_dim == 128u &&
        (h3_env_on("H3_SDPA_MMA") || !h3_env_on("H3_SDPA_WAVE")))
        return h3_gpu_sdpa_bf16_mma(gpu, output, query, key, value, sequence,
                                    heads, scale, head_major_output);
    if (head_dim <= 128u && !h3_env_on("H3_SDPA_WAVE_OFF"))
        return h3_gpu_sdpa_bf16_wave(gpu, output, query, key, value, sequence,
                                     heads, head_dim, scale,
                                     head_major_output);
    size_t shared_need =
        (size_t)sequence * sizeof(float) + 128u * sizeof(float);
    if (shared_need > 48u * 1024u)
        return h3_gpu_sdpa_bf16_gemm(gpu, output, query, key, value, sequence,
                                     heads, head_dim, scale,
                                     head_major_output);
    return h3_gpu_sdpa_bf16_parallel(gpu, output, query, key, value, sequence,
                                     heads, head_dim, scale,
                                     head_major_output);
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    h3_gpu_op_begin(gpu, H3_GPU_OP_SDPA);
    int ok = h3_gpu_sdpa_bf16_impl(gpu, output, query, key, value, sequence,
                                   heads, head_dim, scale, 0);
    h3_gpu_op_end(gpu);
    return ok;
}

int h3_gpu_sdpa_bf16_head_major_output(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query,
    const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale) {
    h3_gpu_op_begin(gpu, H3_GPU_OP_SDPA);
    int ok = h3_gpu_sdpa_bf16_impl(gpu, output, query, key, value, sequence,
                                   heads, head_dim, scale, 1);
    h3_gpu_op_end(gpu);
    return ok;
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
    uint16_t *gated_residual, uint16_t *output, h3_gate_adaln_args args,
    const int32_t *branch_accum, const float *branch_input_scales,
    const float *branch_weight_scales) {
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
    float branch_row_scale = branch_accum ? branch_input_scales[row] : 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        size_t index = (size_t)row * args.width + column;
        float gate = h3_bf16_bits_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        uint16_t gated = h3_f32_to_bf16_bits(
            h3_bf16_bits_to_f32(residual[index]) +
            h3_branch_value_bf16(branch, branch_accum, branch_row_scale,
                                 branch_weight_scales, index, column) *
                gate);
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
    const int32_t *branch_accum = NULL;
    const float *branch_input_scales = NULL;
    const float *branch_weight_scales = NULL;
    h3_take_int8_defer(gpu, rows, width, &branch_accum, &branch_input_scales,
                       &branch_weight_scales);
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
        (uint16_t *)gated_residual->device, (uint16_t *)output->device, args,
        branch_accum, branch_input_scales, branch_weight_scales);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_gate_adaln_bf16");
}


/* Single kernel: fused gate+AdaLN then INT8 row quantize from smem.
 * Avoids writing the BF16 AdaLN temp to HBM. Opt out H3_SPLIT_ADALN_QUANT=1. */
/* branch_accum, when non-NULL, replaces the BF16 branch with the int32
 * accumulator of the INT8 GEMM that produced it, rescaled here. The BF16
 * rounding below is the same one h3_int8_apply_scales_bf16_kernel applies, in
 * the same multiplication order, so the gated result is bit for bit what the
 * split pair produced — the pair just wrote 2 bytes per element to memory and
 * read them straight back. */
__global__ static void h3_gate_adaln_quantize_int8_kernel(
    const uint16_t *residual, const uint16_t *branch,
    const uint16_t *gate_modulation, const uint32_t *row_map,
    const uint16_t *weight, const uint16_t *norm_modulation,
    uint16_t *gated_residual, int8_t *quantized, float *scales,
    h3_gate_adaln_args args, uint32_t padded_rows, float levels,
    const int32_t *branch_accum, const float *branch_input_scales,
    const float *branch_weight_scales) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= padded_rows) return;
    extern __shared__ unsigned char shared_raw[];
    float *reductions = (float *)shared_raw;
    uint16_t *row_values =
        (uint16_t *)(shared_raw + threads * sizeof(float));
    size_t out_base = (size_t)row * args.width;
    if (row >= args.rows) {
        for (uint32_t column = tid; column < args.width; column += threads)
            quantized[out_base + column] = 0;
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    size_t base = (size_t)row_map[row] * args.slots * args.width;
    float local_sum = 0.0f;
    float branch_row_scale = branch_accum ? branch_input_scales[row] : 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        size_t index = (size_t)row * args.width + column;
        float gate = h3_bf16_bits_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        float branch_value;
        if (branch_accum)
            branch_value = h3_bf16_bits_to_f32(h3_f32_to_bf16_bits(
                (float)branch_accum[index] * branch_row_scale *
                branch_weight_scales[column]));
        else
            branch_value = h3_bf16_bits_to_f32(branch[index]);
        uint16_t gated = h3_f32_to_bf16_bits(
            h3_bf16_bits_to_f32(residual[index]) + branch_value * gate);
        gated_residual[index] = gated;
        row_values[column] = gated;
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
    float local_max = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_bits_to_f32(row_values[column]) *
                           inverse * h3_bf16_bits_to_f32(weight[column]);
        float shift = h3_bf16_bits_to_f32(
            norm_modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_bits_to_f32(
            norm_modulation[base + args.scale_slot * args.width + column]);
        uint16_t bits =
            h3_f32_to_bf16_bits(normalized * (1.0f + scale) + shift);
        row_values[column] = bits;
        float absv = fabsf(h3_bf16_bits_to_f32(bits));
        if (absv > local_max) local_max = absv;
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
    float qscale = clipped_max > 0.0f ? clipped_max / levels : 1.0f / levels;
    float qinv = clipped_max > 0.0f ? levels / clipped_max : levels;
    if (tid == 0) scales[row] = qscale;
    __syncthreads();
    for (uint32_t column = tid; column < args.width; column += threads) {
        float value = h3_bf16_bits_to_f32(row_values[column]) * qinv;
        int qv = (int)rintf(value);
        if (qv > (int)levels) qv = (int)levels;
        if (qv < -(int)levels) qv = -(int)levels;
        quantized[out_base + column] = (int8_t)qv;
    }
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

#define H3_SCALE_ADD_RMS_TILE 8u

__global__ static void h3_scale_add_rms_norm_f32_kernel(
    const float *residual, const float *branch, const float *scale,
    const float *weight, float *hidden, float *norm, h3_rms_norm_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;

    extern __shared__ float reductions[];
    const float *row_residual = residual + (size_t)row * args.width;
    const float *row_branch = branch + (size_t)row * args.width;
    float *row_hidden = hidden + (size_t)row * args.width;
    float *row_norm = norm + (size_t)row * args.width;
    float vals[H3_SCALE_ADD_RMS_TILE];
    uint32_t n = 0;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float value = row_residual[column] + row_branch[column] * scale[column];
        vals[n++] = value;
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
    n = 0;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float value = vals[n++];
        row_hidden[column] = value;
        row_norm[column] = value * inverse * weight[column];
    }
}

int h3_gpu_scale_add_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *hidden,
                                  h3_gpu_tensor *norm,
                                  const h3_gpu_tensor *residual,
                                  const h3_gpu_tensor *branch,
                                  const h3_gpu_tensor *scale,
                                  const h3_gpu_tensor *norm_weight,
                                  uint32_t rows, uint32_t width,
                                  float epsilon) {
    size_t count = (size_t)rows * width;
    if (!gpu || !hidden || !norm || !residual || !branch || !scale ||
        !norm_weight || hidden->dtype != H3_GPU_F32 ||
        norm->dtype != H3_GPU_F32 || residual->dtype != H3_GPU_F32 ||
        branch->dtype != H3_GPU_F32 || scale->dtype != H3_GPU_F32 ||
        norm_weight->dtype != H3_GPU_F32 || hidden->elements < count ||
        norm->elements < count || residual->elements < count ||
        branch->elements < count || scale->elements < width ||
        norm_weight->elements < width || !rows || !width)
        return h3_gpu_fail(gpu, "invalid fused scale-add RMSNorm request");
    unsigned threads = 256;
    if (h3_env_on("H3_DISABLE_FUSED_SCALE_ADD_RMS") ||
        width > threads * H3_SCALE_ADD_RMS_TILE) {
        if (!h3_gpu_scale_add_f32(gpu, hidden, residual, branch, scale, rows,
                                  width))
            return 0;
        return h3_gpu_rms_norm_f32(gpu, norm, hidden, norm_weight, rows, width,
                                   epsilon);
    }
    h3_rms_norm_args args = {rows, width, epsilon};
    h3_scale_add_rms_norm_f32_kernel<<<rows, threads, threads * sizeof(float),
                                       gpu->stream>>>(
        (const float *)residual->device, (const float *)branch->device,
        (const float *)scale->device, (const float *)norm_weight->device,
        (float *)hidden->device, (float *)norm->device, args);
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(),
                         "h3_scale_add_rms_norm_f32");
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

/* One block per (head, row): thread 0 still does the serial fmaf sum so the
 * inverse matches the kernel above bit for bit, then every lane applies RoPE.
 * Opt out H3_VIDEO_QKV_SERIAL_RMS=1. */
__global__ static void h3_video_qkv_rope_f32_coop_kernel(
    const float *qkv, const float *rope_cos, const float *rope_sin,
    float *query, float *key, float *value, h3_video_qkv_rope_args args) {
    uint32_t dimension = (uint32_t)threadIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t row = (uint32_t)blockIdx.z;
    if (head >= args.heads || row >= args.sequence) return;
    size_t base =
        ((size_t)row * args.heads + head) * args.head_dim * 3u;
    __shared__ float q_inv;
    __shared__ float k_inv;
    extern __shared__ float pair_smem[];
    float *q_sh = pair_smem;
    float *k_sh = pair_smem + args.head_dim;
    float *v_sh = pair_smem + args.head_dim * 2u;
    if (dimension < args.head_dim) {
        q_sh[dimension] = qkv[base + dimension];
        k_sh[dimension] = qkv[base + args.head_dim + dimension];
        v_sh[dimension] = qkv[base + args.head_dim * 2u + dimension];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) {
            q_sum = fmaf(q_sh[d], q_sh[d], q_sum);
            k_sum = fmaf(k_sh[d], k_sh[d], k_sum);
        }
        q_inv = rsqrtf(q_sum / (float)args.head_dim + args.epsilon);
        k_inv = rsqrtf(k_sum / (float)args.head_dim + args.epsilon);
    }
    __syncthreads();
    float q0 = 0.0f;
    float k0 = 0.0f;
    if (dimension < args.head_dim) {
        q0 = q_sh[dimension] * q_inv;
        k0 = k_sh[dimension] * k_inv;
        q_sh[dimension] = q0;
        k_sh[dimension] = k0;
    }
    __syncthreads();
    if (dimension >= args.head_dim) return;
    if (dimension < args.rope_half) {
        float q1 = q_sh[dimension + args.rope_half];
        float k1 = k_sh[dimension + args.rope_half];
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2u) {
        uint32_t pair = dimension - args.rope_half;
        float q1 = q_sh[pair];
        float k1 = k_sh[pair];
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    size_t output_index =
        ((size_t)row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = v_sh[dimension];
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
    if (!h3_env_on("H3_VIDEO_QKV_SERIAL_RMS")) {
        uint32_t threads_x = (head_dim + 31u) & ~31u;
        if (threads_x < 32u) threads_x = 32u;
        dim3 blocks(1, heads, sequence);
        dim3 threads(threads_x, 1, 1);
        size_t smem = (size_t)head_dim * 3u * sizeof(float);
        h3_video_qkv_rope_f32_coop_kernel<<<blocks, threads, smem,
                                            gpu->stream>>>(
            (const float *)qkv->device, (const float *)rope_cos->device,
            (const float *)rope_sin->device, (float *)query->device,
            (float *)key->device, (float *)value->device, args);
    } else {
        dim3 threads(32, 1, 1);
        dim3 blocks((head_dim + threads.x - 1) / threads.x, heads, sequence);
        h3_video_qkv_rope_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (const float *)qkv->device, (const float *)rope_cos->device,
            (const float *)rope_sin->device, (float *)query->device,
            (float *)key->device, (float *)value->device, args);
    }
    gpu->stats.direct_dispatches++;
    return h3_cuda_check(gpu, cudaGetLastError(), "h3_video_qkv_rope_f32");
}

__global__ static void h3_sdpa_f32_kernel(const float *query, const float *key,
                                          const float *value, float *output,
                                          h3_sdpa_args args) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *reduce = shared + args.sequence;
    uint32_t head = (uint32_t)blockIdx.x;
    uint32_t q_row = (uint32_t)blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (head >= args.heads || q_row >= args.sequence) return;
    size_t q_base = ((size_t)q_row * args.heads + head) * args.head_dim;
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads) {
        size_t k_base = ((size_t)k_row * args.heads + head) * args.head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++)
            dot = fmaf(query[q_base + d], key[k_base + d], dot);
        scores[k_row] = dot * args.scale;
    }
    __syncthreads();
    float local_max = -INFINITY;
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads)
        if (scores[k_row] > local_max) local_max = scores[k_row];
    reduce[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && reduce[tid + stride] > reduce[tid])
            reduce[tid] = reduce[tid + stride];
        __syncthreads();
    }
    float max_score = reduce[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads) {
        float value = expf(scores[k_row] - max_score);
        scores[k_row] = value;
        local_sum += value;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    float inverse = 1.0f / reduce[0];
    __syncthreads();
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads)
        scores[k_row] *= inverse;
    __syncthreads();
    for (uint32_t d = tid; d < args.head_dim; d += threads) {
        float accumulated = 0.0f;
        for (uint32_t k_row = 0; k_row < args.sequence; k_row++) {
            size_t v_base =
                ((size_t)k_row * args.heads + head) * args.head_dim;
            accumulated = fmaf(scores[k_row], value[v_base + d], accumulated);
        }
        output[q_base + d] = accumulated;
    }
}

__global__ static void __launch_bounds__(32)
h3_sdpa_f32_wave_kernel(const float *query, const float *key,
                        const float *value, float *output, h3_sdpa_args args) {
    uint32_t q_pos = (uint32_t)blockIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t lane = (uint32_t)threadIdx.x;
    if (q_pos >= args.sequence || head >= args.heads) return;
    uint32_t q_base = (q_pos * args.heads + head) * args.head_dim;
    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            q[item] = query[q_base + dimension] * args.scale;
    }
    float maximum = -1e30f;
    float sum = 0.0f;
    uint32_t stride = args.heads * args.head_dim;
    uint32_t k_base = head * args.head_dim;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float partial = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            if (dimension < args.head_dim)
                partial = fmaf(q[item], key[k_base + dimension], partial);
        }
        float score = h3_warp_reduce_sum(partial);
        float new_max = fmaxf(maximum, score);
        float alpha = expf(maximum - new_max);
        float probability = expf(score - new_max);
#pragma unroll
        for (int item = 0; item < 4; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            float v = dimension < args.head_dim ? value[k_base + dimension]
                                                : 0.0f;
            acc[item] = fmaf(probability, v, acc[item] * alpha);
        }
        sum = fmaf(sum, alpha, probability);
        maximum = new_max;
        k_base += stride;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            output[q_base + dimension] = acc[item] * inv;
    }
}

__global__ static void __launch_bounds__(32)
h3_sdpa_f32_wave_d64_q2_kernel(const float *query, const float *key,
                               const float *value, float *output,
                               h3_sdpa_args args) {
    uint32_t q_pos = (uint32_t)blockIdx.x * 2u;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t lane = (uint32_t)threadIdx.x;
    if (q_pos >= args.sequence || head >= args.heads) return;
    int q1_live = (q_pos + 1u) < args.sequence;
    uint32_t q_base0 = (q_pos * args.heads + head) * 64u;
    uint32_t q_base1 = q_base0 + args.heads * 64u;
    float a0 = query[q_base0 + lane] * args.scale;
    float a1 = query[q_base0 + 32u + lane] * args.scale;
    float b0 = 0.0f, b1 = 0.0f;
    if (q1_live) {
        b0 = query[q_base1 + lane] * args.scale;
        b1 = query[q_base1 + 32u + lane] * args.scale;
    }
    float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_b0 = 0.0f, acc_b1 = 0.0f;
    float max_a = -1e30f, max_b = -1e30f, sum_a = 0.0f, sum_b = 0.0f;
    uint32_t stride = args.heads * 64u;
    uint32_t k_base = head * 64u;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float k0 = key[k_base + lane];
        float k1v = key[k_base + 32u + lane];
        float v0 = value[k_base + lane];
        float v1 = value[k_base + 32u + lane];
        float score_a = h3_warp_reduce_sum(a0 * k0 + a1 * k1v);
        float new_a = fmaxf(max_a, score_a);
        float alpha_a = expf(max_a - new_a);
        float pa = expf(score_a - new_a);
        acc_a0 = fmaf(pa, v0, acc_a0 * alpha_a);
        acc_a1 = fmaf(pa, v1, acc_a1 * alpha_a);
        sum_a = fmaf(sum_a, alpha_a, pa);
        max_a = new_a;
        if (q1_live) {
            float score_b = h3_warp_reduce_sum(b0 * k0 + b1 * k1v);
            float new_b = fmaxf(max_b, score_b);
            float alpha_b = expf(max_b - new_b);
            float pb = expf(score_b - new_b);
            acc_b0 = fmaf(pb, v0, acc_b0 * alpha_b);
            acc_b1 = fmaf(pb, v1, acc_b1 * alpha_b);
            sum_b = fmaf(sum_b, alpha_b, pb);
            max_b = new_b;
        }
        k_base += stride;
    }
    float inv_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    output[q_base0 + lane] = acc_a0 * inv_a;
    output[q_base0 + 32u + lane] = acc_a1 * inv_a;
    if (q1_live) {
        float inv_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
        output[q_base1 + lane] = acc_b0 * inv_b;
        output[q_base1 + 32u + lane] = acc_b1 * inv_b;
    }
}

__global__ static void __launch_bounds__(32)
h3_sdpa_f32_wave_d64_q4_kernel(const float *query, const float *key,
                               const float *value, float *output,
                               h3_sdpa_args args) {
    uint32_t q_pos = (uint32_t)blockIdx.x * 4u;
    uint32_t head = (uint32_t)blockIdx.y;
    uint32_t lane = (uint32_t)threadIdx.x;
    if (q_pos >= args.sequence || head >= args.heads) return;
    uint32_t q_live = args.sequence - q_pos;
    if (q_live > 4u) q_live = 4u;
    uint32_t q_base0 = (q_pos * args.heads + head) * 64u;
    uint32_t q_step = args.heads * 64u;
    float q0[4], q1[4];
    float acc0[4], acc1[4], maximum[4], sum[4];
#pragma unroll
    for (int qi = 0; qi < 4; qi++) {
        maximum[qi] = -1e30f;
        sum[qi] = 0.0f;
        acc0[qi] = 0.0f;
        acc1[qi] = 0.0f;
        q0[qi] = 0.0f;
        q1[qi] = 0.0f;
        if ((uint32_t)qi < q_live) {
            uint32_t qb = q_base0 + (uint32_t)qi * q_step;
            q0[qi] = query[qb + lane] * args.scale;
            q1[qi] = query[qb + 32u + lane] * args.scale;
        }
    }
    uint32_t stride = args.heads * 64u;
    uint32_t k_base = head * 64u;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float k0 = key[k_base + lane];
        float k1v = key[k_base + 32u + lane];
        float v0 = value[k_base + lane];
        float v1 = value[k_base + 32u + lane];
#pragma unroll
        for (int qi = 0; qi < 4; qi++) {
            if ((uint32_t)qi < q_live) {
                float score = h3_warp_reduce_sum(q0[qi] * k0 + q1[qi] * k1v);
                float new_max = fmaxf(maximum[qi], score);
                float alpha = expf(maximum[qi] - new_max);
                float p = expf(score - new_max);
                acc0[qi] = fmaf(p, v0, acc0[qi] * alpha);
                acc1[qi] = fmaf(p, v1, acc1[qi] * alpha);
                sum[qi] = fmaf(sum[qi], alpha, p);
                maximum[qi] = new_max;
            }
        }
        k_base += stride;
    }
#pragma unroll
    for (int qi = 0; qi < 4; qi++) {
        if ((uint32_t)qi < q_live) {
            float inv = sum[qi] > 0.0f ? 1.0f / sum[qi] : 0.0f;
            uint32_t qb = q_base0 + (uint32_t)qi * q_step;
            output[qb + lane] = acc0[qi] * inv;
            output[qb + 32u + lane] = acc1[qi] * inv;
        }
    }
}

/* Tensor-core attention for the video VAE's F32, head_dim 64 attention. Same
 * flash shape as the DiT's BF16 kernel: one block per 64-query tile per head,
 * four warps, 64-key stride, online softmax in registers. The inputs are F32
 * in memory and get narrowed to FP16 on the way into the tiles — 11 bits of
 * mantissa, more than either BF16 or TF32 offers, and the products still
 * accumulate in F32. */
enum { H3_MMA_F32_M = 64u, H3_MMA_F32_N = 64u, H3_MMA_F32_LD = 72u };

/* Each block re-reads its head's whole K and V, so the row tile sets how many
 * times that traffic is paid: at M=64 a 1797-row head needs 29 blocks and the
 * video VAE's attention moves 854 MB per call. Widening the tile costs warps,
 * not registers or shared memory, because a warp owns 16 rows either way. */
#ifndef H3_MMA_F32_WARPS
#define H3_MMA_F32_WARPS 8u
#endif
#define H3_MMA_F32_ROWS (H3_MMA_F32_WARPS * 16u)
#define H3_MMA_F32_THREADS (H3_MMA_F32_WARPS * 32u)
/* Big enough for the four K/V tiles, or for Q's two halves, whichever wants
 * more: past eight warps the row tile is the larger of the two. */
#define H3_MMA_F32_SHARED                                                      \
    (4u * H3_MMA_F32_N > 2u * H3_MMA_F32_ROWS                                  \
         ? 4u * H3_MMA_F32_N * H3_MMA_F32_LD                                   \
         : 2u * H3_MMA_F32_ROWS * H3_MMA_F32_LD)

/* One FP16 keeps 11 bits of mantissa, which leaves the scores about 0.4% off an
 * exact F32 dot product — fine for the DiT, whose inputs are BF16 anyway, but
 * this attention is the only 16-bit step in an otherwise F32 decoder. So each
 * F32 splits into a high and a low FP16 and Q·Kᵀ is summed as
 * hi·hi + hi·lo + lo·hi, which recovers ~22 bits at the cost of three MMAs per
 * score tile instead of one. P·V stays single FP16: P is in [0, 1] and enters a
 * positive-weight average, so it contributes an order of magnitude less error.
 */
__device__ __forceinline__ static uint32_t h3_pack_f16_hi_pair(float low,
                                                               float high) {
    return h3_pack_f16_pair(low, high);
}

__device__ __forceinline__ static uint32_t h3_pack_f16_lo_pair(float low,
                                                               float high) {
    float low_residual =
        low - __half2float(__ushort_as_half((unsigned short)h3_f16_bits(low)));
    float high_residual =
        high -
        __half2float(__ushort_as_half((unsigned short)h3_f16_bits(high)));
    return h3_pack_f16_pair(low_residual, high_residual);
}

/* hi then lo of two F32s, reusing the hi half2 so the residual convert does
 * not redo the same two RN rounds. Saturate like h3_f16_bits so infs cannot
 * leak into the MMA. */
__device__ __forceinline__ static void h3_pack_f16_split_pair(
    float low, float high, uint32_t *hi_out, uint32_t *lo_out) {
    float2 sat = make_float2(fminf(fmaxf(low, -65504.0f), 65504.0f),
                             fminf(fmaxf(high, -65504.0f), 65504.0f));
    half2 hi = __float22half2_rn(sat);
    memcpy(hi_out, &hi, sizeof(*hi_out));
    float2 back = __half22float2(hi);
    float2 resid = make_float2(
        fminf(fmaxf(low - back.x, -65504.0f), 65504.0f),
        fminf(fmaxf(high - back.y, -65504.0f), 65504.0f));
    half2 lo = __float22half2_rn(resid);
    memcpy(lo_out, &lo, sizeof(*lo_out));
}

__global__ __launch_bounds__(H3_MMA_F32_THREADS) static void
h3_sdpa_f32_mma_d64_kernel(
    const float *__restrict__ query, const float *__restrict__ key,
    const float *__restrict__ value, float *__restrict__ output,
    h3_sdpa_args args) {
    /* One block of shared memory carved into the four K/V tiles, so that Q can
     * borrow it in halves: at H3_MMA_F32_WARPS=8 each Q half needs two tiles'
     * worth of rows, which separate arrays could not guarantee were adjacent. */
    __shared__ uint16_t tiles[H3_MMA_F32_SHARED];
    uint16_t *k_tile = tiles;
    uint16_t *k_low_tile = tiles + H3_MMA_F32_N * H3_MMA_F32_LD;
    uint16_t *v_tile = tiles + 2u * H3_MMA_F32_N * H3_MMA_F32_LD;
    uint16_t *v_low_tile = tiles + 3u * H3_MMA_F32_N * H3_MMA_F32_LD;
    /* Q is read into fragments before the loop's first barrier, so its two
     * halves borrow the K and V tiles. */
    uint16_t *q_tile = tiles;
    uint16_t *q_low_tile = tiles + H3_MMA_F32_ROWS * H3_MMA_F32_LD;

    const uint32_t sequence = args.sequence;
    const uint32_t heads = args.heads;
    const uint32_t head = (uint32_t)blockIdx.y;
    const uint32_t m0 = (uint32_t)blockIdx.x * H3_MMA_F32_ROWS;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t group = lane >> 2u;
    const uint32_t tig = lane & 3u;
    const uint32_t row_a = warp * 16u + group;
    const uint32_t row_b = row_a + 8u;

    for (uint32_t i = tid; i < H3_MMA_F32_ROWS * 32u;
         i += H3_MMA_F32_THREADS) {
        uint32_t row = i >> 5u;
        uint32_t column = (i & 31u) * 2u;
        uint32_t packed = 0;
        uint32_t packed_low = 0;
        uint32_t source = m0 + row;
        if (source < sequence) {
            const float *base =
                query + ((size_t)source * heads + head) * 64u + column;
            h3_pack_f16_split_pair(base[0], base[1], &packed, &packed_low);
        }
        *(uint32_t *)&q_tile[row * H3_MMA_F32_LD + column] = packed;
        *(uint32_t *)&q_low_tile[row * H3_MMA_F32_LD + column] = packed_low;
    }
    __syncthreads();

    uint32_t q_frag[4][4];
    uint32_t q_low_frag[4][4];
#pragma unroll
    for (uint32_t kk = 0; kk < 4u; kk++) {
        uint32_t k0 = kk * 16u + tig * 2u;
        q_frag[kk][0] = *(const uint32_t *)&q_tile[row_a * H3_MMA_F32_LD + k0];
        q_frag[kk][1] = *(const uint32_t *)&q_tile[row_b * H3_MMA_F32_LD + k0];
        q_frag[kk][2] =
            *(const uint32_t *)&q_tile[row_a * H3_MMA_F32_LD + k0 + 8u];
        q_frag[kk][3] =
            *(const uint32_t *)&q_tile[row_b * H3_MMA_F32_LD + k0 + 8u];
        q_low_frag[kk][0] =
            *(const uint32_t *)&q_low_tile[row_a * H3_MMA_F32_LD + k0];
        q_low_frag[kk][1] =
            *(const uint32_t *)&q_low_tile[row_b * H3_MMA_F32_LD + k0];
        q_low_frag[kk][2] =
            *(const uint32_t *)&q_low_tile[row_a * H3_MMA_F32_LD + k0 + 8u];
        q_low_frag[kk][3] =
            *(const uint32_t *)&q_low_tile[row_b * H3_MMA_F32_LD + k0 + 8u];
    }

    float out_acc[8][4];
#pragma unroll
    for (uint32_t dt = 0; dt < 8u; dt++)
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) out_acc[dt][e] = 0.0f;
    float max_a = -INFINITY;
    float max_b = -INFINITY;
    float sum_a = 0.0f;
    float sum_b = 0.0f;

    for (uint32_t n0 = 0; n0 < sequence; n0 += H3_MMA_F32_N) {
        __syncthreads();
        for (uint32_t i = tid; i < H3_MMA_F32_N * 32u;
             i += H3_MMA_F32_THREADS) {
            uint32_t row = i >> 5u;
            uint32_t column = (i & 31u) * 2u;
            uint32_t packed_k = 0;
            uint32_t packed_k_low = 0;
            uint32_t packed_v = 0;
            uint32_t packed_v_low = 0;
            uint32_t source = n0 + row;
            if (source < sequence) {
                size_t base = ((size_t)source * heads + head) * 64u + column;
                h3_pack_f16_split_pair(key[base], key[base + 1], &packed_k,
                                       &packed_k_low);
                h3_pack_f16_split_pair(value[base], value[base + 1], &packed_v,
                                       &packed_v_low);
            }
            *(uint32_t *)&k_tile[row * H3_MMA_F32_LD + column] = packed_k;
            *(uint32_t *)&k_low_tile[row * H3_MMA_F32_LD + column] =
                packed_k_low;
            *(uint32_t *)&v_tile[row * H3_MMA_F32_LD + column] = packed_v;
            *(uint32_t *)&v_low_tile[row * H3_MMA_F32_LD + column] =
                packed_v_low;
        }
        __syncthreads();

        float score[8][4];
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) score[j][e] = 0.0f;
#pragma unroll
        for (uint32_t kk = 0; kk < 4u; kk++) {
            uint32_t k0 = kk * 16u + tig * 2u;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                uint32_t row = (j * 8u + group) * H3_MMA_F32_LD;
                uint32_t b_frag[2] = {
                    *(const uint32_t *)&k_tile[row + k0],
                    *(const uint32_t *)&k_tile[row + k0 + 8u]};
                uint32_t b_low_frag[2] = {
                    *(const uint32_t *)&k_low_tile[row + k0],
                    *(const uint32_t *)&k_low_tile[row + k0 + 8u]};
                h3_mma_m16n8k16_f16(score[j], q_frag[kk], b_frag);
                h3_mma_m16n8k16_f16(score[j], q_frag[kk], b_low_frag);
                h3_mma_m16n8k16_f16(score[j], q_low_frag[kk], b_frag);
            }
        }

        uint32_t live = sequence - n0;
        float tile_max_a = -INFINITY;
        float tile_max_b = -INFINITY;
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            uint32_t column = j * 8u + tig * 2u;
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                if (column + e < live) {
                    score[j][e] *= args.scale;
                    score[j][e + 2u] *= args.scale;
                    tile_max_a = fmaxf(tile_max_a, score[j][e]);
                    tile_max_b = fmaxf(tile_max_b, score[j][e + 2u]);
                } else {
                    score[j][e] = -INFINITY;
                    score[j][e + 2u] = -INFINITY;
                }
            }
        }
#pragma unroll
        for (uint32_t mask = 1u; mask < 4u; mask <<= 1u) {
            tile_max_a = fmaxf(
                tile_max_a, __shfl_xor_sync(0xffffffffu, tile_max_a, (int)mask));
            tile_max_b = fmaxf(
                tile_max_b, __shfl_xor_sync(0xffffffffu, tile_max_b, (int)mask));
        }
        float new_max_a = fmaxf(max_a, tile_max_a);
        float new_max_b = fmaxf(max_b, tile_max_b);
        float alpha_a = isfinite(new_max_a) ? __expf(max_a - new_max_a) : 1.0f;
        float alpha_b = isfinite(new_max_b) ? __expf(max_b - new_max_b) : 1.0f;
        float tile_sum_a = 0.0f;
        float tile_sum_b = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                float p_a = score[j][e] == -INFINITY
                                ? 0.0f
                                : __expf(score[j][e] - new_max_a);
                float p_b = score[j][e + 2u] == -INFINITY
                                ? 0.0f
                                : __expf(score[j][e + 2u] - new_max_b);
                score[j][e] = p_a;
                score[j][e + 2u] = p_b;
                tile_sum_a += p_a;
                tile_sum_b += p_b;
            }
        }
#pragma unroll
        for (uint32_t mask = 1u; mask < 4u; mask <<= 1u) {
            tile_sum_a += __shfl_xor_sync(0xffffffffu, tile_sum_a, (int)mask);
            tile_sum_b += __shfl_xor_sync(0xffffffffu, tile_sum_b, (int)mask);
        }
        sum_a = sum_a * alpha_a + tile_sum_a;
        sum_b = sum_b * alpha_b + tile_sum_b;
        max_a = new_max_a;
        max_b = new_max_b;
#pragma unroll
        for (uint32_t dt = 0; dt < 8u; dt++) {
            out_acc[dt][0] *= alpha_a;
            out_acc[dt][1] *= alpha_a;
            out_acc[dt][2] *= alpha_b;
            out_acc[dt][3] *= alpha_b;
        }

#pragma unroll
        for (uint32_t kk = 0; kk < 4u; kk++) {
            uint32_t p_frag[4] = {
                h3_pack_f16_hi_pair(score[kk * 2u][0], score[kk * 2u][1]),
                h3_pack_f16_hi_pair(score[kk * 2u][2], score[kk * 2u][3]),
                h3_pack_f16_hi_pair(score[kk * 2u + 1u][0],
                                    score[kk * 2u + 1u][1]),
                h3_pack_f16_hi_pair(score[kk * 2u + 1u][2],
                                    score[kk * 2u + 1u][3])};
            uint32_t p_low_frag[4] = {
                h3_pack_f16_lo_pair(score[kk * 2u][0], score[kk * 2u][1]),
                h3_pack_f16_lo_pair(score[kk * 2u][2], score[kk * 2u][3]),
                h3_pack_f16_lo_pair(score[kk * 2u + 1u][0],
                                    score[kk * 2u + 1u][1]),
                h3_pack_f16_lo_pair(score[kk * 2u + 1u][2],
                                    score[kk * 2u + 1u][3])};
            uint32_t n_low = (kk * 16u + tig * 2u) * H3_MMA_F32_LD;
            uint32_t n_high = n_low + 8u * H3_MMA_F32_LD;
#pragma unroll
            for (uint32_t dt = 0; dt < 8u; dt++) {
                uint32_t column = dt * 8u + group;
                uint32_t b_frag[2] = {
                    (uint32_t)v_tile[n_low + column] |
                        ((uint32_t)v_tile[n_low + H3_MMA_F32_LD + column]
                         << 16u),
                    (uint32_t)v_tile[n_high + column] |
                        ((uint32_t)v_tile[n_high + H3_MMA_F32_LD + column]
                         << 16u)};
                uint32_t b_low_frag[2] = {
                    (uint32_t)v_low_tile[n_low + column] |
                        ((uint32_t)v_low_tile[n_low + H3_MMA_F32_LD + column]
                         << 16u),
                    (uint32_t)v_low_tile[n_high + column] |
                        ((uint32_t)v_low_tile[n_high + H3_MMA_F32_LD + column]
                         << 16u)};
                h3_mma_m16n8k16_f16(out_acc[dt], p_frag, b_frag);
                h3_mma_m16n8k16_f16(out_acc[dt], p_frag, b_low_frag);
                h3_mma_m16n8k16_f16(out_acc[dt], p_low_frag, b_frag);
            }
        }
    }

    float inverse_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    float inverse_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
    uint32_t global_a = m0 + row_a;
    uint32_t global_b = m0 + row_b;
#pragma unroll
    for (uint32_t dt = 0; dt < 8u; dt++) {
        uint32_t column = dt * 8u + tig * 2u;
        if (global_a < sequence) {
            float *dst =
                output + h3_sdpa_output_index(args, global_a, head, column);
            dst[0] = out_acc[dt][0] * inverse_a;
            dst[1] = out_acc[dt][1] * inverse_a;
        }
        if (global_b < sequence) {
            float *dst =
                output + h3_sdpa_output_index(args, global_b, head, column);
            dst[0] = out_acc[dt][2] * inverse_b;
            dst[1] = out_acc[dt][3] * inverse_b;
        }
    }
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
    h3_gpu_op_begin(gpu, H3_GPU_OP_SDPA);
    h3_sdpa_args args = {sequence, heads, head_dim, scale, 0u, 0u};
    int ok = 1;
    if (head_dim == 64u && !h3_env_on("H3_SDPA_F32_WAVE") &&
        !h3_env_on("H3_SDPA_PARALLEL") && !h3_env_on("H3_SDPA_WAVE_OFF")) {
        dim3 blocks((sequence + H3_MMA_F32_ROWS - 1u) / H3_MMA_F32_ROWS, heads,
                    1);
        dim3 threads(H3_MMA_F32_THREADS, 1, 1);
        h3_sdpa_f32_mma_d64_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (const float *)query->device, (const float *)key->device,
            (const float *)value->device, (float *)output->device, args);
        gpu->stats.mps_sdpa_dispatches++;
        ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_f32_mma_d64");
        h3_gpu_op_end(gpu);
        return ok;
    }
    if (head_dim <= 128u && !h3_env_on("H3_SDPA_PARALLEL") &&
        !h3_env_on("H3_SDPA_WAVE_OFF")) {
        dim3 threads(32, 1, 1);
        if (head_dim == 64u && !h3_env_on("H3_SDPA_D64_Q1")) {
            if (!h3_env_on("H3_SDPA_D64_Q2")) {
                dim3 blocks((sequence + 3u) / 4u, heads, 1);
                h3_sdpa_f32_wave_d64_q4_kernel<<<blocks, threads, 0,
                                                 gpu->stream>>>(
                    (const float *)query->device, (const float *)key->device,
                    (const float *)value->device, (float *)output->device,
                    args);
                gpu->stats.mps_sdpa_dispatches++;
                ok = h3_cuda_check(gpu, cudaGetLastError(),
                                   "h3_sdpa_f32_wave_q4");
                h3_gpu_op_end(gpu);
                return ok;
            }
            dim3 blocks((sequence + 1u) / 2u, heads, 1);
            h3_sdpa_f32_wave_d64_q2_kernel<<<blocks, threads, 0, gpu->stream>>>(
                (const float *)query->device, (const float *)key->device,
                (const float *)value->device, (float *)output->device, args);
            gpu->stats.mps_sdpa_dispatches++;
            ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_f32_wave_q2");
            h3_gpu_op_end(gpu);
            return ok;
        }
        dim3 blocks(sequence, heads, 1);
        h3_sdpa_f32_wave_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (const float *)query->device, (const float *)key->device,
            (const float *)value->device, (float *)output->device, args);
        gpu->stats.mps_sdpa_dispatches++;
        ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_f32_wave");
        h3_gpu_op_end(gpu);
        return ok;
    }
    unsigned threads = 128;
    while (threads > head_dim && threads > 32) threads >>= 1;
    if (threads < 32) threads = 32;
    size_t shared_bytes =
        (size_t)sequence * sizeof(float) + (size_t)threads * sizeof(float);
    if (shared_bytes > 48u * 1024u) {
        h3_gpu_op_end(gpu);
        return h3_gpu_fail(gpu, "F32 SDPA shared memory exceeds 48KiB");
    }
    dim3 blocks(heads, sequence, 1);
    h3_sdpa_f32_kernel<<<blocks, threads, shared_bytes, gpu->stream>>>(
        (const float *)query->device, (const float *)key->device,
        (const float *)value->device, (float *)output->device, args);
    gpu->stats.mps_sdpa_dispatches++;
    ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_f32");
    h3_gpu_op_end(gpu);
    return ok;
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

/* One thread per output element costs two global loads per FMA, and the audio
 * VAE's 122 convolutions issue 341 GB of them to do 85 GFLOP of work — 0.18
 * TFLOP/s against an FP32 roof of ~31, with both roofs putting the whole decode
 * near 3 ms rather than 478. The loads are the entire cost, so each thread takes
 * a 4x4 block of (time, output channel) instead: one weight load now feeds four
 * outputs and one input load feeds four channels, which is a quarter of the
 * loads per FMA. The accumulation stays fmaf over ic then k, ascending, so every
 * output lands on the same bits as before.
 *
 * 4x4 is the measured optimum: 8x4 and 4x8 give up 0.013 s and 8x8 gives up
 * 0.073 s, because blocking trades parallelism for reuse and the later stages
 * run only 8 or 16 channels wide. */
#ifndef H3_CONV1D_TIME_BLOCK
#define H3_CONV1D_TIME_BLOCK 4u
#endif
#ifndef H3_CONV1D_CHANNEL_BLOCK
#define H3_CONV1D_CHANNEL_BLOCK 4u
#endif

__global__ static void h3_conv1d_f32_blocked_kernel(const float *input,
                                                    const float *weight,
                                                    const float *bias,
                                                    float *output,
                                                    h3_conv1d_args args) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t channel_groups =
        (args.output_channels + H3_CONV1D_CHANNEL_BLOCK - 1u) /
        H3_CONV1D_CHANNEL_BLOCK;
    uint32_t time_groups =
        (args.output_length + H3_CONV1D_TIME_BLOCK - 1u) / H3_CONV1D_TIME_BLOCK;
    size_t total = (size_t)args.batch * time_groups * channel_groups;
    if (index >= total) return;
    /* Consecutive threads walk output channels so that the input load below is
     * one broadcast across the warp. */
    uint32_t channel_group = (uint32_t)(index % channel_groups);
    size_t rest = index / channel_groups;
    uint32_t time_group = (uint32_t)(rest % time_groups);
    uint32_t batch = (uint32_t)(rest / time_groups);
    uint32_t oc0 = channel_group * H3_CONV1D_CHANNEL_BLOCK;
    uint32_t t0 = time_group * H3_CONV1D_TIME_BLOCK;

    float acc[H3_CONV1D_CHANNEL_BLOCK][H3_CONV1D_TIME_BLOCK];
#pragma unroll
    for (uint32_t c = 0; c < H3_CONV1D_CHANNEL_BLOCK; c++) {
        float base = args.has_bias && oc0 + c < args.output_channels
                         ? bias[oc0 + c]
                         : 0.0f;
#pragma unroll
        for (uint32_t t = 0; t < H3_CONV1D_TIME_BLOCK; t++) acc[c][t] = base;
    }

    for (uint32_t ic = 0; ic < args.input_channels; ic++) {
        for (uint32_t k = 0; k < args.kernel; k++) {
            float in_value[H3_CONV1D_TIME_BLOCK];
#pragma unroll
            for (uint32_t t = 0; t < H3_CONV1D_TIME_BLOCK; t++) {
                in_value[t] = 0.0f;
                uint32_t t_out = t0 + t;
                if (t_out >= args.output_length) continue;
                int32_t t_in = (int32_t)t_out * (int32_t)args.stride -
                               (int32_t)args.padding +
                               (int32_t)k * (int32_t)args.dilation;
                if (t_in < 0 || t_in >= (int32_t)args.length) continue;
                in_value[t] = input[((size_t)batch * args.length +
                                     (size_t)t_in) *
                                        args.input_channels +
                                    ic];
            }
#pragma unroll
            for (uint32_t c = 0; c < H3_CONV1D_CHANNEL_BLOCK; c++) {
                if (oc0 + c >= args.output_channels) continue;
                float w = weight[((size_t)(oc0 + c) * args.input_channels + ic) *
                                     args.kernel +
                                 k];
#pragma unroll
                for (uint32_t t = 0; t < H3_CONV1D_TIME_BLOCK; t++)
                    acc[c][t] = fmaf(in_value[t], w, acc[c][t]);
            }
        }
    }

#pragma unroll
    for (uint32_t c = 0; c < H3_CONV1D_CHANNEL_BLOCK; c++) {
        if (oc0 + c >= args.output_channels) continue;
#pragma unroll
        for (uint32_t t = 0; t < H3_CONV1D_TIME_BLOCK; t++) {
            uint32_t t_out = t0 + t;
            if (t_out >= args.output_length) continue;
            output[((size_t)batch * args.output_length + t_out) *
                       args.output_channels +
                   oc0 + c] = acc[c][t];
        }
    }
}

/* The blocked kernel's input read is a warp-wide broadcast, but its weight read
 * is not: consecutive threads walk output channels, and in the checkpoint's
 * [oc][ic][k] layout those are a full ic*k stride apart, so each lane pulls its
 * own 32-byte sector to use 4 bytes of it. That eightfold waste is the whole
 * 341 GB the audio VAE issues to do 85 GFLOP. Transposed to [ic][k][oc] the
 * output channels are adjacent, one float4 per thread covers the block's four,
 * and a warp's loads coalesce into contiguous lines.
 *
 * Same values, same fmaf order over ic then k, so every output keeps its bits;
 * only the address arithmetic moves. */
__global__ static void h3_conv1d_weight_transpose_kernel(
    const float *source, float *destination, uint32_t output_channels,
    uint32_t input_channels, uint32_t kernel) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)output_channels * input_channels * kernel;
    if (index >= total) return;
    uint32_t k = (uint32_t)(index % kernel);
    size_t rest = index / kernel;
    uint32_t ic = (uint32_t)(rest % input_channels);
    uint32_t oc = (uint32_t)(rest / input_channels);
    destination[((size_t)ic * kernel + k) * output_channels + oc] =
        source[index];
}

__global__ static void h3_conv1d_f32_coalesced_kernel(const float *input,
                                                      const float *weight,
                                                      const float *bias,
                                                      float *output,
                                                      h3_conv1d_args args) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t channel_groups = args.output_channels / 4u;
    uint32_t time_groups =
        (args.output_length + H3_CONV1D_TIME_BLOCK - 1u) / H3_CONV1D_TIME_BLOCK;
    size_t total = (size_t)args.batch * time_groups * channel_groups;
    if (index >= total) return;
    uint32_t channel_group = (uint32_t)(index % channel_groups);
    size_t rest = index / channel_groups;
    uint32_t time_group = (uint32_t)(rest % time_groups);
    uint32_t batch = (uint32_t)(rest / time_groups);
    uint32_t oc0 = channel_group * 4u;
    uint32_t t0 = time_group * H3_CONV1D_TIME_BLOCK;

    float acc[4][H3_CONV1D_TIME_BLOCK];
    float4 base4 = args.has_bias ? *(const float4 *)(bias + oc0)
                                 : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float base[4] = {base4.x, base4.y, base4.z, base4.w};
#pragma unroll
    for (uint32_t c = 0; c < 4u; c++)
#pragma unroll
        for (uint32_t t = 0; t < H3_CONV1D_TIME_BLOCK; t++)
            acc[c][t] = base[c];

    for (uint32_t ic = 0; ic < args.input_channels; ic++) {
        for (uint32_t k = 0; k < args.kernel; k++) {
            float in_value[H3_CONV1D_TIME_BLOCK];
#pragma unroll
            for (uint32_t t = 0; t < H3_CONV1D_TIME_BLOCK; t++) {
                in_value[t] = 0.0f;
                uint32_t t_out = t0 + t;
                if (t_out >= args.output_length) continue;
                int32_t t_in = (int32_t)t_out * (int32_t)args.stride -
                               (int32_t)args.padding +
                               (int32_t)k * (int32_t)args.dilation;
                if (t_in < 0 || t_in >= (int32_t)args.length) continue;
                in_value[t] = input[((size_t)batch * args.length +
                                     (size_t)t_in) *
                                        args.input_channels +
                                    ic];
            }
            float4 w4 = *(const float4 *)(
                weight + ((size_t)ic * args.kernel + k) * args.output_channels +
                oc0);
            const float w[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
            for (uint32_t c = 0; c < 4u; c++)
#pragma unroll
                for (uint32_t t = 0; t < H3_CONV1D_TIME_BLOCK; t++)
                    acc[c][t] = fmaf(in_value[t], w[c], acc[c][t]);
        }
    }

#pragma unroll
    for (uint32_t t = 0; t < H3_CONV1D_TIME_BLOCK; t++) {
        uint32_t t_out = t0 + t;
        if (t_out >= args.output_length) continue;
        float4 out4 = make_float4(acc[0][t], acc[1][t], acc[2][t], acc[3][t]);
        *(float4 *)(output + ((size_t)batch * args.output_length + t_out) *
                                 args.output_channels +
                             oc0) = out4;
    }
}

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

/* Transposes the weight into scratch and returns it, or NULL to let the caller
 * fall back to the [oc][ic][k] kernel. */
static const float *h3_conv1d_weight_transposed(h3_gpu *gpu,
                                                const float *source,
                                                uint32_t output_channels,
                                                uint32_t input_channels,
                                                uint32_t kernel) {
    size_t count = (size_t)output_channels * input_channels * kernel;
    size_t bytes = count * sizeof(float);
    if (gpu->conv_weight_scratch_bytes < bytes) {
        if (gpu->conv_weight_scratch) cudaFree(gpu->conv_weight_scratch);
        gpu->conv_weight_scratch = NULL;
        gpu->conv_weight_scratch_bytes = 0;
        if (cudaMalloc(&gpu->conv_weight_scratch, bytes) != cudaSuccess) {
            gpu->conv_weight_scratch = NULL;
            return NULL;
        }
        gpu->conv_weight_scratch_bytes = bytes;
    }
    unsigned threads = 256;
    unsigned blocks = (unsigned)((count + threads - 1) / threads);
    h3_conv1d_weight_transpose_kernel<<<blocks, threads, 0, gpu->stream>>>(
        source, (float *)gpu->conv_weight_scratch, output_channels,
        input_channels, kernel);
    if (cudaGetLastError() != cudaSuccess) return NULL;
    return (const float *)gpu->conv_weight_scratch;
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
    h3_gpu_op_begin(gpu, H3_GPU_OP_CONV);
    h3_conv1d_args args = {batch,         length,         output_length,
                           input_channels, output_channels, kernel,
                           stride,        padding,        dilation,
                           bias ? 1u : 0u};
    unsigned threads = 256;
    size_t blocked_threads =
        (size_t)batch *
        ((output_length + H3_CONV1D_TIME_BLOCK - 1u) / H3_CONV1D_TIME_BLOCK) *
        ((output_channels + H3_CONV1D_CHANNEL_BLOCK - 1u) /
         H3_CONV1D_CHANNEL_BLOCK);
    /* Blocking costs 16x the parallelism, so it only pays while there is still
     * enough of it to fill the device. H3_DISABLE_CONV1D_BLOCK=1 forces the
     * one-output-per-thread kernel. */
    const float *transposed = NULL;
    if (blocked_threads >= 4096u && (output_channels % 4u) == 0u &&
        H3_CONV1D_CHANNEL_BLOCK == 4u && !h3_env_on("H3_DISABLE_CONV1D_BLOCK") &&
        !h3_env_on("H3_DISABLE_CONV1D_COALESCED"))
        transposed = h3_conv1d_weight_transposed(
            gpu, (const float *)weight->device, output_channels,
            input_channels, kernel);
    if (transposed) {
        size_t coalesced_threads =
            (size_t)batch *
            ((output_length + H3_CONV1D_TIME_BLOCK - 1u) /
             H3_CONV1D_TIME_BLOCK) *
            (output_channels / 4u);
        unsigned blocks =
            (unsigned)((coalesced_threads + threads - 1) / threads);
        h3_conv1d_f32_coalesced_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (const float *)input->device, transposed,
            bias ? (const float *)bias->device : NULL,
            (float *)output->device, args);
    } else if (blocked_threads >= 4096u &&
               !h3_env_on("H3_DISABLE_CONV1D_BLOCK")) {
        unsigned blocks =
            (unsigned)((blocked_threads + threads - 1) / threads);
        h3_conv1d_f32_blocked_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (const float *)input->device, (const float *)weight->device,
            bias ? (const float *)bias->device : NULL,
            (float *)output->device, args);
    } else {
        unsigned blocks = (unsigned)((output_count + threads - 1) / threads);
        h3_conv1d_f32_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (const float *)input->device, (const float *)weight->device,
            bias ? (const float *)bias->device : NULL,
            (float *)output->device, args);
    }
    gpu->stats.mps_conv_dispatches++;
    int conv_ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_conv1d_f32");
    h3_gpu_op_end(gpu);
    return conv_ok;
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
    h3_gpu_op_begin(gpu, H3_GPU_OP_CONV);
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
    int ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_conv_transpose1d_f32");
    h3_gpu_op_end(gpu);
    return ok;
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
    h3_gpu_op_begin(gpu, H3_GPU_OP_CONV);
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
    int conv3d_ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_conv3d_f32");
    h3_gpu_op_end(gpu);
    return conv3d_ok;
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
    extern __shared__ float shared[];
    float *scores = shared;
    float *reduce = shared + args.sequence;
    uint32_t head = (uint32_t)blockIdx.x;
    uint32_t q_row = (uint32_t)blockIdx.y;
    uint32_t batch = (uint32_t)blockIdx.z;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (batch >= args.batch || head >= args.heads || q_row >= args.sequence)
        return;
    size_t q_base =
        (((size_t)batch * args.sequence + q_row) * args.heads + head) *
        args.head_dim;
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads) {
        if (k_row > q_row) {
            scores[k_row] = -INFINITY;
            continue;
        }
        size_t k_base =
            (((size_t)batch * args.sequence + k_row) * args.heads + head) *
            args.head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++)
            dot = fmaf(query[q_base + d], key[k_base + d], dot);
        scores[k_row] = dot * args.scale;
    }
    __syncthreads();
    float local_max = -INFINITY;
    for (uint32_t k_row = tid; k_row <= q_row; k_row += threads)
        if (scores[k_row] > local_max) local_max = scores[k_row];
    reduce[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && reduce[tid + stride] > reduce[tid])
            reduce[tid] = reduce[tid + stride];
        __syncthreads();
    }
    float max_score = reduce[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (uint32_t k_row = tid; k_row < args.sequence; k_row += threads) {
        if (k_row > q_row) {
            scores[k_row] = 0.0f;
            continue;
        }
        float value = expf(scores[k_row] - max_score);
        scores[k_row] = value;
        local_sum += value;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    float inverse = 1.0f / reduce[0];
    __syncthreads();
    for (uint32_t k_row = tid; k_row <= q_row; k_row += threads)
        scores[k_row] *= inverse;
    __syncthreads();
    for (uint32_t d = tid; d < args.head_dim; d += threads) {
        float accumulated = 0.0f;
        for (uint32_t k_row = 0; k_row <= q_row; k_row++) {
            size_t v_base =
                (((size_t)batch * args.sequence + k_row) * args.heads +
                 head) *
                args.head_dim;
            accumulated = fmaf(scores[k_row], value[v_base + d], accumulated);
        }
        output[q_base + d] = accumulated;
    }
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
    h3_gpu_op_begin(gpu, H3_GPU_OP_SDPA);
    h3_sdpa_causal_args args = {batch, sequence, heads, head_dim, scale};
    unsigned threads = 128;
    while (threads > head_dim && threads > 32) threads >>= 1;
    if (threads < 32) threads = 32;
    size_t shared_bytes =
        (size_t)sequence * sizeof(float) + (size_t)threads * sizeof(float);
    if (shared_bytes > 48u * 1024u)
        return h3_gpu_fail(gpu, "causal SDPA shared memory exceeds 48KiB");
    dim3 blocks(heads, sequence, batch);
    h3_sdpa_causal_f32_kernel<<<blocks, threads, shared_bytes, gpu->stream>>>(
        (const float *)query->device, (const float *)key->device,
        (const float *)value->device, (float *)output->device, args);
    gpu->stats.mps_sdpa_dispatches++;
    int causal_ok =
        h3_cuda_check(gpu, cudaGetLastError(), "h3_sdpa_causal_f32");
    h3_gpu_op_end(gpu);
    return causal_ok;
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
    float levels;
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
    float levels = args.levels;
    float scale = clipped_max > 0.0f ? clipped_max / levels : 1.0f / levels;
    float inverse = clipped_max > 0.0f ? levels / clipped_max : levels;
    if (tid == 0) scales[row] = scale;
    __syncthreads();
    for (uint32_t column = tid; column < args.columns; column += threads) {
        float value = h3_bf16_bits_to_f32(row_input[column]) * inverse;
        int quantized = (int)rintf(value);
        if (quantized > (int)levels) quantized = (int)levels;
        if (quantized < -(int)levels) quantized = -(int)levels;
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
    h3_int8_quant_args args = {rows, dispatch_rows, columns, clip,
                               h3_int8_levels()};
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

struct h3_int8_swiglu_quant_args {
    uint32_t rows;
    uint32_t dispatch_rows;
    uint32_t width;
    float levels;
};

/* One block per row: reads the FC1 int32 accumulator, applies the row and
 * column scales, runs SwiGLU, and requantizes to INT8 without ever landing
 * the BF16 gate/up pair or the activation in memory. The BF16 roundings match
 * the split apply_scales → swiglu → quantize chain bit for bit. */
__global__ static void h3_int8_swiglu_quant_kernel(
    const int32_t *accum, const float *input_scales,
    const float *weight_scales, int8_t *output, float *output_scales,
    h3_int8_swiglu_quant_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.dispatch_rows) return;

    extern __shared__ float shared[];
    float *reductions = shared;
    uint16_t *activated = (uint16_t *)(shared + threads);
    size_t out_base = (size_t)row * args.width;
    if (row >= args.rows) {
        for (uint32_t column = tid; column < args.width; column += threads)
            output[out_base + column] = 0;
        if (tid == 0) output_scales[row] = 1.0f;
        return;
    }

    const int32_t *row_accum = accum + (size_t)row * args.width * 2u;
    float input_scale = input_scales[row];
    float local_max = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float gate = h3_bf16_bits_to_f32(h3_f32_to_bf16_bits(
            (float)row_accum[column] * input_scale * weight_scales[column]));
        float up = h3_bf16_bits_to_f32(h3_f32_to_bf16_bits(
            (float)row_accum[args.width + column] * input_scale *
            weight_scales[args.width + column]));
        uint16_t bits =
            h3_f32_to_bf16_bits(gate / (1.0f + expf(-gate)) * up);
        activated[column] = bits;
        float magnitude = fabsf(h3_bf16_bits_to_f32(bits));
        if (magnitude > local_max) local_max = magnitude;
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
    float row_max = reductions[0];
    float levels = args.levels;
    float scale = row_max > 0.0f ? row_max / levels : 1.0f / levels;
    float inverse = row_max > 0.0f ? levels / row_max : levels;
    if (tid == 0) output_scales[row] = scale;
    for (uint32_t column = tid; column < args.width; column += threads) {
        int quantized =
            (int)rintf(h3_bf16_bits_to_f32(activated[column]) * inverse);
        if (quantized > (int)levels) quantized = (int)levels;
        if (quantized < -(int)levels) quantized = -(int)levels;
        output[out_base + column] = (int8_t)quantized;
    }
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

__global__ static void h3_linear_int8_grouped_tiled_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, uint32_t rows,
    uint32_t input_dim, uint32_t output_dim, uint32_t group_size,
    uint32_t groups);

/* cuBLAS INT8 GEMM into the persistent int32 accumulator. Returns NULL when
 * the buffer or the GEMM is unavailable so callers can fall back. */
static int32_t *h3_int8_gemm_accum(h3_gpu *gpu, const void *weight,
                                   const void *quantized_input, uint32_t rows,
                                   uint32_t input_dim, uint32_t output_dim) {
    size_t accum_bytes = (size_t)rows * output_dim * sizeof(int32_t);
    if (gpu->int8_accum_bytes < accum_bytes) {
        if (gpu->int8_accum) cudaFree(gpu->int8_accum);
        gpu->int8_accum = NULL;
        gpu->int8_accum_bytes = 0;
        if (cudaMalloc((void **)&gpu->int8_accum, accum_bytes) == cudaSuccess)
            gpu->int8_accum_bytes = accum_bytes;
    }
    if (!gpu->int8_accum) return NULL;
    if (h3_env_on("H3_INT8_SHAPES")) {
        static int seen[64][3];
        static int count = 0;
        int known = 0;
        for (int i = 0; i < count; i++)
            if (seen[i][0] == (int)rows && seen[i][1] == (int)output_dim &&
                seen[i][2] == (int)input_dim)
                known = 1;
        if (!known && count < 64) {
            seen[count][0] = (int)rows;
            seen[count][1] = (int)output_dim;
            seen[count][2] = (int)input_dim;
            count++;
            fprintf(stderr, "h3 int8 gemm shape m=%u n=%u k=%u\n", rows,
                    output_dim, input_dim);
        }
    }
    int32_t alpha = 1;
    int32_t beta = 0;
    cublasStatus_t status = cublasGemmEx(
        gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim, (int)rows,
        (int)input_dim, &alpha, weight, CUDA_R_8I, (int)input_dim,
        quantized_input, CUDA_R_8I, (int)input_dim, &beta, gpu->int8_accum,
        CUDA_R_32I, (int)output_dim, CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS)
        status = cublasGemmEx(
            gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim, (int)rows,
            (int)input_dim, &alpha, weight, CUDA_R_8I, (int)input_dim,
            quantized_input, CUDA_R_8I, (int)input_dim, &beta, gpu->int8_accum,
            CUDA_R_32I, (int)output_dim, CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return NULL;
    gpu->int8_cublas_ok++;
    gpu->stats.direct_dispatches++;
    return gpu->int8_accum;
}

static int h3_gpu_linear_int8_bf16_impl(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
    int use_slower_uncached_int8_scales, int input_is_quantized,
    int defer_scales) {
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

    /* Cleared up front so the deferral record is only ever set by the path that
     * actually skipped the rescale; a consumer that asks to fuse but finds no
     * record falls back to reading the BF16 output, which every other path
     * still writes. */
    gpu->int8_defer_rows = 0;
    gpu->int8_defer_columns = 0;

    if (!input_is_quantized) {
        if (!h3_gpu_quantize_bf16_int8_rows(
                gpu, quantized_input, input_scales, input, rows, padded_rows,
                input_dim, 1.0f))
            return 0;
    }

    /* REJECT as default: 64x64 tile loses badly vs cuBLAS on FC1/QKV
     * (fox-s2 denoise 18s → 69s). Opt in with H3_INT8_TILED=1. */
    if (h3_env_on("H3_INT8_TILED") && (input_dim % 32u) == 0u) {
        dim3 threads(16, 16, 1);
        dim3 blocks((output_dim + 63u) / 64u, (rows + 63u) / 64u, 1);
        h3_linear_int8_grouped_tiled_kernel<<<blocks, threads, 0,
                                              gpu->stream>>>(
            (const int8_t *)quantized_input->device,
            (const int8_t *)weight->device,
            (const float *)input_scales->device,
            (const float *)weight_scales->device, (uint16_t *)output->device,
            rows, input_dim, output_dim, input_dim, 1u);
        gpu->stats.direct_dispatches++;
        return h3_cuda_check(gpu, cudaGetLastError(),
                             "h3_linear_int8_tiled");
    }

    /* Prefer cuBLAS INT8 GEMM; keep a persistent accum. */
    int32_t *accum =
        h3_int8_gemm_accum(gpu, weight->device, quantized_input->device, rows,
                           input_dim, output_dim);
    if (accum) {
        if (defer_scales) {
            gpu->int8_defer_input_scales = (const float *)input_scales->device;
            gpu->int8_defer_weight_scales =
                (const float *)weight_scales->device;
            gpu->int8_defer_rows = rows;
            gpu->int8_defer_columns = output_dim;
            return 1;
        }
        unsigned threads = 256;
        unsigned blocks = (unsigned)((output_count + threads - 1) / threads);
        h3_int8_apply_scales_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
            accum, (const float *)input_scales->device,
            (const float *)weight_scales->device, (uint16_t *)output->device,
            rows, output_dim);
        gpu->stats.direct_dispatches++;
        return h3_cuda_check(gpu, cudaGetLastError(),
                             "h3_int8_apply_scales_bf16");
    }

    if (gpu->int8_naive_fallback++ == 0)
        fprintf(stderr, "h3: INT8 linear falling back to naive GEMM\n");
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
                            int use_slower_uncached_int8_scales,
                            int defer_scales) {
    h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
    int ok = h3_gpu_linear_int8_bf16_impl(
        gpu, output, quantized_input, input_scales, input, weight,
        weight_scales, rows, input_dim, output_dim,
        use_slower_uncached_int8_scales, 0, defer_scales);
    h3_gpu_op_end(gpu);
    return ok;
}

/* FP8-E4M3 linear.
 *
 * Measured on this part at the DiT's shapes, FP8 runs the tensor cores at
 * ~192 TFLOP/s against INT8's ~145, and because cuBLASLt writes BF16 straight
 * out there is no int32 accumulator to land and re-read. The price is accuracy:
 * on real DiT weights E4M3 lands at 2.6% relative error against INT8's 1.2%,
 * which the frame comparison in scripts/quant_sensitivity.sh found acceptable
 * and NVFP4's 9.4% not.
 *
 * Weights carry one scale for the whole tensor, which cuBLASLt folds into the
 * GEMM for free; per-channel weight scales measured no better, so the extra
 * vector would buy nothing. Activations keep a scale per token, applied to the
 * BF16 result afterwards. */
#define H3_FP8_E4M3_MAX 448.0f

__global__ static void h3_fp8_amax_kernel(const uint16_t *input, size_t count,
                                          unsigned int *amax_bits) {
    extern __shared__ float reductions[];
    uint32_t tid = threadIdx.x;
    float local = 0.0f;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + tid; index < count;
         index += (size_t)gridDim.x * blockDim.x) {
        float value = fabsf(h3_bf16_bits_to_f32(input[index]));
        if (value > local) local = value;
    }
    reductions[tid] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride; stride >>= 1u) {
        if (tid < stride) {
            float other = reductions[tid + stride];
            if (other > reductions[tid]) reductions[tid] = other;
        }
        __syncthreads();
    }
    /* Non-negative floats order the same as their bit patterns, so an integer
     * atomic max is exact here. */
    if (tid == 0) atomicMax(amax_bits, __float_as_uint(reductions[0]));
}

__global__ static void h3_fp8_scale_from_amax_kernel(
    const unsigned int *amax_bits, float *scale) {
    float amax = __uint_as_float(*amax_bits);
    scale[0] = amax > 0.0f ? amax / H3_FP8_E4M3_MAX : 1.0f / H3_FP8_E4M3_MAX;
}

__global__ static void h3_quantize_bf16_fp8_tensor_kernel(
    const uint16_t *input, __nv_fp8_storage_t *output, const float *scale,
    size_t count) {
    float inverse = 1.0f / scale[0];
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += (size_t)gridDim.x * blockDim.x)
        output[index] = __nv_cvt_float_to_fp8(
            h3_bf16_bits_to_f32(input[index]) * inverse, __NV_SATFINITE,
            __NV_E4M3);
}

__global__ static void h3_quantize_bf16_fp8_rows_kernel(
    const uint16_t *input, __nv_fp8_storage_t *output, float *scales,
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
    float scale = clipped_max > 0.0f ? clipped_max / H3_FP8_E4M3_MAX
                                     : 1.0f / H3_FP8_E4M3_MAX;
    float inverse = 1.0f / scale;
    if (tid == 0) scales[row] = scale;
    __syncthreads();
    for (uint32_t column = tid; column < args.columns; column += threads)
        output[base + column] = __nv_cvt_float_to_fp8(
            h3_bf16_bits_to_f32(row_input[column]) * inverse, __NV_SATFINITE,
            __NV_E4M3);
}

/* The GEMM already carried the weight's scale, so only the per-token scale is
 * left; doing it in place keeps the traffic to one read and one write. */
__global__ static void h3_fp8_apply_row_scales_bf16_kernel(
    uint16_t *values, const float *row_scales, uint32_t rows,
    uint32_t output_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)rows * output_dim;
    if (index >= count) return;
    float scaled = h3_bf16_bits_to_f32(values[index]) *
                   row_scales[index / output_dim];
    values[index] = h3_f32_to_bf16_bits(scaled);
}

int h3_gpu_quantize_weight_fp8(h3_gpu *gpu, h3_gpu_tensor *output,
                               h3_gpu_tensor *scale,
                               const h3_gpu_tensor *input, uint32_t rows,
                               uint32_t columns) {
    size_t count = (size_t)rows * columns;
    if (!gpu || !output || !scale || !input || !rows || !columns ||
        input->dtype != H3_GPU_BF16 || output->dtype != H3_GPU_F8E4M3 ||
        scale->dtype != H3_GPU_F32 || input->elements < count ||
        output->elements < count || scale->elements < 1)
        return h3_gpu_fail(gpu, "invalid BF16→FP8 weight quantize request");

    unsigned int *amax = NULL;
    if (cudaMalloc((void **)&amax, sizeof(unsigned int)) != cudaSuccess)
        return h3_gpu_fail(gpu, "cannot allocate FP8 amax");
    int ok = h3_cuda_check(
        gpu, cudaMemsetAsync(amax, 0, sizeof(unsigned int), gpu->stream),
        "h3_fp8_amax_reset");
    unsigned threads = 256;
    unsigned blocks = (unsigned)((count + threads - 1) / threads);
    if (blocks > 4096u) blocks = 4096u;
    if (ok) {
        h3_fp8_amax_kernel<<<blocks, threads, threads * sizeof(float),
                             gpu->stream>>>((const uint16_t *)input->device,
                                            count, amax);
        h3_fp8_scale_from_amax_kernel<<<1, 1, 0, gpu->stream>>>(
            amax, (float *)scale->device);
        h3_quantize_bf16_fp8_tensor_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (const uint16_t *)input->device,
            (__nv_fp8_storage_t *)output->device, (const float *)scale->device,
            count);
        gpu->stats.direct_dispatches += 3;
        ok = h3_cuda_check(gpu, cudaGetLastError(),
                           "h3_quantize_bf16_fp8_tensor");
    }
    /* The scale lives on the stream, so the scratch cannot go until it drains. */
    if (ok)
        ok = h3_cuda_check(gpu, cudaStreamSynchronize(gpu->stream),
                           "h3_quantize_weight_fp8 sync");
    cudaFree(amax);
    return ok;
}

typedef struct {
    int rows;
    int input_dim;
    int output_dim;
    cublasLtMatmulAlgo_t algo;
    int valid;
} h3_lt_fp8_plan;

/* FP8 GEMM into a BF16 destination, with the weight's tensor scale folded in.
 * Returns 0 when cuBLASLt has no algorithm, so callers keep their INT8 path. */
static int h3_fp8_gemm_bf16(h3_gpu *gpu, const void *weight,
                            const void *weight_scale, const void *input,
                            void *destination, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim) {
    static h3_lt_fp8_plan plans[16];
    if (!h3_lt_ensure(gpu)) return 0;
    cublasLtMatmulDesc_t desc = NULL;
    cublasLtMatrixLayout_t la = NULL, lb = NULL, ld = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasOperation_t straight = CUBLAS_OP_N;
    int ok = 0;
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
        CUBLAS_STATUS_SUCCESS)
        return 0;
    if (cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transpose, sizeof(transpose)) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &straight, sizeof(straight)) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulDescSetAttribute(desc,
                                       CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                       &weight_scale, sizeof(weight_scale)) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, (int)input_dim,
                                   (int)output_dim, (int)input_dim) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, (int)input_dim,
                                   (int)rows, (int)input_dim) ==
            CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, (int)output_dim,
                                   (int)rows, (int)output_dim) ==
            CUBLAS_STATUS_SUCCESS) {
        h3_lt_fp8_plan *plan = NULL;
        for (unsigned i = 0; i < 16u; i++) {
            if (plans[i].valid && plans[i].rows == (int)rows &&
                plans[i].input_dim == (int)input_dim &&
                plans[i].output_dim == (int)output_dim) {
                plan = &plans[i];
                break;
            }
            if (!plans[i].valid && !plan) plan = &plans[i];
        }
        if (plan && !plan->valid) {
            cublasLtMatmulPreference_t preference = NULL;
            cublasLtMatmulHeuristicResult_t result;
            int found = 0;
            if (cublasLtMatmulPreferenceCreate(&preference) ==
                CUBLAS_STATUS_SUCCESS) {
                cublasLtMatmulPreferenceSetAttribute(
                    preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &gpu->lt_workspace_bytes,
                    sizeof(gpu->lt_workspace_bytes));
                if (cublasLtMatmulAlgoGetHeuristic(gpu->lt, desc, la, lb, ld,
                                                   ld, preference, 1, &result,
                                                   &found) ==
                        CUBLAS_STATUS_SUCCESS &&
                    found > 0) {
                    plan->algo = result.algo;
                    plan->rows = (int)rows;
                    plan->input_dim = (int)input_dim;
                    plan->output_dim = (int)output_dim;
                    plan->valid = 1;
                }
                cublasLtMatmulPreferenceDestroy(preference);
            }
        }
        if (plan && plan->valid) {
            float alpha = 1.0f;
            float beta = 0.0f;
            ok = cublasLtMatmul(gpu->lt, desc, &alpha, weight, la, input, lb,
                                &beta, destination, ld, destination, ld,
                                &plan->algo, gpu->lt_workspace,
                                gpu->lt_workspace_bytes,
                                gpu->stream) == CUBLAS_STATUS_SUCCESS;
        }
    }
    if (ld) cublasLtMatrixLayoutDestroy(ld);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(desc);
    if (ok) gpu->stats.direct_dispatches++;
    return ok;
}

int h3_gpu_linear_fp8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           h3_gpu_tensor *quantized_input,
                           h3_gpu_tensor *input_scales,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *weight_scale, uint32_t rows,
                           uint32_t input_dim, uint32_t output_dim) {
    size_t output_count = (size_t)rows * output_dim;
    if (!gpu || !output || !quantized_input || !input_scales || !input ||
        !weight || !weight_scale || !rows || !input_dim || !output_dim ||
        output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 ||
        quantized_input->dtype != H3_GPU_F8E4M3 ||
        weight->dtype != H3_GPU_F8E4M3 ||
        input_scales->dtype != H3_GPU_F32 ||
        weight_scale->dtype != H3_GPU_F32 ||
        output->elements < output_count ||
        input->elements < (size_t)rows * input_dim ||
        quantized_input->elements < (size_t)rows * input_dim ||
        weight->elements < (size_t)output_dim * input_dim ||
        input_scales->elements < rows || weight_scale->elements < 1)
        return h3_gpu_fail(gpu, "invalid FP8 linear request");

    h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
    h3_int8_quant_args args = {rows, rows, input_dim, 1.0f, H3_FP8_E4M3_MAX};
    unsigned threads = 256;
    h3_quantize_bf16_fp8_rows_kernel<<<rows, threads, threads * sizeof(float),
                                       gpu->stream>>>(
        (const uint16_t *)input->device,
        (__nv_fp8_storage_t *)quantized_input->device,
        (float *)input_scales->device, args);
    gpu->stats.direct_dispatches++;
    if (!h3_cuda_check(gpu, cudaGetLastError(),
                       "h3_quantize_bf16_fp8_rows")) {
        h3_gpu_op_end(gpu);
        return 0;
    }
    if (!h3_fp8_gemm_bf16(gpu, weight->device, weight_scale->device,
                          quantized_input->device, output->device, rows,
                          input_dim, output_dim)) {
        h3_gpu_op_end(gpu);
        return h3_gpu_fail(gpu, "no cuBLASLt FP8 algorithm for this shape");
    }
    unsigned blocks = (unsigned)((output_count + threads - 1) / threads);
    h3_fp8_apply_row_scales_bf16_kernel<<<blocks, threads, 0, gpu->stream>>>(
        (uint16_t *)output->device, (const float *)input_scales->device, rows,
        output_dim);
    gpu->stats.direct_dispatches++;
    int ok = h3_cuda_check(gpu, cudaGetLastError(),
                           "h3_fp8_apply_row_scales_bf16");
    h3_gpu_op_end(gpu);
    return ok;
}

/* FC1's BF16 result straight into FC2's FP8 input: applies the token scale,
 * runs SwiGLU and requantizes, so the gate/up pair never lands in memory. The
 * weight scale is already inside the GEMM, which is why only one vector is
 * read here where the INT8 twin reads two. */
__global__ static void h3_fp8_swiglu_quant_kernel(
    const uint16_t *fused, const float *input_scales,
    __nv_fp8_storage_t *output, float *output_scales,
    h3_int8_swiglu_quant_args args) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.dispatch_rows) return;

    extern __shared__ float shared[];
    float *reductions = shared;
    uint16_t *activated = (uint16_t *)(shared + threads);
    size_t out_base = (size_t)row * args.width;
    if (row >= args.rows) {
        for (uint32_t column = tid; column < args.width; column += threads)
            output[out_base + column] = 0;
        if (tid == 0) output_scales[row] = 1.0f;
        return;
    }

    const uint16_t *row_fused = fused + (size_t)row * args.width * 2u;
    float input_scale = input_scales[row];
    float local_max = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float gate = h3_bf16_bits_to_f32(row_fused[column]) * input_scale;
        float up =
            h3_bf16_bits_to_f32(row_fused[args.width + column]) * input_scale;
        uint16_t bits =
            h3_f32_to_bf16_bits(gate / (1.0f + expf(-gate)) * up);
        activated[column] = bits;
        float magnitude = fabsf(h3_bf16_bits_to_f32(bits));
        if (magnitude > local_max) local_max = magnitude;
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
    float row_max = reductions[0];
    float scale =
        row_max > 0.0f ? row_max / H3_FP8_E4M3_MAX : 1.0f / H3_FP8_E4M3_MAX;
    float inverse = 1.0f / scale;
    if (tid == 0) output_scales[row] = scale;
    for (uint32_t column = tid; column < args.width; column += threads)
        output[out_base + column] = __nv_cvt_float_to_fp8(
            h3_bf16_bits_to_f32(activated[column]) * inverse, __NV_SATFINITE,
            __NV_E4M3);
}

int h3_gpu_mlp_fp8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *fc1_output,
                        h3_gpu_tensor *quantized_activation,
                        h3_gpu_tensor *activation_scales,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc1_scale,
                        const h3_gpu_tensor *fc2_weight,
                        const h3_gpu_tensor *fc2_scale, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    size_t output_count = (size_t)rows * output_dim;
    if (!gpu || !output || !fc1_output || !quantized_activation ||
        !activation_scales || !input || !fc1_weight || !fc1_scale ||
        !fc2_weight || !fc2_scale || !rows || !input_dim || !hidden_dim ||
        !output_dim || output->dtype != H3_GPU_BF16 ||
        fc1_output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 ||
        quantized_activation->dtype != H3_GPU_F8E4M3 ||
        fc1_weight->dtype != H3_GPU_F8E4M3 ||
        fc2_weight->dtype != H3_GPU_F8E4M3 ||
        activation_scales->dtype != H3_GPU_F32 ||
        fc1_scale->dtype != H3_GPU_F32 || fc2_scale->dtype != H3_GPU_F32 ||
        output->elements < output_count ||
        fc1_output->elements < (size_t)rows * hidden_dim * 2u ||
        quantized_activation->elements <
            (size_t)rows * (input_dim > hidden_dim ? input_dim : hidden_dim) ||
        activation_scales->elements < rows)
        return h3_gpu_fail(gpu, "invalid FP8 MLP request");

    /* The SwiGLU row lives in shared memory, so a wide hidden dimension has to
     * fall back to the INT8 path rather than silently overflowing. */
    unsigned fused_threads = 256;
    size_t fused_shared =
        fused_threads * sizeof(float) + (size_t)hidden_dim * sizeof(uint16_t);
    if (fused_shared > 46u * 1024u)
        return h3_gpu_fail(gpu, "FP8 MLP hidden dimension too wide to fuse");

    h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
    unsigned threads = 256;
    h3_int8_quant_args quant = {rows, rows, input_dim, 1.0f, H3_FP8_E4M3_MAX};
    h3_quantize_bf16_fp8_rows_kernel<<<rows, threads, threads * sizeof(float),
                                       gpu->stream>>>(
        (const uint16_t *)input->device,
        (__nv_fp8_storage_t *)quantized_activation->device,
        (float *)activation_scales->device, quant);
    gpu->stats.direct_dispatches++;
    int ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_fp8_mlp input quant");
    if (ok)
        ok = h3_fp8_gemm_bf16(gpu, fc1_weight->device, fc1_scale->device,
                              quantized_activation->device,
                              fc1_output->device, rows, input_dim,
                              hidden_dim * 2u);
    if (ok) {
        /* The scale buffer is read as FC1's input scale and rewritten as
         * FC2's; every read of a row precedes the reduction barrier in the
         * block that owns it, so the reuse is safe. */
        h3_int8_swiglu_quant_args args = {rows, rows, hidden_dim,
                                          H3_FP8_E4M3_MAX};
        h3_fp8_swiglu_quant_kernel<<<rows, fused_threads, fused_shared,
                                     gpu->stream>>>(
            (const uint16_t *)fc1_output->device,
            (const float *)activation_scales->device,
            (__nv_fp8_storage_t *)quantized_activation->device,
            (float *)activation_scales->device, args);
        gpu->stats.direct_dispatches++;
        ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_fp8_swiglu_quant");
    }
    if (ok)
        ok = h3_fp8_gemm_bf16(gpu, fc2_weight->device, fc2_scale->device,
                              quantized_activation->device, output->device,
                              rows, hidden_dim, output_dim);
    if (ok) {
        unsigned blocks = (unsigned)((output_count + threads - 1) / threads);
        h3_fp8_apply_row_scales_bf16_kernel<<<blocks, threads, 0,
                                              gpu->stream>>>(
            (uint16_t *)output->device,
            (const float *)activation_scales->device, rows, output_dim);
        gpu->stats.direct_dispatches++;
        ok = h3_cuda_check(gpu, cudaGetLastError(), "h3_fp8_mlp output scale");
    }
    h3_gpu_op_end(gpu);
    return ok;
}

__global__ static void h3_quantize_bf16_int8_head_major_kernel(
    const uint16_t *input, int8_t *output, float *scales, uint32_t rows,
    uint32_t padded_rows, uint32_t heads, uint32_t head_dim, float levels) {
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
    float scale = clipped_max > 0.0f ? clipped_max / levels : 1.0f / levels;
    float inverse = clipped_max > 0.0f ? levels / clipped_max : levels;
    if (tid == 0) scales[row] = scale;
    __syncthreads();
    for (uint32_t column = tid; column < columns; column += threads) {
        uint32_t head = column / head_dim;
        uint32_t dim = column % head_dim;
        size_t in_index =
            ((size_t)head * rows + row) * head_dim + dim;
        int quantized =
            (int)rintf(h3_bf16_bits_to_f32(input[in_index]) * inverse);
        if (quantized > (int)levels) quantized = (int)levels;
        if (quantized < -(int)levels) quantized = -(int)levels;
        output[out_base + column] = (int8_t)quantized;
    }
}

int h3_gpu_linear_int8_head_major_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t output_dim,
    int defer_scales) {
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

    h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
    unsigned threads = 256;
    h3_quantize_bf16_int8_head_major_kernel<<<padded_rows, threads,
                                                threads * sizeof(float),
                                                gpu->stream>>>(
        (const uint16_t *)input->device, (int8_t *)quantized_input->device,
        (float *)input_scales->device, rows, padded_rows, heads, head_dim,
        h3_int8_levels());
    gpu->stats.direct_dispatches++;
    if (!h3_cuda_check(gpu, cudaGetLastError(),
                       "h3_quantize_bf16_int8_head_major")) {
        h3_gpu_op_end(gpu);
        return 0;
    }
    int ok = h3_gpu_linear_int8_bf16_impl(
        gpu, output, quantized_input, input_scales, NULL, weight,
        weight_scales, rows, input_dim, output_dim, 0, 1, defer_scales);
    h3_gpu_op_end(gpu);
    return ok;
}

struct h3_int8_group_quant_args {
    uint32_t rows;
    uint32_t dispatch_rows;
    uint32_t columns;
    uint32_t group_size;
    uint32_t groups;
    float levels;
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
    float levels = args.levels;
    float scale = clipped_max > 0.0f ? clipped_max / levels : 1.0f / levels;
    float inverse = clipped_max > 0.0f ? levels / clipped_max : levels;
    if (tid == 0) scales[(size_t)row * args.groups + group] = scale;
    __syncthreads();
    for (uint32_t column = tid; column < args.group_size; column += threads) {
        int quantized = (int)rintf(
            h3_bf16_bits_to_f32(input[row_base + group_start + column]) *
            inverse);
        if (quantized > (int)levels) quantized = (int)levels;
        if (quantized < -(int)levels) quantized = -(int)levels;
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
                                     groups, h3_int8_levels()};
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

/* 64x64 output tiles, K=32. Applies per-group input scales (MLP FC2). */
__global__ static void h3_linear_int8_grouped_tiled_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, uint32_t rows,
    uint32_t input_dim, uint32_t output_dim, uint32_t group_size,
    uint32_t groups) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    __shared__ int8_t tile_a[BM][BK];
    __shared__ int8_t tile_b[BN][BK];

    uint32_t row0 = (uint32_t)blockIdx.y * (uint32_t)BM;
    uint32_t col0 = (uint32_t)blockIdx.x * (uint32_t)BN;
    uint32_t tx = (uint32_t)threadIdx.x;
    uint32_t ty = (uint32_t)threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    for (uint32_t group = 0; group < groups; group++) {
        int32_t iacc[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) iacc[i][j] = 0;
        uint32_t k_origin = group * group_size;
        for (uint32_t kk = 0; kk < group_size; kk += (uint32_t)BK) {
#pragma unroll
            for (int step = 0; step < (BM * BK) / 256; step++) {
                uint32_t idx = tid + (uint32_t)step * 256u;
                uint32_t r = idx / (uint32_t)BK;
                uint32_t c = idx % (uint32_t)BK;
                uint32_t gr = row0 + r;
                uint32_t gc = k_origin + kk + c;
                tile_a[r][c] =
                    (gr < rows && gc < input_dim)
                        ? input[(size_t)gr * input_dim + gc]
                        : (int8_t)0;
            }
#pragma unroll
            for (int step = 0; step < (BN * BK) / 256; step++) {
                uint32_t idx = tid + (uint32_t)step * 256u;
                uint32_t r = idx / (uint32_t)BK;
                uint32_t c = idx % (uint32_t)BK;
                uint32_t gc = col0 + r;
                uint32_t gk = k_origin + kk + c;
                tile_b[r][c] =
                    (gc < output_dim && gk < input_dim)
                        ? weight[(size_t)gc * input_dim + gk]
                        : (int8_t)0;
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < BK; k += 4) {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    int a_pack =
                        *reinterpret_cast<const int *>(
                            &tile_a[ty * 4u + (uint32_t)i][k]);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int b_pack =
                            *reinterpret_cast<const int *>(
                                &tile_b[tx * 4u + (uint32_t)j][k]);
                        iacc[i][j] = __dp4a(a_pack, b_pack, iacc[i][j]);
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t row = row0 + ty * 4u + (uint32_t)i;
            float in_scale =
                row < rows ? input_scales[(size_t)row * groups + group] : 0.0f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t col = col0 + tx * 4u + (uint32_t)j;
                float w_scale =
                    col < output_dim ? weight_scales[col] : 0.0f;
                acc[i][j] = fmaf((float)iacc[i][j], in_scale * w_scale,
                                 acc[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row0 + ty * 4u + (uint32_t)i;
        if (row >= rows) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col0 + tx * 4u + (uint32_t)j;
            if (col >= output_dim) continue;
            output[(size_t)row * output_dim + col] =
                h3_f32_to_bf16_bits(acc[i][j]);
        }
    }
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

    h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
    int grouped_ok;
    if (!h3_env_on("H3_INT8_GROUP_NAIVE") && (group_size % 32u) == 0u) {
        dim3 threads(16, 16, 1);
        dim3 blocks((output_dim + 63u) / 64u, (rows + 63u) / 64u, 1);
        h3_linear_int8_grouped_tiled_kernel<<<blocks, threads, 0,
                                              gpu->stream>>>(
            (const int8_t *)quantized_input->device,
            (const int8_t *)weight->device,
            (const float *)input_scales->device,
            (const float *)weight_scales->device, (uint16_t *)output->device,
            rows, input_dim, output_dim, group_size, groups);
        gpu->stats.direct_dispatches++;
        grouped_ok = h3_cuda_check(gpu, cudaGetLastError(),
                                   "h3_linear_int8_grouped_tiled");
    } else {
        dim3 threads(256, 1, 1);
        dim3 blocks((output_dim + threads.x - 1) / threads.x, rows, 1);
        h3_linear_int8_grouped_naive_kernel<<<blocks, threads, 0, gpu->stream>>>(
            (const int8_t *)quantized_input->device,
            (const int8_t *)weight->device,
            (const float *)input_scales->device,
            (const float *)weight_scales->device, (uint16_t *)output->device,
            rows, input_dim, output_dim, group_size, groups);
        gpu->stats.direct_dispatches++;
        grouped_ok = h3_cuda_check(gpu, cudaGetLastError(),
                                   "h3_linear_int8_grouped_naive");
    }
    h3_gpu_op_end(gpu);
    return grouped_ok;
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
                         int input_is_quantized, int defer_output_scales) {
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

    /* Default: one kernel turns the FC1 int32 accumulator into the INT8 FC2
     * input, so neither the gate/up pair nor the activation is ever written as
     * BF16. Opt out with H3_SPLIT_INT8_SWIGLU=1. */
    unsigned fused_threads = 256;
    size_t fused_shared =
        fused_threads * sizeof(float) + (size_t)hidden_dim * sizeof(uint16_t);
    if (int8_fc1 && int8_fc2 && !grouped_fc2 && fused_shared <= 46u * 1024u &&
        !h3_env_on("H3_SPLIT_INT8_SWIGLU")) {
        if (!input_is_quantized &&
            !h3_gpu_quantize_bf16_int8_rows(gpu, quantized_activation,
                                            activation_scales, input, rows,
                                            padded_rows, input_dim, 1.0f))
            return 0;
        h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
        int32_t *accum =
            h3_int8_gemm_accum(gpu, fc1_weight->device,
                               quantized_activation->device, rows, input_dim,
                               hidden_dim * 2u);
        int fused_ok = 0;
        if (accum) {
            /* The scale buffer is read as the FC1 input scale and rewritten as
             * the FC2 input scale; every read of a row precedes the reduction
             * barrier in the block that owns it, so the reuse is safe. */
            h3_int8_swiglu_quant_args args = {rows, padded_rows, hidden_dim,
                                              h3_int8_levels()};
            h3_int8_swiglu_quant_kernel<<<padded_rows, fused_threads,
                                            fused_shared, gpu->stream>>>(
                accum, (const float *)activation_scales->device,
                (const float *)fc1_scales->device,
                (int8_t *)quantized_activation->device,
                (float *)activation_scales->device, args);
            gpu->stats.direct_dispatches++;
            fused_ok = h3_cuda_check(gpu, cudaGetLastError(),
                                     "h3_int8_swiglu_quant");
        }
        h3_gpu_op_end(gpu);
        if (fused_ok)
            return h3_gpu_linear_int8_bf16_impl(
                gpu, output, quantized_activation, activation_scales, NULL,
                fc2_weight, fc2_scales, rows, hidden_dim, output_dim, 0, 1,
                defer_output_scales);
        if (accum) return 0; /* the kernel failed, not the GEMM buffer */
        /* No INT8 GEMM available: fall through to the split path. */
        if (!input_is_quantized) input_is_quantized = 1;
    }

    h3_gpu_tensor *fc1_fused = h3_gpu_workspace_bf16(
        gpu, &gpu->ws_int8_fc1, (size_t)rows * hidden_dim * 2u);
    if (!fc1_fused) return h3_gpu_fail(gpu, "INT8 MLP FC1 temp alloc failed");

    int ok = 1;
    if (int8_fc1) {
        if (!input_is_quantized &&
            !h3_gpu_quantize_bf16_int8_rows(
                gpu, quantized_activation, activation_scales, input, rows,
                padded_rows, input_dim, 1.0f))
            ok = 0;
        if (ok) {
            h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
            if (!h3_gpu_linear_int8_bf16_impl(
                    gpu, fc1_fused, quantized_activation, activation_scales,
                    NULL, fc1_weight, fc1_scales, rows, input_dim,
                    hidden_dim * 2u, 0, 1, 0))
                ok = 0;
            h3_gpu_op_end(gpu);
        }
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
                         output_dim, 0, 1, 0))
                ok = 0;
        }
    } else if (ok &&
               !h3_gpu_linear_bf16(gpu, output, activated, fc2_bf16, NULL, rows,
                                   hidden_dim, output_dim)) {
        ok = 0;
    }

    h3_gpu_workspace_release(fc1_fused);
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
    float epsilon, int fuse_branch_rescale) {
    size_t elements = (size_t)rows * width;
    if (!gpu || !gated_residual || !quantized_output || !quantized_scales ||
        !residual || !branch || !norm_weight || !gate_modulation ||
        !norm_modulation || !row_map || !rows || padded_rows < rows || !width ||
        gate_slot >= slots || shift_slot >= slots || scale_slot >= slots)
        return h3_gpu_fail(gpu, "invalid gate AdaLN quantize request");

    /* Only fold in the producer's rescale if that producer actually left one:
     * the record has to match this call's shape, or the branch tensor already
     * holds finished BF16 and reading the accumulator would be wrong. */
    const int32_t *branch_accum = NULL;
    if (fuse_branch_rescale && gpu->int8_defer_rows == rows &&
        gpu->int8_defer_columns == width && gpu->int8_accum)
        branch_accum = gpu->int8_accum;

    /* Default: single kernel. Opt out H3_SPLIT_ADALN_QUANT=1. */
    if (!h3_env_on("H3_SPLIT_ADALN_QUANT") && width <= 5376u &&
        gated_residual->dtype == H3_GPU_BF16 &&
        quantized_output->dtype == H3_GPU_I8 &&
        quantized_scales->dtype == H3_GPU_F32 &&
        residual->dtype == H3_GPU_BF16 && branch->dtype == H3_GPU_BF16 &&
        norm_weight->dtype == H3_GPU_BF16 &&
        gate_modulation->dtype == H3_GPU_BF16 &&
        norm_modulation->dtype == H3_GPU_BF16 &&
        row_map->dtype == H3_GPU_U32 &&
        gated_residual->elements >= elements &&
        quantized_output->elements >= (size_t)padded_rows * width &&
        quantized_scales->elements >= padded_rows &&
        residual->elements >= elements && branch->elements >= elements &&
        norm_weight->elements >= width && row_map->elements >= rows) {
        h3_gate_adaln_args args = {rows, width, slots, gate_slot, shift_slot,
                                   scale_slot, epsilon};
        unsigned threads = 256;
        size_t shared_bytes =
            threads * sizeof(float) + (size_t)width * sizeof(uint16_t);
        h3_gate_adaln_quantize_int8_kernel<<<padded_rows, threads,
                                               shared_bytes, gpu->stream>>>(
            (const uint16_t *)residual->device,
            (const uint16_t *)branch->device,
            (const uint16_t *)gate_modulation->device,
            (const uint32_t *)row_map->device,
            (const uint16_t *)norm_weight->device,
            (const uint16_t *)norm_modulation->device,
            (uint16_t *)gated_residual->device,
            (int8_t *)quantized_output->device,
            (float *)quantized_scales->device, args, padded_rows,
            h3_int8_levels(), branch_accum, gpu->int8_defer_input_scales,
            gpu->int8_defer_weight_scales);
        gpu->stats.direct_dispatches++;
        return h3_cuda_check(gpu, cudaGetLastError(),
                             "h3_gate_adaln_quantize_int8");
    }

    /* The split path below reads the branch as BF16, which a deferred producer
     * never wrote. Callers gate their deferral request on the same conditions
     * as the fused kernel above, so reaching here with a record in hand means
     * those conditions drifted apart; say so rather than read stale memory. */
    if (branch_accum)
        return h3_gpu_fail(gpu,
                           "fused branch rescale requested but the fused gate "
                           "AdaLN quantize path was not taken");

    h3_gpu_tensor *adaln =
        h3_gpu_workspace_bf16(gpu, &gpu->ws_adaln, elements);
    if (!adaln) return h3_gpu_fail(gpu, "gate AdaLN quantize temp alloc failed");
    int ok =
        h3_gpu_gate_adaln_bf16(gpu, gated_residual, adaln, residual, branch,
                               norm_weight, gate_modulation, norm_modulation,
                               row_map, rows, width, slots, gate_slot,
                               shift_slot, scale_slot, epsilon) &&
        h3_gpu_quantize_bf16_int8_rows(gpu, quantized_output, quantized_scales,
                                       adaln, rows, padded_rows, width, 1.0f);
    h3_gpu_workspace_release(adaln);
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

    /* The projection is rows x 3*inner, the widest intermediate in the block.
     * Reading the accumulator straight into the RoPE kernel keeps it out of
     * memory entirely, which is worth more here than anywhere else in the
     * layer. Opt out with H3_SPLIT_INT8_QKV_ROPE=1. */
    int serial_rms =
        use_slower_scalar_qkv_rms || h3_env_on("H3_QKV_ROPE_SERIAL_RMS");
    if (!serial_rms && !h3_env_on("H3_SPLIT_INT8_QKV_ROPE")) {
        h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
        int32_t *accum =
            h3_int8_gemm_accum(gpu, weight->device, quantized_input->device,
                               rows, input_dim, inner * 3u);
        int fused_ok = 0;
        if (accum) {
            h3_qkv_int8_source source = {accum,
                                         (const float *)input_scales->device,
                                         (const float *)weight_scales->device,
                                         inner * 3u};
            h3_qkv_rope_args args = {rows,      heads, head_dim,
                                     rope_half, 1u,    epsilon};
            fused_ok = h3_qkv_rope_coop_launch(gpu, source, query, key, value,
                                               q_norm, k_norm, rope_cos,
                                               rope_sin, args);
        }
        h3_gpu_op_end(gpu);
        if (fused_ok) return 1;
    }

    h3_gpu_tensor *qkv = h3_gpu_workspace_bf16(
        gpu, &gpu->ws_qkv, (size_t)rows * inner * 3u);
    if (!qkv) return h3_gpu_fail(gpu, "INT8 QKV temp alloc failed");
    h3_gpu_op_begin(gpu, H3_GPU_OP_LINEAR);
    int ok = h3_gpu_linear_int8_bf16_impl(
        gpu, qkv, quantized_input, input_scales, NULL, weight, weight_scales,
        rows, input_dim, inner * 3u, 0, 1, 0);
    h3_gpu_op_end(gpu);
    ok = ok &&
        h3_gpu_qkv_rope_bf16_layout(gpu, query, key, value, qkv, q_norm,
                                    k_norm, rope_cos, rope_sin, rows, heads,
                                    head_dim, rope_half, 1u, epsilon,
                                    use_slower_scalar_qkv_rms);
    h3_gpu_workspace_release(qkv);
    return ok;
}

int h3_gpu_begin(h3_gpu *gpu) {
    if (!gpu) return 0;
    gpu->error[0] = '\0';
    return 1;
}

int h3_gpu_continue(h3_gpu *gpu) {
    /* Match Metal: enqueue without waiting. CUDA stream already orders work. */
    if (!gpu) return 0;
    gpu->stats.submissions++;
    return 1;
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
