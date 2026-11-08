#ifndef H3_TAEH3_KERNELS_H
#define H3_TAEH3_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t batch;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t height;
    uint32_t width;
    uint32_t kernel;
    uint32_t pad;
    uint32_t has_bias;
} h3_taeh3_conv2d_args;

typedef struct {
    uint32_t batch;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
} h3_taeh3_nchw_args;

int h3_launch_taeh3_conv2d(const float *input, const float *weight,
                           const float *bias, float *output,
                           const h3_taeh3_conv2d_args *args,
                           cudaStream_t stream);
int h3_launch_taeh3_tanh_clamp(const float *input, float *output,
                               uint32_t elements, cudaStream_t stream);
int h3_launch_taeh3_relu(const float *input, float *output,
                         uint32_t elements, cudaStream_t stream);
int h3_launch_taeh3_add_relu(const float *left, const float *right,
                             float *output, uint32_t elements,
                             cudaStream_t stream);
int h3_launch_taeh3_upsample2(const float *input, float *output,
                              const h3_taeh3_nchw_args *args,
                              cudaStream_t stream);
int h3_launch_taeh3_time_past(const float *input, const float *memory,
                              float *output, const h3_taeh3_nchw_args *args,
                              cudaStream_t stream);
int h3_launch_taeh3_cat_channels(const float *left, const float *right,
                                 float *output, const h3_taeh3_nchw_args *args,
                                 cudaStream_t stream);
int h3_launch_taeh3_pixel_shuffle2(const float *input, float *output,
                                   const h3_taeh3_nchw_args *args,
                                   cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
