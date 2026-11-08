#ifndef H3_VSA_KERNELS_H
#define H3_VSA_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Scatter token-major Q/K/V/gate into head-major tiled buffers, pool/score
 * from tiled, run sparse MMA fine on tiled layout, gather to output. */
int h3_launch_vsa_bf16(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    const uint16_t *gate, uint16_t *output, const uint32_t *slot,
    const uint32_t *block_sizes, uint16_t *tiled_q, uint16_t *tiled_k,
    uint16_t *tiled_v, uint16_t *tiled_g, uint16_t *tiled_o, float *pool_q,
    float *pool_k, float *pool_v, float *scores, float *fine_acc,
    uint8_t *mask, uint32_t sequence, uint32_t heads, uint32_t dim,
    uint32_t padded, uint32_t n_tiles, uint32_t n_prefix, uint32_t keep,
    float scale, uint32_t kv_head_major, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
