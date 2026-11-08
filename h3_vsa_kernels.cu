/* Opt-in FastH3 VSA. Video tokens are 4x4x4 cubes (64). Prefix segments
 * (text, keyframes, audio) are tiled without straddling a boundary and
 * stay dense. Video queries keep every prefix tile and the top
 * (1 - sparsity) video tiles. The learned gate adds a dense tile-mean
 * branch: output = fine + coarse * gate. Sol-Attn is not used. */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr uint32_t VSA_MMA_M = 64u;
constexpr uint32_t VSA_MMA_N = 64u;
constexpr uint32_t VSA_MMA_LD = 136u;
constexpr uint32_t VSA_MMA_VLD = 72u;

__device__ float vsa_bf16(uint16_t bits) {
    return __uint_as_float((uint32_t)bits << 16);
}

__device__ uint16_t vsa_f32(float value) {
    __nv_bfloat16 bf = __float2bfloat16(value);
    return *reinterpret_cast<uint16_t *>(&bf);
}

__device__ __forceinline__ void vsa_mma_m16n8k16_bf16(float (&d)[4],
                                                     const uint32_t (&a)[4],
                                                     const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void vsa_mma_m16n8k16_f16(float (&d)[4],
                                                    const uint32_t (&a)[4],
                                                    const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t vsa_pack_bf16_pair(float low, float high) {
    return (uint32_t)vsa_f32(low) | ((uint32_t)vsa_f32(high) << 16);
}

__device__ __forceinline__ uint32_t vsa_f16_bits(float value) {
    value = fminf(fmaxf(value, -65504.0f), 65504.0f);
    return (uint32_t)__half_as_ushort(__float2half_rn(value));
}

__device__ __forceinline__ uint32_t vsa_pack_f16_pair(float low, float high) {
    return vsa_f16_bits(low) | (vsa_f16_bits(high) << 16);
}

__device__ __forceinline__ uint32_t vsa_bf16_pair_to_f16(uint32_t raw) {
    nv_bfloat162 in;
    memcpy(&in, &raw, sizeof(in));
    half2 out = __float22half2_rn(__bfloat1622float2(in));
    uint32_t packed;
    memcpy(&packed, &out, sizeof(packed));
    return packed;
}

__device__ __forceinline__ void vsa_ldmatrix_b16_x2(uint32_t &lo, uint32_t &hi,
                                                   const uint16_t *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(lo), "=r"(hi)
        : "r"(addr));
}

uint32_t vsa_grid(uint64_t n) {
    uint64_t blocks = (n + 255ull) / 256ull;
    if (blocks < 1ull) blocks = 1ull;
    if (blocks > 2048ull) blocks = 2048ull;
    return (uint32_t)blocks;
}

__device__ float vsa_warp_sum(float value) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, off);
    return value;
}

__device__ float vsa_warp_max(float value) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, off));
    return value;
}

__device__ void vsa_load4(const uint16_t *p, float &a, float &b, float &c,
                          float &d) {
    uint2 raw = *reinterpret_cast<const uint2 *>(p);
    a = vsa_bf16(static_cast<uint16_t>(raw.x));
    b = vsa_bf16(static_cast<uint16_t>(raw.x >> 16));
    c = vsa_bf16(static_cast<uint16_t>(raw.y));
    d = vsa_bf16(static_cast<uint16_t>(raw.y >> 16));
}

__device__ void vsa_store4(uint16_t *p, float a, float b, float c, float d) {
    uint2 raw;
    raw.x = (uint32_t)vsa_f32(a) | ((uint32_t)vsa_f32(b) << 16);
    raw.y = (uint32_t)vsa_f32(c) | ((uint32_t)vsa_f32(d) << 16);
    *reinterpret_cast<uint2 *>(p) = raw;
}

__global__ void h3_vsa_scatter_kernel(const uint16_t *src, uint16_t *dst,
                                      const uint32_t *slot, uint32_t sequence,
                                      uint32_t heads, uint32_t dim,
                                      uint32_t padded, uint32_t kv_head_major) {
    uint64_t n = (uint64_t)sequence * heads * dim;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += stride) {
        uint32_t d = (uint32_t)(idx % dim);
        uint64_t rest = idx / dim;
        uint32_t head = (uint32_t)(rest % heads);
        uint32_t token = (uint32_t)(rest / heads);
        uint32_t dest = slot[token];
        size_t from = kv_head_major
                          ? ((size_t)head * sequence + token) * dim + d
                          : ((size_t)token * heads + head) * dim + d;
        dst[((size_t)head * padded + dest) * dim + d] = src[from];
    }
}

__global__ void h3_vsa_pool_kernel(const uint16_t *src, float *pooled,
                                   const uint32_t *block_sizes, uint32_t n_tiles,
                                   uint32_t heads, uint32_t dim,
                                   uint32_t padded) {
    constexpr uint32_t TILE = 64u;
    const uint32_t lane = threadIdx.x;
    const uint32_t tile = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    if (lane >= 32u || tile >= n_tiles || head >= heads || dim != 128u) return;
    const uint32_t count = block_sizes[tile];
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float a, b, c, d;
        vsa_load4(src + ((size_t)head * padded + tile * TILE + i) * dim +
                      lane * 4u,
                  a, b, c, d);
        s0 += a;
        s1 += b;
        s2 += c;
        s3 += d;
    }
    float inv_scale = count ? 1.0f / (float)count : 0.0f;
    float *out = pooled + ((size_t)head * n_tiles + tile) * dim + lane * 4u;
    out[0] = s0 * inv_scale;
    out[1] = s1 * inv_scale;
    out[2] = s2 * inv_scale;
    out[3] = s3 * inv_scale;
}

__global__ void h3_vsa_score_kernel(const float *query, const float *key,
                                    float *scores, uint32_t n_tiles,
                                    uint32_t heads, uint32_t dim, float scale) {
    const uint32_t q_tile = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    if (q_tile >= n_tiles || head >= heads || dim != 128u) return;
    const float *q = query + ((size_t)head * n_tiles + q_tile) * dim;
    for (uint32_t k_tile = threadIdx.x; k_tile < n_tiles; k_tile += blockDim.x) {
        const float *k = key + ((size_t)head * n_tiles + k_tile) * dim;
        float sum = 0.0f;
        for (uint32_t d = 0; d < dim; d += 4u) {
            float4 qv = *reinterpret_cast<const float4 *>(q + d);
            float4 kv = *reinterpret_cast<const float4 *>(k + d);
            sum = fmaf(qv.x, kv.x, sum);
            sum = fmaf(qv.y, kv.y, sum);
            sum = fmaf(qv.z, kv.z, sum);
            sum = fmaf(qv.w, kv.w, sum);
        }
        scores[((size_t)head * n_tiles + q_tile) * n_tiles + k_tile] =
            sum * scale;
    }
}

__global__ void h3_vsa_mask_kernel(const float *scores, uint8_t *mask,
                                   uint32_t n_tiles, uint32_t n_prefix,
                                   uint32_t keep, uint32_t heads) {
    uint32_t q_tile = (uint32_t)blockIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    if (q_tile >= n_tiles || head >= heads || threadIdx.x != 0u) return;
    uint8_t *row = mask + ((size_t)head * n_tiles + q_tile) * n_tiles;
    const float *score = scores + ((size_t)head * n_tiles + q_tile) * n_tiles;
    if (q_tile < n_prefix || keep >= n_tiles - n_prefix) {
        for (uint32_t col = 0; col < n_tiles; col++) row[col] = 1u;
        return;
    }
    for (uint32_t col = 0; col < n_prefix; col++) row[col] = 1u;
    for (uint32_t col = n_prefix; col < n_tiles; col++) row[col] = 0u;
    uint32_t video = n_tiles - n_prefix;
    uint32_t k = keep < video ? keep : video;
    for (uint32_t chosen = 0; chosen < k; chosen++) {
        float best = -INFINITY;
        uint32_t best_col = n_prefix;
        for (uint32_t col = n_prefix; col < n_tiles; col++) {
            if (row[col]) continue;
            float value = score[col];
            if (value > best || (value == best && col < best_col)) {
                best = value;
                best_col = col;
            }
        }
        row[best_col] = 1u;
    }
}

__global__ void h3_vsa_coarse_kernel(const float *scores,
                                     const float *value_pool, float *coarse,
                                     uint32_t n_tiles, uint32_t heads,
                                     uint32_t dim) {
    const uint32_t lane = threadIdx.x;
    const uint32_t q_tile = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    if (lane >= 32u || q_tile >= n_tiles || head >= heads || dim != 128u)
        return;
    const float *srow = scores + ((size_t)head * n_tiles + q_tile) * n_tiles;
    float max_score = -INFINITY;
    for (uint32_t kv = lane; kv < n_tiles; kv += 32u)
        max_score = fmaxf(max_score, srow[kv]);
    max_score = vsa_warp_max(max_score);
    float denom = 0.0f;
    for (uint32_t kv = lane; kv < n_tiles; kv += 32u)
        denom += __expf(srow[kv] - max_score);
    denom = vsa_warp_sum(denom);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (denom > 0.0f) {
        for (uint32_t kv = 0; kv < n_tiles; kv++) {
            float prob = __expf(srow[kv] - max_score) / denom;
            float4 value = *reinterpret_cast<const float4 *>(
                value_pool + ((size_t)head * n_tiles + kv) * dim + lane * 4u);
            a0 = fmaf(prob, value.x, a0);
            a1 = fmaf(prob, value.y, a1);
            a2 = fmaf(prob, value.z, a2);
            a3 = fmaf(prob, value.w, a3);
        }
    }
    float *out = coarse + ((size_t)head * n_tiles + q_tile) * dim + lane * 4u;
    out[0] = a0;
    out[1] = a1;
    out[2] = a2;
    out[3] = a3;
}

/* Sparse MMA fine: one block per (q_tile, head), 128 threads. Same shape as
 * dense h3_sdpa_bf16_mma_d128 / sol — m16n8k16 over TILE=64 KV tiles — but
 * walks only mask-selected tiles and finishes with fine + coarse * gate.
 * Head-major tiled layout: (head * padded + slot) * 128. */
__global__ __launch_bounds__(128) void h3_vsa_fine_mma_kernel(
    const uint16_t *__restrict__ query, const uint16_t *__restrict__ key,
    const uint16_t *__restrict__ value, const uint16_t *__restrict__ gate,
    const float *__restrict__ coarse, const uint8_t *__restrict__ mask,
    const uint32_t *__restrict__ block_sizes, uint16_t *__restrict__ output,
    float *fine_acc, uint32_t n_tiles, uint32_t dim, float scale) {
    __shared__ uint16_t k_tile[VSA_MMA_N * VSA_MMA_LD];
    __shared__ uint16_t v_tile[128u * VSA_MMA_VLD];
    uint16_t *q_tile = v_tile;

    constexpr uint32_t TILE = 64u;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t group = lane >> 2u;
    const uint32_t tig = lane & 3u;
    const uint32_t row_a = warp * 16u + group;
    const uint32_t row_b = row_a + 8u;
    const uint32_t q_tile_id = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    (void)fine_acc;
    if (dim != 128u || q_tile_id >= n_tiles || head >= 56u) return;

    const uint32_t padded = n_tiles * TILE;
    const uint32_t q_live = block_sizes[q_tile_id];
    const uint8_t *row = mask + ((size_t)head * n_tiles + q_tile_id) * n_tiles;
    const float *crow =
        coarse + ((size_t)head * n_tiles + q_tile_id) * dim;
    const uint32_t q0 = q_tile_id * TILE;

    for (uint32_t i = tid; i < VSA_MMA_M * 64u; i += 128u) {
        uint32_t r = i >> 6u;
        uint32_t column = (i & 63u) * 2u;
        uint32_t packed = 0;
        if (r < q_live) {
            size_t base =
                ((size_t)head * padded + q0 + r) * dim + column;
            packed = *(const uint32_t *)(query + base);
        }
        *(uint32_t *)&q_tile[r * VSA_MMA_LD + column] = packed;
    }
    __syncthreads();

    uint32_t q_frag[8][4];
#pragma unroll
    for (uint32_t kk = 0; kk < 8u; kk++) {
        uint32_t k0 = kk * 16u + tig * 2u;
        q_frag[kk][0] = *(const uint32_t *)&q_tile[row_a * VSA_MMA_LD + k0];
        q_frag[kk][1] = *(const uint32_t *)&q_tile[row_b * VSA_MMA_LD + k0];
        q_frag[kk][2] =
            *(const uint32_t *)&q_tile[row_a * VSA_MMA_LD + k0 + 8u];
        q_frag[kk][3] =
            *(const uint32_t *)&q_tile[row_b * VSA_MMA_LD + k0 + 8u];
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

    for (uint32_t kv = 0; kv < n_tiles; kv++) {
        if (!row[kv]) continue;
        const uint32_t live = block_sizes[kv];
        if (!live) continue;
        const uint32_t kv0 = kv * TILE;

        __syncthreads();
        for (uint32_t i = tid; i < VSA_MMA_N * 64u; i += 128u) {
            uint32_t r = i >> 6u;
            uint32_t column = (i & 63u) * 2u;
            uint32_t packed_k = 0;
            uint32_t packed_v = 0;
            if (r < live) {
                size_t base =
                    ((size_t)head * padded + kv0 + r) * dim + column;
                packed_k = *(const uint32_t *)(key + base);
                packed_v = vsa_bf16_pair_to_f16(*(const uint32_t *)(value + base));
            }
            *(uint32_t *)&k_tile[r * VSA_MMA_LD + column] = packed_k;
            v_tile[column * VSA_MMA_VLD + r] =
                (uint16_t)(packed_v & 0xffffu);
            v_tile[(column + 1u) * VSA_MMA_VLD + r] =
                (uint16_t)(packed_v >> 16u);
        }
        __syncthreads();

        float score[8][4];
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) score[j][e] = 0.0f;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                uint32_t b_frag[2];
                uint32_t r = j * 8u + (lane & 7u);
                uint32_t c = kk * 16u + ((lane >> 3u) & 1u) * 8u;
                vsa_ldmatrix_b16_x2(b_frag[0], b_frag[1],
                                   &k_tile[r * VSA_MMA_LD + c]);
                vsa_mma_m16n8k16_bf16(score[j], q_frag[kk], b_frag);
            }
        }

        float tile_max_a = -INFINITY;
        float tile_max_b = -INFINITY;
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            uint32_t column = j * 8u + tig * 2u;
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                if (column + e < live) {
                    score[j][e] *= scale;
                    score[j][e + 2u] *= scale;
                    tile_max_a = fmaxf(tile_max_a, score[j][e]);
                    tile_max_b = fmaxf(tile_max_b, score[j][e + 2u]);
                } else {
                    score[j][e] = -INFINITY;
                    score[j][e + 2u] = -INFINITY;
                }
            }
        }
#pragma unroll
        for (uint32_t m = 1u; m < 4u; m <<= 1u) {
            tile_max_a =
                fmaxf(tile_max_a, __shfl_xor_sync(0xffffffffu, tile_max_a,
                                                  (int)m));
            tile_max_b =
                fmaxf(tile_max_b, __shfl_xor_sync(0xffffffffu, tile_max_b,
                                                  (int)m));
        }
        float new_max_a = fmaxf(max_a, tile_max_a);
        float new_max_b = fmaxf(max_b, tile_max_b);
        float alpha_a =
            isfinite(new_max_a) ? __expf(max_a - new_max_a) : 1.0f;
        float alpha_b =
            isfinite(new_max_b) ? __expf(max_b - new_max_b) : 1.0f;
        float tile_sum_a = 0.0f;
        float tile_sum_b = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                float p_a = __expf(score[j][e] - new_max_a);
                float p_b = __expf(score[j][e + 2u] - new_max_b);
                score[j][e] = p_a;
                score[j][e + 2u] = p_b;
                tile_sum_a += p_a;
                tile_sum_b += p_b;
            }
        }
#pragma unroll
        for (uint32_t m = 1u; m < 4u; m <<= 1u) {
            tile_sum_a += __shfl_xor_sync(0xffffffffu, tile_sum_a, (int)m);
            tile_sum_b += __shfl_xor_sync(0xffffffffu, tile_sum_b, (int)m);
        }
        sum_a = sum_a * alpha_a + tile_sum_a;
        sum_b = sum_b * alpha_b + tile_sum_b;
        max_a = new_max_a;
        max_b = new_max_b;
        if (__any_sync(0xffffffffu, alpha_a != 1.0f || alpha_b != 1.0f)) {
#pragma unroll
            for (uint32_t dt = 0; dt < 16u; dt++) {
                out_acc[dt][0] *= alpha_a;
                out_acc[dt][1] *= alpha_a;
                out_acc[dt][2] *= alpha_b;
                out_acc[dt][3] *= alpha_b;
            }
        }
#pragma unroll
        for (uint32_t kk = 0; kk < 4u; kk++) {
            uint32_t p_frag[4] = {
                vsa_pack_f16_pair(score[kk * 2u][0], score[kk * 2u][1]),
                vsa_pack_f16_pair(score[kk * 2u][2], score[kk * 2u][3]),
                vsa_pack_f16_pair(score[kk * 2u + 1u][0],
                                  score[kk * 2u + 1u][1]),
                vsa_pack_f16_pair(score[kk * 2u + 1u][2],
                                  score[kk * 2u + 1u][3])};
#pragma unroll
            for (uint32_t dt = 0; dt < 16u; dt++) {
                uint32_t b_frag[2];
                uint32_t r = dt * 8u + (lane & 7u);
                uint32_t c = kk * 16u + ((lane >> 3u) & 1u) * 8u;
                vsa_ldmatrix_b16_x2(b_frag[0], b_frag[1],
                                   &v_tile[r * VSA_MMA_VLD + c]);
                vsa_mma_m16n8k16_f16(out_acc[dt], p_frag, b_frag);
            }
        }
    }

    float inverse_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    float inverse_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
#pragma unroll
    for (uint32_t dt = 0; dt < 16u; dt++) {
        uint32_t column = dt * 8u + tig * 2u;
        float c0 = crow[column];
        float c1 = crow[column + 1u];
        if (row_a < q_live) {
            size_t base =
                ((size_t)head * padded + q0 + row_a) * dim + column;
            uint32_t gpack = *(const uint32_t *)(gate + base);
            float g0 = vsa_bf16((uint16_t)gpack);
            float g1 = vsa_bf16((uint16_t)(gpack >> 16));
            *(uint32_t *)&output[base] = vsa_pack_bf16_pair(
                fmaf(c0, g0, out_acc[dt][0] * inverse_a),
                fmaf(c1, g1, out_acc[dt][1] * inverse_a));
        }
        if (row_b < q_live) {
            size_t base =
                ((size_t)head * padded + q0 + row_b) * dim + column;
            uint32_t gpack = *(const uint32_t *)(gate + base);
            float g0 = vsa_bf16((uint16_t)gpack);
            float g1 = vsa_bf16((uint16_t)(gpack >> 16));
            *(uint32_t *)&output[base] = vsa_pack_bf16_pair(
                fmaf(c0, g0, out_acc[dt][2] * inverse_b),
                fmaf(c1, g1, out_acc[dt][3] * inverse_b));
        }
    }
}

__global__ void h3_vsa_gather_kernel(const uint16_t *tiled, uint16_t *output,
                                     const uint32_t *slot, uint32_t sequence,
                                     uint32_t heads, uint32_t dim,
                                     uint32_t padded) {
    uint64_t n = (uint64_t)sequence * heads * dim;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += stride) {
        uint32_t d = (uint32_t)(idx % dim);
        uint64_t rest = idx / dim;
        uint32_t head = (uint32_t)(rest % heads);
        uint32_t token = (uint32_t)(rest / heads);
        uint32_t src = slot[token];
        output[((size_t)token * heads + head) * dim + d] =
            tiled[((size_t)head * padded + src) * dim + d];
    }
}


}  // namespace


extern "C" int h3_launch_vsa_bf16(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    const uint16_t *gate, uint16_t *output, const uint32_t *slot,
    const uint32_t *block_sizes, uint16_t *tiled_q, uint16_t *tiled_k,
    uint16_t *tiled_v, uint16_t *tiled_g, uint16_t *tiled_o, float *pool_q,
    float *pool_k, float *pool_v, float *scores, float *fine_acc,
    uint8_t *mask, uint32_t sequence, uint32_t heads, uint32_t dim,
    uint32_t padded, uint32_t n_tiles, uint32_t n_prefix, uint32_t keep,
    float scale, uint32_t kv_head_major, cudaStream_t stream) {
    if (!sequence || !heads || dim != 128u || !n_tiles || padded != n_tiles * 64u)
        return 0;
    /* Skip tiled memsets: pool/fine only touch block_sizes[tile] live slots,
     * and MMA masks dead KV columns with -inf. Padding slots are unread. */
    auto scatter = [&](const uint16_t *src, uint16_t *dst, uint32_t hm) {
        h3_vsa_scatter_kernel<<<vsa_grid((uint64_t)sequence * heads * dim), 256,
                                0, stream>>>(src, dst, slot, sequence, heads,
                                             dim, padded, hm);
        return cudaGetLastError() == cudaSuccess;
    };
    if (!scatter(query, tiled_q, 0u) || !scatter(key, tiled_k, kv_head_major) ||
        !scatter(value, tiled_v, kv_head_major) || !scatter(gate, tiled_g, 0u))
        return 0;
    auto pool = [&](const uint16_t *src, float *dest) {
        h3_vsa_pool_kernel<<<dim3(n_tiles, heads), 32, 0, stream>>>(
            src, dest, block_sizes, n_tiles, heads, dim, padded);
        return cudaGetLastError() == cudaSuccess;
    };
    if (!pool(tiled_q, pool_q) || !pool(tiled_k, pool_k) ||
        !pool(tiled_v, pool_v))
        return 0;
    h3_vsa_score_kernel<<<dim3(n_tiles, heads), 128, 0, stream>>>(
        pool_q, pool_k, scores, n_tiles, heads, dim, scale);
    if (cudaGetLastError() != cudaSuccess) return 0;
    h3_vsa_mask_kernel<<<dim3(n_tiles, heads), 32, 0, stream>>>(
        scores, mask, n_tiles, n_prefix, keep, heads);
    if (cudaGetLastError() != cudaSuccess) return 0;
    h3_vsa_coarse_kernel<<<dim3(n_tiles, heads), 32, 0, stream>>>(
        scores, pool_v, pool_q, n_tiles, heads, dim);
    if (cudaGetLastError() != cudaSuccess) return 0;
    h3_vsa_fine_mma_kernel<<<dim3(n_tiles, heads), 128, 0, stream>>>(
        tiled_q, tiled_k, tiled_v, tiled_g, pool_q, mask, block_sizes, tiled_o,
        fine_acc, n_tiles, dim, scale);
    if (cudaGetLastError() != cudaSuccess) return 0;
    h3_vsa_gather_kernel<<<vsa_grid((uint64_t)sequence * heads * dim), 256, 0,
                           stream>>>(tiled_o, output, slot, sequence, heads,
                                     dim, padded);
    return cudaGetLastError() == cudaSuccess;
}
