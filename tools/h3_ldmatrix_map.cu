/* Probe: scalar MMA-B fragment vs ldmatrix.x2 (no .trans). */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define LD 136u

__device__ __forceinline__ static void ldm_x2(uint32_t &lo, uint32_t &hi,
                                              const uint16_t *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(lo), "=r"(hi)
        : "r"(addr));
}

__global__ void probe_kernel(unsigned long long *mismatch) {
    __shared__ uint16_t k_tile[64 * LD];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t group = lane >> 2u;
    const uint32_t tig = lane & 3u;

    for (uint32_t i = tid; i < 64u * LD; i += 32u) k_tile[i] = 0;
    __syncwarp();
    for (uint32_t i = tid; i < 64u * 128u; i += 32u) {
        uint32_t r = i >> 7u;
        uint32_t c = i & 127u;
        k_tile[r * LD + c] = (uint16_t)((r << 8) | (c & 255u));
    }
    __syncwarp();

    unsigned long long bad = 0;
#pragma unroll 1
    for (uint32_t j = 0; j < 8u; j++) {
#pragma unroll 1
        for (uint32_t kk = 0; kk < 8u; kk++) {
            uint32_t k0 = kk * 16u + tig * 2u;
            uint32_t row = (j * 8u + group) * LD;
            uint32_t sc0 = *(const uint32_t *)&k_tile[row + k0];
            uint32_t sc1 = *(const uint32_t *)&k_tile[row + k0 + 8u];

            uint32_t r = j * 8u + (lane & 7u);
            uint32_t c = kk * 16u + ((lane >> 3u) & 1u) * 8u;
            uint32_t lo, hi;
            ldm_x2(lo, hi, &k_tile[r * LD + c]);
            if (lo != sc0 || hi != sc1) {
                bad++;
                if (bad == 1)
                    printf("first miss lane %u j %u kk %u sc %08x %08x ldm "
                           "%08x %08x\n",
                           lane, j, kk, sc0, sc1, lo, hi);
            }
        }
    }
    atomicAdd(mismatch, bad);
}

int main() {
    unsigned long long *d_bad;
    cudaMalloc(&d_bad, sizeof(unsigned long long));
    cudaMemset(d_bad, 0, sizeof(unsigned long long));
    probe_kernel<<<1, 32>>>(d_bad);
    cudaError_t e = cudaDeviceSynchronize();
    unsigned long long h = 0;
    cudaMemcpy(&h, d_bad, sizeof(h), cudaMemcpyDeviceToHost);
    printf("mismatches (sum over 32 lanes x 8 j x 8 kk) = %llu\n", h);
    if (e != cudaSuccess)
        fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));
    return (e == cudaSuccess && h == 0) ? 0 : 1;
}
