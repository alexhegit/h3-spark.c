# Changelog

## v0.2.0 — 2026-09-02

DGX Spark (GB10) shipping snapshot after the DiT / SDPA / VAE optimization
pass. Same fox-fast pixels as the bit gate (`f5282774d3a4`).

### Speed (warm, `--profile`, seed 42)

| Preset | E2E | Denoise |
|---|---:|---:|
| fox-s2 (512², steps 2, L35 R1) | **8.0 s** | **1.19 s** |
| fox-fast (512², steps 20, L45 R2) | **15.5 s** | **8.2 s** |
| 15 s cinematic (864×480, L45 R2) | **18 min 17 s** | **16 min 51 s** |
| 15 s + `--token-reduction` | **11 min 22 s** | **9 min 53 s** |

v0.1.0 fox-fast was **1726 s** e2e / **1472 s** denoise. Remaining fox-fast
time is mostly INT8 linear, not QK MMA. Remaining 15 s time is long-N SDPA.
GB10 has no wider dense-BF16 MMA (`tcgen05` / WGMMA / WMMA m16n16 / `ldmatrix.x4`
are closed).

`--token-reduction` stays **opt-in**. Fox-fast vs off is ~17.8 dB PSNR / 0.72
SSIM; do not treat it as lossless.

### Product

- `--info` prints `h3-spark` and `H3_VERSION` **0.2.0**.
- README / PERF_BASELINE quote the HIP-page presets measured on this tree.

## v0.1.0 — 2026-08-17

First tagged CUDA port: T2VA / FL2VA / Ref2VA generate on GB10, `--profile`
marks, pre-optimization fox-fast baseline (~28.8 min e2e).
