# Changelog

## v0.2.1 — 2026-09-07

Smaller SDPA default plus an opt-in VAE VRAM path. Fox-fast pixels still
`f5282774d3a4`. 15 s cinematic was **not** re-timed (44800 SDPA −2.4% is
below the 15% bar).

### Speed (warm, `--profile`, seed 42)

| Preset | E2E | Denoise |
|---|---:|---:|
| fox-s2 (512², steps 2, L35 R1) | **8.0 s** | **1.19 s** |
| fox-fast (512², steps 20, L45 R2) | **15.5 s** | **8.06 s** (sdpa 1.38 / linear 5.13) |
| 15 s cinematic (864×480, L45 R2) | **18 min 17 s** (last measured, v0.2.0) | **16 min 51 s** |
| 15 s + `--token-reduction` | **11 min 22 s** (last measured) | **9 min 53 s** |

### Changes

- Default DiT SDPA stores V transposed so P·V uses the same `ldmatrix.x2` B
  map as QK. Fox-fast md5 unchanged; sdpa **1.43 s → 1.38 s**.
- `H3_INT8_VAE=1` quantizes video-VAE block linears. fox-s2 VAE peak
  **9.45 → 2.73 GiB**, PSNR **43.2 dB** vs F32. Not default.
- Quality gate for default opts: fox-fast PSNR ≥ 24 dB vs the v0.2.0 ref.
- `--info` prints `H3_VERSION` **0.2.1**.

### Retest 2026-09-09 (`7420692`)

Same tree, GB10. Logs `/tmp/h3_rebench/`. Pixels unchanged.

| Preset | E2E | Denoise |
|---|---:|---:|
| fox-s2 | **8.2 s** | **1.20 s** |
| fox-fast | **15.6 s** | **8.17 s** (1.40 / 5.22) |
| 15 s cinematic | **17 min 56 s** | **16 min 28 s** (845 / 110) |

`make test` **16.2 s**; `test-conditional` **4 min 33 s** (no `--ref-video`).
`--token-reduction` not re-timed.

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
