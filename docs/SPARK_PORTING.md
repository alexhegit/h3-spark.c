# h3-cuda: DGX Spark Porting Plan

This document describes how to port **h3-metal** to **NVIDIA DGX Spark** while
preserving the two properties that define the project:

1. **Performance** — hand-fused kernels, int8 paths, activation aliasing, and
   explicit residency control, not a generic deep-learning stack.
2. **Simplicity** — a single C binary with minimal runtime dependencies, not
   Python, PyTorch, TVM, or TensorRT.

The working name for the Spark target is **h3-cuda**. The public C API in `h3.h`
should remain stable; only the GPU backend and a few platform-specific pieces
change.

## Codebase structure (current h3-metal)

The repository is ~37k lines across 62 source files. Understanding the layering
is essential: the port is **not** “swap two GPU files”; it is “implement
`h3_gpu.h` on CUDA while keeping the model/host stack intact.”

### Layer diagram

```
main.c / h3_cli.c          CLI, interactive REPL (linenoise), terminal preview
        │
h3.c / h3_multimodal.c     Public API, generate pipeline, FL2VA / Ref2VA routing
        │
h3_host.c                  Layout, sigmas, RNG, Euler/res solver, RGB resize
h3_safetensors.c           Safetensors header parse + pread (portable)
h3_weights.c               Shard discovery → h3_gpu_tensor_load_*()
        │
h3_dit.c                   50-block denoise loop, fusion flags, SSD streaming
h3_dit_schedule.c          Precomputed AdaLN modulation on GPU
h3_text_encoder.c          Qwen3 text (835 lines, ~115 GPU calls)
h3_vision_encoder.c        Qwen3-VL vision (633 lines, ~119 GPU calls)
h3_video_vae.c             Video VAE decode (1252 lines, ~102 GPU calls)
h3_video_encoder.c         Visual VAE encode (814 lines, ~78 GPU calls)
h3_audio_vae.c             BigVGAN / AudioVAE (1343 lines, ~108 GPU calls)
        │
h3_gpu.h                   Stable C API — 87 exported functions
        │
h3_gpu.m + h3_shaders.metal   Metal + MPSGraph + 83 custom kernels
h3_metal.m                 Device probe → h3_device_info
h3_tokenizer.m             BPE tokenizer (Foundation + ICU)
```

### File inventory

| Role | Files | Lines (approx) | Spark action |
|------|-------|---------------:|--------------|
| Public / orchestration | `h3.c`, `h3_internal.h`, `h3.h` | 1,960 | Keep |
| Host math / layout | `h3_host.c`, `h3_host.h` | 790 | Keep; replace vImage (~50 lines) |
| Safetensors / weights | `h3_safetensors.c`, `h3_weights.c` | 880 | Keep unchanged |
| DiT | `h3_dit.c`, `h3_dit_schedule.c`, headers | 3,700 | Keep; retarget device probes |
| Encoders / VAE | 5 module `.c` files + headers | 4,900 | Keep logic; depends on GPU ops |
| Multimodal | `h3_multimodal.c` | 410 | Keep |
| Media | `h3_ffmpeg.c` | 755 | Keep (already POSIX spawn) |
| CLI / terminal | `main.c`, `h3_cli.c`, `h3_terminal.c`, `linenoise.c` | 2,400 | Keep; minor Linux UX |
| GPU backend | `h3_gpu.m`, `h3_shaders.metal`, `h3_gpu.h` | 9,650 | Rewrite `.m`/`.metal` → `.cu` |
| Device probe | `h3_metal.m`, `h3_metal.h` | 55 | Replace with `h3_cuda.c` |
| Tokenizer | `h3_tokenizer.m`, `h3_tokenizer.h` | 550 | Rewrite as `h3_tokenizer.c` |
| Tests | 22 binaries under `tests/` | 5,500 | Adapt runners; same fixtures |

### What is already portable

These files have **no Apple `#ifdef`** and no Metal headers:

- `h3_safetensors.c` — pure POSIX `pread`, fully reusable.
- `h3_weights.c` — opens shards, validates shapes, calls `h3_gpu_tensor_load_bf16()`.
- `h3_ffmpeg.c` — `posix_spawnp` for ffmpeg/ffprobe; works on Linux today.
- `h3_dit_schedule.c`, `h3_host.c` (except vImage), `h3_multimodal.c`.
- DiT orchestration in `h3_dit.c` including pthread SSD prefetch and Qwen weight
  ring prefetch in `h3_text_encoder.c` — host threading is portable; only the
  GPU buffer targets change.

Portability today is enforced by the **Makefile** (Apple frameworks), not by
conditional compilation in C sources.

## Target hardware

DGX Spark (Grace Blackwell GB10 Superchip):

| Component | Specification |
|-----------|---------------|
| CPU | 20-core ARM64 (10× Cortex-X925 + 10× Cortex-A725) |
| GPU | NVIDIA GB10 Blackwell, compute capability **sm_12.1** |
| Memory | 128 GB LPDDR5x unified (UMA), ~273 GB/s bandwidth |
| OS | Linux (Ubuntu / DGX OS) |
| CUDA | Driver 580+, CUDA 13.x |

### What transfers well from Apple Silicon

- **Unified memory** — same conceptual model as M-series UMA; zero-copy weight
  mapping and SSD streaming remain relevant.
- **128 GB capacity** — sufficient for the full H3 pipeline (documented peaks
  around 40 GB for end-to-end reference runs; int8 path around 26 GB peak
  tensor storage).
- **BF16 / int8** — matches the production compute dtype and the fastest M5
  inference path.
- **ARM64 host** — most host-side C code is portable; only Apple-specific APIs
  need replacement.

### Spark-specific constraints

- **Memory bandwidth (273 GB/s)** is likely tighter than M5 Max unified memory;
  int8 quantization and `--ssd-streaming` may matter more, not less.
- **140 W TDP** — thermal throttling is a real concern, as on Mac; fused kernels
  that reduce dispatch count and global memory traffic remain important.
- **Integrated GPU** — not a datacenter H100; peak throughput depends on
  kernel efficiency, not library defaults.

## What does not port directly

| Apple-only component | Role |
|---------------------|------|
| `h3_gpu.m` (~4,711 lines) | Metal context, MPSGraph caches, dispatch, weight I/O |
| `h3_shaders.metal` (~4,332 lines) | 83 custom compute kernels + Metal 4 TensorOps |
| `h3_metal.m` (47 lines) | Fills `h3_device_info` from `MTLDevice` |
| `h3_tokenizer.m` (521 lines) | BPE via Foundation collections + ICU |
| `h3_host.c` vImage block (~50 lines) | `h3_resize_rgb24_high_quality()` for reference scaling |
| `h3_cli.c` `open` command | macOS `posix_spawnp("open", …)` for `!show` playback |
| `Makefile` | `-framework Metal/MPS/MPSGraph/Accelerate`, `-licucore` |

A direct `make` on Linux fails immediately (no Apple frameworks, no Metal).

## The `h3_gpu.h` boundary (critical)

All model code talks to the GPU only through **`h3_gpu.h`** (87 functions).
No `.c` file includes Metal headers. This boundary is the correct port surface.

### Capability probes (must generalize, not copy verbatim)

Metal-specific probes leak into portable C today:

| API | Metal meaning | Used in | Spark equivalent |
|-----|---------------|---------|------------------|
| `h3_gpu_is_m5()` | Device name contains `"M5"` | `h3_dit.c`, `h3_text_encoder.c` | `h3_gpu_has_fast_path()` or detect GB10 / sm ≥ 12.0 |
| `h3_gpu_has_nax_mlp()` | Metal 4 tensor MLP mode | `h3_dit.c` L1632 | Blackwell fused MLP availability |
| `h3_gpu_has_int8_mlp()` | Metal 4 tensor ops enabled | `h3_dit.c` L1633–1642 | Blackwell int8 MMA path available |
| `h3_device_info.metal4` | `MTLGPUFamilyMetal4` | `h3.c` L562 (`--use-int8-row-fc2`) | e.g. `tensor_core_int8` or reuse `metal4` as generic “fast GPU” flag |
| `h3_metal_probe()` | Populates device info | `h3.c`, `test_h3.c` | `h3_cuda_probe()` filling same struct |

Recommendation: keep `h3_device_info` layout stable for CLI `--info`, but document
field semantics on Spark (map `metal4` → “supports default int8 fast path”).

### Shader source path parameter

`h3_gpu_create(const char *shader_source_path, …)` reads and **runtime-compiles**
`h3_shaders.metal` on Metal (see `h3_gpu.m` L353–403). The string
`"h3_shaders.metal"` is hardcoded at **30+ call sites** in `h3.c`, encoders,
VAE modules, and tests.

On CUDA:

- Kernels are **build-time compiled** by `nvcc` into the binary (or a loaded
  `.so`); there is no runtime MSL compile step.
- Keep the parameter for API compatibility; ignore it or use it only in debug
  (e.g. load PTX from path). Production builds embed cubins at link time.

### MPSGraph is a hidden backend layer

Custom Metal kernels are only part of the story. `h3_gpu.m` also maintains
**cached MPSGraph** objects for:

- BF16 SDPA / GQA when native flash path does not apply
- BF16 linear / MLP (non-NAX shapes)
- **Conv1d** (audio VAE) and **Conv3d** (video VAE encoder/decoder)

Classes: `H3SDPA`, `H3GQA`, `H3Linear`, `H3MLP`, `H3Conv` with shape-keyed caches.

The CUDA port must plan explicit implementations for these fallback paths—not
only the 83 named kernels. During bring-up, **cuBLASLt** may substitute for
wide BF16 GEMM; conv stacks need custom CUDA or a minimal im2col+GEMM layer
(avoid linking cuDNN).

### Stats and profiling

`h3_gpu_stats` (`h3_gpu.h` L17–33) names Metal concepts (`mps_linear_dispatches`,
`mps_sdpa_dispatches`, …). Either:

- Rename to neutral counters (`graph_gemm_dispatches`), or
- Keep names as opaque buckets filled by the CUDA backend.

Tests in `test_bf16.c` and `test_text_metal.c` assert MPSGraph was used for
certain wide shapes; CUDA tests should assert the equivalent fallback fired.

## Rejected approaches

| Approach | Why rejected |
|----------|--------------|
| **Official PyTorch / SGLang** | Heavy dependency stack; not a single C binary. |
| **TileLang** | Requires Python + TVM + PyTorch; changes the product shape even though it supports Linux AArch64 and SM120. Useful only as an offline SM120 GEMM spike, not as the shipping backend. |
| **TensorRT / cuDNN** | Large, version-coupled runtime; conflicts with “minimal dependencies”. |
| **Mechanical Metal → CUDA translation** | MSL and MPSGraph have no portable equivalent; fusion layouts must be re-designed. |

## Recommended architecture: h3-cuda

```
┌──────────────────────────────────────────────────────────┐
│  h3.c, h3_dit.c, h3_dit_schedule.c, h3_multimodal.c     │  keep
│  h3_weights.c, h3_safetensors.c                          │
│  h3_text_encoder.c, h3_vision_encoder.c                  │
│  h3_video_vae.c, h3_video_encoder.c, h3_audio_vae.c      │
│  h3_cli.c, h3_ffmpeg.c, h3_terminal.c, main.c            │
├──────────────────────────────────────────────────────────┤
│  h3_gpu.h          (stable C API, 87 functions)        │  keep interface
├──────────────────────────────────────────────────────────┤
│  h3_gpu.cu + h3_kernels.cu   (replaces .m + .metal)      │
│  h3_cuda.c                   (replaces h3_metal.m)       │
├──────────────────────────────────────────────────────────┤
│  h3_tokenizer.c              (replaces h3_tokenizer.m)   │
├──────────────────────────────────────────────────────────┤
│  h3_host.c                   (vImage → vendored scaler)  │
└──────────────────────────────────────────────────────────┘
```

Suggested new files (do not `#ifdef` the existing Metal tree initially; use
`Makefile.linux` on the `spark` branch):

| File | Purpose |
|------|---------|
| `h3_gpu.cu` | Context, streams, tensors, dispatch, graph-cache equivalent |
| `h3_kernels.cu` | Custom kernels (GEMM tiles, fusion, int8, token pool, …) |
| `h3_cuda.c` / `h3_cuda.h` | `h3_cuda_probe()`, optional alias `h3_device_probe()` |
| `h3_tokenizer.c` | Pure C BPE + ICU |
| `Makefile.linux` | `gcc` + `nvcc`, link `-lcudart -licuuc -lm -pthread` |

### Code reuse estimate (by line count)

| Category | Lines | Share | Action |
|----------|------:|------:|--------|
| Host / model / CLI / safetensors | ~19,000 | 52% | Keep |
| GPU backend (`.m` + `.metal`) | ~9,000 | 25% | Rewrite |
| Tests | ~5,500 | 15% | Adapt |
| Tokenizer + vImage + probe | ~620 | 2% | Rewrite |
| Vendored linenoise | ~1,900 | 5% | Keep |
| Docs / misc | ~500 | 1% | — |

## Dependency budget

Spark builds should link only system-level, single-purpose libraries:

| Dependency | Purpose | Required |
|------------|---------|:--------:|
| CUDA driver + **libcudart** | GPU allocation, streams, kernels | yes |
| **libicu** (`icu-uc`) | BPE tokenizer (Mac uses `-licucore`) | yes |
| **FFmpeg / FFprobe** | Media I/O via spawn (not linked today) | yes (PATH) |
| **libm**, **pthread** | Standard C runtime | yes |
| **libcublasLt** | BF16 GEMM oracle / bootstrap; replace with custom tiles | optional |

Do **not** depend on: PyTorch, Python, TVM/TileLang, cuDNN, TensorRT, NCCL,
`libav*` (linking FFmpeg libraries would break the current spawn-only model),
Hugging Face `transformers`, or any pip/conda environment at runtime.

### Image resize without new link dependencies

`h3_host.c` uses Accelerate vImage for reference image / video canvas scaling.
`h3_ffmpeg.c` does **not** link libav — do not add `libswscale` unless explicitly
accepted as a new dependency. Preferred options:

1. Small **vendored** high-quality scaler (single-file C), or
2. Minimal bilinear/bicubic in `h3_host.c` for the one code path (~50 lines of
   call site), validated against existing `test_h3.c` resize test.

## Weight loading and memory modes

### Safetensors (unchanged)

`h3_safetensors.c` parses JSON headers and reads payloads with `pread`. No GPU
 involvement. All inventory / `--info` paths stay as-is.

### GPU tensor I/O (`h3_gpu.m` → `h3_gpu.cu`)

| API | Metal behavior | Linux / CUDA plan |
|-----|----------------|-------------------|
| `h3_gpu_tensor_load_bf16` | Shared `MTLBuffer`; pread or mmap | UMA: `cudaMallocManaged` or device alloc + pread; optional `mmap` for `/transformer/` when `H3_ZERO_COPY_WEIGHTS` |
| `h3_gpu_tensor_read_file_bf16` | pread into existing buffer | Same; used by Qwen prefetch threads |
| `h3_gpu_tensor_stream_file_bf16` | pread + `F_NOCACHE` (Darwin) | `posix_fadvise(DONTNEED)` after read, or `O_DIRECT` where aligned |

M5 auto-mmaps transformer weights (`H3_ZERO_COPY_WEIGHTS`); Spark should default
to an analogous file-backed UMA mapping on NVMe.

### SSD streaming (`--ssd-streaming`)

Implemented in `h3_dit.c` (double buffer, pthread prefetch, norms-only resident).
Constraints **carry over unchanged**:

- Uses **original BF16** checkpoint weights only (no conversion).
- **Incompatible with int8 paths** — `h3_dit.c` L1633–1642 disables int8 when
  `ssd_streaming` is set; `h3.c` L553–556 rejects `--use-int8-row-fc2` combo.

### Qwen layer prefetch

`h3_text_encoder.c` uses a pthread ring (depth 3 on M5, 2 otherwise via
`h3_gpu_is_m5()`). Host logic is portable; map “M5 depth 3” to “GB10 depth 3”
via the generalized fast-path probe.

## GPU kernel porting priorities

Performance comes from **custom fusion** plus selective fallbacks. The Metal
side has **83 kernels** and **five MPSGraph cache families**; CUDA must cover both.

### P0 — DiT hot path (first useful inference)

| Work item | Notes |
|-----------|-------|
| BF16 linear / MLP | Tiled GEMM; cuBLASLt as oracle only |
| int8 QKV + MLP + attention output | WMMA / Blackwell MMA (sm_120); gates mirror `h3_gpu_has_int8_mlp()` |
| SDPA / GQA | Custom flash-attention-style kernel (DiT + Qwen shapes) |
| SwiGLU | Fused FC1 epilogue |
| **`test_real_dit_block`** | Recommended **first vertical slice** — real block 0 vs MLX fixture |

### P1 — fusion kernels (competitive speed)

| Kernel family | Metal diagnostic flag |
|---------------|----------------------|
| QKV + RMSNorm + RoPE (grouped layout) | `H3_DISABLE_FUSED_*` / `--use-slower-unfused-qkv-rope` |
| AdaLN + gate + int8 quantize | `--use-slower-unfused-int8-inputs` |
| Token pool / expand | `--token-reduction` |
| Patch linear 16×16 tiles | `H3_SCALAR_PATCH`, `H3_DISABLE_FUSED_PATCH_*` |
| Fused final AdaLN + head | `H3_DISABLE_FUSED_FINAL_*` |
| Gate + cross-block AdaLN | `H3_DISABLE_FUSED_GATE_ADALN`, `H3_DISABLE_FUSED_CROSS_BLOCK_ADALN` |

Port the **~40 `H3_DISABLE_*` / `H3_NAX*` environment variables** read in
`h3_dit.c` and `h3_gpu.m` so each fusion has a CUDA oracle for regression and
`tests/bench_dit.c` A/B remains meaningful.

### P2 — full pipeline modules

| Module | Primary GPU ops | Approx GPU calls |
|--------|-----------------|-----------------:|
| `h3_text_encoder.c` | BF16 GQA, SwiGLU, embedding | 115 |
| `h3_vision_encoder.c` | BF16 attention, patch embed | 119 |
| `h3_video_vae.c` | Conv3d, group norm, SiLU, decode loops | 102 |
| `h3_video_encoder.c` | Conv3d encode, pad | 78 |
| `h3_audio_vae.c` | Conv1d, ConvTranspose1d, Snake, SDPA | 108 |
| GPU Euler sampler | `h3_gpu_euler_bf16` | M5 default in `h3_dit.c` L2584 |

Conv stacks currently use **MPSGraph** (`H3Conv` cache). Plan custom CUDA conv
or im2col+GEMM; do not assume cuDNN.

Algorithm ideas in `h3_shaders.metal` (Morton scheduling, dynamic int8 quant;
see `THIRD_PARTY_NOTICES.md`) port at the **algorithm** level.

## Host-side and UX changes

### Tokenizer (`h3_tokenizer.m` → `h3_tokenizer.c`)

- Pure C + ICU (`unicode/uchar.h` already included on Mac).
- Drop Foundation (`NSDictionary`, `NSString`, BPE cache objects).
- Validate with `h3_tokenizer_tests` against `MiniMax-H3/tokenizer/tokenizer.json`.

### Device probe (`h3_metal.m` → `h3_cuda.c`)

Populate existing `h3_device_info`:

```c
typedef struct {
    char name[128];
    char architecture[128];
    uint64_t physical_memory;
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
    int apple_gpu_family;   /* repurpose or add sm_major/minor on Spark */
    int metal4;             /* 1 = default int8 / fused fast path enabled */
    int unified_memory;
} h3_device_info;
```

Wire `h3.c` L453 from `h3_metal_probe` to CUDA probe (or thin `h3_device_probe`
wrapper used on both platforms).

### CLI / terminal (minor)

| Item | Change |
|------|--------|
| `h3_cli.c` `open_video()` | Use `xdg-open` on Linux, or no-op with message |
| `h3_terminal.c` | Kitty / iTerm2 protocols work on Linux; keep FFmpeg spawn for PNG |
| `main.c` `--info` | Print CUDA device name, compute capability, UMA size |

All CLI flags in `main.c` (including 12× `--use-slower-*`, `--ssd-streaming`,
`--use-int8-row-fc2`) must retain semantics on Spark.

## Build structure

Current `Makefile`:

- **`LIB_C`** (14 files) → static `libh3.a`
- **`LIB_M`** (3 ObjC files) → same archive
- **`h3`** ← `main.o`, `h3_cli.o`, `linenoise.o` + `libh3.a`
- Link: `-framework Foundation Metal MPS MPSGraph Accelerate -licucore -lm`

Linux target:

```makefile
# Makefile.linux (sketch)
NVCC = nvcc
CC   = gcc
LIB_C = … (same 14 files)
LIB_CUDA = h3_gpu.o h3_kernels.o   # from .cu
LIB_HOST = h3_cuda.o h3_tokenizer.o
# h3: main.o h3_cli.o linenoise.o $(LIB_C:.c=.o) $(LIB_CUDA) $(LIB_HOST)
# LDLIBS = -lcudart -licuuc -lm -pthread  (+ optional -lcublasLt)
```

Keep **`make test`** structure; add CUDA parity runners alongside Metal names:

| Metal test | CUDA counterpart |
|------------|------------------|
| `h3_metal_tests` | `h3_cuda_tests` (same fixtures) |
| `h3_bf16_tests` | keep name or alias |
| `test_h3.c::test_metal_probe` | `test_cuda_probe` |
| `tests/bench_dit.c` | unchanged driver; CUDA backend |

## Implementation phases

### Phase 0 — Scaffold (≈1 week)

- [ ] `Makefile.linux`, `h3_cuda.c` probe, `h3_tokenizer.c` smoke test
- [ ] `h3_gpu.cu`: context, streams, tensor alloc/free, BF16 copy/cast
- [ ] vImage replacement in `h3_host.c`; `test_h3.c` resize passes
- [ ] `./h3 --info -d ./MiniMax-H3` prints CUDA device

### Phase 1 — DiT block parity (≈3–4 weeks)

- [ ] Implement P0 ops behind `h3_gpu.h`
- [ ] `h3_cuda_tests` / adapted `test_bf16.c` vs `misc/fixtures/`
- [ ] **`test_real_dit_block`** — complete block 0 forward vs MLX golden
- [ ] cuBLASLt oracle for wide GEMM shapes covered by MPSGraph on Mac

### Phase 2 — Performance path (≈4–6 weeks)

- [ ] int8 QKV / MLP / attention-output (default when `metal4`-equivalent set)
- [ ] P1 fusion kernels + `H3_DISABLE_*` parity
- [ ] Zero-copy / SSD streaming on NVMe
- [ ] `tests/bench_dit.c` A/B on Spark; `--profile` phase breakdown

### Phase 3 — Full pipeline (≈3–4 weeks)

- [ ] Text + vision encoders, audio/video VAE conv paths
- [ ] `h3_generate()` end-to-end; interactive CLI
- [ ] README presets (512² fox, Ref2VA, FL2VA) validated on Spark

**Estimated calendar time:** 3–4 months for one experienced GPU engineer with
weights and fixtures on Spark.

## Validation strategy

1. **Host tests** — `h3_tests`: layout, schedule, RNG, safetensors, resize (no GPU kernels).
2. **Kernel parity** — `misc/fixtures/` MLX golden tensors; same relative bounds as Metal.
3. **Real-weight tests** — 15+ `test_real_*` binaries (conditional on weights in Makefile).
4. **Fusion regression** — `bench_dit.c` + `H3_DISABLE_*` oracles.
5. **End-to-end** — visual review; no byte-identical requirement vs MLX (already documented for Metal).

## Risks

| Risk | Mitigation |
|------|------------|
| GB10 sm_12.1 toolchain immaturity | cuBLASLt oracle; pin CUDA 13.x; test on Spark early |
| MPSGraph fallback surface underestimated | Explicit conv + wide-GEMM plan in Phase 2, not just 83 kernels |
| `h3_gpu_is_m5()` string matching | Generalize capability API before forking model logic |
| Lower memory bandwidth | Prioritize int8, fusion, `--ssd-streaming` benchmarks |
| ~9k lines GPU rewrite | Strict `h3_gpu.h` contract; block-0 vertical slice first |
| Dual backend maintenance | Shared C core; identical CLI; fusion flags documented |
| Thermal throttling (140 W) | Warm repeated runs; `--profile` wall vs GPU time |

## Optional: TileLang as a spike only

[TileLang](https://github.com/tile-ai/tilelang) supports Linux AArch64 and SM120,
but is **not** part of the shipping dependency model. Offline use only: prototype
a GEMM or attention tile, then reimplement as static CUDA in `h3_kernels.cu`.

## Success criteria

1. `./h3` is a **standalone Linux AArch64 binary** with dependencies in
   [Dependency budget](#dependency-budget) only.
2. Prompt-to-video/audio at 512×512 with README default presets.
3. `--profile` DiT denoise within a reasonable factor of M5 Max (TBD after Phase 2).
4. `make test` passes host + parity + real-weight tests when fixtures installed.
5. No Python or PyTorch at runtime.
6. All existing CLI flags and `H3_DISABLE_*` diagnostics behave consistently.

## References

- [NVIDIA DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark porting guide (UMA)](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)
- h3-metal README — presets, memory modes, fusion flags
- `h3_gpu.h` — 87-function backend contract
- `h3_dit.c` L1630–1660 — int8 / SSD streaming / fast-path gating
- `THIRD_PARTY_NOTICES.md` — Morton / int8 matmul design lineage

---

*Branch: `spark` · Status: planning · Last updated: 2026-08-12*
