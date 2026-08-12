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

h3-metal cannot be compiled or run on Spark as-is:

| Apple-only component | Role |
|---------------------|------|
| `h3_gpu.m` (~4,700 lines) | Metal + MPS + MPSGraph backend |
| `h3_shaders.metal` (~4,300 lines) | 83 custom GPU kernels |
| `h3_metal.m` | Device probe |
| `h3_tokenizer.m` | Objective-C + Foundation tokenizer |
| `h3_host.c` (partial) | Accelerate / vImage frame scaling |
| `Makefile` | `-framework Metal`, `-framework Accelerate`, etc. |

A direct `make` on Linux fails immediately (no Apple frameworks, no Metal).

## Rejected approaches

These paths were evaluated and **do not meet the project goals**:

| Approach | Why rejected |
|----------|--------------|
| **Official PyTorch / SGLang** | Heavy dependency stack; not a single C binary. |
| **TileLang** | Requires Python + TVM + PyTorch; changes the product shape even though it supports Linux AArch64 and SM120. Useful only as an optional SM120 GEMM spike, not as the shipping backend. |
| **TensorRT / cuDNN** | Large, version-coupled runtime; conflicts with “minimal dependencies”. |
| **Translating Metal → CUDA mechanically** | Metal Shading Language and MPSGraph have no portable equivalent; fusion layout and threadgroup semantics must be re-designed. |

## Recommended architecture: h3-cuda

Mirror the existing vertical-slice design: keep the C host and model logic,
replace only the GPU layer behind `h3_gpu.h`.

```
┌─────────────────────────────────────────────┐
│  h3.c, h3_dit.c, h3_dit_schedule.c          │  keep
│  h3_weights.c, h3_safetensors.c             │
│  h3_cli.c, h3_ffmpeg.c, h3_multimodal.c     │
│  h3_text_encoder.c, h3_*_vae.c, ...         │
├─────────────────────────────────────────────┤
│  h3_gpu.h          (stable C API)           │  keep interface
├─────────────────────────────────────────────┤
│  h3_gpu.cu + h3_kernels.cu                  │  replace h3_gpu.m + .metal
├─────────────────────────────────────────────┤
│  h3_tokenizer.c                             │  replace h3_tokenizer.m
├─────────────────────────────────────────────┤
│  h3_host.c (vImage → swscale or inline)     │  small platform swap
└─────────────────────────────────────────────┘
```

### Code reuse estimate

| Category | Share | Action |
|----------|------:|--------|
| Host / model / CLI / safetensors | ~55% | Keep, minor `#ifdef` or Linux paths |
| GPU backend | ~40% | Rewrite as CUDA |
| Tokenizer + image scale | ~5% | Pure C / FFmpeg |

## Dependency budget

Spark builds should link only system-level, single-purpose libraries:

| Dependency | Purpose | Required |
|------------|---------|:--------:|
| CUDA driver + **libcudart** | GPU allocation, streams, kernels | yes |
| **libicu** (`icu-uc`) | BPE tokenizer (already used via `icucore` on Mac) | yes |
| **FFmpeg / FFprobe** | Media I/O (already required) | yes |
| **libm**, **pthread** | Standard C runtime | yes |
| **libcublasLt** | BF16 GEMM oracle / bootstrap only; replace with custom tiles later | optional |

Do **not** depend on: PyTorch, Python, TVM/TileLang, cuDNN, TensorRT, NCCL,
Hugging Face `transformers`, or any pip/conda environment at runtime.

Target deliverable:

```sh
./h3 -d ./MiniMax-H3 -p "..." -o outputs/out.mp4
```

One executable, same CLI surface as h3-metal.

## GPU kernel porting priorities

Performance comes from **custom fusion**, not from calling generic BLAS for
everything. The Metal side has **83 kernels** and extensive MPSGraph use; the
CUDA side must reproduce the same fusion strategy.

### P0 — DiT hot path (required for useful inference)

| Metal / MPS area | CUDA plan |
|------------------|-----------|
| BF16 linear / MLP | Tiled GEMM kernels; cuBLASLt only as numerical oracle during bring-up |
| int8 QKV + MLP + attention output | WMMA / Blackwell MMA int8 paths (sm_120) |
| SDPA / GQA causal attention | Custom flash-attention-style kernel |
| SwiGLU | Fused epilogue on FC1 output |

### P1 — fusion kernels (required for competitive speed)

| Kernel family | Notes |
|---------------|-------|
| QKV + RMSNorm + RoPE | H3 grouped QKV layout; preserve checkpoint row order |
| AdaLN + gate + int8 quantize | Single-kernel boundaries documented via `H3_DISABLE_FUSED_*` on Mac |
| Token pool / expand | H3-specific token-reduction mode |
| Patch linear 16×16 tiles | DiT / VAE boundary kernels |
| Fused final AdaLN + head | Large activation savings at 512² and 864-class shapes |

### P2 — full pipeline

| Component | Notes |
|-----------|-------|
| Qwen text encoder layers | BF16 GQA + SwiGLU |
| Qwen vision tower | BF16 attention + conv paths |
| Video / audio VAE | Conv3d / Conv1d / ConvTranspose1d in CUDA |
| GPU Euler sampler | Optional; CPU sampler can bootstrap first |

Algorithm ideas in `h3_shaders.metal` (Morton scheduling, dynamic int8 quant,
ccv-derived matmul structure per `THIRD_PARTY_NOTICES.md`) should be ported at
the **algorithm** level, not by adopting a new framework.

## Host-side changes

### Tokenizer (`h3_tokenizer.m` → `h3_tokenizer.c`)

- Rewrite in **pure C** using existing ICU usage (`unicode/uchar.h`).
- Drop Foundation / NSString / NSDictionary.
- Preserve byte-level BPE compatibility; validate against
  `MiniMax-H3/tokenizer/tokenizer.json` and existing tests.

### Image scaling (`h3_host.c`)

- Replace Accelerate **vImage** with **FFmpeg libswscale** (already a project
  dependency for media) or a small vendored scaler.
- Keep the same RGB24 frame API for callbacks and MP4 output.

### Weight I/O

- Keep safetensors parsing (`h3_safetensors.c`) unchanged.
- Map `--ssd-streaming` to Linux `pread()` with the same double-buffer overlap
  model as Darwin uncached reads.
- Use UMA-appropriate allocation (`cudaMallocManaged` or device allocations on
  Spark’s shared memory model) for persistent weights.

### Build

- Add `Makefile.linux` (or conditional blocks in `Makefile`):
  - Host: `gcc` or `clang`
  - Device: `nvcc` for `h3_gpu.cu`, `h3_kernels.cu`
  - Link: `-lcudart -lm -pthread -licuuc` (+ optional `-lcublasLt`)
- Keep `make test` structure; rename / add `test_cuda.c` parity targets using
  existing `misc/fixtures/` golden tensors.

## Implementation phases

### Phase 0 — Scaffold (≈1 week)

- [ ] Create `spark` branch workflow and Linux Makefile
- [ ] `h3_gpu.cu`: context, streams, tensor alloc/free, BF16 copy
- [ ] `h3_tokenizer.c`: load tokenizer JSON, encode/decode smoke test
- [ ] Replace vImage path in `h3_host.c` for Linux
- [ ] `./h3 --info -d ./MiniMax-H3` prints CUDA device, no generation

### Phase 1 — Numerical parity (≈3–4 weeks)

- [ ] BF16 linear, RMSNorm, AdaLN, gate, SwiGLU
- [ ] SDPA / GQA for DiT block shape
- [ ] Pass adapted `test_metal.c` / `test_bf16.c` fixtures on Linux
- [ ] Single 512×512 DiT block forward vs MLX golden

### Phase 2 — Performance path (≈4–6 weeks)

- [ ] int8 QKV, MLP, attention-output (default M5-equivalent fast path)
- [ ] Fusion kernels: gate+AdaLN, QKV+RoPE, token pool/expand
- [ ] Weight zero-copy / SSD streaming on NVMe
- [ ] `--profile` phase timings comparable to Metal `--profile`

### Phase 3 — End-to-end (≈3–4 weeks)

- [ ] Qwen text + vision encoder on CUDA
- [ ] Video / audio VAE encode + decode
- [ ] Full prompt-to-video/audio CLI and interactive session
- [ ] Reference presets from README validated on Spark (512² fox, etc.)

**Estimated calendar time:** 3–4 months for one experienced GPU engineer,
assuming MiniMax-H3 weights and fixtures are available on the Spark machine.

## Validation strategy

Reuse the existing test philosophy:

1. **Deterministic host tests** — `make test` C suite without GPU.
2. **Kernel parity** — compare CUDA outputs to `misc/fixtures/` MLX golden
   tensors (same relative/error bounds as Metal tests).
3. **Semantic tests** — real-weight block / schedule / VAE tests already in
   `tests/test_real_*.c`; port runners to CUDA backend.
4. **End-to-end quality** — visual review + optional SSIM against a reference
   render; do not require byte-identical output vs MLX (already documented for
   Metal).

Environment variables like `H3_DISABLE_FUSED_*` should gain CUDA equivalents so
each fusion can be A/B tested against an unfused oracle, matching Mac
diagnostics.

## Risks

| Risk | Mitigation |
|------|------------|
| GB10 sm_12.1 toolchain immaturity | Keep cuBLASLt oracle; pin CUDA 13.x; test early on real Spark hardware |
| Lower memory bandwidth vs Mac | Prioritize int8 and fusion; measure `--ssd-streaming` early |
| Large CUDA rewrite (~9k lines GPU code) | Strict `h3_gpu.h` boundary; vertical slices; fixture-driven parity |
| Thermal throttling on 140 W | Same as Mac: report wall time, warm repeated runs, `--profile` |
| Two backends to maintain (Metal + CUDA) | Shared C core; fusion semantics documented; identical CLI flags |

## Optional: TileLang as a spike only

[TileLang](https://github.com/tile-ai/tilelang) supports Linux AArch64 and
SM120, but it is **not** part of the shipping dependency model. It may be used
offline to prototype a BF16 GEMM or flash-attention tile shape, then reimplement
the winning schedule as a static CUDA kernel in `h3_kernels.cu`.

## Success criteria

The Spark port is complete when:

1. `./h3` is a **standalone Linux AArch64 binary** with the dependencies listed
   in [Dependency budget](#dependency-budget) only.
2. Prompt-to-video/audio works at 512×512 with README default presets.
3. `--profile` shows DiT denoise time within a reasonable factor of M5 Max on
   equivalent presets (exact ratio TBD after Phase 2 benchmarking).
4. `make test` passes kernel parity and real-weight tests on Spark CI or manual
   sign-off.
5. No Python interpreter or PyTorch is required at runtime.

## References

- [NVIDIA DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark porting guide (UMA)](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)
- h3-metal README — presets, memory modes, and fusion flags
- `h3_gpu.h` — backend interface to implement on CUDA
- `THIRD_PARTY_NOTICES.md` — Morton / int8 matmul design lineage

---

*Branch: `spark` · Status: planning · Last updated: 2026-08-12*
