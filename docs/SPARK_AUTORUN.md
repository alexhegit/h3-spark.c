# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark (`spark` branch).

## Plan (revised 2026-08-13)

| Phase | Goal | INT8? |
|-------|------|-------|
| **0** | Scaffold, host tests, CUDA probe | no |
| **1** | DiT **BF16** block parity + text encoder ops | no |
| **2** | **Metal-aligned runtime INT8** on GB10 (WMMA/MMA) | yes — default fast path |
| **3** | Full pipeline + end-to-end generate | inherits Phase 2 |

**Weights:** official `MiniMaxAI/MiniMax-H3` → `FL2VA/*` **BF16** shards only.
Do not use Comfy-Org repack or pre-quantized int8/fp8/nvfp4 files.

**Not in plan:** INT4; MVFP8/FP8 as primary path (optional benchmark after INT8 parity).

GB10 `sm_121` already sets `h3_gpu_has_int8_mlp()=true`; int8 kernels are stubs until Phase 2.

## Gate commands

```bash
make -f Makefile.linux test          # h3_tests + cuda smoke/ops + dit block smoke
make -f Makefile.linux test-step      # 50-block full DiT step (~71s, needs weights)
make -f Makefile.linux probe          # CUDA device probe (GB10 sm_121)
make -f Makefile.linux h3             # full CLI binary
H3_TOKENIZER_JSON=path/to/tokenizer.json make -f Makefile.linux test
make -f Makefile.linux tokenizer_test # full vocab parity (needs MiniMax-H3 tokenizer)
```

## Phase 0 (complete)

- [x] `Makefile.linux`, `h3_cuda.c`, `h3_device.c`, `h3_gpu.cu` scaffold
- [x] Tensor alloc, copy/cast/add BF16, weight pread
- [x] `h3_gpu_stubs.c` + `scripts/gen_gpu_stubs.py`
- [x] Linux host portability fixes
- [x] `tests/test_cuda_smoke.c`, `./h3_tests` 1768 checks

## Phase 1 (in progress) — BF16 only

### CUDA BF16 ops implemented

| API | Status |
|-----|--------|
| `h3_gpu_silu_bf16` | done |
| `h3_gpu_rms_norm_bf16` | done |
| `h3_gpu_linear_bf16` | done (cuBLAS `GemmEx`) |
| `h3_gpu_adaln_bf16` / `_offset` | done |
| `h3_gpu_gate_bf16` | done |
| `h3_gpu_swiglu_bf16` | done |
| `h3_gpu_mlp_bf16` | done (linear→swiglu→linear) |
| `h3_gpu_gelu_bf16` | done |
| `h3_gpu_embedding_bf16` | done |
| `h3_gpu_qkv_rope_bf16` / grouped variants | done |
| `h3_gpu_sdpa_bf16` / `_head_major_output` | done |
| `h3_gpu_linear_f32` / `silu_f32` | done |
| `h3_gpu_head_rms_norm_bf16` | done (text encoder) |
| `h3_gpu_rope_text_bf16` | done (F32 rope tables) |
| `h3_gpu_gqa_causal_bf16` | done (causal GQA) |
| `h3_gpu_silu_mul_bf16` | done (SwiGLU fuse) |

- [x] `tests/test_cuda_ops.c` — numerical checks vs CPU oracle
- [x] `tests/test_cuda_dit_block_smoke.c` — block-0 + optional 50-block step (no MLX fixture)
- [x] `tests/test_cuda_text_smoke.c` — Qwen layer-0 + optional 50-layer (no MLX fixture)
- [x] Stubs remaining: **39**
- [ ] patch_linear, conv/VAE paths
- [ ] `test_real_dit_block` (needs MLX golden fixture — optional)

### Tokenizer

- [x] Full `h3_tokenizer.c` BPE port (cJSON + ICU NFC)
- [ ] Full `tests/test_tokenizer.c` — needs `FL2VA/tokenizer/tokenizer.json` symlink

## Phase 2 (not started) — INT8 like Metal

- [ ] `quantize_weight_int8`, dynamic activation quant kernels
- [ ] `linear_int8_bf16`, `mlp_int8_bf16`, `grouped_qkv_linear_rope_int8`
- [ ] `gate_adaln_quantize_int8`, head-major SDPA→int8 path
- [ ] Fusion kernels + `bench_dit.c` A/B vs BF16 oracle

## Commits (Phase 0–1 autoloop)

```
2773758 GELU + embedding
815c580 Tokenizer BPE port
05f8991 SwiGLU + decomposed MLP
0b98314 AdaLN + gate
0985ea4 linear_bf16 cuBLAS
d009804 SiLU + RMSNorm
36d5f0b Phase 0 scaffold
```

---

*Last updated: 2026-08-13*
