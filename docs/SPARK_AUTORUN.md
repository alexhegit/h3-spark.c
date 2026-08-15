# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark (`spark` branch).

## Plan (revised 2026-08-13)

| Phase | Goal | INT8? | Status |
|-------|------|-------|--------|
| **0** | Scaffold, host tests, CUDA probe | no | ✅ Done |
| **1** | DiT **BF16** block parity + text + vision encoder | no | ✅ Done |
| **2** | **Metal-aligned runtime INT8** on GB10 | yes | ✅ Done |
| **3** | Full pipeline + end-to-end generate | inherits Phase 2 | ⏳ Not started |

**Weights:** official `MiniMaxAI/MiniMax-H3` → `FL2VA/*` **BF16** shards only.
Do not use Comfy-Org repack or pre-quantized int8/fp8/nvfp4 files.

## Gate commands

```bash
make -f Makefile.linux test
make -f Makefile.linux test-step   # includes BF16 forward + INT8-MLP forward
H3_MODEL_ROOT=/path/to/MiniMax-H3 make -f Makefile.linux test
./h3_cuda_dit_forward_smoke $H3_MODEL_ROOT int8
```

## Phase 1 — BF16 DiT + text + vision ✅

(unchanged; see prior commits)

## Phase 2 — Metal-aligned INT8 ✅

Implemented (correctness-first CUDA; cuBLAS INT8 GEMM when available):

- [x] `h3_gpu_quantize_weight_int8` — per-output-channel symmetric quant
- [x] `h3_gpu_linear_int8_bf16` — dynamic act quant + INT8 GEMM + scales
- [x] `h3_gpu_linear_int8_head_major_bf16` — SDPA head-major gather-quant
- [x] `h3_gpu_mlp_int8_bf16` — FC1/SwiGLU/FC2 (+ optional grouped FC2)
- [x] `h3_gpu_gate_adaln_quantize_int8`
- [x] `h3_gpu_grouped_qkv_linear_rope_int8`
- [x] CPU oracles in `tests/test_cuda_ops.c`
- [x] INT8 DiT forward smoke (`int8` / `int8-full`; MLP INT8, QKV/attn BF16 on tiny layout)
- [ ] Stretch: larger-layout smoke with QKV + attention INT8 (`sequence >= 128`)

Tiny forward smoke (`TEXT=6`, 2×2×2 latent) only exercises INT8 MLP because
`h3_dit` requires `sequence >= 128` for QKV/attention INT8 load.

Stubs remaining: **19** (Phase 3 VAE/audio F32 + Metal-only `mlp_nax_bf16`).

## Commits (Phase 2)

```
b6f9e26 INT8-MLP DiT forward smoke + test-step gate
6656f8a gate_adaln_quantize_int8 + INT8 QKV/RoPE
9b39dcf head-major linear + mlp_int8_bf16
03a9d22 quantize_weight_int8 + linear_int8_bf16
97ec6d5 Phase 1 vision smoke (complete)
```

---

*Last updated: 2026-08-15*
