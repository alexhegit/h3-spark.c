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

GB10 `sm_121` already sets `h3_gpu_has_int8_mlp()=true`; int8 kernels are stubs until Phase 2.

## Gate commands

```bash
make -f Makefile.linux test          # h3_tests + cuda smoke/ops + dit/text/tokenizer smoke
make -f Makefile.linux test-step      # 50-block DiT step + 50-layer Qwen + dit forward
make -f Makefile.linux probe          # CUDA device probe (GB10 sm_121)
make -f Makefile.linux h3             # full CLI binary
H3_MODEL_ROOT=/path/to/MiniMax-H3 make -f Makefile.linux test
```

## Phase 1 (in progress) — BF16 only

### CUDA BF16 ops implemented

Core DiT block: silu, RMSNorm, linear (cuBLAS), AdaLN, gate, SwiGLU, MLP, GELU,
QKV+RoPE, SDPA, F32 linear/SiLU, token pool/expand (+ fused AdaLN variants),
patch_linear (F32→BF16 tiled), gate_adaln, adaln_linear (final head fuse),
sub, euler.

Text encoder: head_rms_norm, rope_text (F32 tables), gqa_causal, silu_mul.

Audio/VAE prep (F32): scale_add, add_scaled, clip, layer_norm, swiglu, geglu.

Vision prep: layer_norm_bf16, text_qk_rope_bf16, vision_qkv_rope_bf16.

- [x] `tests/test_cuda_ops.c` — CPU oracle for all above
- [x] `tests/test_cuda_dit_block_smoke.c` — block-0 + 50-block step
- [x] `tests/test_cuda_text_smoke.c` — Qwen layer-0 + optional 50-layer
- [x] `tests/test_cuda_dit_forward_smoke.c` — production `h3_dit_forward` (25/50 blocks)
- [x] Tokenizer smoke in `make test`
- [x] Stubs remaining: **28**
- [ ] Phase 2 INT8 stack

### Tokenizer

- [x] Full `h3_tokenizer.c` BPE port
- [x] Tokenizer smoke in `make test`

## Commits (autoloop)

```
d7b6b1e DiT forward smoke (25/50 blocks)
5dcff89 patch_linear, gate_adaln, adaln_linear
cfc365e Token pool/expand + sub/euler
de4e1c3 DiT attention + Qwen text ops + weight smokes
```

---

*Last updated: 2026-08-14*
