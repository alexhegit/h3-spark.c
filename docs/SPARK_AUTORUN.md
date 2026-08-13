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
make -f Makefile.linux test          # h3_tests + cuda smoke/ops + dit/text smoke
make -f Makefile.linux test-step      # 50-block DiT step + 50-layer Qwen (~minutes)
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

- [x] `tests/test_cuda_ops.c` — CPU oracle for all above
- [x] `tests/test_cuda_dit_block_smoke.c` — block-0 + 50-block step
- [x] `tests/test_cuda_text_smoke.c` — Qwen layer-0 + optional 50-layer
- [x] Stubs remaining: **34**
- [ ] `text_qk_rope_bf16` (refiner), `layer_norm_bf16` (vision)
- [ ] Full `h3_dit` forward smoke via production API
- [ ] Phase 2 INT8 stack

### Tokenizer

- [x] Full `h3_tokenizer.c` BPE port
- [ ] Tokenizer smoke in `make test` — symlink `MiniMax-H3/tokenizer/tokenizer.json`

## Commits (autoloop)

```
cfc365e Token pool/expand + sub/euler
de4e1c3 DiT attention + Qwen text ops + weight smokes
2773758 GELU + embedding
```

---

*Last updated: 2026-08-13*
