# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark (`spark` branch).

## Plan (revised 2026-08-13)

| Phase | Goal | INT8? | Status |
|-------|------|-------|--------|
| **0** | Scaffold, host tests, CUDA probe | no | ✅ Done |
| **1** | DiT **BF16** block parity + text + vision encoder | no | ✅ Done |
| **2** | **Metal-aligned runtime INT8** on GB10 (WMMA/MMA) | yes | ⏳ Not started |
| **3** | Full pipeline + end-to-end generate | inherits Phase 2 | ⏳ Not started |

**Weights:** official `MiniMaxAI/MiniMax-H3` → `FL2VA/*` **BF16** shards only.
Do not use Comfy-Org repack or pre-quantized int8/fp8/nvfp4 files.

GB10 `sm_121` already sets `h3_gpu_has_int8_mlp()=true`; int8 kernels are stubs until Phase 2.

## Gate commands

```bash
make -f Makefile.linux test          # h3_tests + cuda smoke/ops + dit/text/vision/tokenizer
make -f Makefile.linux test-step      # 50-block DiT step + 50-layer Qwen + dit forward
make -f Makefile.linux probe          # CUDA device probe (GB10 sm_121)
make -f Makefile.linux h3             # full CLI binary
H3_MODEL_ROOT=/path/to/MiniMax-H3 make -f Makefile.linux test
```

## Phase 1 — BF16 DiT + text + vision ✅

### CUDA BF16 ops (Phase 1 scope)

Core DiT block: silu, RMSNorm, linear (cuBLAS), AdaLN, gate, SwiGLU, MLP, GELU,
QKV+RoPE, SDPA, token pool/expand (+ fused AdaLN variants),
patch_linear, gate_adaln, adaln_linear, sub, euler.

Text encoder: head_rms_norm, rope_text (F32 tables), gqa_causal, silu_mul.

Vision encoder: layer_norm_bf16, vision_qkv_rope_bf16.

### Smoke tests (all in `make test` when weights present)

- [x] `tests/test_cuda_ops.c` — CPU oracle for implemented ops
- [x] `tests/test_cuda_dit_block_smoke.c` — block-0 (+ 50-block in `test-step`)
- [x] `tests/test_cuda_text_smoke.c` — Qwen layer-0 (+ 50-layer in `test-step`)
- [x] `tests/test_cuda_vision_smoke.c` — Qwen vision 27-layer tower (64×64)
- [x] `tests/test_cuda_dit_forward_smoke.c` — production `h3_dit_forward` (25/50 in `test-step`)
- [x] Tokenizer smoke

Phase 1 BF16 production paths use env flags / slower-BF16 toggles to avoid INT8 stubs.
Remaining **25 GPU stubs** are Phase 2 INT8, Phase 3 VAE/audio F32, or Metal-only (`mlp_nax_bf16`).

## Phase 3 prep (out of Phase 1 scope)

F32 helpers already landed on `spark` for later VAE/audio work: scale_add, add_scaled,
clip, layer_norm, swiglu, geglu. Not required for Phase 1 gate.

## Commits (autoloop)

```
b0dbf59 F32 layer_norm, swiglu, geglu (Phase 3 prep)
ac4d1ca BF16 layer_norm, text_qk_rope, vision_qkv_rope
d7b6b1e DiT forward smoke (25/50 blocks)
5dcff89 patch_linear, gate_adaln, adaln_linear
```

---

*Last updated: 2026-08-15*
