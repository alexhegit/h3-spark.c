# Known issues (CUDA / DGX Spark)

Tracked gaps that do **not** block the default `./h3` T2VA generate path on
GB10 (`sm_121`). Production DiT uses BF16 (+ optional INT8 MLP from Phase 2).

## KI-001: Unimplemented DiT F32 GPU ops

**Status:** deferred (won't-fix for Spark unless a caller forces F32 DiT)

**Stubs remaining in `h3_gpu_stubs.c`:**

| API | Notes |
|-----|--------|
| `h3_gpu_adaln_f32` | F32 AdaLN; BF16 `h3_gpu_adaln_bf16` is used instead |
| `h3_gpu_gate_f32` | F32 residual gate; BF16 `h3_gpu_gate_bf16` is used instead |
| `h3_gpu_qkv_rope_f32` | F32 QKV+RoPE; BF16 / INT8 paths are used instead |

**Why deferred on Spark GB10**

- Official MiniMax-H3 FL2VA DiT weights and the CUDA port's generate path run
  **BF16** (with optional Metal-aligned **INT8** MLP).
- FP32 DiT compute is not a realistic Spark production mode; porting these
  three kernels would mainly complete the `h3_gpu.h` surface for parity with
  Metal's unused-or-rare F32 DiT helpers.

**Impact**

- Default CLI / `h3_generate()`: **none**.
- Any code that explicitly calls the F32 DiT APIs above will fail with
  `… is not implemented on CUDA yet`.

**Resolution options (when needed)**

1. Port from Metal shaders / existing BF16 CUDA kernels (`adaln_bf16`,
   `gate_bf16`, `qkv_rope_bf16`) with F32 tensors.
2. Or document that CUDA builds intentionally omit F32 DiT and keep the stubs.

## KI-002: Metal-only `h3_gpu_mlp_nax_bf16`

**Status:** intentional stub on CUDA

Apple M5 NAX MLP fusion. `h3_gpu_has_nax_mlp()` returns false on CUDA; leave
unimplemented indefinitely unless a CUDA equivalent is designed.

## Related

- Progress log: [`docs/SPARK_AUTORUN.md`](SPARK_AUTORUN.md)
- Porting plan: [`docs/SPARK_PORTING.md`](SPARK_PORTING.md)
