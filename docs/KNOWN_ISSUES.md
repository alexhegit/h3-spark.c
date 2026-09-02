# Known issues (CUDA / DGX Spark)

Tracked gaps that do **not** block the default `./h3` T2VA generate path on
GB10 (`sm_121`). Production DiT uses BF16 (+ optional INT8 MLP from Phase 2).

## KI-001: Unimplemented DiT F32 GPU ops

**Status:** **won't-fix** on Spark (product decision 2026-08-17)

Spark edge devices ship BF16 / INT8 (and may add FP8 later). Porting unused F32
DiT helpers is skipped.

**Stubs remaining in `h3_gpu_stubs.c`:**

| API | Notes |
|-----|--------|
| `h3_gpu_adaln_f32` | F32 AdaLN; BF16 `h3_gpu_adaln_bf16` is used instead |
| `h3_gpu_gate_f32` | F32 residual gate; BF16 `h3_gpu_gate_bf16` is used instead |
| `h3_gpu_qkv_rope_f32` | F32 QKV+RoPE; BF16 / INT8 paths are used instead |

**Impact**

- Default CLI / `h3_generate()`: **none**.
- Explicit F32 DiT API callers fail with `… is not implemented on CUDA yet`.

## KI-002: Metal-only `h3_gpu_mlp_nax_bf16`

**Status:** **ignore** / intentional stub on CUDA

Apple M5 NAX MLP fusion. `h3_gpu_has_nax_mlp()` returns false on CUDA; leave
unimplemented. CUDA performance work should use its own fused MLP / INT8 path
(see [`SPARK_AUTORUN.md`](SPARK_AUTORUN.md)).

## Related backlog

| ID | Topic | Status |
|----|-------|--------|
| D | Perf + CUDA `--profile` | **v0.2.0 snapshot** — fox-fast ~15.5 s e2e / ~8.2 s denoise. Remaining: fox-fast INT8 linear; 15 s long-N SDPA. No wider dense-BF16 MMA on `sm_121`. [`PERF_BASELINE.md`](PERF_BASELINE.md) |
| E | MLX fixture numerical parity | Pending (no Mac / fixtures) |

Progress log: [`docs/SPARK_AUTORUN.md`](SPARK_AUTORUN.md) · Porting:
[`docs/SPARK_PORTING.md`](SPARK_PORTING.md)
