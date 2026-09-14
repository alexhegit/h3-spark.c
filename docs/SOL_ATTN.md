# `--sol-attn` — sparse SDPA for long T2VA

Opt-in, **off by default**. Training-free block-sparse attention inspired by
[Sol-Attn](https://arxiv.org/abs/2607.24027) (NVIDIA), implemented on this
port’s MMA SDPA kernel. It is **not** the Sol-H3-Spark 4-step + LTX pipeline.

The default generate path is unchanged. `H3_SOL_ATTN_TAU=-100` (keep every
KV tile) is bit-identical to dense MMA.

## When to use it

| Workload | Use `--sol-attn`? |
|---|---|
| HIP-page / showcase md5, publish | **No** — quality path only |
| fox-fast / 512² ~22 frames | **No** — SDPA is not the wall; `--reuse 3` is better |
| 15 s / long T2VA (seq tens of thousands) | **Yes**, if you can accept ~19 dB vs the quality path |

Do not stack with `--token-reduction` unless you are measuring that combo
on purpose. Do not treat this as KEEP (fox-fast PSNR ≥ 24 dB / SSIM ≥ 0.85).

## How to run

```bash
MODEL=/path/to/MiniMax-H3

./h3 --profile -d "$MODEL" -p "$PROMPT_15S" \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  --sol-attn \
  -o out-15s-sol-attn.mp4
```

Interactive CLI: `!sol-attn on`. Same effect as the flag: sets `H3_SOL_ATTN=1`
for the process.

Kernel-only microbench (no DiT layer mask):

```bash
make -f Makefile.linux h3_sdpa_bench
./h3_sdpa_bench 44800 56 128 3
H3_SOL_ATTN=1 H3_SOL_ATTN_TAU=0.5 ./h3_sdpa_bench 44800 56 128 3
```

Keep-all correctness (ops test + fox-fast md5 `f5282774d3a4`):

```bash
H3_SOL_ATTN=1 H3_SOL_ATTN_TAU=-100 ./h3_sdpa_bench 1874 56 128 5
# or: make -f Makefile.linux h3_cuda_ops   # prints "sol-attn keep-all bitwise diffs 0"
```

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `H3_SOL_ATTN` | unset (`--sol-attn` sets `1`) | Master switch for the sparse MMA kernel |
| `H3_SOL_ATTN_TAU` | `0.5` | Keep KV tiles with proxy score ≥ mean + τ·std. Lower τ keeps more tiles (slower, closer to dense). `-100` keeps all |
| `H3_SOL_ATTN_BAND` | `1` | Always keep \|kv_block − q_block\| ≤ band |
| `H3_SOL_ATTN_PREFIX` | from DiT: `(video_target_start+63)/64` | Always keep this many leading KV blocks (text / cond / audio). Env override if the GPU helper is not configured |
| `H3_SOL_ATTN_BLOCKS` | `4:40` | DiT residual blocks that may run sparse SDPA; others stay dense |
| `H3_SOL_ATTN_DROP` | unset | If `1`, skipped tiles contribute nothing (worse quality; A/B only) |

Sequence must be ≥ 512 tokens or the kernel stays on dense MMA.

## Measured GB10 (2026-09-14)

PSNR/SSIM are ffmpeg lavfi vs this port’s **quality path** (same prompt, size,
seed, `--layers 45 --reuse 2`, no speed flag). Not a camera metric.

### Kernel (`h3_sdpa_bench`, 56 heads, dim 128)

| seq | dense | τ=0.5 | ratio |
|---:|---:|---:|---:|
| 1874 (fox-fast) | 2.84 ms | 1.63–2.38 ms | ~1.2–1.7× |
| 8192 | 44.4 ms | 20.3 ms | **2.2×** |
| 16384 | 164.4 ms | 75.5 ms | **2.2×** |
| 44800 (15 s cinematic) | 1682–1693 ms | **548–572 ms** | **3.0×** |

### fox-fast (512², 22f, steps 20, L45 R2, seed 42)

vs md5 `f5282774d3a4`. Sparse τ=0.5: denoise 8.17→7.74 s, sdpa 1.40→1.05 s,
**PSNR 17.6 / SSIM 0.71**. Fails KEEP. Keep-all τ=-100: md5 unchanged, SSIM 1.0.

### 15 s cinematic (864×480, L45 R2, seed 42)

vs quality-path md5 `60fd70cc309c` (e2e **1076 s**, denoise 988 s, sdpa 845 s).

| Flag | E2E | Denoise (sdpa / linear) | vs quality | PSNR / SSIM | md5 prefix |
|---|---:|---|---:|---|---|
| (none) | 1076 s | 988 s (845 / 110) | — | — | `60fd70cc309c` |
| `--sol-attn` | **694.7 s** (11 min 35 s) | **608.9 s** (464 / 111) | **−35%** | **19.2 / 0.72** | `6ad88ffb989a` |
| `--reuse 3` | 805 s (13 min 25 s) | 719 s (614 / 80) | −25% | 18.8 / 0.70 | `d9c4482a551a` |
| `--token-reduction` | 674 s (11 min 14 s) | 589 s (482 / 81) | −37% | 17.8 / 0.66 | `19c109ebb0cb` |

On this clip Sol-Attn is **faster than reuse 3 and higher-PSNR than both
reuse 3 and TR**. It is ~20 s slower than TR. Kernel 3× does not become e2e
3×: only DiT blocks 4–40 are sparse, prefix KV is exact, linear and VAE do
not shrink. Log: `/tmp/h3_perf4h/sol-attn-15s.log`.

## What the code does

1. **CLI / params** — `h3_params.sol_attn` (`h3.h`). `--sol-attn` in `main.c`,
   `!sol-attn` in `h3_cli.c`. `h3_generate` `setenv("H3_SOL_ATTN","1")` and
   prints that output is not bit-identical.
2. **DiT** — `h3_dit.c` `run_block` calls `h3_gpu_sol_attn_configure` so sparse
   SDPA runs only on `H3_SOL_ATTN_BLOCKS` (default 4:40). Prefix block count
   is `(video_target_start + 63) / 64` so text/cond/audio KV stay exact.
3. **GPU** — `h3_gpu.cu`:
   - `h3_sdpa_kv_block_summary_kernel` — per-head K-mean and V-sum for each
     64-token tile
   - `h3_sdpa_bf16_mma_d128_sol_kernel` — default MMA path (ldmatrix, online
     softmax) that skips unselected tiles and mixes pooled K/V into the
     running softmax
   - `h3_gpu_sdpa_bf16_mma` dispatches the sol kernel when `H3_SOL_ATTN` is on,
     `gpu->sol_attn_layer` is set, and `sequence >= 512`
4. **Tests** — `tests/test_cuda_ops.c` compares τ=-100 to dense MMA (bitwise
   on seq 512). `tools/h3_sdpa_bench.c` times the kernel.

This is a **keep-or-approximate** tile router, not a reimplementation of the
Sol-Attn paper’s fused one-pass CUDA. Quality is therefore a measured
trade-off on this port, not a claim of paper-table PSNR.

Recipes: [`BEST_PRACTICE.md`](BEST_PRACTICE.md). Dated benches:
[`PERF_BASELINE.md`](PERF_BASELINE.md) (2026-09-14).
