# Spark performance baseline (v0.1.x)

Recorded on **NVIDIA DGX Spark (GB10)** for optimization work. Do not treat
these as Metal-parity targets yet — they are the **pre-optimization CUDA**
baseline after functional completeness (`v0.1.0` + CUDA `--profile`).

**Run date:** 2026-08-17  
**Binary / tree:** `h3-spark.c` `main` @ `483ffdf` (CUDA `h3_gpu_profile_mark`)  
**Log:** `/tmp/h3_perf/fox-fast-512-profile.log`  
**Output:** `/tmp/h3_perf/fox-fast-512-profile.mp4`

## Workload (fox-fast)

Same validated balanced preset as the README showcase T2VA clip:

```bash
./h3 --profile \
  -d /path/to/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  -o outputs/fox-fast.mp4
```

- Spatial: **512×512**
- Temporal: **22 frames** @ 24 fps (~0.92 s clip)
- Denoise: **20 steps**, **45 / 50** DiT layers, **reuse 2** → **11** DiT evaluations
- Weights: official MiniMax-H3 BF16 (`FL2VA`), runtime INT8 MLP path as defaulted by the port

## End-to-end wall clock

| Metric | Value |
|--------|------:|
| Total wall (`/usr/bin/time`) | **1725.65 s** (~28.8 min) |
| Max RSS | ~2.26 GiB (`MAX_RSS_KB 2370012`) |

## CUDA `--profile` phase breakdown

Lines are emitted as `h3 profile: <label> <phase> wall=…` after
`cudaStreamSynchronize`, so **wall includes GPU work** for that mark interval.

| Label | Phase | Wall (s) | Share of total | Notes |
|-------|-------|---------:|---------------:|-------|
| Qwen text encoder | total | 14.803 | 0.9% | Full text encode context |
| H3 DiT | load | 53.043 | 3.1% | Weight / AdaLN prep before denoise |
| **H3 DiT** | **GPU Euler denoise** | **1471.700** | **85.3%** | **Primary hotspot** (11 evals) |
| H3 DiT | total | 1524.966 | 88.4% | load + denoise (+ teardown) |
| audio VAE decoder | total | 1.259 | 0.1% | Negligible |
| video VAE decoder | total | 183.605 | 10.6% | Second hotspot (2×2 tiles @ 288) |

Sanity check: profiled phases ≈ text + DiT total + audio VAE + video VAE  
≈ 14.8 + 1525.0 + 1.3 + 183.6 ≈ **1724.7 s**, matches wall within ~1 s.

### DiT denoise counters (from profile line)

```
H3 DiT  GPU Euler denoise
  wall=1471.700s
  peak=17.053 GiB
  alloc≈105.6 GiB (delta accounting)
  submissions=11
  direct=7606
  attention=495
```

`linear` / `conv` CUDA stats buckets are still mostly unused (Metal-oriented
counter names); use `direct` + `attention` + wall until counters are remapped.

## Comparison vs upstream Metal (antirez/h3.c docs)

Upstream numbers are **M5 Max**, same **512-square / 22-frame / 45 layers +
reuse 2** denoise profile (README performance sections). They report **denoise
segment** time, not always full process wall.

| Metric | Metal M5 Max (docs) | Spark CUDA (this baseline) | Approx. ratio |
|--------|--------------------:|---------------------------:|--------------:|
| Denoise (`45 + reuse 2`) | **16.69 s** | **1471.7 s** | **~88×** |
| Same + token reduction | **12.60 s** | (not used in this run) | **~117×** vs 12.60 |
| Full e2e wall | (not directly tabulated for fox-fast) | **1725.7 s** | — |

**Conclusion:** With denoise-to-denoise口径对齐, Spark CUDA is still on the
order of **~10²×** slower than Metal on this preset. The gap is **not** an
artifact of comparing full wall to denoise-only: DiT denoise alone is already
~1472 s vs ~13–17 s.

Secondary gap: video VAE decode **183.6 s** is large vs Metal’s overall
fox-fast experience (tens of seconds e2e on M5), but still **~8× smaller**
than DiT denoise — optimize DiT first.

## Optimization priority (derived from this baseline)

1. **P0 — DiT / GPU Euler denoise (~85%)**  
   SDPA / attention, GEMM+MLP (BF16 + INT8), fusion, occupancy, fewer host syncs.
2. **P1 — Video VAE decode (~11%)**  
   Tile path, conv/attention in VAE, residency.
3. **P2 — DiT load (~3%)**  
   Weight I/O / prep; only after P0/P1.
4. **Ignore for speed** — text encoder, audio VAE (sub-second to ~15 s).

## How to re-measure

```bash
make -f Makefile.linux -j$(nproc) h3
mkdir -p /tmp/h3_perf
/usr/bin/time -f 'WALL_SEC %e\nMAX_RSS_KB %M' ./h3 --profile \
  -d "$H3_MODEL_ROOT" \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 \
  -o /tmp/h3_perf/fox-fast-512-profile.mp4 \
  2>&1 | tee /tmp/h3_perf/fox-fast-512-profile.log

# Extract comparable denoise line:
rg "GPU Euler denoise|WALL_SEC" /tmp/h3_perf/fox-fast-512-profile.log
```

Prefer **warm** second runs when comparing micro-optimizations (filesystem
cache + GPU clocks). Keep this fox-fast preset as the **default A/B gate**
unless intentionally changing resolution/steps.

## Related

- Progress / backlog: [`SPARK_AUTORUN.md`](SPARK_AUTORUN.md) (item **D**)
- Known stubs: [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)
- Showcase clip: [`../assets/showcase/t2va-fox-fast.mp4`](../assets/showcase/t2va-fox-fast.mp4)
- Upstream Metal timings: [antirez/h3.c](https://github.com/antirez/h3.c) README

---

---

## 2026-08-17 — Parallel SDPA (autoloop, `perf/dit-denoise-opt`)

**Change:** Replace thread-0 serial SDPA score loop with block-parallel SDPA
(`h3_sdpa_bf16_parallel_kernel`). Keep naive path behind `H3_SDPA_NAIVE=1`.
Also make `h3_gpu_continue` non-blocking (Metal semantics).

**Commits:** `abf1ef8` (first parallel), `92a190b` (fewer syncthreads on scores).

### Fox-fast A/B (same preset as baseline)

| Metric | Baseline (naive SDPA) | Parallel SDPA (`abf1ef8` binary) | Speedup |
|--------|----------------------:|---------------------------------:|--------:|
| **GPU Euler denoise** | **1471.7 s** | **912.3 s** | **1.61×** |
| Video VAE decode | 183.6 s | 178.2 s | ~1.03× |
| DiT load | 53.0 s | 58.5 s | — |
| Text encode | 14.8 s | 14.8 s | — |
| **E2E wall** | **1725.7 s** | **1166.1 s** | **1.48×** |

Log: `/tmp/h3_perf/fox-fast-512-sdpa-parallel.log`

### Proxy A/B (512² / 22f / steps4 / layers35 / reuse1)

| Backend | Denoise wall | E2E wall |
|---------|-------------:|---------:|
| Parallel (default) | 255.2 s | 501.5 s |
| `H3_SDPA_NAIVE=1` | 409.4 s | 655.2 s |

Denoise proxy speedup **1.60×**, consistent with full fox-fast.

### Remaining hotspot share (after parallel SDPA)

At fox-fast: denoise still **~78%** of wall (912/1166); video VAE **~15%**.
Next: GEMM/Flash-style SDPA, INT8 GEMM reliability, VAE decode.

### Follow-up: SDPA v2 kernel (`92a190b`)

Full fox-fast remeasure with the fewer-syncthreads score/softmax kernel:

| Metric | Parallel v1 (`abf1ef8`) | Parallel v2 (`92a190b`) |
|--------|------------------------:|------------------------:|
| GPU Euler denoise | **912.3 s** | 940.5 s |
| E2E wall | **1166.1 s** | 1193.9 s |

v2 was **not** faster on this workload; keep **v1-style** scheduling as the
reference win. Log: `/tmp/h3_perf/fox-fast-512-sdpa-v2.log`.

### Autoloop close-out (2026-08-18)

11h deadline reached. Branch `perf/dit-denoise-opt` holds:

- Parallel BF16 DiT SDPA (+ continue non-blocking)
- Parallel F32 / causal SDPA (VAE paths)
- Documented ~**1.61×** denoise / **1.48×** e2e vs naive baseline

Merged to `main` as `f64b726`. Work continued on `perf/dit-denoise-opt`.

---

## 2026-08-22 — Wave / Q2 SDPA (HIP port)

**KEEP** default warp online-softmax SDPA (d128 Q2, F32 d64 Q2). Opt-out:
`H3_SDPA_PARALLEL=1`, `H3_SDPA_D128_Q1=1`, `H3_SDPA_D64_Q1=1`.
`--profile` prints exclusive `gpu-op linear/sdpa/conv`.

Fox-s2 (`512² / 22f / steps=2 / L35 / reuse=1`):

| Metric | Parallel SDPA | Wave + Q2 | Speedup |
|--------|--------------:|----------:|--------:|
| GPU Euler denoise wall | 131.1 s | **99.9 s** | 1.31× |
| denoise gpu-op **sdpa** | 33.9 s | **2.86 s** | **11.9×** |
| denoise gpu-op **linear** | 94.1 s | 94.1 s | — |
| video VAE wall | 44.1 s | **19.2 s** | 2.29× |
| VAE gpu-op sdpa | 27.7 s | **4.02 s** | **6.9×** |
| E2E wall | 247.0 s | **189.2 s** | 1.31× |

Logs: `/tmp/h3_perf_day5/fox-s2-wave.log`. After this, denoise is ~94% linear
(INT8 MLP/QKV). Next: tiled grouped INT8 FC2.

---

## 2026-08-23 — Tiled grouped INT8 FC2

**KEEP** `h3_linear_int8_grouped_tiled_kernel` (64×64, K=32) as default MLP
FC2. Opt-out `H3_INT8_GROUP_NAIVE=1`. Persistent cuBLAS INT8 accum buffer
(no per-call `cudaMalloc`). Profile: `int8-cublas=210 naive=0` on fox-s2.

Same fox-s2 preset, wave SDPA on both sides:

| Metric | Naive grouped FC2 | Tiled FC2 | Speedup |
|--------|------------------:|----------:|--------:|
| GPU Euler denoise wall | 98.9 s | **18.0 s** | **5.5×** |
| denoise gpu-op linear | 94.3 s | **13.4 s** | **7.0×** |
| denoise gpu-op sdpa | 2.84 s | 2.85 s | — |
| E2E wall | 190.3 s | **112.3 s** | 1.69× |

vs original CUDA naive-SDPA baseline denoise 1472 s / e2e 1726 s, fox-s2 is a
different step count; on this 2-step gate, denoise is now **18 s**. Remaining
denoise split: linear 13.4 · sdpa 2.9.

### Fox-fast remeasure (`55a2710`, 2026-08-23 02:54 CST)

Same 512² / 22f / steps20 / L45 / reuse2 preset as the original baseline.
Log: `/tmp/h3_perf_day6/fox-fast-wave-int8tile.log`.

| Metric | Naive SDPA baseline | Parallel SDPA | **Wave + tiled FC2** | vs naive |
|--------|--------------------:|--------------:|---------------------:|---------:|
| GPU Euler denoise | 1471.7 s | 912.3 s | **127.3 s** | **11.6×** |
| denoise gpu-op linear | — | — | 94.8 s | — |
| denoise gpu-op sdpa | — | — | 20.5 s | — |
| DiT load | 53.0 s | 58.5 s | 65.8 s | — |
| video VAE | 183.6 s | 178.2 s | **20.6 s** | **8.9×** |
| **E2E wall** | **1725.7 s** | **1166.1 s** | **232.0 s** | **7.4×** |

Remaining fox-fast denoise: **linear ~75%** (94.8 / 127). Metal M5 docs denoise
~16.7 s → still ~**7.6×**. Next: FC1/QKV INT8 GEMM, then load I/O.

---

## 2026-08-23 — Spark defaults to BF16 MLP

GB10 BF16 tensor cores beat the Metal-aligned runtime INT8 MLP. fox-s2
(`512² / 22f / steps=2 / L35 / reuse=1`):

| Path | Denoise | gpu-op linear | Load | E2E |
|------|--------:|--------------:|-----:|----:|
| Tiled FC2 + cuBLAS INT8 FC1 (`55a2710`) | 18.0 s | 13.4 s | ~52 s | 112 s |
| Same + tiled FC1 (`groups=1`) | 69.3 s | 19.9 s† | 51.5 s | 158 s |
| cuBLASLt INT8 (vs GemmEx) | 18.1 s | 14.0 s | 53 s | 110 s |
| INT8 MLP + dp4a FC2 (`H3_INT8_MLP=1`) | 7.24 s | 3.22 s | 42.9 s | 82 s |
| **BF16 MLP default** (INT8 QKV/out kept) | **4.87 s** | **1.10 s** | **35.0 s** | **71.6 s** |

† FC1 was not in `gpu-op linear` until this pass, so tiled-FC1 denoise (69 s)
is the real signal. **REJECT** default tiled FC1 (`H3_INT8_TILED=1` opt-in).
**REJECT** cuBLASLt (no faster than `cublasGemmEx`). **REJECT**
`H3_DISABLE_INT8_QKV=1` (DiT QKV linear request failed).

**KEEP**

- Default **BF16 MLP** on Spark. Restore INT8 MLP with `H3_INT8_MLP=1`.
- `__dp4a` inner loop on grouped INT8 FC2 (opt-in INT8 path: linear 13.4 s → 3.2 s).
- Compile `h3_gpu.cu` for `sm_121`.
- Pinned host staging for weight `pread`; one GPU submit per block after INT8 quantize.
- Count MLP FC1 / QKV INT8 GEMM in `gpu-op linear`.

### Fox-fast remeasure (BF16 MLP default, 2026-08-23 08:53 CST)

Log: `/tmp/h3_perf_day7/fox-fast-bf16mlp.log`.

| Metric | Wave + tiled INT8 FC2 | **BF16 MLP default** | vs prior |
|--------|----------------------:|---------------------:|---------:|
| GPU Euler denoise | 127.3 s | **35.3 s** | **3.6×** |
| denoise gpu-op linear | 94.8 s | **7.84 s** | **12.1×** |
| denoise gpu-op sdpa | 20.5 s | 22.0 s | — |
| DiT load | 65.8 s | **18.3 s** | **3.6×** |
| video VAE | 20.6 s | **17.1 s** | — |
| **E2E wall** | **232.0 s** | **83.8 s** | **2.8×** |

vs naive CUDA baseline denoise 1472 s / e2e 1726 s: **41.7× / 20.6×**.
Metal M5 docs denoise ~16.7 s → Spark now ~**2.1×**. Remaining fox-fast
denoise is **SDPA ~62%** (22 / 35). Next: SDPA, then load I/O on a cold cache.

---

## 2026-08-23 — Wave SDPA d128 Q4

**KEEP** default d128 Q4 (four queries share K/V). Opt-out `H3_SDPA_D128_Q2=1`.
VAE still uses d64 Q2. Logs: `/tmp/h3_perf_day7/fox-s2-sdpa-q4.log`,
`/tmp/h3_perf_day7/fox-fast-sdpa-q4.log`.

fox-s2 DiT:

| Metric | Q2 | **Q4 default** |
|--------|---:|---------------:|
| GPU Euler denoise | 4.96 s | **4.42 s** |
| denoise gpu-op sdpa | 3.08 s | **2.55 s** |

fox-fast (load is cache-noisy):

| Metric | BF16 MLP + Q2 | **+ Q4** |
|--------|--------------:|---------:|
| GPU Euler denoise | 35.3 s | **31.6 s** |
| denoise gpu-op sdpa | 22.0 s | **18.3 s** |
| denoise gpu-op linear | 7.84 s | 7.82 s |
| **E2E wall** | 83.8 s (warm load 18 s) | 100 s (colder load 36 s) |

Denoise SDPA **1.20×**. Remaining fox-fast denoise is still mostly SDPA
(18.3 / 31.6). Next: d64 VAE SDPA, then load I/O.

---

## 2026-08-23 — VAE d64 Q4 + overlapped weight copies

**KEEP** F32 d64 wave Q4 as default VAE SDPA. Opt-out `H3_SDPA_D64_Q2=1`.
fox-s2 `video VAE` `gpu-op sdpa` **4.11 s → 3.54 s** (~1.16×). VAE wall is
still ~17 s because the phase includes loading 36 VAE blocks.

**KEEP** double-buffered pinned `pread` + `cudaMemcpyAsync` for weight
staging. fox-s2 DiT load **38.4 s (sync) → 35.6 s (overlap)** on a warm-ish
cache. Opt-out `H3_LOAD_SYNC_COPY=1`.

**REJECT** always-on safetensor fd cache (`H3_LOAD_FD_CACHE=1` opt-in): DiT
load did not drop vs open/close per tensor (pread of large tensors dominates).

Logs: `/tmp/h3_perf_day8/`.

fox-fast (`82f2936`, colder load): denoise **31.4 s** (sdpa 18.4), VAE **16.6 s**
(sdpa 3.57 vs ~4.1 on Q2), DiT load **42.8 s**, e2e **107 s**. Compute moved;
e2e still tracks load temperature.

---

## 2026-08-23 — Prefetch video VAE weights during DiT denoise

**KEEP** a background `read()` of `FL2VA/video_vae/source/*.safetensors` (~9.7 GiB)
started after DiT load and joined before video VAE load. Opt-out
`H3_VAE_NO_PREFETCH=1`.

`readahead(2)` / `POSIX_FADV_WILLNEED` return immediately on this kernel and
do not populate ~10 GiB, so the thread uses a real 8 MiB `read` loop.

Cold VAE file (`posix_fadvise DONTNEED` between runs). GPU compute unchanged.

fox-s2 (steps 2 / L35 / reuse 1):

| Metric | No prefetch | **Prefetch** |
|--------|------------:|-------------:|
| video VAE decoder wall | 17.0 s | **8.9 s** |
| prefetch thread | — | 6.8 s (partially overlaps 4.5 s denoise) |
| denoise gpu-op sdpa / linear | 2.56 / 1.11 | 2.57 / 1.11 |

fox-fast (steps 20 / L45 / reuse 2):

| Metric | No prefetch | **Prefetch** |
|--------|------------:|-------------:|
| video VAE decoder wall | 16.7 s | **9.0 s** |
| prefetch thread | — | 5.2 s (fully hidden in 31.8 s denoise) |
| GPU Euler denoise | 31.7 s | 31.8 s |
| denoise gpu-op sdpa | 18.4 s | 18.4 s |

VAE wall **~1.85–1.91×**. Remaining fox-fast VAE (~9 s) is mostly GPU
(linear 3.7 + sdpa 3.6). Remaining fox-fast denoise is still mostly SDPA
(18.4 / 31.8). Logs: `/tmp/h3_perf_day8/fox-s2-prefetch-read.log`,
`fox-s2-noprefetch-read.log`, `fox-fast-prefetch.log`, `fox-fast-noprefetch.log`.

---

## 2026-08-23 — DiT d128 wave SDPA Q8

**KEEP** d128 wave SDPA Q8 as default (eight queries share each K/V load).
Opt-out `H3_SDPA_D128_Q4=1` (previous Q4), `H3_SDPA_D128_Q2=1`, or
`H3_SDPA_D128_Q1=1`. `readahead`-style GEMM SDPA is still rejected.

`h3_cuda_ops` and DiT block smoke passed.

fox-s2 (steps 2 / L35 / reuse 1):

| Metric | Q4 | **Q8** |
|--------|---:|-------:|
| GPU Euler denoise | 4.47 s | **3.82 s** |
| denoise gpu-op sdpa | 2.57 s | **1.91 s** |
| denoise gpu-op linear | 1.14 s | 1.13 s |

fox-fast (steps 20 / L45 / reuse 2):

| Metric | Q4 | **Q8** |
|--------|---:|-------:|
| GPU Euler denoise | 31.7 s | **27.3 s** |
| denoise gpu-op sdpa | 18.4 s | **14.0 s** |
| denoise gpu-op linear | 7.87 s | 7.92 s |
| video VAE decoder | 9.2 s | 9.1 s |

Denoise SDPA **~1.31–1.34×**. Remaining fox-fast denoise is SDPA 14.0 + linear
7.9 of 27.3 s. Logs: `/tmp/h3_perf_day8/fox-s2-q4.log`, `fox-s2-q8.log`,
`fox-fast-sdpa-q8.log`, `fox-fast-sdpa-q4.log`.

---

## 2026-08-23 — REJECT d128 wave SDPA Q16

Tried sharing K/V across 16 queries (`H3_SDPA_D128_Q16=1`). fox-s2 denoise
gpu-op sdpa **1.93 s → 5.01 s** (denoise 3.86 s → 6.94 s). Register spill
dominates; Q8 stays default. Kernel not landed. Log:
`/tmp/h3_perf_day8/fox-s2-q8.log`, `fox-s2-q16.log`.

---

## 2026-08-23 — DiT Q8 SDPA K-unroll 2

**KEEP** Q8 wave SDPA consuming two K/V positions per inner iteration (loads
overlapped before the online-softmax update). Partial Q tiles still walk K
one-by-one.

fox-s2 (steps 2 / L35 / reuse 1):

| Metric | Q8 | **Q8 K2** |
|--------|---:|----------:|
| GPU Euler denoise | 3.86 s | **3.67 s** |
| denoise gpu-op sdpa | 1.93 s | **1.74 s** |
| denoise gpu-op linear | 1.17 s | 1.15 s |

fox-fast (steps 20 / L45 / reuse 2):

| Metric | Q8 | **Q8 K2** |
|--------|---:|----------:|
| GPU Euler denoise | 27.3 s | **26.9 s** |
| denoise gpu-op sdpa | 14.0 s | **13.0 s** |
| denoise gpu-op linear | 7.92 s | 8.01 s |

Denoise SDPA **~1.07–1.10×**. Remaining fox-fast denoise is SDPA 13.0 + linear
8.0 of 26.9 s. Logs: `/tmp/h3_perf_day8/fox-s2-q8k2.log`,
`fox-fast-q8k2.log`.

---

## 2026-08-23 — REJECT cuBLAS tensor-op math / GEMM algo for BF16 linear

Tried `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)`: F32 linear tests failed
(TF32 vs reference at ~1e-4). Reverted.

Tried `CUBLAS_GEMM_DEFAULT_TENSOR_OP` on BF16 DiT linear
(`H3_CUBLAS_TENSOR_ALGO=1`). fox-s2 denoise gpu-op linear **1.177 s → 1.156 s**
(noise). Default `CUBLAS_GEMM_DEFAULT` stays. Logs:
`/tmp/h3_perf_day8/fox-s2-linear-default.log`, `fox-s2-linear-tensor.log`.

---

## 2026-08-23 — REJECT Q8 SDPA K-unroll 4

Prefetching four K/V steps before softmax did not beat K2 outside noise
(fox-s2 denoise sdpa **1.70 s** vs K2 **1.74 s**; later K2 reruns were 1.78 s).
Keep K2. Log: `/tmp/h3_perf_day8/fox-s2-q8k4.log`.

---

## 2026-08-23 — REJECT CUBLAS_COMPUTE_32F_FAST_16BF for DiT linear

Ops still passed. fox-s2 denoise gpu-op linear **1.139 s → 1.186 s** (slightly
slower). Stay on `CUBLAS_COMPUTE_32F`. Logs:
`/tmp/h3_perf_day8/fox-s2-fast16bf-off.log`, `fox-s2-fast16bf-on.log`.

---

## 2026-08-23 — DiT Q8 SDPA software-pipelined K/V load

**KEEP** overlapping the next K/V load with the current online-softmax
update, instead of loading two positions then consuming both. Full Q8
tiles only.

fox-s2 (steps 2 / L35 / reuse 1), contemporaneous A/B (two warm runs):

| Metric | Q8 K2 | **Q8 pipe** |
|--------|------:|------------:|
| GPU Euler denoise | 3.74 / 3.82 s | **3.64 / 3.66 s** |
| denoise gpu-op sdpa | 1.77 / 1.82 s | **1.69 / 1.70 s** |
| denoise gpu-op linear | 1.17 / 1.19 s | 1.16 / 1.15 s |

fox-fast (steps 20 / L45 / reuse 2):

| Metric | Q8 K2 | **Q8 pipe** |
|--------|------:|------------:|
| GPU Euler denoise | 26.9 s | **25.2 s** |
| denoise gpu-op sdpa | 13.0 s | **11.9 s** |
| denoise gpu-op linear | 8.01 s | 7.92 s |

Denoise SDPA **~1.05–1.10×** vs K2. Remaining fox-fast denoise is SDPA 11.9 +
linear 7.9 of 25.2 s (~1.51× vs Metal ~16.7 s). Logs:
`/tmp/h3_perf_day8/fox-s2-k2-ab1.log`, `fox-s2-pipe-ab1.log`,
`fox-fast-q8k2.log`, `fox-fast-q8pipe.log`.

---

## 2026-08-23 — REJECT Q8 SDPA launch_bounds(32, 8)

Forcing eight blocks per SM compiled, but fox-s2 denoise sdpa stayed inside
noise (**1.63 / 1.68 s** vs pipelined K2 **1.69 / 1.70 s**). fox-fast denoise
**25.2 s → 25.0 s**, sdpa **11.86 s → 11.70 s**. Keep default
`__launch_bounds__(32)`. Logs: `/tmp/h3_perf_day8/fox-s2-q8lb8.log`,
`fox-s2-q8lb8-b.log`, `fox-fast-q8lb8.log`.

---

## 2026-08-24 — DiT persistent MLP/QKV workspace

**KEEP** caching BF16 temps used by fused MLP, INT8 QKV, INT8 MLP FC1, and
gate-AdaLN quantize. Those paths called `cudaMalloc`/`cudaFree` every DiT
block (stream sync). Opt out with `H3_DISABLE_GPU_WORKSPACE=1`.

fox-s2 (steps 2 / L35 / reuse 1), contemporaneous A/B (two warm runs):

| Metric | per-block alloc | **workspace** |
|--------|----------------:|--------------:|
| GPU Euler denoise | 3.51 / 3.62 s | **3.11 / 3.14 s** |
| denoise gpu-op sdpa | 1.65 / 1.68 s | 1.71 / 1.72 s |
| denoise gpu-op linear | 1.14 / 1.15 s | 1.12 / 1.14 s |
| denoise alloc | 17.0 GiB | **0.24 GiB** |

fox-fast (steps 20 / L45 / reuse 2):

| Metric | Q8 pipe | **workspace** |
|--------|--------:|--------------:|
| GPU Euler denoise | 25.2 s | **22.0 s** |
| denoise gpu-op sdpa | 11.9 s | 12.2 s |
| denoise gpu-op linear | 7.92 s | 7.87 s |

Kernel gpu-op times are unchanged; denoise wall dropped because malloc no
longer stalls the stream. Remaining fox-fast denoise is SDPA 12.2 + linear
7.9 of 22.0 s (~1.32× vs Metal ~16.7 s). Logs:
`/tmp/h3_perf_day8/fox-s2-ws-off-ab1.log`, `fox-s2-ws-on-ab1.log`,
`fox-fast-q8pipe.log`, `fox-fast-ws.log`.

---

## 2026-08-24 — REJECT 2-warp Q16 SDPA (shared K/V smem)

Two warps per block, 16 queries, K/V staged through shared memory. fox-s2
denoise sdpa **1.67 / 1.74 s → 3.17 / 3.21 s** (denoise 3.06 / 3.18 s →
4.54 / 4.62 s). Barrier + split K/V load is slower than pipelined Q8.
Reverted. Logs: `/tmp/h3_perf_day8/fox-s2-q8-ab1.log`, `fox-s2-q8x2-ab1.log`.

---

## 2026-08-24 — REJECT Q8 paired warp-reduce

Interleaving two query `__shfl_xor` reductions per K/V step did not beat
serial `h3_warp_reduce_sum`. fox-s2 denoise sdpa **1.85 / 1.87 s** (serial)
vs **1.95 / 1.85 s** (paired); denoise wall **3.27 / 3.33 s** vs
**3.52 / 3.29 s**. Reverted. Logs: `/tmp/h3_perf_day8/fox-s2-serial-ab1.log`,
`fox-s2-pair-ab1.log`.

---

## 2026-08-24 — REJECT persistent cuBLAS workspace

`cublasSetWorkspace` with 128 MiB did not drop fox-s2 denoise (off
**3.12 / 3.13 s**, linear **1.14 / 1.15 s**; on **3.26 / 3.15 s**, linear
**1.22 / 1.15 s**). Reverted. Logs:
`/tmp/h3_perf_day8/fox-s2-cublasws-off-ab1.log`, `fox-s2-cublasws-on-ab1.log`.

---

## 2026-08-24 — REJECT register-cached fused gate+AdaLN

Keeping gated activations in registers and using a warp-shuffle block
reduce did not beat the shared-memory cache. fox-s2 denoise **3.14 / 3.20 s**
(smem) vs **3.23 / 3.16 s** (registers). Reverted. Logs:
`/tmp/h3_perf_day8/fox-s2-adaln-smem-ab1.log`, `fox-s2-adaln-reg-ab1.log`.

---

## 2026-08-24 — REJECT vec4 SwiGLU

Four-wide SwiGLU with `__expf`/`__frcp_rn` did not beat scalar `expf`.
fox-s2 denoise **3.15 / 3.11 s** (scalar) vs **3.13 / 3.12 s** (vec4).
Reverted. Logs: `/tmp/h3_perf_day8/fox-s2-swiglu-scalar-ab1.log`,
`fox-s2-swiglu-vec4-ab1.log`.

---

## 2026-08-24 — REJECT Q8 K/V streaming loads (ld.global.cs)

`ld.global.cs` for pipelined K/V did not beat `__ldg`. fox-s2 denoise
**3.31 / 3.27 s** (`__ldg`) vs **3.31 / 3.31 s** (cs); sdpa **1.86 / 1.85 s**
vs **1.88 / 1.86 s**. The dual-path kernel also inflated both vs the
single-`__ldg` baseline (~1.70 s sdpa). Reverted. Logs:
`/tmp/h3_perf_day8/fox-s2-q8-ldg-ab1.log`, `fox-s2-q8-ldcs-ab1.log`.

---

## 2026-08-24 — REJECT 2D INT8 scale epilogue

A 2D grid (row × column) to avoid `index / output_dim` did not drop fox-s2
denoise: **3.18 / 3.14 s** (1D) vs **3.16 / 3.19 s** (2D); linear
**1.19 / 1.15 s** vs **1.16 / 1.17 s**. Reverted. Logs:
`/tmp/h3_perf_day8/fox-s2-iscale-1d-ab1.log`, `fox-s2-iscale-2d-ab1.log`.

---

## 2026-08-24 — REJECT Q8 SDPA PreferL1 cache config

`cudaFuncCachePreferL1` plus zero shared-memory carveout did not drop
fox-s2 denoise: **3.08 / 3.13 s** (default cache) vs **3.11 / 3.29 s**
(PreferL1); sdpa **1.67 / 1.70 s** vs **1.69 / 1.76 s**. Reverted. Logs:
`/tmp/h3_perf_day8/fox-s2-q8-cache-def-ab1.log`, `fox-s2-q8-cache-l1-ab1.log`.

---

## 2026-08-24 — REJECT warp-shuffle INT8 row quantize

Warp `__shfl_xor` max plus packed BF16 loads did not beat the shared-memory
tree. fox-s2 denoise **3.19 / 3.15 s** (smem) vs **3.09 / 3.23 s** (warp);
linear **1.16 / 1.14 s** vs **1.13 / 1.21 s**. Reverted. Logs:
`/tmp/h3_perf_day8/fox-s2-quant-smem-ab1.log`, `fox-s2-quant-warp-ab1.log`.

---

## 2026-08-24 — KEEP cooperative QKV/RoPE RMS

Each QKV/RoPE thread used to square **all** `head_dim` values (O(d²) loads).
Default is now one load per thread plus warp/block reduce. Opt out with
`H3_QKV_ROPE_SERIAL_RMS=1` (or `--use-slower-scalar-qkv-rms` on the INT8 path).

fox-s2 (steps 2 / L35 / reuse 1), contemporaneous A/B (two warm runs):

| Metric | serial RMS | **coop RMS** |
|--------|-----------:|-------------:|
| GPU Euler denoise | 3.19 / 3.14 s | **3.01 / 3.06 s** |
| denoise leftover (wall − sdpa − linear) | 0.29 / 0.29 s | **0.18 / 0.18 s** |
| denoise gpu-op sdpa | 1.73 / 1.71 s | 1.69 / 1.70 s |
| denoise gpu-op linear | 1.18 / 1.15 s | 1.14 / 1.18 s |

fox-fast (steps 20 / L45 / reuse 2) vs workspace KEEP 22.0 s:

| Metric | workspace | **+ coop RMS** |
|--------|----------:|---------------:|
| GPU Euler denoise | 22.0 s | **21.1 s** |
| denoise leftover | 1.96 s | **1.19 s** |
| denoise gpu-op sdpa | 12.2 s | 12.1 s |
| denoise gpu-op linear | 7.87 s | 7.78 s |

Logs: `/tmp/h3_perf_day8/fox-s2-qkvrms-serial-ab1.log`, `fox-s2-qkvrms-coop-ab1.log`,
`fox-fast-qkvrms.log`.

---

## 2026-08-24 — REJECT vec4 INT8 scale epilogue

Vector `int4`/`float4` loads on the 1D INT8 scale kernel did not drop fox-s2
denoise: **3.05 / 3.07 s** (scalar) vs **3.00 / 3.07 s** (vec4); linear
**1.16 / 1.18 s** vs **1.12 / 1.17 s**. Reverted. Logs:
`/tmp/h3_perf_day8/fox-s2-iscale-1d-ab1.log`, `fox-s2-iscale-vec4-ab1.log`.

---

## 2026-08-24 — REJECT Q8 SDPA interleaved warp-reduce

A separate Q8 kernel that shuffles all 8 query scores each mask step did not
drop fox-s2 denoise vs serial `h3_warp_reduce_sum`. Interleaved warm A/B:
**3.11 / 3.02 s** (serial) vs **3.00 / 3.11 s** (ILP); sdpa **1.75 / 1.69 s**
vs **1.69 / 1.73 s**. Reverted. Logs: `/tmp/h3_perf_day8/fox-s2-q8-serial-ab3.log`,
`fox-s2-q8-ilp-ab3.log`.

---

## 2026-08-24 — REJECT scalar SwiGLU __expf/__frcp_rn

Isolating fast math on the scalar SwiGLU kernel (not vec4) did not drop fox-s2
denoise. Interleaved A/B: **3.04 / 3.06 s** (`expf`) vs **3.07 / 3.07 s**
(`__expf`+`__frcp_rn`). Reverted. Logs:
`/tmp/h3_perf_day8/fox-s2-swiglu-expf-ab1.log`, `fox-s2-swiglu-fast-ab1.log`.

---

## 2026-08-24 — REJECT Q8 SDPA depth-2 K/V pipeline

Prefetching K/V two steps ahead (triple-buffer, separate kernel) did not drop
fox-s2 denoise vs the depth-1 software pipeline. Interleaved A/B:
**3.08 / 3.07 s** (pipe1) vs **3.04 / 3.28 s** (pipe3); sdpa **1.71 / 1.74 s**
vs **1.72 / 1.86 s**. Reverted. Logs: `/tmp/h3_perf_day9/fox-s2-q8-pipe1-ab1.log`,
`fox-s2-q8-pipe3-ab1.log`.

