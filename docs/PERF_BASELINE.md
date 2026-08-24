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

---

## 2026-08-24 — KEEP fused gate AdaLN + INT8 row quantize

Replace the two-kernel wrapper (`h3_gpu_gate_adaln_bf16` then
`h3_gpu_quantize_bf16_int8_rows`) with one kernel that keeps gated/AdaLN BF16
in smem and writes INT8+scales. Skips the BF16 AdaLN HBM temp. Default on;
opt-out `H3_SPLIT_ADALN_QUANT=1`.

`h3_cuda_ops` and DiT block smoke passed with fused and split paths.

fox-s2 (steps 2 / L35 / reuse 1), interleaved:

| Metric | Two kernels | **Fused** |
|--------|------------:|----------:|
| GPU Euler denoise | 3.051 / 3.046 s | **3.022 / 3.026 s** |
| denoise gpu-op sdpa | 1.728 / 1.699 s | 1.705 / 1.717 s |
| denoise gpu-op linear | 1.147 / 1.170 s | 1.152 / 1.145 s |
| direct launches | 862 | **796** |
| denoise alloc | 0.244 GiB | **0.225 GiB** |

fox-fast (steps 20 / L45 / reuse 2), interleaved:

| Metric | Two kernels | **Fused** |
|--------|------------:|----------:|
| GPU Euler denoise | 21.967 / 21.974 s | **21.911 / 21.867 s** |
| denoise gpu-op sdpa | 12.848 / 12.878 s | 12.837 / 12.858 s |
| denoise gpu-op linear | 7.896 / 7.898 s | 7.916 / 7.867 s |
| direct launches | 6121 | **5670** |
| denoise alloc | 0.247 GiB | **0.228 GiB** |

Denoise wall dropped on both gates. Logs: `/tmp/h3_perf_day9/fox-s2-fused-ab1.log`,
`fox-s2-two-ab1.log`, `fox-fast-fused-ab1.log`, `fox-fast-two-ab1.log`.

---

## 2026-08-24 — REJECT fused INT8 QKV scale+RoPE

Folding INT8 GEMM int32 accum + scales into the cooperative QKV/RoPE kernel
skipped the BF16 QKV temp (alloc 0.225 → 0.150 GiB, 796 → 726 launches) but did
not drop fox-s2 denoise wall. Interleaved A/B:

| Run | Split | **Fused** |
|-----|------:|----------:|
| ab1 | 3.011 s | 3.017 s |
| ab2 | 3.184 s | 2.933 s |
| ab3 | 3.005 s | 3.129 s |

ab2 split looks like SDPA noise (1.803 s vs ~1.71 s). ab1/ab3 fused is not
faster. Reverted. Logs: `/tmp/h3_perf_day9/fox-s2-qkvrope-fused-ab1.log`,
`fox-s2-qkvrope-split-ab1.log`.

---

## 2026-08-24 — REJECT two-wide SwiGLU ILP

A separate kernel doing two columns per thread with the same `expf` math
(not vec4, not `__expf`) did not drop fox-s2 denoise. Interleaved A/B:

| Run | Scalar | **X2** |
|-----|-------:|-------:|
| ab1 | 3.012 s | 3.089 s |
| ab2 | 3.160 s | 3.103 s |

ab2 scalar SDPA looks noisy (1.785 s vs ~1.70 s). ab1 x2 is slower. Reverted.
Logs: `/tmp/h3_perf_day9/fox-s2-swiglu-x2-ab1.log`, `fox-s2-swiglu-scalar-ab1.log`.

---

## 2026-08-24 — REJECT warp AdaLN reduce

Separate kernels using warp-shuffle + 8-wide smem reduce (instead of a 256-wide
tree) did not drop leftover AdaLN. Interleaved fox-s2 denoise:

| Run | Tree | **Warp** |
|-----|-----:|---------:|
| ab1 | 3.146 s | 3.035 s |
| ab2 | 3.110 s | 3.086 s |

Leftover denoise − sdpa − linear stayed **0.167 / 0.166 s** on both sides. ab1
wall tracks SDPA noise (1.766 s vs 1.688 s). Reverted. Logs:
`/tmp/h3_perf_day9/fox-s2-adaln-warp-ab1.log`, `fox-s2-adaln-tree-ab1.log`.

---

## 2026-08-24 — REJECT Q8 SDPA L2 prefetch

A separate Q8 kernel issuing `prefetch.global.L2` on K/V two/three steps ahead
(depth-1 register pipe unchanged) slowed fox-s2 denoise. Interleaved A/B:

| Run | Default Q8 | **L2 prefetch** |
|-----|-----------:|----------------:|
| ab1 | 3.005 s (sdpa 1.704) | 3.192 s (sdpa 1.851) |
| ab2 | 3.034 s (sdpa 1.711) | 3.152 s (sdpa 1.822) |

Not the rejected depth-2 register pipe; still a net loss. Reverted. Logs:
`/tmp/h3_perf_day9/fox-s2-q8-l2pre-ab1.log`, `fox-s2-q8-base-ab1.log`.
---

## 2026-08-24 — REJECT fused INT8 attn-out scales into MLP AdaLN

Leaving INT8 attention-out GEMM in int32 accum and folding
`accum * input_scale * weight_scale` into the BF16 MLP AdaLN kernel skipped
70 apply-scale launches (796 → 726) but did not drop fox-s2 denoise wall.
Interleaved A/B (`H3_INT8_ATTN_ADALN=1`):

| Run | Default | **Fused accum AdaLN** |
|-----|--------:|----------------------:|
| ab1 | 3.145 s (sdpa 1.780, linear 1.199, leftover 0.166) | 3.059 s (sdpa 1.730, linear 1.153, leftover 0.176) |
| ab2 | 2.960 s (sdpa 1.685, linear 1.111, leftover 0.164) | 3.025 s (sdpa 1.720, linear 1.132, leftover 0.173) |

Leftover rose ~10 ms (int32 branch reads are heavier than BF16). Wall is a
tie inside SDPA noise. Same failure mode as fused INT8 QKV scale+RoPE.
Reverted. Logs: `/tmp/h3_perf_day9/fox-s2-attn-adaln-ab1.log`,
`fox-s2-attn-base-ab1.log`.
---

## 2026-08-24 — REJECT head-major QKV for wave SDPA

RoPE storing Q/K/V as `[heads, seq, 128]` so Q8 SDPA walks K/V with 256 B
stride instead of `heads*256` (14 KiB). Distinct from L2 prefetch.
fox-s2 interleaved (`H3_SDPA_HEAD_MAJOR_QKV=1`):

| Run | Token-major | **Head-major QKV** |
|-----|------------:|-------------------:|
| ab1 | 3.044 s (sdpa 1.706, linear 1.168) | 3.022 s (sdpa 1.684, linear 1.173) |
| ab2 | 3.025 s (sdpa 1.709, linear 1.148) | 3.025 s (sdpa 1.673, linear 1.185) |

SDPA ~25–35 ms faster both pairs; denoise wall tied (linear noise). Reverted.
Logs: `/tmp/h3_perf_day9/fox-s2-hm-ab1.log`, `fox-s2-hm-base-ab1.log`.
---

## 2026-08-24 — REJECT cuBLASLt for BF16 DiT linear

`cublasLtMatmul` + heuristic (64 MiB workspace, `H3_BF16_CUBLASLT=1`) did not
beat `cublasGemmEx` / WMMA. fox-s2 interleaved:

| Run | GemmEx | **cuBLASLt** |
|-----|-------:|-------------:|
| ab1 | 3.033 s (linear 1.155, sdpa 1.710) | 3.096 s (linear 1.185, sdpa 1.746) |
| ab2 | 3.049 s (linear 1.138, sdpa 1.744) | 3.049 s (linear 1.173, sdpa 1.707) |

Linear is slower, not just SDPA noise. Distinct from INT8 cuBLASLt REJECT.
Reverted. Logs: `/tmp/h3_perf_day9/fox-s2-lt-ab1.log`, `fox-s2-lt-base-ab1.log`.
---

## 2026-08-24 — REJECT Q10 wave SDPA

Separate d128 Q10 kernel (158 regs, no local spill; occupancy below Q8's 127
regs). fox-s2 interleaved `H3_SDPA_D128_Q10=1`:

| Run | Q8 | **Q10** |
|-----|---:|--------:|
| ab1 | 3.115 s (sdpa 1.759) | 3.385 s (sdpa 2.024) |
| ab2 | 3.203 s (sdpa 1.777) | 3.436 s (sdpa 2.050) |

Slower than Q8; not Q16 spill. Reverted. Logs:
`/tmp/h3_perf_day9/fox-s2-q10-ab1.log`, `fox-s2-q8-ab1.log`.
---

## 2026-08-24 — REJECT vec2 cooperative QKV/RoPE

Separate kernel: two dims/thread, packed `uint32` BF16 loads/stores, same coop
RMS. fox-s2 interleaved `H3_QKV_ROPE_VEC2=1`:

| Run | Coop RMS | **Vec2** |
|-----|---------:|---------:|
| ab1 | 3.197 s (sdpa 1.795, leftover 0.170) | 3.019 s (sdpa 1.711, leftover 0.162) |
| ab2 | 3.089 s (sdpa 1.738, leftover 0.165) | 3.177 s (sdpa 1.772, leftover 0.164) |

Leftover denoise − sdpa − linear unchanged. ab1 wall tracks SDPA noise.
Reverted. Logs: `/tmp/h3_perf_day9/fox-s2-vec2-ab1.log`, `fox-s2-coop-ab1.log`.
---

## 2026-08-24 — REJECT Q8 SDPA exp2f softmax

Separate Q8 kernel using `exp2f((x)*log2(e))` instead of `__expf`. fox-s2
interleaved `H3_SDPA_Q8_EXP2=1`:

| Run | `__expf` | **exp2f** |
|-----|---------:|----------:|
| ab1 | 3.094 s (sdpa 1.754) | 3.194 s (sdpa 1.781) |
| ab2 | 3.076 s (sdpa 1.735) | 3.109 s (sdpa 1.752) |

Slower SDPA and denoise wall. Reverted. Logs:
`/tmp/h3_perf_day9/fox-s2-exp2-ab1.log`, `fox-s2-expf-ab1.log`.
---

## 2026-08-24 — REJECT SwiGLU 4-row packing

Separate kernel: four rows, same column, still scalar `expf`. Distinct from
two-wide column ILP and vec4. fox-s2 interleaved `H3_SWIGLU_X4ROW=1`:

| Run | One row | **x4row** |
|-----|--------:|----------:|
| ab1 | 3.150 s (sdpa 1.779, leftover 0.167) | 2.994 s (sdpa 1.687, leftover 0.168) |
| ab2 | 3.094 s (sdpa 1.741, leftover 0.165) | 3.010 s (sdpa 1.703, leftover 0.168) |

Leftover denoise − sdpa − linear unchanged. Wall tracks SDPA noise. Reverted.
Logs: `/tmp/h3_perf_day9/fox-s2-x4row-ab1.log`, `fox-s2-swiglu-ab1.log`.
---

## 2026-08-24 — REJECT smem-tiled INT8 scale epilogue

Separate kernel: one column-tile block, weight scales in smem, loop over rows.
Distinct from REJECT 2D grid and vec4 1D. fox-s2 interleaved
`H3_INT8_SCALE_SMEM=1`:

| Run | 1D scalar | **Smem tile** |
|-----|----------:|--------------:|
| ab1 | 3.066 s (linear 1.149, sdpa 1.752) | 3.042 s (linear 1.200, sdpa 1.678) |
| ab2 | 2.955 s (linear 1.106, sdpa 1.687) | 3.096 s (linear 1.221, sdpa 1.712) |

Linear is slower (row-strided accum). Wall does not drop. Reverted. Logs:
`/tmp/h3_perf_day9/fox-s2-smem-ab1.log`, `fox-s2-scale-ab1.log`.

## 2026-08-24 — REJECT AdaLN packed vec2 loads

Separate MLP AdaLN kernel: two consecutive columns per step, packed `uint32`
BF16 residual/branch/gate/weight loads. Distinct from warp AdaLN reduce and
register-cached fuse. fox-s2 interleaved `H3_ADALN_VEC2=1`:

| Run | Default | **Vec2 packed** |
|-----|--------:|----------------:|
| ab1 | 3.165 s (sdpa 1.788, linear 1.209, leftover 0.168) | 3.004 s (sdpa 1.702, linear 1.136, leftover 0.166) |
| ab2 | 3.032 s (sdpa 1.702, linear 1.167, leftover 0.163) | 2.998 s (sdpa 1.693, linear 1.144, leftover 0.161) |

Leftover denoise − sdpa − linear unchanged (~2 ms). ab1 wall tracks SDPA/linear
noise. Reverted. Logs: `/tmp/h3_perf_day9/fox-s2-adaln-vec2-ab1.log`,
`fox-s2-adaln-base-ab1.log`.
---

## 2026-08-24 — REJECT Q8 bf16x2 QK dots

Separate Q8 kernel: native `__nv_bfloat16` K/Q loads, `__hmul2` 4-term QK,
float `__expf` online softmax unchanged. Distinct from exp2f softmax REJECT.
fox-s2 interleaved `H3_SDPA_Q8_BF16X2=1`:

| Run | Float QK | **bf16x2 QK** |
|-----|---------:|--------------:|
| ab1 | 3.012 s (sdpa 1.693) | 3.207 s (sdpa 1.908) |
| ab2 | 3.071 s (sdpa 1.730) | 3.253 s (sdpa 1.944) |

SDPA ~210 ms slower; denoise wall follows. Reverted. Logs:
`/tmp/h3_perf_day9/fox-s2-q8-bf16x2-ab1.log`, `fox-s2-q8-float-ab1.log`.
---

## 2026-08-24 — REJECT Q8 pair-K consume

Separate Q8 kernel: step K by two, consume K[i] then prefetch K[i+2], consume
K[i+1] then prefetch K[i+3]. Same two live K/V slots as the depth-1 pipe;
not K-unroll 4 or depth-2 triple-buffer. fox-s2 interleaved
`H3_SDPA_Q8_PAIRK=1`:

| Run | Depth-1 pipe | **Pair-K** |
|-----|-------------:|-----------:|
| ab1 | 3.254 s (sdpa 1.818, linear 1.268) | 3.145 s (sdpa 1.763, linear 1.213) |
| ab2 | 3.099 s (sdpa 1.746, linear 1.187) | 3.039 s (sdpa 1.731, linear 1.143) |

SDPA delta is within noise; linear moved as much as SDPA so wall is not a
real denoise win. Reverted. Logs: `/tmp/h3_perf_day9/fox-s2-q8-pairk-ab1.log`,
`fox-s2-q8-pipe-ab1.log`.
---

## 2026-08-24 — REJECT Q6 wave SDPA

Separate d128 Q6 kernel (depth-1 K/V pipe, 6 queries/block). Occupancy
between Q4 and Q8; not Q10 register spill. fox-s2 interleaved
`H3_SDPA_D128_Q6=1`:

| Run | Q8 | **Q6** |
|-----|---:|-------:|
| ab1 | 3.038 s (sdpa 1.711) | 3.170 s (sdpa 1.815) |
| ab2 | 3.016 s (sdpa 1.712) | 3.235 s (sdpa 1.840) |

SDPA ~100–130 ms slower; denoise wall follows. Reverted. Logs:
`/tmp/h3_perf_day9/fox-s2-q6-ab1.log`, `fox-s2-q8-ab1.log`.
---

## 2026-08-24 — REJECT Q8 hardware warp reduce

Separate Q8 kernel replacing the shfl-xor sum with `__reduce_add_sync`.
nvcc 12.x / sm_121 only overloads that intrinsic for `int` / `unsigned`;
there is no `float` form, so the kernel does not compile. Not an A/B.
Reverted. Distinct from Q6 occupancy REJECT.
---

## 2026-08-24 — REJECT Q8 __ldcg K/V

Separate Q8 kernel: K/V via `__ldcg` (cache-global). Distinct from
`ld.global.cs` streaming and L2 prefetch. fox-s2 interleaved
`H3_SDPA_Q8_LDCG=1`:

| Run | `__ldg` | **`__ldcg`** |
|-----|--------:|-------------:|
| ab1 | 3.028 s (sdpa 1.719) | 3.038 s (sdpa 1.720) |
| ab2 | 3.051 s (sdpa 1.701) | 3.259 s (sdpa 1.848) |

No denoise wall drop; ab2 SDPA slower. Reverted. Logs:
`/tmp/h3_perf_day9/fox-s2-q8-ldcg-ab1.log`, `fox-s2-q8-ldg-ab1.log`.
---

## 2026-08-24 — REJECT Q8 lane-packed QKV

Separate RoPE store + Q8 kernel: Q/K/V as `[seq, heads, 32, 4]` so each
lane's 4 chunks are contiguous (`uint32` pairs), not 64 B apart. Distinct
from head-major (which kept dim 0..127 contiguous). fox-s2 interleaved
`H3_SDPA_LANE_PACK=1`. DiT block smoke passed.

| Run | Token-major Q8 | **Lane-pack** |
|-----|---------------:|--------------:|
| ab1 | 3.067 s (sdpa 1.729, linear 1.170) | 2.857 s (sdpa 1.555, linear 1.136) |
| ab2 | 2.982 s (sdpa 1.687, linear 1.128) | 3.082 s (sdpa 1.651, linear 1.259) |

SDPA is faster both pairs (~36–174 ms). Denoise wall is mixed (ab2 linear
noise +131 ms). Same KEEP failure as head-major. Reverted. Logs:
`/tmp/h3_perf_day9/fox-s2-lanepack-ab1.log`, `fox-s2-lanepack-base-ab1.log`.
---

## 2026-08-25 — GB10 GEMM roofline: INT8 is 1.6× BF16

Standalone `cublasGemmEx` bench on the DiT shapes (`/tmp/h3_perf_day10/gemm_bench.cu`),
`n` = sequence rows:

| Shape | BF16 | INT8 |
|-------|-----:|-----:|
| QKV `21504x5376`, n=6144 | 103.4 TFLOP/s | **166.9** |
| attn out `5376x7168`, n=6144 | 94.4 | **159.3** |
| FC1 `28672x5376`, n=6144 | 103.8 | — |
| FC2 `5376x14336`, n=6144 | 96.4 | — |
| square `8192³` | 97.9 | **150.4** |

Two conclusions that redirected the plan:

1. **cuBLAS BF16 is already saturated.** The DiT shapes hit 94–104 TFLOP/s and a
   compute-bound `8192³` square hits 97.9, so the shapes are not the problem and
   there is no tiling win to reclaim. A hand or CUTLASS BF16 GEMM would be
   competing with cuBLAS at its own ceiling — **dropped**, not attempted.
2. **INT8 runs at ~1.6× the BF16 rate** on exactly these shapes. That is where
   the linear time is.

## 2026-08-25 — REJECT cuBLASLt INT8 scale epilogue

Goal was to fold `accum * input_scale * weight_scale` into the GEMM and delete
the `h3_int8_apply_scales_bf16_kernel` launch, using
`CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` (per-`m` A scale × per-`n` B
scale). Not supported on `sm_121` / CUDA 13.0. Probe
(`/tmp/h3_perf_day10/lt_probe2.cu`), `heur` is `cublasLtMatmulAlgoGetHeuristic`:

| A/B | D | scales | heuristic |
|-----|---|--------|----------:|
| i8 | i32 | none | 0 (ok) |
| i8 | i8 | none | 0 (ok) |
| i8 | f32 / bf16 / f16 | none | **15 NOT_SUPPORTED** |
| i8 | i32 | alpha device vector | **7** |
| i8 | bf16 / f32 | outer vec | **15** |
| bf16 | bf16 | none | 0 (ok) |
| bf16 | bf16 | outer vec | **7** |

INT8 matmul on this GPU can only write `i32` or `i8`, and takes no per-vector
scale, so the epilogue has to stay a separate kernel. fox-s2 with the code in
place confirmed it: `ltfail=140`, every call fell back. Reverted.

---

## 2026-08-25 — KEEP INT8 MLP as the Spark default (row FC2)

**Supersedes the 2026-08-23 "Spark defaults to BF16 MLP" entry.** That
conclusion was an artifact of the *grouped* FC2 scale, not of INT8. A grouped
1024-wide scale cannot be expressed as one cuBLAS GEMM, so it fell back to the
`__dp4a` tile kernel; one row scale keeps FC2 on the cuBLAS INT8 GEMM.

fox-s2 (`512² / 22f / steps=2 / L35 / reuse=1`), same binary:

| MLP path | Denoise | gpu-op linear | Peak |
|----------|--------:|--------------:|-----:|
| INT8, grouped FC2 (`H3_INT8_GROUP_FC2=1`) | 5.16 s | 3.33 s | 13.3 GiB |
| BF16 (previous default) | 3.01 s | 1.14 s | 20.6 GiB |
| **INT8, row FC2 (new default)** | **2.87 s** | **0.81 s** | **13.3 GiB** |

Interleaved fox-s2, two pairs:

| Run | BF16 MLP | **INT8 row FC2** |
|-----|---------:|-----------------:|
| ab1 | 3.051 s (linear 1.167, sdpa 1.717) | 2.875 s (linear 0.810, sdpa 1.696) |
| ab2 | 3.069 s (linear 1.177, sdpa 1.727) | 2.822 s (linear 0.817, sdpa 1.652) |

Interleaved fox-fast (`steps=20 / L45 / reuse=2`, 11 evals), two pairs:

| Metric | BF16 MLP | **INT8 row FC2** | Delta |
|--------|---------:|-----------------:|------:|
| GPU Euler denoise | 20.95 / 21.11 s | **19.66 / 19.84 s** | **−1.28 s** |
| denoise gpu-op linear | 7.76 / 7.85 s | **5.35 / 5.39 s** | **−2.43 s (−31%)** |
| denoise gpu-op sdpa | 12.09 / 12.14 s | 11.77 / 11.91 s | −0.28 s |
| denoise leftover | 1.12 s | 2.54 s | **+1.42 s** |
| DiT peak live | 26.63 GiB | **17.20 GiB** | −9.4 GiB |

Denoise wall drops and the driver is linear, so this is a KEEP under the usual
rule. Note the leftover **rises 1.42 s**: the INT8 path adds an
`apply_scales` pass after each GEMM plus a re-quantize before FC2. That is the
next target, not a reason to reject — the wall still drops 1.28 s.

vs Metal M5 docs denoise 16.69 s → Spark **1.18×** (was 1.31×).

### Accuracy

The old INT8 mode of `h3_cuda_dit_forward_smoke` was a silent no-op: it only
unset `H3_DISABLE_INT8_MLP` while the day-7 default flip also required
`H3_INT8_MLP`, so it re-ran the BF16 path and printed an identical hash. Fixed,
and the numerics are now gated where they mean something:

- **`h3_cuda_ops` gates the MLP op at DiT width** (`rows 64, in 5376, hidden
  14336, out 5376`): INT8 MLP vs BF16 MLP **relL2 0.0069**, gate `< 3e-2`. That
  is ordinary INT8 error, and row FC2 is included in it.
- The forward smoke's deep relL2 (video 0.87 / audio 1.56 over 25 blocks) only
  measures how 0.7% per block compounds through a residual stream whose
  velocity nearly cancels. It is now reported, not tightly gated.
- Output-level check on the 20-step fox-fast videos, PSNR average:
  **BF16 vs BF16 (two runs of the same config) 19.04 dB**, INT8 vs BF16
  **20.69 / 19.82 dB**. The config change moves the output *less* than
  re-running the same config does.

The last line also documents that the pipeline is not bit-reproducible
run-to-run even though `h3_dit_forward` itself is (the smoke's BF16 control is
relL2 0). The nondeterminism is downstream of the DiT; not chased here.

Opt out with `H3_BF16_MLP=1` or `--use-slower-bf16-mlp`; opt back into grouped
FC2 scales with `H3_INT8_GROUP_FC2=1`. Logs: `/tmp/h3_perf_day10/fox-s2-i8row-ab*.log`,
`fox-s2-bf16-ab*.log`, `fox-fast-i8-ff*.log`, `fox-fast-bf16-ff*.log`.
---

## 2026-08-25 — KEEP fused INT8 FC1 epilogue (scale + SwiGLU + requantize)

The INT8 MLP's leftover cost was three separate passes between the two GEMMs:
`apply_scales` wrote the `[rows, 2*hidden]` gate/up pair as BF16, SwiGLU read it
and wrote `[rows, hidden]`, then the row quantizer read that twice (max, then
quantize). One kernel now does all of it from the int32 accumulator — one block
per row, the activation staged in 28 KiB of shared memory instead of HBM — so
neither the gate/up pair nor the activation is ever written to memory.

The BF16 roundings are applied at the same three points as the split chain, so
this is bit-identical, not an approximation: `h3_cuda_ops` reports the same
relL2 0.00691 and the forward smoke the same 0.8693 / 1.556.

fox-s2 interleaved (`H3_SPLIT_INT8_SWIGLU=1` is the split control):

| Run | **Fused** | Split |
|-----|----------:|------:|
| ab1 | 2.713 s (linear 0.766) | 2.834 s (linear 0.806) |
| ab2 | 2.798 s (linear 0.793) | 2.877 s (linear 0.807) |

Both fused runs beat both split runs. Dispatches per run 866 vs 1006 (two
kernels × 70 block invocations), denoise scratch alloc 0.075 vs 0.175 GiB — the
`ws_int8_fc1` workspace is gone.

fox-fast (20 steps, 45 layers):

| Metric | **Fused** | Split | Δ |
|--------|----------:|------:|--:|
| denoise wall | **20.10 s** | 20.70 s | −0.60 s |
| denoise gpu-op linear | 5.414 s | 5.475 s | −0.06 s |
| denoise gpu-op sdpa | 12.687 s | 12.689 s | ±0 |
| denoise leftover | **2.00 s** | 2.54 s | −0.54 s |
| dispatches | 6165 | 7155 | −990 |

So the +1.42 s leftover the INT8 flip introduced is now +0.88 s, and the win is
entirely in the leftover as intended: SDPA is unchanged to 2 ms. Denoise vs
Metal M5 docs 16.69 s → **1.20×**.

Opt out with `H3_SPLIT_INT8_SWIGLU=1`. The fused path requires row FC2 scales
(the grouped variant cannot share the row reduction) and `hidden*2 B` of shared
memory, so it self-disables above ~23k hidden. Logs:
`/tmp/h3_perf_day11/fox-s2-fused*.log`, `fox-s2-split*.log`,
`fox-fast-fused.log`, `fox-fast-split.log`.
---

## 2026-08-25 — KEEP tensor-core SDPA: DiT denoise 20.1 s → 8.8 s

The wave kernels ran both attention products on FP32 CUDA cores. At the DiT's
shape (sequence 1874, 56 heads, head_dim 128 — 101 GFLOP per attention call)
they took 25.6 ms, i.e. **~4 TFLOP/s**, while the same part sustains ~98
TFLOP/s of BF16 through the MMA pipes on GEMM shapes. That gap, not memory
traffic, was the SDPA cost.

`h3_sdpa_bf16_mma_d128_kernel` is flash-attention shaped: one block per
(64-query tile, head), four warps, each owning 16 query rows across the whole
head_dim; K/V stream through shared memory in 64-key tiles and Q never leaves
the MMA fragments. Two properties of the `m16n8k16` fragment maps do the heavy
lifting:

- The accumulator's lane map (lane `l` holds rows `l/4` and `l/4 + 8` at
  columns `2*(l%4)` and `+1`) puts each row's max and sum inside a group of
  four lanes, so the online softmax needs three `shfl_xor`s and rescaling the
  output accumulator by `alpha` needs no cross-lane traffic at all.
- The A fragment wants exactly the (row, column) pairs the accumulator already
  holds, so P feeds the second MMA straight out of the score registers — no
  shared-memory round trip for the probabilities.

Scores are BF16 (Q and K are already BF16 in memory, and the products
accumulate exactly in F32), but **P·V runs in FP16**. P lives in [0, 1] and V
is O(1), so FP16's 11-bit mantissa is free accuracy over BF16's 8:

| P·V precision | relL2 vs F32 reference, sequence 1874 |
|---------------|--------------------------------------:|
| BF16 | 0.01335 |
| **FP16** | **0.00239** |
| FP32 wave kernel (the baseline) | 0.00164 |

fox-fast (20 steps, 45 layers):

| Metric | **MMA** | Wave | Δ |
|--------|--------:|-----:|--:|
| denoise wall | **8.80 / 9.06 s** | 20.08 s | −11.2 s |
| denoise gpu-op sdpa | **1.60 s** | 12.78 s | **−87%** |
| denoise gpu-op linear | 5.21 s | 5.34 s | ±noise |
| e2e wall | 56.3 s | 67.7 s | −11.4 s |

That is 3.24 ms per attention call, **31 TFLOP/s**, an 8.0× rate improvement.
fox-s2 denoise 2.78 s → 1.27 s in the same A/B.

**Denoise vs Metal M5 docs 16.69 s → Spark 8.80 s, 1.90× faster**, where the
BF16 baseline was 1.31× slower.

### Accuracy

- `h3_cuda_ops` gates the kernel against the F32 CPU reference at sequence 200
  (deliberately not a multiple of the 64-wide tile, so both masks are
  exercised) in both output layouts: worst absolute error **0.00037**.
- A second gate compares both kernels against an F64-accumulated reference at
  the DiT's own sequence 1874, where the output is a heavily cancelling
  average: wave 0.00164, MMA 0.00239. Comparing the kernels to each other says
  nothing about which is closer, so the gate is written against the reference.
- fox-fast output PSNR: MMA vs wave **25.63 dB**, against a run-to-run floor of
  **18.72 dB** for two runs of the same config. Swapping the kernel perturbs
  the video less than re-running the same config does.

Opt back to the FP32 wave kernel with `H3_SDPA_WAVE=1`. Only head_dim 128 takes
the MMA path; the wave kernels still cover every other shape. The video VAE's
attention is unaffected (it is not head_dim 128) and still costs 3.57 s. Logs:
`/tmp/h3_perf_day11/fox-fast-mma16.log`, `fox-fast-wave16.log`,
`fox-s2-mma*.log`, `fox-s2-wave*.log`.

Unrelated fix found on the way: the `gate_adaln` fixture in `h3_cuda_ops`
indexed its modulation by a row map containing 0 and 1 while allocating only
one map row, so both the reference and the kernel read one row past the end of
their buffers and agreed only while that memory happened to be zero. It now
allocates both rows, which also makes the row map meaningful.
---

## 2026-08-25 — TF32 F32 GEMM: 2.1× on the video VAE, kept opt-in

`h3_gpu_linear_f32` ran `CUBLAS_COMPUTE_32F`, i.e. no tensor cores at all.
`H3_F32_TF32=1` switches the large shapes (all of `input_dim`, `output_dim` and
`rows` at least 512, so the op gate's small cases stay exact) to
`CUBLAS_COMPUTE_32F_FAST_TF32`. fox-s2, interleaved:

| Run | **TF32** | FP32 |
|-----|---------:|-----:|
| ab1 | video VAE 7.29 s (linear 1.78 s) | 9.16 s (linear 3.65 s) |
| ab2 | video VAE 7.24 s (linear 1.70 s) | 9.19 s (linear 3.69 s) |

The GEMM more than halves and the VAE wall drops 1.9 s. It is **not** the
default, because of what the output stability says:

| PSNR pair | average |
|-----------|--------:|
| FP32 run vs FP32 run (same config) | **32.65 dB** |
| TF32 run vs TF32 run (same config) | 22.24 dB |
| TF32 vs FP32 | 23.24 dB |

The video VAE is already not bit-reproducible run to run (documented earlier as
nondeterminism downstream of the DiT, most likely a split-k reduction), and
TF32 raises that floor by ~10 dB: two TF32 runs differ from each other about as
much as a TF32 run differs from an FP32 one. So TF32's error is not a small
bias on a stable output, it makes the decode measurably noisier, and 23 dB is
large enough to see. A 1.9 s VAE win is not worth spending that silently, so
the flag exists and the default stays exact.

The other half of the video VAE, its 3.55 s of FP32 attention, is untouched:
the DiT's MMA kernel is head_dim 128 only, and an FP16 path there would raise
the same question this measurement just answered.
---

## 2026-08-25 — REJECT 128-query MMA tile

Eight warps per block sharing each K/V tile instead of four, i.e. 128 query
rows per block. fox-fast, interleaved:

| Run | 64 rows (kept) | **128 rows** |
|-----|---------------:|-------------:|
| ab1 | 8.793 s (sdpa 1.602) | 9.609 s (sdpa 2.444) |
| ab2 | 8.778 s (sdpa 1.588) | 9.750 s (sdpa 2.437) |

53% slower attention. The output accumulator is already 64 registers per lane,
so doubling the warps per block halves the blocks resident per SM without
buying anything: the K/V tile was never the constraint. Reverted.
---

## 2026-08-25 — KEEP parallel weight reads (e2e 56.3 s → 37.0 s)

With denoise down to 8.8 s, the run is dominated by weight loading: 22.3 s for
the DiT and 13.3 s for the text encoder out of a 56.3 s fox-fast wall. Both ran
at 0.6 GB/s, which is not a disk limit and not a copy limit — the reads and the
H2D copies already overlap through the pinned double buffer, so the wall is the
read rate of a single `pread` stream. The device just needs queue depth:

| Concurrent readers, 8 MiB requests | Throughput |
|-----------------------------------:|-----------:|
| 1 | 0.94 GB/s |
| 2 | 1.32 GB/s |
| 4 | 2.55 GB/s |
| 8 | 4.50 GB/s |

So `h3_read_file_at` now fans a tensor out over up to 12 `pread` slices on
1 MiB boundaries (each worker opens its own descriptor; the calling thread takes
slice 0 and joins the rest). Tensors under 8 MiB, and `H3_LOAD_READ_THREADS=1`,
still take the original serial path. DiT weights are 231–308 MB per tensor, so
nearly all of the bytes go through the parallel path.

fox-s2 (`512² / 22f / steps=2 / L35 / reuse=1`), interleaved:

| Metric | **12 readers** | 1 reader | Δ |
|--------|---------------:|---------:|--:|
| text encoder | **6.36 / 6.50 s** | 12.72 / 13.70 s | −2.05× |
| DiT load | **8.52 / 8.56 s** | 14.64 / 17.32 s | −1.87× |
| denoise | 1.31 s | 1.31 s | ±noise |

fox-fast (20 steps, 45 layers), interleaved:

| Metric | **12 readers** | 1 reader | Δ |
|--------|---------------:|---------:|--:|
| text encoder | **6.38 s** | 13.13 s | −6.8 s |
| DiT load | **9.94 s** | 22.06 s | −12.1 s |
| DiT total | **19.02 s** | 31.15 s | −12.1 s |
| video VAE prefetch | **1.35 s** | 1.76 s | −0.4 s |
| denoise | 8.86 s | 8.88 s | ±noise |
| **e2e wall** | **37.0 s** | 56.3 s | **−19.3 s** |

Reader-count sweep on fox-s2 DiT load: 4 → 10.64 s, 8 → 8.52 s, 12 → 8.05 s,
16 → 7.98 s. Twelve is where the curve flattens, so that is the default.

### Correctness

Bytes are bytes: `h3_cuda_dit_forward_smoke` produces video hash
`ab054bb20d9c4c83` and audio hash `863553009bcf1b83` with both 12 readers and
`H3_LOAD_READ_THREADS=1`, i.e. the weights land bit-identically. `h3_tests`
(1772 checks), `h3_cuda_ops`, `h3_cuda_dit_block_smoke` and
`h3_cuda_video_vae_smoke` all pass. The parallel path also touches no shared
loader state, so it is safer than the serial one under the background video VAE
prefetch, which reads on its own thread.

The remaining load time is no longer a single stream's read rate; the DiT's
9.9 s for its weight set is a mix of per-tensor GPU quantization and the read
fan-out not reaching full depth on the smaller tensors. Logs:
`/tmp/h3_perf_day12/fox-s2-{par,ser,t4,t12,t16}*.log`,
`fox-fast-{ffpar,ffser,default}.log`.
---

## 2026-08-25 — KEEP tensor-core video VAE attention (VAE 8.72 s → 5.74 s)

The video VAE's attention was the last big FP32 CUDA-core kernel: 144 calls,
3.57 s, head_dim 64 across 32 heads. It now runs the same flash shape as the
DiT's MMA kernel (one block per 64-query tile per head, four warps, 64-key
stride, online softmax in registers), reading F32 and writing F32 with the
tiles narrowed to FP16 on the way in.

Narrowing is where this differs from the DiT. There, everything is BF16
already, so a 16-bit MMA costs nothing. Here the decoder is F32 end to end, and
a single FP16 pass is visibly worse than the CUDA-core kernel it replaces. So
each F32 splits into a high FP16 and the FP16 of its residual, and both
products are summed the way a double-double does it — `hi·hi + hi·lo + lo·hi`,
dropping only `lo·lo` at ~2⁻²². The ladder, measured against an F64 CPU
reference at sequence 900 (not a multiple of the 64-wide tile, so both masks
are live):

| Variant | relL2 vs F64 | fox-fast VAE sdpa |
|---------|-------------:|------------------:|
| FP32 wave kernel (the baseline) | 0.000010 | 3.565 s |
| Single FP16 | 0.004411 | 0.302 s |
| Split Q·K only | 0.003120 | 0.423 s |
| Split Q·K and V | 0.002231 | — |
| **Split Q·K, V and P** | **0.000020** | **0.622 s** |

The interesting part is which split mattered. Splitting the score product got
only a third of the error, and V another third; the rest was P's own FP16
rounding, which the `p_lo·V_hi` term removes. All three splits together land
within 2× of the exact kernel's own error, i.e. this is no longer a precision
trade at all — it is 5.7× the throughput at F32 accuracy, so it is on by
default rather than behind a flag. `H3_SDPA_F32_WAVE=1` restores the CUDA-core
kernel.

fox-fast (20 steps, 45 layers):

| Metric | **MMA** | Wave | Δ |
|--------|--------:|-----:|--:|
| video VAE sdpa | **0.622 s** | 3.565 s | **−82%** |
| video VAE wall | **5.74 s** | 8.72 s | −3.0 s |
| **e2e wall** | **33.5 s** | 37.0 s | −3.5 s |

`h3_cuda_video_vae_smoke` reports abs-max 0.677619 with the MMA kernel and
0.677619 with the wave kernel — the decode is unchanged to every digit it
prints, which the single-FP16 version was not (0.677606). `h3_tests` (1772
checks), `h3_cuda_ops`, `h3_cuda_smoke`, `h3_cuda_dit_block_smoke` and
`h3_cuda_dit_forward_smoke` all pass; the forward smoke's hashes are unchanged.

Shared memory is four 64×72 half tiles (36 KB), still under the 48 KB static
limit, and the accumulator is 32 registers instead of the d128 kernel's 64.
Logs: `/tmp/h3_perf_day12/fox-fast-{vmma,vwave,vaemma}.log`,
`split-{qk,full}.log`.
---

## 2026-08-25 — KEEP pooled device allocation and a coarse staging buffer

With the reads parallelized, an `nsys` trace of the load showed the remaining
time was not I/O at all — it was the CUDA allocator:

| API | Total | Calls | Avg |
|-----|------:|------:|----:|
| `cudaMalloc` | 6.73 s | 2833 | 2.4 ms |
| `cudaFree` | 2.01 s | 2837 | 0.7 ms |
| `cudaMallocHost` | 2.43 s | 24 | 101 ms |
| `cudaFreeHost` | 1.17 s | 24 | 49 ms |

Confirmed independently: a diagnostic build that skipped the file reads
entirely still took 6.46 s of the DiT's 9.88 s load, so the reads were already
almost free and the driver was the wall.

Two changes:

- Tensors allocate through `cudaMallocAsync`/`cudaFreeAsync` on the GPU stream,
  with the default pool's release threshold raised to 24 GiB so freed blocks
  stay resident for reuse. The loader allocates the same handful of shapes over
  and over (a BF16 staging tensor per weight, quantized to INT8 and dropped),
  which is exactly what a pool is for. `H3_GPU_SYNC_ALLOC=1` restores
  `cudaMalloc`.
- The pinned staging pair rounds up to 256 MiB granularity and never shrinks,
  instead of being resized to each tensor. That takes the host-side allocation
  from 24 calls to 2.

fox-s2 (`512² / 22f / steps=2 / L45 / reuse=1`), interleaved:

| Metric | **Pooled** | `cudaMalloc` | Δ |
|--------|-----------:|-------------:|--:|
| text encoder | **5.65 s** | 6.27 s | −0.6 s |
| DiT load | **6.53 s** | 9.34 s | −2.8 s |
| denoise | 1.71 s | 1.67 s | ±noise |
| video VAE | **5.47 s** | 5.68 s | −0.2 s |

fox-fast e2e **29.9 s**, from 33.5 s before this change. Peak live memory is
unchanged (16.9 GiB), and the pool does not bloat: the minimum `MemAvailable`
during an 864×480 run is 96 GiB of 124 GiB. The whole suite passes with the
forward smoke's hashes unchanged (`ab054bb20d9c4c83` / `863553009bcf1b83`).

Also measured and rejected on the way: `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` for
the video VAE's 3.6 s of FP32 GEMM. Nine BF16 products at 98 TFLOP/s is no
faster than FP32's own ~31 TFLOP/s, cuBLAS declines the emulation under its
default strategy, and the wall time does not move (3.597 s vs 3.634 s). That
GEMM is at the hardware's exact-FP32 limit unless TF32 is accepted, which an
earlier entry already rejected. Logs: `/tmp/h3_perf_day12/alloc-{pool,sync}.log`,
`fox-fast-{pool,pool2,bfx9,exact}.log`, `dit_load.nsys-rep`.
---

## 2026-08-25 — KEEP a fixed 128 MiB staging buffer with chunked copies

Pooling the device allocations left pinned host memory as the next driver cost:
`cudaMallocHost` was 2.41 s over 10 calls (up to 682 ms each) and `cudaFreeHost`
another 0.98 s, because the staging pair still grew to fit the largest tensor
seen so far and pinning runs at roughly 1.5 GB/s.

The staging pair is now allocated once at 128 MiB (`H3_LOAD_STAGE_MIB`) and
tensors larger than that are copied in pieces. Chunking is not a cost: the
double buffer already alternates, so a large tensor's second piece is read while
its first piece is copying, which the old whole-tensor path could not do.

fox-s2 (`512² / 22f / steps=2 / L45 / reuse=1`):

| Metric | **128 MiB chunked** | Grown to fit | Δ |
|--------|--------------------:|-------------:|--:|
| text encoder | **4.50 s** | 5.91 s | −1.4 s |
| DiT load | **5.41 s** | 6.18 s | −0.8 s |

Chunk sweep (fox-s2 DiT load): 64 MiB → 5.62 s, 128 MiB → 5.41 s,
256 MiB → 5.48 s.

fox-fast is now **26.8 s** end to end, with DiT load 5.45 s, denoise 8.80 s,
text encoder 4.52 s, video VAE 5.37 s, audio VAE 1.15 s. The whole suite passes
with the forward smoke's hashes unchanged.

Also measured and rejected: pre-warming the memory pool with one large
`cudaMallocAsync`/`cudaFreeAsync` pair at startup. It does move time — DiT load
5.28 s → 4.35 s with an 18 GiB reserve — but e2e does not change (26.6 s vs
26.6 s, 26.4 s vs 26.4 s in two interleaved pairs), because the page mapping
just happens earlier instead of not happening. The pool's ~0.5 ms per allocation
is page mapping, not bookkeeping, so the only real fix is allocating fewer
bytes.

Also measured: the INT8 GEMM has no headroom left. A microbenchmark of the four
shapes the DiT actually issues (all M=1874: N=21504 K=5376 for QKV, N=28672
K=5376 for FC1, N=5376 K=14336 for FC2, N=5376 K=7168 for the attention output)
runs at **143 TFLOP/s** through `cublasGemmEx`, and cuBLASLt's best heuristic
candidate is within 1% of it (144 TFLOP/s). Those shapes at that rate account
for essentially all of the denoise's 4.96 s of INT8 GEMM time, so the GEMM is
done — 89% of the 160 TFLOP/s INT8 ceiling. Re-confirmed the INT8-vs-BF16 MLP
choice at the same time: BF16 denoise linear is 8.19 s against INT8's 5.31 s.
Logs: `/tmp/h3_perf_day12/stage-{64,128,256}.log`, `pw-{0,18}.log`,
`pwf-{0,16}.log`, `mlp-{i8,bf16}.log`, `int8_bench.cu`.
---

## 2026-08-25 — state of play after the load and VAE work

fox-fast (`512² / 22f / steps=20 / L45 / reuse=2`), one clean run per column:

| Stage | 2026-08-24 evening | **Now** | Δ |
|-------|-------------------:|--------:|--:|
| text encoder | 13.27 s | **4.52 s** | −2.9× |
| DiT load | 22.33 s | **5.45 s** | −4.1× |
| DiT denoise | 8.80 s | **8.80 s** | — |
| audio VAE | 0.95 s | **1.15 s** | — |
| video VAE | 9.10 s | **5.37 s** | −1.7× |
| **e2e wall** | **56.3 s** | **26.8 s** | **−2.1×** |

Where the remaining 26.8 s goes, and what is left in each:

- **denoise 8.80 s.** INT8 GEMM 4.96 s at 143 TFLOP/s, i.e. 89% of this chip's
  INT8 ceiling — done. Attention 1.60 s at 31 TFLOP/s of a ~98 TFLOP/s BF16
  peak, the largest single remaining headroom. The other ~2.2 s is the
  memory-bound glue between GEMMs (int32→scale→BF16 0.74 s, fused
  SwiGLU-quantize 0.54 s, QKV RoPE 0.36 s, gate/AdaLN quantize 0.35 s), all of
  it running at DRAM bandwidth; the only way down is fusing consumers, worth
  about 0.3 s for the RoPE pair and less for the others.
- **video VAE 5.37 s.** 3.6 s of it is FP32 GEMM, which cuBLAS runs at
  **10–14 TFLOP/s** (measured across the five shapes). TF32 reaches 25–39
  TFLOP/s but was rejected on output noise, and both BF16x9 emulation and a
  hand-split FP16 GEMM land back at ~13 TFLOP/s effective because three 16-bit
  products cost what one FP32 product does. The checkpoint itself is F32, so
  dropping precision here is not free.
- **DiT load 5.45 s + text encoder 4.52 s.** Now genuinely I/O bound: the run
  reads 58 GB (`/usr/bin/time -v`, File system inputs 113,553,992 blocks) at an
  effective 5.8 GB/s across the loading stages. Reading fewer bytes — caching
  quantized INT8 weights on disk — is the only lever left, and that is a
  behaviour change, not a tuning change.

The parallel reads matter most where reads are on the critical path. With
`--ssd-streaming`, which fetches each BF16 layer during denoise instead of
resident INT8:

| fox-s2 `--ssd-streaming` | denoise wall |
|--------------------------|-------------:|
| 12 readers | **12.85 s** |
| 1 reader | 71.23 s |

That is 5.5× on the streaming path, versus 1.9× on the resident path, because
streaming has nothing else to hide the read behind.

Verified on this tree: `make -f Makefile.linux test` (unit, ops, and the DiT
block, text, vision, video VAE, audio VAE, video encoder and audio encoder
smokes), `h3_cuda_dit_forward_smoke` with unchanged hashes, and
`make -f Makefile.linux test-conditional` (FL2VA first/last plus Ref2VA image,
audio and silent paths, 5m20s). Logs: `/tmp/h3_perf_day12/conditional.log`,
`ssd.log`, `ssd-ser.log`, `f32_bench.cu`.
---
