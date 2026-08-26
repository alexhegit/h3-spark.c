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

> **Reversed 2026-08-26.** Every PSNR figure below was taken while the staging
> buffer race was corrupting weights nondeterministically, which is what the
> "FP32 run vs FP32 run 32.65 dB" line was actually measuring. With that race
> fixed both arms are bit-reproducible and TF32 sits 45.3 dB from exact FP32, so
> the reasoning here does not survive its own premise. TF32 is now the default —
> see "TF32 on the video VAE, on the second look".


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
done — 89% of the 160 TFLOP/s INT8 ceiling. **(Corrected 2026-08-26: that
ceiling was wrong. FP8 runs these shapes at 190–203 TFLOP/s and NVFP4 at
289–365 through the same tensor cores, so 143 was a cuBLAS INT8 kernel limit,
not the chip's. See "narrow formats" below.)** Re-confirmed the INT8-vs-BF16 MLP
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
  INT8 ceiling — done. *(Corrected 2026-08-26: not done, and the accounting
  missed the apply pass. FP8 does these shapes at ~195 TFLOP/s and NVFP4 at
  ~320; the reachable saving is ~2.2 s, at a quality cost that rejects it for
  now.)* Attention 1.60 s at 31 TFLOP/s of a ~98 TFLOP/s BF16
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

## 2026-08-25 — KEEP a cheaper online softmax, REJECT a transposed V tile

Two attempts at the DiT's attention kernel, which sits at 31 TFLOP/s of a
~98 TFLOP/s BF16 peak.

**Rejected: storing the V tile transposed.** A B fragment for P·V needs two
keys at one dimension, which the row-major tile serves as four 16-bit shared
reads plus shifts. Dimension-major (pitch 72 halves, so a quad-group's eight
columns cover all 32 banks) makes it two aligned 32-bit reads. It is 82% slower:
attention 1.62 s → 2.95 s, denoise 8.80 s → 10.0 s. Transposing moves the cost
to the store side — each thread then writes one dimension's key pair, whose
addresses collide four ways — and the read side was evidently not the
constraint: two lanes reading different halves of one word broadcast rather than
conflict, so the "half the banks" reasoning that motivated this was wrong.

**Kept: two cheap softmax edits.** Masked columns hold −inf and `expf(-inf)` is
already 0, so the per-element comparison guarding the exponential is
unnecessary; and after the first few key tiles the running row max rarely moves,
so the 64 accumulator rescales per tile are usually multiplication by 1 and can
be skipped behind a warp vote. Interleaved fox-fast, two pairs:

| Run | Baseline attention | **With both** |
|-----|-------------------:|--------------:|
| ab1 | 1.599 s | **1.538 s** |
| ab2 | 1.592 s | **1.534 s** |

3.8% off the kernel, accuracy unchanged (relL2 0.00239 at sequence 1874, worst
absolute error 0.00037 at sequence 200). The denoise wall does not move by a
measurable amount — 0.06 s is inside its run-to-run spread — so this is banked
as reduced GPU work rather than as a wall-clock win. Logs:
`/tmp/h3_perf_day12/ab-{base,soft}-{1,2}.log`, `vt-{1,2}.log`.
---

## 2026-08-25 — KEEP the F32 bias in the GEMM epilogue (video VAE −0.27 s)

`h3_gpu_linear_f32` added its bias in a separate kernel, which is a full read
and write of the GEMM output for an operation the GEMM could have done on the
way out: 590 launches and 0.27 s a run. It now goes through `cublasLtMatmul`
with `CUBLASLT_EPILOGUE_BIAS`. The Lt plan depends only on the shape and the
VAE issues five, so heuristics are looked up once per shape and cached; any
failure anywhere in the Lt path falls back to the old `cublasGemmEx` plus bias
kernel, and `H3_F32_SPLIT_BIAS=1` forces that path.

Interleaved fox-fast, two pairs:

| Run | **Fused** | Separate kernel |
|-----|----------:|----------------:|
| video VAE wall | **5.09 s / 5.13 s** | 5.39 s / 5.38 s |
| video VAE GEMM | **3.46 s / 3.50 s** | 3.74 s / 3.77 s |
| dispatches | **1468** | 2056 |

The audio VAE is convolution-only and does not change (0.94 s either way). Both
VAE smokes report identical values (video abs-max 0.677619, audio 0.16749) and
the whole suite passes. Logs: `/tmp/h3_perf_day12/bias-{fused,split,fused2,split2}.log`.
---

## 2026-08-25 — two output-correctness findings

> **Superseded:** both findings below are one bug, a data race in the weight
> staging buffer introduced on 2026-08-23 by `55a61f0`, fixed the same day it
> was found. See *"the staging buffer was shared between loader threads"* at the
> end of this document. The conclusion drawn here — that the problems predate
> the optimizations — was wrong: the env switches used to test it
> (`H3_LOAD_READ_THREADS=1 H3_GPU_SYNC_ALLOC=1 H3_SDPA_F32_WAVE=1
> H3_F32_SPLIT_BIAS=1`) do not disable the shared staging buffer, because
> nothing did. Rebuilding an older commit, rather than switching flags off, is
> what settled it.

Checking the actual pixels at the end of the session, rather than only hashes
and wall clocks, turned up two problems that are independent of every change in
this document. Both reproduce with all of them disabled
(`H3_LOAD_READ_THREADS=1 H3_GPU_SYNC_ALLOC=1 H3_SDPA_F32_WAVE=1
H3_F32_SPLIT_BIAS=1`).

**1. Output does not depend on the prompt.** Three prompts — "A red fox walks
through fresh snow in a pine forest…", "An astronaut rides a white horse on the
surface of the moon…", and a 321-word prompt written in the documented
H3-Context-IR shape (`integrated_multimodal_description:` / `[Shot 1]` /
`overall_soundscape:` / `non_diegetic_music:`) — all produce the same family of
scene: hands preparing food or pouring water on a wooden surface, i.e. what
looks like the training distribution's mode rather than anything conditioned.

The text encoder is not the problem. `h3_real_prompt_test` (now buildable on
Linux) shows it tokenizes and encodes correctly and prompt-dependently:

| Prompt | Tokens | Layer-50 embedding hash |
|--------|-------:|-------------------------|
| "A red fox walks through fresh snow in a pine forest." | 12 | `bfdb59bb391c82aa` |
| "An astronaut rides a white horse on the moon." | 10 | `9757e55740146973` |

So the conditioning is lost somewhere between that embedding and the denoised
latent: `refine_text` projects it through `condition_proj` and two token-refiner
blocks into `dit->refined_text`, and that is the next thing to audit — likely
whether `refined_text` actually reaches the joint attention, and whether the
prompt template the checkpoint expects differs from what is fed in. Note the
model card is explicit that H3 expects Context-IR input of ~5,000–12,800 prompt
tokens, so short prompts are out of distribution by design; but the structured
321-word attempt failing the same way says this is not only a prompt-length
issue.

**2. Identical invocations produce different videos.** The same command twice,
same default seed 42, gives different files (md5 `0c65f6dd…` vs `9b66fe82…`)
and 22.87 dB PSNR between them — a different scene, not rounding. The latent
noise is seeded deterministically (`h3_rng_seed(&video_rng, params->seed)` for
both modalities) and `h3_cuda_dit_forward_smoke` hashes identically run to run,
so the divergence is introduced after that seeding and outside the forward
smoke's coverage. This is the same nondeterminism noted earlier in this document
as "downstream of the DiT; not chased here", but it is worth recording that its
magnitude is semantic, not numerical: it is why the PSNR floor between runs of
one configuration is ~17–23 dB, and why PSNR cannot be used to validate any
kernel change end to end.

Neither finding affects the performance measurements above — the work per step
is the same either way — but both outrank further tuning: the pipeline is
currently fast at producing something other than what was asked for.
---

### What the nondeterminism is not

Three hypotheses tested and eliminated, so the next attempt can skip them:

- **Uninitialized device memory.** `H3_GPU_ZERO_ALLOC=1` (new, off by default)
  zeroes every tensor allocation on the stream. Two runs under it still diverge
  (20.54 dB, different md5), so no op is reading a tensor before writing it.
- **Uninitialized host memory.** Two runs at a fixed `MALLOC_PERTURB_=1` also
  diverge (17.35 dB), so it is not malloc content leaking into the result
  either.
- **The weight load.** `h3_cuda_dit_forward_smoke` hashes identically across
  every invocation tonight (`ab054bb20d9c4c83` / `863553009bcf1b83`) while
  loading the same weights through the same chunked staging path, so a racing
  host-to-device copy would have to be invisible there to be the cause.

That leaves the parts of a real run the forward smoke does not exercise: the
multi-step loop with core reuse, the preview decode interleaving, the audio
branch, and the tiled VAE decode. The obvious next step is to dump the latent
after each denoise step across two runs and find the first step that differs,
which localizes it to a step boundary rather than a kernel.

The third hypothesis is the one that was true, and the reasoning against it was
the flaw: `h3_cuda_dit_forward_smoke` loads DiT weights on one thread, and only
the Qwen text encoder reads a layer from several threads at once, so the smoke
covers the staging path without ever exercising the race in it.
---

## 2026-08-25 — the staging buffer was shared between loader threads

**Both findings above are one bug, and it is a regression, not a pre-existing
condition.** The 2026-08-17 CUDA baseline (`483ffdf`) renders the fox prompt as
a fox; every build from 2026-08-23 onward renders an unrelated scene (a
spreadsheet, a document, handwriting) and a different one on each run.

### Bisect

Same command on each build, at 256×256 so the pre-optimization commits are
affordable, scored by PSNR against the `483ffdf` output:

| Build | Output | PSNR vs `483ffdf` |
|-------|--------|------------------:|
| `483ffdf` 2026-08-17 CUDA baseline | fox in snow | — |
| `55a2710` wave SDPA + tiled INT8 FC2 | fox in snow | 17.62 dB |
| **`55a61f0` default DiT MLP to BF16** | spreadsheet | **7.84 dB** |
| `82b58b5` pre-session | spreadsheet | 8.52 dB |
| `494f271` session HEAD | spreadsheet | 8.90 dB |

`55a61f0` is the first bad commit, but not for the reason its message suggests:
re-enabling the INT8 MLP it turned off (`H3_INT8_MLP=1`) still renders a
spreadsheet (7.82 dB), and so does dropping the `-gencode
arch=compute_121,code=sm_121` it added (7.85 dB). Reverting only its `h3_dit.c`
half leaves the bug (7.83 dB), which puts it in the `h3_gpu.cu` half.

### Root cause

That half replaced the per-read `malloc` in `h3_gpu_tensor_load_*` and
`h3_gpu_tensor_read_file_bf16` with one staging buffer hanging off `h3_gpu`
(later a pinned pair, then chunked). The DiT loads weights on one thread, so
that is safe there — but `prefetch_slot_start` in `h3_text_encoder.c` reads each
Qwen layer's eleven tensors from up to eight lanes at once, and every lane went
through the same buffer. Lanes overwrote each other's bytes between the `pread`
and the `cudaMemcpy`, so an arbitrary subset of the text encoder's 50 layers got
weights belonging to a different tensor.

That single race explains both symptoms exactly: a corrupted text encoder emits
an embedding unrelated to the prompt (so the DiT is conditioned on noise and
falls back to arbitrary scenes), and which bytes lose the race differs per run
(so two invocations of one command differ semantically, not numerically).

### Fix

`h3_gpu` now owns four staging slots instead of one buffer. A reader claims a
slot for one chunk under a mutex, waits on that slot's copy event before
overwriting it, and releases it as soon as its own copy is enqueued, so slots
still pipeline reads against copies. Slots are handed out round-robin: handing
back the slot just released would make a single-threaded loader wait on its own
copy instead of overlapping with it (DiT load 10.8 s round-robin vs 11.7 s
first-free). The `H3_LOAD_FD_CACHE` descriptor cache became `__thread` for the
same reason — one lane must not close a descriptor another lane is reading.

### Cost

fox-fast 512×512, 22 frames, 20 steps, 45 layers, reuse 2, warm:

| | Fixed | Racy (`494f271`) |
|--|------:|-----------------:|
| e2e wall | **32.1 s / 32.4 s** | 31.3 s / 31.5 s |
| Qwen text encoder | 4.43 s / 4.35 s | 4.15 s / 4.16 s |
| DiT load | 10.82 s / 10.86 s | 10.70 s / 10.87 s |
| GPU Euler denoise | 8.75 s / 8.87 s | 8.75 s |
| output md5 | **identical** | differs every run |

Correctness costs ~0.9 s (2.9%) of end-to-end wall, all of it in the text
encoder and the pinning of two extra slots. Serializing the whole staging path
with one mutex instead — the first fix tried — costs 3.5 s in the text encoder
(7.65 s) and was replaced by the slot pool. 128 MiB chunks remain the best
setting with four slots (32/64/96/192/256 MiB all measure slower).

Three prompts now render what they ask for: fox in snow, astronaut planting a
flag on the moon, coffee cup by a rainy window. `--ssd-streaming` also renders a
correct fox (79.2 s, BF16 weights). Whole suite passes (10 ok checks, all step
smokes). Logs and frames: `/tmp/h3_perf_day12/verab/`.

### What this says about the gates (2026-08-25)

Every gate in this document passed while the pipeline was producing wrong
videos, because all of them compare a build against itself: hashes of a
single-threaded forward smoke, kernel-level relative L2 against a reference
kernel, wall clocks. None of them looked at a frame, and none of them compared
against a known-good older build. The cheap check that would have caught this on
2026-08-23 is the one used to bisect it here: render one fixed prompt at 256×256
and PSNR it against a stored reference clip.
---

## 2026-08-25 — the AdaLN tables did not have to be recomputed

### Where the load time actually went

The staging path now reports its own bytes and seconds (`stage … read= copy=
pin=` on every `--profile` line), which changed the picture completely:

| Phase | Bytes staged | read | H2D copy | pin |
|-------|-------------:|-----:|---------:|----:|
| Qwen text encoder | 46.86 GiB | 2.92 s | 0.03 s | 0.28 s |
| DiT load | 58.14 GiB | 8.73 s | 0.02 s | 0.24 s |

One generation moved **105 GiB** of weights. The host-to-device copy is free on
GB10 — unified memory, and the copies are async — and pinning is a quarter of a
second, so essentially all of the load time is getting bytes out of the file and
into the staging slots. Reads scale with the fan-out up to about 16 readers on
this 20-core part (DiT load 28.1 s at 4, 14.2 s at 8, 11.1 s at 12, 9.9 s at 16,
9.4 s at 20, 9.2 s at 32), so the default went from 12 to 16.

That is a constant factor. The structural finding is what the bytes are:

| Tensor (per block, 50 blocks) | Total | Share of transformer |
|-------------------------------|------:|---------------------:|
| `adaln_proj.linear.weight` `[96768, 2688]` | **24.22 GiB** | **39.2%** |
| `ff.net.0.proj.weight` | 14.36 GiB | 23.3% |
| `ff.net.2.weight` | 7.18 GiB | 11.6% |
| `attn.to_{q,k,v,out}.weight` | 14.36 GiB | 23.3% |

24.22 GiB of AdaLN projection, read on every run, produces the modulation tables
in `h3_dit_schedule_precompute` — and its only inputs are the sigma schedule and
the two condition flags. **Nothing about it depends on the prompt.** The tables
it produces are `time_rows × 96768` BF16 per block: 361 MiB for the fox-fast
schedule, 64× smaller than the weights that generate them. It also ran for all
50 blocks even though the gate ranking prunes 5 of them, which is exactly why
the measured 58.139 GiB matched 45 core blocks plus 50 AdaLN blocks.

### The cache

`h3_dit_schedule_precompute` now looks for `adaln-<key>.bin` under
`$H3_ADALN_CACHE`, else `$XDG_CACHE_HOME/h3-spark`, else `~/.cache/h3-spark`.
The key hashes the schedule's timestep features, the condition flags, the table
dimensions, and the size and mtime of the shard holding
`blocks.0.adaln_proj.linear.weight`, so a different step count, a different
sigma schedule or a swapped checkpoint misses instead of reusing. A miss
computes the tables as before and writes them; `H3_ADALN_CACHE=off` disables the
whole thing. Because BF16 tables round-trip exactly, a hit is bit-identical to a
miss, not merely close.

fox-fast 512×512, 22 frames, 20 steps, 45 layers, reuse 2:

| | Cold cache (writes 361 MiB) | Warm cache |
|--|---------------------------:|-----------:|
| e2e wall | 32.7 s | **21.9 / 22.6 / 24.5 s** |
| DiT load | 11.5 s | **2.6 / 3.1 / 5.1 s** |
| bytes staged in DiT load | 58.14 GiB | **34.15 GiB** |
| DiT load read time | 9.29 s | **1.4 / 2.6 s** |
| GPU Euler denoise | 8.83 s | 8.81 / 8.83 / 8.89 s |
| output md5 | `a90568fa…` | `a90568fa…` (identical) |

Read time falls further than the byte count does, because 105 GiB per run was
thrashing a 121 GiB machine's page cache: dropping the AdaLN reads lets the
remaining 34 GiB stay resident, so it reads at cache speed (24 GB/s) rather than
device speed (6 GB/s).

Two checks that matter more than the numbers: a cache built by the fox prompt,
used by the astronaut prompt, gives the same md5 (`fd3f4b38e0d71af3`) as the
same astronaut run with the cache off — the tables really are prompt-independent
— and `--steps 12` writes a second entry instead of reusing the 20-step one.
Whole suite passes (10 ok checks, 5 step smokes). Logs: `/tmp/h3_perf_day13/`.

### Where fox-fast stands now

| Phase | Wall | Share |
|-------|-----:|------:|
| GPU Euler denoise | 8.8 s | 39% |
| video VAE decode | 5.2 s | 23% |
| DiT load | ~3.1 s | 14% |
| Qwen text encoder | ~2.5 s | 11% |
| audio VAE + mux | ~1.5 s | 7% |

Loading is no longer the head: denoise and the video VAE are, and inside the VAE
it is 3.5 s of F32 GEMM. The text encoder still stages 46.86 GiB per run for a
twelve-token prompt, which is the next structural target — the same
sidecar-cache reasoning does not apply (its output depends on the prompt), but
an INT8 or FP8 weight cache would halve the bytes for every prompt.
---

## 2026-08-26 — narrow formats: FP8 is 1.8× and NVFP4 2.8× the INT8 path, and both cost visible quality

This set out to confirm a decision that had already been made on architectural
grounds: INT8 sits at 89% of this chip's 8-bit ceiling, FP8 shares the same
8-bit tensor cores so it cannot be faster, therefore sub-8-bit work has no speed
upside and buys only error. Every load-bearing part of that was wrong. It is the
clearest case so far for measuring instead of reasoning from the architecture.

`tests/bench_gemm_precision.cu`, now with a build rule so it can be re-taken:

    make -f Makefile.linux h3_gemm_precision && ./h3_gemm_precision

### Rate, at the token count fox-fast actually runs

CUDA-event timed, 30 iterations, `tokens=1870`. `int8+apply` is what the DiT
pays today: the `i32` GEMM plus the separate `h3_int8_apply_scales_bf16_kernel`
pass, charged together, because INT8 matmul here cannot write BF16.

| Shape | bf16 | int8 | **int8+apply** | fp8 e4m3 | int8 D=i8 | nvfp4 | mxfp8 |
|-------|-----:|-----:|---------------:|---------:|----------:|------:|------:|
| qkv `out=21504 k=5376` | 92.6 | 140.9 | **105.5** | 193.0 | 176.1 | **289.5** | 195.7 |
| attn-out `out=5376 k=7168` | 93.5 | 144.8 | **112.8** | 203.1 | 179.6 | **343.3** | 199.5 |
| fc1 `out=28672 k=5376` | 93.0 | 143.6 | **107.6** | 190.8 | 182.3 | **293.5** | 51.4 |
| fc2 `out=5376 k=14336` | 93.7 | 145.2 | **129.1** | 194.8 | 165.5 | **365.2** | 135.4 |

All figures TFLOP/s. Summed over one layer's four GEMMs, in elapsed time:

| Path | One layer | vs today |
|------|----------:|---------:|
| int8 + apply (today) | 12.97 ms | 1.00× |
| int8, D=`i8` | 8.16 ms | 0.63× |
| fp8 e4m3 → BF16 | 7.45 ms | **0.57×** |
| nvfp4 → BF16 | 4.67 ms | **0.36×** |

**The 160 TFLOP/s INT8 ceiling quoted on 2026-08-25 was not the hardware's.**
FP8 reaches 190–203 and NVFP4 289–365 through the same tensor cores, so 143 was
the limit of cuBLAS's INT8 kernels, not of the chip. Two other costs surface in
the same table: the `int32` accumulator write alone is 19% of INT8 GEMM time
(`int8` 10.06 ms vs `int8 D=i8` 8.16 ms for the layer), and the apply pass adds
another 2.9 ms on top — together nearly a third of the INT8 linear path spent on
being unable to write BF16 out of the GEMM. FP8 writes BF16 directly.

Against the denoise's measured 5.25 s of `gpu-op linear`, FP8 would be worth
about **2.2 s** and NVFP4 about **3.4 s** — 10% and 15% of a 22.6 s e2e. Upper
bounds: that 5.25 s also contains the quantize kernels, which do not shrink.

### Accuracy, on the real weights

`tests/probe_fp8_weight_error.py` against the checkpoint, relL2 of the
quantized weight against BF16. Four projections at blocks 0, 3, 25, 49; spread
across blocks is small, so one representative block:

| Tensor | int8/chan | fp8/tensor | fp8/chan | nvfp4/16 |
|--------|----------:|-----------:|---------:|---------:|
| `blocks.25.mlp.fc1` | 0.01005 | 0.02652 | 0.02661 | 0.09450 |
| `blocks.25.mlp.fc2` | 0.01308 | 0.02647 | 0.02646 | 0.09447 |
| `blocks.25.attn.qkv_proj` | 0.01061 | 0.02647 | 0.02646 | 0.09387 |
| `blocks.25.attn.out_proj` | 0.01047 | 0.02651 | 0.02644 | 0.09540 |

**Per-channel FP8 is not better than per-tensor FP8** — identical to four
digits, across every tensor measured. E4M3's error is mantissa-bound, not
range-bound, so a finer scale buys nothing; the 4-bit exponent is spending eight
bits' worth of budget on dynamic range that a per-channel scale already
supplies. That justifies the cheapest granularity for weights, which is also
the one the GEMM can fold. It also means FP8 is 2.1× INT8's weight error with no
way to close the gap by rescaling, and NVFP4 is 7.4×.

Through a whole MLP, weights and activations together
(`tests/probe_fp8_mlp_error.py`):

| Scheme | MLP relL2 |
|--------|----------:|
| int8, per-channel weights / per-token activations (today) | 0.00440 |
| fp8, per-tensor weights / per-token activations | **0.02125** |

4.8× today's error, and it is the activations that dominate: the linear probe
splits FP8's 0.04697 into 0.01774 from weights alone and 0.04362 from
activations alone. Per-token is already the finest activation granularity a
row-major GEMM can use, so this is not fixable by rescaling either.

The shipped kernels are worse still than the model of them. `h3_cuda_ops`, at
the DiT's own width, against a BF16 MLP:

| Path | relL2 |
|------|------:|
| INT8 MLP | 0.00691 |
| FP8 MLP | **0.07783** |

11× rather than 4.8×, because the fused SwiGLU quantizes its intermediate too
and E4M3 loses more there than the probe's per-tensor model assumed. This is the
number the decision rests on: it is measured through the code that would ship,
not through an approximation of it.

### What that error does to the output

`scripts/quant_sensitivity.sh` puts a format's error through the whole pipeline
before anyone implements it: `H3_INT8_LEVELS` coarsens every quantized tensor by
a known factor, so 48 levels reproduces FP8's *weight* error and 13 levels lands
on FP8's *whole-MLP* error (0.02216 against FP8's 0.02125) as well as NVFP4's
weight error.

| Levels | MLP relL2 | PSNR vs 127 | Denoise |
|-------:|----------:|------------:|--------:|
| 127 (today) | 0.00440 | — | 8.72 s |
| 48 | 0.00771 | 18.68 dB | 8.70 s |
| 13 | 0.02216 | 15.96 dB | 8.77 s |

**PSNR is the wrong gate here, and reading it as one would have given the wrong
answer.** 18.68 dB sounds fatal, but the 48-level frame is a sharp, correctly
lit, on-prompt fox — as good as the reference, just a different sample. Euler
denoise is chaotic: any perturbation moves the trajectory to a neighbouring
valid sample, so a low PSNR against a reference render says the output moved,
not that it got worse. Only the frames answer the quality question.

They do answer it, though. At 13 levels — FP8's error level, by whole-MLP relL2
— the fox is still coherent but the fur has lost its fine structure, the
background pines have gone muddy, and the face is misshapen. That is the
admissibility bar FP8 fails: not a different sample, a worse one.

### Where this leaves the narrow-precision work

The FP8 path is implemented and unit-tested in `h3_gpu.cu`
(`h3_gpu_quantize_weight_fp8`, `h3_gpu_linear_fp8_bf16`, `h3_gpu_mlp_fp8_bf16`,
with a plan cache for the cuBLASLt descriptors) and deliberately **not wired
into the DiT**. The speed is real and larger than expected; the quality cost is
also real and larger than the usual "FP8 needs no calibration" story admits.
Blanket FP8 is a **REJECT** on that evidence.

Two things stay open, both grounded in the tables above rather than in
speculation:

- **MXFP8**, e4m3 with a UE8M0 scale every 32 values of K, is supported here and
  matches plain FP8's rate on two of the four shapes (195.7, 199.5 TFLOP/s).
  Its scale granularity attacks exactly the activation error that dominates
  FP8's, which is the only identified route to 8-bit that is both fast and
  accurate. Unresolved: the single heuristic candidate collapses on fc1
  (51.4 TFLOP/s), so it needs an algorithm search before it means anything.
- **Dropping the `i32` accumulator** is worth 19% of INT8 GEMM time at today's
  accuracy — `int8 D=i8` measures it. That needs the epilogue to produce `i8`,
  which is a real change to the scale handling, but it costs no precision.

MXFP4 is not available (`no algorithm` on every shape); NVFP4 is, and is the
fastest thing on the chip by a wide margin, but at 7.4× the weight error it is
past the 13-level frame, not short of it.

Logs: `/tmp/h3_perf_day14/bench_gemm_precision.log`, `/tmp/h3_levels/`.
---

## 2026-08-26 — KEEP TF32 on the video VAE, on the second look (e2e 21.9 s → 19.8 s)

An nsys pass taken for an unrelated reason put the largest single kernel in the
run at `cutlass_80_simt_sgemm_128x256_8x4_tn_align1`, 27.1% of GPU time. `simt`
means it is on the FP32 CUDA cores with the tensor cores idle — the video VAE's
F32 GEMM, which already had a measured 2.1× TF32 path sitting behind an opt-in
flag since 2026-08-25.

That flag was off for a documented reason, and the reason was wrong. The
rejection rested on TF32 raising the decode's noise floor, evidenced by an FP32
run differing from another FP32 run by only 32.65 dB. But that measurement
predates the staging-buffer race fix: what looked like a noisy decoder was
weights being corrupted differently on each run. Re-taken now, two runs per arm,
interleaved (`/tmp/h3_perf_day14/tf32_ab.sh`):

| PSNR pair | 2026-08-25 | **Now** |
|-----------|-----------:|--------:|
| FP32 vs FP32 | 32.65 dB | **identical** (`a90568fa7ec1`) |
| TF32 vs TF32 | 22.24 dB | **identical** (`f5282774d3a4`) |
| TF32 vs FP32 | 23.24 dB | **45.27 dB** |

Both arms are bit-reproducible, and the gap between them is 22 dB better than
the old measurement claimed. The frames are indistinguishable: same composition,
same fur detail, same background.

There is a structural reason to trust PSNR here, having just argued it is the
wrong gate for DiT quantization. The video VAE decodes a latent the denoise has
already finalised, so a change inside it cannot move the trajectory to a
different sample the way an FP8 MLP does — it can only perturb the decode of a
fixed one. Same composition means same sample, so the dB figure is measuring
what it looks like it is measuring.

| Arm | video VAE | e2e |
|-----|----------:|----:|
| exact FP32 (`H3_DISABLE_F32_TF32=1`) | 5.41 / 5.68 / 5.27 s | 24.3 / 22.1 / 21.9 s |
| **TF32 (default)** | **3.34 / 3.27 / 3.29 s** | **19.8 / 20.7 / 19.8 s** |

−2.0 s on the VAE and −2.1 s e2e. The flag inverted to
`H3_DISABLE_F32_TF32=1`, which still reproduces the historical reference md5
`a90568fa7ec1` exactly — the exact path is intact, not merely nearby.

### Where fox-fast stands now

| Phase | Wall | Share |
|-------|-----:|------:|
| GPU Euler denoise | 8.8 s | 44% |
| video VAE decode | 3.3 s | 17% |
| DiT load | ~3.1 s | 16% |
| Qwen text encoder | ~2.5 s | 13% |
| audio VAE + mux | ~1.5 s | 8% |
| **e2e wall** | **19.8 s** | |

2.8× off the 56.3 s this started at. Denoise is now 44% of the run and the only
block above 4 s. Its own split is 5.2 s of INT8 linear — where the narrow-format
work above says the next 0.5 s is the three unfused `int32 → scale → BF16`
passes, and the rest needs a format the quality does not currently allow — and
1.5 s of attention still at 31 TFLOP/s against a ~98 TFLOP/s BF16 peak, which is
the largest single unclaimed gap left in the pipeline.

**Amended 2026-08-26.** That last sentence is wrong. The attention kernel is not
tensor-core bound, so the ratio to the GEMM peak does not describe a gap that
can be claimed — see "REJECT further work on the DiT attention tile" below.

Logs: `/tmp/h3_perf_day14/tf32/`.
---

## 2026-08-26 — KEEP fusing the QKV rescale into RoPE (denoise −0.35 s), REJECT vectorizing the row quantizer

The denoise's INT8 linear path spends time on nothing but changing
representation: the GEMM cannot read BF16 or write BF16, so each one is wrapped
in a quantize before and an `int32 → scale → BF16` rescale after. nsys on a
4-step run put `h3_int8_apply_scales_bf16_kernel` at 405 instances / 210.9 ms,
which is three per layer — QKV, the attention output and FC2. Only FC1's is
already fused, into `h3_int8_swiglu_quant`.

QKV is the one worth taking. Its projection is `rows × 3*inner`, four times
wider than the other two, so it alone is 69% of that time (145 ms of 210.9,
against 66.7% predicted from the width ratio).

`h3_qkv_rope_coop_kernel` is now templated over where it reads the projection
from: a materialized BF16 tensor, or the INT8 accumulator plus the two scale
vectors. The fused source rounds each value to BF16 before returning it, exactly
where the split path's store did, so the fusion is **bit-identical** rather than
merely close — same trick as the SwiGLU fusion. Per element this drops a 2-byte
write and a 2-byte read and turns the RoPE kernel's read from 2 bytes into 4:
8 bytes of traffic becomes 4.

Interleaved, `H3_SPLIT_INT8_QKV_ROPE=1` for the split arm:

| Run | **Fused** | Split |
|-----|----------:|------:|
| ab1 | **8.330 s** | 8.702 s |
| ab2 | **8.386 s** | 8.717 s |

−0.35 s of denoise, output md5 `f5282774d3a4` on all four runs. nsys confirms
the mechanism rather than just the wall: `apply_scales` drops to 270 instances /
65.9 ms, and the new `h3_qkv_rope_coop_kernel<h3_qkv_int8_source>` costs 142.9 ms
against the old pair's 145 + 101 ms.

### REJECT the vectorized row quantizer

`h3_quantize_bf16_int8_rows_kernel` looked like the bigger fish: 250.9 ms at
~1.3 ms per call, which for ~40 MB of traffic is about 31 GB/s on a machine with
roughly 270. It issues scalar 2-byte loads and 1-byte stores, so the obvious
read is latency-bound with 2 bytes in flight per thread. A `uint4`-load,
`uint2`-store variant, bit-identical and with a scalar fallback for unaligned or
non-multiple-of-8 rows, measured:

| Arm | Kernel total | Denoise |
|-----|-------------:|--------:|
| vector | 239.1 ms | 8.347 / 8.386 s |
| scalar | 250.9 ms | 8.367 / 8.331 s |

5% on the kernel and nothing on the wall, so it is not latency-bound and the
premise was wrong. Two things were also wrong about the attribution: those 192
instances are mostly the **text encoder**, not the denoise, and the denoise's
attention-output quantize goes through `h3_quantize_bf16_int8_head_major_kernel`
(24.2 ms), which this change does not touch. Reverted rather than kept for 0.2%
of e2e worth of extra branch.

### What is left of the glue

Scaled to a full run, after the fusion:

| Kernel | Full run | Fusable into |
|--------|---------:|--------------|
| `h3_int8_swiglu_quant` | 0.53 s | already fused |
| `h3_qkv_rope_coop` | 0.52 s | already fused |
| `h3_gate_adaln_quantize_int8` | 0.36 s | already fused |
| `h3_int8_apply_scales_bf16` | 0.24 s | attention out, FC2 |
| `h3_quantize_bf16_int8_head_major` | 0.09 s | the SDPA epilogue |

The remaining rescales are 0.24 s for two kernels' worth of work, so the
"fuse the glue" seam is close to exhausted — the estimate of 0.8–1.0 s for this
line of work was too high, and 0.35 s is what was there. e2e is **19.4 s**.

Logs: `/tmp/h3_perf_day14/qkv*.log`, `q-{vec,scalar}-*.log`, `vec.nsys-rep`.
---

## 2026-08-26 — REJECT further work on the DiT attention tile; it is not tensor-core bound

The 2026-08-26 TF32 entry called the attention kernel "1.5 s of attention still
at 31 TFLOP/s against a ~98 TFLOP/s BF16 peak, the largest single unclaimed gap
left in the pipeline". That framing is wrong, and it is why two earlier tile
rewrites (a 128-query MMA, a transposed V tile) both came back empty. The
number is right — `sdpa=1.512s` on a fox-fast profile, 3.06 ms per call for
100.8 GFLOP — but the ratio to the GEMM peak does not describe the kernel.

`ncu` cannot answer this on this machine: `/proc/driver/nvidia/params` has
`RmProfilingAdminOnly: 1`, so a non-root profile attaches, collects nothing and
exits without a diagnostic. Everything below is from `tools/h3_sdpa_bench.c`,
which runs `h3_gpu_sdpa_bf16` on fox-fast's own shape (1874 tokens, 56 heads,
d=128) and reproduces the pipeline exactly — 3.09 ms against nsys's 3.05 — with
a one-second edit-measure loop instead of a twenty-second one.

### Pricing each part by deleting it

Each row is a deliberately wrong kernel that keeps the work of interest and
drops one other thing, so the difference prices that thing:

| Ablation | Wall | Cost of the part |
|----------|-----:|-----------------:|
| baseline | 3.06 ms | |
| both MMA loops replaced by two float adds | 2.99 ms | **0.11 ms** |
| K/V loaded from tile 0 every iteration | 2.58 ms | 0.52 ms |
| V fragments as aligned 32-bit loads | 2.90 ms | 0.19 ms |
| V stored to shared without the BF16→F16 convert | 2.90 ms | 0.17 ms |

The first row settles it. Removing **both** matmuls saves 0.11 ms of 3.06, yet
100.8 GFLOP cannot cross the tensor cores in less than 1.03 ms at the ~98
TFLOP/s the GEMMs reach. So the MMA pipe is busy for at least a third of the
kernel and is almost entirely *hidden underneath* something else: there is a
~2.9 ms non-MMA critical path, and the tensor cores are idling inside it, not
limiting it. A faster matmul, a bigger tile or a better fragment layout all
attack the wrong term.

### Occupancy is not the term either

180 registers, 34,816 bytes of shared memory, 128 threads, no spills. Both
limits independently allow 2 blocks per SM — 8 warps of the 48 an sm_121 SM can
hold, 16.7%. That looks like the obvious answer, and the instruction count
agrees: the non-MMA work is ~0.6 ms at a perfect issue rate against 2.56 ms
measured, so ~23% issue efficiency, which is what 2 warps per scheduler with
long LDS and MUFU chains produces.

It is still not the answer. Shared memory is 683 bytes over the threshold for a
third block, so trimming the row padding plus `__launch_bounds__(128, 3)` buys
one, with **no spills**:

| Config | Registers | Spills | smem | Blocks/SM | Wall |
|--------|----------:|-------:|-----:|----------:|-----:|
| shipping | 180 | 0 | 34816 | 2 | 3.04–3.09 ms |
| `LD=132`, min 3 blocks | 168 | 0 | 33792 | **3** | 3.09 ms |
| `LD=132`, min 4 blocks | 128 | 348 B | 33792 | 4 | 7.81 ms |
| `LD=130` | 180 | 0 | 33280 | 3 | 3.35 ms |

50% more warps, no spills, no gain. Padding is the reverse: `LD=130` costs 10%
to bank conflicts, and `LD=136` is already the best of `{130,132,136,144}`.

So the costs are diffuse — nothing above 0.52 ms, most of it under 0.2 — over a
floor of 1.03 ms, and the two levers that usually move a kernel like this are
measured to do nothing. Attention is within roughly 1.5–2× of its structural
floor, and 15% of it is 0.2 s of a 19 s run. Everything else here is worth
more; the tile stays as it is. `tools/h3_sdpa_bench.c` stays too, so the next
attempt starts with a measurement instead of a rewrite.

Logs: `/tmp/h3_perf_day14/ncu_sdpa.txt` (the empty profile).
---

## 2026-08-26 — KEEP caching the DiT's quantized weights (load 2.68 → 1.64 s, and the variance collapses)

`load_core` reads `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1` and `mlp.fc2` as
BF16, quantizes each to INT8 plus a scale per row, and then
`release_block_bf16_after_int8` frees the BF16 — nothing else ever reads it.
Those four are essentially the whole warm load: 18.005 GiB of INT8 against the
34.145 GiB of BF16 that produced it, and the transform's only inputs are the
checkpoint's own bytes. Same shape as the AdaLN tables, so the same fix.

`dit-int8-<key>-b<NN>.bin` under `$H3_DIT_INT8_CACHE`, else
`$XDG_CACHE_HOME/h3-spark`, else `~/.cache/h3-spark`, holds one block's four
INT8 payloads and their scales, each section aligned to 4 KiB so the hit path
is the same staged parallel pread the checkpoint uses. The key hashes the
version, the three dimensions, and the path, size and mtime of the shard
holding `blocks.0.attn.qkv_proj.weight`, so a swapped checkpoint misses. A hit
skips the four BF16 reads, the three quantize kernels and the BF16 allocations,
and reads INT8 straight into the tensors the GEMM already wants; a truncated or
unreadable file falls back to the checkpoint instead of failing. The cache is
bypassed unless all three INT8 paths are on and no `keep_bf16_*` wants the
original. `H3_DIT_INT8_CACHE=off` disables it.

Because the file holds exactly what the quantizer produced, a hit is
**bit-identical**, not merely close: md5 `f5282774d3a4` on the cold write, on
both warm runs, on all four arms of the A/B and on all five steady-state runs.

### It moves more than the read

| | Cache off | Cache on |
|--|----------:|---------:|
| bytes staged in DiT load | 34.145 GiB | **18.005 GiB** |
| DiT load allocations | 50.600 GiB | **18.300 GiB** |
| DiT load wall | 2.681 s | **1.61–1.67 s** |
| DiT load read | 1.350 s | **0.49–0.51 s** |

### The page cache is the real story

The first interleaved A/B looked better than the truth — 21.8/22.1 s off
against 19.7/20.1 s on — and the reason is instructive. The hot files are 63 GB
of text encoder, 62 GB of transformer and 10 GB of VAE against 121 GB of RAM
and ~93 GB of usable page cache, so the working set does not fit and each run
evicts what the next one needs. Adding an 18 GiB cache file to that made the
`off` arm slower than it had been before the change (its DiT read went 1.35 →
3.5 s), and one `on` run paid 7.36 s for a text-encoder read that costs 2.9 s
when resident. Interleaving cannot measure this: each arm hands the next a
poisoned cache.

Run consecutively instead, and the arms separate for a better reason than
bytes. Warm, the 62 GB of BF16 transformer shards stop being read at all, which
takes the working set from ~135 GB to ~92 GB — under the limit — and everything
stops fighting:

| Five consecutive warm runs | e2e | DiT load | DiT read | TE read |
|---|----:|---------:|---------:|--------:|
| | 18.23 s | 1.666 s | 0.506 s | 3.122 s |
| | 18.00 s | 1.610 s | 0.505 s | 2.865 s |
| | 18.37 s | 1.639 s | 0.502 s | 2.940 s |
| | 18.42 s | 1.647 s | 0.504 s | 2.974 s |
| | 18.19 s | 1.666 s | 0.493 s | 2.930 s |

Against 19.29 s before the change, that is −1.1 s, but the spread matters as
much as the median: **±0.2 s where it used to be ±2 s**, and the text encoder's
read stops oscillating between 2.9 and 7.4 s even though nothing in the text
encoder changed. Fitting in RAM is worth more than it looks.

The cold run pays for it once: 43.4 s, with the DiT load at 26.6 s to read back
and write 18 GiB, and 18 GiB of disk. `tests/test_cuda_ops.c` gates the new
primitives — `h3_gpu_tensor_read_file_i8`, `..._read_file_f32`,
`..._read_i8` — on an exact multi-chunk round trip, since a hit is only
bit-identical if the payload is.

e2e is **18.2 s**, 3.1× off the 56.3 s this started at.

Logs: `/tmp/h3_perf_day14/q8_{cold,warm1,warm2}.log`, `q8ab/`.
---

## KEEP pooling the pinned staging slots across `h3_gpu` contexts

With the DiT's weights cached, the remaining phases were re-profiled against the
CUDA API trace rather than the kernel trace, and the largest single entry in the
run was not a kernel at all:

| CUDA API | total | calls | per call |
|---|----:|----:|----:|
| `cudaStreamSynchronize` | 4900 ms | 77 | — |
| `cudaEventSynchronize` | 3343 ms | 2488 | — |
| **`cudaMallocHost`** | **940 ms** | **16** | **58.8 ms** |
| `cudaMallocAsync` | 810 ms | 2526 | — |
| **`cudaFreeHost`** | **441 ms** | **16** | **27.6 ms** |

The two synchronize rows are the pipeline waiting for work it asked for. The
`cudaMallocHost` and `cudaFreeHost` rows are not: they are 1.38 s of page-locking
and unlocking, and the sixteen calls are four staging slots times four `h3_gpu`
contexts. A generation builds those contexts in sequence — text encoder, DiT,
audio VAE, video VAE — and each one page-locks 4 × 128 MiB at the ~1.5 GB/s the
kernel manages, then hands it back. The same half gigabyte gets pinned four
times and freed four times, and the phase profile had been reporting it all
along as an innocuous `pin=0.2s` on every line.

The slots are all `h3_stage_chunk_bytes()` and no two contexts hold them at
once, so a retired slot goes to a process-wide pool instead of `cudaFreeHost`,
and the next context takes it as-is. Only the first context of a run pays:

| | e2e | TE | DiT load | denoise | audio VAE | video VAE |
|---|----:|---:|---------:|--------:|----------:|----------:|
| per-context pinning | 17.98 s | 2.190 s | 1.469 s | 8.453 s | 0.853 s | 3.310 s |
| pooled slots | 16.76 s | 2.175 s | 1.330 s | 8.468 s | 0.611 s | 2.905 s |

`pin=` goes from five charges of ~0.25 s to one, denoise does not move (it never
loads anything), and the win lands where the loading is. Output md5 is
`f5282774d3a4` — the same bytes as before, because nothing about the data
changed, only who owns the buffer it lands in.

## KEEP timing cuBLASLt's candidates for the F32 GEMM

The biased F32 linear asked `cublasLtMatmulAlgoGetHeuristic` for one result and
used it. Asking the same call for `CUBLAS_COMPUTE_32F_FAST_16BF` instead of
`..._FAST_TF32` exposed the problem: cuBLAS returned *TF32* kernels either way —
still `s1688gemm`, the m16n8k8 instruction — but under the BF16 request it chose
128x256 and 128x128x32 tiles instead of 256x128 and 256x64, and ran 0.18 s
faster for identical output bits. Identical is expected: these GEMMs do not
split K, so tile shape cannot change the order a dot product accumulates in.

So the ranking, not the math, was wrong. The plan cache is per shape and the
video VAE uses each of its four shapes 144 times per decode, so timing six
candidates once and keeping the winner is cheap against getting the order wrong
144 times:

| video VAE | linear | phase |
|---|----:|----:|
| first heuristic result (`H3_DISABLE_LT_AUTOTUNE=1`) | 1.616 s | 3.041 s |
| fastest of six timed | 1.391 s | 2.835 s |

Do not read this off the e2e wall: at ±0.15 s of run-to-run spread the two arms
overlap, and the arm with the slower VAE measured the faster wall. The gpu-op
accumulators separate cleanly and repeatably (1.614/1.616/1.622 s against
1.559/1.371/1.391 s).

## REJECT BF16 operands for the video VAE's GEMM

TF32 is m16n8k8 where BF16 is m16n8k16, so feeding BF16 operands into the F32
accumulator should double the rate of the VAE's 1.45 s of GEMM. Casting the
weight once per buffer and the activation per call, it does, almost exactly:

| video VAE, nsys | GPU kernels |
|---|----:|
| TF32 | 2439 ms |
| BF16 operands (incl. 172 ms of casts) | 1900 ms |

And the phase got **slower**: 2.835 → 3.183 s, for output that is no longer
`f5282774d3a4`. Trading 539 ms of GPU time for 348 ms of extra wall is only
possible because the decode was never GPU-bound — the same reason the staging
pool was worth four times what any of this was. The BF16 weight copies also want
4.5 GiB that the phase did not previously need, taking its peak from 9.45 to
14.0 GiB, and a run that allocates them leaves the page cache poisoned enough
that the *next* run cost 28.9 s.

Getting the tensor cores' other half would mean the VAE holding BF16 weights
from the start, not casting F32 ones at the call site. That is a real change and
the numbers above say to price the decode's non-GPU 0.45 s first, because there
is no point halving 1.45 s of GEMM inside a phase that only yields 15 % of what
is taken off its GPU.

e2e is **16.7 s**, 3.4× off the 56.3 s this started at.

## The GPU is 84.5 % busy, and most of the idle is not for sale

The section above left `cudaStreamSynchronize` (4.9 s) and `cudaEventSynchronize`
(3.3 s) labelled as benign without measuring them, which is not the same as
knowing. An API total is not a cost: a sync that waits for work which had to
happen anyway is free. What costs is the GPU going idle while the host sits
inside a call. So rebuild the GPU's busy timeline from the nsys SQLite — every
kernel, memcpy and memset interval, merged — and attribute each idle gap to
whatever API call spanned it:

| | |
|---|----:|
| GPU span | 15.397 s |
| GPU busy | 13.012 s (84.5 %) |
| GPU idle | 2.384 s (15.5 %) |
| in 229 gaps >2 ms | 2.025 s |

| GPU idle | at | host was inside |
|---:|---:|---|
| 341.6 ms | 0.00 s | `cudaMallocHost` |
| 316.8 ms | 0.57 s | `cuLibraryLoadData` |
| 228.7 ms | 12.95 s | `cuLibraryLoadData` |
| 26.5 ms | 0.44 s | `cudaEventSynchronize` |
| 14.4 ms | 0.51 s | `cudaEventSynchronize` |

The two synchronize rows really are benign, but for a reason worth stating:
aggregating all 229 gaps by the API that spans them puts 147.7 ms of idle on
`cudaEventSynchronize` and none on `cudaStreamSynchronize`. Eight seconds of API
time against 148 ms of consequence is what a correctly-placed wait looks like.
(Reading only the largest gaps understates this at 41 ms; the aggregate is the
number to quote.)

The same aggregate names the real host-side costs, and launch latency is not
among them:

| API spanning the gap | GPU idle | gaps |
|---|----:|----:|
| `cudaMallocAsync` | 665.1 ms | 110 |
| `cuLibraryLoadData` | 574.2 ms | 5 |
| `cudaMallocHost` | 341.6 ms | 1 |
| `cudaMemcpyAsync` | 214.9 ms | 70 |
| `cudaEventSynchronize` | 147.7 ms | 33 |

Device allocation, not kernel launching, is what the GPU waits on most after the
two items already priced and rejected below.

### REJECT priming the memory pool

The 665 ms on `cudaMallocAsync` localizes precisely. All 110 stalls fall between
t=2.70s and t=3.63s, and the last kernel to run before each one is the RoPE
setup's cast — the only cast in that window, and a one-shot. So the picture is
not a cast that allocates repeatedly; it is the DiT's load allocating its 109
weight tensors back to back, 18.3 GiB in all, with no kernel in between for the
GPU to run.

The pool's release threshold is already 24 GiB, but that only keeps blocks that
have been *freed*. A fresh process starts with an empty pool, so the DiT's first
18 GiB comes from the driver one tensor at a time. Consolidating that into a
single allocation up front, immediately freed, should leave the memory parked in
the pool for the per-tensor requests to hit.

It does exactly that, and it buys nothing:

| prime | wall | text encoder | DiT load |
|---|---:|---:|---:|
| off | 15.90, 15.94, 15.88 | 2.181s | 1.227s |
| 17 GiB, background thread | 15.92, 15.90, 15.99 | 2.515s | 0.931s |

DiT load gives up 0.29s every run and the text encoder takes on 0.31s every run.
Chunking the prime into 512 MiB and 256 MiB pieces, on the theory that one giant
mapping holds the pool's lock long enough to stall the encoder's own
allocations, changed nothing: 15.80 and 15.94, with the same 0.33s handed to the
encoder. A synchronous prime at context creation moves the same cost outside the
named phases instead, where it shows up as 0.47s of startup against 0.58s of
phase gains.

Three placements, one result — and the reason is not the one to reach for first.
It is not bandwidth. Device allocation is *serialized against host-to-device
copies* on this platform, and a weight load is nothing but host-to-device
copies. See the section below, which measures it; the explanation that used to
sit here blamed memory bandwidth and was wrong.

### REJECT pipelining the DiT load's allocations

Since priming only relocates the cost, the obvious next move is to overlap
instead of pre-pay: allocate block N+1 while block N is still being read. A
microbenchmark says that should be free — mapping 12.5 GiB takes 0.446 s alone
and 0.470 s while four threads read a weight file, a 1.05× slowdown for
something that hides 100% of the reads.

It is not free, and the microbenchmark was measuring the wrong thing. The
loader's reads are not file reads; each one lands in the device through
`cudaMemcpyAsync`. Asking the same question about *that*:

| mapping 12.5 GiB in 100 allocations | time | vs alone |
|---|---:|---:|
| alone | 0.446 s | |
| beside 4 threads reading a weight file | 0.470 s | 1.05× |
| beside continuous H2D `cudaMemcpyAsync` | 5.938 s | **13.31×** |

Allocation drops from 30 GB/s to 2.3 GB/s the moment copies are in flight. That
single fact explains every attempt in this section:

- The prime during the text encoder phase contended with the encoder's copies,
  so it gave back exactly what it saved.
- A prefetch thread allocating on the loader's own stream was worse still
  (DiT load 1.16 s → 1.81 s), because `cudaMallocAsync` is stream-ordered: it
  queued behind the copies rather than running beside them.
- Giving the prefetch thread its own stream removed that regression and left
  DiT load unchanged at 1.27 s, which is the honest answer — the contention is
  per-context, not per-stream.

So the serial alternation the loader already does is not a missed opportunity,
it is the shape the platform forces. The 665 ms is only reducible by mapping
fewer bytes, not by mapping them at a better time. Worth remembering before the
next "the GPU is idle here" reading: idle that coincides with allocation is not
idle waiting to be filled.

One measurement note that cost real time here. The first probe used plain
`read()` for the concurrent load and reported that overlap was free. Anything
claiming to model the loader has to include the loader's CUDA calls, not just
its file I/O — the two answers differ by 13×.

`cuLibraryLoadData` is 545 ms over 21 calls, matching the two bursts almost
exactly, and it is not our code: the binary carries one `sm_121` cubin at 1.5 MB
total, so this is cuBLAS lazily loading its own kernel modules — once at the
first GEMM, once more when the VAE asks for shapes nothing has used yet.

**Which is a trap, and `CUDA_MODULE_LOADING=EAGER` is the way out of it.** Eager
loading moves every module to context creation, so if those 545 ms were on the
critical path the phases would get shorter:

| | e2e | TE | video VAE |
|---|----:|---:|----------:|
| `LAZY` (default) | 16.52 s | 2.152 s | 2.836 s |
| `EAGER` | 17.48 s | 2.156 s | 2.799 s |

The text encoder does not move at all and the VAE gives back 37 ms of 228,
while e2e loses a second to loading modules the run never calls. The bursts sit
inside phases that are waiting on file reads, so the module load was already
free — it shows up as GPU idle without being wall. **GPU idle is not recoverable
wall**, and 545 ms of it here buys 37 ms.

That leaves the 341.6 ms of first-context pinning, which the pool cannot remove
because nothing can be read until a buffer exists to read into. Pinning less is
the obvious lever and it is a wash, because chunk size sets read efficiency too:

| slot size | pin | TE read | DiT read | e2e (3 runs) |
|---|----:|--------:|---------:|-------------:|
| 32 MiB | 0.078 s | 3.647 s | 0.653 s | 16.51 s |
| 64 MiB | 0.151 s | 3.863 s | 0.614 s | 16.51 / 16.48 / 16.45 s |
| 128 MiB (default) | 0.458 s | 2.916 s | 0.514 s | 16.47 / 16.46 / 16.52 s |
| 256 MiB | 0.478 s | 2.965 s | 0.507 s | 16.95 s |

64 MiB pins 0.3 s faster and reads 0.9 s slower, and the medians land 10 ms
apart. Keep 128 MiB.

So the remaining 1.5 s of idle is 220-odd gaps of 2–10 ms, which is launch
latency and per-op overhead with no single lever behind it. The host-side
surface that the staging pool came out of is now closed; what is left is either
serial by construction or already hidden behind I/O.

One thing this did settle: the candidate timing added two module loads and no
time (544.8 ms / 21 calls against 553.9 ms / 19 with `H3_DISABLE_LT_AUTOTUNE=1`),
so it is not paying for itself in `cuLibraryLoadData`.

Warm steady state after these runs is **16.5 s**.

## KEEP widening the F32 attention's row tile to eight warps

The video VAE's attention was the one line item left unexplained: 615 ms over
144 calls, and an earlier attempt to price it derived 98 TFLOP/s, which F32 data
cannot reach. That arithmetic was wrong because the shape was. The decoder works
in `TILE_PIXELS = 256` tiles at `SPATIAL_RATIO = 16`, so a 512×512 frame is 2×2
tiles of 18×18 latents, and the sequence is 7 × 18 × 18 + `SUFFIX` = **2273**,
not the 7173 a full-frame latent would give. 36 layers × 4 tiles = the 144 calls.
`grid=36x32` from the kernel trace confirms it: 36 row blocks of 64, 32 heads.

Three ablations, cheapest first. **The MMA count is free.** Q·K is
hi·hi + hi·lo + lo·hi and P·V is the same — the comment claiming P·V stays single
FP16 is stale — so dropping two of Q·K's three halves the score matmul:

| | sdpa | phase |
|---|----:|----:|
| as shipped | 0.622 s | 2.788 s |
| two of three Q·K MMAs removed | 0.625 s | 2.790 s |
| whole hi/lo scheme removed, both matmuls | 0.604 s | 2.816 s |

Removing the *entire* split — every residual pack, four of six MMAs — buys 32 ms
of 636. So **you cannot buy speed with decode quality here**: ~22 bits of score
precision costs 5 %, and that line of enquiry is closed, not deferred.

What did cost was traffic. Every block walks its head's whole K and V, so the
row tile sets how many times that is paid: at M=64 a 2273-row head takes 36
blocks, and a warp owns 16 rows either way, so a wider tile costs warps rather
than registers or shared memory. Eight warps halves the block count:

| | sdpa | phase | e2e |
|---|----:|----:|----:|
| 4 warps, M=64 | 0.635 / 0.633 s | 2.813 / 2.816 s | 16.47 s |
| 8 warps, M=128 | 0.320 / 0.320 s | 2.536 / 2.528 s | 16.25 s |

**1.98×, and bit-identical** — `f5282774d3a4` either way, because grouping more
rows into a block changes no row's accumulation order. Q's staging has to be
re-laid-out for it: it borrowed the V and low-K tiles, which only hold 64 rows,
so the four tiles become one carved buffer and Q takes it in halves.

Two traps worth recording. The first measurement of this said 0.631 vs 0.630 s —
a clean null — because `make h3 NVCC_EXTRA=-DH3_MMA_F32_WARPS=8u` does not
rebuild `h3_gpu.o` when only the flag changed, so both arms ran the 4-warp
binary. Checking `blockX` in the kernel trace is what caught it; **an ablation
that changes launch geometry has to be verified in the trace, not the wall.**
The second: 12 and 16 warps do not build. Q's two halves want
2 × ROWS × LD, which passes 48 KB of static shared memory at 12 warps
(`uses too much shared data (0xd800 bytes, 0xc000 max)`), and a failed build
silently leaves the previous binary in place to be measured again.

So eight warps is the ceiling until Q stops being staged through shared memory
at all — each thread's fragment rows are known, so it could load them straight
from global and leave the buffer to K/V, which would allow 16 warps and halve
the traffic again. That is the open end here.

e2e is **16.25 s**, 3.5× off the 56.3 s this started at.

## KEEP a 4×4 register block in the audio VAE's Conv1d

This kernel had never been looked at, and pricing it before touching it was the
whole trick. Instrumenting the launch gives the 122 real shapes — seven stages of
`b=2` with channels doubling from 8 to 512 as the length halves from 29600 to 185,
kernels of 3, 7 and 11 — and from those the roofs:

| | |
|---|----:|
| total work | 85.3 GFLOP |
| measured | 477.7 ms |
| achieved | **0.18 TFLOP/s** against an FP32 roof of ~31 |
| traffic if each value were read once | 0.65 GB, i.e. 1.4 GB/s of ~273 |
| loads actually issued | 341 GB — **525×** the ideal |
| floor if compute-bound (31 TFLOP/s) | 2.8 ms |
| floor if traffic-bound (250 GB/s) | 2.6 ms |

Both roofs say ~3 ms and the kernel took 478. Nothing about the convolution is
expensive; the schedule was. One thread per output element means two global loads
per FMA — 85.3 G loads, about 178 G/s, near the rate the device can issue them —
and consecutive lanes walk output channels while the weight index strides by
`ic*k`, so each 4-byte weight drags in its own 32-byte sector.

Both problems are the same problem: nothing is reused. Giving each thread a 4×4
block of (time, output channel) makes one weight load feed four outputs and one
input load feed four channels, taking loads per FMA from 2 to 0.5:

| | conv | phase | e2e |
|---|----:|------:|----:|
| one output per thread | 0.498 s | 0.614 s | 16.10 s |
| 4×4 register block | 0.178 s | 0.294 s | 15.80 s |

**2.8×, bit-identical.** Exactness is not luck: the accumulation stays `fmaf`
over `ic` then `k` ascending, so each output sums in the order it always did.
Out-of-range taps now multiply by a zeroed operand instead of skipping, which is
the same bits for finite weights.

Bigger blocks are worse — 8×4 and 4×8 lose 0.013 s, 8×8 loses 0.073 s — because
blocking pays 16× parallelism for its reuse and the last stages are only 8 or 16
channels wide. All four configurations produce identical output, which is the
check that blocking is not what makes it exact.

At 0.178 s this is still 60× off the roof and still load-bound, now at ~120 G
loads/s.

### KEEP letting the gate kernel apply the projection's INT8 rescale

Two of the block's four INT8 projections still ended in a separate rescale pass.
`h3_int8_apply_scales_bf16_kernel` runs 270 times in a four-step generation for
69.6 ms — exactly twice per block-step, which is `attn.out_proj` and FC2, the two
that were never folded into a consumer the way QKV and FC1 were.

Both feed the same consumer, `h3_gate_adaln_quantize_int8`, so one fusion covers
them. The rescale is pure bandwidth: 4 bytes of int32 accumulator in, 2 bytes of
BF16 out, 257 µs per call for 64.5 MB, which is 250 GB/s and therefore the whole
cost. The gate kernel then reads those same 2 bytes straight back. Letting it
read the accumulator instead and scale inline removes 6 bytes per element and
adds 2, so two thirds of the pass should disappear:

| | apply_scales | gate kernel | net |
|---|---:|---:|---:|
| split | 69.6 ms / 270 | 95.0 ms / 258 | |
| fused | 3.0 ms / 12 | 116.9 ms / 258 | −44.7 ms |

Scaled to twenty steps that is −164 ms of kernel time against a −169 ms
prediction from the bandwidth arithmetic alone. The denoise phase gives back
0.105 s of it (8.291 s → 8.186 s over paired runs); the rest is latency the
phase was already hiding.

Bit-identical because the fused path reproduces the rounding rather than
skipping it: it computes `bf16(accum * input_scale * weight_scale)` in that
multiplication order and converts back to float, which is precisely what the
separate kernel wrote to memory. The twelve surviving calls are the last block
of each pass, whose gate is not the quantizing one.

Two things this cost that are worth writing down. The producer defers by *not*
writing its BF16 output, so a kill switch that disables only the consumer leaves
the gate reading a buffer nobody wrote — the first A/B produced a different md5
for exactly that reason. One switch has to govern both ends. And because the
deferral is only valid until the next INT8 GEMM reuses the shared accumulator,
the record is cleared on entry to every GEMM and the consumer checks that it
matches its own shape, so a consumer that asks to fuse when the producer took a
different path falls back to reading BF16 instead of reading stale memory.

### KEEP transposing the Conv1d weight to `[ic][k][oc]`

The register-blocked kernel's input read is a warp-wide broadcast, but its
weight read is not. Consecutive threads walk output channels, and in the
checkpoint's `[oc][ic][k]` layout those sit a full `ic*k` stride apart, so every
lane pulls its own 32-byte sector to use 4 bytes of it. That eightfold waste is
not part of the 341 GB — it *is* the 341 GB: `8 × Σ(out_len × oc × ic × k)`
comes to 341 GB exactly, so the entire excess traffic is this one access
pattern.

Transposed to `[ic][k][oc]` the output channels are adjacent, one `float4` per
thread covers the block's four, and a warp's loads coalesce into contiguous
lines. Same values, same `fmaf` order over ic then k, so only the address
arithmetic moves:

| | conv | audio VAE phase |
|---|---:|---:|
| `[oc][ic][k]` | 0.179, 0.178 s | 0.298, 0.295 s |
| `[ic][k][oc]` | 0.092, 0.093 s | 0.208, 0.213 s |

1.93×, and md5 unchanged.

The transpose is rebuilt per call into one grown-on-demand scratch buffer, which
is not the obvious choice and is the point worth recording. Keying a cache on
the weight's device pointer looks free and is wrong: the VAE streams its
weights, so the allocator hands a later layer the address a freed one had, and
the cache returns a filter transposed for a different shape. That is what
happened — 127 of 128 convolutions matched bit for bit and the last one differed
in every single element. Rebuilding costs two passes over ~46 MB across the whole
decode, well under a millisecond, against the 86 ms the layout wins.

Worth noting how cheaply this was caught: running both kernels into separate
buffers and printing per-call shape and mismatch count named the failing
convolution and its exact shape on the first try, where the md5 alone said only
that something among 128 calls was wrong. The unit tests passed throughout —
their conv shapes are too small to reach the blocked path at all.

e2e is **15.5 s** (15.40, 15.49, 15.57), 3.6× off the 56.3 s this started at. The
two fusions above are worth 0.19 s between them at the phase level, and the e2e
mean moved from 15.91 s to 15.72 s — the same 0.19 s. Run-to-run spread is now
wider than either individual change, which is why both were accepted on paired
phase measurements rather than on wall time.
