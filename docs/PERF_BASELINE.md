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
