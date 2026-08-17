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

*Baseline frozen 2026-08-17. Append new dated sections when a material speedup
lands; do not overwrite this table without noting the old numbers.*
