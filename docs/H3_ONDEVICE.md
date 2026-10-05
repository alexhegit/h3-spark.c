# H3-OnDevice comparison (DGX Spark, 2026-10-05)

Same job as NVIDIA's
[H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/) Spark
column: stock MiniMax-H3 weights, **832×480**, 24 fps, **124 frames** (5.17 s
from `--seconds 5`), **50 scheduler steps**, T2VA with audio. No FastH3 LoRA.

Script: [`scripts/h3_ondevice_5s.sh`](../scripts/h3_ondevice_5s.sh).
Logs and MP4s: `/tmp/h3_ondevice/`.

Their 710.6 s column is a **PyTorch eager** reference on the unmodified BF16
checkpoint, before kernel fusion, Sol-Attn, and cross-step cache. Their
181.3 s column is that stack added on top (Sol-Attn **τ=1**, First Block
Cache). Page note: warm, batch 1.

## What we ran

Prompt is the sentence the page prints. The page marks it `[full prompt compacted]`,
so the real prompt and seed are unpublished. These clips are **not** their
sample frames. Do not compute PSNR against their videos.

| Run | Flags | Evals | E2E | Denoise (sdpa / linear) | Video VAE | md5 prefix |
|---|---|---:|---:|---|---:|---|
| exact | `--layers 50 --reuse 1` | 50 | **641.9 s** | 598.2 s (351 / 188) | 26.7 s | `7e2a92b49a3d` |
| fbc | `--fbc` (50 layers, reuse 1) | 16 full / 34 skip | **236.9 s** | 200.1 s (117 / 62) | 26.6 s | `70ca55e095f8` |
| fbc+sol | `--fbc --layers 45 --sol-attn`, `H3_SOL_ATTN_TAU=1` | 16 full / 34 skip | **169.6 s** | 137.4 s (62 / 56) | 26.1 s | `1cf883e369bb` |
| all-opt | `--layers 45 --reuse 3 --sol-attn`, `H3_SOL_ATTN_TAU=1` | 18 | **180.8 s** | 144.6 s (64 / 61) | 26.5 s | `8119514ece0e` |

`--fbc` is off by default. It runs block 0 every step and skips the rest when
that residual's relative L2 stays under `H3_FBC_REL` (default 0.10). The first
and last 4 steps stay dense (`H3_FBC_WARMUP`, `H3_FBC_TAIL`), and at most 4
steps are skipped in a row (`H3_FBC_MAX_STREAK`). On this 50-step job the
streak cap is what refreshes the cache: middle-step relative L2 sat around
0.014–0.03, under 0.10. A skip still runs block 0, so 16 full / 34 skip is
834 attention calls at 50 layers and 754 at 45 layers.

Default CUDA kernels stay on (INT8 MLP, INT8 QKV, MMA SDPA).
`--token-reduction` is not stacked with `--sol-attn` or `--fbc`. Qwen was
about 5.5 s on the first three runs. The fbc+sol run saw 2.1 s because the
text encoder was already hot; that is about 3 s of its wall-clock gap, not a
denoise change. Each figure is one shot, not a three-run median. Weight reads
were not fully hot; that is a few seconds of load, not the denoise gap.

Against our own exact clip:

| Run | PSNR | SSIM |
|---|---:|---:|
| fbc | **18.7 dB** | **0.73** |
| fbc+sol | 12.6 dB | 0.58 |
| all-opt | 12.8 dB | 0.58 |

All three are below the repo KEEP line (24 dB / 0.85). fbc+sol is faster than
all-opt (169.6 s vs 180.8 s; denoise 137.4 s vs 144.6 s) and does not improve
on 12.8 dB. The quality win is `--fbc` alone, at 236.9 s. Their published
full-stack figure is **LPIPS 0.293** vs their eager reference. Different
metric, different frames. Wall-clock proximity is not a quality match.

## Apple-to-apple wall clock

Same GPU class, resolution, frame count, and nominal 50 steps. Recipes differ.

| Stack | E2E | vs their eager 710.6 s |
|---|---:|---:|
| H3-OnDevice, PyTorch eager | 710.6 s | 1.00× |
| H3-OnDevice + kernel fusion | 511.8 s | 1.39× |
| H3-OnDevice + Sol-Attn τ=1 | 451.2 s | 1.57× |
| **h3-spark.c exact** (50 layers, reuse 1) | **641.9 s** | **1.11×** |
| **h3-spark.c `--fbc`** (50 layers, dense attention) | **236.9 s** | **3.00×** |
| H3-OnDevice + kernel + Sol-Attn + First Block Cache | **181.3 s** | **3.92×** |
| **h3-spark.c all-opt** (45 layers, reuse 3, Sol-Attn τ=1) | **180.8 s** | **3.93×** |
| **h3-spark.c `--fbc` + 45 layers + Sol-Attn τ=1** | **169.6 s** | **4.19×** |

641.9 s is already past their eager baseline because this tree is not PyTorch
eager. 180.8 s matches their full stack on the clock with a different recipe:
45 layers and 18 fixed-stride evaluations. `--fbc` is the closer mechanism
(50 layers, block-0 residual, early and late steps dense) and lands at 236.9 s
with 18.7 dB against our exact clip. Stacking it with 45 layers and Sol-Attn
reaches 169.6 s, ahead of both 180.8 s and their 181.3 s on the clock, at
12.6 dB. That is the same quality class as all-opt, not a match to their
LPIPS figure.

Video VAE stays ~26.5 s once denoise shrinks, about 15% of the 181 s.
