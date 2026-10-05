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
| all-opt | `--layers 45 --reuse 3 --sol-attn`, `H3_SOL_ATTN_TAU=1` | 18 | **180.8 s** | 144.6 s (64 / 61) | 26.5 s | `8119514ece0e` |

Default CUDA kernels stay on in both runs (INT8 MLP, INT8 QKV, MMA SDPA).
`--token-reduction` is not stacked with `--sol-attn`. Qwen was about 5.5 s.
Each figure is one shot, not a three-run median. Weight reads were not fully hot;
that is a few seconds of load, not the denoise gap.

all-opt vs our own exact clip: **PSNR 12.8 dB / SSIM 0.58**. That is a
different picture, below the repo KEEP line (24 dB / 0.85). Their published
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
| H3-OnDevice + kernel + Sol-Attn + First Block Cache | **181.3 s** | **3.92×** |
| **h3-spark.c all-opt** (45 layers, reuse 3, Sol-Attn τ=1) | **180.8 s** | **3.93×** |

641.9 s is already past their eager baseline because this tree is not PyTorch
eager. 180.8 s matches their full stack on the clock. The mechanisms are not
the same: they keep a 50-layer net and skip steps with First Block Cache; we
drop to 45 layers and evaluate 18 of 50 steps on a fixed stride.

Video VAE stays ~26.5 s once denoise shrinks, about 15% of the 181 s.
