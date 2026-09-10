# Quality vs speed: how to pick generate knobs

Numbers below are **DGX Spark (GB10), v0.2.1**, seed 42, `--profile`.
PSNR/SSIM are ffmpeg lavfi vs the **same prompt/size/seed without the speed
flag**. They are not a claim about “good video” in absolute terms — only how
far you moved from this port’s quality path.

Default `./h3` (no extra flags beyond size/prompt) is the **quality path**.
Do not turn on speed flags if you need that look, or a bit-identical replay
of the HIP-page clips.

## What actually costs time

| Knob | What it changes | When it matters |
|---|---|---|
| `--width` / `--height` / `--seconds` | DiT sequence length **N**. Attention is **N²** | **15 s / 864×480** is ~78% SDPA. This is the long-video wall |
| `--steps` | Denoise passes (default **20**) | Fewer steps = fewer DiT evals, more under-denoised frames |
| `--layers` | DiT blocks: **50** exact, **45** fast, **40** aggressive | 40 is a bigger quality hit than reuse 3; prefer reuse first |
| `--reuse` | How often a denoise step **reuses** the last DiT residual: **1** close, **2** fast (default for fox-fast/15 s), **3** aggressive | Cuts **eval count**, not N. fox-fast 11→8 evals; 15 s same |
| `--token-reduction` | Pools adjacent **horizontal video tokens** in middle blocks | Cuts **N** on those blocks. Big on long T2VA, modest on 512² 22f |
| `H3_INT8_VAE=1` | INT8 video-VAE weights | **VRAM**, not wall clock (fox-s2 peak 9.45→2.73 GiB) |

Leave `--core-reuse` alone unless you already know that knob. `H3_BF16_MLP=1`
is slower **and** fails the 24 dB floor — do not use it for speed.

## Recipes (start here)

Same 20-step, 45-layer baseline as the showcase / HIP page. Only the speed
flag changes.

### Short clip (512², ~22 frames) — fox-fast class

Warm GB10, steps 20, L45.

| Intent | Flags | Wall (this box) | vs quality-path pixels |
|---|---|---:|---|
| **Ship / reference** | `--layers 45 --reuse 2` | **15.6 s** | md5 `f5282774d3a4` |
| **Faster, still watchable** | `--layers 45 --reuse 3` | denoise **5.9 s** (e2e ~13 s) | **PSNR 20.9 / SSIM 0.75** |
| Draft / “is the motion right?” | `--token-reduction` | denoise ~6.1 s | **PSNR 17.8 / 0.72** (worst frame ~15 dB) |
| Smoke the pipeline | `--steps 2 --layers 35 --reuse 1` (fox-s2) | **8.2 s** | not a quality preset; wall is VAE+Qwen |

On a short clip **`--reuse 3` beats token-reduction**: similar or better speed,
~3 dB more PSNR, and a much higher worst-frame floor (20.6 vs 15.2).

Do **not** stack `--reuse 3` and `--token-reduction` on short clips. That
combo is **PSNR 17.9 / 0.72** — token-reduction quality at reuse-3 speed, so
you paid the uglier tax for little extra.

### Long clip (864×480, `--seconds 15`) — cinematic class

HIP-page office prompt, L45, steps 20. Quality path **1076 s**.

| Intent | Flags | Wall | vs quality-path `60fd70cc309c` |
|---|---|---:|---|
| **Final / publish** | `--layers 45 --reuse 2` | **18 min** | bit-stable quality path |
| **Faster, keep more structure** | `--reuse 3` (keep L45) | **13 min 25 s (−25%)** | **PSNR 18.8 / 0.70** |
| **Need the minutes back** | `--token-reduction` | **11 min 14 s (−37%)** | **PSNR 17.8 / 0.66** (Y ~16.3) |

Long video is N². Token-reduction shrinks N; reuse 3 only drops 11→8 evals, so
**TR is the long-T2VA speed knob**, reuse 3 is the **less-ugly** knob.

~18 dB vs the quality path is a **visible** hit (soft fur, edges, luma). It
is not a “slight” trade. Use it when wall clock matters more than matching the
reference clip. Do not use TR when you need the HIP-page md5.

### Iterate, then polish

1. fox-s2 or a small `--frames` to check prompt and composition.
2. Short fox-fast **without** TR to judge look.
3. If the 15 s encode is too slow, generate a **`--reuse 3`** 15 s first.
4. Only add `--token-reduction` if 13 minutes is still too long.
5. Final delivery: drop the speed flags (`--reuse 2`, no TR).

## How to read the quality numbers

These are **vs this repo’s quality path**, not vs a camera.

| PSNR vs quality path | What we treat it as |
|---|---|
| inf / SSIM 1.0 | Default KEEP (bit-identical or equivalent) |
| **≥ 24 dB and SSIM ≥ 0.85** | Allowed as a **default** kernel change |
| ~21 dB / 0.75 | `--reuse 3`: obviously different, usually still “the same shot” |
| **≤ 18 dB / SSIM ~0.66–0.72** | TR: luma/detail collapse; draft or time-critical only |

A VAE-tile experiment at **26.8 dB / SSIM 0.80** was still REJECT as default
because seams were visible. Speed flags that land at 18 dB are **opt-in**,
never default.

## Resolution and duration

- Prefer **`--seconds`** for 24 fps duration instead of guessing `--frames`.
- 512² 22f is a **DiT-cheap** preset (seq ~1874). 864×480 15 s is seq ~44800;
  almost all extra time is attention, not VAE.
- `--render-width` / `--render-height` lower the **internal** generate size.
  Use them only when you understand the upsample; they are not a free quality
  win.

## Memory

`H3_INT8_VAE=1` is for **peak VRAM**, not a 15% e2e win. fox-s2 VAE peak
**9.45 → 2.73 GiB**, PSNR vs F32 VAE **43 dB**. Leave it off unless you are
tight on memory.

## Command sketches

```bash
MODEL=/path/to/MiniMax-H3

# Quality (showcase / HIP-page fox-fast)
./h3 -d "$MODEL" -p "$PROMPT" \
  --width 512 --height 512 --frames 22 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  -o out-quality.mp4

# Short clip, faster
./h3 -d "$MODEL" -p "$PROMPT" \
  --width 512 --height 512 --frames 22 \
  --steps 20 --layers 45 --reuse 3 --seed 42 \
  -o out-faster.mp4

# 15 s, less-ugly speed
./h3 -d "$MODEL" -p "$PROMPT_15S" \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 3 --seed 42 \
  -o out-15s-reuse3.mp4

# 15 s, maximum wall-clock cut (visible quality loss)
./h3 -d "$MODEL" -p "$PROMPT_15S" \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  --token-reduction \
  -o out-15s-tr.mp4
```

Advanced TR tuning (`H3_TOKEN_REDUCTION_BLOCKS`, `H3_TOKEN_REDUCTION_EARLY`)
exists but milder ranges on fox-fast never reached 19 dB and were **slower
than `--reuse 3`**. Prefer reuse 3 before inventing a custom TR band.

Dated tables: [`PERF_BASELINE.md`](PERF_BASELINE.md).
