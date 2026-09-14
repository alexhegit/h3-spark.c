# Quality vs speed: how to pick generate knobs

Numbers below are **DGX Spark (GB10), v0.2.2**, seed 42, `--profile`.
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
| `--sol-attn` | Sparse SDPA: keep important 64-token KV tiles, approximate the rest | Cuts **attention traffic**. Kernel **3×** at seq 44800; fox-fast **~17.6 dB** (KEEP fail). Long T2VA only |
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
| **Faster, keep more structure** | `--sol-attn` | **11 min 35 s (−35%)** | **PSNR 19.2 / 0.72** |
| **Faster, reuse only** | `--reuse 3` (keep L45) | **13 min 25 s (−25%)** | **PSNR 18.8 / 0.70** |
| **Need the minutes back** | `--token-reduction` | **11 min 14 s (−37%)** | **PSNR 17.8 / 0.66** (Y ~16.3) |

Long video is N². `--sol-attn` cuts attention tiles; `--token-reduction`
shrinks N; reuse 3 only drops 11→8 evals. On this 15 s clip **Sol-Attn
beats reuse 3 on both wall and PSNR**, and beats TR on PSNR at almost the
same speed.

All three speed flags also move the **audio**, even though nothing sparsifies
or pools audio tokens directly: video hidden states feed the audio branch
through cross-modal attention. Waveform SNR vs the quality-path track on this
clip is `--sol-attn` **8.6 dB**, `--token-reduction` 3.1 dB, `--reuse 3`
2.6 dB. If the soundtrack matters, use the quality path. Separately, H3 audio
is inherently dull — the quality path puts only ~1% of its energy above
4 kHz — so a muffled soundtrack is usually the model, not a flag.

~18–19 dB vs the quality path is a **visible** hit (soft fur, edges, luma).
It is not a “slight” trade. Do not use these flags when you need the
HIP-page md5.

### Iterate, then polish

1. fox-s2 or a small `--frames` to check prompt and composition.
2. Short fox-fast **without** TR to judge look.
3. If the 15 s encode is too slow, generate with **`--sol-attn`** first.
4. Only add `--token-reduction` if you still need the last ~20 s and accept
   the extra quality drop.
5. Final delivery: drop the speed flags (`--reuse 2`, no TR / sol-attn).

## How to read the quality numbers

These are **vs this repo’s quality path**, not vs a camera.

| PSNR vs quality path | What we treat it as |
|---|---|
| inf / SSIM 1.0 | Default KEEP (bit-identical or equivalent) |
| **≥ 24 dB and SSIM ≥ 0.85** | Allowed as a **default** kernel change |
| ~21 dB / 0.75 | `--reuse 3`: obviously different, usually still “the same shot” |
| ~19 dB / 0.72 | `--sol-attn` on 15 s: visible, but the best long-T2VA speed/quality point measured here |
| **≤ 18 dB / SSIM ~0.66–0.72** | TR: luma/detail collapse; last-resort minutes |

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

# 15 s, recommended speed flag (visible vs quality path, best PSNR of the fast set)
./h3 -d "$MODEL" -p "$PROMPT_15S" \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  --sol-attn \
  -o out-15s-sol-attn.mp4

# 15 s, reuse only (slower than sol-attn, similar look)
./h3 -d "$MODEL" -p "$PROMPT_15S" \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 3 --seed 42 \
  -o out-15s-reuse3.mp4

# 15 s, maximum wall-clock cut (uglier than sol-attn)
./h3 -d "$MODEL" -p "$PROMPT_15S" \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  --token-reduction \
  -o out-15s-tr.mp4
```

Do **not** stack `--sol-attn` with `--token-reduction` unless you are
measuring that pair. Advanced TR tuning (`H3_TOKEN_REDUCTION_BLOCKS`,
`H3_TOKEN_REDUCTION_EARLY`) exists but milder ranges on fox-fast never
reached 19 dB and were **slower than `--reuse 3`**. Sol-Attn knobs:
[`SOL_ATTN.md`](SOL_ATTN.md).

Dated tables: [`PERF_BASELINE.md`](PERF_BASELINE.md).
