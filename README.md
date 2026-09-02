# h3-spark.c

**v0.2.0** — NVIDIA DGX Spark (GB10) CUDA port of
[antirez/h3.c](https://github.com/antirez/h3.c). MiniMax-H3 inference with the
same CLI and model stack; the GPU backend is CUDA.

**Original project:** [antirez/h3.c](https://github.com/antirez/h3.c)  
**Official weights:** [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)  
**Changelog:** [`CHANGELOG.md`](CHANGELOG.md)

## Showcase (DGX Spark)

Clips below were generated on NVIDIA DGX Spark with this CUDA port
(512×512, 22 frames, `--steps 20 --layers 45 --reuse 2`). Click a poster for
the MP4.

| Mode | Sample |
|------|--------|
| **T2VA** — text → video+audio | [![T2VA fox](assets/showcase/t2va-fox-fast.jpg)](assets/showcase/t2va-fox-fast.mp4) [mp4](assets/showcase/t2va-fox-fast.mp4) |
| **Ref2VA** — `--ref-image` | [![Ref2VA](assets/showcase/ref2va-person-lamb.jpg)](assets/showcase/ref2va-person-lamb.mp4) [mp4](assets/showcase/ref2va-person-lamb.mp4) |
| **FL2VA** — `--first-frame` + `--last-frame` | [![FL2VA](assets/showcase/fl2va-family-dinner.jpg)](assets/showcase/fl2va-family-dinner.mp4) [mp4](assets/showcase/fl2va-family-dinner.mp4) |

### Reproduce the showcase clips

```bash
MODEL=/path/to/MiniMax-H3
COMMON=(--width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2)

# T2VA — fox in snow (assets/showcase/t2va-fox-fast.mp4)
./h3 -d "$MODEL" "${COMMON[@]}" \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  -o assets/showcase/t2va-fox-fast.mp4

# Ref2VA — reference still + motion prompt (assets/showcase/ref2va-person-lamb.mp4)
./h3 -d "$MODEL" "${COMMON[@]}" \
  -p "The person in Picture 1 smiles and gently waves while holding the black lamb on the grassy hillside. Soft natural light, cinematic, realistic motion." \
  --ref-image assets/showcase/refs/ref2va-person-lamb.png \
  -o assets/showcase/ref2va-person-lamb.mp4

# FL2VA — first + last frame anchors (assets/showcase/fl2va-family-dinner.mp4)
./h3 -d "$MODEL" "${COMMON[@]}" \
  -p "A warm family dinner continues: steam rises from the ramen bowl, people chat softly, natural window light, cinematic." \
  --first-frame assets/showcase/refs/fl2va-first.png \
  --last-frame assets/showcase/refs/fl2va-last.png \
  -o assets/showcase/fl2va-family-dinner.mp4
```

Reference stills under `assets/showcase/refs/` were taken from the official
MiniMax-H3 demo assets (`ref2va.mp4` / `fl2va.mp4`).

## Status (v0.2.0, 2026-09-02)

| Capability | Status |
|------------|--------|
| T2VA (text → video+audio) | ✅ |
| FL2VA (`--first-frame` / `--last-frame`) | ✅ |
| Ref2VA (`--ref-image`, `--ref-video`, …) | ✅ |
| Runtime INT8 MLP (opt-in `H3_INT8_MLP=1`; BF16 MLP is faster on GB10) | ✅ |
| `--ref-audio` + preview UX (`--frames-dir`, `--show`) | ✅ |
| fox-s2 / fox-fast / 15 s cinematic | GB10 scoreboard below — [`docs/PERF_BASELINE.md`](docs/PERF_BASELINE.md) |

Progress log: [`docs/SPARK_AUTORUN.md`](docs/SPARK_AUTORUN.md) · Known gaps:
[`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) · Porting notes:
[`docs/SPARK_PORTING.md`](docs/SPARK_PORTING.md) · Perf baseline:
[`docs/PERF_BASELINE.md`](docs/PERF_BASELINE.md)

## Requirements

- DGX Spark (or Linux + CUDA 12.x with GB10-class GPU)
- Official BF16 checkpoint at `MiniMax-H3/` (`FL2VA/*`, optional `Ref2VA/*`)
- FFmpeg / FFprobe on `PATH`
- ICU (`libicu-dev`), build tools, `nvcc`

## Build

```bash
git clone https://github.com/alexhegit/h3-spark.c.git
cd h3-spark.c
make -f Makefile.linux -j$(nproc) h3
./h3 --info -d /path/to/MiniMax-H3
```

## Quick generate (T2VA)

Same fox-fast preset as the T2VA showcase clip:

```bash
./h3 --profile \
  -d /path/to/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  -o outputs/fox-fast.mp4
```

First run pays model load + filesystem cache; repeat runs for timing.
On DGX Spark (GB10) at `03adb33`, **warm** repeats of this command are about
**15.5 s** wall (**8.2 s** GPU Euler denoise); output md5 prefix
`f5282774d3a4`. Dated tables:
[`docs/PERF_BASELINE.md`](docs/PERF_BASELINE.md).

## HIP-page presets (GB10, v0.2.0)

Same CLI knobs as the [h3-hip.c](https://alexhegit.github.io/h3-hip.c/)
reproduce section (`03adb33`, `--seed 42`, `--profile`). These are Spark
measurements of those commands, not a vendor bake-off.

| Preset | knobs | GB10 E2E | denoise (sdpa / linear) | md5 prefix |
|---|---|---:|---|---|
| **fox-s2** | 512² 22f, steps 2, L35 R1 | **8.0 s** warm | **1.19 s** (0.20 / 0.76) | `aeb5ae10e105` |
| **fox-fast** | 512² 22f, steps 20, L45 R2 | **15.5 s** warm | **8.20 s** (1.43 / 5.19) | `f5282774d3a4` |
| **15 s cinematic** | 864×480, `--seconds 15`, L45 R2 | **18 min 17 s** | **16 min 51 s** (866 / 110) | `60fd70cc309c` |
| same + `--token-reduction` | opt-in; quality trade | **11 min 22 s** | **9 min 53 s** (486 / 82) | `19c109ebb0cb` |

fox-s2 wall is mostly video VAE (~2.5 s) + Qwen (~2.1 s), not DiT. 15 s wall
is still long-N SDPA. Logs: `/tmp/h3_perf/hip-examples-20260902/`.

```bash
# fox-s2 — short A/B smoke
./h3 --profile -d /path/to/MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 --seed 42 \
  -o outputs/fox-s2.mp4

# 15 s cinematic — same knobs as the HIP page (paste that office prompt)
./h3 --profile -d /path/to/MiniMax-H3 \
  -p "$PROMPT_15S" \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  -o outputs/long-15s-cinematic.mp4
# optional: append --token-reduction  (11 min 22 s on this box; not bit-identical)
```

The 15 s prompt is the HIP-page office/cinematic text. For Ref2VA / FL2VA
reproduction, see [Showcase](#showcase-dgx-spark).

## `--token-reduction` (faster, worse picture)

Opt-in. **Off by default.** The wall-clock win is paid in quality: it is not
a lossless shortcut.

Middle DiT blocks (4–30; deeper on early denoise steps) pair adjacent
**horizontal video tokens** and keep a full-resolution residual. Text/audio
tokens are unchanged.

On the fox-fast showcase preset (512², 22 frames, seed 42, steps 20 / L45 /
reuse 2), decoded pixels vs the same run without the flag:

| | `--token-reduction` vs off |
|---|---:|
| Average PSNR | **17.8 dB** (Y 16.0, U 31.2, V 32.8) |
| Worst / best frame | 15.2 / 21.2 dB |
| SSIM All | **0.72** (Y 0.63) |
| Output md5 | `f5282774d3a4` → `4d1d250e5ab9` |
| Audio | also not identical |

The speed is real (15 s cinematic **1097 s → 682 s** on the HIP-page 864×480
A/B at `03adb33`) and so is the quality hit (fur, edges). Do not use this
flag when you need the bit-identical fox-fast reference. Details:
[`docs/PERF_BASELINE.md`](docs/PERF_BASELINE.md).

## Conditional paths

Ordered references (`--ref-image`, `--ref-video`, `--ref-audio`, …) and
first/last-frame anchors (`--first-frame`, `--last-frame`) follow the same CLI
as [antirez/h3.c](https://github.com/antirez/h3.c). Exact commands for the
three showcase samples are under [Showcase](#showcase-dgx-spark).
## Preview while generating

- **`--frames-dir DIR`** — write each decoded frame as PPM (works everywhere)
- **`--show`** — live denoise preview in Kitty/Ghostty/iTerm2/WezTerm/Konsole  
  Override detection: `H3_TERMINAL=kitty` (useful over SSH).  
  Default terminal zoom: **1× on Linux**, 2× on macOS.

## Tests

```bash
make -f Makefile.linux test
make -f Makefile.linux test-conditional
# full ref-video smoke (~20 min extra):
H3_CONDITIONAL_SKIP_REF_VIDEO=0 make -f Makefile.linux test-conditional
```

Set `H3_MODEL_ROOT=/path/to/MiniMax-H3` if weights are not under `./MiniMax-H3`.

## Repository layout

Host/model/CLI code follows [antirez/h3.c](https://github.com/antirez/h3.c).
The Spark build uses the CUDA backend (`h3_gpu.cu`, `Makefile.linux`). Apple
Metal sources (`h3_gpu.m`, `h3_shaders.metal`, root `Makefile`) remain for
reference against the original project and are not the Spark build path.

## License

See [`LICENSE`](LICENSE). Model weights are subject to the MiniMax-H3 license on
Hugging Face.
