# h3-spark.c

CUDA port of [antirez/h3.c](https://github.com/antirez/h3.c) for **NVIDIA DGX
Spark** (GB10). The original project is a native MiniMax-H3 inference engine
(Apple Metal / macOS); this repository reimplements the GPU backend for CUDA so
the same CLI and model stack run on Spark.

**Original project:** [antirez/h3.c](https://github.com/antirez/h3.c)  
**Official weights:** [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)

## Showcase (DGX Spark)

Clips below were generated on NVIDIA DGX Spark with this CUDA port
(512×512, 22 frames, `--steps 20 --layers 45 --reuse 2`). Click a poster for
the MP4.

| Mode | Sample |
|------|--------|
| **T2VA** — text → video+audio | [![T2VA fox](assets/showcase/t2va-fox-fast.jpg)](assets/showcase/t2va-fox-fast.mp4) [mp4](assets/showcase/t2va-fox-fast.mp4) |
| **Ref2VA** — `--ref-image` | [![Ref2VA](assets/showcase/ref2va-person-lamb.jpg)](assets/showcase/ref2va-person-lamb.mp4) [mp4](assets/showcase/ref2va-person-lamb.mp4) |
| **FL2VA** — `--first-frame` + `--last-frame` | [![FL2VA](assets/showcase/fl2va-family-dinner.jpg)](assets/showcase/fl2va-family-dinner.mp4) [mp4](assets/showcase/fl2va-family-dinner.mp4) |

- **T2VA:** fox walking through snow (prompt-only).
- **Ref2VA:** identity/scene from a real reference still (person in pink suit holding a black lamb), then motion from the prompt.
- **FL2VA:** warm dinner / ramen scene anchored by real first and last frames from an official demo clip.

## Status (2026-08-17)

| Capability | Status |
|------------|--------|
| T2VA (text → video+audio) | ✅ |
| FL2VA (`--first-frame` / `--last-frame`) | ✅ |
| Ref2VA (`--ref-image`, `--ref-video`, …) | ✅ |
| Runtime INT8 MLP (Metal-aligned) | ✅ |
| `--ref-audio` + preview UX (`--frames-dir`, `--show`) | ✅ |
| Performance vs Metal / CUDA `--profile` phases | 📋 planned (see docs) |

Progress log: [`docs/SPARK_AUTORUN.md`](docs/SPARK_AUTORUN.md) · Known gaps:
[`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) · Porting notes:
[`docs/SPARK_PORTING.md`](docs/SPARK_PORTING.md)

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

Validated balanced preset (512², ~22 frames @ 24 fps):

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

## Conditional paths

```bash
# FL2VA first frame
./h3 -d /path/to/MiniMax-H3 -p "Continue the scene." \
  --first-frame fox.png --width 512 --height 512 --frames 22 --steps 20 \
  -o outputs/fl2va.mp4

# Ref2VA image + standalone audio
./h3 -d /path/to/MiniMax-H3 \
  -p "Use the animal in Picture 1 with soft wind. <Picture 1>" \
  --ref-image fox.png --ref-audio wind.wav \
  --width 512 --height 512 --frames 22 --steps 20 \
  -o outputs/ref2va.mp4
```

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
