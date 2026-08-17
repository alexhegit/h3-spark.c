# h3-spark.c

Native **MiniMax-H3** inference for **NVIDIA DGX Spark** (GB10, CUDA). This
repository is the long-term home for the Spark/CUDA port. It was split from
[`alexhegit/h3.c`](https://github.com/alexhegit/h3.c) (`spark` branch history)
so Spark work can evolve independently of the upstream Apple Metal tree.

**Upstream (Metal / macOS):** [alexhegit/h3.c](https://github.com/alexhegit/h3.c)  
**Official weights:** [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)

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

This tree retains the full h3 host/model stack plus CUDA backend
(`h3_gpu.cu`, `Makefile.linux`). Apple Metal sources (`h3_gpu.m`,
`h3_shaders.metal`, root `Makefile`) remain for reference and upstream parity
but are not the Spark build path.

## License

See [`LICENSE`](LICENSE). Model weights are subject to the MiniMax-H3 license on
Hugging Face.
