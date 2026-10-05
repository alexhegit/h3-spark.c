#!/usr/bin/env bash
# 15 s cinematic quality path on DGX Spark.
# Same prompt and knobs as the h3-hip.c wiki "Reproduce the 15 s clip":
#   https://github.com/alexhegit/h3-hip.c/wiki/Long-video
#   864×480, --seconds 15 (362 frames), steps 20, layers 45, reuse 2, seed 42.
# h3-hip.c bench/fox-15s.sh uses this prompt and adds --token-reduction.
# That flag is not part of this quality path (Spark md5 60fd70cc309c).
# Extra flags in "$@" override (getopt last-wins), e.g. --fbc, --sol-attn,
# --reuse 3, --token-reduction.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${H3_MODEL_ROOT:-/home/alex/HF-MODELS/MiniMax-H3}"
if [[ $# -gt 0 && -d "$1" ]]; then
  MODEL="$1"
  shift
fi

exec ./h3 --profile -d "$MODEL" \
  -p "15 seconds, 16:9 landscape cinematic. A lone software engineer works late in a dim home office lit only by monitor glow and a desk lamp. Photoreal live-action feel with subtle handheld camera breathing.

[0–3 seconds] Medium shot from behind the desk. Code scrolls on dual monitors; warm red accent light reflects on glass. Ambient: quiet keyboard clicks, soft fan hum, distant city rain.

[3–6 seconds] Slow push-in over the shoulder. On screen, glowing matrix tiles and magenta wavefronts visualize a neural network training. The engineer pauses, sips coffee. Sound: gentle electronic pulse, a single soft notification chime.

[6–9 seconds] Cut to close-up of hands typing, then rack focus to a small window showing a red fox walking through digital snow inside the monitor reflection. Sound: rising synthesized tone, subtle wind.

[9–12 seconds] Smooth lateral move across the desk: terminal windows, GPU metrics, and a grid of video frames assembling on screen. Warm amber grade, volumetric dust in the lamp beam.

[12–15 seconds] Controlled pullback reveals the full workspace at rest. The engineer leans back, satisfied. Sound: clean final impact, room tone fades.

No readable text, no logos, no subtitles. Premium technology documentary aesthetic." \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  -o outputs/long-15s-cinematic.mp4 \
  "$@"
