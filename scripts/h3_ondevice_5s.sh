#!/usr/bin/env bash
# H3-OnDevice Spark workload on the stock MiniMax-H3 weights.
# Matches https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/ :
#   832x480, 24 fps, --seconds 5 (aligns to 124 frames), 50 steps,
#   50 DiT layers, reuse 1. No LoRA, no --sol-attn, no token reduction.
# Their published baseline for this job is 710.6 s; full Sol Engine is 181.3 s.
# The page only prints a compacted prompt, so this uses that published sentence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${H3_MODEL_ROOT:-/home/alex/HF-MODELS/MiniMax-H3}"
OUTDIR="${H3_ONDEVICE_OUT:-/tmp/h3_ondevice}"
mkdir -p "$OUTDIR"
NAME="${1:-spark-832x480-5s-50step}"
shift 2>/dev/null || true
MP4="$OUTDIR/${NAME}.mp4"
LOG="$OUTDIR/${NAME}.log"

/usr/bin/time -f 'WALL_SEC %e\nMAX_RSS_KB %M' ./h3 --profile \
  -d "$MODEL" \
  -p "Cinematic push-in on a starship bridge: a captain in a high-collared navy tunic stands silhouetted at the observation window as a dreadnought armada charges its hyperdrives. A blinding flash, the bridge shudders, and the fleet is gone." \
  --width 832 --height 480 --seconds 5 \
  --steps 50 --layers 50 --reuse 1 --seed 42 \
  -o "$MP4" \
  "$@" \
  >"$LOG" 2>&1 || { tail -80 "$LOG"; exit 1; }

md5sum "$MP4" | tee "$OUTDIR/${NAME}.md5"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,duration,avg_frame_rate \
  -of default=nw=1 "$MP4" | tee "$OUTDIR/${NAME}.ffprobe"
python3 - "$LOG" "$NAME" <<'PY'
import re, sys
log_path, name = sys.argv[1:]
log = open(log_path).read()
def grab(pat):
    m = re.search(pat, log, re.S)
    return m.group(1) if m else "?"
wall = grab(r"WALL_SEC ([0-9.]+)")
m = re.search(
    r"GPU Euler denoise wall=\s*([0-9.]+)s.*?gpu-op linear=([0-9.]+)s sdpa=([0-9.]+)s",
    log, re.S)
den, linear, sdpa = (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?")
vae = grab(r"video VAE decoder\s+total\s+wall=\s*([0-9.]+)s")
qwen = grab(r"Qwen text encoder\s+total\s+wall=\s*([0-9.]+)s")
evals = grab(r"selected GPU reuse schedule has (\d+) evaluations")
print(f"SUMMARY {name} WALL={wall} denoise={den} linear={linear} sdpa={sdpa} vae={vae} qwen={qwen} evals={evals}")
PY
