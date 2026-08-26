#!/usr/bin/env bash
# Puts a narrower format's error through the whole pipeline before anyone
# implements it. H3_INT8_LEVELS coarsens every quantized tensor by a known
# factor; 48 lands on FP8-E4M3's measured weight error and 13 on NVFP4's, so
# the resulting clips answer whether those formats are admissible at all.
set -euo pipefail

MODEL_ROOT="${H3_MODEL_ROOT:-/home/alex/HF-MODELS/MiniMax-H3}"
OUT_DIR="${1:-/tmp/h3_levels}"
PROMPT="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind."

mkdir -p "$OUT_DIR"
for levels in 127 48 13; do
    echo "=== H3_INT8_LEVELS=$levels ==="
    H3_INT8_LEVELS="$levels" /usr/bin/time -f 'WALL_SEC %e' ./h3 --profile \
        -d "$MODEL_ROOT" -p "$PROMPT" \
        --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 \
        -o "$OUT_DIR/levels-$levels.mp4" \
        > "$OUT_DIR/levels-$levels.log" 2>&1
    grep -E 'GPU Euler denoise|WALL_SEC' "$OUT_DIR/levels-$levels.log" || true
done

echo
echo "=== PSNR against the 127-level reference ==="
for levels in 48 13; do
    printf '%-4s ' "$levels"
    ffmpeg -hide_banner -i "$OUT_DIR/levels-$levels.mp4" \
        -i "$OUT_DIR/levels-127.mp4" -lavfi psnr -f null - 2>&1 |
        grep -o 'average:[0-9.a-z]*' || echo "psnr failed"
done
