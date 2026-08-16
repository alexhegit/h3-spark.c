#!/usr/bin/env bash
# Conditional-path smoke: FL2VA first/last frame + Ref2VA image/video.
# Usage: ./scripts/smoke_conditional.sh [MODEL_ROOT]
# Env: H3_CONDITIONAL_SKIP_REF_VIDEO=1 to skip long --ref-video (72 frames)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-${H3_MODEL_ROOT:-/home/alex/HF-MODELS/MiniMax-H3}}"
H3="${ROOT}/h3"
ASSETS="${TMPDIR:-/tmp}/h3_func_assets"
mkdir -p "$ASSETS"

if ! test -x "$H3"; then
  echo "skip: $H3 not built (make -f Makefile.linux h3)"
  exit 0
fi
if ! test -f "$MODEL/FL2VA/transformer/config.json"; then
  echo "skip: FL2VA weights not available at $MODEL"
  exit 0
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "skip: ffmpeg required for image/video fixtures"
  exit 0
fi

make_png() {
  local path="$1" color="$2" size="${3:-256x256}"
  ffmpeg -y -f lavfi -i "color=c=${color}:s=${size}:d=1" -frames:v 1 "$path" \
    >/dev/null 2>&1
}

make_png "$ASSETS/first.png" "0x4060a0"
make_png "$ASSETS/last.png" "0xa06040"
make_png "$ASSETS/ref.png" "0x308050" "320x240"
ffmpeg -y -f lavfi -i color=c=0x7030a0:s=320x240:r=24:d=1 \
  -c:v libx264 -pix_fmt yuv420p "$ASSETS/ref_silent.mp4" >/dev/null 2>&1
ffmpeg -y -f lavfi -i color=c=0x7030a0:s=320x240:r=24:d=3 \
  -f lavfi -i sine=f=440:d=3 -shortest \
  -c:v libx264 -pix_fmt yuv420p -c:a aac "$ASSETS/ref_audio.mp4" \
  >/dev/null 2>&1

COMMON=(-d "$MODEL" --width 256 --height 256 --frames 22 --steps 2 --layers 35)

echo "== FL2VA --first-frame =="
"$H3" "${COMMON[@]}" --seed 7 \
  -p "conditional first-frame smoke" \
  --first-frame "$ASSETS/first.png" \
  -o "$ASSETS/out_first.mp4"
test -s "$ASSETS/out_first.mp4"
echo "ok: first-frame"

echo "== FL2VA --last-frame =="
"$H3" "${COMMON[@]}" --seed 8 \
  -p "conditional last-frame smoke" \
  --last-frame "$ASSETS/last.png" \
  -o "$ASSETS/out_last.mp4"
test -s "$ASSETS/out_last.mp4"
echo "ok: last-frame"

echo "== FL2VA --first-frame + --last-frame =="
"$H3" "${COMMON[@]}" --seed 9 \
  -p "conditional both-frames smoke" \
  --first-frame "$ASSETS/first.png" \
  --last-frame "$ASSETS/last.png" \
  -o "$ASSETS/out_both.mp4"
test -s "$ASSETS/out_both.mp4"
echo "ok: first+last frames"

if test -f "$MODEL/Ref2VA/transformer/config.json" || \
   test -f "$MODEL/Ref2VA/transformer/model.safetensors.index.json"; then
  echo "== Ref2VA --ref-image =="
  "$H3" "${COMMON[@]}" --seed 10 \
    -p "A red fox walks through snow. <Picture 1>" \
    --ref-image "$ASSETS/ref.png" \
    -o "$ASSETS/out_ref_image.mp4"
  test -s "$ASSETS/out_ref_image.mp4"
  echo "ok: ref-image"

  echo "== Ref2VA --ref-silent-video =="
  "$H3" "${COMMON[@]}" --seed 11 \
    -p "Follow the motion in the clip. <Video 1>" \
    --ref-silent-video "$ASSETS/ref_silent.mp4" \
    -o "$ASSETS/out_ref_silent.mp4"
  test -s "$ASSETS/out_ref_silent.mp4"
  echo "ok: ref-silent-video"

  if test "${H3_CONDITIONAL_SKIP_REF_VIDEO:-}" = "1"; then
    echo "skip: --ref-video (H3_CONDITIONAL_SKIP_REF_VIDEO=1)"
  else
    # Embedded audio needs >=48 decoded ref frames (~2s @24fps).
    echo "== Ref2VA --ref-video (72 frames) =="
    "$H3" -d "$MODEL" --width 256 --height 256 --frames 72 --steps 2 \
      --layers 35 --seed 12 \
      -p "Follow the motion and sound. <Video 1>" \
      --ref-video "$ASSETS/ref_audio.mp4" \
      -o "$ASSETS/out_ref_video.mp4"
    test -s "$ASSETS/out_ref_video.mp4"
    echo "ok: ref-video"
  fi
else
  echo "skip: Ref2VA weights not available"
fi

echo "ok: conditional-path smoke suite passed"
