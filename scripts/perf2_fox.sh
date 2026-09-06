#!/usr/bin/env bash
# fox-fast A/B for perf2. Quality vs the v0.2.0 reference MP4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${H3_MODEL_ROOT:-/home/alex/HF-MODELS/MiniMax-H3}"
OUTDIR="${PERF2_OUT:-/tmp/h3_perf2}"
REF="${PERF2_REF:-$OUTDIR/ref-fox-fast.mp4}"
mkdir -p "$OUTDIR"
NAME="${1:?usage: perf2_fox.sh <name> [extra h3 args]}"
shift || true
MP4="$OUTDIR/${NAME}.mp4"
LOG="$OUTDIR/${NAME}.log"

/usr/bin/time -f 'WALL_SEC %e\nMAX_RSS_KB %M' ./h3 --profile \
  -d "$MODEL" \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 --seed 42 \
  -o "$MP4" \
  "$@" \
  >"$LOG" 2>&1 || { tail -50 "$LOG"; exit 1; }

md5sum "$MP4" | tee "$OUTDIR/${NAME}.md5"
python3 - "$LOG" "$NAME" "$REF" "$MP4" "$OUTDIR/${NAME}.quality" <<'PY'
import re, subprocess, sys
log_path, name, ref, test, qpath = sys.argv[1:]
log=open(log_path).read()
def grab(pat, g=1):
    m=re.search(pat, log, re.S)
    return m.group(g) if m else "?"
wall=grab(r"WALL_SEC ([0-9.]+)")
m=re.search(r"GPU Euler denoise wall=\s*([0-9.]+)s.*?gpu-op linear=([0-9.]+)s sdpa=([0-9.]+)s", log, re.S)
den, linear, sdpa = (m.group(1), m.group(2), m.group(3)) if m else ("?","?","?")
vae=grab(r"video VAE decoder\s+total\s+wall=\s*([0-9.]+)s")
print(f"SUMMARY {name} WALL={wall} denoise={den} linear={linear} sdpa={sdpa} vae={vae}")
import os
if os.path.exists(ref):
    p=subprocess.run(["ffmpeg","-hide_banner","-i",ref,"-i",test,"-lavfi","[0:v][1:v]psnr;[0:v][1:v]ssim","-f","null","-"],capture_output=True,text=True)
    err=p.stderr
    open(qpath,"w").write(err)
    psnr=re.search(r"average:([0-9.]+)", err)
    ssim=re.search(r"All:([0-9.]+)", err)
    print(f"QUALITY vs ref PSNR_avg={psnr.group(1) if psnr else '?'} SSIM_all={ssim.group(1) if ssim else '?'}")
PY
