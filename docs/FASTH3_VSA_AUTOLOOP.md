# Overnight autoloop — FastH3 / VSA (2026-11-07 → morning)

**Stopped:** goal complete (~22:05 +0800). **Gate:** fox-fast md5 `f5282774d3a4`.  
**Progress file:** this path · overnight loop killed.

## Baseline → current best

| Run | Before | Now | Verdict |
|---|---:|---:|---|
| fox-fast | md5 `f5282774d3a4` | same, ~22 s cold / 15.5 s warm | **KEEP** |
| fasth3+taeh3 | 72.4 s / denoise 47.7 s | **69.4 s** / 47.5 s | **KEEP** skip VAE prefetch under `--taeh3` |
| vsa+taeh3 | 180 s / denoise **151 s** | **~58–61 s** / denoise **~34 s** | **KEEP** sparse-MMA fine + scatter tiled (+ skip retile memset) |

VSA denoise now **beats dense FastH3** (34 s vs 47 s). E2E also faster (58–61 vs 69).

## REJECT / KEEP log

| Time | Change | Result |
|---|---|---|
| 21:30 | Skip VAE prefetch under `--taeh3` | **KEEP** fasth3 e2e 72.4→69.4 s |
| 21:31 | VSA fine serial-per-tile (32 thr) | **REJECT** denoise 282 s |
| 21:37 | VSA fine 1024-thr tile (32 warps∥) | ≈baseline denoise 149 s — no win |
| 21:45 | VSA sparse-MMA fine (`h3_vsa_fine_mma_kernel`, 128 thr, m16n8k16) + skip tiled memsets | **KEEP** denoise **151→34 s** (−77%), e2e **180→58 s**; beats dense denoise 47 s |
| 21:50 | Drop scatter/gather; pool/fine read token-major via inv | **REJECT** denoise 34→**38 s**, e2e 58→**68.5 s** (coalescing lost) |
| 22:01 | Revert to MMA+scatter KEEP | **restored** denoise **33.855 s**, e2e **60.58 s**; fox-fast md5 `f5282774d3a4` |
| 22:05 | VSA gate BF16→INT8 (`H3_VSA_BF16_GATE=1` restores old) | **no ≥5% e2e** (denoise 33.9 / e2e 61.1, linear still ~17.3 s); peak **24.1→22.4 GiB** — keep as memory win |

## Code touched

- `h3.c` — no VAE prefetch when `--taeh3`
- `h3_vsa_kernels.cu` / `h3_gpu.cu` — sparse-MMA fine + scatter (**KEEP**); no-retile **REJECT**
- `h3_dit.c` — VSA gate INT8 quantize (default; opt out `H3_VSA_BF16_GATE=1`)

## Stop decision

Primary goal **met**. Exhausted ≥5% e2e knobs on this path:
1. VSA sparse-MMA — **KEEP** (151→34 s denoise; beats dense 47 s)
2. TAEH3 prefetch skip — **KEEP** (72.4→69.4 s)
3. Retile elimination — **REJECT**
4. INT8 VSA gate — neutral wall, memory only

fox-fast md5 still `f5282774d3a4`. No commit unless asked.

Best speed log: `vsa-taeh3-mma.log` / `vsa-taeh3-revert.log` (~34 s denoise / ~58–61 s e2e).
