# FastH3 / TAEH3 / Pika / VSA sample clips (2026-11-07)

H3-OnDevice shape: 832×480, 5 s, 124f, seed 42, starship prompt. Measured on
GB10. Numbers in [`docs/PERF_BASELINE.md`](../../docs/PERF_BASELINE.md).

| File | Path | Notes |
|---|---|---|
| `fasth3-4step.mp4` | FastH3 dense + full VAE | ~94 s e2e; clean from frame 0 |
| `fasth3-taeh3.mp4` | FastH3 + `--taeh3` | ~69–72 s; **first ~3 frames are snow** (MemBlock cold start) |
| `pika-6step.mp4` | Pika 6-step renoise + full VAE | ~119 s |
| `vsa-taeh3-mma.mp4` | `--vsa` sparse-MMA + TAEH3 | ~58 s; same TAEH3 leading-snow |

`--taeh3` is a preview decoder. Final exports should use the stock video VAE.
