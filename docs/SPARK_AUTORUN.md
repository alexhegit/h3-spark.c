# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark (`spark` branch).

## Plan (revised 2026-08-17)

| Phase | Goal | INT8? | Status |
|-------|------|-------|--------|
| **0** | Scaffold, host tests, CUDA probe | no | ✅ Done |
| **1** | DiT **BF16** block parity + text + vision encoder | no | ✅ Done |
| **2** | **Metal-aligned runtime INT8** on GB10 | yes | ✅ Done |
| **3** | Full pipeline + end-to-end generate | inherits Phase 2 | ✅ Done |
| **3b** | Conditional paths (FL2VA keyframes + Ref2VA) | inherits | ✅ Done |
| **3c** | UX: `--ref-audio` gate + `--show` / frames-dir | inherits | ✅ Done |

**Weights:** official `MiniMaxAI/MiniMax-H3` → `FL2VA/*` + optional `Ref2VA/*`.

## Gate commands

```bash
make -f Makefile.linux test
make -f Makefile.linux test-conditional   # FL2VA + Ref2VA image/audio/silent
# full --ref-video (72 frames, ~20 min extra):
H3_CONDITIONAL_SKIP_REF_VIDEO=0 make -f Makefile.linux test-conditional
```

## Phase 3b — Conditional generate ✅

| Path | Status | Notes |
|------|--------|-------|
| `--first-frame` | ✅ | includes `--frames-dir` PPM check |
| `--last-frame` | ✅ | |
| first+last | ✅ | |
| `--ref-image` | ✅ | Ref2VA transformer |
| `--ref-image` + `--ref-audio` | ✅ | standalone WAV ≥2s @32 kHz |
| `--ref-silent-video` | ✅ | |
| `--ref-video` (w/ audio) | ✅ | needs ≥48 ref frames + long audio; fixed snake grid |

**Bug fix:** `h3_gpu_alias_free_snake_f32` used `blockIdx.y = length`, which exceeds
CUDA's 65535 grid limit once AudioVAE upsamples past ~2s of audio. Switched to
1D linear launch.

Script: `scripts/smoke_conditional.sh`

## Phase 3c — Preview UX

| Item | Status | Notes |
|------|--------|-------|
| `--frames-dir` | ✅ | always works; smoke checks `frame-0000.ppm` |
| `--show` | ✅ code | Kitty/Ghostty/iTerm2/…; `H3_TERMINAL=` override for SSH/IDE |
| default `--zoom` | ✅ | 2 on macOS, 1 on Linux/Spark |

## Decision backlog (2026-08-17)

| ID | Item | Decision |
|----|------|----------|
| **A** | DiT F32 GPU API surface (KI-001) | **Skip** — BF16/INT8 (+ optional FP8 later) enough for Spark |
| **B** | Metal NAX MLP (KI-002) | **Ignore** — Apple-only |
| **C** | `--show` / frames preview UX | **Do now** (this phase) |
| **D** | Perf + CUDA `--profile` phase marks | **Planned later** — high value, large effort |
| **E** | MLX fixture numerical parity | **Pending** — no Mac / no `misc/fixtures` |
| **F** | `--ref-audio` hard gate | **Do now** (this phase) |

### Remaining stubs (non-blocking)

See [`docs/KNOWN_ISSUES.md`](KNOWN_ISSUES.md) KI-001 / KI-002 (won't-fix on Spark).

### Planned later (D)

- Implement real `h3_gpu_profile_mark` on CUDA (wall / encode / kernel buckets)
- Hotspot: SDPA, GEMM/MLP fusion, VAE decode tiling
- Target: close the gap vs Metal fox-fast wall times on GB10

### Pending (E)

- Import or regenerate MLX golden fixtures when available
- Block-level rel/abs checks; optional full-video SSIM vs reference renders

---

*Last updated: 2026-08-17 — UX gates + backlog decisions*
