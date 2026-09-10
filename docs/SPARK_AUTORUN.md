# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark. Shipping tag: **v0.2.1**
(`perf2`).

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
make -f Makefile.linux test                 # 16.2 s (2026-09-09)
make -f Makefile.linux test-conditional     # 4 min 33 s; FL2VA + Ref2VA image/audio/silent
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
| **D** | Perf + CUDA `--profile` phase marks | **Done** (v0.2.1) — scoreboard in README / PERF_BASELINE |
| **E** | MLX fixture numerical parity | **Pending** — no Mac / no `misc/fixtures` |
| **F** | `--ref-audio` hard gate | **Do now** (this phase) |

### Remaining stubs (non-blocking)

See [`docs/KNOWN_ISSUES.md`](KNOWN_ISSUES.md) KI-001 / KI-002 (won't-fix on Spark).

### D — perf (closed at v0.2.0)

- Implement real `h3_gpu_profile_mark` on CUDA (wall / encode / kernel buckets) — **done** (`483ffdf`)
- **v0.1 baseline (2026-08-17):** fox-fast DiT denoise **1471.7 s** vs Metal **~16.7 s** (~88×) — keep as history in [`PERF_BASELINE.md`](PERF_BASELINE.md)
- **v0.2.0 shipping (2026-09-02, `03adb33`):** HIP-page knobs on GB10 — fox-s2 **~8.0 s** / fox-fast **~15.5 s** / 15 s cinematic **18 min 17 s** (TR **11 min 22 s**). Fox-fast denoise **~8.2 s**, md5 `f5282774d3a4`.
- **v0.2.1 retest (2026-09-09, `7420692`):** fox-s2 **8.2 s** / fox-fast **15.6 s** / 15 s **17 min 56 s** (md5 unchanged). `make test` **16.2 s**, `test-conditional` **4 min 33 s**.
- **15 s + TR (2026-09-10, `59d307b`):** **11 min 14 s** e2e / **9 min 49 s** denoise (sdpa 482 / linear 81), md5 `19c109ebb0cb` (same as v0.2.0 TR). vs quality 1076 s: **−37.4 %**.

### Pending (E)

- Import or regenerate MLX golden fixtures when available
- Block-level rel/abs checks; optional full-video SSIM vs reference renders

---

*Last updated: 2026-09-09 — v0.2.1 retest*
