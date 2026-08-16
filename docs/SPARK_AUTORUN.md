# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark (`spark` branch).

## Plan (revised 2026-08-16)

| Phase | Goal | INT8? | Status |
|-------|------|-------|--------|
| **0** | Scaffold, host tests, CUDA probe | no | ✅ Done |
| **1** | DiT **BF16** block parity + text + vision encoder | no | ✅ Done |
| **2** | **Metal-aligned runtime INT8** on GB10 | yes | ✅ Done |
| **3** | Full pipeline + end-to-end generate | inherits Phase 2 | ✅ Done |
| **3b** | Conditional paths (FL2VA keyframes + Ref2VA) | inherits | ✅ Done |

**Weights:** official `MiniMaxAI/MiniMax-H3` → `FL2VA/*` + optional `Ref2VA/*`.

## Gate commands

```bash
make -f Makefile.linux test
make -f Makefile.linux test-conditional   # FL2VA first/last + Ref2VA image/silent
# full --ref-video (72 frames, ~20 min extra):
H3_CONDITIONAL_SKIP_REF_VIDEO=0 make -f Makefile.linux test-conditional
```

## Phase 3b — Conditional generate ✅

| Path | Status | Notes |
|------|--------|-------|
| `--first-frame` | ✅ | ~135s @256²/22f/2steps |
| `--last-frame` | ✅ | |
| first+last | ✅ | |
| `--ref-image` | ✅ | Ref2VA transformer |
| `--ref-silent-video` | ✅ | |
| `--ref-video` (w/ audio) | ✅ | needs ≥48 ref frames + long audio; fixed snake grid |

**Bugfix:** `h3_gpu_alias_free_snake_f32` used `blockIdx.y = length`, which exceeds
CUDA's 65535 grid limit once AudioVAE upsamples past ~2s of audio. Switched to
1D linear launch.

Script: `scripts/smoke_conditional.sh`

## Phase 3 — VAE + generate ✅

(see prior commits)

### Remaining stubs (**4**, non-blocking)

See [`docs/KNOWN_ISSUES.md`](KNOWN_ISSUES.md) KI-001 / KI-002.

### Optional polish
- MLX fixture parity when `misc/fixtures` present
- 512² perf / CUDA `--profile` phase marks

---

*Last updated: 2026-08-16 — conditional paths green*
