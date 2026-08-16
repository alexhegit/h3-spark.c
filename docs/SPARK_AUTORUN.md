# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark (`spark` branch).

## Plan (revised 2026-08-16)

| Phase | Goal | INT8? | Status |
|-------|------|-------|--------|
| **0** | Scaffold, host tests, CUDA probe | no | ✅ Done |
| **1** | DiT **BF16** block parity + text + vision encoder | no | ✅ Done |
| **2** | **Metal-aligned runtime INT8** on GB10 | yes | ✅ Done |
| **3** | Full pipeline + end-to-end generate | inherits Phase 2 | ✅ Done |

**Weights:** official `MiniMaxAI/MiniMax-H3` → `FL2VA/*` **BF16** shards only.

## Gate commands

```bash
make -f Makefile.linux test
./h3_cuda_video_vae_smoke $H3_MODEL_ROOT
./h3_cuda_audio_vae_smoke $H3_MODEL_ROOT
./h3_cuda_video_encoder_smoke $H3_MODEL_ROOT
./h3_cuda_audio_encoder_smoke $H3_MODEL_ROOT
./h3 -d $H3_MODEL_ROOT -p "a red fox walking" \
  --width 256 --height 256 --frames 22 --steps 2 --layers 35 \
  -o /tmp/h3_phase3_smoke.mp4
```

## Phase 3 — VAE + generate ✅

| Path | Status | Evidence |
|------|--------|----------|
| Video decode | ✅ | 5×32×32, 36 SDPA / 38 sub |
| Audio decode | ✅ | 2×3200@32k, 136 conv / 16 sub |
| Video encode | ✅ | 64² → 24×1×4×4, 34 conv / 18 sub |
| Audio encode | ✅ | 1600 PCM → 32×2×2, 38 conv / 1 SDPA |
| `./h3` e2e | ✅ | 256²×22 MP4 written |

### Remaining stubs (**4**, non-blocking)

See [`docs/KNOWN_ISSUES.md`](KNOWN_ISSUES.md):

- **KI-001** DiT F32: `adaln_f32` / `gate_f32` / `qkv_rope_f32` — deferred on Spark (BF16/INT8 path); not planned unless something forces FP32 DiT
- **KI-002** `mlp_nax_bf16` — Metal-only; intentional CUDA stub

### Optional polish
- MLX fixture parity when `misc/fixtures` present
- 512² preset wall-clock / `--profile` baseline on Spark

## Commits (Phase 3)

```
ec2228b audio encoder attention + smoke (Phase 3 close)
b302ca7 video encoder pad/conv3d/group_norm + smoke
7fe5d98 Phase 3 e2e generate documented
117e538 audio VAE F32 decode + smoke
e27b1a3 video VAE F32 decode + smoke
```

---

*Last updated: 2026-08-16*
