# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark (`spark` branch).

## Plan (revised 2026-08-16)

| Phase | Goal | INT8? | Status |
|-------|------|-------|--------|
| **0** | Scaffold, host tests, CUDA probe | no | ✅ Done |
| **1** | DiT **BF16** block parity + text + vision encoder | no | ✅ Done |
| **2** | **Metal-aligned runtime INT8** on GB10 | yes | ✅ Done |
| **3** | Full pipeline + end-to-end generate | inherits Phase 2 | ✅ Core done |

**Weights:** official `MiniMaxAI/MiniMax-H3` → `FL2VA/*` **BF16** shards only.

## Gate commands

```bash
make -f Makefile.linux test
./h3_cuda_video_vae_smoke $H3_MODEL_ROOT
./h3_cuda_audio_vae_smoke $H3_MODEL_ROOT
./h3 -d $H3_MODEL_ROOT -p "a red fox walking" \
  --width 256 --height 256 --frames 22 --steps 2 --layers 35 \
  -o /tmp/h3_phase3_smoke.mp4
```

## Phase 3 — VAE + generate ✅ (core)

### Video VAE decode ✅
- `rms_norm_f32`, `video_qkv_rope_f32`, `sdpa_f32`
- Smoke: 5×32×32 finite RGB, 36 SDPA, 38 submissions

### Audio VAE decode ✅
- `weight_norm_f32`, `conv1d(_stride)`, `conv_transpose1d`, `alias_free_snake`, `snake1d`
- Smoke: 2×3200 @32kHz, **136 conv / 16 submissions**

### End-to-end generate ✅
- Bare T2VA `./h3 -p …` completed: text → DiT denoise → audio/video VAE → FFmpeg MP4
- Verified smoke: 256²×22 frames, 2 steps, 35 layers → `/tmp/h3_phase3_smoke.mp4`

### Stretch (not blocking core generate)
- [ ] Video encoder: `vae_encoder_pad_f32`, `conv3d_f32`, `group_norm_silu_f32` (`--first-frame`)
- [ ] Audio encoder attention chain (Ref2VA audio refs)
- [ ] DiT-only F32 stubs (`adaln_f32` / `gate_f32` / `qkv_rope_f32`)
- [ ] MLX fixture parity when `misc/fixtures` present
- [ ] README default 512² presets
- [ ] Skip Metal-only `mlp_nax_bf16`

Stubs remaining: **10**.

## Commits (Phase 3)

```
117e538 audio VAE F32 decode + smoke
e27b1a3 video VAE F32 decode + smoke
(pending) Phase 3 e2e generate smoke documented
```

---

*Last updated: 2026-08-16 — Phase 3 core generate green*
