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
./h3_cuda_video_encoder_smoke $H3_MODEL_ROOT
./h3 -d $H3_MODEL_ROOT -p "a red fox walking" \
  --width 256 --height 256 --frames 22 --steps 2 --layers 35 \
  -o /tmp/h3_phase3_smoke.mp4
```

## Phase 3 — VAE + generate ✅ (core + encoder stretch)

### Decode ✅
- Video: `rms_norm_f32`, `video_qkv_rope_f32`, `sdpa_f32` + smoke
- Audio: weight_norm / conv1d / transpose / snake + smoke (136 conv / 16 sub)

### Encode stretch ✅
- `vae_encoder_pad_f32`, `conv3d_f32`, `group_norm_silu_f32`
- Smoke: 64×64 → latent 24×1×4×4, 34 conv / 18 submissions

### End-to-end generate ✅
- Bare T2VA `./h3` → MP4 (256²×22, 2 steps, 35 layers)

### Remaining stubs (**7**)
- Audio encoder attention: `audio_qkv_split`, `sdpa_causal`, `audio_attention_pool`
- DiT-only F32: `adaln_f32`, `gate_f32`, `qkv_rope_f32`
- Metal-only skip: `mlp_nax_bf16`

## Commits (Phase 3)

```
(pending) video encoder pad/conv3d/group_norm + smoke
7fe5d98 Phase 3 e2e generate documented
117e538 audio VAE F32 decode + smoke
e27b1a3 video VAE F32 decode + smoke
```

---

*Last updated: 2026-08-16*
