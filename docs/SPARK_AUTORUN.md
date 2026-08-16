# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark (`spark` branch).

## Plan (revised 2026-08-16)

| Phase | Goal | INT8? | Status |
|-------|------|-------|--------|
| **0** | Scaffold, host tests, CUDA probe | no | ✅ Done |
| **1** | DiT **BF16** block parity + text + vision encoder | no | ✅ Done |
| **2** | **Metal-aligned runtime INT8** on GB10 | yes | ✅ Done |
| **3** | Full pipeline + end-to-end generate | inherits Phase 2 | 🔄 In progress |

**Weights:** official `MiniMaxAI/MiniMax-H3` → `FL2VA/*` **BF16** shards only.

## Gate commands

```bash
make -f Makefile.linux test          # includes video + audio VAE smokes
./h3_cuda_video_vae_smoke $H3_MODEL_ROOT
./h3_cuda_audio_vae_smoke $H3_MODEL_ROOT
# e2e (stretch): ./h3 -d $H3_MODEL_ROOT -p "..." --frames 22 -o /tmp/out.mp4
```

## Phase 3 — VAE + generate 🔄 (6h autoloop)

### Video VAE decode ✅

- [x] `rms_norm_f32`, `video_qkv_rope_f32`, `sdpa_f32`
- [x] `h3_cuda_video_vae_smoke` — 5×32×32 finite RGB, 36 SDPA, 38 submissions
- [ ] MLX fixture parity when `misc/fixtures` present

### Audio VAE decode ✅

- [x] `weight_norm_f32`, `conv1d_f32`, `conv1d_stride_f32`
- [x] `conv_transpose1d_f32`, `alias_free_snake_f32`, `snake1d_f32`
- [x] `h3_cuda_audio_vae_smoke` — 2×3200 @32kHz, **136 conv / 16 submissions**

### Remaining / stretch

- [ ] `./h3` end-to-end generate smoke (no first-frame; decode stacks ready)
- [ ] Video encoder: pad / conv3d / group_norm_silu
- [ ] Audio encoder attention: qkv_split / sdpa_causal / attention_pool
- [ ] DiT-only F32 stubs (`adaln_f32`, `gate_f32`, `qkv_rope_f32`) — not on generate hot path
- [ ] Skip Metal-only `mlp_nax_bf16`

Stubs remaining: **10**.

## Commits (Phase 3)

```
(pending) audio VAE F32 conv/snake decode path + smoke
e27b1a3 video VAE F32: rms_norm + video_qkv_rope + sdpa + smoke
```

---

*Last updated: 2026-08-16*
