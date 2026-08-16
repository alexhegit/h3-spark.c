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
Do not use Comfy-Org repack or pre-quantized int8/fp8/nvfp4 files.

## Gate commands

```bash
make -f Makefile.linux test
make -f Makefile.linux test-step
H3_MODEL_ROOT=/path/to/MiniMax-H3 make -f Makefile.linux test
./h3_cuda_video_vae_smoke $H3_MODEL_ROOT
# later: ./h3_cuda_audio_vae_smoke ; ./h3 -d $H3_MODEL_ROOT -p "..." -o /tmp/out.mp4
```

## Phase 3 — VAE + generate 🔄 (6h autoloop)

**Priority:** video decode → audio decode → encoder/conditioning → `./h3` e2e.

### Video VAE decode (blocking for generate)

- [x] `h3_gpu_rms_norm_f32`
- [x] `h3_gpu_video_qkv_rope_f32`
- [x] `h3_gpu_sdpa_f32` (naive correctness-first)
- [ ] `h3_cuda_video_vae_smoke` finite RGB + 36 SDPA
- [ ] MLX fixture parity (`test_real_video_vae`) when fixtures present

### Audio VAE decode

- [ ] `weight_norm_f32`, `conv1d_f32`, `conv1d_stride_f32`
- [ ] `conv_transpose1d_f32`, `alias_free_snake_f32`
- [ ] audio decode smoke / `test_real_audio_vae`

### Stretch / later

- [ ] Video encoder: `vae_encoder_pad_f32`, `conv3d_f32`, `group_norm_silu_f32`
- [ ] Audio encoder attention chain
- [ ] `./h3` end-to-end generate smoke (512²×22)
- [ ] Skip Metal-only `mlp_nax_bf16`

Stubs remaining at Phase 3 start: **19** → video trio removes 3 → **16** after first landing.

## Phase 2 — Metal-aligned INT8 ✅

(see prior commits; MLP INT8 forward smoke green)

## Commits (Phase 3)

```
(pending) video VAE F32: rms_norm + video_qkv_rope + sdpa
```

---

*Last updated: 2026-08-16 — Phase 3 6h autoloop started*
