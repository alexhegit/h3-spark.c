# Spark autoloop progress log

Automated **h3-cuda** port on DGX Spark (`spark` branch).

## Gate commands

```bash
make -f Makefile.linux test          # h3_tests + cuda smoke/ops + tokenizer smoke (if JSON present)
make -f Makefile.linux probe         # CUDA device probe (GB10 sm_121)
make -f Makefile.linux h3            # full CLI binary
H3_TOKENIZER_JSON=path/to/tokenizer.json make -f Makefile.linux test
make -f Makefile.linux tokenizer_test  # full vocab parity (needs MiniMax-H3 tokenizer)
```

## Phase 0 (complete)

- [x] `Makefile.linux`, `h3_cuda.c`, `h3_device.c`, `h3_gpu.cu` scaffold
- [x] Tensor alloc, copy/cast/add BF16, weight pread
- [x] `h3_gpu_stubs.c` + `scripts/gen_gpu_stubs.py`
- [x] Linux host portability fixes
- [x] `tests/test_cuda_smoke.c`, `./h3_tests` 1768 checks

## Phase 1 progress (2026-08-12 autoloop)

### CUDA BF16 ops implemented

| API | Status |
|-----|--------|
| `h3_gpu_silu_bf16` | done |
| `h3_gpu_rms_norm_bf16` | done |
| `h3_gpu_linear_bf16` | done (cuBLAS `GemmEx`) |
| `h3_gpu_adaln_bf16` / `_offset` | done |
| `h3_gpu_gate_bf16` | done |
| `h3_gpu_swiglu_bf16` | done |
| `h3_gpu_mlp_bf16` | done (linear→swiglu→linear) |

- [x] `tests/test_cuda_ops.c` — numerical checks vs CPU oracle
- [x] Stubs reduced: **58** remaining (from 66)

### Tokenizer

- [x] Full `h3_tokenizer.c` BPE port (cJSON + ICU NFC)
- [x] `third_party/cJSON.c` vendored
- [x] `tests/test_tokenizer_smoke.c` — roundtrip when JSON available
- [ ] Full `tests/test_tokenizer.c` parity — needs `MiniMax-H3/tokenizer/tokenizer.json`

## Not yet done

- [ ] SDPA, QKV+RoPE, embedding, remaining ~58 GPU stubs
- [ ] `test_real_dit_block` parity (needs weights + fixtures)
- [ ] `./h3_generate` end-to-end on Spark

## Commits (Phase 1 autoloop)

```
05f8991 SwiGLU + decomposed MLP
0b98314 AdaLN + gate
0985ea4 linear_bf16 cuBLAS
d009804 SiLU + RMSNorm
36d5f0b Phase 0 scaffold
```

---

*Last autoloop run: 2026-08-12*
