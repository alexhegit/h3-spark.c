# Spark autoloop progress log

Automated Phase 0 scaffold for **h3-cuda** on DGX Spark.

## Gate commands

```bash
make -f Makefile.linux test          # h3_tests + h3_cuda_smoke
make -f Makefile.linux probe          # CUDA device probe
make -f Makefile.linux h3             # full CLI binary
./h3 --info -d ./MiniMax-H3           # model inventory (weights required)
```

## Completed (Phase 0)

- [x] `Makefile.linux` — gcc + nvcc, `-lcudart -lstdc++ -licuuc`
- [x] `h3_cuda.c` / `h3_device.c` — CUDA device probe → `h3_device_info`
- [x] `h3_gpu.cu` — context, streams, tensor alloc, copy/cast/add BF16, weight pread
- [x] `h3_gpu_stubs.c` — 66 unimplemented `h3_gpu_*` APIs (link-safe)
- [x] `h3_tokenizer.c` — stub (encode/decode pending Phase 1)
- [x] Linux host fixes: `h3_host.c` bilinear resize, `h3.c` stat time, `h3_ffmpeg.c`
- [x] `h3_cli.c` `xdg-open`, `main.c` CUDA `--info` labels
- [x] `tests/test_cuda_smoke.c` — GPU tensor roundtrip
- [x] `./h3_tests` — 1768 host checks pass on Spark

## Not yet done (next session)

- [ ] `h3_tokenizer.c` full BPE port (ICU + JSON)
- [ ] BF16 linear / RMSNorm / AdaLN kernels
- [ ] `test_real_dit_block` parity
- [ ] `./h3_generate` end-to-end

## New files

| File | Role |
|------|------|
| `Makefile.linux` | Linux/CUDA build |
| `h3_cuda.c`, `h3_cuda.h` | GB10 probe |
| `h3_device.c`, `h3_device.h` | Platform dispatch |
| `h3_gpu.cu` | CUDA backend (implemented subset) |
| `h3_gpu_stubs.c` | Generated API stubs |
| `h3_gpu_cuda_internal.h` | Stub error helper |
| `h3_tokenizer.c` | Tokenizer stub |
| `tools/h3_cuda_probe.c` | Standalone probe binary |
| `tests/test_cuda_smoke.c` | GPU smoke test |

## Modified (portable / Linux)

- `h3.c`, `h3_host.c`, `h3_ffmpeg.c`, `h3_cli.c`, `main.c`, `tests/test_h3.c`

---

*Last autoloop run: 2026-08-12*
