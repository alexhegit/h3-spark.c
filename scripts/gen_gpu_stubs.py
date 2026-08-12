#!/usr/bin/env python3
"""Regenerate h3_gpu_stubs.c for functions not implemented in h3_gpu.cu."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "h3_gpu.h"
OUTPUT = ROOT / "h3_gpu_stubs.c"

IMPLEMENTED = {
    "h3_gpu_create", "h3_gpu_free", "h3_gpu_is_m5", "h3_gpu_has_nax_mlp",
    "h3_gpu_has_int8_mlp", "h3_gpu_tensor_new_f32", "h3_gpu_tensor_new_bf16",
    "h3_gpu_tensor_new_i8", "h3_gpu_tensor_from_f32", "h3_gpu_tensor_from_bf16",
    "h3_gpu_tensor_from_u32", "h3_gpu_tensor_load_bf16", "h3_gpu_tensor_load_f32",
    "h3_gpu_tensor_read_file_bf16", "h3_gpu_tensor_stream_file_bf16",
    "h3_gpu_tensor_free", "h3_gpu_tensor_elements", "h3_gpu_tensor_dtype",
    "h3_gpu_tensor_read_f32", "h3_gpu_tensor_read_f32_range",
    "h3_gpu_tensor_read_bf16", "h3_gpu_tensor_write_f32",
    "h3_gpu_tensor_write_f32_range", "h3_gpu_tensor_write_bf16",
    "h3_gpu_tensor_write_bf16_range", "h3_gpu_begin", "h3_gpu_continue",
    "h3_gpu_submit", "h3_gpu_error", "h3_gpu_get_stats",
    "h3_gpu_profile_set_label", "h3_gpu_profile_mark",
    "h3_gpu_cast_f32_to_bf16", "h3_gpu_cast_bf16_to_f32",
    "h3_gpu_copy_bf16", "h3_gpu_copy_f32", "h3_gpu_add_bf16",
    "h3_gpu_silu_bf16", "h3_gpu_rms_norm_bf16",
    "h3_gpu_linear_bf16",
    "h3_gpu_cuda_set_error",
}

text = HEADER.read_text()
parts = re.split(
    r"(?=\n(?:int|void|h3_gpu \*|h3_gpu_tensor \*|const char \*) h3_gpu_)", text
)
lines = [
    '#include "h3_gpu.h"',
    '#include "h3_gpu_cuda_internal.h"',
    "",
    "static int h3_gpu_stub_impl(h3_gpu *gpu, const char *name) {",
    "    h3_gpu_cuda_set_error(gpu, name);",
    "    return 0;",
    "}",
    "",
]
for part in parts:
    match = re.match(
        r"\n((?:int|void|h3_gpu \*|h3_gpu_tensor \*|const char \*) (h3_gpu_\w+)\([^;]+\);)",
        part,
        re.S,
    )
    if not match:
        continue
    decl = match.group(1).strip()
    name = match.group(2)
    if name in IMPLEMENTED or decl.startswith("void "):
        continue
    sig = decl[:-1]
    lines.append(f"{sig} {{")
    if sig.split("(")[1].strip().startswith("h3_gpu *"):
        lines.append(f'    return h3_gpu_stub_impl(gpu, "{name}");')
    else:
        lines.append(f'    return h3_gpu_stub_impl(NULL, "{name}");')
    lines.append("}\n")

OUTPUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUTPUT.name} with {sum(1 for l in lines if l.startswith('int h3_gpu_'))} stubs")
