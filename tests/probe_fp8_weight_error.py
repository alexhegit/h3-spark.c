"""Quantization error of real DiT weights, so the FP8 scheme is chosen against
the INT8 scheme that ships today rather than against an ideal.

The bar is simple: per-tensor FP8 only earns the right to skip the output
rescale pass if its error is no worse than the per-channel INT8 the engine
already accepts.
"""

import glob
import json
import struct
import sys

import torch


def load_tensor(name):
    for path in sorted(glob.glob(sys.argv[1] + "/*.safetensors")):
        with open(path, "rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
            if name not in header:
                continue
            entry = header[name]
            start, end = entry["data_offsets"]
            handle.seek(8 + header_length + start)
            raw = handle.read(end - start)
        flat = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        return flat.view(entry["shape"]).float()
    raise SystemExit("tensor not found: " + name)


def relative_l2(reference, approximation):
    return (
        torch.linalg.vector_norm(approximation - reference)
        / torch.linalg.vector_norm(reference)
    ).item()


def int8_per_channel(weight):
    scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(weight / scale).clamp(-127, 127) * scale


def fp8_quantize(weight, scale):
    return (weight / scale).to(torch.float8_e4m3fn).float() * scale


def fp8_per_tensor(weight):
    scale = weight.abs().amax() / 448.0
    return fp8_quantize(weight, scale if scale > 0 else torch.ones_like(scale))


def fp8_per_channel(weight):
    scale = weight.abs().amax(dim=1, keepdim=True) / 448.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return fp8_quantize(weight, scale)


E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def round_to_e2m1(values):
    """Nearest representable E2M1 magnitude, sign preserved."""
    magnitude = values.abs().unsqueeze(-1)
    nearest = (magnitude - E2M1_MAGNITUDES).abs().argmin(dim=-1)
    return torch.sign(values) * E2M1_MAGNITUDES[nearest]


def nvfp4_block16(weight):
    """E2M1 elements with one UE4M3 scale per 16 values along the reduction
    axis, which is the layout the Blackwell tensor cores consume."""
    rows, columns = weight.shape
    blocks = weight.reshape(rows, columns // 16, 16)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / 6.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scale = scale.to(torch.float8_e4m3fn).float()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (round_to_e2m1(blocks / scale) * scale).reshape(rows, columns)


NAMES = [
    "blocks.{}.mlp.fc1.weight",
    "blocks.{}.mlp.fc2.weight",
    "blocks.{}.attn.qkv_proj.weight",
    "blocks.{}.attn.out_proj.weight",
]

print(f"{'tensor':<40}{'int8/chan':>10}{'fp8/tensor':>11}{'fp8/chan':>10}"
      f"{'nvfp4/16':>10}")
for block in (0, 3, 25, 49):
    for template in NAMES:
        name = template.format(block)
        weight = load_tensor(name)
        row = (
            f"{name:<40}"
            f"{relative_l2(weight, int8_per_channel(weight)):>10.5f}"
            f"{relative_l2(weight, fp8_per_tensor(weight)):>11.5f}"
            f"{relative_l2(weight, fp8_per_channel(weight)):>10.5f}"
            f"{relative_l2(weight, nvfp4_block16(weight)):>10.5f}"
        )
        print(row, flush=True)
