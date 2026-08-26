"""Independent simulation of the INT8 and FP8 MLP schemes, to separate a bug in
the CUDA path from the formats simply behaving this way.

Mirrors what the kernels do: INT8 keeps a scale per weight channel and per
token; FP8 keeps one scale for the whole weight tensor and one per token, with
the SwiGLU intermediate landing in BF16 either way.
"""

import sys

import torch

ROWS, IN_DIM, HIDDEN, OUT_DIM = 64, 1024, 2048, 1024
if len(sys.argv) > 1:
    ROWS, IN_DIM, HIDDEN, OUT_DIM = (int(value) for value in sys.argv[1:5])
FP8_MAX = 448.0


def bf16(values):
    return values.to(torch.bfloat16).float()


def int8_per_channel(weight, levels=127.0):
    scale = weight.abs().amax(dim=1, keepdim=True) / levels
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(weight / scale).clamp(-levels, levels) * scale


def int8_per_row(activation, levels=127.0):
    scale = activation.abs().amax(dim=1, keepdim=True) / levels
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(activation / scale).clamp(-levels, levels) * scale


def fp8_per_tensor(weight):
    scale = weight.abs().amax() / FP8_MAX
    scale = scale if scale > 0 else torch.ones_like(scale)
    return (weight / scale).to(torch.float8_e4m3fn).float() * scale


def fp8_per_row(activation):
    scale = activation.abs().amax(dim=1, keepdim=True) / FP8_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (activation / scale).to(torch.float8_e4m3fn).float() * scale


def mlp(activation, fc1, fc2, quantize_weight, quantize_activation):
    fused = bf16(quantize_activation(activation) @ quantize_weight(fc1).T)
    gate, up = fused[:, :HIDDEN], fused[:, HIDDEN:]
    activated = bf16(torch.nn.functional.silu(gate) * up)
    return quantize_activation(activated) @ quantize_weight(fc2).T


def relative_l2(reference, approximation):
    return (
        torch.linalg.vector_norm(approximation - reference)
        / torch.linalg.vector_norm(reference)
    ).item()


index = torch.arange(ROWS * IN_DIM, dtype=torch.float32)
activation = bf16(
    (torch.sin(index * 0.017) + 0.3 * torch.cos(index * 0.101)).reshape(
        ROWS, IN_DIM
    )
)
index = torch.arange(HIDDEN * 2 * IN_DIM, dtype=torch.float32)
fc1 = bf16((torch.sin(index * 0.0031) * 0.02).reshape(HIDDEN * 2, IN_DIM))
index = torch.arange(OUT_DIM * HIDDEN, dtype=torch.float32)
fc2 = bf16((torch.cos(index * 0.0027) * 0.01).reshape(OUT_DIM, HIDDEN))

reference = mlp(activation, fc1, fc2, lambda w: w, lambda a: a)
schemes = {
    "int8 chan/row": (int8_per_channel, int8_per_row),
    "int8 48 levels": (
        lambda w: int8_per_channel(w, 48.0),
        lambda a: int8_per_row(a, 48.0),
    ),
    "int8 13 levels": (
        lambda w: int8_per_channel(w, 13.0),
        lambda a: int8_per_row(a, 13.0),
    ),
    "fp8 tensor/row": (fp8_per_tensor, fp8_per_row),
}
for name, (quantize_weight, quantize_activation) in schemes.items():
    got = mlp(activation, fc1, fc2, quantize_weight, quantize_activation)
    print(f"{name:<16}{relative_l2(reference, got):.5f}")
