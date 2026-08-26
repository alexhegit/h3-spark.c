"""Simulates exactly the single FP8 linear the CUDA unit test runs, so the two
numbers can be compared directly and any gap blamed on the implementation
rather than on E4M3."""

import torch

ROWS, IN_DIM, OUT_DIM = 128, 5376, 5376
FP8_MAX = 448.0


def bf16(values):
    return values.to(torch.bfloat16).float()


index = torch.arange(ROWS * IN_DIM, dtype=torch.float32)
activation = bf16(
    (torch.sin(index * 0.017) + 0.3 * torch.cos(index * 0.101)).reshape(
        ROWS, IN_DIM
    )
)
index = torch.arange(OUT_DIM * IN_DIM, dtype=torch.float32)
weight = bf16((torch.cos(index * 0.0027) * 0.02).reshape(OUT_DIM, IN_DIM))

reference = bf16(activation @ weight.T)

weight_scale = weight.abs().amax() / FP8_MAX
weight_fp8 = (weight / weight_scale).to(torch.float8_e4m3fn).float()
row_scale = activation.abs().amax(dim=1, keepdim=True) / FP8_MAX
activation_fp8 = (activation / row_scale).to(torch.float8_e4m3fn).float()

# The kernel folds the weight scale into the GEMM and applies the token scale
# to the BF16 result afterwards.
got = bf16(bf16(activation_fp8 @ (weight_fp8 * weight_scale).T) * row_scale)

error = torch.linalg.vector_norm(got - reference)
print(f"fp8 linear relL2: {(error / torch.linalg.vector_norm(reference)):.5f}")

# How much of that is the weights alone, and how much the activations alone?
weight_only = bf16(activation @ (weight_fp8 * weight_scale).T)
activation_only = bf16((activation_fp8 * row_scale) @ weight.T)
for name, value in (("weight only", weight_only),
                    ("activation only", activation_only)):
    gap = torch.linalg.vector_norm(value - reference)
    print(f"  {name:<16}{(gap / torch.linalg.vector_norm(reference)):.5f}")
